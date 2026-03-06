#!/usr/bin/env python3
"""API 服务逻辑：与 test2 run_test 同套执行环境（D-Bus/AT-SPI、GPU、黑屏重试等），进程池执行。
cd /data/zhijieliu/AppEvalPilot
conda activate appeval
no_proxy='*' python -m appeval.api_server.server --port 8888


"""
import argparse
import asyncio
from datetime import datetime
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from loguru import logger
from pydantic import BaseModel, model_validator

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
API_LOG_BASE = PROJECT_DIR / "appeval" / "api_log"
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
CALLBACK_BASE_URL = "https://test-tool.deepwisdomai.com/api/v1/agent-eval/notice/detail"
SHARED_MODEL_CACHE = os.path.join(PROJECT_DIR, ".cache", "modelscope")


class StartRequest(BaseModel):
    task_id: Union[int, str]
    detail_id: Union[int, str]
    case_name: str
    test_arry: List[str]
    prod_url: str

    @model_validator(mode="before")
    @classmethod
    def accept_test_array(cls, data):
        """兼容对方传 test_array 或 test_arry"""
        if isinstance(data, dict) and "test_array" in data and "test_arry" not in data:
            data = {**data, "test_arry": data["test_array"]}
        return data


class BatchRequest(BaseModel):
    """一批任务：仅接单并后台执行，结果通过回调推送"""
    tasks: List[StartRequest]
    callback_url: Optional[str] = None


class StartResponse(BaseModel):
    code: int = 0
    message: str = "ok"
    data: dict = {"success": True}


app = FastAPI(title="AppEval API", version="1.0")
_task_status: Dict[int, dict] = {}
_executor: Optional[ProcessPoolExecutor] = None
_preset: dict = {}
_full_cfg: dict = {}
_case_counter: Dict[str, int] = {}   # "{task_id}/{case_name}" -> 下一个可用编号
_counter_lock = asyncio.Lock()   # 分配 start_index 时加锁，避免并发都拿到 0
_worker_slots: Optional[asyncio.Queue] = None


def load_config(path: str) -> tuple:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    model = (cfg.get("model") or "remote").strip().lower()
    if model not in ("local", "remote"):
        model = "remote"
    preset = cfg.get(model, {})
    workers = int(cfg.get("workers", 5))
    return workers, preset, cfg


def _apply_llm_env(preset: dict, for_local: bool = False):
    llm = preset.get("llm") or {}
    os.environ["llm__api_type"] = str(llm.get("api_type", "openai"))
    os.environ["llm__model"] = str(llm.get("model", ""))
    os.environ["llm__base_url"] = str(llm.get("base_url", ""))
    os.environ["llm__api_key"] = str(llm.get("api_key", ""))
    os.environ["llm__stream"] = str(llm.get("stream", "false"))
    if "max_token" in llm:
        os.environ["llm__max_token"] = str(llm["max_token"])
    if for_local:
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[k] = ""


def _start_xvfb(display: int):
    return subprocess.Popen(
        ["Xvfb", f":{display}", "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _start_dbus_and_atspi(display_num: int, worker_id: int):
    dbus_proc = atspi_launcher = atspi_registryd = None
    env = os.environ.copy()
    env["DISPLAY"] = f":{display_num}"
    try:
        result = subprocess.run(
            ["dbus-launch", "--sh-syntax"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        if result.returncode != 0:
            return None, None, None
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, _, val = line.partition("=")
                val = val.strip().rstrip(";").strip("'\"")
                os.environ[key] = val
        for path, var in [
            ("/usr/libexec/at-spi-bus-launcher", "atspi_launcher"),
            ("/usr/libexec/at-spi2-registryd", "atspi_registryd"),
        ]:
            if os.path.exists(path):
                p = subprocess.Popen(
                    [path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env=os.environ.copy(),
                )
                time.sleep(2 if "launcher" in path else 1)
                if "launcher" in path:
                    atspi_launcher = p
                else:
                    atspi_registryd = p
    except Exception:
        pass
    return dbus_proc, atspi_launcher, atspi_registryd


def _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id: int):
    for proc in (atspi_registryd, atspi_launcher):
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    pid = os.environ.get("DBUS_SESSION_BUS_PID")
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass


def _start_wm_for_display(display_num: int):
    env = os.environ.copy()
    env["DISPLAY"] = f":{display_num}"
    for wm in ("openbox", "fluxbox"):
        exe = shutil.which(wm)
        if not exe:
            for p in (f"/usr/bin/{wm}", f"/usr/local/bin/{wm}"):
                if os.path.isfile(p):
                    exe = p
                    break
        if exe:
            try:
                subprocess.Popen(
                    [exe], env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(1)
                return
            except Exception:
                continue


def _setup_xauthority(display_num: int, worker_id: int):
    auth_file = f"/tmp/.Xauthority_api_w{worker_id}"
    try:
        import secrets
        cookie = secrets.token_hex(16)
        subprocess.run(
            ["xauth", "-f", auth_file, "add", f":{display_num}", ".", "MIT-MAGIC-COOKIE-1", cookie],
            check=True, capture_output=True,
        )
    except Exception:
        Path(auth_file).touch()
    os.environ["XAUTHORITY"] = auth_file


class _StderrFilter:
    def __init__(self, stream):
        self._stream = stream
    def write(self, msg):
        if "xauthority" in msg.lower():
            return
        self._stream.write(msg)
    def flush(self):
        self._stream.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)


def _do_callback_sync(url: str, payload: dict, retry_times: int = 3, retry_interval: int = 30):
    """子进程内同步回调：落盘后立即 POST，不依赖主进程。"""
    import httpx as _httpx
    detail_id = payload.get("detail_id", "")
    for attempt in range(1, retry_times + 1):
        try:
            with _httpx.Client(timeout=_httpx.Timeout(30.0, connect=10.0)) as c:
                resp = c.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning(f"[Worker-CB] detail_id={detail_id} 第{attempt}/{retry_times}次失败: HTTP {resp.status_code}")
            else:
                cb_ok = False
                try:
                    rj = resp.json()
                    if isinstance(rj, dict):
                        d = rj.get("data", {})
                        cb_ok = d.get("success") is True or rj.get("ok") is True or rj.get("success") is True
                except Exception:
                    pass
                if cb_ok:
                    logger.info(f"[Worker-CB] detail_id={detail_id} 回调成功: HTTP {resp.status_code}")
                    return
                logger.warning(f"[Worker-CB] detail_id={detail_id} 第{attempt}/{retry_times}次 data.success!=true")
        except Exception as e:
            logger.error(f"[Worker-CB] detail_id={detail_id} 第{attempt}/{retry_times}次异常: {type(e).__name__} {e}")
        if attempt < retry_times:
            time.sleep(retry_interval)
    logger.error(f"[Worker-CB] detail_id={detail_id} 回调最终失败")


def _worker_callback(callback_url: str, full_cfg: dict, result: dict):
    """在子进程内执行回调，落盘后立即调用，不依赖主进程存活。"""
    if not callback_url or full_cfg.get("skip_callback"):
        logger.info(f"[Worker-CB] detail_id={result.get('detail_id')} 跳过回调")
        return
    try:
        transformed = _transform_result_for_callback(result)
        url = _callback_url_with_detail_id(callback_url, result.get("detail_id", ""))
        retry_times = max(1, int(full_cfg.get("callback_retry_times", 3) or 3))
        retry_interval = max(1, int(full_cfg.get("callback_retry_interval_sec", 30) or 30))
        _do_callback_sync(url, transformed, retry_times, retry_interval)
    except Exception as e:
        logger.error(f"[Worker-CB] 子进程内回调失败: {e}")


def _run_task_in_process(
    worker_id: int,
    detail_id: int,
    task_id: int,
    case_name: str,
    prod_url: str,
    test_arry: List[str],
    preset: dict,
    full_cfg: dict,
    start_index: int = 0,
    run_group_ts: Optional[str] = None,
    callback_url: str = "",
) -> dict:
    """单任务在子进程中执行，落盘后立即在子进程内回调，不依赖主进程。"""
    logger.info(f"[Worker] pid={os.getpid()} worker_id={worker_id} detail_id={detail_id} 开始执行")
    warnings.filterwarnings("ignore", message=".*xauthority.*")
    sys.stderr = _StderrFilter(sys.stderr)

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    os.chdir(str(PROJECT_DIR))

    worker_cache = f"/tmp/modelscope_api_w{worker_id}"
    os.environ["MODELSCOPE_CACHE"] = worker_cache
    if os.path.exists(SHARED_MODEL_CACHE) and not os.path.exists(os.path.join(worker_cache, "hub")):
        shutil.copytree(SHARED_MODEL_CACHE, worker_cache, dirs_exist_ok=True)
    os.makedirs(worker_cache, exist_ok=True)

    cuda_devices = preset.get("cuda_devices", [0])
    assigned_gpu = cuda_devices[worker_id % len(cuda_devices)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    _apply_llm_env(preset, for_local=(preset.get("config_file", "").find("local") >= 0))
    if full_cfg.get("chrome_disable_web_security"):
        os.environ["APPEVAL_CHROME_DISABLE_WEB_SECURITY"] = "1"

    base_display = int(preset.get("base_display", 300)) + 200
    base_chrome_port = int(preset.get("base_chrome_port", 9500)) + 2000
    llm_config = preset.get("llm", {})
    base_log_prefix = preset.get("log_dir_prefix", "api")
    main_model_raw = llm_config.get("model", "unknown")
    safe_model = main_model_raw.replace("/", "_").replace(" ", "_").replace(":", "_")
    for c in '\\*?"<>|':
        safe_model = safe_model.replace(c, "_")
    log_dir_prefix = f"{base_log_prefix}_{safe_model}"

    case_generator_config = preset.get("case_generator", llm_config)
    tell_verifier_config = full_cfg.get("tell_verifier", {})
    config_file = f"/tmp/appeval_api_cfg_w{worker_id}.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump({
            "llm": llm_config,
            "case_generator": case_generator_config,
            "tell_verifier": tell_verifier_config,
        }, f, default_flow_style=False)

    display_num = base_display + worker_id
    port = base_chrome_port + worker_id
    user_data_dir = f"/tmp/chrome_api_{log_dir_prefix}_w{worker_id}_d{detail_id}"

    # 错峰启动，避免 20 个 worker 同时初始化 Xvfb/D-Bus/Chrome 导致卡死
    time.sleep(worker_id * 1.2)
    xvfb = _start_xvfb(display_num)
    time.sleep(2)
    os.environ["DISPLAY"] = f":{display_num}"
    _setup_xauthority(display_num, worker_id)
    _start_wm_for_display(display_num)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["GTK_MODULES"] = "gail:atk-bridge"
    os.environ["GNOME_ACCESSIBILITY"] = "1"
    os.environ["NO_AT_BRIDGE"] = "0"
    dbus_proc, atspi_launcher, atspi_registryd = _start_dbus_and_atspi(display_num, worker_id)

    shutil.rmtree(user_data_dir, ignore_errors=True)
    os.makedirs(user_data_dir, exist_ok=True)

    llm_cfg = preset.get("llm", {})
    if not run_group_ts:
        run_group_ts = datetime.now().strftime("%Y%m%d%H%M")
    test_cases = {str(start_index + i): {"case_desc": desc, "result": "", "evidence": ""} for i, desc in enumerate(test_arry)}

    def _default_case_item(i: int, err: str) -> dict:
        return {
            "test_id": f"{case_name}{start_index + i}",
            "case_desc": test_arry[i],
            "evidence": f"Error: {err}",
            "result": False,
            "cost": "time=0.0s, usd=$0.000000",
        }

    def _load_partial_cases(err: str) -> list:
        """Load per-case results from api_log to preserve completed cases on late exceptions."""
        partial = []
        for i, _desc in enumerate(test_arry):
            case_num = start_index + i
            case_dir = API_LOG_BASE / str(task_id) / run_group_ts / case_name / f"{case_name}{case_num}"
            candidates = []
            try:
                ts_dirs = [p for p in case_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{12,}", p.name)]
                ts_dirs = sorted(ts_dirs, key=lambda p: p.name, reverse=True)
                candidates.extend([p / "test_case.json" for p in ts_dirs])
            except Exception:
                pass
            candidates.append(case_dir / "test_case.json")

            loaded = None
            for fp in candidates:
                try:
                    if not fp.exists():
                        continue
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    arr = data.get("test_cases", [])
                    if arr and isinstance(arr[0], dict):
                        loaded = dict(arr[0])
                        break
                except Exception:
                    continue

            if loaded:
                loaded.setdefault("test_id", f"{case_name}{case_num}")
                loaded.setdefault("case_desc", test_arry[i])
                loaded.setdefault("evidence", f"Error: {err}")
                loaded.setdefault("result", False)
                loaded.setdefault("cost", "time=0.0s, usd=$0.000000")
                partial.append(loaded)
            else:
                partial.append(_default_case_item(i, err))
        return partial

    try:
        from appeval.roles.eval_runner import AppEvalRole

        api_log_dir = str(API_LOG_BASE)
        os.makedirs(api_log_dir, exist_ok=True)
        role = AppEvalRole(
            config_file=config_file,
            remote_debugging_port=port,
            user_data_dir=user_data_dir,
            use_chrome_debugger=False,
            agent_class=preset.get("agent_class", "osagent"),
            a11y_mode=preset.get("a11y_mode", "atspi"),
            max_iters=preset.get("max_iters", 15),
            use_ocr=True,
            post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
            log_dirs=api_log_dir,
            use_timestamp_log_dir=True,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        t0 = time.perf_counter()
        result, _ = loop.run_until_complete(
            role.run_api(
                task_name="test_case",
                test_cases=test_cases,
                start_func=prod_url,
                log_dir=f"{task_id}/{run_group_ts}/{case_name}",
                sequential_mode=True,
                case_name_for_log=case_name,
            )
        )
        elapsed = time.perf_counter() - t0
        loop.close()

        # 统计美元成本（CaseGenerator + OSAgent）
        # 先读 metagpt 的 total_cost；若模型名未命中价表导致 total_cost=0，则按兜底单价估算。
        total_usd = 0.0
        fallback_pricing = {
            # USD / 1K tokens（$0.5/$3.0 per 1M tokens）
            "gemini-3-flash-preview": {"prompt": 0.0005, "completion": 0.003},
        }

        def _cost_to_usd(cost_obj, model_name: str) -> float:
            direct = float(
                getattr(cost_obj, "total_cost_usd", None)
                or getattr(cost_obj, "total_cost", None)
                or getattr(cost_obj, "cost", None)
                or 0.0
            )
            if direct > 0:
                return direct
            pt = int(getattr(cost_obj, "total_prompt_tokens", 0) or 0)
            ct = int(getattr(cost_obj, "total_completion_tokens", 0) or 0)
            if pt == 0 and ct == 0:
                return 0.0
            price = fallback_pricing.get(model_name or "")
            if not price:
                return 0.0
            return (pt * float(price["prompt"]) + ct * float(price["completion"])) / 1000.0

        try:
            if hasattr(role.test_generator, "llm") and hasattr(role.test_generator.llm, "get_costs"):
                c = role.test_generator.llm.get_costs()
                model_name = getattr(role.test_generator.llm, "model", "") or llm_cfg.get("model", "")
                total_usd += _cost_to_usd(c, model_name)
            if hasattr(role, "osagent") and role.osagent and hasattr(role.osagent, "llm") and hasattr(role.osagent.llm, "get_costs"):
                c = role.osagent.llm.get_costs()
                model_name = getattr(role.osagent.llm, "model", "") or llm_cfg.get("model", "")
                total_usd += _cost_to_usd(c, model_name)
        except Exception:
            total_usd = 0.0

        case_count = max(1, len(test_arry))
        avg_time = elapsed / case_count
        avg_usd = total_usd / case_count

        cases = []
        for i, desc in enumerate(test_arry):
            key = str(start_index + i)
            res = (result or {}).get(key, {})
            if isinstance(res, dict):
                result_value = res.get("result", "Fail")
                evidence = res.get("evidence", "")
                # Use per-case cost from runner when present (sequential mode), else fallback to avg
                res_cost = res.get("cost", "")
                if res_cost and "time=" in str(res_cost) and "usd=" in str(res_cost):
                    cost_str = res_cost
                else:
                    cost_str = f"time={avg_time:.1f}s, usd=${avg_usd:.6f}"
            else:
                result_value = str(res)
                evidence = str(res)
                cost_str = f"time={avg_time:.1f}s, usd=${avg_usd:.6f}"
            passed = result_value.lower().strip() in ("pass", "true", "1")
            cases.append({
                "test_id": f"{case_name}{key}",
                "case_desc": desc,
                "evidence": evidence,
                "result": passed,
                "cost": cost_str,
            })

        black_kws = ["black screen", "blank screen", "black page", "blank page"]
        if not passed and any(k in evidence.lower() for k in black_kws):
            try:
                from appeval.utils.window_utils import kill_windows
                loop2 = asyncio.new_event_loop()
                asyncio.set_event_loop(loop2)
                loop2.run_until_complete(kill_windows(user_data_dir=user_data_dir))
                loop2.close()
                time.sleep(2)
                shutil.rmtree(user_data_dir, ignore_errors=True)
                os.makedirs(user_data_dir, exist_ok=True)
                role2 = AppEvalRole(
                    config_file=config_file,
                    remote_debugging_port=port,
                    user_data_dir=user_data_dir,
                    use_chrome_debugger=False,
                    agent_class=preset.get("agent_class", "osagent"),
                    a11y_mode=preset.get("a11y_mode", "atspi"),
                    max_iters=preset.get("max_iters", 15),
                    use_ocr=True,
                    post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
                    log_dirs=api_log_dir,
                    use_timestamp_log_dir=True,
                )
                loop3 = asyncio.new_event_loop()
                asyncio.set_event_loop(loop3)
                result2, _ = loop3.run_until_complete(
                    role2.run_api(
                        task_name="test_case",
                        test_cases=test_cases,
                        start_func=prod_url,
                        log_dir=f"{task_id}/{run_group_ts}/{case_name}",
                        sequential_mode=True,
                        case_name_for_log=case_name,
                    )
                )
                loop3.close()
                if result2:
                    cases = []
                    for j, desc in enumerate(test_arry):
                        key2 = str(start_index + j)
                        r2 = result2.get(key2, {})
                        if isinstance(r2, dict):
                            rv, ev = r2.get("result", "Fail"), r2.get("evidence", "")
                            rc = r2.get("cost", "")
                            cost_str2 = rc if (rc and "time=" in str(rc) and "usd=" in str(rc)) else f"time={avg_time:.1f}s, usd=${avg_usd:.6f}"
                        else:
                            rv, ev = str(r2), str(r2)
                            cost_str2 = f"time={avg_time:.1f}s, usd=${avg_usd:.6f}"
                        cases.append({
                            "test_id": f"{case_name}{key2}",
                            "case_desc": desc,
                            "evidence": ev,
                            "result": rv.lower().strip() in ("pass", "true", "1"),
                            "cost": cost_str2,
                        })
            except Exception:
                pass

        # 回写每个 case 的 test_case.json，确保文件内包含 cost / bool result / test_id
        try:
            for case_item in cases:
                test_id = str(case_item.get("test_id", ""))
                if not test_id.startswith(case_name):
                    continue
                case_suffix = test_id[len(case_name):]
                case_dir = API_LOG_BASE / str(task_id) / run_group_ts / case_name / f"{case_name}{case_suffix}"
                case_dir.mkdir(parents=True, exist_ok=True)
                ts_dirs = [p for p in case_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{12,}", p.name)]
                output_dir = sorted(ts_dirs, key=lambda p: p.name)[-1] if ts_dirs else case_dir
                output_file = output_dir / "test_case.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({"test_cases": [case_item]}, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

        final_result = {
            "detail_id": str(detail_id),
            "case_name": case_name,
            "eval_cost": f"avg_time={avg_time:.1f}s, avg_usd=${avg_usd:.6f}",
            "version": llm_cfg.get("model", ""),
            "test_cases": cases,
        }

        # 子进程内直接回调，不再依赖主进程
        _worker_callback(callback_url, full_cfg, final_result)

        return final_result
    except Exception as e:
        traceback.print_exc()
        err_msg = str(e).strip() or repr(e)
        cases = _load_partial_cases(err_msg)
        times = [_parse_cost_to_seconds(str(c.get("cost", ""))) for c in cases]
        avg_time = (sum(times) / max(1, len(times))) if times else 0.0
        avg_usd = 0.0

        # 异常场景也回写文件，避免 test_case.json 缺少 cost/result 规范字段
        try:
            for case_item in cases:
                test_id = str(case_item.get("test_id", ""))
                if not test_id.startswith(case_name):
                    continue
                case_suffix = test_id[len(case_name):]
                case_dir = API_LOG_BASE / str(task_id) / run_group_ts / case_name / f"{case_name}{case_suffix}"
                case_dir.mkdir(parents=True, exist_ok=True)
                ts_dirs = [p for p in case_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{12,}", p.name)]
                output_dir = sorted(ts_dirs, key=lambda p: p.name)[-1] if ts_dirs else case_dir
                output_file = output_dir / "test_case.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({"test_cases": [case_item]}, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

        err_result = {
            "detail_id": str(detail_id),
            "case_name": case_name,
            "eval_cost": f"avg_time={avg_time:.1f}s, avg_usd=$0.000000",
            "version": llm_cfg.get("model", ""),
            "test_cases": cases,
        }

        # 异常分支也在子进程内回调
        _worker_callback(callback_url, full_cfg, err_result)

        return err_result
    finally:
        _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id)
        xvfb.terminate()
        xvfb.wait()
        shutil.rmtree(user_data_dir, ignore_errors=True)
        try:
            from appeval.tools.ocr import release_ocr_memory
            release_ocr_memory()
        except Exception:
            pass


def _run_task_in_process_wrapper(args_tuple):
    """包装：ProcessPoolExecutor 用 tuple 传参。"""
    return _run_task_in_process(*args_tuple)


def _should_skip_callback() -> bool:
    """本地测试不请求对方时跳过回调。由 config skip_callback 或启动参数 --no-callback 控制。"""
    return bool(_full_cfg.get("skip_callback", False))


async def _callback(detail_id: int, result: dict):
    if _should_skip_callback():
        logger.info(f"[Callback] detail_id={detail_id} 已跳过（本地测试，不回调对方）")
        return
    url = f"{CALLBACK_BASE_URL}/{detail_id}"
    payload = [result]
    retry_times = int(_full_cfg.get("callback_retry_times", 3) or 3)
    retry_interval_sec = int(_full_cfg.get("callback_retry_interval_sec", 300) or 300)
    retry_times = max(1, retry_times)
    retry_interval_sec = max(1, retry_interval_sec)
    logger.info(f"[Callback] POST {url} | body 前 300 字: {json.dumps(payload, ensure_ascii=False)[:300]}")

    for attempt in range(1, retry_times + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                resp = await c.post(url, json=payload)
                if resp.status_code >= 400:
                    logger.warning(
                        f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: "
                        f"HTTP {resp.status_code} | {resp.text[:500]}"
                    )
                else:
                    callback_ok = False
                    callback_msg = ""
                    callback_code = None
                    try:
                        resp_json = resp.json()
                        data = resp_json.get("data", {}) if isinstance(resp_json, dict) else {}
                        callback_ok = (data.get("success") is True)
                        callback_msg = str(resp_json.get("msg", resp_json.get("message", "")))
                        callback_code = resp_json.get("code", None)
                    except Exception:
                        callback_ok = False
                        callback_msg = "回调响应非 JSON 或缺少 data.success"

                    if callback_ok:
                        logger.info(
                            f"[Callback] detail_id={detail_id} 成功: HTTP {resp.status_code} "
                            f"| code={callback_code} msg={callback_msg}"
                        )
                        return
                    logger.warning(
                        f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: "
                        f"回调响应 data.success!=true | HTTP {resp.status_code} | body={resp.text[:500]}"
                    )
        except Exception as e:
            err_msg = str(e).strip() or repr(e)
            logger.error(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次异常: {type(e).__name__} {err_msg}")

        if attempt < retry_times:
            logger.warning(f"[Callback] detail_id={detail_id} {retry_interval_sec}s 后重试...")
            await asyncio.sleep(retry_interval_sec)

    logger.error(f"[Callback] detail_id={detail_id} 回调最终失败，已达最大重试次数 {retry_times}")


def _callback_url_with_detail_id(base_url: str, detail_id) -> str:
    """将 callback_base_url 末尾的数字换成具体的 detail_id。例如 .../detail/1 -> .../detail/123"""
    base_url = base_url.strip().rstrip("/")
    base_url = re.sub(r"/\d+$", "", base_url)  # 去掉末尾的 /数字
    return f"{base_url}/{detail_id}"


def _parse_cost_to_seconds(cost_str: str) -> int:
    """从 'time=10.0s, usd=...' 或 'avg_time=10.0s, ...' 中解析出秒数，返回整数。"""
    if not cost_str or not isinstance(cost_str, str):
        return 0
    m = re.search(r"(?:avg_)?time[=:]?\s*([\d.]+)\s*s", cost_str, re.I)
    if m:
        try:
            return int(float(m.group(1)))
        except (ValueError, TypeError):
            pass
    return 0


def _transform_result_for_callback(result: dict) -> dict:
    """对方接口要求 eval_cost/test_id/cost 为整数，将我们的 result 转成对方格式。"""
    out = dict(result)
    out["eval_cost"] = _parse_cost_to_seconds(str(out.get("eval_cost", "")))
    did = result.get("detail_id")
    try:
        out["detail_id"] = int(did) if did is not None and str(did).strip() != "" else did
    except (TypeError, ValueError):
        out["detail_id"] = did
    cases = []
    for i, tc in enumerate(out.get("test_cases", [])):
        c = dict(tc)
        tid = tc.get("test_id", i)
        if isinstance(tid, str):
            nums = re.findall(r"\d+", tid)
            raw = int(nums[0]) if nums else i
        else:
            try:
                raw = int(tid)
            except (TypeError, ValueError):
                raw = i
        c["test_id"] = max(1, raw)  # 对方要求 test_id 必须为正整数
        cost = tc.get("cost", "")
        c["cost"] = _parse_cost_to_seconds(str(cost))
        cases.append(c)
    out["test_cases"] = cases
    return out


async def _callback_one(callback_url: str, result: dict):
    """单条结果立即回调：POST 到 .../detail/{detail_id}，任务完成即触发，不等其他任务。"""
    detail_id = result.get("detail_id", "")
    if _should_skip_callback():
        logger.info(f"[Callback] detail_id={detail_id} 已跳过（本地测试不回调）")
        return
    retry_times = int(_full_cfg.get("callback_retry_times", 3) or 3)
    retry_interval_sec = int(_full_cfg.get("callback_retry_interval_sec", 300) or 300)
    retry_times = max(1, retry_times)
    retry_interval_sec = max(1, retry_interval_sec)
    url = _callback_url_with_detail_id(callback_url.strip(), detail_id)
    payload = _transform_result_for_callback(result)
    logger.info(f"[Callback] POST {url} | body 前 300 字: {json.dumps(payload, ensure_ascii=False)[:300]}")
    for attempt in range(1, retry_times + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False) as c:
                resp = await c.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: HTTP {resp.status_code} | {resp.text[:500]}")
            else:
                callback_ok = False
                try:
                    resp_json = resp.json()
                    if isinstance(resp_json, dict):
                        data = resp_json.get("data", {})
                        callback_ok = (
                            data.get("success") is True
                            or resp_json.get("ok") is True
                            or resp_json.get("success") is True
                        )
                except Exception:
                    pass
                if callback_ok:
                    logger.info(f"[Callback] detail_id={detail_id} 成功: HTTP {resp.status_code}")
                    return
                logger.warning(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: 回调需 data.success 或 ok=true | {resp.text[:500]}")
        except Exception as e:
            logger.error(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次异常: {type(e).__name__} {e}")
        if attempt < retry_times:
            await asyncio.sleep(retry_interval_sec)
    logger.error(f"[Callback] detail_id={detail_id} 回调最终失败，已达最大重试次数 {retry_times}")


async def _dispatch(req: StartRequest, start_index: int, run_group_ts: Optional[str] = None):
    _task_status[req.detail_id] = {"status": "running", "case_name": req.case_name}
    cb_url = (_full_cfg.get("callback_base_url") or CALLBACK_BASE_URL or "").strip().rstrip("/")

    worker_id = await _worker_slots.get()
    try:
        logger.info(f"[API] 提交任务 detail_id={req.detail_id} worker_id={worker_id} "
                    f"start_index={start_index}")
        loop = asyncio.get_event_loop()
        args_tuple = (
            worker_id, req.detail_id, req.task_id, req.case_name,
            req.prod_url, req.test_arry,
            dict(_preset), dict(_full_cfg), start_index, run_group_ts,
            cb_url,
        )
        result = await loop.run_in_executor(
            _executor,
            _run_task_in_process_wrapper,
            args_tuple,
        )
        _task_status[req.detail_id] = {"status": "done", "result": result}
        logger.info(f"[API] detail_id={req.detail_id} 任务完成（回调已在子进程内执行）")
    except Exception as e:
        _task_status[req.detail_id] = {"status": "error", "error": str(e)}
        logger.error(f"[API] detail_id={req.detail_id} 子进程异常: {e}（回调已在子进程内执行或进程被杀）")
    finally:
        _worker_slots.put_nowait(worker_id)


@app.post("/start", response_model=StartResponse)
async def start_task(req: StartRequest):
    logger.info(f"[API] task_id={req.task_id} detail_id={req.detail_id} "
                f"case={req.case_name} tests={len(req.test_arry)} url={req.prod_url}")

    if _task_status.get(req.detail_id, {}).get("status") == "running":
        return StartResponse(code=1, message="任务执行中", data={"success": False})

    # 全局编号：同一 task_id/case_name 下连续编号，加锁避免并发时两个请求都拿到 0
    counter_key = f"{req.task_id}/{req.case_name}"
    async with _counter_lock:
        start_index = _case_counter.get(counter_key, 0)
        _case_counter[counter_key] = start_index + len(req.test_arry)
    run_group_ts = datetime.now().strftime("%Y%m%d%H%M")

    asyncio.create_task(_dispatch(req, start_index, run_group_ts))
    return StartResponse()


def _parse_eval_cost(cost_text: str) -> tuple[float, float]:
    m = re.search(r"avg_time=([0-9]+(?:\.[0-9]+)?)s,\s*avg_usd=\$([0-9]+(?:\.[0-9]+)?)", str(cost_text))
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


async def _run_batch_then_callback(
    req: BatchRequest,
    start_index_map: Dict[int, int],
    callback_url: str,
    run_group_ts: str,
):
    """后台并发执行本批所有任务，回调在子进程内完成，主进程仅更新状态。"""
    loop = asyncio.get_event_loop()
    for t in req.tasks:
        _task_status[t.detail_id] = {"status": "running", "case_name": t.case_name}

    async def _run_one(t: StartRequest):
        start_index = start_index_map[t.detail_id]

        worker_id = await _worker_slots.get()
        try:
            logger.info(f"[API] /batch detail_id={t.detail_id} worker_id={worker_id} 开始执行")
            args_tuple = (
                worker_id, t.detail_id, t.task_id, t.case_name,
                t.prod_url, t.test_arry,
                dict(_preset), dict(_full_cfg), start_index, run_group_ts,
                callback_url,
            )
            result = await loop.run_in_executor(_executor, _run_task_in_process_wrapper, args_tuple)
            _task_status[t.detail_id] = {"status": "done", "result": result}
            logger.info(f"[API] /batch detail_id={t.detail_id} 完成（回调已在子进程内执行）")
        except Exception as e:
            _task_status[t.detail_id] = {"status": "error", "error": str(e)}
            logger.error(f"[API] /batch detail_id={t.detail_id} 子进程异常: {e}（回调已在子进程内执行或进程被杀）")
        finally:
            _worker_slots.put_nowait(worker_id)

    await asyncio.gather(*[_run_one(t) for t in req.tasks])
    logger.info(f"[API] /batch 全部 {len(req.tasks)} 个任务已完成")


@app.post("/batch")
async def batch_tasks(request: Request):
    """接收一批任务（JSON 文件或 JSON body），仅返回接单结果；每个 detail_id 完成后立即单独回调。
    支持三种方式：
    1. POST body (JSON) - 标准方式
    2. multipart/form-data - 对方使用的方式
    3. URL 参数
    """
    req = None
    
    # 请求调试日志
    content_type = request.headers.get('content-type', '').lower()
    logger.info("[API] /batch 收到请求")
    logger.debug(f"  Method: {request.method}")
    logger.debug(f"  URL: {request.url}")
    logger.debug(f"  Query params: {dict(request.query_params)}")
    logger.debug(f"  Content-Type: {content_type}")
    
    # 1. 尝试 multipart/form-data
    if 'multipart/form-data' in content_type:
        try:
            form = await request.form()
            logger.debug(f"  Form fields: {list(form.keys())}")
            form_dict = {}
            for key, value in form.items():
                # 如果是文件对象，读取内容
                if hasattr(value, 'read'):
                    content = await value.read()
                    form_dict[key] = content.decode('utf-8')
                else:
                    form_dict[key] = value
                logger.debug(f"    {key}: {str(form_dict[key])[:200]}")
            
            # 优先：JSON 文件上传（对方传 json 文件，文件内包含 tasks 等）
            _json_file_keys = ('file', 'json_file', 'json', 'tasks_file', 'tasks_json')
            for fkey in _json_file_keys:
                if fkey not in form_dict:
                    continue
                raw = form_dict[fkey]
                if not raw or not isinstance(raw, str):
                    continue
                raw = raw.strip()
                if not (raw.startswith('{') or raw.startswith('[')):
                    continue
                try:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        req = BatchRequest(tasks=data, callback_url=form_dict.get('callback_url'))
                    else:
                        req = BatchRequest(
                            tasks=data.get('tasks', []),
                            callback_url=data.get('callback_url') or form_dict.get('callback_url'),
                        )
                    if req.tasks:
                        logger.info(f"  ✓ 从 form-data 文件字段 '{fkey}' 解析 JSON 成功: {len(req.tasks)} 个任务")
                        break
                except Exception as e:
                    logger.warning(f"  解析 form 字段 '{fkey}' 为 JSON 失败: {e}")
            else:
                req = None
            
            # 对方发送的格式：分散的字段，不是 tasks 数组
            if not req and all(k in form_dict for k in ['task_id', 'case_name', 'prod_url']):
                # detail_id 可选，如果没有就用 task_id
                detail_id = form_dict.get('detail_id', form_dict.get('task_id'))
                
                # test_arry 处理：可能是 JSON 字符串，或者多个字段
                test_arry = []
                if 'test_arry' in form_dict:
                    try:
                        test_arry = json.loads(form_dict['test_arry'])
                    except:
                        test_arry = [form_dict['test_arry']]
                elif 'test_array' in form_dict:
                    try:
                        test_arry = json.loads(form_dict['test_array'])
                    except:
                        test_arry = [form_dict['test_array']]
                else:
                    # 查找所有 test_* 字段
                    test_fields = {k: v for k, v in form_dict.items() if k.startswith('test_')}
                    if test_fields:
                        test_arry = list(test_fields.values())
                
                # 如果没有 test_arry，设置默认值
                if not test_arry:
                    test_arry = ["默认测试步骤"]
                
                task = StartRequest(
                    task_id=form_dict.get('task_id'),
                    detail_id=detail_id,
                    case_name=form_dict.get('case_name'),
                    prod_url=form_dict.get('prod_url'),
                    test_arry=test_arry
                )
                callback_url = form_dict.get('callback_url')
                req = BatchRequest(tasks=[task], callback_url=callback_url)
                logger.info("  ✓ 从 form-data 解析成功: 1 个任务")
                logger.debug(f"    task_id={task.task_id}, detail_id={task.detail_id}")
                logger.debug(f"    case_name={task.case_name}")
                logger.debug(f"    test_arry={task.test_arry}")
            elif not req and 'tasks' in form_dict:
                # tasks 字段包含 JSON 字符串
                tasks_json = json.loads(form_dict['tasks'])
                callback_url = form_dict.get('callback_url')
                req = BatchRequest(tasks=tasks_json, callback_url=callback_url)
                logger.info(f"  ✓ 从 form-data (tasks JSON) 解析成功: {len(req.tasks)} 个任务")
            
            if req:
                logger.info(f"  最终解析: {len(req.tasks)} 个任务, callback={req.callback_url}")
        except Exception as e:
            logger.exception(f"  Form-data 解析失败: {e}")
    
    # 2. 尝试从 JSON body 解析
    if not req and 'application/json' in content_type:
        try:
            body = await request.json()
            logger.debug(f"  Body (JSON): {json.dumps(body, ensure_ascii=False)[:200]}")
            req = BatchRequest(**body)
            logger.info("  ✓ 从 JSON 解析成功")
        except Exception as e:
            logger.debug(f"  JSON 解析失败: {e}")
    
    # 3. 尝试从原始 body 解析
    if not req:
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode('utf-8')
            if body_text:
                logger.debug(f"  Body (raw): {body_text[:500]}")
                body_data = json.loads(body_text)
                req = BatchRequest(**body_data)
                logger.info("  ✓ 从原始 body 解析成功")
        except Exception as e:
            logger.debug(f"  原始 body 解析失败: {e}")
    
    # 4. 尝试从 query parameters 读取
    if not req:
        query_params = dict(request.query_params)
        if query_params:
            try:
                if "tasks" in query_params:
                    tasks_json = json.loads(query_params["tasks"])
                    callback_url = query_params.get("callback_url")
                    req = BatchRequest(tasks=tasks_json, callback_url=callback_url)
                    logger.info("  ✓ 从 query 参数解析成功")
            except Exception as e:
                logger.debug(f"  Query 参数解析失败: {e}")
    
    if not req or not req.tasks:
        return {
            "code": 1, 
            "message": "无法解析请求，请确保提供 tasks 数据（支持 JSON body 或 form-data）", 
            "data": {"success": False}
        }

    running_ids = [t.detail_id for t in req.tasks if _task_status.get(t.detail_id, {}).get("status") == "running"]
    if running_ids:
        return {
            "code": 1,
            "message": f"任务执行中: {running_ids}",
            "data": {"success": False},
        }

    callback_url = (_full_cfg.get("callback_base_url") or CALLBACK_BASE_URL or req.callback_url or "").strip().rstrip("/")
    if not callback_url:
        return {"code": 1, "message": "未配置 callback_base_url 或 callback-url", "data": {"success": False}}

    logger.info(f"[API] /batch 收到 {len(req.tasks)} 个任务，已接单并后台执行（全部完成后一次性回调）")
    run_group_ts = datetime.now().strftime("%Y%m%d%H%M")
    start_index_map = {}
    async with _counter_lock:
        for t in req.tasks:
            counter_key = f"{t.task_id}/{t.case_name}"
            start_index = _case_counter.get(counter_key, 0)
            _case_counter[counter_key] = start_index + len(t.test_arry)
            start_index_map[t.detail_id] = start_index
    asyncio.create_task(_run_batch_then_callback(req, start_index_map, callback_url, run_group_ts))
    return {"code": 0, "message": "ok", "data": {"success": True}}


@app.get("/status")
async def list_status():
    return {"total": len(_task_status), "tasks": _task_status}


@app.get("/status/{detail_id}")
async def get_status(detail_id: int):
    if detail_id not in _task_status:
        raise HTTPException(404, "任务不存在")
    return _task_status[detail_id]


@app.get("/health")
async def health():
    return {"status": "ok", "workers": _executor._max_workers if _executor else 0}


# 本地测试用：接收自身回调，直接存到 _callback_results，无需启动 receive_callback.py
_callback_results: Dict[str, dict] = {}

@app.post("/api/v1/agent-eval/notice/detail/{detail_id}")
async def receive_callback(detail_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    _callback_results[detail_id] = body
    logger.info(f"[LocalCallback] detail_id={detail_id} 回调已接收，test_cases={len(body.get('test_cases', []))}")
    return {"data": {"success": True}, "code": 0, "msg": "ok"}


@app.get("/callback_results")
async def list_callback_results():
    return {"total": len(_callback_results), "results": _callback_results}


def main():
    global _executor, _preset, _full_cfg, CALLBACK_BASE_URL, _worker_slots

    parser = argparse.ArgumentParser(description="AppEval API Server")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--workers", "--worker", type=int, default=None, dest="workers")
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--callback-url", type=str, default=None)
    parser.add_argument("--no-callback", dest="no_callback", action="store_true", help="本地测试不回调对方")
    args = parser.parse_args()

    if args.callback_url:
        CALLBACK_BASE_URL = args.callback_url.rstrip("/")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    workers, _preset, _full_cfg = load_config(str(config_path))
    if args.workers is not None:
        workers = args.workers
    if args.no_callback:
        _full_cfg["skip_callback"] = True
    # 未通过命令行指定时，用配置文件里的 callback_base_url（本地测试可设为 http://127.0.0.1:9999）
    if not args.callback_url and _full_cfg.get("callback_base_url"):
        CALLBACK_BASE_URL = _full_cfg["callback_base_url"].strip().rstrip("/")

    _preset["cuda_devices"] = _full_cfg.get("cuda_devices", [0])
    _apply_llm_env(_preset, for_local=(_full_cfg.get("model") == "local"))

    _executor = ProcessPoolExecutor(max_workers=workers)
    _worker_slots = asyncio.Queue(maxsize=workers)
    for _i in range(workers):
        _worker_slots.put_nowait(_i)
    API_LOG_BASE.mkdir(parents=True, exist_ok=True)
    callback_hint = "已关闭(本地)" if _full_cfg.get("skip_callback") else CALLBACK_BASE_URL
    _lan_ip = "127.0.0.1"
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        _lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    logger.info(f"AppEval API | port={args.port} workers={workers} model={_full_cfg.get('model','?')} callback={callback_hint}")
    logger.info(f"  内网服务地址: http://{_lan_ip}:{args.port}")
    logger.info(f"  并行: 最多 {workers} 个任务同时执行 (ProcessPoolExecutor)")
    logger.info(f"  api_log: {API_LOG_BASE}")
    for r in app.routes:
        if hasattr(r, "methods") and hasattr(r, "path"):
            logger.info(f"  注册路由: {list(r.methods)} {r.path}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
