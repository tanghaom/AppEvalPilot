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
from pydantic import BaseModel

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
) -> dict:
    """单任务在子进程中执行，环境与 test2 run_test worker 一致。"""
    print(f"[Worker] pid={os.getpid()} worker_id={worker_id} detail_id={detail_id} 开始执行", flush=True)
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
    try:
        from appeval.roles.eval_runner import AppEvalRole

        api_log_dir = str(API_LOG_BASE)
        os.makedirs(api_log_dir, exist_ok=True)
        role = AppEvalRole(
            config_file=config_file,
            remote_debugging_port=port,
            user_data_dir=user_data_dir,
            use_chrome_debugger=False,
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
            else:
                result_value = str(res)
                evidence = str(res)
            passed = result_value.lower().strip() in ("pass", "true", "1")
            cases.append({
                "test_id": f"{case_name}{key}",
                "case_desc": desc,
                "evidence": evidence,
                "result": passed,
                "cost": f"time={avg_time:.1f}s, usd=${avg_usd:.6f}",
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
                        else:
                            rv, ev = str(r2), str(r2)
                        cases.append({
                            "test_id": f"{case_name}{key2}",
                            "case_desc": desc,
                            "evidence": ev,
                            "result": rv.lower().strip() in ("pass", "true", "1"),
                            "cost": f"time={avg_time:.1f}s, usd=${avg_usd:.6f}",
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

        return {
            "detail_id": str(detail_id),
            "case_name": case_name,
            "eval_cost": f"avg_time={avg_time:.1f}s, avg_usd=${avg_usd:.6f}",
            "version": llm_cfg.get("model", ""),
            "test_cases": cases,
        }
    except Exception as e:
        traceback.print_exc()
        cases = [{
            "test_id": f"{case_name}{start_index + i}",
            "case_desc": d,
            "evidence": f"Error: {e}",
            "result": False,
            "cost": "time=0.0s, usd=$0.000000",
        } for i, d in enumerate(test_arry)]

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

        return {
            "detail_id": str(detail_id),
            "case_name": case_name,
            "eval_cost": "avg_time=0.0s, avg_usd=$0.000000",
            "version": llm_cfg.get("model", ""),
            "test_cases": cases,
        }
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
        print(f"[Callback] detail_id={detail_id} 已跳过（本地测试，不回调对方）")
        return
    url = f"{CALLBACK_BASE_URL}/{detail_id}"
    payload = [result]
    retry_times = int(_full_cfg.get("callback_retry_times", 3) or 3)
    retry_interval_sec = int(_full_cfg.get("callback_retry_interval_sec", 300) or 300)
    retry_times = max(1, retry_times)
    retry_interval_sec = max(1, retry_interval_sec)
    print(f"[Callback] POST {url} | body 前 300 字: {json.dumps(payload, ensure_ascii=False)[:300]}")

    for attempt in range(1, retry_times + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                resp = await c.post(url, json=payload)
                if resp.status_code >= 400:
                    print(
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
                        print(
                            f"[Callback] detail_id={detail_id} 成功: HTTP {resp.status_code} "
                            f"| code={callback_code} msg={callback_msg}"
                        )
                        return
                    print(
                        f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: "
                        f"回调响应 data.success!=true | HTTP {resp.status_code} | body={resp.text[:500]}"
                    )
        except Exception as e:
            err_msg = str(e).strip() or repr(e)
            print(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次异常: {type(e).__name__} {err_msg}")

        if attempt < retry_times:
            print(f"[Callback] detail_id={detail_id} {retry_interval_sec}s 后重试...")
            await asyncio.sleep(retry_interval_sec)

    print(f"[Callback] detail_id={detail_id} 回调最终失败，已达最大重试次数 {retry_times}")


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


async def _callback_batch(callback_url: str, results: list):
    """batch 专用：全部任务完成后，按条回调；每条 POST 到 .../detail/{detail_id}，body 为 {"results": [单条]}。"""
    if _should_skip_callback():
        print("[Callback] batch 已跳过（本地测试不回调）")
        return
    retry_times = int(_full_cfg.get("callback_retry_times", 3) or 3)
    retry_interval_sec = int(_full_cfg.get("callback_retry_interval_sec", 300) or 300)
    retry_times = max(1, retry_times)
    retry_interval_sec = max(1, retry_interval_sec)
    base_url = callback_url.strip()
    print(f"[Callback] batch 全部完成，按 detail_id 分别回调共 {len(results)} 条")
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=True) as c:
        for result in results:
            detail_id = result.get("detail_id", "")
            url = _callback_url_with_detail_id(base_url, detail_id)
            # 对方接口要求 eval_cost、test_id、cost 为整数，先做转换
            payload = _transform_result_for_callback(result)
            for attempt in range(1, retry_times + 1):
                try:
                    resp = await c.post(url, json=payload)
                    if resp.status_code >= 400:
                        print(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: HTTP {resp.status_code} | {resp.text[:500]}")
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
                            print(f"[Callback] detail_id={detail_id} 成功: HTTP {resp.status_code}")
                            break
                        print(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次失败: 回调需 data.success 或 ok=true | {resp.text[:500]}")
                except Exception as e:
                    print(f"[Callback] detail_id={detail_id} 第{attempt}/{retry_times}次异常: {type(e).__name__} {e}")
                if attempt < retry_times:
                    await asyncio.sleep(retry_interval_sec)
            else:
                print(f"[Callback] detail_id={detail_id} 回调最终失败，已达最大重试次数 {retry_times}")


async def _dispatch(req: StartRequest, start_index: int, run_group_ts: Optional[str] = None):
    max_workers = _executor._max_workers if _executor else 1
    worker_id = hash(str(req.detail_id)) % max_workers
    _task_status[req.detail_id] = {"status": "running", "case_name": req.case_name}
    print(f"[API] 提交任务 detail_id={req.detail_id} worker_id={worker_id} "
          f"start_index={start_index} (并行槽位 0~{max_workers - 1})", flush=True)
    loop = asyncio.get_event_loop()
    args_tuple = (
        worker_id, req.detail_id, req.task_id, req.case_name,
        req.prod_url, req.test_arry,
        dict(_preset), dict(_full_cfg), start_index, run_group_ts,
    )
    try:
        result = await loop.run_in_executor(
            _executor,
            _run_task_in_process_wrapper,
            args_tuple,
        )
        _task_status[req.detail_id] = {"status": "done", "result": result}
        print(f"[API] detail_id={req.detail_id} 任务完成，准备回调")
    except Exception as e:
        _task_status[req.detail_id] = {"status": "error", "error": str(e)}
        result = {
            "detail_id": str(req.detail_id), "case_name": req.case_name,
            "eval_cost": "avg_time=0.0s, avg_usd=$0.000000", "version": "",
            "test_cases": [
                {
                    "test_id": f"{req.case_name}{start_index + i}",
                    "case_desc": d,
                    "evidence": f"Error: {e}",
                    "result": False,
                    "cost": "time=0.0s, usd=$0.000000",
                }
                for i, d in enumerate(req.test_arry)
            ],
        }
        print(f"[API] detail_id={req.detail_id} 任务异常: {e}，准备回调")
    await _callback(req.detail_id, result)


@app.post("/start", response_model=StartResponse)
async def start_task(req: StartRequest):
    print(f"[API] task_id={req.task_id} detail_id={req.detail_id} "
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
    """后台执行本批全部任务，完成后一次性回调推送完整 results。"""
    loop = asyncio.get_event_loop()
    max_workers = _executor._max_workers if _executor else 1
    for t in req.tasks:
        _task_status[t.detail_id] = {"status": "running", "case_name": t.case_name}
    futures = []
    for t in req.tasks:
        worker_id = hash(str(t.detail_id)) % max_workers
        args_tuple = (
            worker_id, t.detail_id, t.task_id, t.case_name,
            t.prod_url, t.test_arry,
            dict(_preset), dict(_full_cfg), start_index_map[t.detail_id], run_group_ts,
        )
        fut = loop.run_in_executor(_executor, _run_task_in_process_wrapper, args_tuple)
        futures.append((t.detail_id, fut))
    results = []
    for detail_id, fut in futures:
        try:
            result = await fut
            _task_status[detail_id] = {"status": "done", "result": result}
            results.append(result)
        except Exception as e:
            t = next((x for x in req.tasks if x.detail_id == detail_id), None)
            test_arry = t.test_arry if t else []
            start_index = start_index_map.get(detail_id, 0)
            _task_status[detail_id] = {"status": "error", "error": str(e)}
            results.append({
                "detail_id": str(detail_id),
                "case_name": t.case_name if t else "",
                "eval_cost": "avg_time=0.0s, avg_usd=$0.000000",
                "version": "",
                "test_cases": [
                    {
                        "test_id": f"{t.case_name}{start_index + i}" if t else str(start_index + i),
                        "case_desc": d,
                        "evidence": f"Error: {e}",
                        "result": False,
                        "cost": "time=0.0s, usd=$0.000000",
                    }
                    for i, d in enumerate(test_arry)
                ],
            })
        print(f"[API] /batch detail_id={detail_id} 完成")
    detail_to_group = {t.detail_id: (t.task_id, t.case_name) for t in req.tasks}
    group_cost_acc = {}
    for r in results:
        try:
            detail_id = int(str(r.get("detail_id", "0")))
        except Exception:
            continue
        group_key = detail_to_group.get(detail_id)
        if not group_key:
            continue
        t_cost, u_cost = _parse_eval_cost(r.get("eval_cost", ""))
        if group_key not in group_cost_acc:
            group_cost_acc[group_key] = [0.0, 0.0, 0]
        group_cost_acc[group_key][0] += t_cost
        group_cost_acc[group_key][1] += u_cost
        group_cost_acc[group_key][2] += 1
    for r in results:
        try:
            detail_id = int(str(r.get("detail_id", "0")))
        except Exception:
            continue
        group_key = detail_to_group.get(detail_id)
        if not group_key or group_key not in group_cost_acc:
            continue
        sum_t, sum_u, cnt = group_cost_acc[group_key]
        if cnt <= 0:
            continue
        r["eval_cost"] = f"avg_time={sum_t / cnt:.1f}s, avg_usd=${sum_u / cnt:.6f}"
    print(f"[API] /batch 全部 {len(results)} 个任务完成，一次性回调")
    await _callback_batch(callback_url, results)


@app.post("/batch")
async def batch_tasks(request: Request):
    """接收一批任务，仅返回接单结果；全部任务完成后一次性回调推送完整结果数组。
    支持三种方式：
    1. POST body (JSON) - 标准方式
    2. multipart/form-data - 对方使用的方式
    3. URL 参数
    """
    req = None
    
    # 打印调试信息
    content_type = request.headers.get('content-type', '').lower()
    print(f"[API] /batch 收到请求:")
    print(f"  Method: {request.method}")
    print(f"  URL: {request.url}")
    print(f"  Query params: {dict(request.query_params)}")
    print(f"  Content-Type: {content_type}")
    
    # 1. 尝试 multipart/form-data
    if 'multipart/form-data' in content_type:
        try:
            form = await request.form()
            print(f"  Form fields: {list(form.keys())}")
            form_dict = {}
            for key, value in form.items():
                # 如果是文件对象，读取内容
                if hasattr(value, 'read'):
                    content = await value.read()
                    form_dict[key] = content.decode('utf-8')
                else:
                    form_dict[key] = value
                print(f"    {key}: {str(form_dict[key])[:200]}")
            
            # 对方发送的格式：分散的字段，不是 tasks 数组
            # 检查必填字段
            if all(k in form_dict for k in ['task_id', 'case_name', 'prod_url']):
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
                print(f"  ✓ 从 form-data 解析成功: 1 个任务")
                print(f"    task_id={task.task_id}, detail_id={task.detail_id}")
                print(f"    case_name={task.case_name}")
                print(f"    test_arry={task.test_arry}")
            elif 'tasks' in form_dict:
                # tasks 字段包含 JSON 字符串
                tasks_json = json.loads(form_dict['tasks'])
                callback_url = form_dict.get('callback_url')
                req = BatchRequest(tasks=tasks_json, callback_url=callback_url)
                print(f"  ✓ 从 form-data (tasks JSON) 解析成功: {len(req.tasks)} 个任务")
            
            if req:
                print(f"  最终解析: {len(req.tasks)} 个任务, callback={req.callback_url}")
        except Exception as e:
            print(f"  Form-data 解析失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. 尝试从 JSON body 解析
    if not req and 'application/json' in content_type:
        try:
            body = await request.json()
            print(f"  Body (JSON): {json.dumps(body, ensure_ascii=False)[:200]}")
            req = BatchRequest(**body)
            print(f"  ✓ 从 JSON 解析成功")
        except Exception as e:
            print(f"  JSON 解析失败: {e}")
    
    # 3. 尝试从原始 body 解析
    if not req:
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode('utf-8')
            if body_text:
                print(f"  Body (raw): {body_text[:500]}")
                body_data = json.loads(body_text)
                req = BatchRequest(**body_data)
                print(f"  ✓ 从原始 body 解析成功")
        except Exception as e:
            print(f"  原始 body 解析失败: {e}")
    
    # 4. 尝试从 query parameters 读取
    if not req:
        query_params = dict(request.query_params)
        if query_params:
            try:
                if "tasks" in query_params:
                    tasks_json = json.loads(query_params["tasks"])
                    callback_url = query_params.get("callback_url")
                    req = BatchRequest(tasks=tasks_json, callback_url=callback_url)
                    print(f"  ✓ 从 query 参数解析成功")
            except Exception as e:
                print(f"  Query 参数解析失败: {e}")
    
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

    callback_url = (req.callback_url or _full_cfg.get("callback_base_url") or CALLBACK_BASE_URL or "").strip().rstrip("/")
    if not callback_url:
        return {"code": 1, "message": "未配置 callback_base_url 或 callback-url", "data": {"success": False}}

    print(f"[API] /batch 收到 {len(req.tasks)} 个任务，已接单并后台执行（全部完成后一次性回调）")
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


def main():
    global _executor, _preset, _full_cfg, CALLBACK_BASE_URL

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
    API_LOG_BASE.mkdir(parents=True, exist_ok=True)
    callback_hint = "已关闭(本地)" if _full_cfg.get("skip_callback") else CALLBACK_BASE_URL
    print(f"AppEval API | port={args.port} workers={workers} model={_full_cfg.get('model','?')} callback={callback_hint}")
    print(f"  并行: 最多 {workers} 个任务同时执行 (ProcessPoolExecutor)")
    print(f"  api_log: {API_LOG_BASE}")
    for r in app.routes:
        if hasattr(r, "methods") and hasattr(r, "path"):
            print(f"  注册路由: {list(r.methods)} {r.path}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
