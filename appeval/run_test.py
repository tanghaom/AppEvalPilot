#!/usr/bin/env python3
"""
独立测试执行引擎 —— 与服务层解耦，可被 server_v2.py 调用，也可 CLI 直接跑。

用法:
  # 1. 跑单个任务 JSON（由 server 生成，或手动构建）
  python -m appeval.run_test --task-json api_log/tasks/2537.json --config appeval/api_server/config.yaml --worker-id 0

  # 2. 从 server_v2.py 进程池调用
  from appeval.run_test import run_single_task
  result = run_single_task("task.json", "config.yaml", worker_id=0)

Task JSON 格式:
  {
    "task_id": 138,
    "detail_id": 2537,
    "case_name": "MenuExpress",
    "prod_url": "https://xxx.netlify.app",
    "test_arry": ["step 1 desc", "step 2 desc"],
    "start_index": 0,
    "run_group_ts": "202602261513"
  }
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
from pathlib import Path
from typing import List, Optional

import httpx
import yaml
from loguru import logger

PROJECT_DIR = Path(__file__).resolve().parent.parent
API_LOG_BASE = PROJECT_DIR / "appeval" / "api_log"
SHARED_MODEL_CACHE = os.path.join(PROJECT_DIR, ".cache", "modelscope")


# ---------------------------------------------------------------------------
#  Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> tuple:
    """Load config.yaml → (preset, full_cfg)."""
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    model = (cfg.get("model") or "remote").strip().lower()
    if model not in ("local", "remote"):
        model = "remote"
    preset = cfg.get(model, {})
    return preset, cfg


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


def _resolve_resume_checkpoint_path(path: str, target_step: Optional[int]) -> str:
    """Resolve resume checkpoint path for per-step rollback.

    Supports:
    - direct step json: /.../checkpoints/step_005.json
    - checkpoint index: /.../checkpoints/checkpoint_index.json + target_step
    - run dir: /.../<case>/<timestamp>/ + target_step
    - legacy latest: /.../resume_checkpoint.json (when target_step is None)
    """
    if not path:
        return ""

    p = Path(path)
    if p.is_file():
        if p.name == "checkpoint_index.json" and target_step is not None:
            try:
                idx = json.loads(p.read_text(encoding="utf-8"))
                steps = idx.get("steps", {})
                step_item = steps.get(str(int(target_step)), {})
                resolved = str(step_item.get("checkpoint", "") or "")
                if resolved:
                    return resolved
            except Exception:
                pass
        return str(p)

    if p.is_dir():
        if target_step is not None:
            step_file = p / "checkpoints" / f"step_{int(target_step):03d}.json"
            if step_file.exists():
                return str(step_file)
        legacy = p / "resume_checkpoint.json"
        if legacy.exists():
            return str(legacy)

    return str(p)


# ---------------------------------------------------------------------------
#  Environment setup helpers (Xvfb / D-Bus / AT-SPI / WM / Xauthority)
# ---------------------------------------------------------------------------

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
        for path in ("/usr/libexec/at-spi-bus-launcher", "/usr/libexec/at-spi2-registryd"):
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
    auth_file = f"/tmp/.Xauthority_run_w{worker_id}"
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


# ---------------------------------------------------------------------------
#  Result / cost helpers
# ---------------------------------------------------------------------------

def _parse_cost_to_seconds(cost_str: str) -> int:
    if not cost_str or not isinstance(cost_str, str):
        return 0
    m = re.search(r"(?:avg_)?time[=:]?\s*([\d.]+)\s*s", cost_str, re.I)
    if m:
        try:
            return int(float(m.group(1)))
        except (ValueError, TypeError):
            pass
    return 0


FALLBACK_PRICING = {
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
    price = FALLBACK_PRICING.get(model_name or "")
    if not price:
        return 0.0
    return (pt * float(price["prompt"]) + ct * float(price["completion"])) / 1000.0


def _get_total_usd(role, llm_config: dict) -> float:
    total_usd = 0.0
    try:
        if hasattr(role.test_generator, "llm") and hasattr(role.test_generator.llm, "get_costs"):
            c = role.test_generator.llm.get_costs()
            mn = getattr(role.test_generator.llm, "model", "") or llm_config.get("model", "")
            total_usd += _cost_to_usd(c, mn)
        if hasattr(role, "osagent") and role.osagent and hasattr(role.osagent, "llm") and hasattr(role.osagent.llm, "get_costs"):
            c = role.osagent.llm.get_costs()
            mn = getattr(role.osagent.llm, "model", "") or llm_config.get("model", "")
            total_usd += _cost_to_usd(c, mn)
    except Exception:
        pass
    return total_usd


def _do_callback_sync(url: str, payload: dict, retry_times: int = 3, retry_interval: int = 30):
    """子进程内同步回调。"""
    detail_id = payload.get("detail_id", "")
    for attempt in range(1, retry_times + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as c:
                resp = c.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning(f"[Runner-CB] detail_id={detail_id} 第{attempt}/{retry_times}次失败: HTTP {resp.status_code}")
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
                    logger.info(f"[Runner-CB] detail_id={detail_id} 回调成功")
                    return
                logger.warning(f"[Runner-CB] detail_id={detail_id} 第{attempt}/{retry_times}次 data.success!=true")
        except Exception as e:
            logger.error(f"[Runner-CB] detail_id={detail_id} 第{attempt}/{retry_times}次异常: {type(e).__name__} {e}")
        if attempt < retry_times:
            time.sleep(retry_interval)
    logger.error(f"[Runner-CB] detail_id={detail_id} 回调最终失败")


def _transform_result_for_callback(result: dict) -> dict:
    """转换结果为回调格式（test_id/cost 为整数）。"""
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
        c["test_id"] = max(1, raw)
        c["cost"] = _parse_cost_to_seconds(str(tc.get("cost", "")))
        cases.append(c)
    out["test_cases"] = cases
    return out


def _callback_url_with_detail_id(base_url: str, detail_id) -> str:
    base_url = base_url.strip().rstrip("/")
    base_url = re.sub(r"/\d+$", "", base_url)
    return f"{base_url}/{detail_id}"


def _worker_callback(callback_url: str, full_cfg: dict, result: dict):
    """在子进程内执行回调，落盘后立即调用。"""
    if not callback_url or full_cfg.get("skip_callback"):
        logger.info(f"[Runner-CB] detail_id={result.get('detail_id')} 跳过回调")
        return
    try:
        transformed = _transform_result_for_callback(result)
        url = _callback_url_with_detail_id(callback_url, result.get("detail_id", ""))
        retry_times = max(1, int(full_cfg.get("callback_retry_times", 3) or 3))
        retry_interval = max(1, int(full_cfg.get("callback_retry_interval_sec", 30) or 30))
        _do_callback_sync(url, transformed, retry_times, retry_interval)
    except Exception as e:
        logger.error(f"[Runner-CB] 回调失败: {e}")


def _write_per_case_json(cases: list, case_name: str, task_id, run_group_ts: str):
    """Write test_case.json for each individual case under api_log."""
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


# ---------------------------------------------------------------------------
#  Core execution function
# ---------------------------------------------------------------------------

def run_single_task(
    task_json_path: str,
    config_path: str,
    worker_id: int = 0,
) -> dict:
    """Execute a single test task defined by *task_json_path*.

    This is the sole entry-point for test execution, used by both:
      - server_v2.py (via ProcessPoolExecutor)
      - CLI (``python -m appeval.run_test --task-json ...``)

    Returns a result dict compatible with the callback contract.
    """
    # ---- 读取任务 JSON ----
    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    task_id = task["task_id"]
    detail_id = task["detail_id"]
    case_name = task["case_name"]
    prod_url = task["prod_url"]
    test_arry: List[str] = task["test_arry"]
    start_index: int = task.get("start_index", 0)
    run_group_ts: str = task.get("run_group_ts") or datetime.now().strftime("%Y%m%d%H%M")
    callback_url: str = task.get("callback_url", "")

    # resume_mode: read from task JSON
    #   "resume_mode": true             → save checkpoint at end (round 1)
    #   "resume_checkpoint_path": "..."  → load checkpoint from previous round (round 2)
    resume_mode: bool = bool(task.get("resume_mode", False))
    resume_checkpoint_path: str = str(task.get("resume_checkpoint_path", "") or "")
    resume_target_step = task.get("resume_target_step", None)
    save_checkpoint_per_step: bool = bool(task.get("save_checkpoint_per_step", False))
    save_profile_per_step: bool = bool(task.get("save_profile_per_step", False))
    resume_checkpoint_path = _resolve_resume_checkpoint_path(
        resume_checkpoint_path,
        int(resume_target_step) if resume_target_step is not None else None,
    )

    # ---- 加载配置 ----
    preset, full_cfg = load_config(config_path)
    llm_config = preset.get("llm", {})

    logger.info(
        f"[Runner] pid={os.getpid()} worker={worker_id} "
        f"detail_id={detail_id} task_id={task_id} case={case_name} "
        f"cases={len(test_arry)} url={prod_url}"
    )

    # ---- 进程级初始化 ----
    warnings.filterwarnings("ignore", message=".*xauthority.*")
    sys.stderr = _StderrFilter(sys.stderr)

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    os.chdir(str(PROJECT_DIR))

    # 模型缓存
    worker_cache = f"/tmp/modelscope_run_w{worker_id}"
    os.environ["MODELSCOPE_CACHE"] = worker_cache
    if os.path.exists(SHARED_MODEL_CACHE) and not os.path.exists(os.path.join(worker_cache, "hub")):
        shutil.copytree(SHARED_MODEL_CACHE, worker_cache, dirs_exist_ok=True)
    os.makedirs(worker_cache, exist_ok=True)

    # GPU
    cuda_devices = preset.get("cuda_devices", full_cfg.get("cuda_devices", [0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices[worker_id % len(cuda_devices)])

    # LLM 环境变量
    _apply_llm_env(preset, for_local=(preset.get("config_file", "").find("local") >= 0))
    if full_cfg.get("chrome_disable_web_security"):
        os.environ["APPEVAL_CHROME_DISABLE_WEB_SECURITY"] = "1"

    # Display / port / user_data_dir
    base_display = int(preset.get("base_display", 300)) + 200
    base_chrome_port = int(preset.get("base_chrome_port", 9500)) + 2000
    base_log_prefix = preset.get("log_dir_prefix", "api")
    main_model_raw = llm_config.get("model", "unknown")
    safe_model = re.sub(r'[/\s:\\*?"<>|]', "_", main_model_raw)
    log_dir_prefix = f"{base_log_prefix}_{safe_model}"

    # CaseGenerator / TellVerifier 临时配置文件
    case_generator_config = preset.get("case_generator", llm_config)
    tell_verifier_config = full_cfg.get("tell_verifier", {})
    temp_config_file = f"/tmp/appeval_run_cfg_w{worker_id}.yaml"
    with open(temp_config_file, "w", encoding="utf-8") as f:
        yaml.dump({
            "llm": llm_config,
            "case_generator": case_generator_config,
            "tell_verifier": tell_verifier_config,
        }, f, default_flow_style=False)

    display_num = base_display + worker_id
    port = base_chrome_port + worker_id
    user_data_dir = f"/tmp/chrome_run_{log_dir_prefix}_w{worker_id}_d{detail_id}"

    # 错峰启动
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

    # ---- 构造 test_cases ----
    test_cases = {
        str(start_index + i): {"case_desc": desc, "result": "", "evidence": ""}
        for i, desc in enumerate(test_arry)
    }

    def _default_case_item(i: int, err: str) -> dict:
        return {
            "test_id": f"{case_name}{start_index + i}",
            "case_desc": test_arry[i],
            "evidence": f"Error: {err}",
            "result": False,
            "cost": "time=0.0s, usd=$0.000000",
        }

    def _load_partial_cases(err: str) -> list:
        """从已写入的 test_case.json 恢复已完成的 case 结果。"""
        partial: list = []
        for i, _desc in enumerate(test_arry):
            case_num = start_index + i
            case_dir = API_LOG_BASE / str(task_id) / run_group_ts / case_name / f"{case_name}{case_num}"
            candidates: list = []
            try:
                ts_dirs = sorted(
                    [p for p in case_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{12,}", p.name)],
                    key=lambda p: p.name, reverse=True,
                )
                candidates.extend([p / "test_case.json" for p in ts_dirs])
            except Exception:
                pass
            candidates.append(case_dir / "test_case.json")

            loaded = None
            for fp in candidates:
                try:
                    if not fp.exists():
                        continue
                    data = json.loads(fp.read_text(encoding="utf-8"))
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

    # ---- 执行测试 ----
    result_dict: dict = {}
    try:
        from appeval.roles.eval_runner import AppEvalRole

        api_log_dir = str(API_LOG_BASE)
        os.makedirs(api_log_dir, exist_ok=True)

        role = AppEvalRole(
            config_file=temp_config_file,
            remote_debugging_port=port,
            user_data_dir=user_data_dir,
            use_chrome_debugger=False,
            agent_class=preset.get("agent_class", "osagent"),
            a11y_mode=preset.get("a11y_mode", "atspi"),
            max_iters=preset.get("max_iters", 15),
            run_id=f"run_{run_group_ts}_d{detail_id}",
            worker_id=worker_id,
            save_checkpoint_per_step=save_checkpoint_per_step,
            save_profile_per_step=save_profile_per_step,
            use_ocr=True,
            post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
            log_dirs=api_log_dir,
            use_timestamp_log_dir=True,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        t0 = time.perf_counter()

        # resume_mode=True  → round 1: save checkpoint after last case, keep profile
        # resume_checkpoint_path set → round 2: restore from checkpoint
        result, _ = loop.run_until_complete(
            role.run_api(
                task_name="test_case",
                test_cases=test_cases,
                start_func=prod_url,
                log_dir=f"{task_id}/{run_group_ts}/{case_name}",
                sequential_mode=True,
                case_name_for_log=case_name,
                resume_checkpoint_path=resume_checkpoint_path,
                save_checkpoint=resume_mode,
                chrome_profile_src=user_data_dir if resume_mode else "",
            )
        )
        elapsed = time.perf_counter() - t0
        loop.close()

        total_usd = _get_total_usd(role, llm_config)
        case_count = max(1, len(test_arry))
        avg_time = elapsed / case_count
        avg_usd = total_usd / case_count

        cases = _build_cases(result, test_arry, case_name, start_index, avg_time, avg_usd)

        # 黑屏重试（最后一个 case 判 Fail 且含黑屏关键词时触发）
        last = cases[-1] if cases else {}
        if not last.get("result") and _is_black_screen(str(last.get("evidence", ""))):
            retry_cases = _black_screen_retry(
                role_kwargs=dict(
                    config_file=temp_config_file,
                    remote_debugging_port=port,
                    user_data_dir=user_data_dir,
                    agent_class=preset.get("agent_class", "osagent"),
                    a11y_mode=preset.get("a11y_mode", "atspi"),
                    max_iters=preset.get("max_iters", 15),
                    post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
                    log_dirs=api_log_dir,
                ),
                test_cases=test_cases,
                prod_url=prod_url,
                task_id=task_id,
                run_group_ts=run_group_ts,
                case_name=case_name,
                test_arry=test_arry,
                start_index=start_index,
                avg_time=avg_time,
                avg_usd=avg_usd,
                user_data_dir=user_data_dir,
            )
            if retry_cases:
                cases = retry_cases

        _write_per_case_json(cases, case_name, task_id, run_group_ts)

        result_dict = {
            "detail_id": str(detail_id),
            "case_name": case_name,
            "eval_cost": f"avg_time={avg_time:.1f}s, avg_usd=${avg_usd:.6f}",
            "version": llm_config.get("model", ""),
            "test_cases": cases,
        }

        _worker_callback(callback_url, full_cfg, result_dict)

    except Exception as e:
        traceback.print_exc()
        err_msg = str(e).strip() or repr(e)
        cases = _load_partial_cases(err_msg)
        times = [_parse_cost_to_seconds(str(c.get("cost", ""))) for c in cases]
        avg_time = (sum(times) / max(1, len(times))) if times else 0.0

        _write_per_case_json(cases, case_name, task_id, run_group_ts)

        result_dict = {
            "detail_id": str(detail_id),
            "case_name": case_name,
            "eval_cost": f"avg_time={avg_time:.1f}s, avg_usd=$0.000000",
            "version": llm_config.get("model", ""),
            "test_cases": cases,
        }

        _worker_callback(callback_url, full_cfg, result_dict)

    finally:
        _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id)
        xvfb.terminate()
        xvfb.wait()
        # In resume_mode (round 1): profile is copied inside eval_runner before cleanup,
        # so we still delete the live working directory here.
        # In round 2 (resume_checkpoint_path set): the profile is already a copy;
        # delete the working dir as usual.
        shutil.rmtree(user_data_dir, ignore_errors=True)
        try:
            from appeval.tools.ocr import release_ocr_memory
            release_ocr_memory()
        except Exception:
            pass

    # 将聚合结果写到 task JSON 同目录
    result_path = str(Path(task_json_path).with_suffix("")) + "_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"[Runner] Result saved → {result_path}")

    return result_dict


# ---------------------------------------------------------------------------
#  Build / retry helpers
# ---------------------------------------------------------------------------

def _build_cases(result, test_arry, case_name, start_index, avg_time, avg_usd):
    cases = []
    for i, desc in enumerate(test_arry):
        key = str(start_index + i)
        res = (result or {}).get(key, {})
        if isinstance(res, dict):
            result_value = res.get("result", "Fail")
            evidence = res.get("evidence", "")
            res_cost = res.get("cost", "")
            cost_str = res_cost if (res_cost and "time=" in str(res_cost) and "usd=" in str(res_cost)) \
                else f"time={avg_time:.1f}s, usd=${avg_usd:.6f}"
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
    return cases


_BLACK_KWS = ["black screen", "blank screen", "black page", "blank page"]


def _is_black_screen(evidence: str) -> bool:
    ev = evidence.lower()
    return any(k in ev for k in _BLACK_KWS)


def _black_screen_retry(
    role_kwargs, test_cases, prod_url, task_id, run_group_ts, case_name,
    test_arry, start_index, avg_time, avg_usd, user_data_dir,
) -> Optional[list]:
    """Kill Chrome, rebuild user_data_dir, re-run. Returns new cases list or None."""
    try:
        from appeval.utils.window_utils import kill_windows
        from appeval.roles.eval_runner import AppEvalRole

        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)
        loop2.run_until_complete(kill_windows(user_data_dir=user_data_dir))
        loop2.close()
        time.sleep(2)
        shutil.rmtree(user_data_dir, ignore_errors=True)
        os.makedirs(user_data_dir, exist_ok=True)

        role2 = AppEvalRole(
            use_chrome_debugger=False,
            use_ocr=True,
            use_timestamp_log_dir=True,
            **role_kwargs,
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
            return _build_cases(result2, test_arry, case_name, start_index, avg_time, avg_usd)
    except Exception:
        traceback.print_exc()
    return None


# ---------------------------------------------------------------------------
#  ProcessPoolExecutor wrapper (used by server_v2.py)
# ---------------------------------------------------------------------------

def run_single_task_wrapper(args_tuple):
    """Unpack tuple for ProcessPoolExecutor compatibility."""
    task_json_path, config_path, worker_id = args_tuple
    return run_single_task(task_json_path, config_path, worker_id)


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AppEval 独立测试执行器")
    parser.add_argument("--task-json", required=True, help="Task JSON 文件路径")
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID (资源隔离)")
    args = parser.parse_args()

    result = run_single_task(args.task_json, args.config, args.worker_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
