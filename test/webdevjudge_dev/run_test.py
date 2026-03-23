#!/usr/bin/env python3
"""
webdevjudge_dev 跑测入口
从 case/web_*/task_*/metadata.json 枚举任务，URL=file://本地HTML，GUI agent 测试。
执行: conda activate appeval && cd /data/zhijieliu/AppEvalPilot/test/webdevjudge_dev && python run_test.py
"""
import os
import json
import signal
import shutil
import sys
import time
import subprocess
import asyncio
import yaml
from collections import defaultdict
from pathlib import Path
from multiprocessing import Process, Queue
import queue

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = "/data/zhijieliu/AppEvalPilot"
CONFIG_PATH = SCRIPT_DIR / "run_config.yaml"
CASE_DIR = SCRIPT_DIR / "case"
SHARED_MODEL_CACHE = os.path.join(PROJECT_DIR, ".cache", "modelscope")

sys.path.insert(0, PROJECT_DIR)


# ── 任务加载 ────────────────────────────────────────────────────────────────

def load_tasks_from_cases(case_dir: Path, web_filter=None, task_filter=None):
    """从 case/web_*/task_*/metadata.json 枚举所有任务，返回有序 list。

    每项包含:
      idx        - 全局顺序编号（0-based）
      web_id     - "web_2"
      task_id    - 1
      task_name  - "web_2_1"（与 label Excel 的 case_name 列对齐）
      url        - "file:///abs/path/case/web_2/web.html"
      instruction- 测试指令
      max_steps  - metadata 中的 max_steps
    """
    tasks = []
    idx = 0
    for web_dir in sorted(case_dir.iterdir(), key=lambda p: _web_sort_key(p.name)):
        if not web_dir.is_dir():
            continue
        web_id = web_dir.name  # e.g. "web_2"
        if web_filter and web_id not in web_filter:
            continue
        html_path = web_dir / "web.html"
        if not html_path.exists():
            print(f"[WARNING] {web_dir} 缺少 web.html，跳过")
            continue
        url = f"file://{html_path.resolve()}"
        for task_dir in sorted(web_dir.iterdir(), key=lambda p: _task_sort_key(p.name)):
            if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
                continue
            meta_path = task_dir / "metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            task_id = int(meta.get("task_id", task_dir.name.split("_")[-1]))
            if task_filter and task_id not in task_filter:
                continue
            tasks.append({
                "idx": idx,
                "web_id": web_id,
                "task_id": task_id,
                "task_name": f"{web_id}_{task_id}",   # e.g. "web_2_1"，与 label Excel 对齐
                "url": url,
                "instruction": meta.get("instruction", ""),
                "max_steps": int(meta.get("max_steps", 15)),
            })
            idx += 1
    return tasks


def _web_sort_key(name: str):
    try:
        return (0, int(name.split("_")[-1]))
    except Exception:
        return (1, name)


def _task_sort_key(name: str):
    try:
        return (0, int(name.split("_")[-1]))
    except Exception:
        return (1, name)


# ── 配置加载 ────────────────────────────────────────────────────────────────

def load_run_config(config_path=None):
    path = Path(config_path or CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = (cfg.get("model") or "remote").strip().lower()
    if model not in ("local", "remote"):
        model = "remote"
    preset = cfg.get(model, {})
    if not preset:
        raise ValueError(f"run_config.yaml 中缺少 '{model}' 配置段")
    tasks = int(cfg.get("tasks", 0))
    workers = int(cfg.get("workers", 5))
    serial_per_url = bool(cfg.get("serial_per_url", True))
    label_excel = cfg.get("label_excel", "")
    return tasks, workers, model, preset, cfg, serial_per_url, label_excel


def load_label_map(label_excel_path: str) -> dict:
    """从 label Excel 加载 {case_name: label} 映射，case_name 格式为 web_0_1。"""
    if not label_excel_path or not Path(label_excel_path).exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_excel(label_excel_path)
        label_map = {}
        for _, row in df.iterrows():
            cn = str(row.get("case_name", "") or "").strip()
            lbl = row.get("label")
            if cn and lbl is not None:
                try:
                    label_map[cn] = int(lbl)
                except Exception:
                    pass
        print(f"[标签] 从 {Path(label_excel_path).name} 加载 {len(label_map)} 条标签")
        return label_map
    except Exception as e:
        print(f"[标签] 加载失败: {e}")
        return {}


def apply_llm_env(preset, for_local=False):
    llm = preset.get("llm") or {}
    os.environ["llm__api_type"] = str(llm.get("api_type", "openai"))
    os.environ["llm__model"] = str(llm.get("model", ""))
    os.environ["llm__base_url"] = str(llm.get("base_url", ""))
    os.environ["llm__api_key"] = str(llm.get("api_key", ""))
    os.environ["llm__stream"] = str(llm.get("stream", "false"))
    if "max_token" in llm:
        os.environ["llm__max_token"] = str(llm["max_token"])
    if for_local:
        for k in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[k] = "localhost,127.0.0.1" if "proxy" not in k.lower() or "no" in k.lower() else ""


def resolve_resume_checkpoint_path(path: str, target_step):
    """Resolve resume checkpoint path for per-step rollback."""
    if not path:
        return ""

    p = Path(path)
    if p.is_file():
        if p.name == "checkpoint_index.json" and target_step is not None:
            try:
                idx = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
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


def resolve_task_resume_checkpoint_path(base_path: str, task_name: str, target_step):
    """Resolve per-task checkpoint when base_path points to a folder with many tasks."""
    if not base_path:
        return ""

    p = Path(base_path)
    task_dir = p / task_name
    if p.is_dir() and task_dir.is_dir():
        # nested runs: <task>/<task>/<timestamp>/
        nested_root = task_dir / task_name
        if nested_root.is_dir():
            run_dirs = sorted(
                [d for d in nested_root.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True,
            )
            if target_step is not None:
                for run_dir in run_dirs:
                    step_file = run_dir / "checkpoints" / f"step_{int(target_step):03d}.json"
                    if step_file.exists():
                        return str(step_file)
            for run_dir in run_dirs:
                cp = run_dir / "resume_checkpoint.json"
                if cp.exists():
                    return str(cp)

        # top-level fallback
        if target_step is not None:
            step_file = task_dir / "checkpoints" / f"step_{int(target_step):03d}.json"
            if step_file.exists():
                return str(step_file)
        top_legacy = task_dir / "resume_checkpoint.json"
        if top_legacy.exists():
            return str(top_legacy)

    return resolve_resume_checkpoint_path(base_path, target_step)


# ── 显示/Chrome 工具 ─────────────────────────────────────────────────────────

def _find_free_display(start, max_search=200):
    for n in range(start, start + max_search):
        if not os.path.exists(f"/tmp/.X{n}-lock") and not os.path.exists(f"/tmp/.X11-unix/X{n}"):
            return n
    return start + max_search


def _find_free_port(start, max_search=500):
    import socket
    for p in range(start, start + max_search):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start + max_search


def start_xvfb(display_num):
    actual = _find_free_display(display_num)
    xvfb = subprocess.Popen(
        ["Xvfb", f":{actual}", "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    return xvfb, actual


def _start_dbus_and_atspi(display_num, worker_id):
    dbus_proc = atspi_launcher = atspi_registryd = None
    if not shutil.which("dbus-launch"):
        return None, None, None
    env = os.environ.copy()
    env["DISPLAY"] = f":{display_num}"
    try:
        result = subprocess.run(
            ["dbus-launch", "--sh-syntax"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().rstrip(";").strip("'\"")
                    os.environ[key] = val
        for path, attr in [
            ("/usr/libexec/at-spi-bus-launcher", "atspi_launcher"),
            ("/usr/libexec/at-spi2-registryd", "atspi_registryd"),
        ]:
            if os.path.exists(path):
                proc = subprocess.Popen([path], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, env=os.environ.copy())
                time.sleep(1)
                if attr == "atspi_launcher":
                    atspi_launcher = proc
                else:
                    atspi_registryd = proc
    except Exception as e:
        print(f"[Worker {worker_id}] AT-SPI setup failed: {e}")
    return dbus_proc, atspi_launcher, atspi_registryd


def _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id):
    for proc in (atspi_registryd, atspi_launcher):
        if proc and proc.poll() is None:
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                proc.kill()
    dbus_pid = os.environ.get("DBUS_SESSION_BUS_PID")
    if dbus_pid:
        try:
            os.kill(int(dbus_pid), signal.SIGTERM)
        except Exception:
            pass


def start_wm_for_display(display_num: int):
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
                subprocess.Popen([exe], env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
                time.sleep(1)
                return
            except Exception:
                continue


def setup_xauthority(display_num, worker_id):
    auth_file = f"/tmp/.Xauthority_wdj_worker_{worker_id}"
    try:
        import secrets
        cookie = secrets.token_hex(16)
        subprocess.run(
            ["xauth", "-f", auth_file, "add", f":{display_num}", ".", "MIT-MAGIC-COOKIE-1", cookie],
            check=True, capture_output=True,
        )
        os.environ["XAUTHORITY"] = auth_file
    except Exception:
        os.environ["XAUTHORITY"] = auth_file
        Path(auth_file).touch()


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


# ── Worker ──────────────────────────────────────────────────────────────────

def worker_process(task_queue, result_queue, worker_id, preset, full_cfg):
    import warnings
    warnings.filterwarnings("ignore", message=".*xauthority.*")
    sys.stderr = _StderrFilter(sys.stderr)

    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)

    # 每个 worker 独立缓存目录，从共享缓存复制（避免重复下载，与 test2 一致）
    worker_cache = f"/tmp/modelscope_worker_{worker_id}"
    os.environ["MODELSCOPE_CACHE"] = worker_cache
    if os.path.exists(SHARED_MODEL_CACHE) and not os.path.exists(os.path.join(worker_cache, "hub")):
        shutil.copytree(SHARED_MODEL_CACHE, worker_cache, dirs_exist_ok=True)
        print(f"[Worker {worker_id}] 从共享缓存复制模型 -> {worker_cache}")
    os.makedirs(worker_cache, exist_ok=True)

    # 本地文件访问：必须禁用 web security
    os.environ["APPEVAL_CHROME_DISABLE_WEB_SECURITY"] = "1"
    # file:// 协议需要允许文件访问
    os.environ["APPEVAL_CHROME_EXTRA_ARGS"] = (
        os.environ.get("APPEVAL_CHROME_EXTRA_ARGS", "")
        + " --allow-file-access-from-files --disable-web-security"
    ).strip()

    if full_cfg.get("chrome_disable_web_security"):
        os.environ["APPEVAL_CHROME_DISABLE_WEB_SECURITY"] = "1"

    apply_llm_env(preset, for_local=(preset.get("config_file", "").find("local") >= 0))

    llm_config = preset.get("llm", {})
    case_generator_config = preset.get("case_generator", llm_config)
    tell_verifier_config = full_cfg.get("tell_verifier", {})
    supervisor_judge_config = full_cfg.get("supervisor_judge", tell_verifier_config)
    temp_config = {
        "llm": llm_config,
        "case_generator": case_generator_config,
        "tell_verifier": tell_verifier_config,
        "supervisor_judge": supervisor_judge_config,
    }
    config_file = f"/tmp/appeval_wdj_config_worker_{worker_id}.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(temp_config, f, default_flow_style=False)

    cuda_devices = preset.get("cuda_devices", [0])
    assigned_gpu = cuda_devices[worker_id % len(cuda_devices)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    base_display = int(preset.get("base_display", 400))
    base_chrome_port = int(preset.get("base_chrome_port", 11000))
    base_log_prefix = preset.get("log_dir_prefix", "webdevjudge")
    main_model_raw = llm_config.get("model", "unknown")
    safe_model = main_model_raw.replace("/", "_").replace(" ", "_").replace(":", "_")
    for c in '\\*?"<>|':
        safe_model = safe_model.replace(c, "_")
    log_dir_prefix = f"{base_log_prefix}_{safe_model}"

    display_num = base_display + worker_id
    port_step = 100

    time.sleep(worker_id * 0.3)
    xvfb, display_num = start_xvfb(display_num)
    os.environ["DISPLAY"] = f":{display_num}"
    setup_xauthority(display_num, worker_id)
    start_wm_for_display(display_num)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["GTK_MODULES"] = "gail:atk-bridge"
    os.environ["GNOME_ACCESSIBILITY"] = "1"
    os.environ["NO_AT_BRIDGE"] = "0"
    dbus_proc, atspi_launcher, atspi_registryd = _start_dbus_and_atspi(display_num, worker_id)

    from appeval.roles.eval_runner import AppEvalRole

    print(f"[Worker {worker_id}] 启动 | 模型={llm_config.get('model')} | display=:{display_num}")

    task_count = 0
    tasks_since_recycle = 0
    RECYCLE_EVERY = 20

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            break
        if item is None:
            break
        tasks_to_do = item if isinstance(item, list) else [item]

        for task in tasks_to_do:
            port = _find_free_port(base_chrome_port + worker_id + task_count * port_step)
            user_data_dir = f"/tmp/chrome_wdj_{log_dir_prefix}_w{worker_id}_t{task_count}"
            task_count += 1

            if os.path.exists(user_data_dir):
                shutil.rmtree(user_data_dir, ignore_errors=True)
            os.makedirs(user_data_dir, exist_ok=True)

            idx = task["idx"]
            web_id = task["web_id"]
            task_id = task["task_id"]
            task_name = task["task_name"]   # e.g. "web_2_task_1"
            url = task["url"]               # file:///...
            instruction = task["instruction"]
            max_steps = int(task.get("max_steps", preset.get("max_iters", 15)))
            save_checkpoint = bool(task.get("save_checkpoint", False))
            save_checkpoint_per_step = bool(task.get("save_checkpoint_per_step", False))
            save_profile_per_step = bool(task.get("save_profile_per_step", False))
            resume_target_step = task.get("resume_target_step", None)
            resume_checkpoint_path = resolve_task_resume_checkpoint_path(
                str(task.get("resume_checkpoint_path", "") or ""),
                task_name,
                int(resume_target_step) if resume_target_step is not None else None,
            )

            print(f"[Worker {worker_id}] #{idx} {task_name} | {url[:60]}...")

            try:
                test_cases = {
                    "0": {
                        "case_desc": instruction,
                        "result": "",
                        "evidence": "",
                    }
                }
                appeval = AppEvalRole(
                    config_file=config_file,
                    remote_debugging_port=port,
                    user_data_dir=user_data_dir,
                    use_chrome_debugger=False,
                    a11y_mode=preset.get("a11y_mode", "cdp"),
                    max_iters=max_steps,
                    run_id=f"wdj_{int(time.time())}_{idx}",
                    worker_id=worker_id,
                    save_checkpoint_per_step=save_checkpoint_per_step,
                    save_profile_per_step=save_profile_per_step,
                    use_ocr=bool(preset.get("use_ocr", False)),
                    post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
                    log_dirs="work_dirs",
                    use_timestamp_log_dir=True,
                    agent_class=preset.get("agent_class", "osagent"),
                    branching_n_candidates=int(
                        preset.get("branching_n_candidates", full_cfg.get("branching_n_candidates", 0))
                    ),
                    branching_k=int(
                        preset.get("branching_k", full_cfg.get("branching_k", 1))
                    ),
                )
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                t0 = time.perf_counter()
                result, _ = loop.run_until_complete(
                    appeval.run_api(
                        task_name=task_name,
                        test_cases=test_cases,
                        start_func=url,
                        log_dir=f"{log_dir_prefix}/{task_name}",
                        max_retry_uncertain=preset.get("max_retry_uncertain", 1),
                        resume_checkpoint_path=resume_checkpoint_path,
                        save_checkpoint=save_checkpoint,
                        chrome_profile_src=user_data_dir if save_checkpoint else "",
                        sequential_mode=bool(resume_checkpoint_path),
                    )
                )
                elapsed_sec = time.perf_counter() - t0
                loop.close()

                # Token 消耗（全量：OSAgent + CaseGenerator + TellVerifier + SupervisorJudge）
                token_usage = {}
                try:
                    token_usage = appeval.get_all_token_usage()
                except Exception:
                    pass
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                sv_prompt_tokens = token_usage.get("sv_prompt_tokens", 0)
                sv_completion_tokens = token_usage.get("sv_completion_tokens", 0)

                score = 0
                evidence = ""
                if result and "0" in result:
                    res = result["0"]
                    result_value = res.get("result", "Fail") if isinstance(res, dict) else str(res)
                    evidence = res.get("evidence", "") if isinstance(res, dict) else str(res)
                    score = 1 if result_value.lower().strip() in ("pass", "true", "1") else 0
                else:
                    evidence = "No result returned"

                print(f"[Worker {worker_id}] ✓ #{idx} {task_name} → {score} ({elapsed_sec:.0f}s, tokens={prompt_tokens+completion_tokens}, sv_tokens={sv_prompt_tokens+sv_completion_tokens})")
                result_queue.put({
                    "idx": idx,
                    "web_id": web_id,
                    "task_id": task_id,
                    "task_name": task_name,
                    "instruction": instruction,
                    "score": score,
                    "evidence": evidence,
                    "elapsed_sec": elapsed_sec,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "sv_prompt_tokens": sv_prompt_tokens,
                    "sv_completion_tokens": sv_completion_tokens,
                })
            except Exception as e:
                print(f"[Worker {worker_id}] ✗ #{idx} {task_name}: {e}")
                result_queue.put({
                    "idx": idx,
                    "web_id": web_id,
                    "task_id": task_id,
                    "task_name": task_name,
                    "instruction": instruction,
                    "score": 0,
                    "evidence": f"Error: {str(e)}",
                    "elapsed_sec": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                })

            tasks_since_recycle += 1
            if tasks_since_recycle >= RECYCLE_EVERY:
                try:
                    subprocess.run(
                        f"pkill -f 'user-data-dir={user_data_dir}'",
                        shell=True, capture_output=True,
                    )
                    time.sleep(1)
                    tasks_since_recycle = 0
                except Exception:
                    pass

    _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id)
    xvfb.terminate()
    xvfb.wait()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="webdevjudge_dev 跑测（配置见 run_config.yaml）")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tasks", type=int, default=None, help="限制任务数，0=全部")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--model", type=str, choices=["local", "remote"], default=None)
    parser.add_argument("--web", type=str, default=None, help="只跑指定 web，逗号分隔或范围，如 web_2,web_5 或 11-30")
    parser.add_argument("--task-names", type=str, default=None, help="只跑指定 task_name，JSON 文件路径（如 fn_case_names.json）或逗号分隔列表")
    parser.add_argument("--case-dir", type=str, default=None, help="case 目录路径（默认 case/，也可指定 cases_export/）")
    parser.add_argument("--resume", action="store_true", help="跳过结果文件中已有分数的任务")
    parser.add_argument("--save-checkpoint", action="store_true", help="保存可用于第二轮恢复的 checkpoint")
    parser.add_argument("--save-checkpoint-per-step", action="store_true", help="每步保存 step_xxx checkpoint")
    parser.add_argument("--save-profile-per-step", action="store_true", help="每步保存 profile 快照（磁盘占用大）")
    parser.add_argument("--resume-checkpoint-path", type=str, default="", help="第二轮恢复用 checkpoint 路径（文件/目录/index）")
    parser.add_argument("--resume-target-step", type=int, default=None, help="第二轮恢复到指定步（如 7）")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    if args.cleanup:
        cleanup_script = SCRIPT_DIR.parent / "cleanup.sh"
        if cleanup_script.exists():
            print("🧹 清理残留进程...")
            subprocess.run(["bash", str(cleanup_script)], timeout=30)

    try:
        subprocess.run(
            ["sysctl", "-w", "fs.inotify.max_user_instances=8192"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    config_path = args.config or str(CONFIG_PATH)
    tasks_limit, workers, model, preset, full_cfg, serial_per_url, label_excel = load_run_config(config_path)

    if args.tasks is not None:
        tasks_limit = args.tasks
    if args.workers is not None:
        workers = args.workers
    if args.model is not None:
        model = args.model
        preset = full_cfg.get(model, preset)

    preset["cuda_devices"] = full_cfg.get("cuda_devices", [0])
    apply_llm_env(preset, for_local=(model == "local"))

    # 加载真实标签
    if label_excel and not Path(label_excel).is_absolute():
        label_excel = str(SCRIPT_DIR / label_excel)
    label_map = load_label_map(label_excel)

    # 解析 --web 参数：支持 "web_2,web_5"、"11-30"、"web_11-web_30" 三种格式
    web_filter = None
    if args.web:
        raw = args.web.strip()
        # 纯数字范围：11-30
        import re as _re
        range_m = _re.match(r'^(\d+)-(\d+)$', raw)
        if range_m:
            lo, hi = int(range_m.group(1)), int(range_m.group(2))
            web_filter = {f"web_{i}" for i in range(lo, hi + 1)}
        else:
            # 逗号分隔，每项可带或不带 "web_" 前缀
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            web_filter = {p if p.startswith("web_") else f"web_{p}" for p in parts}

    # 确定 case 目录
    case_dir = Path(args.case_dir).resolve() if args.case_dir else CASE_DIR
    if not case_dir.exists():
        raise FileNotFoundError(f"case 目录不存在: {case_dir}")

    all_tasks = load_tasks_from_cases(case_dir, web_filter=web_filter)

    # --task-names 过滤：支持 JSON 文件或逗号分隔列表
    if args.task_names:
        raw_tn = args.task_names.strip()
        if raw_tn.endswith(".json"):
            tn_path = Path(raw_tn) if Path(raw_tn).is_absolute() else SCRIPT_DIR / raw_tn
            task_name_set = set(json.loads(tn_path.read_text(encoding="utf-8")))
        else:
            task_name_set = {n.strip() for n in raw_tn.split(",") if n.strip()}
        before = len(all_tasks)
        all_tasks = [t for t in all_tasks if t["task_name"] in task_name_set]
        print(f"--task-names 过滤: {before} → {len(all_tasks)} 个任务")

    if tasks_limit and tasks_limit > 0:
        all_tasks = all_tasks[:tasks_limit]

    print(f"case目录: {case_dir.name} | 共扫描到任务: {len(all_tasks)} 个")

    # 结果文件
    result_excel = preset.get("result_excel", "webdevjudge_results.xlsx")
    result_path = SCRIPT_DIR / result_excel
    result_json = result_path.with_suffix(".json")

    # 续跑：跳过已有分数的任务
    done_set = set()
    if args.resume and result_json.exists():
        try:
            existing = json.loads(result_json.read_text(encoding="utf-8"))
            done_set = {r["task_name"] for r in existing if r.get("score") is not None}
            print(f"[续跑] 已完成 {len(done_set)} 个任务，跳过")
        except Exception:
            pass
    task_list = [
        {
            **t,
            "save_checkpoint": bool(args.save_checkpoint or full_cfg.get("save_checkpoint", False)),
            "save_checkpoint_per_step": bool(
                args.save_checkpoint_per_step or full_cfg.get("save_checkpoint_per_step", False)
            ),
            "save_profile_per_step": bool(
                args.save_profile_per_step or full_cfg.get("save_profile_per_step", False)
            ),
            "resume_checkpoint_path": str(args.resume_checkpoint_path or full_cfg.get("resume_checkpoint_path", "")),
            "resume_target_step": args.resume_target_step if args.resume_target_step is not None else full_cfg.get("resume_target_step", None),
        }
        for t in all_tasks
        if t["task_name"] not in done_set
    ]

    if not task_list:
        print("没有待跑任务，退出")
        return

    llm_model = preset.get("llm", {}).get("model", "?")
    print(f"模型={llm_model} | 任务={len(task_list)} | 并行={workers} | serial_per_url={serial_per_url}")

    task_queue: Queue = Queue()
    result_queue: Queue = Queue()

    if serial_per_url:
        # 同一 web_id 的 tasks 串行，不同 web 并行
        by_web: dict = defaultdict(list)
        for t in task_list:
            by_web[t["web_id"]].append(t)
        for group in by_web.values():
            task_queue.put(group)
        print(f"同 web 串行模式: {len(by_web)} 个 web, {len(task_list)} 个任务")
    else:
        for t in task_list:
            task_queue.put(t)
    for _ in range(workers):
        task_queue.put(None)

    total_start = time.perf_counter()
    processes = []
    for i in range(workers):
        p = Process(
            target=worker_process,
            args=(task_queue, result_queue, i, preset, full_cfg),
        )
        p.start()
        processes.append(p)

    # 加载历史结果，续跑时保留
    results = {}
    if args.resume and result_json.exists():
        try:
            for r in json.loads(result_json.read_text(encoding="utf-8")):
                results[r["task_name"]] = r
        except Exception:
            pass

    completed = 0
    last_save_time = time.perf_counter()
    last_result_time = time.perf_counter()
    SAVE_INTERVAL = 120
    STALL_TIMEOUT = 1200

    while completed < len(task_list):
        try:
            r = result_queue.get(timeout=30)
            results[r["task_name"]] = r
            completed += 1
            last_result_time = time.perf_counter()
            pct = completed / len(task_list) * 100
            print(f"进度: {completed}/{len(task_list)} ({pct:.1f}%) | {r['task_name']} → {r['score']}")

            if time.perf_counter() - last_save_time > SAVE_INTERVAL:
                _save_results(results, result_json, result_path)
                print(f"  💾 增量保存 ({completed} 条)")
                last_save_time = time.perf_counter()

        except queue.Empty:
            alive = [p for p in processes if p.is_alive()]
            if not alive:
                print(f"所有 worker 已退出，完成 {completed}/{len(task_list)}")
                break
            if time.perf_counter() - last_result_time > STALL_TIMEOUT:
                print(f"⚠️  {STALL_TIMEOUT/60:.0f} 分钟无新结果，强制退出")
                break
        except Exception as e:
            print(f"收集结果异常: {e}")
            break

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    total_elapsed = time.perf_counter() - total_start
    _save_results(results, result_json, result_path)

    # 统计
    all_res = list(results.values())
    n = len(all_res)
    if n:
        pass_n = sum(1 for r in all_res if r.get("score") == 1)
        avg_sec = sum(r.get("elapsed_sec", 0) for r in all_res) / n
        total_tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in all_res)

        # 准确率（与 label 对比）
        labeled = [(r, label_map[r["task_name"]]) for r in all_res if r["task_name"] in label_map]
        accuracy_str = "N/A (无标签)"
        if labeled:
            consistent = sum(1 for r, lbl in labeled if r.get("score") == lbl)
            accuracy_str = f"{consistent}/{len(labeled)} ({consistent/len(labeled)*100:.1f}%)"

        # 按 web 分组统计
        by_web: dict = defaultdict(list)
        for r in all_res:
            by_web[r["web_id"]].append(r.get("score", 0))

        print(f"\n✅ 结果已保存: {result_path}")
        print(f"\n=== 统计 ===")
        print(f"总任务: {n} | Pass: {pass_n} ({pass_n/n*100:.1f}%) | 准确率: {accuracy_str}")
        sv_tokens = sum(r.get("sv_prompt_tokens", 0) + r.get("sv_completion_tokens", 0) for r in all_res)
        print(f"平均耗时: {avg_sec:.1f}s | Token: {total_tokens} (其中 SV: {sv_tokens}) | 总耗时: {total_elapsed:.1f}s")
        print(f"\n=== 各 Web 通过率 ===")
        for web_id in sorted(by_web.keys(), key=lambda x: _web_sort_key(x)):
            scores = by_web[web_id]
            p_n = sum(scores)
            # 准确率（如果有标签）
            web_labeled = [(s, label_map.get(f"{web_id}_{i+1}")) for i, s in enumerate(scores)]
            web_consistent = sum(1 for s, lbl in web_labeled if lbl is not None and s == lbl)
            web_labeled_n = sum(1 for _, lbl in web_labeled if lbl is not None)
            acc_str = f" acc={web_consistent}/{web_labeled_n}" if web_labeled_n else ""
            print(f"  {web_id}: pass={p_n}/{len(scores)}{acc_str}")

    # 写回 label Excel（写入 os_agent_score 和一致性列）
    if label_map and results and label_excel and Path(label_excel).exists():
        _write_back_to_label_excel(label_excel, results, preset)


def _save_results(results: dict, json_path: Path, excel_path: Path):
    rows = sorted(results.values(), key=lambda r: (r.get("web_id", ""), r.get("task_id", 0)))
    try:
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[save] JSON 保存失败: {e}")
    try:
        import pandas as pd
        cols = ["web_id", "task_id", "task_name", "instruction", "score", "evidence",
                "elapsed_sec", "prompt_tokens", "completion_tokens",
                "sv_prompt_tokens", "sv_completion_tokens"]
        df = pd.DataFrame(rows, columns=[c for c in cols if any(c in r for r in rows)])
        df.to_excel(excel_path, index=False)
    except Exception as e:
        print(f"[save] Excel 保存失败（JSON 已保存）: {e}")


def _write_back_to_label_excel(label_excel: str, results: dict, preset: dict):
    """把本次跑测结果写回 label Excel 的 os_agent_score / os_agent判断是否一致？ 列。"""
    try:
        import pandas as pd
        df = pd.read_excel(label_excel)
        score_col = preset.get("score_column", "os_agent_score")
        evidence_col = preset.get("evidence_column", "evidence")
        consistent_col = "os_agent判断是否一致？"
        for col in (score_col, evidence_col, consistent_col):
            if col not in df.columns:
                df[col] = None
            df[col] = df[col].astype(object)

        updated = 0
        for task_name, r in results.items():
            mask = df["case_name"] == task_name
            if not mask.any():
                continue
            df.loc[mask, score_col] = r.get("score")
            df.loc[mask, evidence_col] = str(r.get("evidence", ""))
            lbl = df.loc[mask, "label"].iloc[0]
            try:
                df.loc[mask, consistent_col] = 1 if int(lbl) == r.get("score") else 0
            except Exception:
                pass
            updated += 1

        # 写回（加 _scored 后缀，不覆盖原始文件）
        out = Path(label_excel).with_stem(Path(label_excel).stem + "_scored")
        df.to_excel(out, index=False)
        print(f"[标签] 已写回 {updated} 条结果 → {out.name}")
    except Exception as e:
        print(f"[标签] 写回失败: {e}")


if __name__ == "__main__":
    main()
