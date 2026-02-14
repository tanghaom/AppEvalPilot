#!/usr/bin/env python3
"""
test2 统一跑测入口 - RealDevBench MGX
URL=prod_id，真实标签=A角分数，任务=测试点。
配置见 run_config.yaml：任务数、并行数、本地/远程模型。
执行: cd /root/zhijieliu/AppEvalPilot/test/realdevbench/test2 && /data/miniconda3/envs/appeval/bin/python run_test.py
"""
import os
import signal
import sys
import time
import subprocess
import asyncio
import pandas as pd
import yaml
from collections import defaultdict
from pathlib import Path
from multiprocessing import Process, Queue
import queue

# GPU 分配由 run_config.yaml 的 cuda_devices 控制，worker 启动时按轮询分配单卡

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = "/data/hongsirui/AppEvalPilot"
CONFIG_PATH = SCRIPT_DIR / "run_config.yaml"

sys.path.insert(0, PROJECT_DIR)

def load_run_config(config_path=None):
    """加载 run_config.yaml，返回 (tasks, workers, model, preset, full_cfg, excel_file)."""
    path = Path(config_path or CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = (cfg.get("model") or "remote").strip().lower()
    if model not in ("local", "remote", "text"):
        model = "remote"
    preset = cfg.get(model, {})
    if not preset:
        raise ValueError(f"run_config.yaml 中缺少 '{model}' 配置段")
    tasks = int(cfg.get("tasks", 5))
    workers = int(cfg.get("workers", 5))
    serial_per_url = bool(cfg.get("serial_per_url", False))
    excel_file = cfg.get("excel_file", "")
    true_label_column = cfg.get("true_label_column", "A 角分数")
    return tasks, workers, model, preset, cfg, excel_file, true_label_column, serial_per_url


def apply_llm_env(preset, for_local=False):
    """用 preset['llm'] 设置 metagpt 环境变量（需在 import metagpt 前调用）。"""
    llm = preset.get("llm") or {}
    os.environ["llm__api_type"] = str(llm.get("api_type", "openai"))
    os.environ["llm__model"] = str(llm.get("model", ""))
    os.environ["llm__base_url"] = str(llm.get("base_url", ""))
    os.environ["llm__api_key"] = str(llm.get("api_key", ""))
    os.environ["llm__stream"] = str(llm.get("stream", "false"))
    if for_local:
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""


def start_xvfb(display_num):
    xvfb = subprocess.Popen(
        ["Xvfb", f":{display_num}", "-screen", "0", "1920x1080x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return xvfb


def _start_dbus_and_atspi(display_num, worker_id):
    """Start D-Bus session bus and AT-SPI services for accessibility tree support.

    Returns (dbus_proc, atspi_launcher, atspi_registryd) or Nones on failure.
    """
    dbus_proc = atspi_launcher = atspi_registryd = None
    env = os.environ.copy()
    env["DISPLAY"] = f":{display_num}"

    try:
        # 1. Start D-Bus session bus
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
            bus_pid = os.environ.get("DBUS_SESSION_BUS_PID")
            print(f"[Worker {worker_id}] D-Bus started (PID: {bus_pid})")
        else:
            print(f"[Worker {worker_id}] D-Bus launch failed: {result.stderr}")
            return None, None, None

        # 2. Start AT-SPI bus launcher
        atspi_launcher_path = "/usr/libexec/at-spi-bus-launcher"
        if os.path.exists(atspi_launcher_path):
            atspi_launcher = subprocess.Popen(
                [atspi_launcher_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
            time.sleep(2)
            print(f"[Worker {worker_id}] AT-SPI bus launcher started (PID: {atspi_launcher.pid})")

        # 3. Start AT-SPI registry daemon
        atspi_registryd_path = "/usr/libexec/at-spi2-registryd"
        if os.path.exists(atspi_registryd_path):
            atspi_registryd = subprocess.Popen(
                [atspi_registryd_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
            time.sleep(1)
            print(f"[Worker {worker_id}] AT-SPI registryd started (PID: {atspi_registryd.pid})")

    except Exception as e:
        print(f"[Worker {worker_id}] AT-SPI setup failed: {e}")

    return dbus_proc, atspi_launcher, atspi_registryd


def _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id):
    """Stop AT-SPI services."""
    for name, proc in [("registryd", atspi_registryd), ("bus-launcher", atspi_launcher)]:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    # Kill D-Bus by PID stored in env
    dbus_pid = os.environ.get("DBUS_SESSION_BUS_PID")
    if dbus_pid:
        try:
            os.kill(int(dbus_pid), signal.SIGTERM)
        except Exception:
            pass


SHARED_MODEL_CACHE = os.path.join(PROJECT_DIR, ".cache", "modelscope")

def worker_process(task_queue, result_queue, worker_id, preset, excel_file):
    """Worker 进程：根据 preset 使用本地或远程模型。"""
    import shutil

    agent_class = preset.get("agent_class", "osagent")
    is_text_agent = agent_class == "text_agent"

    if is_text_agent:
        # TextAgent 不需要 OCR/GPU，跳过模型缓存和 GPU 分配
        print(f"[Worker {worker_id}] TextAgent 模式: 跳过 OCR 模型和 GPU 分配")
    else:
        # 每个 worker 独立缓存目录，从共享缓存复制（避免重复下载）
        worker_cache = f"/tmp/modelscope_worker_{worker_id}"
        os.environ["MODELSCOPE_CACHE"] = worker_cache
        if os.path.exists(SHARED_MODEL_CACHE) and not os.path.exists(os.path.join(worker_cache, "hub")):
            # 共享缓存已有模型，直接复制到 worker 目录
            shutil.copytree(SHARED_MODEL_CACHE, worker_cache, dirs_exist_ok=True)
            print(f"[Worker {worker_id}] 从共享缓存复制模型 -> {worker_cache}")
        os.makedirs(worker_cache, exist_ok=True)

        # GPU 轮询分配：每个 worker 只使用一张卡，避免所有 worker 挤同一张卡
        cuda_devices = preset.get("cuda_devices", [0])
        assigned_gpu = cuda_devices[worker_id % len(cuda_devices)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        print(f"[Worker {worker_id}] 分配 GPU {assigned_gpu}")

    apply_llm_env(preset, for_local=(preset.get("config_file", "").find("local") >= 0))

    base_display = int(preset.get("base_display", 300))
    base_chrome_port = int(preset.get("base_chrome_port", 9500))
    config_file = preset.get("config_file", "config/config2.yaml")
    log_dir_prefix = preset.get("log_dir_prefix", "test2")

    display_num = base_display + worker_id
    port = base_chrome_port + worker_id
    user_data_dir = f"/tmp/chrome_test2_{log_dir_prefix}_{worker_id}"

    if os.path.exists(user_data_dir):
        shutil.rmtree(user_data_dir, ignore_errors=True)
    os.makedirs(user_data_dir, exist_ok=True)

    time.sleep(worker_id * 0.3)
    xvfb = start_xvfb(display_num)
    os.environ["DISPLAY"] = f":{display_num}"

    # 防止 opencv-python 自带的 Qt 插件在 Xvfb 下崩溃
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # AT-SPI: text_agent(cdp 模式) 不需要 D-Bus/AT-SPI；osagent(atspi 模式) 才需要
    dbus_proc = atspi_launcher = atspi_registryd = None
    if not is_text_agent or preset.get("a11y_mode", "cdp") == "atspi":
        os.environ["GTK_MODULES"] = "gail:atk-bridge"
        os.environ["GNOME_ACCESSIBILITY"] = "1"
        os.environ["NO_AT_BRIDGE"] = "0"
        dbus_proc, atspi_launcher, atspi_registryd = _start_dbus_and_atspi(display_num, worker_id)
    else:
        print(f"[Worker {worker_id}] TextAgent(cdp 模式): 跳过 D-Bus/AT-SPI 启动")

    os.chdir(PROJECT_DIR)
    from appeval.roles.eval_runner import AppEvalRole

    print(f"[Worker {worker_id}] 配置: {config_file}")

    tasks_since_recycle = 0
    RECYCLE_EVERY = 20  # 每 20 个任务回收一次 Chrome 进程，防止内存泄漏

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            break
        if item is None:
            break
        # serial_per_url 时 item 为同 URL 的一批任务 list；否则为单个 task dict
        tasks_to_do = item if isinstance(item, list) else [item]

        for task in tasks_to_do:
            idx = task["idx"]
            case_name = task["case_name"]
            url = task["url"]
            test_point = task["test_point"]

            print(f"[Worker {worker_id}] #{idx}: {case_name}")
            print(f"[Worker {worker_id}] a11y_mode: {preset.get('a11y_mode', 'atspi')}, preset: {preset}")
            # breakpoint()
            try:
                test_cases = {
                    "0": {
                        "case_desc": test_point,
                        "result": "",
                        "evidence": "",
                    }
                }
                agent_class = preset.get("agent_class", "osagent")  # osagent | text_agent
                appeval = AppEvalRole(
                    config_file=config_file,
                    remote_debugging_port=port,
                    user_data_dir=user_data_dir,
                    use_chrome_debugger=False,
                    a11y_mode=preset.get("a11y_mode", "atspi"),  # atspi=默认(需D-Bus), cdp=轻量(有坐标漂移)
                    max_iters=preset.get("max_iters", 15),
                    use_ocr=preset.get("use_ocr", agent_class != "text_agent"),  # text_agent 默认关 OCR
                    post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
                    agent_class=agent_class,
                    debug_screenshots=preset.get("debug_screenshots", True),
                )
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                t0 = time.perf_counter()
                result, _ = loop.run_until_complete(
                    appeval.run_api(
                        task_name=f"{case_name}_{idx}",
                        test_cases=test_cases,
                        start_func=url,
                        log_dir=f"{log_dir_prefix}/{case_name}_{idx}",
                    )
                )
                elapsed_sec = time.perf_counter() - t0
                loop.close()

                # Token 消耗（CaseGenerator + OSAgent 的 LLM）
                prompt_tokens = completion_tokens = 0
                try:
                    if hasattr(appeval.test_generator, "llm") and hasattr(appeval.test_generator.llm, "get_costs"):
                        c = appeval.test_generator.llm.get_costs()
                        prompt_tokens += getattr(c, "total_prompt_tokens", 0) or 0
                        completion_tokens += getattr(c, "total_completion_tokens", 0) or 0
                    if hasattr(appeval, "osagent") and appeval.osagent and hasattr(appeval.osagent, "llm") and hasattr(appeval.osagent.llm, "get_costs"):
                        c = appeval.osagent.llm.get_costs()
                        prompt_tokens += getattr(c, "total_prompt_tokens", 0) or 0
                        completion_tokens += getattr(c, "total_completion_tokens", 0) or 0
                except Exception:
                    pass

                score = 0
                evidence = ""
                if result and "0" in result:
                    res = result["0"]
                    if isinstance(res, dict):
                        result_value = res.get("result", "Fail")
                        evidence = res.get("evidence", "")
                    else:
                        # LLM 返回了字符串而非 dict（格式不稳定的兜底）
                        result_value = str(res)
                        evidence = str(res)
                    score = 1 if result_value.lower().strip() in ("pass", "true", "1") else 0
                else:
                    evidence = "No result returned"

                # 黑屏检测：如果 evidence 含黑屏关键词且判 Fail，尝试重启 Chrome 重试一次
                black_kws = ["black screen", "blank screen", "black page", "blank page"]
                if score == 0 and any(k in evidence.lower() for k in black_kws):
                    print(f"[Worker {worker_id}] ⚠️ #{idx} 黑屏检测! 重启 Chrome 重试...")
                    try:
                        from appeval.utils.window_utils import kill_windows
                        loop2 = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop2)
                        loop2.run_until_complete(kill_windows(user_data_dir=user_data_dir))
                        loop2.close()
                        time.sleep(2)
                        # 清理 Chrome profile 避免锁文件
                        import shutil as _shutil
                        if os.path.exists(user_data_dir):
                            _shutil.rmtree(user_data_dir, ignore_errors=True)
                            os.makedirs(user_data_dir, exist_ok=True)
                        # 重试
                        appeval2 = AppEvalRole(
                            config_file=config_file,
                            remote_debugging_port=port,
                            user_data_dir=user_data_dir,
                            use_chrome_debugger=False,
                            a11y_mode=preset.get("a11y_mode", "atspi"),
                            max_iters=preset.get("max_iters", 15),
                            use_ocr=preset.get("use_ocr", agent_class != "text_agent"),
                            post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
                            agent_class=agent_class,
                            debug_screenshots=preset.get("debug_screenshots", True),
                        )
                        loop3 = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop3)
                        t1 = time.perf_counter()
                        result2, _ = loop3.run_until_complete(
                            appeval2.run_api(
                                task_name=f"{case_name}_{idx}",
                                test_cases=test_cases,
                                start_func=url,
                                log_dir=f"{log_dir_prefix}/{case_name}_{idx}",
                            )
                        )
                        elapsed_sec += time.perf_counter() - t1
                        loop3.close()
                        if result2 and "0" in result2:
                            res2 = result2["0"]
                            evidence = res2.get("evidence", evidence)
                            score = 1 if res2.get("result", "Fail").lower() == "pass" else 0
                            print(f"[Worker {worker_id}] ♻️ #{idx} 重试结果: {score}")
                    except Exception as retry_err:
                        print(f"[Worker {worker_id}] ♻️ #{idx} 重试失败: {retry_err}")

                print(f"[Worker {worker_id}] 完成 #{idx}: {case_name} -> {score} ({elapsed_sec:.0f}s, {prompt_tokens + completion_tokens} tokens)")
                result_queue.put({
                    "idx": idx, "case_name": case_name, "score": score, "evidence": evidence,
                    "elapsed_sec": elapsed_sec, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                })
            except Exception as e:
                print(f"[Worker {worker_id}] 错误 #{idx}: {e}")
                result_queue.put({
                    "idx": idx, "case_name": case_name, "score": 0, "evidence": f"Error: {str(e)}",
                    "elapsed_sec": 0, "prompt_tokens": 0, "completion_tokens": 0,
                })

            # 定期回收 Chrome 进程，防止长时间运行内存泄漏导致黑屏
            tasks_since_recycle += 1
            if tasks_since_recycle >= RECYCLE_EVERY:
                try:
                    import subprocess as _sp
                    _sp.run(f"pkill -f 'user-data-dir={user_data_dir}'", shell=True, capture_output=True)
                    time.sleep(1)
                    print(f"[Worker {worker_id}] ♻️ Chrome 定期回收 (每 {RECYCLE_EVERY} 任务)")
                    tasks_since_recycle = 0
                except Exception:
                    pass

    _stop_atspi(dbus_proc, atspi_launcher, atspi_registryd, worker_id)
    xvfb.terminate()
    xvfb.wait()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="test2: RealDevBench MGX（配置见 run_config.yaml）")
    parser.add_argument("--config", type=str, default=None, help="run_config.yaml 路径")
    parser.add_argument("--tasks", type=int, default=None, help="覆盖配置中的任务数")
    parser.add_argument("--workers", type=int, default=None, help="覆盖配置中的并行数")
    parser.add_argument("--model", type=str, choices=["local", "remote", "text"], default=None, help="覆盖配置: local | remote | text")
    parser.add_argument("--resume", action="store_true", default=True, help="续跑：跳过结果表中已有分数的任务，只跑未完成的（默认启用）")
    parser.add_argument("--rerun-failed", action="store_true", default=False,
                        help="只重跑上一轮判断错误的 case（需要结果表和真实标签列同时存在）")
    parser.add_argument("--no-cleanup", action="store_true", default=True, help="跳过启动前的残留进程清理")

    args = parser.parse_args()

    # ── 跑测前自动清理残留进程 ──
    if not args.no_cleanup:
        cleanup_script = SCRIPT_DIR / "cleanup.sh"
        if cleanup_script.exists():
            print("🧹 跑测前清理残留进程...")
            subprocess.run(["bash", str(cleanup_script)], timeout=30)
        else:
            print(f"⚠️  清理脚本不存在: {cleanup_script}，跳过清理")

    # ── 提高系统 inotify 限制，防止 Chrome 报 "Too many open files" ──
    try:
        subprocess.run(
            ["sysctl", "-w", "fs.inotify.max_user_instances=8192"],
            capture_output=True, timeout=5
        )
        print("✅ inotify.max_user_instances 已设置为 8192")
    except Exception:
        print("⚠️  无法设置 inotify 限制（需要 root 权限），多 worker 时 Chrome 可能报错")

    config_path = args.config or str(CONFIG_PATH)
    tasks, workers, model, preset, full_cfg, excel_file, true_label_column, serial_per_url = load_run_config(config_path)

    if args.tasks is not None:
        tasks = args.tasks
    if args.workers is not None:
        workers = args.workers
    if args.model is not None:
        model = args.model
        preset = full_cfg.get(model, preset)

    # 将顶层 cuda_devices 注入 preset，供 worker 轮询分配 GPU
    preset["cuda_devices"] = full_cfg.get("cuda_devices", [0])

    apply_llm_env(preset, for_local=(model == "local"))

    excel_path = SCRIPT_DIR / (excel_file or full_cfg.get("excel_file", "RealDevBench_MGX_20260130.xlsx"))
    if not excel_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {excel_path}")

    result_excel = preset.get("result_excel", "test2_results.xlsx")
    score_col = preset.get("score_column", "os_agent_score")
    evidence_col = preset.get("evidence_column", "evidence")
    out_path = SCRIPT_DIR / result_excel

    # 自动检测 URL 列名：支持 prod_id 或 prod_url
    def _get_url_col(dataframe):
        if "prod_id" in dataframe.columns:
            return "prod_id"
        elif "prod_url" in dataframe.columns:
            return "prod_url"
        else:
            raise ValueError(f"Excel 中找不到 URL 列（需要 prod_id 或 prod_url），可用列: {list(dataframe.columns)}")

    # --rerun-failed: 只重跑上一轮判断错误的 case
    if args.rerun_failed and out_path.exists():
        df = pd.read_excel(out_path)
        url_col = _get_url_col(df)
        valid_df = df[
            df[url_col].notna() & df[true_label_column].notna() & df["测试点"].notna()
        ].copy()
        if tasks > 0:
            valid_df = valid_df.head(tasks)
        # 筛选判断错误的行：有预测分数、有真实标签、且两者不一致
        if score_col in valid_df.columns:
            has_both = valid_df[valid_df[score_col].notna() & valid_df[true_label_column].notna()]
            wrong_mask = has_both[score_col].astype(int) != has_both[true_label_column].astype(int)
            valid_df = has_both[wrong_mask].copy()
            # 清除旧分数，让 worker 重跑
            valid_df[score_col] = None
            valid_df[evidence_col] = None
            # 同步清除 df 中这些行的旧分数
            for idx in valid_df.index:
                df.at[idx, score_col] = None
                if evidence_col in df.columns:
                    df.at[idx, evidence_col] = None
            df.to_excel(out_path, index=False)
        else:
            valid_df = valid_df.head(0)
        print(f"[重跑失败] 结果表: {out_path}，筛选判断错误 case: {len(valid_df)} 个")
    # 续跑：从结果表恢复 df，只跑尚未有分数的任务
    elif args.resume and out_path.exists():
        df = pd.read_excel(out_path)
        url_col = _get_url_col(df)
        valid_df = df[
            df[url_col].notna() & df[true_label_column].notna() & df["测试点"].notna()
        ].copy()
        if tasks > 0:
            valid_df = valid_df.head(tasks)
        # 只保留尚未有分数的行
        if score_col in valid_df.columns:
            valid_df = valid_df[valid_df[score_col].isna()]
        else:
            valid_df = valid_df.head(0)
        print(f"[续跑] 结果表已存在: {out_path}，跳过已有分数任务，待跑: {len(valid_df)}")
    else:
        df = pd.read_excel(excel_path)
        url_col = _get_url_col(df)
        valid_df = df[
            df[url_col].notna() & df[true_label_column].notna() & df["测试点"].notna()
        ].copy()
        if tasks > 0:
            valid_df = valid_df.head(tasks)

    agent_class = preset.get("agent_class", "osagent")
    print(f"模型: {model} | Agent: {agent_class} | 任务数: {tasks} | 并行: {workers} | URL列: {url_col}")
    print(f"配置: {preset.get('config_file')} | 结果: {result_excel}")
    if agent_class == "text_agent":
        print(f"📝 TextAgent 模式: 纯文本 a11y tree, a11y_mode={preset.get('a11y_mode', 'cdp')}, 无 OCR/GPU")

    if len(valid_df) == 0:
        print("没有有效任务（或续跑时已全部完成），退出")
        return

    task_list = [
        {
            "idx": idx,
            "case_name": row["case_name"],
            "url": row[url_col],
            "test_point": row["测试点"],
        }
        for idx, row in valid_df.iterrows()
    ]

    # 预下载 OCR 模型到共享缓存，避免 worker 并发下载
    # _ensure_ocr_model_cached()

    task_queue = Queue()
    result_queue = Queue()
    if serial_per_url:
        by_url = defaultdict(list)
        for t in task_list:
            by_url[t["url"]].append(t)
        for _url, group in by_url.items():
            task_queue.put(group)  # 同一 URL 的一批任务作为一个 job，worker 内串行执行
        print(f"同一网站串行、不同网站并行: 共 {len(by_url)} 个 URL，{len(task_list)} 个任务")
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
            args=(task_queue, result_queue, i, preset, str(excel_path)),
        )
        p.start()
        processes.append(p)

    results = {}
    completed = 0
    last_save_time = time.perf_counter()
    last_result_time = time.perf_counter()
    save_interval = 120       # 每 2 分钟增量保存一次
    stall_timeout = 1200      # 20 分钟无新结果 → 判定卡死，强制退出

    while completed < len(task_list):
        try:
            r = result_queue.get(timeout=30)  # 30s 短超时，靠下面检查 worker 存活
            results[r["idx"]] = r
            completed += 1
            last_result_time = time.perf_counter()
            print(f"进度: {completed}/{len(task_list)} ({100 * completed / len(task_list):.1f}%)")

            # 增量保存
            now = time.perf_counter()
            if now - last_save_time > save_interval:
                for col in (score_col, evidence_col):
                    if col not in df.columns:
                        df[col] = None
                    df[col] = df[col].astype(object)
                for idx, res in results.items():
                    df.at[idx, score_col] = res["score"]
                    df.at[idx, evidence_col] = str(res["evidence"]) if res["evidence"] else ""
                df.to_excel(out_path, index=False)
                print(f"  💾 增量保存 ({completed} 条)")
                last_save_time = now

        except queue.Empty:
            # 检查是否还有 worker 存活
            alive = [p for p in processes if p.is_alive()]
            if not alive:
                print(f"所有 worker 已退出，已完成 {completed}/{len(task_list)}")
                break
            # worker 还活着，检查是否卡死（长时间无新结果）
            stall_sec = time.perf_counter() - last_result_time
            if stall_sec > stall_timeout:
                print(f"⚠️  已 {stall_sec/60:.0f} 分钟无新结果，{len(alive)} 个 worker 可能卡死，强制退出")
                break
            continue
        except Exception as e:
            print(f"收集结果异常: {e}")
            break

    # 等待 worker 退出
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            print(f"  强制终止 worker PID {p.pid}")
            p.terminate()
            p.join(timeout=5)

    total_elapsed = time.perf_counter() - total_start
    total_prompt = sum(r.get("prompt_tokens", 0) for r in results.values())
    total_completion = sum(r.get("completion_tokens", 0) for r in results.values())
    total_tokens = total_prompt + total_completion

    # Ensure columns accept both numbers and strings (avoid pandas dtype warning)
    for col in (score_col, evidence_col):
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)
    for idx, res in results.items():
        df.at[idx, score_col] = res["score"]
        df.at[idx, evidence_col] = str(res["evidence"]) if res["evidence"] else ""
        true_label = df.at[idx, true_label_column]
        if pd.notna(true_label):
            df.at[idx, "判断一致？"] = 1 if res["score"] == int(true_label) else 0

    df.to_excel(out_path, index=False)
    print(f"\n✅ 结果已保存: {out_path}")

    tested = df[df[score_col].notna()]
    if len(tested) > 0:
        valid = tested[tested[true_label_column].notna()]
        if len(valid) > 0:
            consistent = (valid[score_col] == valid[true_label_column]).sum()
            accuracy = consistent / len(valid) * 100
            n = len(task_list)
            avg_sec = (sum(r.get("elapsed_sec", 0) for r in results.values()) / n) if n else 0
            print(f"\n=== 统计 ===")
            print(f"测试数量: {len(valid)} | 准确率: {accuracy:.2f}%")
            print(f"总耗时: {total_elapsed:.1f}s | 平均每任务: {avg_sec:.1f}s")
            print(f"Token 消耗: prompt {total_prompt} + completion {total_completion} = 总计 {total_tokens}")


if __name__ == "__main__":
    main()