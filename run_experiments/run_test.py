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
    if model not in ("local", "remote"):
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


def worker_process(task_queue, result_queue, worker_id, preset, excel_file):
    """Worker 进程：根据 preset 使用本地或远程模型。"""
    import shutil

    # 每个 worker 独立 modelscope 缓存，并行时开 OCR 不冲突
    os.environ["MODELSCOPE_CACHE"] = f"/tmp/modelscope_worker_{worker_id}"
    os.makedirs(os.environ["MODELSCOPE_CACHE"], exist_ok=True)

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

    os.chdir(PROJECT_DIR)
    from appeval.roles.eval_runner import AppEvalRole

    print(f"[Worker {worker_id}] 配置: {config_file}")

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

            try:
                test_cases = {
                    "0": {
                        "case_desc": test_point,
                        "result": "",
                        "evidence": "",
                    }
                }
                appeval = AppEvalRole(
                    config_file=config_file,
                    remote_debugging_port=port,
                    user_data_dir=user_data_dir,
                    use_chrome_debugger=False,
                    max_iters=15,
                    use_ocr=True,  # 并行开 OCR，每 worker 独立 MODELSCOPE_CACHE
                    post_action_wait_sec=preset.get("post_action_wait_sec", 1.5),
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
                    result_value = res.get("result", "Fail")
                    evidence = res.get("evidence", "")
                    score = 1 if result_value.lower() == "pass" else 0
                else:
                    evidence = "No result returned"

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

    xvfb.terminate()
    xvfb.wait()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="test2: RealDevBench MGX（配置见 run_config.yaml）")
    parser.add_argument("--config", type=str, default=None, help="run_config.yaml 路径")
    parser.add_argument("--tasks", type=int, default=None, help="覆盖配置中的任务数")
    parser.add_argument("--workers", type=int, default=None, help="覆盖配置中的并行数")
    parser.add_argument("--model", type=str, choices=["local", "remote"], default=None, help="覆盖配置: local | remote")
    parser.add_argument("--resume", action="store_true", help="续跑：跳过结果表中已有分数的任务，只跑未完成的")
    args = parser.parse_args()

    config_path = args.config or str(CONFIG_PATH)
    tasks, workers, model, preset, full_cfg, excel_file, true_label_column, serial_per_url = load_run_config(config_path)

    if args.tasks is not None:
        tasks = args.tasks
    if args.workers is not None:
        workers = args.workers
    if args.model is not None:
        model = args.model
        preset = full_cfg.get(model, preset)

    apply_llm_env(preset, for_local=(model == "local"))

    excel_path = SCRIPT_DIR / (excel_file or full_cfg.get("excel_file", "RealDevBench_MGX_20260130.xlsx"))
    if not excel_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {excel_path}")

    result_excel = preset.get("result_excel", "test2_results.xlsx")
    score_col = preset.get("score_column", "os_agent_score")
    evidence_col = preset.get("evidence_column", "evidence")
    out_path = SCRIPT_DIR / result_excel

    # 续跑：从结果表恢复 df，只跑尚未有分数的任务
    if args.resume and out_path.exists():
        df = pd.read_excel(out_path)
        valid_df = df[
            df["prod_id"].notna() & df[true_label_column].notna() & df["测试点"].notna()
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
        valid_df = df[
            df["prod_id"].notna() & df[true_label_column].notna() & df["测试点"].notna()
        ].copy()
        if tasks > 0:
            valid_df = valid_df.head(tasks)

    print(f"模型: {model} | 任务数: {tasks} | 并行: {workers}")
    print(f"配置: {preset.get('config_file')} | 结果: {result_excel}")

    if len(valid_df) == 0:
        print("没有有效任务（或续跑时已全部完成），退出")
        return

    task_list = [
        {
            "idx": idx,
            "case_name": row["case_name"],
            "url": row["prod_id"],
            "test_point": row["测试点"],
        }
        for idx, row in valid_df.iterrows()
    ]

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
    while completed < len(task_list):
        try:
            r = result_queue.get(timeout=600)
            results[r["idx"]] = r
            completed += 1
            print(f"进度: {completed}/{len(task_list)} ({100 * completed / len(task_list):.1f}%)")
        except Exception:
            print("收集结果超时")
            break

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

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
