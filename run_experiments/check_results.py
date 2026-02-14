#!/usr/bin/env python3
"""
跑测结果检查脚本 — 独立运行，查看准确率、混淆矩阵、错误分析。

用法:
    # 检查 TextAgent 结果
    python check_results.py

    # 检查指定结果文件
    python check_results.py --file test2_sf_api_atspi_test_gemini_3_flash_preview_c46o.xlsx --score gemini_score --evidence gemini_evidence

    # 对比两个结果文件
    python check_results.py --compare test2_text_agent_results.xlsx test2_sf_api_atspi_test_gemini_3_flash_preview_c46o.xlsx

    # 只看错误案例
    python check_results.py --errors 20

    # 实时刷新（跑测期间监控）
    python check_results.py --watch 30
"""
import argparse
import time
import sys
from pathlib import Path

import pandas as pd


def load_results(file_path, score_col, evidence_col, true_col):
    """加载结果文件并返回统计信息。"""
    df = pd.read_excel(file_path)
    tested = df[df[score_col].notna()]
    valid = tested[tested[true_col].notna()]
    return df, tested, valid


def print_summary(file_path, score_col, evidence_col, true_col="A 角分数"):
    """打印结果摘要。"""
    df = pd.read_excel(file_path)
    tested = df[df[score_col].notna()]
    n_tested = len(tested)
    valid = tested[tested[true_col].notna()]
    n_valid = len(valid)

    print(f"\n{'='*60}")
    print(f"📊 结果文件: {file_path}")
    print(f"   评分列: {score_col} | 证据列: {evidence_col}")
    print(f"{'='*60}")
    print(f"总任务: {len(df)} | 已完成: {n_tested} | 有真实标签: {n_valid}")

    if n_valid == 0:
        print("⚠️  没有可对比的结果")
        return

    consistent = (valid[score_col] == valid[true_col]).sum()
    acc = 100 * consistent / n_valid
    print(f"\n🎯 准确率: {acc:.2f}% ({consistent}/{n_valid})")

    # 混淆矩阵
    tp = ((valid[score_col] == 1) & (valid[true_col] == 1)).sum()
    tn = ((valid[score_col] == 0) & (valid[true_col] == 0)).sum()
    fp = ((valid[score_col] == 1) & (valid[true_col] == 0)).sum()
    fn = ((valid[score_col] == 0) & (valid[true_col] == 1)).sum()

    print(f"\n📋 混淆矩阵:")
    print(f"                   真实 Pass    真实 Fail")
    print(f"  预测 Pass:       {tp:>4}  (TP)   {fp:>4}  (FP)")
    print(f"  预测 Fail:       {fn:>4}  (FN)   {tn:>4}  (TN)")
    print()
    if tp + fn > 0:
        print(f"  Pass 召回率: {tp}/{tp+fn} = {100*tp/(tp+fn):.1f}%")
    if tp + fp > 0:
        print(f"  Pass 精确率: {tp}/{tp+fp} = {100*tp/(tp+fp):.1f}%")
    if tn + fp > 0:
        print(f"  Fail 召回率: {tn}/{tn+fp} = {100*tn/(tn+fp):.1f}%")

    # API 错误统计
    fn_cases = valid[(valid[score_col] == 0) & (valid[true_col] == 1)]
    if len(fn_cases) > 0 and evidence_col in fn_cases.columns:
        api_errors = fn_cases[fn_cases[evidence_col].astype(str).str.contains(
            'Error:|额度已用尽|401|429|500|token|timeout', case=False, na=False
        )]
        if len(api_errors) > 0:
            print(f"\n⚠️  API/系统错误导致的 FN: {len(api_errors)} ({100*len(api_errors)/len(fn_cases):.0f}% of FN)")
            real_valid = valid[~valid.index.isin(api_errors.index)]
            if len(real_valid) > 0:
                real_correct = (real_valid[score_col] == real_valid[true_col]).sum()
                real_acc = 100 * real_correct / len(real_valid)
                print(f"   剔除后真实准确率: {real_acc:.2f}% ({real_correct}/{len(real_valid)})")

    # 评分分布
    print(f"\n📊 评分分布:")
    print(f"  预测 Pass(1): {(valid[score_col]==1).sum()} | 预测 Fail(0): {(valid[score_col]==0).sum()}")
    print(f"  真实 Pass(1): {(valid[true_col]==1).sum()} | 真实 Fail(0): {(valid[true_col]==0).sum()}")


def print_errors(file_path, score_col, evidence_col, true_col="A 角分数", n=15):
    """打印错误案例详情。"""
    df = pd.read_excel(file_path)
    tested = df[df[score_col].notna()]
    valid = tested[tested[true_col].notna()]

    fn_cases = valid[(valid[score_col] == 0) & (valid[true_col] == 1)]
    fp_cases = valid[(valid[score_col] == 1) & (valid[true_col] == 0)]

    test_col = "测试点" if "测试点" in df.columns else None

    print(f"\n{'='*60}")
    print(f"❌ 漏判 (FN): 实际 Pass 但判成 Fail ({len(fn_cases)} 个)")
    print(f"{'='*60}")
    for i, (idx, row) in enumerate(fn_cases.head(n).iterrows()):
        ev = str(row[evidence_col])[:150] if pd.notna(row.get(evidence_col)) else "N/A"
        tp_text = str(row[test_col])[:80] if test_col and pd.notna(row.get(test_col)) else ""
        is_api_err = any(kw in ev.lower() for kw in ['error:', '额度已用尽', '401', '429', 'timeout'])
        err_tag = " 🔴API错误" if is_api_err else ""
        print(f"  [{idx}]{err_tag}")
        if tp_text:
            print(f"    测试点: {tp_text}")
        print(f"    evidence: {ev}")
    if len(fn_cases) > n:
        print(f"  ... 还有 {len(fn_cases)-n} 个")

    print(f"\n{'='*60}")
    print(f"⚡ 误判 (FP): 实际 Fail 但判成 Pass ({len(fp_cases)} 个)")
    print(f"{'='*60}")
    for i, (idx, row) in enumerate(fp_cases.head(n).iterrows()):
        ev = str(row[evidence_col])[:150] if pd.notna(row.get(evidence_col)) else "N/A"
        tp_text = str(row[test_col])[:80] if test_col and pd.notna(row.get(test_col)) else ""
        print(f"  [{idx}]")
        if tp_text:
            print(f"    测试点: {tp_text}")
        print(f"    evidence: {ev}")
    if len(fp_cases) > n:
        print(f"  ... 还有 {len(fp_cases)-n} 个")


def print_compare(file1, file2, true_col="A 角分数"):
    """对比两个结果文件。"""
    configs = {
        "text_agent": ("text_agent_score", "text_agent_evidence"),
        "gemini": ("gemini_score", "gemini_evidence"),
        "os_agent": ("os_agent_score", "evidence"),
    }

    def detect_config(path):
        df = pd.read_excel(path)
        for name, (sc, ev) in configs.items():
            if sc in df.columns:
                return df, sc, ev, name
        # 尝试用文件名猜
        for col in df.columns:
            if "score" in col.lower():
                ev_col = col.replace("score", "evidence")
                if ev_col not in df.columns:
                    ev_col = "evidence"
                return df, col, ev_col, col
        return df, None, None, "unknown"

    df1, sc1, ev1, name1 = detect_config(file1)
    df2, sc2, ev2, name2 = detect_config(file2)

    if not sc1 or not sc2:
        print("❌ 无法自动检测评分列，请用 --score 参数指定")
        return

    print(f"\n{'='*60}")
    print(f"📊 对比: {name1} vs {name2}")
    print(f"{'='*60}")

    header = f"{'指标':<25} {'  ' + name1:<15} {'  ' + name2:<15}"
    print(header)
    print("-" * len(header))

    for name, df, sc in [(name1, df1, sc1), (name2, df2, sc2)]:
        pass  # just for validation

    results = {}
    for label, df, sc, ev in [(name1, df1, sc1, ev1), (name2, df2, sc2, ev2)]:
        tested = df[df[sc].notna()]
        valid = tested[tested[true_col].notna()]
        n_valid = len(valid)
        if n_valid > 0:
            correct = (valid[sc] == valid[true_col]).sum()
            acc = 100 * correct / n_valid
            tp = ((valid[sc] == 1) & (valid[true_col] == 1)).sum()
            tn = ((valid[sc] == 0) & (valid[true_col] == 0)).sum()
            fp = ((valid[sc] == 1) & (valid[true_col] == 0)).sum()
            fn = ((valid[sc] == 0) & (valid[true_col] == 1)).sum()
            pass_recall = 100 * tp / (tp + fn) if tp + fn > 0 else 0
            pass_prec = 100 * tp / (tp + fp) if tp + fp > 0 else 0
            results[label] = {
                "n": n_valid, "acc": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "pass_recall": pass_recall, "pass_prec": pass_prec,
            }
        else:
            results[label] = None

    def row(metric, key, fmt=".1f"):
        v1 = f"{results[name1][key]:{fmt}}" if results.get(name1) else "N/A"
        v2 = f"{results[name2][key]:{fmt}}" if results.get(name2) else "N/A"
        print(f"  {metric:<23} {v1:>13} {v2:>13}")

    row("已完成", "n", "d")
    row("准确率 (%)", "acc")
    row("Pass 召回率 (%)", "pass_recall")
    row("Pass 精确率 (%)", "pass_prec")
    row("TP", "tp", "d")
    row("TN", "tn", "d")
    row("FP", "fp", "d")
    row("FN", "fn", "d")


def watch_mode(file_path, score_col, evidence_col, true_col, interval):
    """实时监控模式。"""
    print(f"👀 监控模式: 每 {interval}s 刷新 | Ctrl+C 退出")
    while True:
        try:
            print("\033[2J\033[H", end="")  # clear screen
            print(f"⏰ {time.strftime('%H:%M:%S')}")
            print_summary(file_path, score_col, evidence_col, true_col)
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 退出监控")
            break
        except Exception as e:
            print(f"⚠️ {e}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="跑测结果检查脚本")
    parser.add_argument("--file", type=str, default="test2_text_agent_no_images_results.xlsx",
                        help="结果 Excel 文件路径")
    parser.add_argument("--score", type=str, default=None,
                        help="评分列名 (自动检测)")
    parser.add_argument("--evidence", type=str, default=None,
                        help="证据列名 (自动检测)")
    parser.add_argument("--true-label", type=str, default="A 角分数",
                        help="真实标签列名")
    parser.add_argument("--errors", type=int, default=0,
                        help="显示 N 个错误案例详情")
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"),
                        help="对比两个结果文件")
    parser.add_argument("--watch", type=int, default=0,
                        help="实时刷新间隔(秒), 0=不刷新")
    args = parser.parse_args()

    # 对比模式
    if args.compare:
        print_compare(args.compare[0], args.compare[1], args.true_label)
        return

    # 自动检测列名
    file_path = args.file
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    df = pd.read_excel(file_path)
    score_col = args.score
    evidence_col = args.evidence

    if not score_col:
        for col in df.columns:
            if "score" in col.lower() and col != args.true_label:
                score_col = col
                break
    if not evidence_col:
        for col in df.columns:
            if "evidence" in col.lower():
                evidence_col = col
                break

    if not score_col:
        print(f"❌ 无法自动检测评分列，可用列: {list(df.columns)}")
        sys.exit(1)

    print(f"ℹ️  自动检测: score={score_col}, evidence={evidence_col}")

    # 监控模式
    if args.watch > 0:
        watch_mode(file_path, score_col, evidence_col, args.true_label, args.watch)
        return

    # 摘要
    print_summary(file_path, score_col, evidence_col, args.true_label)

    # 错误详情
    if args.errors > 0:
        print_errors(file_path, score_col, evidence_col, args.true_label, args.errors)


if __name__ == "__main__":
    main()

