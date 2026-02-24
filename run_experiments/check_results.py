#!/usr/bin/env python3
"""
跑测结果检查脚本 — 独立运行，查看准确率、混淆矩阵、错误分析。

用法:
    # 检查 TextAgent 结果（自动检测列名）
    python check_results.py

    # 检查指定结果文件
    python check_results.py --file test2_sf_api_atspi_test_gemini_3_flash_preview_c46o.xlsx --score gemini_score --evidence gemini_evidence

    # 对比两个结果文件
    python check_results.py --compare test2_text_agent_results.xlsx test2_sf_api_atspi_test_gemini_3_flash_preview_c46o.xlsx

    # 只看错误案例
    python check_results.py --errors 20

    # 实时刷新（跑测期间监控）
    python check_results.py --watch 30

    # 🆕 完整错误归因分析（分类 + 按App聚合 + 可优化建议）
    python check_results.py --analyze

    # 分析并导出 markdown
    python check_results.py --analyze --export analysis_report.md
"""
import argparse
import re
import time
import sys
from collections import defaultdict
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

    # ── 🏥 页面崩溃 / A11y Tree 健康检查 ──
    if evidence_col and evidence_col in tested.columns:
        crash_cases = []
        for idx, row in tested.iterrows():
            ev_text = str(row.get(evidence_col, ""))
            is_crash, subtype = _detect_page_crash(ev_text)
            if is_crash:
                crash_cases.append((idx, subtype, row))

        n_crash = len(crash_cases)
        if n_crash > 0:
            subtypes = defaultdict(int)
            crash_apps = set()
            app_col = "case_name" if "case_name" in tested.columns else None
            for idx, subtype, row in crash_cases:
                subtypes[subtype] += 1
                if app_col and pd.notna(row.get(app_col)):
                    crash_apps.add(row[app_col])

            print(f"\n🏥 页面崩溃 / A11y Tree 为空: {n_crash} 个 ({100*n_crash/n_tested:.1f}% of 已完成)")
            for st, cnt in sorted(subtypes.items(), key=lambda x: -x[1]):
                print(f"   {st}: {cnt} 个")
            if crash_apps:
                print(f"   涉及 App ({len(crash_apps)}): {', '.join(sorted(crash_apps)[:8])}"
                      f"{'...' if len(crash_apps) > 8 else ''}")

            # 判断是否需要增加等待时间
            initial_fail = subtypes.get("初始加载失败", 0)
            mid_crash = subtypes.get("操作中崩溃", 0)
            if initial_fail > n_crash * 0.4:
                print(f"\n   💡 建议: 初始加载失败占比高 ({initial_fail}/{n_crash}={100*initial_fail/n_crash:.0f}%)")
                print(f"      → 可增加页面加载等待时间 (eval_runner.py SLEEP_AFTER_START_WEB, 当前=10s)")
                print(f"      → 或降低并发 workers 数以减少 CPU/内存竞争")
                print(f"      → 或增加 a11y tree 获取重试次数")
            if mid_crash > n_crash * 0.3:
                print(f"\n   💡 建议: 操作中崩溃占比高 ({mid_crash}/{n_crash}={100*mid_crash/n_crash:.0f}%)")
                print(f"      → 可能是并发过高导致 Chrome 渲染资源不足")
                print(f"      → 建议降低 workers 数或增加操作间 sleep 时间")
        else:
            print(f"\n✅ 页面健康: 未检测到页面崩溃/A11y Tree 为空")

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


############################################################
# ── 错误归因分析 ──
############################################################

# ── 页面崩溃 / A11y Tree 为空的检测关键词 ──
# ⚠️ 注意：关键词要足够精确，避免匹配正常 evidence 中的 "content", "page" 等常见词
PAGE_CRASH_KEYWORDS = [
    # A11y Tree 明确为空
    r"accessibility tree.*(?:is |was |became |remained |appears? )?empty",
    r"empty accessibility tree",
    r"a11y.*tree.*empty", r"empty.*a11y",
    r"accessibility tree.*(?:shows?|contains?|has)\s+no\s+(?:elements|nodes|entries)",
    # 页面崩溃/白屏（要求 page 紧邻 blank/empty/crash）
    r"(?:page|screen)\s+(?:is|was|became|went|turned|remained)\s+(?:blank|empty|black|unresponsive)",
    r"(?:blank|black)\s+(?:screen|page)\s+(?:after|when|during)",
    r"render(?:ed|s)?\s+a\s+blank\s+page",
    r"became\s+(?:blank|empty|unresponsive)",
    # 页面整体加载失败（非图片/元素级别）
    r"page\s+failed\s+to\s+(?:load|render)",
    r"page\s+(?:did not|didn'?t)\s+(?:load|render)",
    # 中文关键词
    r"页面.*崩溃", r"页面.*(?:全部|整个|完全).*空", r"加载失败", r"a11y.*tree.*为空",
]

# 区分崩溃子类型
_CRASH_SUBTYPE_RULES = [
    # 操作过程中崩溃
    ("操作中崩溃", [
        r"became\s+(?:blank|empty|unresponsive)",
        r"(?:after|upon)\s+(?:clicking|adjusting|attempting)",
        r"causes?\s+(?:the\s+)?(?:application|page)\s+to\s+crash",
        r"crashes\s+and\s+disappears",
        r"consistently\s+crash",
    ]),
    # 初始加载就失败
    ("初始加载失败", [
        r"remained\s+empty", r"page\s+(?:is|was)\s+empty",
        r"page\s+failed\s+to\s+(?:load|render)",
        r"empty.*throughout",
        r"(?:blank|black)\s+(?:screen|page)\s+(?:after|upon)\s+(?:loading|refresh|open)",
    ]),
]


def _detect_page_crash(evidence_text: str) -> tuple:
    """检测是否为页面崩溃 / A11y Tree 为空。

    Returns:
        (is_crash: bool, subtype: str)
        subtype: "操作中崩溃" | "初始加载失败" | "其他空树" | ""
    """
    ev = evidence_text.lower()
    is_crash = any(re.search(pat, ev, re.IGNORECASE) for pat in PAGE_CRASH_KEYWORDS)
    if not is_crash:
        return False, ""

    for subtype, patterns in _CRASH_SUBTYPE_RULES:
        if any(re.search(p, ev, re.IGNORECASE) for p in patterns):
            return True, subtype
    return True, "其他空树"


# 关键词 → 错误类别映射（优先级从高到低）
ERROR_CATEGORY_RULES = [
    # ── 系统/API 错误 ──
    {
        "name": "API/系统错误",
        "icon": "🔴",
        "keywords_evidence": [
            r"Error:", r"额度已用尽", r"40[13]", r"429", r"500",
            r"timeout", r"Errno", r"ConnectionError", r"rate.?limit",
        ],
        "optimizable": False,
        "desc": "LLM API 调用失败、超时、限流等系统级错误。",
    },
    # ── 页面崩溃 / A11y Tree 为空 ──
    {
        "name": "页面崩溃/A11y Tree为空",
        "icon": "💀",
        "keywords_evidence": PAGE_CRASH_KEYWORDS + [
            r"consistently\s+crash", r"crashes\s+and\s+disappears",
            r"(?:application|app|interface)\s+(?:crashed|crashes|crash\b)",
            r"causes?\s+(?:the\s+)?(?:application|page)\s+to\s+crash",
        ],
        "optimizable": "partial",
        "desc": "页面加载失败或操作过程中崩溃，A11y Tree 为空导致无法测试。",
    },
    # ── 视觉渲染 ──
    {
        "name": "视觉渲染盲区",
        "icon": "👁️",
        "keywords_a_reason": [
            r"显示失败", r"未显示", r"无法显示", r"渲染失败", r"不显示",
            r"缩略图", r"3[Dd].*模型", r"图片.*加载", r"样式.*异常",
            r"动画.*效果", r"颜色.*不对", r"布局.*错", r"无法展示",
        ],
        "optimizable": False,
        "desc": "A角标注为渲染/视觉问题，TextAgent 的 a11y tree 无法感知视觉渲染结果。",
    },
    # ── 功能缺失误报 (FP: Agent 看到 a11y 元素就报 Pass，实际功能不存在/不可用) ──
    {
        "name": "功能缺失误报",
        "icon": "🚫",
        "keywords_a_reason": [
            r"无此功能", r"没有此功能", r"没有该功能", r"没有该工具",
            r"无该功能", r"无注册功能", r"缺少.*页面", r"没有.*功能",
            r"无数据", r"搜索结果为空", r"mock数据", r"是mock",
            r"展示原模版", r"原模版",
            r"不可用", r"功能不可用", r"组合功能不可用",
            r"无法正常结束", r"一直thinking",
            r"交互后报错", r"无交互",
            r"存在但无法访问", r"无法访问",
            r"元素无法选择", r"点击.*无交互",
            r"不是全屏", r"不是.*模式",
            r"资源数量不正确", r"数量不正确",
            r"没有信息卡片", r"没有提示",
            r"页面上未找到", r"点击后未展示",
            r"无法使用.*手势",
        ],
        "optimizable": "partial",
        "desc": "Agent 在 a11y tree 中看到相关元素就报 Pass，但实际功能不存在、不可用或数据不正确。",
    },
    # ── 控件暴露不完整 ──
    {
        "name": "A11y Tree 暴露不完整",
        "icon": "🌳",
        "keywords_evidence": [
            r"no.*slider", r"no.*button.*found", r"not.*found.*element",
            r"not.*exposed", r"not.*accessible", r"no.*interactive",
            r"search.*return.*no.*result", r"Ctrl\+F.*nothing",
            r"no.*control", r"static text",
            # 标签全相同 / 内容不可区分
            r"all.*labeled", r"all are labeled", r"does not expose",
            r"not.*visible.*on.*page", r"not.*present.*on.*page",
            r"not found on the page",
            r"was not found on", r"were not found",
            r"input.*not.*visible", r"input.*not.*present",
            r"no.*tooltip", r"no.*text.*for",
        ],
        "keywords_a_reason": [
            r"无法.*操作", r"控件.*缺失",
        ],
        "optimizable": "partial",
        "desc": "页面控件（slider, 图标按钮等）在 a11y tree 中暴露不完整，或标签内容不可区分。",
    },
    # ── 交互验证不足 ──
    {
        "name": "交互验证薄弱",
        "icon": "🔄",
        "keywords_evidence": [
            r"success", r"updated", r"changed",  # Agent 认为成功
        ],
        "keywords_a_reason": [
            r"无法.*编辑", r"无法.*修改", r"无法.*打开", r"无法.*操作",
            r"不生效", r"未保存", r"无法.*提交",
            r"无法添加", r"无法.*平移",
        ],
        "optimizable": True,
        "desc": "Agent 报 Pass（看到文字变化就认为成功），但实际功能未真正生效。",
    },
    # ── 操作执行失败 ──
    {
        "name": "操作执行问题",
        "icon": "⚙️",
        "keywords_evidence": [
            r"No functions detected", r"placeholder", r"did not.*update",
            r"did not.*change", r"remained", r"still.*show",
            r"no.*result", r"no.*output", r"no.*response",
            r"failed to", r"could not",
            r"does not support", r"only supports? single",
            r"0 km/h despite", r"vehicle.*remains",
            r"button.*not.*found", r"was interrupted",
        ],
        "optimizable": True,
        "desc": "Agent 的操作没有正确执行（代码编辑器输入失败、按钮无响应等）。",
    },
    # ── 步数/探索不足 ──
    {
        "name": "步数/探索不足",
        "icon": "⏱️",
        "keywords_evidence": [
            r"same ending", r"not.*consistently", r"only.*one",
            r"insufficient", r"not.*fully",
            r"maximum steps", r"reached.*max",
            r"did not progress beyond", r"stopped at",
            r"not.*completed?\.?\s", r"was not complete",
            r"waited.*\d+.*seconds.*but no",
        ],
        "optimizable": True,
        "desc": "在有限步数内无法完成复杂验证（如需多次游戏 playthrough、长时间等待）。",
    },
]


def classify_error(row, score_col, evidence_col, true_col="A 角分数"):
    """对单个错误 case 进行自动归因分类。

    Returns:
        (category_name, icon, is_optimizable, matched_rule_detail)
    """
    pred = int(row[score_col])
    truth = int(row[true_col])
    evidence = str(row.get(evidence_col, "")).lower()
    a_reason = str(row.get("A 角0分原因", "")).lower()

    is_fp = pred == 1 and truth == 0  # Agent 报 Pass，实际 Fail
    is_fn = pred == 0 and truth == 1  # Agent 报 Fail，实际 Pass

    for rule in ERROR_CATEGORY_RULES:
        matched = False

        # 检查 evidence 关键词
        for pat in rule.get("keywords_evidence", []):
            if re.search(pat, evidence, re.IGNORECASE):
                matched = True
                break

        # 检查 A角0分原因 关键词（仅 FP 时有意义：Agent 说 Pass 但实际 Fail）
        if not matched and is_fp:
            for pat in rule.get("keywords_a_reason", []):
                if re.search(pat, a_reason, re.IGNORECASE):
                    matched = True
                    break

        # 交互验证薄弱：需要 FP + evidence 包含成功词 + A角原因包含失败词
        if rule["name"] == "交互验证薄弱" and is_fp:
            ev_has_success = any(
                re.search(p, evidence, re.IGNORECASE)
                for p in rule.get("keywords_evidence", [])
            )
            a_has_fail = any(
                re.search(p, a_reason, re.IGNORECASE)
                for p in rule.get("keywords_a_reason", [])
            )
            matched = ev_has_success and a_has_fail

        # 视觉渲染盲区 for FP only
        if rule["name"] == "视觉渲染盲区" and not is_fp:
            matched = False  # 只对 FP 生效（Agent 误以为功能正常）

        # 功能缺失误报 for FP only（Agent 看到元素报 Pass，实际功能不存在）
        if rule["name"] == "功能缺失误报" and not is_fp:
            matched = False

        if matched:
            return rule["name"], rule["icon"], rule["optimizable"], rule["desc"]

    return "其他/未分类", "❓", True, "无法自动归因，建议人工审查。"


def analyze_errors(file_path, score_col, evidence_col, true_col="A 角分数",
                   export_path=None):
    """完整的错误归因分析。"""
    df = pd.read_excel(file_path)
    tested = df[df[score_col].notna()]
    valid = tested[tested[true_col].notna()].copy()
    valid[score_col] = valid[score_col].astype(int)
    valid[true_col] = valid[true_col].astype(int)

    wrong = valid[valid[score_col] != valid[true_col]]
    correct = valid[valid[score_col] == valid[true_col]]

    n_total = len(valid)
    n_wrong = len(wrong)
    n_correct = len(correct)
    acc = 100 * n_correct / n_total if n_total > 0 else 0

    tp = len(valid[(valid[score_col] == 1) & (valid[true_col] == 1)])
    fp = len(valid[(valid[score_col] == 1) & (valid[true_col] == 0)])
    fn = len(valid[(valid[score_col] == 0) & (valid[true_col] == 1)])
    tn = len(valid[(valid[score_col] == 0) & (valid[true_col] == 0)])

    lines = []  # for export

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"{'='*90}")
    out(f"📊 错误归因分析报告")
    out(f"   文件: {file_path}")
    out(f"   时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"{'='*90}")
    out()
    out(f"## 1. 总体性能")
    out()
    out(f"| 指标 | 数值 |")
    out(f"|------|------|")
    out(f"| 已评估 | {n_total} |")
    out(f"| 正确 | {n_correct} ({acc:.1f}%) |")
    out(f"| 错误 | {n_wrong} ({100-acc:.1f}%) |")
    out(f"| TP (真阳) | {tp} |")
    out(f"| FP (假阳/误报Pass) | {fp} |")
    out(f"| FN (假阴/漏报Pass) | {fn} |")
    out(f"| TN (真阴) | {tn} |")
    if tp + fn > 0:
        out(f"| Pass 召回率 | {100*tp/(tp+fn):.1f}% |")
    if tp + fp > 0:
        out(f"| Pass 精确率 | {100*tp/(tp+fp):.1f}% |")
    if tn + fp > 0:
        out(f"| Fail 召回率 | {100*tn/(tn+fp):.1f}% |")
    out()

    if n_wrong == 0:
        out("✅ 没有错误案例！")
        if export_path:
            Path(export_path).write_text("\n".join(lines), encoding="utf-8")
            out(f"\n📝 报告已导出到: {export_path}")
        return

    # ── 分类所有错误 ──
    categories = defaultdict(list)
    for idx, row in wrong.iterrows():
        cat_name, icon, optimizable, desc = classify_error(
            row, score_col, evidence_col, true_col
        )
        pred = "Pass" if row[score_col] == 1 else "Fail"
        truth = "Pass" if row[true_col] == 1 else "Fail"
        error_type = "FP" if pred == "Pass" else "FN"
        categories[cat_name].append({
            "idx": idx,
            "row": row,
            "icon": icon,
            "optimizable": optimizable,
            "desc": desc,
            "pred": pred,
            "truth": truth,
            "error_type": error_type,
        })

    # ── 2. 分类汇总表 ──
    out(f"## 2. 错误分类汇总")
    out()
    out(f"| 类别 | 数量 | 占比 | FP | FN | 可优化 |")
    out(f"|------|------|------|----|----|--------|")
    for cat_name in ERROR_CATEGORY_RULES:
        name = cat_name["name"]
        if name in categories:
            items = categories[name]
            icon = items[0]["icon"]
            n_fp = sum(1 for x in items if x["error_type"] == "FP")
            n_fn = sum(1 for x in items if x["error_type"] == "FN")
            opt = items[0]["optimizable"]
            opt_str = "❌ 不可" if opt is False else ("⚠️ 部分" if opt == "partial" else "✅ 可以")
            out(f"| {icon} {name} | {len(items)} | {100*len(items)/n_wrong:.0f}% | {n_fp} | {n_fn} | {opt_str} |")
    # 其他/未分类
    if "其他/未分类" in categories:
        items = categories["其他/未分类"]
        n_fp = sum(1 for x in items if x["error_type"] == "FP")
        n_fn = sum(1 for x in items if x["error_type"] == "FN")
        out(f"| ❓ 其他/未分类 | {len(items)} | {100*len(items)/n_wrong:.0f}% | {n_fp} | {n_fn} | ✅ 待审查 |")
    out()

    # 可优化的比例
    n_optimizable = sum(
        len(items) for items in categories.values()
        if items and items[0]["optimizable"] is not False
    )
    out(f"**可优化错误**: {n_optimizable}/{n_wrong} ({100*n_optimizable/n_wrong:.0f}%)")
    if n_total > 0:
        theoretical_acc = 100 * (n_correct + n_optimizable) / n_total
        out(f"**理论最优准确率**: {theoretical_acc:.1f}% (修复所有可优化错误后)")
    out()

    # ── 2.5 页面崩溃 / A11y Tree 子类型分析 ──
    crash_cat_name = "页面崩溃/A11y Tree为空"
    if crash_cat_name in categories:
        crash_items = categories[crash_cat_name]
        subtypes = defaultdict(list)
        for item in crash_items:
            ev_text = str(item["row"].get(evidence_col, ""))
            _, subtype = _detect_page_crash(ev_text)
            subtypes[subtype or "其他空树"].append(item)

        out(f"### 💀 页面崩溃子类型分析 ({len(crash_items)}个)")
        out()
        out(f"| 子类型 | 数量 | 占比 | 涉及 App |")
        out(f"|--------|------|------|----------|")
        app_col = "case_name" if "case_name" in valid.columns else None
        for st, items in sorted(subtypes.items(), key=lambda x: -len(x[1])):
            apps = set()
            if app_col:
                for item in items:
                    a = item["row"].get(app_col, "")
                    if pd.notna(a):
                        apps.add(str(a))
            apps_str = ", ".join(sorted(apps)[:5]) + ("..." if len(apps) > 5 else "")
            out(f"| {st} | {len(items)} | {100*len(items)/len(crash_items):.0f}% | {apps_str} |")
        out()

        # 延时建议
        initial_fail = len(subtypes.get("初始加载失败", []))
        mid_crash = len(subtypes.get("操作中崩溃", []))
        out(f"**💡 A11y Tree 延时 / 稳定性建议:**")
        if initial_fail > 0:
            out(f"  - 初始加载失败 {initial_fail} 个 → 建议增加 `SLEEP_AFTER_START_WEB` (当前=10s, 建议 15~20s)")
            out(f"    或在 Agent 首次获取空 a11y tree 时自动重试 (sleep 5s → 再取一次)")
        if mid_crash > 0:
            out(f"  - 操作中崩溃 {mid_crash} 个 → Chrome 渲染资源不足, 建议降低 `workers` 并发数")
            out(f"    或增加操作后 sleep 时间让页面有足够 CPU 重新渲染")
        if initial_fail + mid_crash == 0:
            out(f"  - 全部为'其他空树'类型, 建议人工检查具体 evidence")
        out()

    # ── 3. 按 App 聚合 ──
    out(f"## 3. 按 App 聚合错误")
    out()
    app_col = "case_name" if "case_name" in df.columns else None
    if app_col:
        app_errors = defaultdict(lambda: {"total": 0, "wrong": 0, "fp": 0, "fn": 0, "cats": defaultdict(int)})
        for _, row in valid.iterrows():
            app = row[app_col]
            app_errors[app]["total"] += 1
            if row[score_col] != row[true_col]:
                app_errors[app]["wrong"] += 1
                if row[score_col] == 1:
                    app_errors[app]["fp"] += 1
                else:
                    app_errors[app]["fn"] += 1

        for cat_name, items in categories.items():
            for item in items:
                app = item["row"].get(app_col, "N/A")
                app_errors[app]["cats"][cat_name] += 1

        # 按错误数降序
        sorted_apps = sorted(app_errors.items(), key=lambda x: x[1]["wrong"], reverse=True)
        out(f"| App | 评估 | 错误 | 错误率 | FP | FN | 主要错因 |")
        out(f"|-----|------|------|--------|----|----|----------|")
        for app, info in sorted_apps:
            if info["wrong"] == 0:
                continue
            err_rate = 100 * info["wrong"] / info["total"] if info["total"] > 0 else 0
            main_cat = max(info["cats"].items(), key=lambda x: x[1])[0] if info["cats"] else "-"
            out(f"| {app} | {info['total']} | {info['wrong']} | {err_rate:.0f}% | {info['fp']} | {info['fn']} | {main_cat} |")
        out()

    # ── 4. 逐条错误详情（表格） ──
    out(f"## 4. 逐条错误详情 ({n_wrong}个)")
    out()

    test_col = "test_case" if "test_case" in df.columns else None
    test_ch_col = "测试点" if "测试点" in df.columns else None
    a_reason_col = "A 角0分原因" if "A 角0分原因" in df.columns else None

    # 收集所有分类的错误，按类别输出表格
    all_cat_names = [r["name"] for r in ERROR_CATEGORY_RULES] + ["其他/未分类"]
    for cat_name in all_cat_names:
        if cat_name not in categories:
            continue
        items = categories[cat_name]
        icon = items[0]["icon"]
        desc = items[0]["desc"]
        out(f"### {icon} {cat_name} ({len(items)}个) — {desc}")
        out()

        # 表头
        header =  f"  {'#':<3} {'类型':<4} {'App':<28} {'测试点':<35} {'Agent证据(摘要)':<50} {'A角原因':<20}"
        sep =     f"  {'─'*3} {'─'*4} {'─'*28} {'─'*35} {'─'*50} {'─'*20}"
        out(header)
        out(sep)

        for i, item in enumerate(items, 1):
            row = item["row"]
            etype = item["error_type"]
            app = str(row.get(app_col, ""))[:26] if app_col else ""
            test_desc = ""
            if test_ch_col and pd.notna(row.get(test_ch_col)):
                test_desc = str(row.get(test_ch_col, ""))[:33]
            elif test_col:
                test_desc = str(row.get(test_col, ""))[:33]
            evidence = str(row.get(evidence_col, ""))[:48].replace("\n", " ")
            a_reason = ""
            if a_reason_col and pd.notna(row.get(a_reason_col)) and str(row.get(a_reason_col)) != "nan":
                a_reason = str(row.get(a_reason_col))[:18]
            out(f"  {i:<3} {etype:<4} {app:<28} {test_desc:<35} {evidence:<50} {a_reason:<20}")

        out()

    # ── 5. 优化建议（表格） ──
    out(f"## 5. 优化建议")
    out()
    suggestions = {
        "页面崩溃/A11y Tree为空": "降低workers并发数；增加SLEEP_AFTER_START_WEB(当前10s)；增加a11y tree重试",
        "视觉渲染盲区": "需截图/VLM验证视觉结果；混合模式：TextAgent操作+VLM截图验证",
        "功能缺失误报": "prompt增加规则：仅凭a11y tree有元素不能判Pass，需实际交互验证功能可用；对搜索/数据类验证内容非空",
        "A11y Tree 暴露不完整": "Tab遍历获取焦点；CDP DOM.getDocument直接查询；aria-label搜索",
        "交互验证薄弱": "prompt增加规则：操作后必须验证元素状态变化（检查value属性）",
        "操作执行问题": "CodeMirror等用CDP Runtime.evaluate注入代码；增加输入后验证",
        "步数/探索不足": "增加max_iters；prompt中提示高效探索策略",
        "API/系统错误": "更换API Key；增加retry配置；降低并发数",
    }
    out(f"  {'类别':<22} {'数量':<6} {'建议':<70}")
    out(f"  {'─'*22} {'─'*6} {'─'*70}")
    for cat_name, suggestion in suggestions.items():
        if cat_name in categories:
            n = len(categories[cat_name])
            out(f"  {cat_name:<22} {n:<6} {suggestion:<70}")
    out()

    # ── 导出 ──
    if export_path:
        Path(export_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"\n📝 报告已导出到: {export_path}")


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
    parser.add_argument("--file", type=str, default="test2_text_agent_full_visual_test_results.xlsx",
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
    parser.add_argument("--analyze", action="store_true",
                        help="完整错误归因分析（分类 + 按App聚合 + 可优化建议）")
    parser.add_argument("--export", type=str, default=None,
                        help="导出分析报告到 markdown 文件 (配合 --analyze 使用)")
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

    # 分析模式
    if args.analyze:
        analyze_errors(file_path, score_col, evidence_col, args.true_label,
                       export_path=args.export)
        return

    # 摘要
    print_summary(file_path, score_col, evidence_col, args.true_label)

    # 错误详情
    if args.errors > 0:
        print_errors(file_path, score_col, evidence_col, args.true_label, args.errors)


if __name__ == "__main__":
    main()

