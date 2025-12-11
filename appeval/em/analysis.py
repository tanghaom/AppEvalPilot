"""
EM 分析模块

提供分析函数，包括翻转分析、混淆矩阵、纠偏分析等。
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def analyze_flips(val_df: pd.DataFrame, out_dir: Optional[str] = None):
    """
    区分两类子集: 原判断正确 vs 错误，查看flip比例

    Args:
        val_df: 验证数据 DataFrame，需包含 human_gt, agent_original, corrected_label 列
        out_dir: 输出目录，如果提供则保存详细数据
    """
    df = val_df.copy()
    df["is_correct_before"] = df["human_gt"] == df["agent_original"]
    df["is_correct_after"] = df["human_gt"] == df["corrected_label"]
    df["is_flipped"] = df["agent_original"] != df["corrected_label"]

    # subset A: 原判断正确
    subset_A = df[df["is_correct_before"]]
    misflipped = subset_A[subset_A["is_flipped"]]
    if len(subset_A) > 0:
        print(
            f"[Subset A] 原判断正确: {len(subset_A)} cases, 被误翻 {len(misflipped)} ({len(misflipped)/len(subset_A):.2%})")

        # 进一步区分原始标签是0和原始标签是1的误翻情况
        misflipped_0 = misflipped[misflipped["agent_original"] == 0]
        misflipped_1 = misflipped[misflipped["agent_original"] == 1]
        subset_A_0 = subset_A[subset_A["agent_original"] == 0]
        subset_A_1 = subset_A[subset_A["agent_original"] == 1]

        if len(subset_A_0) > 0:
            print(
                f"  - 原始标签=0: {len(subset_A_0)} cases, 被误翻 {len(misflipped_0)} ({len(misflipped_0)/len(subset_A_0):.2%})")
        if len(subset_A_1) > 0:
            print(
                f"  - 原始标签=1: {len(subset_A_1)} cases, 被误翻 {len(misflipped_1)} ({len(misflipped_1)/len(subset_A_1):.2%})")

        # 保存数据
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # 保存被误翻的数据（按原始标签分组）
            if len(misflipped_0) > 0:
                misflipped_0.to_csv(
                    out_path / "misflipped_original_0.csv", index=False)
                print(
                    f"  已保存原始标签=0的误翻数据到: {out_path / 'misflipped_original_0.csv'}")
            if len(misflipped_1) > 0:
                misflipped_1.to_csv(
                    out_path / "misflipped_original_1.csv", index=False)
                print(
                    f"  已保存原始标签=1的误翻数据到: {out_path / 'misflipped_original_1.csv'}")

            # 保存原判断正确的所有数据（按原始标签分组）
            if len(subset_A_0) > 0:
                subset_A_0.to_csv(
                    out_path / "subset_A_original_0.csv", index=False)
                print(
                    f"  已保存原始标签=0的原判断正确数据到: {out_path / 'subset_A_original_0.csv'}")
            if len(subset_A_1) > 0:
                subset_A_1.to_csv(
                    out_path / "subset_A_original_1.csv", index=False)
                print(
                    f"  已保存原始标签=1的原判断正确数据到: {out_path / 'subset_A_original_1.csv'}")
    else:
        print(
            f"[Subset A] 原判断正确: {len(subset_A)} cases, 被误翻 {len(misflipped)}")

    # subset B: 原判断错误
    subset_B = df[~df["is_correct_before"]]
    corrected = subset_B[subset_B["is_correct_after"]]
    if len(subset_B) > 0:
        print(
            f"[Subset B] 原判断错误: {len(subset_B)} cases, 被成功纠正 {len(corrected)} ({len(corrected)/len(subset_B):.2%})")
    else:
        print(
            f"[Subset B] 原判断错误: {len(subset_B)} cases, 被成功纠正 {len(corrected)}")

    # 再打印最终 confusion
    print("\n=== Confusion Matrix (GT vs Corrected) ===")
    print(pd.crosstab(df["human_gt"], df["corrected_label"],
          rownames=["GT"], colnames=["Corrected"]))


def confusion_matrix(val_df: pd.DataFrame) -> pd.DataFrame:
    """
    统计人类标注 vs 矫正后结果

    Args:
        val_df: 验证数据 DataFrame

    Returns:
        混淆矩阵 DataFrame
    """
    conf_matrix = pd.crosstab(val_df["human_gt"], val_df["corrected_label"], rownames=[
                              "GT"], colnames=["Corrected"])
    print("=== Confusion Matrix (GT vs Corrected) ===")
    print(conf_matrix)
    acc = (val_df["human_gt"] == val_df["corrected_label"]).mean()
    print(f"Accuracy after correction: {acc:.3f}")
    return conf_matrix


def correct_agent_judgment(
    df: pd.DataFrame,
    em,
    tau_agentfail: float = 0.7,
    tau_envfail: float = 0.7,
    alpha: float = 0.75,
    tau_envfail_high: float = 0.7,
    col_case: str = "test_case_id",
    col_agent: str = "agent_testcase_score_x",
) -> pd.DataFrame:
    """
    对每个 case：
      - 汇总 step-level posterior 得到 P_case_AgentFail
      - 与 agent 原判 (C_case) 结合，给出纠偏动作

    Args:
        df: 证据 DataFrame
        em: SimpleEM4EvidenceH_Refine 模型实例
        tau_agentfail: AgentFail 判断阈值
        tau_envfail: EnvFail 判断阈值
        alpha: 阻塞概率指数
        tau_envfail_high: 高置信度 EnvFail 阈值
        col_case: case ID 列名
        col_agent: agent 评分列名

    Returns:
        纠偏结果 DataFrame
    """
    post = em.predict_proba(df)
    df_tmp = df.copy()
    df_tmp["P_EnvFail"] = post[:, 0]
    # AgentRetryFail (1) 和 AgentReasoningFail (2) 都视为 AgentFail
    df_tmp["P_AgentFail"] = post[:, 1] + post[:, 2]

    rows = []
    for cid, g in df_tmp.groupby(col_case):
        # agent 原判: 取该 case 最后一个非 nan
        C_vals = g[col_agent].dropna().values
        # 1=PASS?, 0=FAIL? 按你现在的定义自行对应
        C_case = int(C_vals[-1]) if len(C_vals) else None

        gt = g["phi"].dropna().values[-1] if len(g["phi"].dropna()) > 0 else None

        # 聚合为 case-level AgentFail 概率（阻塞口径）
        q = np.clip(g["P_AgentFail"].values, 0.0, 1.0)
        # 阻塞概率: 越多高风险 step, 越趋向 AgentFail
        P_case_AgentFail = 1.0 - float(np.prod((1.0 - q) ** alpha))
        P_case_EnvFail = 1.0 - P_case_AgentFail

        action = "keep_AgentJudge"
        corrected = C_case

        if C_case is not None:
            # 场景1: agent 判 FAIL (0)，我们要区分 Env vs Agent
            if C_case == 0:
                if P_case_AgentFail >= tau_agentfail:
                    corrected = 1  # 认定 AgentFail
                    action = "flip_to_AgentFail"
                elif P_case_EnvFail >= tau_envfail:
                    corrected = 0  # 环境问题，维持
                    action = "keep_EnvFail"
                # 介于两阈值之间可以保守 keep_AgentJudge/UNK

            # 场景2: agent 判 PASS (1)，可选：当强 EnvFail 证据时 flip
            elif C_case == 1:
                # 如果你只关心纠正误报，可以先不动这里
                if P_case_EnvFail >= tau_envfail_high:  # tau_envfail:
                    corrected = 0
                    action = "flip_to_EnvFail"
        rows.append(
            dict(
                case_id=cid,
                human_gt=gt,
                agent_original=C_case,
                P_case_EnvFail=P_case_EnvFail,
                P_case_AgentFail=P_case_AgentFail,
                corrected_label=corrected,
                action=action,
            )
        )
    val_df = pd.DataFrame(rows).sort_values("case_id")
    acc_original = val_df["agent_original"] == val_df["human_gt"]
    print(f"Accuracy (original): {acc_original.mean():.3f}")

    acc_correct = val_df["corrected_label"] == val_df["human_gt"]
    print(f"Accuracy (corrected): {acc_correct.mean():.3f}")
    analyze_flips(val_df)
    confusion_matrix(val_df)
    return val_df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算分类评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        包含各种指标的字典
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 基本指标
    accuracy = (y_true == y_pred).mean()

    # 对于二分类
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compare_before_after_correction(
    df: pd.DataFrame,
    col_gt: str = "human_gt",
    col_original: str = "agent_original",
    col_corrected: str = "corrected_label",
) -> dict:
    """
    比较纠偏前后的性能变化

    Args:
        df: 包含真值、原始预测、纠偏后预测的 DataFrame
        col_gt: 真值列名
        col_original: 原始预测列名
        col_corrected: 纠偏后预测列名

    Returns:
        包含对比指标的字典
    """
    metrics_before = calculate_metrics(
        df[col_gt].values, df[col_original].values)
    metrics_after = calculate_metrics(
        df[col_gt].values, df[col_corrected].values)

    print("=== 纠偏前后对比 ===")
    print(f"原始准确率: {metrics_before['accuracy']:.3f}")
    print(f"纠偏后准确率: {metrics_after['accuracy']:.3f}")
    print(
        f"准确率提升: {metrics_after['accuracy'] - metrics_before['accuracy']:.3f}")

    return {
        "before": metrics_before,
        "after": metrics_after,
        "improvement": {
            "accuracy": metrics_after["accuracy"] - metrics_before["accuracy"],
            "precision": metrics_after["precision"] - metrics_before["precision"],
            "recall": metrics_after["recall"] - metrics_before["recall"],
            "f1": metrics_after["f1"] - metrics_before["f1"],
        },
    }


def analyze_error_cases(
    df: pd.DataFrame,
    col_gt: str = "human_gt",
    col_pred: str = "corrected_label",
    col_case: str = "case_id",
) -> pd.DataFrame:
    """
    分析错误案例

    Args:
        df: 数据 DataFrame
        col_gt: 真值列名
        col_pred: 预测列名
        col_case: 案例 ID 列名

    Returns:
        错误案例 DataFrame
    """
    errors = df[df[col_gt] != df[col_pred]].copy()

    # 分类错误类型
    errors["error_type"] = errors.apply(
        lambda row: "FP" if row[col_gt] == 0 and row[col_pred] == 1 else (
            "FN" if row[col_gt] == 1 and row[col_pred] == 0 else "Unknown"),
        axis=1,
    )

    print(f"=== 错误案例分析 ===")
    print(f"总错误数: {len(errors)}")
    print(f"假阳性 (FP): {(errors['error_type'] == 'FP').sum()}")
    print(f"假阴性 (FN): {(errors['error_type'] == 'FN').sum()}")

    return errors


def generate_correction_report(
    val_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """
    生成纠偏报告

    Args:
        val_df: 验证数据 DataFrame
        output_path: 输出文件路径

    Returns:
        报告字符串
    """
    lines = []
    lines.append("=" * 50)
    lines.append("EM 纠偏分析报告")
    lines.append("=" * 50)
    lines.append("")

    # 基本统计
    total = len(val_df)
    flipped = (val_df["agent_original"] != val_df["corrected_label"]).sum()
    lines.append(f"总案例数: {total}")
    lines.append(f"被翻转案例数: {flipped} ({flipped/total:.2%})")
    lines.append("")

    # 准确率
    if "human_gt" in val_df.columns:
        acc_before = (val_df["human_gt"] == val_df["agent_original"]).mean()
        acc_after = (val_df["human_gt"] == val_df["corrected_label"]).mean()
        lines.append(f"原始准确率: {acc_before:.3f}")
        lines.append(f"纠偏后准确率: {acc_after:.3f}")
        lines.append(f"准确率变化: {acc_after - acc_before:+.3f}")
        lines.append("")

    # 动作分布
    lines.append("纠偏动作分布:")
    action_counts = val_df["action"].value_counts()
    for action, count in action_counts.items():
        lines.append(f"  {action}: {count} ({count/total:.2%})")
    lines.append("")

    # 概率分布
    lines.append("概率统计:")
    lines.append(f"  P_EnvFail 均值: {val_df['P_case_EnvFail'].mean():.3f}")
    lines.append(f"  P_AgentFail 均值: {val_df['P_case_AgentFail'].mean():.3f}")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存到: {output_path}")

    return report
