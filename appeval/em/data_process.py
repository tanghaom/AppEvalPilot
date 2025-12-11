"""
EM 数据处理模块

用于处理 OSAgent 收集的证据数据，将其转换为 EM 模型训练所需的格式。
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from appeval.em.evidence import Evidence


def get_gui_evidence_from_evidence(evidence: Evidence) -> int:
    """
    从 Evidence 对象中提取 GUI 证据评分

    Args:
        evidence: Evidence 对象

    Returns:
        GUI 证据评分：1=命中, 0=未命中, -1=不需要分析
    """
    if evidence.coordinate_match is not None:
        return evidence.coordinate_match
    return -1  # -1 表示不需要进行分析的动作


def load_evidences_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    从 JSONL 文件加载证据数据

    Args:
        jsonl_path: JSONL 文件路径

    Returns:
        证据字典列表
    """
    evidences = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            evidences.append(data)
    return evidences


def evidences_to_dataframe(evidences: List[Dict], project_name: str = "default") -> pd.DataFrame:
    """
    将证据列表转换为 DataFrame 格式

    Args:
        evidences: 证据字典列表（来自 Evidence.to_dict() 或 JSONL 文件）
        project_name: 项目名称

    Returns:
        包含证据数据的 DataFrame
    """
    rows = []
    for data in evidences:
        action_id = f"{project_name}_iter_{data.get('iter_num', 0)}"

        # 提取 GUI 证据
        gui_evidence = -1
        if data.get("coordinate_match") is not None:
            gui_evidence = data["coordinate_match"]

        row = {
            "test_case_id": project_name,
            "action_id": action_id,
            "iter_num": data.get("iter_num", 0),
            "operation_desc": data.get("operation_desc"),
            "reflection_thought": data.get("reflection_thought"),
            "action_content": data.get("action_content"),
            "gui_evidence": gui_evidence,
            "tell_evidence": data.get("tell_evidence"),
            "agent_noresp": data.get("agent_noresp"),
            "error_flag": data.get("error_flag", False),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def convert_evidences_to_em_format(
    evidences: List[Dict],
    project_name: str = "default",
    code_evidence: Optional[Dict] = None,
    agent_judge: Optional[Dict] = None,
    gt_label: Optional[int] = None,
) -> pd.DataFrame:
    """
    将证据数据转换为 EM 模型训练格式

    Args:
        evidences: 证据字典列表
        project_name: 项目名称
        code_evidence: 代码审查证据（可选）
        agent_judge: Agent 判断评分（可选）
        gt_label: 真值标签（可选）

    Returns:
        EM 模型训练格式的 DataFrame
    """
    rows_out = []

    for data in evidences:
        action_id = f"{project_name}_iter_{data.get('iter_num', 0)}"
        weight = 1.0

        # 处理 GUI 证据
        gui_evidence_val = data.get("coordinate_match")
        if gui_evidence_val is None or gui_evidence_val == -1:
            M_gui = 1  # mask
            gui_evidence = 0
        else:
            M_gui = 0
            gui_evidence = gui_evidence_val

        # 处理 agent_noresp
        agent_noresp_val = data.get("agent_noresp")
        if agent_noresp_val is None:
            M_noresp = 1  # mask
            agent_noresp = 0
            is_reflection = 0
            M_reflect = 1
            weight = 0.3
        else:
            M_noresp = 0
            # agent_noresp: 1=有问题, 0=正常（已经是正确格式）
            agent_noresp = agent_noresp_val
            is_reflection = 1
            M_reflect = 1
            M_gui = 1  # 有 Tell 动作时，GUI 证据可能不适用
            weight = 0.5

        # 代码证据
        code_evidence_val = 0
        if code_evidence and action_id in code_evidence:
            code_evidence_val = code_evidence[action_id]
        M_code = 0 if code_evidence else 1

        # Agent 判断评分
        agent_score = np.nan
        if agent_judge and project_name in agent_judge:
            agent_score = agent_judge[project_name]

        rows_out.append(
            {
                "test_case_id": project_name,
                "step": action_id,
                "phi": gt_label if gt_label is not None else np.nan,
                "operation_desc": data.get("operation_desc", ""),
                "action_content": data.get("action_content", ""),
                "E1_gui": gui_evidence,
                "E2_code": code_evidence_val,
                "E3_reflect": agent_score,
                "E4_noresp": agent_noresp,
                "M_gui": M_gui,
                "M_code": M_code,
                "M_reflect": M_reflect,
                "M_noresp": M_noresp,
                "weight": weight,
                "is_reflection": is_reflection,
                "agent_testcase_score": agent_score,
            }
        )

    return pd.DataFrame(rows_out)


def process_osagent_evidences(
    evidence_collector,
    project_name: str = None,
    gt_label: Optional[int] = None,
) -> pd.DataFrame:
    """
    处理 OSAgent 的证据收集器数据，转换为 EM 模型格式

    Args:
        evidence_collector: OSAgent 的证据收集器实例
        project_name: 项目名称，如果为 None 则使用收集器的项目名称
        gt_label: 真值标签

    Returns:
        EM 模型训练格式的 DataFrame
    """
    if project_name is None:
        project_name = evidence_collector.project_name

    evidences = evidence_collector.get_evidences_for_em()
    return convert_evidences_to_em_format(
        evidences=evidences,
        project_name=project_name,
        gt_label=gt_label,
    )


# ============== 以下为兼容旧版本的辅助函数 ==============


def get_gui_evidence(traj: List[Dict]) -> List[int]:
    """
    从轨迹数据中提取 GUI 证据列表（兼容旧版本）

    Args:
        traj: 轨迹数据列表

    Returns:
        GUI 证据评分列表
    """
    gui_evidence = []
    for action in traj:
        if action.get("coordinate_analysis") is not None:
            try:
                gui_score = action["coordinate_analysis"]["accuracy"]
                gui_evidence.append(gui_score)
            except:
                gui_evidence.append(0)  # 0 means fail
        else:
            gui_evidence.append(-1)  # -1 means 不需要进行分析的动作
    return gui_evidence


def load_gui_evidence_from_jsonl(jsonl_path: str) -> pd.DataFrame:
    """
    从 JSONL 文件加载 GUI 证据数据（兼容旧版本）

    Args:
        jsonl_path: JSONL 文件路径

    Returns:
        包含 GUI 证据的 DataFrame
    """
    gui_evidence = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            project_name = data.get("project_name", "default")
            action_id = f"{project_name}_iter_{data.get('iter_num', 0)}"

            gui_evidence_dict = {
                "test_case_id": project_name,
                "action_id": action_id,
                "operation_desc": data.get("operation_desc"),
                "reflection_thought": data.get("reflection_thought"),
                "action_content": data.get("action_content"),
                "gui_evidence": get_gui_evidence([data])[-1],
                "tell_evidence": data.get("tell_evidence"),
                "agent_noresp": data.get("agent_noresp"),
            }
            gui_evidence.append(gui_evidence_dict)

    return pd.DataFrame(gui_evidence)


def merge_evidences_for_em(
    gui_evidence_df: Optional[pd.DataFrame] = None,
    reflection_evidence_df: Optional[pd.DataFrame] = None,
    code_evidence_df: Optional[pd.DataFrame] = None,
    merge_on: str = "test_case_id",
) -> pd.DataFrame:
    """
    合并 GUI、反思、代码三种证据用于 EM 预测

    Args:
        gui_evidence_df: GUI 证据 DataFrame，需包含 test_case_id 和 gui_evidence 列
        reflection_evidence_df: 反思证据 DataFrame，需包含 test_case_id 和 reflection_evidence 列
        code_evidence_df: 代码证据 DataFrame，需包含 test_case_id 和 code_evidence 列
        merge_on: 合并的键列名

    Returns:
        合并后的 DataFrame，包含 E1_gui, E2_code, E3_reflect 等 EM 所需的列
    """
    # 收集所有非空的 DataFrame
    dfs = []

    if gui_evidence_df is not None and not gui_evidence_df.empty:
        # 重命名列以符合 EM 格式
        gui_df = gui_evidence_df.copy()
        if "gui_evidence" in gui_df.columns:
            gui_df["E1_gui"] = gui_df["gui_evidence"].apply(
                lambda x: 0 if x == -1 else x)
            gui_df["M_gui"] = gui_df["gui_evidence"].apply(
                lambda x: 1 if x == -1 else 0)
        dfs.append(gui_df)

    if code_evidence_df is not None and not code_evidence_df.empty:
        code_df = code_evidence_df.copy()
        if "code_evidence" in code_df.columns:
            code_df["E2_code"] = code_df["code_evidence"]
            code_df["M_code"] = 0  # 有 code evidence 时不 mask
        dfs.append(code_df)

    if reflection_evidence_df is not None and not reflection_evidence_df.empty:
        reflect_df = reflection_evidence_df.copy()
        if "reflection_evidence" in reflect_df.columns:
            reflect_df["E3_reflect"] = reflect_df["reflection_evidence"]
            reflect_df["M_reflect"] = 0
        dfs.append(reflect_df)

    if not dfs:
        return pd.DataFrame()

    # 合并所有 DataFrame
    merged_df = dfs[0]
    for df in dfs[1:]:
        # 获取共同列（除了 merge_on）
        common_cols = set(merged_df.columns) & set(df.columns) - {merge_on}
        # 只保留需要合并的列
        df_to_merge = df[[
            merge_on] + [c for c in df.columns if c not in common_cols or c == merge_on]]
        merged_df = pd.merge(merged_df, df_to_merge, on=merge_on, how="outer")

    # 填充缺失值和 mask
    if "E1_gui" not in merged_df.columns:
        merged_df["E1_gui"] = 0
        merged_df["M_gui"] = 1
    if "E2_code" not in merged_df.columns:
        merged_df["E2_code"] = 0
        merged_df["M_code"] = 1
    if "E3_reflect" not in merged_df.columns:
        merged_df["E3_reflect"] = np.nan
        merged_df["M_reflect"] = 1

    # 填充 NaN
    merged_df["E1_gui"] = merged_df["E1_gui"].fillna(0)
    merged_df["E2_code"] = merged_df["E2_code"].fillna(0)
    merged_df["M_gui"] = merged_df["M_gui"].fillna(1)
    merged_df["M_code"] = merged_df["M_code"].fillna(1)
    merged_df["M_reflect"] = merged_df["M_reflect"].fillna(1)

    return merged_df


def load_code_evidence_from_jsonl(jsonl_path: str) -> pd.DataFrame:
    """
    从 JSONL 文件加载 code review 证据数据

    Args:
        jsonl_path: JSONL 文件路径（如 webdevjudge_with_code_review_concate.jsonl）

    Returns:
        包含 code evidence 的 DataFrame，格式：
        - case_name: 网站ID（如 web_0）
        - test_case_id: 测试用例ID（如 web_001）
        - test_case_zh: 测试用例描述
        - code_evidence: 1 表示代码已实现，0 表示未实现
    """
    logger = logging.getLogger(__name__)

    data_list = []
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            web_id = data.get("web_id", "default")
            task_id = data.get("task_id", 0)
            task = data.get("task", "")
            code_review = data.get("code_review", {})

            # 提取 is_implemented 字段
            is_implemented = code_review.get("is_implemented", False)

            # 生成 test_case_id：web_id + 格式化的 task_id
            test_case_id = f"{web_id}_{task_id:02d}"

            data_list.append(
                {
                    "case_name": web_id,
                    "test_case_id": test_case_id,
                    "test_case_zh": task,
                    "code_evidence": 1 if is_implemented is True else 0,
                }
            )
            count += 1

    logger.info(f"Loaded {count} code evidence entries from {jsonl_path}")

    return pd.DataFrame(data_list)


def get_code_evidence_dict_from_jsonl(jsonl_path: str) -> Dict[str, Dict[str, int]]:
    """
    从 JSONL 文件加载 code evidence 为嵌套字典格式

    Args:
        jsonl_path: JSONL 文件路径

    Returns:
        嵌套字典: {web_id: {test_case_id: code_evidence}}
    """
    code_evidence = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            web_id = data.get("web_id", "default")
            task_id = data.get("task_id", 0)
            code_review = data.get("code_review", {})

            is_implemented = code_review.get("is_implemented", False)
            test_case_id = f"{web_id}_{task_id:02d}"

            if web_id not in code_evidence:
                code_evidence[web_id] = {}
            code_evidence[web_id][test_case_id] = 1 if is_implemented is True else 0

    return code_evidence


if __name__ == "__main__":
    # 示例用法
    pass
