"""
EM Data Process Module

此模块用于处理 OSAgent 收集的证据数据，将其转换为 EM 模型训练所需的格式。
功能已集成到 OnlineEvidenceCollector 中，此模块提供数据转换和辅助功能。
"""
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from appeval.utils.evidence_annotator import Evidence, OnlineEvidenceCollector


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
    evidence_collector: OnlineEvidenceCollector,
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


if __name__ == "__main__":
    # 示例用法
    pass
