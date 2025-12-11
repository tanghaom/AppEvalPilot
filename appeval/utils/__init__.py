"""
Utils 模块

此模块为向后兼容保留，新代码请使用 appeval.em 模块。

所有 EM 相关功能已迁移到 appeval.em 模块：
- Evidence, OnlineEvidenceCollector, AutoAnnotationTool -> appeval.em
- SimpleEM4EvidenceH_Refine, EMManager, create_em_manager -> appeval.em
- convert_evidences_to_em_format, process_osagent_evidences -> appeval.em
- load_gui_evidence_from_jsonl, load_code_evidence_from_jsonl -> appeval.em
- get_code_evidence_dict_from_jsonl, merge_evidences_for_em -> appeval.em
- evidences_to_dataframe, correct_agent_judgment -> appeval.em
"""

# 向后兼容：从 appeval.em 模块导入
from appeval.em import (
    # Evidence Annotator
    Evidence,
    OnlineEvidenceCollector,
    AutoAnnotationTool,
    # EM Data Process
    convert_evidences_to_em_format,
    process_osagent_evidences,
    load_gui_evidence_from_jsonl,
    load_code_evidence_from_jsonl,
    get_code_evidence_dict_from_jsonl,
    merge_evidences_for_em,
    evidences_to_dataframe,
    # EM Model
    SimpleEM4EvidenceH_Refine,
    EMManager,
    create_em_manager,
    correct_agent_judgment,
)

__all__ = [
    # Evidence Annotator
    "Evidence",
    "OnlineEvidenceCollector",
    "AutoAnnotationTool",
    # EM Data Process
    "convert_evidences_to_em_format",
    "process_osagent_evidences",
    "load_gui_evidence_from_jsonl",
    "load_code_evidence_from_jsonl",
    "get_code_evidence_dict_from_jsonl",
    "merge_evidences_for_em",
    "evidences_to_dataframe",
    # EM Model
    "SimpleEM4EvidenceH_Refine",
    "EMManager",
    "create_em_manager",
    "correct_agent_judgment",
]
