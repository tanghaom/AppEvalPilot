"""
EM 模块

提供证据收集、EM 模型训练与预测、根因分析与纠偏功能。

模块结构：
- evidence.py: Evidence 数据结构
- collector.py: OnlineEvidenceCollector 在线证据收集器
- annotator.py: AutoAnnotationTool 自动标注工具
- model.py: SimpleEM4EvidenceH_Refine EM 模型核心
- manager.py: EMManager 管理器和纠偏逻辑
- data_process.py: 数据处理和转换功能
- analysis.py: 分析函数（翻转分析、混淆矩阵等）

使用示例:
```python
from appeval.em import EMManager, OnlineEvidenceCollector, Evidence

# 创建证据收集器
collector = OnlineEvidenceCollector(
    output_dir="evidence",
    project_name="my_project"
)

# 收集证据
evidence = collector.collect_evidence(
    iter_num=1,
    action_content="click button",
    click_coords=(100, 200),
)

# 创建 EM 管理器
em_manager = EMManager(params_path="appeval/data/em_params.json")

# 添加证据
em_manager.add_evidence(
    gui_evidence=1,
    code_evidence=0,
)

# 获取预测
prediction = em_manager.predict()

# 进行纠偏
result = em_manager.correct_judgment(agent_original=0)
```
"""

# Evidence 数据结构
from appeval.em.evidence import Evidence

# 在线证据收集器
from appeval.em.collector import OnlineEvidenceCollector

# 自动标注工具
from appeval.em.annotator import AutoAnnotationTool

# EM 模型核心
from appeval.em.model import SimpleEM4EvidenceH_Refine

# EM 管理器
from appeval.em.manager import EMManager, create_em_manager

# 数据处理函数
from appeval.em.data_process import (
    convert_evidences_to_em_format,
    process_osagent_evidences,
    load_gui_evidence_from_jsonl,
    load_code_evidence_from_jsonl,
    get_code_evidence_dict_from_jsonl,
    merge_evidences_for_em,
    evidences_to_dataframe,
    load_evidences_from_jsonl,
    get_gui_evidence,
    get_gui_evidence_from_evidence,
)

# 分析函数
from appeval.em.analysis import (
    analyze_flips,
    confusion_matrix,
    correct_agent_judgment,
    calculate_metrics,
    compare_before_after_correction,
    analyze_error_cases,
    generate_correction_report,
)

__all__ = [
    # Evidence 数据结构
    "Evidence",
    # 在线证据收集器
    "OnlineEvidenceCollector",
    # 自动标注工具
    "AutoAnnotationTool",
    # EM 模型核心
    "SimpleEM4EvidenceH_Refine",
    # EM 管理器
    "EMManager",
    "create_em_manager",
    # 数据处理函数
    "convert_evidences_to_em_format",
    "process_osagent_evidences",
    "load_gui_evidence_from_jsonl",
    "load_code_evidence_from_jsonl",
    "get_code_evidence_dict_from_jsonl",
    "merge_evidences_for_em",
    "evidences_to_dataframe",
    "load_evidences_from_jsonl",
    "get_gui_evidence",
    "get_gui_evidence_from_evidence",
    # 分析函数
    "analyze_flips",
    "confusion_matrix",
    "correct_agent_judgment",
    "calculate_metrics",
    "compare_before_after_correction",
    "analyze_error_cases",
    "generate_correction_report",
]
