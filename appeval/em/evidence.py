"""
Evidence 数据结构模块

定义单步操作的证据数据结构，用于在线学习和 EM 模型。
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Evidence:
    """单步操作的证据数据结构，用于在线学习"""

    iter_num: int  # 迭代编号
    timestamp: float = field(default_factory=time.time)  # 时间戳

    # 操作信息
    action_content: Optional[str] = None  # 执行的Action内容
    operation_desc: Optional[str] = None  # 操作描述
    click_coords: Optional[Tuple[int, int]] = None  # 点击坐标

    # 思考与反思
    thought: Optional[str] = None  # 当前思考内容
    reflection_thought: Optional[str] = None  # 反思内容

    # 坐标匹配分析结果
    coordinate_match: Optional[int] = None  # 坐标匹配结果：1=命中, 0=未命中, None=无分析
    matched_element: Optional[Dict] = None  # 匹配到的元素信息
    element_distance_sorting: Optional[List[Dict]] = None  # 元素距离排序

    # 元素树信息
    ui_elements: Optional[List[Dict]] = None  # 当前屏幕的UI元素

    # 图片路径
    screenshot_path: Optional[str] = None  # 原始截图路径
    annotated_screenshot_path: Optional[str] = None  # 带标注的截图路径

    # 任务信息
    task_list: Optional[str] = None  # 当前任务列表
    instruction: Optional[str] = None  # 用户指令

    # 执行结果
    error_flag: bool = False  # 是否发生错误
    error_message: Optional[str] = None  # 错误信息

    # Tell 动作相关字段（用于 EM 模型）
    tell_evidence: Optional[str] = None  # 从 Tell 动作中提取的 evidence
    agent_noresp: Optional[int] = None  # agent 无响应/异常判断：1=有问题, 0=正常, None=未分析

    def to_dict(self) -> Dict:
        """转换为字典，用于序列化"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict) -> "Evidence":
        """从字典创建Evidence实例"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Evidence":
        """从JSON字符串创建Evidence实例"""
        data = json.loads(json_str)
        return cls.from_dict(data)
