#!/usr/bin/env python3

import base64
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from metagpt.utils.common import encode_image
from PIL import Image, ImageDraw


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


class OnlineEvidenceCollector:
    """在线证据收集器，用于实时收集每步操作的证据"""

    # 用于判断 agent_noresp 的 prompt 模板
    AGENT_NORESP_PROMPT = """You are a "precise GUI test adjudicator."

Task:
Decide if an action succeeded based only on the Reflection text below. If the Reflection is empty, whitespace-only, or contains no meaningful information, the outcome is failure.

Input:
{reflection_thought}

Decision rules (strict):
- Success (output Yes): The Reflection clearly and explicitly states that the intended UI change or functional effect occurred, e.g.:
  - "Search results updated/refreshed"
  - "Modal opened/closed"
  - "Button became enabled/disabled"
  - "Status 200 and content refreshed/list updated"
  - "Navigated to target page / expected element is visible"
- Failure (output No): Any of the following:
  - Page unresponsive, stuck, or loading never completes (e.g., "no change," "spinner keeps spinning," "timeout," "no results returned")
  - Result not as expected (e.g., "navigated to wrong page," "content differs from expectation," "button still disabled," "filter did not apply," "still old data/same page")
  - Errors/exceptions/stack traces/not implemented (e.g., "4xx/5xx error," "threw exception," "not implemented")
  - Only browser default behavior is triggered (e.g., "right-click shows browser menu only")
  - Ambiguous, contradictory, or unverifiable statements (be conservative: treat as failure)
- Negative examples (always failure): "No visible change," "not sure if it worked," "might have succeeded," "seems updated but no new data," "loading not finished."

Output format:
Return exactly one line of JSON with no extra keys:

```json
{{ "result" : "Yes" }}
```

or

```json
{{ "result" : "No" }}
```}}
"""

    def __init__(
        self,
        output_dir: str = "evidence",
        enable_coordinate_analysis: bool = True,
        llm=None,  # 直接使用OSAgent的llm实例
        fallback_to_mllm: bool = False,  # 默认不使用MLLM回退，加快处理速度
        project_name: str = "default",
        enable_tell_analysis: bool = True,  # 是否启用 Tell 动作分析
    ):
        """
        初始化在线证据收集器

        Args:
            output_dir: 证据输出目录
            enable_coordinate_analysis: 是否启用坐标分析
            llm: LLM实例，直接使用OSAgent的llm
            fallback_to_mllm: 当元素树匹配失败时是否回退到MLLM
            project_name: 项目名称
            enable_tell_analysis: 是否启用 Tell 动作分析
        """
        self.output_dir = output_dir
        self.enable_coordinate_analysis = enable_coordinate_analysis
        self.llm = llm
        self.fallback_to_mllm = fallback_to_mllm
        self.project_name = project_name
        self.enable_tell_analysis = enable_tell_analysis

        # 存储收集的证据
        self.evidences: List[Evidence] = []

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 配置日志
        self.logger = logging.getLogger(f"{__name__}.OnlineCollector")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 记录LLM状态
        if self.llm:
            self.logger.info("在线证据收集器: 使用OSAgent的LLM实例")
        else:
            self.logger.info("在线证据收集器: 使用简单分析模式")

    def collect_evidence(
        self,
        iter_num: int,
        action_content: str = None,
        operation_desc: str = None,
        click_coords: Tuple[int, int] = None,
        thought: str = None,
        reflection_thought: str = None,
        perception_infos: List[Dict] = None,
        screenshot_path: str = None,
        task_list: str = None,
        instruction: str = None,
        error_flag: bool = False,
        error_message: str = None,
        width: int = 0,
        height: int = 0,
    ) -> Evidence:
        """
        收集单步操作的证据

        Args:
            iter_num: 迭代编号
            action_content: 执行的Action内容
            operation_desc: 操作描述
            click_coords: 点击坐标
            thought: 当前思考内容
            reflection_thought: 反思内容
            perception_infos: 感知信息（元素树）
            screenshot_path: 截图路径
            task_list: 当前任务列表
            instruction: 用户指令
            error_flag: 是否发生错误
            error_message: 错误信息
            width: 屏幕宽度
            height: 屏幕高度

        Returns:
            Evidence: 收集的证据对象
        """
        evidence = Evidence(
            iter_num=iter_num,
            action_content=action_content,
            operation_desc=operation_desc,
            click_coords=click_coords,
            thought=thought,
            reflection_thought=reflection_thought,
            task_list=task_list,
            instruction=instruction,
            error_flag=error_flag,
            error_message=error_message,
            screenshot_path=screenshot_path,
        )

        # 处理UI元素信息
        if perception_infos:
            ui_elements = self._convert_perception_to_elements(perception_infos)
            evidence.ui_elements = ui_elements

            # 如果有点击坐标，进行坐标分析
            if click_coords and self.enable_coordinate_analysis:
                self._analyze_coordinate(evidence, click_coords, ui_elements, screenshot_path, operation_desc or action_content or "")

        # 处理 Tell 动作：提取 evidence 内容
        if action_content and self.enable_tell_analysis:
            tell_evidence = self._extract_evidence_from_tell(action_content)
            if tell_evidence:
                evidence.tell_evidence = tell_evidence
                self.logger.info(f"检测到 Tell 动作，提取到 evidence: {tell_evidence[:100]}...")

        # 存储证据
        self.evidences.append(evidence)

        self.logger.info(f"收集到第 {iter_num} 步证据, 坐标匹配: {evidence.coordinate_match}")

        return evidence

    def _extract_evidence_from_tell(self, action_content: str) -> Optional[str]:
        """
        从 Tell ({"0": {"result": "Pass", "evidence": "..."}}) 格式中提取 evidence

        Args:
            action_content: 动作内容

        Returns:
            提取的 evidence 字符串，如果不是 Tell 动作则返回 None
        """
        if action_content is None or "Tell (" not in action_content:
            return None

        try:
            # 定位 Tell ( 后面的 JSON 对象
            if '"evidence":' in action_content or '"evidence": ' in action_content:
                # 尝试提取 evidence 字段的值
                evidence = action_content.split('"evidence":')[1] if '"evidence":' in action_content else action_content.split('"evidence": ')[1]
                # 清理提取的内容
                evidence = evidence.strip()
                if evidence.startswith('"'):
                    # 找到匹配的结束引号
                    end_idx = evidence.find('"', 1)
                    if end_idx != -1:
                        evidence = evidence[1:end_idx]
                return evidence
            return None
        except Exception as e:
            self.logger.error(f"提取 Tell evidence 失败: {e}")
            return None

    async def analyze_tell_action(self, evidence: Evidence) -> Optional[int]:
        """
        分析 Tell 动作，判断 agent 是否给出了实质性响应

        Args:
            evidence: 证据对象，需要包含 tell_evidence 字段

        Returns:
            agent_noresp 结果：1=有问题（无响应/异常），0=正常，None=无法分析
        """
        if not evidence.tell_evidence:
            return None

        if not self.llm:
            self.logger.warning("LLM 未配置，无法分析 Tell 动作")
            return None

        try:
            prompt = self.AGENT_NORESP_PROMPT.format(evidence=evidence.tell_evidence)

            response = await self.llm.aask(
                prompt,
                stream=False,
            )

            # 解析响应
            result = self._parse_agent_noresp_response(response)
            evidence.agent_noresp = result

            self.logger.info(f"Tell 动作分析完成: agent_noresp={result}")
            return result

        except Exception as e:
            self.logger.error(f"分析 Tell 动作失败: {e}")
            return None

    def _parse_agent_noresp_response(self, response: str) -> int:
        """
        解析 LLM 对 agent_noresp 的判断响应

        Args:
            response: LLM 的响应

        Returns:
            1 表示有问题，0 表示正常
        """
        # 去除 markdown 代码块标记
        result = response.strip()
        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        try:
            import json

            parsed = json.loads(result)
            if parsed.get("result") == "Yes":
                return 1  # 有问题
            else:
                return 0  # 正常
        except:
            # 如果解析失败，尝试简单匹配
            if "Yes" in response:
                return 1
            return 0

    def _convert_perception_to_elements(self, perception_infos: List[Dict]) -> List[Dict]:
        """将感知信息转换为元素列表格式"""
        elements = []
        for idx, info in enumerate(perception_infos):
            coords = info.get("coordinates", [])
            text = info.get("text", "")

            # 解析控件类型
            control_type = "Unknown"
            if "control_type:" in text:
                match = re.search(r"control_type:(\w+)", text)
                if match:
                    control_type = match.group(1)
            elif "icon" in text.lower():
                control_type = "Icon"
            elif "text:" in text:
                control_type = "Text"

            # 获取UI名称
            ui_name = ""
            if "text:" in text:
                match = re.search(r"text:([^;]+)", text)
                if match:
                    ui_name = match.group(1).strip()

            # 计算bbox
            if len(coords) == 2:
                # 中心点格式 [x, y]，需要估算bbox
                center_x, center_y = coords
                # 估算一个默认大小
                bbox = [center_x - 20, center_y - 10, center_x + 20, center_y + 10]
            elif len(coords) == 4:
                # bbox格式 [x1, y1, x2, y2]
                bbox = coords
            else:
                continue

            # 从rect字段提取精确bbox
            if "rect:" in text:
                rect_match = re.search(r"rect:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", text)
                if rect_match:
                    bbox = [int(rect_match.group(1)), int(rect_match.group(2)), int(rect_match.group(3)), int(rect_match.group(4))]

            element = {"id": f"element_{idx}", "ui_name": ui_name, "control_type": control_type, "bbox": bbox, "raw_text": text}
            elements.append(element)

        return elements

    def _analyze_coordinate(self, evidence: Evidence, coords: Tuple[int, int], elements: List[Dict], screenshot_path: str, description: str):
        """分析坐标是否命中UI元素"""
        x, y = coords

        # 计算所有元素的距离并排序
        elements_with_distance = []
        matched_elements = []

        for element in elements:
            bbox = element.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            area = (x2 - x1) * (y2 - y1)
            is_inside = x1 <= x <= x2 and y1 <= y <= y2

            element_info = {**element, "distance": distance, "area": area, "is_inside": is_inside}
            elements_with_distance.append(element_info)

            if is_inside:
                matched_elements.append((element_info, distance, area))

        # 按距离排序
        elements_with_distance.sort(key=lambda e: (e["distance"], e["area"]))
        # 只保存前10个
        evidence.element_distance_sorting = elements_with_distance[:10]

        # 确定匹配结果
        if matched_elements:
            # 选择距离中心最近且面积最小的元素
            matched_elements.sort(key=lambda x: (x[1], x[2]))
            best_match = matched_elements[0][0]
            evidence.coordinate_match = 1
            evidence.matched_element = best_match
        else:
            evidence.coordinate_match = 0
            evidence.matched_element = None

        # 生成带标注的截图
        if screenshot_path and os.path.exists(screenshot_path):
            self._draw_annotated_screenshot(evidence, screenshot_path, coords)

    def _draw_annotated_screenshot(self, evidence: Evidence, screenshot_path: str, coords: Tuple[int, int]):
        """在截图上绘制标注"""
        try:
            clean_project_name = re.sub(r'[<>:"/\\|?*]', "_", self.project_name)

            status = "success" if evidence.coordinate_match == 1 else "fail"
            output_filename = f"{clean_project_name}_iter_{evidence.iter_num}_coords_{coords[0]}_{coords[1]}_{status}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)

            with Image.open(screenshot_path) as img:
                draw = ImageDraw.Draw(img)

                # 绘制红点
                x, y = coords
                radius = 4
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red", outline="darkred", width=2)

                # 如果匹配到元素，绘制元素边框
                if evidence.matched_element:
                    bbox = evidence.matched_element.get("bbox", [])
                    if len(bbox) == 4:
                        draw.rectangle(bbox, outline="green", width=2)

                img.save(output_path)
                evidence.annotated_screenshot_path = output_path

        except Exception as e:
            self.logger.error(f"绘制标注截图失败: {e}")

    def get_all_evidences(self) -> List[Evidence]:
        """获取所有收集的证据"""
        return self.evidences

    def get_evidences_for_em(self) -> List[Dict]:
        """获取适合传递给EM模型的证据格式"""
        return [e.to_dict() for e in self.evidences]

    def get_latest_evidence(self) -> Optional[Evidence]:
        """获取最新的证据"""
        return self.evidences[-1] if self.evidences else None

    def save_evidences(self, output_file: str = None):
        """保存所有证据到JSONL文件"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "evidences.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for evidence in self.evidences:
                f.write(evidence.to_json() + "\n")

        self.logger.info(f"证据已保存到: {output_file}")

    def clear_evidences(self):
        """清空证据列表"""
        self.evidences = []

    def get_summary_stats(self) -> Dict:
        """获取证据统计摘要"""
        total = len(self.evidences)
        coord_analyzed = sum(1 for e in self.evidences if e.coordinate_match is not None)
        coord_matched = sum(1 for e in self.evidences if e.coordinate_match == 1)
        errors = sum(1 for e in self.evidences if e.error_flag)

        return {
            "total_steps": total,
            "coordinate_analyzed": coord_analyzed,
            "coordinate_matched": coord_matched,
            "coordinate_accuracy": coord_matched / coord_analyzed if coord_analyzed > 0 else 0,
            "error_count": errors,
        }


class AutoAnnotationTool:
    """自动标注工具主类"""

    def __init__(
        self,
        info_file: str,
        material_dir: str,
        llm=None,
        project_name: str = None,
        results_dir: str = None,
        enable_coordinate_analysis: bool = True,
        fallback_to_mllm: bool = True,
        coordinate_analysis_condition: str = "error_only",
    ):
        """
        初始化标注工具

        Args:
            info_file: info.txt 文件路径
            material_dir: material 目录路径，包含 origin_*.jpg 图片
            llm: LLM实例
            project_name: 项目名称，用于图片命名
            results_dir: 统一的结果文件夹路径，如果提供则使用统一保存
            enable_coordinate_analysis: 是否启用坐标分析，默认True
            fallback_to_mllm: 当元素树匹配失败时是否回退到MLLM分析，默认True
            coordinate_analysis_condition: 坐标分析条件，"error_only"或"all_clicks"，默认"error_only"
        """
        self.info_file = info_file
        self.material_dir = material_dir
        self.llm = llm
        self.project_name = project_name or "Unknown"
        self.enable_coordinate_analysis = enable_coordinate_analysis
        self.fallback_to_mllm = fallback_to_mllm
        self.coordinate_analysis_condition = coordinate_analysis_condition
        self.results = []

        # 创建坐标分析图片输出目录
        if results_dir:
            # 使用统一的结果文件夹
            self.coordinate_analysis_dir = results_dir
        else:
            # 使用项目内的文件夹（向后兼容）
            self.coordinate_analysis_dir = os.path.join(material_dir, "coordinate_analysis")

        os.makedirs(self.coordinate_analysis_dir, exist_ok=True)

        # 配置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 创建日志处理器（如果还没有的话）
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 记录LLM状态
        if self.llm:
            self.logger.info("MLLM已配置（使用与OSAgent相同的LLM实例）")
        else:
            self.logger.info("使用简单分析模式")

        # 记录坐标分析开关状态
        if self.enable_coordinate_analysis:
            self.logger.info("坐标分析已启用")
        else:
            self.logger.info("坐标分析已禁用")

    def parse_info_file(self) -> List[Dict]:
        """解析 info.txt 文件，按 iter 切块，并从最后一个iter中提取History operations"""
        with open(self.info_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 按 iter 切块
        iter_pattern = r"#### iter:(\d+)"
        iter_matches = list(re.finditer(iter_pattern, content))

        iter_blocks = []
        for i, match in enumerate(iter_matches):
            iter_num = int(match.group(1))
            start_pos = match.start()

            # 找到下一个 iter 的位置，或者文件结尾
            if i + 1 < len(iter_matches):
                end_pos = iter_matches[i + 1].start()
            else:
                end_pos = len(content)

            iter_content = content[start_pos:end_pos]

            # 如果是最后一个iter，提取History operations内容
            history_operations_content = ""
            if i == len(iter_matches) - 1:  # 最后一个iter
                history_operations_content = self.extract_history_operations_content(content)

            iter_blocks.append({"iter_num": iter_num, "content": iter_content, "history_operations": history_operations_content})

        return iter_blocks

    def extract_history_operations_content(self, content: str) -> str:
        """提取最后一个iter中History operations和Last Task List之间的内容"""
        # 查找最后一个iter中的History operations
        history_pattern = r"### History operations ###(.*?)(?=### Last Task List ###|$)"
        history_matches = list(re.finditer(history_pattern, content, re.DOTALL))

        if not history_matches:
            self.logger.warning("未找到History operations内容")
            return ""

        # 取最后一个匹配（最后一个iter中的）
        last_history_match = history_matches[-1]
        history_content = last_history_match.group(1).strip()

        self.logger.info(f"提取到History operations内容，长度: {len(history_content)} 字符")
        return history_content

    def extract_coordinates_from_history(self, history_content: str) -> List[Tuple[int, int, str]]:
        """从History operations中提取所有坐标信息"""
        coordinates = []

        # 查找所有Step中的Action部分
        step_pattern = r"Step-(\d+):(.*?)(?=Step-\d+:|$)"
        step_matches = re.finditer(step_pattern, history_content, re.DOTALL)

        for match in step_matches:
            step_num = match.group(1)
            step_content = match.group(2)

            # 查找Action部分
            action_pattern = r"Action:\s*(.*?)(?=Reflection_thought:|$)"
            action_match = re.search(action_pattern, step_content, re.DOTALL)

            if action_match:
                action_text = action_match.group(1).strip()

                # 查找pyautogui.click坐标
                click_pattern = r"pyautogui\.click\((\d+),\s*(\d+)\)"
                click_match = re.search(click_pattern, action_text)

                if click_match:
                    x = int(click_match.group(1))
                    y = int(click_match.group(2))
                    coordinates.append((x, y, f"Step-{step_num}"))

        self.logger.info(f"从History operations中提取到 {len(coordinates)} 个坐标")
        return coordinates

    def extract_descriptions_from_history(self, history_content: str) -> List[Tuple[str, str]]:
        """从History operations中提取所有操作描述"""
        descriptions = []

        # 查找所有Step中的Operation部分
        step_pattern = r"Step-(\d+):(.*?)(?=Step-\d+:|$)"
        step_matches = re.finditer(step_pattern, history_content, re.DOTALL)

        for match in step_matches:
            step_num = match.group(1)
            step_content = match.group(2)

            # 查找Operation部分
            operation_pattern = r"Operation:\s*(.*?)(?=Action:|$)"
            operation_match = re.search(operation_pattern, step_content, re.DOTALL)

            if operation_match:
                operation_text = operation_match.group(1).strip()
                descriptions.append((operation_text, f"Step-{step_num}"))

        self.logger.info(f"从History operations中提取到 {len(descriptions)} 个操作描述")
        return descriptions

    def extract_action_contents_from_history(self, history_content: str) -> List[Tuple[str, str]]:
        """从History operations中提取所有Action内容"""
        action_contents = []

        # 查找所有Step中的Action部分
        step_pattern = r"Step-(\d+):(.*?)(?=Step-\d+:|$)"
        step_matches = re.finditer(step_pattern, history_content, re.DOTALL)

        for match in step_matches:
            step_num = match.group(1)
            step_content = match.group(2)

            # 查找Action部分
            action_pattern = r"Action:\s*(.*?)(?=Reflection_thought:|$)"
            action_match = re.search(action_pattern, step_content, re.DOTALL)

            if action_match:
                action_text = action_match.group(1).strip()
                action_contents.append((action_text, f"Step-{step_num}"))

        self.logger.info(f"从History operations中提取到 {len(action_contents)} 个Action内容")
        return action_contents

    def extract_reflection_thoughts_from_history(self, history_content: str) -> List[Tuple[str, str]]:
        """从History operations中提取所有Reflection_thought内容"""
        reflection_thoughts = []

        # 查找所有Step中的Reflection_thought部分
        step_pattern = r"Step-(\d+):(.*?)(?=Step-\d+:|$)"
        step_matches = re.finditer(step_pattern, history_content, re.DOTALL)

        for match in step_matches:
            step_num = match.group(1)
            step_content = match.group(2)

            # 查找Reflection_thought部分 - 更精确的匹配，确保在Memory之前停止
            reflection_pattern = r"Reflection_thought:\s*(.*?)(?=\n\tMemory:|\nMemory:|$)"
            reflection_match = re.search(reflection_pattern, step_content, re.DOTALL)

            if reflection_match:
                reflection_text = reflection_match.group(1).strip()

                # 进一步清理：如果文本中包含Memory:，则截取到Memory:之前
                if "Memory:" in reflection_text:
                    memory_index = reflection_text.find("Memory:")
                    reflection_text = reflection_text[:memory_index].strip()

                # 清理末尾的换行符和多余空白
                reflection_text = reflection_text.rstrip("\n\t ")

                if reflection_text:  # 只有当文本不为空时才添加
                    reflection_thoughts.append((reflection_text, f"Step-{step_num}"))

        self.logger.info(f"从History operations中提取到 {len(reflection_thoughts)} 个Reflection_thought")
        return reflection_thoughts

    def extract_all_history_data(self, history_content: str) -> Dict:
        """提取所有历史操作数据"""
        return {
            "coordinates": self.extract_coordinates_from_history(history_content),
            "descriptions": self.extract_descriptions_from_history(history_content),
            "action_contents": self.extract_action_contents_from_history(history_content),
            "reflection_thoughts": self.extract_reflection_thoughts_from_history(history_content),
        }

    def extract_element_tree_from_iter(self, iter_content: str) -> List[Dict]:
        """
        从iter内容中提取元素树信息

        Args:
            iter_content: iter块的内容

        Returns:
            元素树列表，每个元素包含id, ui_name, bbox, control_type等信息
        """
        elements = []

        # 查找元素树信息块
        element_tree_pattern = r"### Screenshot information ###(.*?)### Hints ###"
        element_tree_match = re.search(element_tree_pattern, iter_content, re.DOTALL)

        if not element_tree_match:
            self.logger.debug("未找到元素树信息")
            return elements

        element_tree_content = element_tree_match.group(1)

        # 提取"The information is as follow:" 和 "Please note that this information is not necessarily accurate" 之间的内容
        info_pattern = r"The information is as follow:(.*?)Please note that this information is not necessarily accurate"
        info_match = re.search(info_pattern, element_tree_content, re.DOTALL)

        if not info_match:
            self.logger.debug("未找到元素信息内容")
            return elements

        info_content = info_match.group(1).strip()

        # 解析每一行元素信息
        for idx, line in enumerate(info_content.split("\n")):
            line = line.strip()
            if not line:
                continue

            # 解析元素信息格式: (x, y); text:content; control_type:type; rect: (x1, y1, x2, y2)
            element_match = re.match(r"\((\d+),\s*(\d+)\);\s*text:(.*?);\s*control_type:(.*?);\s*rect:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", line)

            if element_match:
                int(element_match.group(1))
                int(element_match.group(2))
                ui_name = element_match.group(3).strip()
                control_type = element_match.group(4).strip()
                rect_x1 = int(element_match.group(5))
                rect_y1 = int(element_match.group(6))
                rect_x2 = int(element_match.group(7))
                rect_y2 = int(element_match.group(8))

                # bbox就是rect坐标 [x1, y1, x2, y2]
                bbox = [rect_x1, rect_y1, rect_x2, rect_y2]

                element = {"id": f"element_{idx}", "ui_name": ui_name, "control_type": control_type, "bbox": bbox}  # 添加唯一id

                elements.append(element)

        self.logger.debug(f"从iter中提取到 {len(elements)} 个元素")
        return elements

    def deduplicate_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        去重元素，相同bbox只保留有ui_name的，并过滤掉Image类型元素和包含其他元素的容器元素

        Args:
            elements: 元素列表

        Returns:
            去重后的元素列表
        """
        # 使用bbox作为key进行去重
        bbox_to_element = {}

        for element in elements:
            bbox = element["bbox"]
            control_type = element["control_type"]

            # 过滤掉Image类型的元素
            if control_type == "Image":
                self.logger.debug(f"过滤掉Image类型元素: {element['ui_name'] or '无名称'} ({control_type})")
                continue

            bbox_key = tuple(bbox)

            # 如果bbox已存在，优先保留有ui_name的元素
            if bbox_key in bbox_to_element:
                existing = bbox_to_element[bbox_key]
                if not existing["ui_name"] and element["ui_name"]:
                    bbox_to_element[bbox_key] = element
                # 如果都有ui_name或都没有，保留第一个
            else:
                bbox_to_element[bbox_key] = element

        deduplicated = list(bbox_to_element.values())
        self.logger.debug(f"去重后剩余 {len(deduplicated)} 个元素")

        # 过滤包含其他元素的容器元素
        filtered_elements = self._filter_container_elements(deduplicated)
        self.logger.debug(f"过滤容器元素后剩余 {len(filtered_elements)} 个元素")

        return filtered_elements

    def _filter_container_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        过滤掉包含其他元素的容器元素和非功能元素

        Args:
            elements: 元素列表

        Returns:
            过滤后的元素列表
        """
        if len(elements) <= 1:
            return elements

        # 按面积排序，从大到小
        elements_with_area = []
        for element in elements:
            bbox = element["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            elements_with_area.append((element, area))

        elements_with_area.sort(key=lambda x: x[1], reverse=True)

        filtered_elements = []

        for i, (element_a, area_a) in enumerate(elements_with_area):
            bbox_a = element_a["bbox"]
            is_container = False

            # 首先检查是否是容器类型，直接过滤掉
            if element_a["control_type"] in ["Pane", "Window", "Document"]:
                is_container = True
                self.logger.debug(f"过滤掉容器类型元素: {element_a['ui_name'] or '无名称'} ({element_a['control_type']})")
            else:
                # 检查当前元素是否包含其他元素
                for j, (element_b, area_b) in enumerate(elements_with_area):
                    if i == j:  # 跳过自己
                        continue

                    bbox_b = element_b["bbox"]

                    # 检查元素A是否包含元素B
                    if self._bbox_contains(bbox_a, bbox_b):
                        # 元素A包含元素B，检查是否应该过滤掉元素A
                        if self._should_filter_container(element_a, element_b):
                            is_container = True
                            self.logger.debug(
                                f"过滤掉容器元素: {element_a['ui_name'] or '无名称'} ({element_a['control_type']}) 包含 {element_b['ui_name'] or '无名称'} ({element_b['control_type']})"
                            )
                            break

            if not is_container:
                filtered_elements.append(element_a)

        return filtered_elements

    def _bbox_contains(self, bbox_a: List[int], bbox_b: List[int]) -> bool:
        """
        检查bbox_a是否包含bbox_b

        Args:
            bbox_a: 容器元素的bbox [x1, y1, x2, y2]
            bbox_b: 被包含元素的bbox [x1, y1, x2, y2]

        Returns:
            是否包含
        """
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b

        # 检查bbox_b是否完全在bbox_a内部
        return x1_a <= x1_b and y1_a <= y1_b and x2_a >= x2_b and y2_a >= y2_b

    def _should_filter_container(self, container_element: Dict, contained_element: Dict) -> bool:
        """
        判断是否应该过滤掉容器元素

        Args:
            container_element: 容器元素
            contained_element: 被包含的元素

        Returns:
            是否应该过滤
        """
        container_type = container_element["control_type"]
        contained_type = contained_element["control_type"]

        # 如果容器是Pane、Window等容器类型，直接过滤掉
        if container_type in ["Pane", "Window", "Document"]:
            return True

        # 如果容器元素没有ui_name，更可能是容器
        if not container_element["ui_name"]:
            return True

        # 如果被包含元素有ui_name且容器元素没有，过滤容器
        if contained_element["ui_name"] and not container_element["ui_name"]:
            return True

        # 如果被包含元素是Button、Edit等功能元素，且容器不是功能元素，过滤容器
        if contained_type in ["Button", "Edit", "Text"] and container_type not in ["Button", "Edit", "Text"]:
            return True

        return False

    def calculate_element_distances(self, coordinates: Tuple[int, int], elements: List[Dict]) -> List[Dict]:
        """
        计算所有元素到指定坐标的距离，并按距离从近到远排序

        Args:
            coordinates: 要检查的坐标 (x, y)
            elements: 元素列表

        Returns:
            按距离排序的元素列表，每个元素包含距离信息
        """
        x, y = coordinates
        elements_with_distance = []

        for element in elements:
            bbox = element["bbox"]
            x1, y1, x2, y2 = bbox

            # 计算元素中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 计算坐标到元素中心的距离
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            # 计算元素面积
            area = (x2 - x1) * (y2 - y1)

            # 检查坐标是否在元素范围内
            is_inside = x1 <= x <= x2 and y1 <= y <= y2

            element_with_distance = {
                "id": element["id"],
                "ui_name": element["ui_name"],
                "control_type": element["control_type"],
                "bbox": element["bbox"],
                "distance": distance,
                "area": area,
                "is_inside": is_inside,
            }

            elements_with_distance.append(element_with_distance)

        # 按距离排序，距离相同时按面积排序（优先选择面积更小的元素）
        elements_with_distance.sort(key=lambda x: (x["distance"], x["area"]))

        return elements_with_distance

    def check_coordinate_in_elements(self, coordinates: Tuple[int, int], elements: List[Dict]) -> Dict:
        """
        检查坐标是否在元素范围内，当命中多个元素时选择最接近元素中心的那个

        Args:
            coordinates: 要检查的坐标 (x, y)
            elements: 元素列表

        Returns:
            匹配结果字典
        """
        x, y = coordinates
        matched_elements = []

        # 找到所有匹配的元素
        for element in elements:
            bbox = element["bbox"]
            x1, y1, x2, y2 = bbox

            # 检查坐标是否在元素范围内
            if x1 <= x <= x2 and y1 <= y <= y2:
                # 计算元素中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 计算坐标到元素中心的距离
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

                # 计算元素面积，用于距离相同时的排序
                area = (x2 - x1) * (y2 - y1)

                matched_elements.append((element, distance, area))

        if not matched_elements:
            return {"matched": False, "element": None, "element_id": None, "ui_name": None, "control_type": None, "bbox": None}

        # 按距离排序，距离相同时按面积排序（优先选择面积更小的元素）
        matched_elements.sort(key=lambda x: (x[1], x[2]))
        best_element = matched_elements[0][0]
        best_distance = matched_elements[0][1]

        return {
            "matched": True,
            "element": best_element,
            "element_id": best_element["id"],  # 添加元素id
            "all_matches": len(matched_elements),  # 记录总匹配数
            "distance_to_center": best_distance,  # 记录到最佳元素中心的距离
        }

    def _map_step_to_iter(self, result: Dict, iter_num: int, history_data: Dict):
        """将History operations中的step信息映射到对应的iter"""
        # 查找当前iter对应的step信息
        step_id = f"Step-{iter_num}"

        # 查找对应的描述
        for desc_info in history_data["descriptions"]:
            if desc_info[1] == step_id:
                result["operation_desc"] = desc_info[0]
                break

        # 查找对应的Action内容
        for action_info in history_data["action_contents"]:
            if action_info[1] == step_id:
                result["action_content"] = action_info[0]
                break

        # 查找对应的Reflection_thought
        for reflection_info in history_data["reflection_thoughts"]:
            if reflection_info[1] == step_id:
                result["reflection_thought"] = reflection_info[0]
                break

        # 查找对应的坐标
        for coord_info in history_data["coordinates"]:
            if coord_info[2] == step_id:
                result["click_coords"] = (coord_info[0], coord_info[1])
                break

        # 调试信息：显示映射情况
        if result.get("operation_desc") or result.get("action_content") or result.get("reflection_thought") or result.get("click_coords"):
            self.logger.debug(
                f"Iter {iter_num} 映射到 {step_id}: desc={bool(result.get('operation_desc'))}, action={bool(result.get('action_content'))}, reflection={bool(result.get('reflection_thought'))}, coords={bool(result.get('click_coords'))}"
            )

    def extract_reflection_result(self, iter_content: str) -> Optional[int]:
        """提取反思结果标注"""
        # 查找 reflection_output 块
        reflection_pattern = r"######################## reflection_output:(.*?)######################## reflection_output end"
        reflection_match = re.search(reflection_pattern, iter_content, re.DOTALL)

        if not reflection_match:
            return None

        reflection_content = reflection_match.group(1)

        # 查找 Answer 部分
        answer_pattern = r"### Answer ###\s*(.*?)(?=\n\n|\Z)"
        answer_match = re.search(answer_pattern, reflection_content, re.DOTALL)

        if not answer_match:
            return None

        answer_content = answer_match.group(1).strip()

        # 判断结果
        if "CORRECT" in answer_content:
            return 1
        elif "ERROR" in answer_content:
            return 0
        else:
            return None

    def extract_click_coordinates(self, iter_content: str) -> Optional[Tuple[int, int]]:
        """提取点击坐标"""
        # 查找 output_action 块
        action_pattern = r"######################## output_action:(.*?)######################## output_action end"
        action_match = re.search(action_pattern, iter_content, re.DOTALL)

        if not action_match:
            return None

        action_content = action_match.group(1)

        # 查找 pyautogui.click 坐标
        click_pattern = r"pyautogui\.click\((\d+),\s*(\d+)\)"
        click_match = re.search(click_pattern, action_content)

        if not click_match:
            return None

        x = int(click_match.group(1))
        y = int(click_match.group(2))
        return (x, y)

    def extract_operation_description(self, iter_content: str) -> Optional[str]:
        """提取操作描述"""
        # 查找 output_action 块
        action_pattern = r"######################## output_action:(.*?)######################## output_action end"
        action_match = re.search(action_pattern, iter_content, re.DOTALL)

        if not action_match:
            return None

        action_content = action_match.group(1)

        # 查找 Operation 部分
        operation_pattern = r"### Operation ###\s*(.*?)(?=### Task List ###)"
        operation_match = re.search(operation_pattern, action_content, re.DOTALL)

        if not operation_match:
            return None

        return operation_match.group(1).strip()

    def extract_action_content(self, iter_content: str) -> Optional[str]:
        """提取Action块内容（### Action ### 和 ### Operation ### 之间的内容）"""
        # 查找 output_action 块
        action_pattern = r"######################## output_action:(.*?)######################## output_action end"
        action_match = re.search(action_pattern, iter_content, re.DOTALL)

        if not action_match:
            return None

        action_content = action_match.group(1)

        # 查找 Action 和 Operation 之间的内容
        action_operation_pattern = r"### Action ###\s*(.*?)(?=### Operation ###)"
        action_operation_match = re.search(action_operation_pattern, action_content, re.DOTALL)

        if not action_operation_match:
            return None

        return action_operation_match.group(1).strip()

    def extract_reflection_thought_from_iter(self, iter_content: str) -> Optional[str]:
        """
        从单个iter中提取Reflection Thought内容，过滤掉prompt模板

        支持两种格式：
        1. 旧格式 (baseline): "Analyze whether the screen content and state match..."
        2. 新格式 (experiment): "Write a comprehensive analysis of the last operation's outcome..."

        匹配格式：
        ### Reflection Thought ###
        [内容]
        ### Thought ###
        """
        # 查找 ### Reflection Thought ### 标记
        if "### Reflection Thought ###" not in iter_content:
            return None

        # 匹配 ### Reflection Thought ### 到下一个 ### Thought ### 或 ### Action ### 之间的内容
        # 使用更精确的模式，确保匹配到正确的结束标记
        reflection_thought_pattern = r"### Reflection Thought ###\s*\n(.*?)(?=\n### Thought ###|\n### Action ###|$)"
        reflection_thought_matches = re.finditer(reflection_thought_pattern, iter_content, re.DOTALL | re.MULTILINE)

        # 两种prompt模板的特征文本（应该被过滤）
        prompt_template_markers = [
            "Analyze whether the screen content and state match the expected outcome",  # 旧格式
            "Write a comprehensive analysis of the last operation's outcome",  # 新格式
            "What was expected vs. what actually happened",  # 新格式的特征
            "Success evaluation:",  # 新格式的特征
            "Failure analysis",  # 新格式的特征
        ]

        for match in reflection_thought_matches:
            reflection_thought_content = match.group(1).strip()

            # 检查是否是prompt模板文本（包含任何特征标记）
            is_prompt_template = any(marker in reflection_thought_content for marker in prompt_template_markers)

            if is_prompt_template:
                self.logger.debug("跳过prompt模板文本")
                continue

            # 清理内容，移除多余的空白和换行
            reflection_thought_content = re.sub(r"\n\s*\n+", "\n", reflection_thought_content)
            reflection_thought_content = reflection_thought_content.strip()

            # 返回第一个非模板的reflection内容
            if reflection_thought_content:
                self.logger.debug(f"提取到Reflection Thought内容，长度: {len(reflection_thought_content)}")
                return reflection_thought_content

        return None

    def draw_coordinate_on_image(self, image_path: str, coordinates: Tuple[int, int], output_path: str):
        """在图片上绘制红点标记坐标"""
        try:
            # 打开图片
            with Image.open(image_path) as img:
                # 创建绘图对象
                draw = ImageDraw.Draw(img)

                # 绘制红点（缩小尺寸）
                x, y = coordinates
                radius = 4  # 从15缩小到5
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red", outline="darkred", width=2)

                # 移除坐标文本显示
                # draw.text((x+20, y-10), f"({x},{y})", fill='red')

                # 保存图片
                img.save(output_path)
                self.logger.debug(f"已绘制坐标到: {output_path}")

        except Exception as e:
            self.logger.error(f"绘制坐标失败: {e}")

    def image_to_base64(self, image_path: str) -> str:
        """将图片转换为 base64 编码"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"图片编码失败: {e}")
            return ""

    async def analyze_coordinate_accuracy_with_mllm(self, image_path: str, description: str) -> Dict:
        """
        使用 MLLM 分析坐标是否命中目标元素，带重试机制和详细日志
        使用与OSAgent相同的LLM调用方式

        Args:
            image_path: 带红点标记的图片路径
            description: 操作描述

        Returns:
            Dict: 包含分析结果的详细信息
        """
        analysis_result = {
            "accuracy": 0,  # 1 (命中), 0 (未命中)
            "retry_count": 0,  # 重试次数
            "api_errors": [],  # API错误列表
            "raw_response": "",  # 原始响应
            "analysis_time": 0.0,  # 分析耗时
            "image_size": (0, 0),  # 图片尺寸
            "coordinates": (0, 0),  # 分析的坐标
        }

        if not self.llm:
            self.logger.warning("LLM未配置，使用模拟结果")
            analysis_result["accuracy"] = 1
            return analysis_result

        # 获取图片信息
        try:
            with Image.open(image_path) as img:
                analysis_result["image_size"] = img.size
        except Exception as e:
            self.logger.warning(f"无法获取图片尺寸: {e}")

        # 从文件名提取坐标
        filename = os.path.basename(image_path)
        coord_match = re.search(r"coords_(\d+)_(\d+)", filename)
        if coord_match:
            analysis_result["coordinates"] = (int(coord_match.group(1)), int(coord_match.group(2)))

        max_retries = 5
        start_time = time.time()

        self.logger.info(f"MLLM分析开始: {os.path.basename(image_path)}")
        self.logger.info(f"操作描述: {description[:100]}...")
        self.logger.info(f"图片尺寸: {analysis_result['image_size']}, 坐标: {analysis_result['coordinates']}")

        # 构建与OSAgent相同格式的prompt
        prompt = f"""请分析这张截图中的红点坐标是否命中了目标元素。

操作描述: {description}

请仔细观察红点中心位置，判断红点中心是否命中在了正确的目标元素上。

请只回答 1 或 0：
- 1: 红点中心命中了目标元素
- 0: 红点中心没有命中目标元素

注意：请严格按照要求只回答数字 1 或 0，不要包含其他文字。
"""

        # 使用与OSAgent相同的方式编码图片
        images = [encode_image(image_path)]

        for attempt in range(max_retries):
            analysis_result["retry_count"] = attempt + 1
            try:
                # 使用与OSAgent相同的LLM调用方式
                response = await self.llm.aask(
                    prompt,
                    system_msgs=["你是一个专业的UI分析助手，负责判断点击坐标是否命中目标元素。"],
                    images=images,
                    stream=False,
                )

                # 解析结果
                answer = response.strip()
                analysis_result["raw_response"] = answer
                self.logger.debug(f"MLLM 原始回答: '{answer}'")

                # 验证回答是否有效
                if self._is_valid_answer(answer):
                    result = self._extract_result(answer)
                    analysis_result["accuracy"] = result

                    analysis_result["analysis_time"] = time.time() - start_time

                    self.logger.info(f"MLLM分析完成: 结果={result}, 耗时={analysis_result['analysis_time']:.2f}s, 重试={attempt}次")
                    return analysis_result
                else:
                    error_msg = f"无效答案: '{answer}'"
                    analysis_result["api_errors"].append(error_msg)
                    self.logger.warning(f"无效答案，重试... (尝试 {attempt + 1}/{max_retries})")

            except Exception as e:
                error_msg = f"API请求失败: {str(e)}"
                analysis_result["api_errors"].append(error_msg)
                self.logger.warning(f"MLLM分析失败，重试: {e} (尝试 {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    self.logger.error("MLLM分析失败，使用默认值")
                    analysis_result["accuracy"] = 1  # 默认返回命中

        # 如果所有重试都失败，返回默认值
        analysis_result["analysis_time"] = time.time() - start_time
        analysis_result["accuracy"] = 1

        self.logger.error(f"MLLM分析最终失败，使用默认值。总耗时: {analysis_result['analysis_time']:.2f}s")
        return analysis_result

    def _is_valid_answer(self, answer: str) -> bool:
        """
        验证 MLLM 回答是否有效

        Args:
            answer: MLLM 的回答

        Returns:
            bool: 是否包含有效的 1 或 0
        """
        # 清理回答，移除空格和换行
        clean_answer = answer.strip().replace("\n", "").replace(" ", "")

        # 检查是否包含 1 或 0
        has_one = "1" in clean_answer
        has_zero = "0" in clean_answer

        # 如果只包含1或只包含0，则有效
        if has_one and not has_zero:
            return True
        elif has_zero and not has_one:
            return True
        else:
            return False

    def _extract_result(self, answer: str) -> int:
        """
        从 MLLM 回答中提取结果

        Args:
            answer: MLLM 的回答

        Returns:
            int: 1 (命中) 或 0 (未命中)
        """
        # 清理回答
        clean_answer = answer.strip().replace("\n", "").replace(" ", "")

        # 优先检查 1，如果包含 1 就返回 1
        if "1" in clean_answer:
            return 1
        # 否则返回 0
        else:
            return 0

    def analyze_coordinate_accuracy_simple(self, image_path: str, description: str) -> Dict:
        """
        简单的坐标分析（不使用 MLLM）

        Args:
            image_path: 带红点标记的图片路径
            description: 操作描述

        Returns:
            Dict: 包含分析结果的详细信息
        """
        analysis_result = {
            "accuracy": 1,  # 默认返回命中
            "retry_count": 0,
            "api_errors": [],
            "raw_response": "",
            "analysis_time": 0.0,
            "image_size": (0, 0),
            "coordinates": (0, 0),
        }

        # 获取图片信息
        try:
            with Image.open(image_path) as img:
                analysis_result["image_size"] = img.size
        except Exception as e:
            self.logger.warning(f"无法获取图片尺寸: {e}")

        # 从文件名提取坐标
        filename = os.path.basename(image_path)
        coord_match = re.search(r"coords_(\d+)_(\d+)", filename)
        if coord_match:
            analysis_result["coordinates"] = (int(coord_match.group(1)), int(coord_match.group(2)))

        self.logger.info(f"简单分析: {os.path.basename(image_path)}")
        self.logger.info(f"操作描述: {description[:100]}...")
        self.logger.info(f"图片尺寸: {analysis_result['image_size']}, 坐标: {analysis_result['coordinates']}")

        # 这里可以实现基于规则的简单分析
        # 比如检查红点是否在按钮区域内等

        # 暂时返回模拟结果
        return analysis_result

    async def process_single_iter(self, iter_block: Dict) -> Dict:
        """处理单个 iter 块，保持原有反思结果逻辑，其他信息从最后一个iter的History operations中提取"""
        iter_num = iter_block["iter_num"]
        content = iter_block["content"]
        iter_block.get("history_operations", "")

        result = {
            "iter_num": iter_num,
            "reflection": None,  # 反思结果：1=正确, 0=错误, None=无反思
            "click_coords": None,  # 点击坐标 (x, y)
            "operation_desc": None,  # 操作描述
            "action_content": None,  # Action内容
            "reflection_thought": None,  # 反思思考内容
            "coordinate_match": None,  # 坐标匹配结果：1=命中, 0=未命中, None=无分析
            "coordinate_analysis": None,  # 详细分析结果
            "element_distance_sorting": None,  # 元素距离排序（按距离从近到远）
        }

        # 1. 反思结果标注（保持原有逻辑，从每个iter中提取）
        reflection_result = self.extract_reflection_result(content)
        result["reflection"] = reflection_result

        # 2. 提取Action内容（保持原有逻辑，从每个iter中提取）
        action_content = self.extract_action_content(content)
        if action_content:
            result["action_content"] = action_content

        # 2.5. 提取Reflection Thought内容（从每个iter中提取）
        reflection_thought = self.extract_reflection_thought_from_iter(content)
        if reflection_thought:
            result["reflection_thought"] = reflection_thought

        # 3. 提取坐标和描述（保持原有逻辑，从每个iter中提取）
        coordinates = self.extract_click_coordinates(content)
        description = self.extract_operation_description(content)

        if coordinates and description:
            result["click_coords"] = coordinates
            result["operation_desc"] = description

        # 4. 提取元素树信息
        elements = self.extract_element_tree_from_iter(content)
        if elements:
            # 去重元素
            elements = self.deduplicate_elements(elements)

            # 如果有坐标，计算元素距离排序
            if result["click_coords"]:
                distance_sorted_elements = self.calculate_element_distances(result["click_coords"], elements)
                result["element_distance_sorting"] = distance_sorted_elements

        # 5. 根据条件决定是否进行坐标分析
        should_analyze = False
        if self.enable_coordinate_analysis and result["click_coords"] and result["operation_desc"]:
            if self.coordinate_analysis_condition == "all_clicks":
                # 对所有点击操作进行分析
                should_analyze = True
            elif self.coordinate_analysis_condition == "error_only" and reflection_result == 0:
                # 仅对错误情况进行分析
                should_analyze = True

        if should_analyze:
            await self._analyze_single_coordinate(result, iter_num, result["click_coords"], result["operation_desc"], elements)

        return result

    async def _analyze_single_coordinate(
        self, result: Dict, iter_num: int, coordinates: Tuple[int, int], description: str, elements: List[Dict] = None
    ):
        """分析单个坐标（新逻辑：基于元素树匹配）"""
        # 检查对应的图片文件
        image_path = os.path.join(self.material_dir, f"origin_{iter_num}.jpg")
        if os.path.exists(image_path):
            # 清理项目名，移除特殊字符，确保文件名合法
            clean_project_name = re.sub(r'[<>:"/\\|?*]', "_", self.project_name)

            # 使用新的基于元素树的匹配逻辑
            if elements:
                match_result = self.check_coordinate_in_elements(coordinates, elements)

                if match_result["matched"]:
                    # 坐标命中元素
                    accuracy = 1
                    element_id = match_result["element_id"]

                    # 绘制坐标并重命名为success
                    draw_filename = f"{clean_project_name}_iter_{iter_num}_coords_{coordinates[0]}_{coordinates[1]}_success.jpg"
                else:
                    # 坐标未命中任何元素
                    accuracy = 0
                    element_id = None

                    # 绘制坐标并重命名为fail
                    draw_filename = f"{clean_project_name}_iter_{iter_num}_coords_{coordinates[0]}_{coordinates[1]}_fail.jpg"

                draw_path = os.path.join(self.coordinate_analysis_dir, draw_filename)
                self.draw_coordinate_on_image(image_path, coordinates, draw_path)

                # 创建精简的分析结果
                analysis_details = {"accuracy": accuracy, "method": "element_tree", "matched_element_id": element_id}

            else:
                # 根据参数决定是否回退到MLLM逻辑
                if self.fallback_to_mllm:
                    # 回退到原有MLLM逻辑
                    draw_filename = f"{clean_project_name}_iter_{iter_num}_coords_{coordinates[0]}_{coordinates[1]}.jpg"
                    draw_path = os.path.join(self.coordinate_analysis_dir, draw_filename)
                    self.draw_coordinate_on_image(image_path, coordinates, draw_path)

                    if self.llm:
                        analysis_details = await self.analyze_coordinate_accuracy_with_mllm(draw_path, description)
                    else:
                        analysis_details = self.analyze_coordinate_accuracy_simple(draw_path, description)

                    # 根据MLLM结果重命名文件
                    if analysis_details["accuracy"] == 1:
                        success_filename = f"{clean_project_name}_iter_{iter_num}_coords_{coordinates[0]}_{coordinates[1]}_success.jpg"
                        success_path = os.path.join(self.coordinate_analysis_dir, success_filename)
                        os.rename(draw_path, success_path)
                        self.logger.info(f"文件已重命名为: {success_filename}")
                    else:
                        fail_filename = f"{clean_project_name}_iter_{iter_num}_coords_{coordinates[0]}_{coordinates[1]}_fail.jpg"
                        fail_path = os.path.join(self.coordinate_analysis_dir, fail_filename)
                        os.rename(draw_path, fail_path)
                        self.logger.info(f"文件已重命名为: {fail_filename}")
                else:
                    # 不回退到MLLM，直接标记为失败
                    self.logger.info("元素树解析失败且禁用MLLM回退，标记为失败")
                    draw_filename = f"{clean_project_name}_iter_{iter_num}_coords_{coordinates[0]}_{coordinates[1]}_fail.jpg"
                    draw_path = os.path.join(self.coordinate_analysis_dir, draw_filename)
                    self.draw_coordinate_on_image(image_path, coordinates, draw_path)

                    analysis_details = {"accuracy": 0, "method": "element_tree_failed", "matched_element_id": None}

            # 保存详细分析结果
            result["coordinate_analysis"] = analysis_details
            result["coordinate_match"] = analysis_details["accuracy"]

            # 详细日志记录
            self.logger.info(f"Iter {iter_num}: 坐标分析完成")
            self.logger.info(f"  结果: {analysis_details['accuracy']} ({'命中' if analysis_details['accuracy'] == 1 else '未命中'})")
            if "analysis_time" in analysis_details:
                self.logger.info(f"  分析耗时: {analysis_details['analysis_time']:.2f}s")
            if analysis_details.get("api_errors"):
                self.logger.warning(f"  API错误: {len(analysis_details['api_errors'])} 个")
                for error in analysis_details["api_errors"]:
                    self.logger.warning(f"    - {error}")
        else:
            self.logger.warning(f"Iter {iter_num}: 未找到对应的图片文件 {image_path}")
            result["coordinate_analysis"] = {"accuracy": 0, "method": "no_image", "matched_element_id": None}

    async def run_annotation(self) -> List[Dict]:
        """运行完整的标注流程"""
        iter_blocks = self.parse_info_file()
        self.logger.info(f"解析完成: {len(iter_blocks)} 个 iter")

        # 提取History operations数据（从最后一个iter中）
        history_data = None
        for iter_block in iter_blocks:
            if iter_block.get("history_operations"):
                history_data = self.extract_all_history_data(iter_block["history_operations"])
                break

        results = []
        for iter_block in iter_blocks:
            self.logger.debug(f"处理 iter {iter_block['iter_num']}")
            result = await self.process_single_iter(iter_block)

            # 如果有History operations数据，进行映射
            if history_data:
                self._map_step_to_iter(result, iter_block["iter_num"], history_data)

            results.append(result)

        self.results = results
        return results

    def save_to_jsonl(self, output_file: str):
        """保存结果到 JSONL 文件"""
        with open(output_file, "w", encoding="utf-8") as f:
            for result in self.results:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

        self.logger.info(f"结果已保存到: {output_file}")

    def get_coordinate_analysis_summary(self) -> Dict:
        """获取坐标分析图片的简化统计信息"""
        total_coordinate_cases = 0
        successfully_analyzed = 0
        element_tree_cases = 0
        element_tree_matched = 0
        mllm_cases = 0

        for r in self.results:
            # 统计有坐标的案例
            if r.get("click_coords"):
                total_coordinate_cases += 1

            # 统计成功分析的案例
            if r.get("coordinate_analysis"):
                if isinstance(r["coordinate_analysis"], list):
                    successfully_analyzed += len(r["coordinate_analysis"])
                else:
                    successfully_analyzed += 1

                    # 统计分析方法
                    method = r["coordinate_analysis"].get("method", "unknown")
                    if method == "element_tree":
                        element_tree_cases += 1
                        if r["coordinate_analysis"].get("accuracy") == 1:
                            element_tree_matched += 1
                    elif method == "element_tree_failed":
                        element_tree_cases += 1
                        # element_tree_failed 不算作匹配成功
                    else:
                        mllm_cases += 1

            # 统计有距离排序的案例（替代ui_elements统计）
            if r.get("element_distance_sorting"):
                element_tree_cases += 1

        summary = {
            "total_coordinate_cases": total_coordinate_cases,
            "successfully_analyzed": successfully_analyzed,
            "coordinate_analysis_dir": self.coordinate_analysis_dir,
            "coordinate_files_created": 0,
            "element_tree_cases": element_tree_cases,
            "element_tree_matched": element_tree_matched,
            "mllm_cases": mllm_cases,
        }

        # 统计实际创建的坐标分析图片文件数量
        if os.path.exists(self.coordinate_analysis_dir):
            coordinate_files = [f for f in os.listdir(self.coordinate_analysis_dir) if f.endswith(".jpg")]
            summary["coordinate_files_created"] = len(coordinate_files)

        return summary

    def print_summary(self):
        """打印详细的处理摘要"""
        total = len(self.results)
        reflection_found = sum(1 for r in self.results if r["reflection"] is not None)
        error_cases = sum(1 for r in self.results if r["reflection"] == 0)
        coordinate_analyzed = sum(1 for r in self.results if r["coordinate_match"] is not None)
        action_found = sum(1 for r in self.results if r["action_content"] is not None)
        reflection_thought_found = sum(1 for r in self.results if r["reflection_thought"] is not None)
        element_tree_found = sum(1 for r in self.results if r.get("element_distance_sorting"))

        print("\n=== 处理摘要 ===")
        print(f"总 iter 数: {total}")
        print(f"找到反思结果: {reflection_found}")
        print(f"ERROR 案例: {error_cases}")
        print(f"坐标分析案例: {coordinate_analyzed}")
        print(f"找到Action内容: {action_found}")
        print(f"找到Reflection_thought: {reflection_thought_found}")
        print(f"找到元素树: {element_tree_found}")

        # 显示简化的坐标分析统计信息
        coord_summary = self.get_coordinate_analysis_summary()
        if coord_summary["total_coordinate_cases"] > 0:
            print("\n=== 坐标分析统计 ===")
            print(f"坐标分析文件夹: {coord_summary['coordinate_analysis_dir']}")
            print(f"创建的图片文件: {coord_summary['coordinate_files_created']} 个")
            print(f"成功分析案例: {coord_summary['successfully_analyzed']}/{coord_summary['total_coordinate_cases']}")
            print(f"元素树匹配案例: {coord_summary['element_tree_matched']}/{coord_summary['element_tree_cases']}")
            print(f"MLLM分析案例: {coord_summary['mllm_cases']}")

        # 显示有坐标的案例详情
        coordinate_cases = [r for r in self.results if r.get("click_coords")]
        if coordinate_cases:
            print("\n=== 坐标案例详情 ===")
            for case in coordinate_cases:
                print(f"Iter {case['iter_num']}: 坐标 {case['click_coords']}")
                if case.get("operation_desc"):
                    print(f"  描述: {case['operation_desc'][:50]}...")
                if case.get("action_content"):
                    print(f"  Action: {case['action_content'][:50]}...")
                if case.get("reflection_thought"):
                    print(f"  Reflection_thought: {case['reflection_thought'][:50]}...")

                # 显示距离排序信息
                if case.get("element_distance_sorting"):
                    print(f"  距离排序: {len(case['element_distance_sorting'])} 个元素")

                # 显示坐标分析结果
                if case.get("coordinate_analysis"):
                    details = case["coordinate_analysis"]
                    if isinstance(details, list):
                        for analysis in details:
                            step_id = analysis.get("step_id", "Unknown")
                            accuracy = analysis.get("accuracy", "N/A")
                            analysis_time = analysis.get("analysis_time", 0.0)
                            method = analysis.get("method", "unknown")

                            print(f"  {step_id}: 结果={accuracy} ({'命中' if accuracy == 1 else '未命中' if accuracy == 0 else 'N/A'})")
                            print(f"    方法: {method}")
                            print(f"    耗时: {analysis_time:.2f}s")
                            if analysis.get("api_errors"):
                                print(f"    API错误: {len(analysis['api_errors'])} 个")
                    else:
                        # 向后兼容旧格式
                        accuracy = details.get("accuracy", "N/A")
                        analysis_time = details.get("analysis_time", 0.0)
                        method = details.get("method", "unknown")

                        print(f"  分析结果: {accuracy} ({'命中' if accuracy == 1 else '未命中' if accuracy == 0 else 'N/A'})")
                        print(f"  方法: {method}")
                        print(f"  耗时: {analysis_time:.2f}s")
                        if details.get("api_errors"):
                            print(f"  API错误: {len(details['api_errors'])} 个")

                        # 显示元素树匹配详情
                        if method == "element_tree" and details.get("match_result"):
                            match_result = details["match_result"]
                            if match_result["matched"]:
                                element = match_result["element"]
                                print(f"  匹配元素: {element['ui_name'] or '无名称'} ({element['control_type']})")
                                print(f"  元素bbox: {element['bbox']}")
                            else:
                                print("  未匹配任何元素")
                print()
