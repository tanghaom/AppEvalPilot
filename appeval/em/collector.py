"""
在线证据收集器模块

用于实时收集 OS Agent 每步操作的证据。
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from appeval.em.evidence import Evidence


class OnlineEvidenceCollector:
    """在线证据收集器，用于实时收集每步操作的证据"""

    # 用于判断 agent_noresp 的 prompt 模板
    AGENT_NORESP_PROMPT = """You are a "precise GUI test adjudicator."

Task:
Decide if an action succeeded based only on the Reflection text below. If the Reflection is empty, whitespace-only, or contains no meaningful information, the outcome is failure.

Input:
{evidence}

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
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
            ui_elements = self._convert_perception_to_elements(
                perception_infos)
            evidence.ui_elements = ui_elements

            # 如果有点击坐标，进行坐标分析
            if click_coords and self.enable_coordinate_analysis:
                self._analyze_coordinate(
                    evidence, click_coords, ui_elements, screenshot_path, operation_desc or action_content or "")

        # 处理 Tell 动作：提取 evidence 内容
        if action_content and self.enable_tell_analysis:
            tell_evidence = self._extract_evidence_from_tell(action_content)
            if tell_evidence:
                evidence.tell_evidence = tell_evidence
                self.logger.info(
                    f"检测到 Tell 动作，提取到 evidence: {tell_evidence[:100]}...")

        # 存储证据
        self.evidences.append(evidence)

        self.logger.info(
            f"收集到第 {iter_num} 步证据, 坐标匹配: {evidence.coordinate_match}")

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
                evidence = action_content.split('"evidence":')[
                    1] if '"evidence":' in action_content else action_content.split('"evidence": ')[1]
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
            prompt = self.AGENT_NORESP_PROMPT.format(
                evidence=evidence.tell_evidence)

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
                bbox = [center_x - 20, center_y -
                        10, center_x + 20, center_y + 10]
            elif len(coords) == 4:
                # bbox格式 [x1, y1, x2, y2]
                bbox = coords
            else:
                continue

            # 从rect字段提取精确bbox
            if "rect:" in text:
                rect_match = re.search(
                    r"rect:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", text)
                if rect_match:
                    bbox = [int(rect_match.group(1)), int(rect_match.group(2)), int(
                        rect_match.group(3)), int(rect_match.group(4))]

            element = {"id": f"element_{idx}", "ui_name": ui_name,
                       "control_type": control_type, "bbox": bbox, "raw_text": text}
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

            element_info = {**element, "distance": distance,
                            "area": area, "is_inside": is_inside}
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
            clean_project_name = re.sub(
                r'[<>:"/\\|?*]', "_", self.project_name)

            status = "success" if evidence.coordinate_match == 1 else "fail"
            output_filename = f"{clean_project_name}_iter_{evidence.iter_num}_coords_{coords[0]}_{coords[1]}_{status}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)

            with Image.open(screenshot_path) as img:
                draw = ImageDraw.Draw(img)

                # 绘制红点
                x, y = coords
                radius = 4
                draw.ellipse([x - radius, y - radius, x + radius,
                             y + radius], fill="red", outline="darkred", width=2)

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
        coord_analyzed = sum(
            1 for e in self.evidences if e.coordinate_match is not None)
        coord_matched = sum(
            1 for e in self.evidences if e.coordinate_match == 1)
        errors = sum(1 for e in self.evidences if e.error_flag)

        return {
            "total_steps": total,
            "coordinate_analyzed": coord_analyzed,
            "coordinate_matched": coord_matched,
            "coordinate_accuracy": coord_matched / coord_analyzed if coord_analyzed > 0 else 0,
            "error_count": errors,
        }
