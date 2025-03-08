#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : reflection.py
@Desc    : 用于对操作结果进行反思的Action
"""
import asyncio
from typing import Any, Dict, List, Tuple

from metagpt.actions.action import Action
from metagpt.const import TEST_DATA_PATH
from metagpt.logs import logger
from metagpt.utils.common import encode_image


class Reflection(Action):
    name: str = "Reflection"
    desc: str = "用于对操作结果进行反思的Action"

    # 添加提示词模板作为类变量
    REFLECTION_PROMPT_TEMPLATE: str = """
These images are two {platform} screenshots before and after an operation. Their widths are {width} pixels and their heights are {height} pixels.

### Before the current operation ###
Screenshot information (format - coordinates; content):
{before_info}

### After the current operation ###
Screenshot information (format - coordinates; content):
{after_info}

### Operation Context ###
User instruction: {instruction}
{additional_info}
Operation thought: {operation_thought}
Operation action: {action}

### Analysis Requirements ###
Compare the screenshots before and after the operation. Focus on:
1. Whether the screen content changed as expected based on the operation thought
2. Whether the operation action produced the intended result
3. Whether any unexpected changes occurred

### Output format ###
### Thought ###
Your analysis of the changes between the two screenshots and whether they match the expected outcome.

### Answer ###
CORRECT: If the operation produced the expected result
ERROR: If the operation failed or produced unexpected results
"""

    def __init__(self, platform="Android"):
        super().__init__()
        self.platform = platform  # 平台类型：Android/PC

    def get_reflection_prompt(
        self,
        instruction: str,
        last_perception_infos: List[Dict[str, Any]],
        perception_infos: List[Dict[str, Any]],
        width: int,
        height: int,
        summary: str,
        action: str,
        add_info: str,
    ) -> str:
        """生成反思提示词"""
        # 处理感知信息
        before_info = "\n".join(
            f"{info['coordinates']}; {info['text']}"
            for info in last_perception_infos
            if self._is_valid_perception(info)
        )

        after_info = "\n".join(
            f"{info['coordinates']}; {info['text']}" for info in perception_infos if self._is_valid_perception(info)
        )

        # 处理额外信息
        additional_info = f"You also need to note the following requirements: {add_info}." if add_info else ""

        # 处理操作思考
        operation_thought = summary.split(" to ")[0].strip()

        # 填充模板
        return self.REFLECTION_PROMPT_TEMPLATE.format(
            platform=self.platform,
            width=width,
            height=height,
            before_info=before_info,
            after_info=after_info,
            instruction=instruction,
            additional_info=additional_info,
            operation_thought=operation_thought,
            action=action,
        )

    def _is_valid_perception(self, perception: Dict[str, Any]) -> bool:
        """检查感知信息是否有效"""
        return perception["text"] != "" and perception["text"] != "icon: None" and perception["coordinates"] != (0, 0)

    async def run(
        self,
        instruction: str,
        last_perception_infos: List[Dict[str, Any]],
        perception_infos: List[Dict[str, Any]],
        width: int,
        height: int,
        summary: str,
        action: str,
        add_info: str,
        last_screenshot: str,
        current_screenshot: str,
    ) -> Tuple[str, str]:
        """执行反思任务

        Args:
            instruction (str): 用户指令
            last_perception_infos (List[Dict]): 上一次的感知信息
            perception_infos (List[Dict]): 当前的感知信息
            width (int): 屏幕宽度
            height (int): 屏幕高度
            summary (str): 操作总结
            action (str): 执行的动作
            add_info (str): 额外信息
            last_screenshot (str): 上一次截图路径
            current_screenshot (str): 当前截图路径

        Returns:
            Tuple[str, str]: (反思结果, 反思思考)
        """
        prompt = self.get_reflection_prompt(
            instruction, last_perception_infos, perception_infos, width, height, summary, action, add_info
        )
        logger.info(
            f"\n\n######################## reflection_prompt:\n{prompt}\n\n######################## reflection_prompt end\n\n\n\n"
        )

        output = await self.llm.aask(
            prompt,
            system_msgs=[
                f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant."
            ],
            images=[encode_image(last_screenshot), encode_image(current_screenshot)],
            stream=False,
        )

        logger.info(
            f"\n\n######################## reflection_output:\n{output}\n\n######################## reflection_output end\n\n\n\n"
        )

        reflection_thought = output.split("### Thought ###")[-1].split("### Answer ###")[0].replace("\n", " ").strip()
        reflect = output.split("### Answer ###")[-1].replace("\n", " ").strip()

        # 验证输出结果是否符合预期
        if reflect.strip().upper().startswith("CORRECT"):
            reflect = "CORRECT"
        elif reflect.strip().upper().startswith("ERROR"):
            reflect = "ERROR"
        else:
            logger.warning(f"Unexpected reflection result: {reflect}, defaulting to ERROR")
            reflect = "ERROR"

        return reflect, reflection_thought


if __name__ == "__main__":

    async def main():
        from PIL import Image

        # 创建Reflection实例
        reflection_action = Reflection()

        # 测试参数 - 伪造缺失的参数
        instruction = "打开微信"
        last_perception_infos = [{"coordinates": (10, 10), "text": "上一次的感知信息"}]
        perception_infos = [{"coordinates": (20, 20), "text": "当前的感知信息"}]
        summary = "打开微信"
        action = "点击微信图标"
        add_info = ""

        # 指定截图路径
        last_screenshot_path = str(TEST_DATA_PATH / "screenshots" / "android.jpg")  # 替换为实际的截图路径
        current_screenshot_path = str(TEST_DATA_PATH / "screenshots" / "android.jpg")  # 替换为实际的截图路径

        # 利用PIL读取当前截图获取宽高
        try:
            with Image.open(current_screenshot_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"读取截图尺寸失败: {str(e)}，将使用默认尺寸 1080x1920")
            width, height = 1080, 1920

        try:
            # 执行反思
            reflect, reflection_thought = await reflection_action.run(
                instruction,
                last_perception_infos,
                perception_infos,
                width,
                height,
                summary,
                action,
                add_info,
                last_screenshot_path,
                current_screenshot_path,
            )
            print("\n反思结果:")
            print(reflect)
            print("\n反思思考:")
            print(reflection_thought)
        except Exception as e:
            print(f"执行失败: {str(e)}")

    asyncio.run(main())
