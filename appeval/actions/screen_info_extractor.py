#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : screen_info_extractor.py
@Desc    : 用于从屏幕截图中提取关键信息的Action
"""
import asyncio

from metagpt.actions.action import Action
from metagpt.const import TEST_DATA_PATH
from metagpt.logs import logger
from metagpt.utils.common import encode_image


class ScreenInfoExtractor(Action):
    name: str = "ScreenInfoExtractor"
    desc: str = "用于从屏幕截图中提取关键信息的Action"

    # 基础提示模板
    prompt_template: str = """
{task_content}

### Response requirements ###
You are an information extractor. {extractor_requirement}

Respond with a single paragraph in plain text, without any special formatting or headers."""

    def __init__(self, platform: str):
        super().__init__()
        self.platform = platform

        # 提取器要求模板
        self.extractor_requirement_template = {
            "with_task": "First summarize the core content of the current page, then describe details related to the task. Use only page information without inferences.",
            "no_task": "Summarize the core content of the current page using only observable information.",
        }

    def get_extractor_prompt(self, task_content: str) -> str:
        """生成提取信息的提示词

        Args:
            task_content (str): 需要关注的内容/任务描述

        Returns:
            str: 生成的提示词
        """
        task_content = ("### task\n" + task_content) if task_content else ""
        extractor_requirement = self.extractor_requirement_template["with_task" if task_content else "no_task"]

        return self.prompt_template.format(task_content=task_content, extractor_requirement=extractor_requirement)

    async def run(self, task_content: str, screenshot_file: str) -> str:
        """执行信息提取任务

        Args:
            task_content (str): 需要关注的内容/任务描述
            screenshot_file (str): 截图文件路径

        Returns:
            str: 提取的重要内容
        """
        prompt = self.get_extractor_prompt(task_content)
        logger.info(
            f"\n\n######################## extractor_prompt:\n{prompt}\n\n######################## extractor_prompt end\n\n\n\n"
        )

        output = await self.llm.aask(
            prompt,
            system_msgs=[
                f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant."
            ],
            images=[encode_image(screenshot_file)],
            stream=False,
        )

        logger.info(
            f"\n\n######################## extractor_output:\n{output}\n\n######################## extractor_output end\n\n\n\n"
        )

        # 直接返回输出，因为现在输出已经是格式化好的单段文本
        return output.strip()


if __name__ == "__main__":

    async def main():
        # 创建ScreenInfoExtractorAction实例
        extractor = ScreenInfoExtractor(platform="Android")

        # 测试参数
        screenshot_path = str(TEST_DATA_PATH / "screenshots" / "android.jpg")
        task_content = "请总结界面上显示的主要内容."

        try:
            # 执行信息提取
            result = await extractor.run(task_content, screenshot_path)
            print("\n提取结果:")
            print(result)

        except Exception as e:
            print(f"执行失败: {str(e)}")

    asyncio.run(main())
