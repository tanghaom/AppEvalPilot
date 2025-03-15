#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : screen_info_extractor.py
@Desc    : Action for extracting key information from screen screenshots
"""

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.utils.common import encode_image


class ScreenInfoExtractor(Action):
    name: str = "ScreenInfoExtractor"
    desc: str = "Action for extracting key information from screen screenshots"

    # Base prompt template
    prompt_template: str = """
{task_content}

### Response requirements ###
You are an information extractor. {extractor_requirement}

Respond with a single paragraph in plain text, without any special formatting or headers."""

    def __init__(self, platform: str):
        super().__init__()
        self.platform = platform

        # Extractor requirement templates
        self.extractor_requirement_template = {
            "with_task": "First summarize the core content of the current page, then describe details related to the task. Use only page information without inferences.",
            "no_task": "Summarize the core content of the current page using only observable information.",
        }

    def get_extractor_prompt(self, task_content: str) -> str:
        """Generate prompt for information extraction

        Args:
            task_content (str): Content/task description to focus on

        Returns:
            str: Generated prompt
        """
        task_content = ("### task\n" + task_content) if task_content else ""
        extractor_requirement = self.extractor_requirement_template["with_task" if task_content else "no_task"]

        return self.prompt_template.format(task_content=task_content, extractor_requirement=extractor_requirement)

    async def run(self, task_content: str, screenshot_file: str) -> str:
        """Execute information extraction task

        Args:
            task_content (str): Content/task description to focus on
            screenshot_file (str): Path to screenshot file

        Returns:
            str: Extracted important content
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

        return output.strip()
