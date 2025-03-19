#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : reflection.py
@Desc    : Action for reflecting on operation results
"""
from typing import Any, Dict, List, Tuple

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.utils.common import encode_image


class Reflection(Action):
    name: str = "Reflection"
    desc: str = "Action for reflecting on operation results"

    # Add prompt template as class variable
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
        self.platform = platform  # Platform type: Android/PC

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
        """Generate reflection prompt

        Args:
            instruction: User instruction
            last_perception_infos: Previous perception information
            perception_infos: Current perception information
            width: Screen width
            height: Screen height
            summary: Operation summary
            action: Executed action
            add_info: Additional information

        Returns:
            str: Reflection prompt
        """
        # Process perception information
        before_info = "\n".join(
            f"{info['coordinates']}; {info['text']}"
            for info in last_perception_infos
            if self._is_valid_perception(info)
        )

        after_info = "\n".join(
            f"{info['coordinates']}; {info['text']}" for info in perception_infos if self._is_valid_perception(info)
        )

        # Process additional information
        additional_info = f"You also need to note the following requirements: {add_info}." if add_info else ""

        # Fill template
        return self.REFLECTION_PROMPT_TEMPLATE.format(
            platform=self.platform,
            width=width,
            height=height,
            before_info=before_info,
            after_info=after_info,
            instruction=instruction,
            additional_info=additional_info,
            operation_thought=summary,
            action=action,
        )

    def _is_valid_perception(self, perception: Dict[str, Any]) -> bool:
        """Check if perception information is valid"""
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
        """Execute reflection task

        Args:
            instruction (str): User instruction
            last_perception_infos (List[Dict]): Previous perception information
            perception_infos (List[Dict]): Current perception information
            width (int): Screen width
            height (int): Screen height
            summary (str): Operation summary
            action (str): Executed action
            add_info (str): Additional information
            last_screenshot (str): Previous screenshot path
            current_screenshot (str): Current screenshot path

        Returns:
            Tuple[str, str]: (Reflection result, Reflection thought)
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
        reflect = output.split("### Answer ###")[-1].strip()

        # Validate if output matches expectations
        if "CORRECT" in reflect.upper():
            reflect = "CORRECT"
        elif "ERROR" in reflect.upper():
            reflect = "ERROR"
        else:
            logger.warning(f"Unexpected reflection result: {reflect}, defaulting to ERROR")
            reflect = "ERROR"

        return reflect, reflection_thought
