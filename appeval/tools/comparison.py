#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@File    : comparison.py
@Desc    : Screenshot comparison tool class, provides LLM-based screenshot comparison functionality
"""

from pathlib import Path

import yaml
from metagpt.config2 import Config
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.utils.common import encode_image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from appeval.prompts.comparison import COMPARISON_PROMPT_TEMPLATE


class ComparisonTool:
    """Screenshot comparison tool class"""

    def __init__(self, config_path: str = "config/config2.yaml"):
        """Initialize comparison tool

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        # Load configuration
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file).get("comparison")
            self.config = Config.from_llm_config(config)
        logger.info(f"ComparisonTool Config: {self.config}")
        self.llm = LLM(self.config.llm)

    async def compare_screenshots(
        self,
        before_screenshot: str,
        after_screenshot: str,
        instruction: str = "",
        operation_thought: str = "",
        action: str = "",
    ) -> str:
        """Compare two screenshots and analyze changes using LLM

        Args:
            before_screenshot: Path to the screenshot taken before the operation
            before_screenshot: Path to the screenshot taken after the operation
            instruction: User instruction
            operation_thought: Operation thought
            action: Operation action

        Returns:
            str: Analysis result of the changes
        """
        try:
            # Check if screenshots exist
            before_path = Path(before_screenshot)
            after_path = Path(after_screenshot)

            if not before_path.exists():
                logger.warning(
                    f"Before screenshot not found: {before_screenshot}")
                return "No previous screenshot available for comparison."

            if not after_path.exists():
                logger.warning(
                    f"After screenshot not found: {after_screenshot}")
                return "No current screenshot available for comparison."

            # Build prompt
            prompt = COMPARISON_PROMPT_TEMPLATE.format(
                instruction=instruction,
                operation_thought=operation_thought,
                action=action,
            )

            # Encode images
            images = [
                encode_image(str(before_path)),
                encode_image(str(after_path)),
            ]

            # Call LLM for analysis
            result = await self.llm.aask(
                prompt,
                system_msgs=[
                    "You are a helpful AI assistant specialized in analyzing UI changes between screenshots."],
                images=images,
                stream=False,
            )

            return result.strip()

        except Exception as e:
            logger.error(
                f"Error occurred during screenshot comparison: {str(e)}")
            return f"Comparison analysis failed: {str(e)}"
