#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/14
@File    : task_validator.py
@Desc    : Validator for checking if agent truly completed the task
"""
import json
import re
from typing import Any, Dict, List

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.utils.common import encode_image


# Compact prompt template for task validation
TASK_VALIDATOR_PROMPT = """Task: {instruction}

Actions: {action_summary}

Screenshots show task execution (oldest→newest). Is task completed?

Output JSON only (keep brief):
{{"completed": true/false, "reason": "why", "suggestion": "how to fix if failed"}}"""


class TaskValidator(Action):
    """Validator to check if agent truly completed the task using Claude 4"""

    name: str = "TaskValidator"
    desc: str = "Validates task completion before accepting Tell action"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_action_summary(self, summary_history: List[str], max_steps: int = 5) -> str:
        """Build a compact summary of recent actions.

        Args:
            summary_history: List of operation summaries
            max_steps: Maximum number of recent steps to include

        Returns:
            Compact string summary of recent actions
        """
        if not summary_history:
            return "None"

        # Get the last N steps, truncate each
        recent = summary_history[-max_steps:]
        lines = []
        for i, s in enumerate(recent, 1):
            short = s.replace("\n", " ").strip()[:80]
            lines.append(f"{i}.{short}")

        return "; ".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract completed, reason and suggestion.

        Args:
            response: Raw LLM response

        Returns:
            Dict with 'completed' (bool), 'reason' (str), 'suggestion' (str)
        """
        default_result = {
            "completed": False,
            "reason": "Parse failed",
            "suggestion": "",
        }

        # Try to extract JSON from response
        try:
            result = json.loads(response.strip())
            return {
                "completed": bool(result.get("completed", False)),
                "reason": str(result.get("reason", ""))[:200],
                "suggestion": str(result.get("suggestion", ""))[:200],
            }
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        json_match = re.search(r"\{[^{}]*\}", response)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return {
                    "completed": bool(result.get("completed", False)),
                    "reason": str(result.get("reason", ""))[:200],
                    "suggestion": str(result.get("suggestion", ""))[:200],
                }
            except json.JSONDecodeError:
                pass

        # Fallback
        return default_result

    async def run(
        self,
        instruction: str,
        screenshot_paths: List[str],
        summary_history: List[str],
        max_steps: int = 5,
    ) -> Dict[str, Any]:
        """Validate if the task is truly completed.

        Args:
            instruction: Original task instruction
            screenshot_paths: List of screenshot paths (oldest to newest)
            summary_history: List of operation summaries
            max_steps: Max recent steps for action summary

        Returns:
            Dict with 'completed' (bool), 'reason' (str), 'suggestion' (str)
        """
        # Build compact action summary
        action_summary = self._build_action_summary(summary_history, max_steps)

        # Build prompt
        prompt = TASK_VALIDATOR_PROMPT.format(
            instruction=instruction,
            action_summary=action_summary,
        )

        logger.info(
            f"\n######################## TaskValidator prompt:\n{prompt}\n########################\n")
        logger.info(f"TaskValidator using {len(screenshot_paths)} screenshots")

        # Encode all screenshots
        images = []
        for path in screenshot_paths:
            try:
                images.append(encode_image(path))
            except Exception as e:
                logger.warning(f"Failed to encode screenshot {path}: {e}")

        if not images:
            return {"completed": True, "reason": "No screenshots", "suggestion": ""}

        # Call LLM
        try:
            response = await self.llm.aask(
                prompt,
                system_msgs=["Task validator. Output brief JSON only."],
                images=images,
                stream=False,
            )

            logger.info(
                f"\n######################## TaskValidator response:\n{response}\n########################\n")

            result = self._parse_response(response)
            logger.info(
                f"TaskValidator: completed={result['completed']}, reason={result['reason'][:50]}...")

            return result

        except Exception as e:
            logger.error(f"TaskValidator failed: {str(e)}")
            return {"completed": True, "reason": f"Error: {str(e)}", "suggestion": ""}
