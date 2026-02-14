#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/14
@File    : text_agent.py
@Desc    : Text-only OS Agent that uses accessibility tree / DOM tree instead of screenshots.
           Inherits from OSAgent and overrides image-dependent methods.
"""
import copy
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.schema import AIMessage

from appeval.prompts.osagent import ActionPromptContext
from appeval.prompts.text_agent import TextPrompt, text_agent_system_prompt
from appeval.roles.osagent import OSAgent


class TextAgent(OSAgent):
    """Text-only agent that operates via accessibility tree / DOM tree.

    Key differences from OSAgent:
    - No screenshots sent to LLM (pure text prompt)
    - Perception uses only a11y tree / CDP DOM tree
    - No OCR, no icon detection, no SOM
    - No TellVerifier (it depends on screenshots)
    - Can use cheaper text-only LLM models (no VLM needed)
    - Optional debug screenshots saved to disk but never sent to LLM
    """

    def __init__(self, **kwargs):
        # Force disable all image-dependent features
        kwargs["use_ocr"] = False
        kwargs["use_icon_detect"] = False
        kwargs["use_icon_caption"] = False
        kwargs["use_som"] = False
        kwargs["use_tell_verifier"] = False
        kwargs["use_reflection"] = False
        kwargs.setdefault("use_chrome_debugger", False)
        kwargs.setdefault("think_history_images", 0)

        # Extract debug_screenshots before super().__init__ (pydantic doesn't allow early setattr)
        debug_screenshots = kwargs.pop("debug_screenshots", True)

        # Default system prompt for text-only testing
        if not kwargs.get("system_prompt"):
            kwargs["system_prompt"] = text_agent_system_prompt

        super().__init__(**kwargs)

        # Set after pydantic init completes
        self._debug_screenshots = debug_screenshots

        # Replace prompt utils with text-only version
        self.prompt_utils = TextPrompt()

        logger.info(f"TextAgent initialized: a11y_mode={self.a11y_mode}, "
                     f"debug_screenshots={self._debug_screenshots}")

    # ================================================================
    # Override: Perception — only a11y tree, no screenshot/OCR/icon
    # ================================================================

    async def _get_perception_infos(
        self, screenshot_file: str, screenshot_som_file: str
    ) -> Tuple[List[Dict[str, Any]], int, int, str]:
        """Get perception info from accessibility tree only.

        No screenshots are taken for LLM reasoning.
        Optionally captures a screenshot for debug logging.

        Returns:
            tuple: (perception_infos, width, height, output_image_path)
        """
        # Optional: take screenshot for debug/human review only
        if self._debug_screenshots:
            try:
                self.controller.get_screenshot(screenshot_file)
            except Exception as e:
                logger.debug(f"Debug screenshot failed (non-critical): {e}")

        # Core: get structured elements from a11y tree / CDP
        perception_infos = []
        if self.extend_xml_infos and self.platform in ["Android", "Windows", "Linux"]:
            perception_infos = self.controller.get_screen_xml(self.location_info)

        # Determine screen size
        width, height = 1920, 1080  # Default
        if self._debug_screenshots:
            try:
                from PIL import Image
                img = Image.open(screenshot_file)
                width, height = img.size
            except Exception:
                pass

        logger.info(f"TextAgent perception: {len(perception_infos)} elements from a11y tree")

        # output_image_path is empty — no annotated image
        return perception_infos, width, height, ""

    # ================================================================
    # Override: Think — text-only LLM call, no images
    # ================================================================

    async def _think(self) -> bool:
        """Generate operation decisions using text-only LLM (no images)."""
        add_info = self.add_info

        # Build action prompt context (same as parent)
        ctx = ActionPromptContext(
            instruction=self.instruction,
            clickable_infos=self.rc.perception_infos,
            width=self.width,
            height=self.height,
            thought_history=self.rc.thought_history,
            summary_history=self.rc.summary_history,
            action_history=self.rc.action_history,
            reflection_thought_history=self.rc.reflection_thought_history,
            last_summary=self.rc.summary,
            last_action=self.rc.action,
            reflection_thought=self.rc.reflection_thought,
            add_info=add_info,
            error_flag=self.rc.error_flag,
            error_message=self.rc.error_message,
            completed_content=self.rc.completed_requirements,
            memory=self.rc.memory,
            task_list=self.rc.task_list,
            use_som=False,
            location_info=self.location_info,
            is_first_step=(self.rc.iter == 1),
            previous_assumption=self.rc.assumption,
        )

        prompt_action = self.prompt_utils.get_action_prompt(ctx)
        logger.info(
            f"\n\n######################## prompt_action (TextAgent):\n{prompt_action}\n"
            f"\n######################## prompt_action end\n\n"
        )

        # Use custom system prompt
        system_msg = (
            self.system_prompt
            if self.system_prompt
            else "You are a helpful AI PC operating assistant. You operate web pages by reading their accessibility tree."
        )

        # KEY DIFFERENCE: Empty images list — pure text LLM call (no screenshots)
        output_action = await self.llm.aask(
            prompt_action,
            system_msgs=[system_msg],
            images=[],  # Explicitly empty — no images sent to LLM
            stream=False,
        )

        # Parse output (same parsing logic as parent, adapted for text-only output format)
        self._parse_think_output(output_action)

        logger.info(
            f"\n\n######################## output_action (TextAgent):\n{output_action}\n"
            f"\n######################## output_action end\n\n"
        )
        logger.info(f"#### Assumption: {self.rc.assumption}")
        logger.info(f"#### Confidence: {self.rc.confidence}")

        if self.rc.action.startswith("Stop"):
            return False
        return True

    def _parse_think_output(self, output_action: str) -> None:
        """Parse LLM output into structured fields.

        Shared parsing logic extracted from OSAgent._think() for reuse.
        Handles both "### Image Description ###" (OSAgent) and "### Screen State ###" (TextAgent).
        """
        def _extract_between(text, start, end=None, normalize=False, escape_newlines=False):
            if start not in text:
                return ""
            start_idx = text.find(start) + len(start)
            if end is not None:
                end_idx = text.find(end, start_idx)
                if end_idx == -1:
                    return ""
                content = text[start_idx:end_idx]
            else:
                content = text[start_idx:]
            content = content.strip()
            if escape_newlines:
                content = content.replace("\n", "\\n")
            if normalize:
                content = content.replace(":", "")
                content = re.sub(r"\s{2,}", " ", content)
            return content.strip()

        # TextAgent uses "### Screen State ###" instead of "### Image Description ###"
        # Try both to be resilient
        screen_state = _extract_between(
            output_action, "### Screen State ###", "### Reflection Thought ###", escape_newlines=True
        )
        if not screen_state:
            screen_state = _extract_between(
                output_action, "### Image Description ###", "### Reflection Thought ###", escape_newlines=True
            )
        self.rc.image_description = screen_state  # Reuse field for compatibility

        self.rc.reflection_thought = _extract_between(
            output_action, "### Reflection Thought ###", "### Thought ###", escape_newlines=True
        )
        self.rc.thought = _extract_between(
            output_action, "### Thought ###", "### Action ###", normalize=True
        )
        self.rc.action = _extract_between(
            output_action, "### Action ###", "### Operation ###"
        )
        self.rc.summary = _extract_between(
            output_action, "### Operation ###", "### Task List ###", escape_newlines=True
        )
        self.rc.task_list = _extract_between(
            output_action, "### Task List ###", "### Assumption ###"
        )
        self.rc.assumption = _extract_between(
            output_action, "### Assumption ###", "### Confidence ###", escape_newlines=True
        )

        # Parse confidence value
        confidence_str = _extract_between(output_action, "### Confidence ###")
        try:
            confidence_match = re.search(r"(\d+\.?\d*)", confidence_str)
            if confidence_match:
                self.rc.confidence = float(confidence_match.group(1))
                self.rc.confidence = max(0.0, min(1.0, self.rc.confidence))
            else:
                self.rc.confidence = 0.0
        except (ValueError, AttributeError):
            self.rc.confidence = 0.0

    # ================================================================
    # Override: Act — simplified, no screenshot management or TellVerifier
    # ================================================================

    async def _act(self) -> AIMessage:
        """Execute action step — simplified for text-only mode.

        Differences from OSAgent._act():
        - No TellVerifier (depends on screenshots)
        - No screenshot file management (rename last/current)
        - Debug screenshots are optional
        """
        self.run_action_failed = False
        self.run_action_failed_exception = ""

        # Execute action
        if "Stop" in self.rc.action:
            return AIMessage(content=self.rc.action, cause_by=Action)
        elif "Open App" in self.rc.action:
            await self._handle_open_app()
        else:
            try:
                if self.platform in ["Android", "Windows", "Linux"]:
                    self.controller.run_action(self.rc.action)
                else:
                    logger.error("Currently only supports Android, Windows and Linux")
            except Exception as e:
                if isinstance(e, SystemExit) and e.code == 0:
                    return AIMessage(content=self.rc.action, cause_by=Action)
                logger.error(f"run action failed: {e}")
                self.run_action_failed = True
                self.run_action_failed_exception = e

        time.sleep(0.5)

        # Save previous perception info
        self.rc.last_perception_infos = copy.deepcopy(self.rc.perception_infos)

        # Get new perception info (a11y tree only)
        self.rc.perception_infos, self.width, self.height, self.output_image_path = (
            await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)
        )

        # Save debug screenshot if enabled
        if self._debug_screenshots:
            self._save_iteration_images(self.rc.iter)

        # Update history records (same as parent, but no TellVerifier)
        self.rc.thought_history.append(self.rc.thought)
        self.rc.summary_history.append(self.rc.summary)
        self.rc.action_history.append(self.rc.action)
        self.rc.assumption_history.append(self.rc.assumption)
        self.rc.confidence_history.append(self.rc.confidence)

        # Memory: store screen state description (from element tree)
        self.rc.memory.append(getattr(self.rc, "image_description", "") or "")

        # Reflection history
        self.rc.reflection_thought_history.append(self.rc.reflection_thought)

        # Handle execution errors
        if self.run_action_failed:
            self.rc.error_message = f"ERROR(run action code failed): {self.run_action_failed_exception}\\n"
            self.rc.error_flag = True
        else:
            self.rc.error_message = ""

        return AIMessage(content=self.rc.action, cause_by=Action)

    # ================================================================
    # Override: React loop — simplified, no forced TellVerifier at max_iters
    # ================================================================

    async def _react(self) -> AIMessage:
        """Main react loop — text-only version.

        Differences from OSAgent._react():
        - No TellVerifier at max_iters
        - Simplified screenshot handling
        """
        self.rc.iter = 0
        rsp = AIMessage(content="No actions taken yet", cause_by=Action)

        while self.rc.iter < self.max_iters and not self._check_last_three_start_with_wait(
            self.rc.action_history
        ):
            self.rc.iter += 1
            logger.info(f"\n\n\n\n\n\n#### iter:{self.rc.iter} (TextAgent)\n\n")

            # First iteration: get initial perception
            if self.rc.iter == 1:
                (
                    self.rc.perception_infos,
                    self.width,
                    self.height,
                    self.output_image_path,
                ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

                if self._debug_screenshots:
                    self._save_iteration_images(0)

                # Generate initial task list (text-only)
                self.rc.task_list = await self._generate_initial_task_list(
                    self.instruction, self.screenshot_file, None
                )

            # Think
            has_todo = await self._think()
            if not has_todo:
                rsp = AIMessage(content="TextAgent has finished all tasks", cause_by=Action)
                break

            # Act
            logger.debug(f"{self._setting}: {self.rc.state=}, will do {self.rc.todo}")
            rsp = await self._act()

            # Exit loop after Tell action
            if self.rc.action.startswith("Tell"):
                logger.info("Tell action completed, exiting loop")
                break

        # If reached max_iters without Tell, force a Tell action
        if self.rc.iter >= self.max_iters and not (
            self.rc.action_history and self.rc.action_history[-1].startswith("Tell")
        ):
            logger.info(f"Reached max_iters ({self.max_iters}), forcing Tell action...")

            # Get latest element tree
            (
                self.rc.perception_infos,
                self.width,
                self.height,
                self.output_image_path,
            ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

            # Force think
            has_todo = await self._think()
            if has_todo:
                rsp = await self._act()
                if not self.rc.action.startswith("Tell"):
                    # Force Tell
                    current_state = self.rc.image_description or "Unknown state"
                    self.rc.action = (
                        f"Tell (Reached maximum steps ({self.max_iters}). "
                        f"Task may be incomplete. Current state: {current_state[:200]})"
                    )
                    self.rc.summary = "Reached max steps, reporting current state"
                    if self.rc.action_history:
                        self.rc.action_history[-1] = self.rc.action
                    else:
                        self.rc.action_history.append(self.rc.action)
            else:
                # Default Tell
                self.rc.action = (
                    f"Tell (Reached maximum steps ({self.max_iters}). "
                    f"Current state: {self.rc.image_description or 'Unknown'})"
                )
                self.rc.summary = "Reached max steps, reporting current state"
                self.rc.thought_history.append(self.rc.thought or "Reached max steps")
                self.rc.summary_history.append(self.rc.summary)
                self.rc.action_history.append(self.rc.action)
                self.rc.assumption_history.append(self.rc.assumption)
                self.rc.confidence_history.append(self.rc.confidence)
                self.rc.memory.append(self.rc.image_description or "")
                self.rc.reflection_thought_history.append(self.rc.reflection_thought)

        return rsp

    # ================================================================
    # Override: Initial task list — text-only
    # ================================================================

    async def _generate_initial_task_list(
        self, instruction: str, screenshot_file: str = None, screenshot_som_file: str = None
    ) -> str:
        """Generate initial task list from element tree (no screenshots)."""
        # Format current element tree as text
        elements_text = self._format_elements_text(self.rc.perception_infos)

        prompt = f"""Based on the following instruction and the current page's accessibility tree, generate an initial task list.

**Instruction:** {instruction}

**Current Page Elements:**
{elements_text}

Please output the task list in the following format:
* **[Completed Tasks]:**
  * None
* **[Current Task]:** <describe the first high-level task to execute>
* **[Next Operation]:**
  * <describe the first step in detail>
* **[Remaining Tasks]:**
  * <describe remaining high-level task 1>
  * <describe remaining high-level task 2>
  * ...
"""

        system_msg = (
            self.system_prompt
            if self.system_prompt
            else "You are a helpful AI PC operating assistant that reads accessibility trees to understand web pages."
        )

        # Text-only LLM call — no images
        initial_task_list = await self.llm.aask(
            prompt,
            system_msgs=[system_msg],
            images=[],  # Explicitly empty
            stream=False,
        )

        task_list = initial_task_list.strip()
        logger.info(
            f"\n\n######################## Initial Task List (TextAgent):\n{task_list}\n"
            f"\n######################## End of Initial Task List\n\n"
        )
        return task_list

    # ================================================================
    # Helper methods
    # ================================================================

    @staticmethod
    def _format_elements_text(elements: List[Dict], max_elements: int = 200) -> str:
        """Format a11y tree elements as readable text for prompts."""
        if not elements:
            return "(No elements detected — the page may be loading or empty)"

        lines = []
        for i, el in enumerate(elements[:max_elements]):
            coords = el.get("coordinates", ())
            text = el.get("text", "")
            lines.append(f"  [{coords}] {text}")

        result = "\n".join(lines)
        if len(elements) > max_elements:
            result += f"\n  ... ({len(elements) - max_elements} more elements truncated)"
        return result

    def _save_iteration_images(self, iter_num: int) -> None:
        """Save debug screenshots if enabled."""
        if not self._debug_screenshots:
            return

        try:
            origin_path = f"{self.save_img}/origin_{iter_num}.jpg"
            if Path(self.screenshot_file).exists():
                shutil.copy2(self.screenshot_file, origin_path)
        except Exception as e:
            logger.debug(f"Failed to save debug screenshot: {e}")

    def _update_screenshot_files(self) -> None:
        """No-op for TextAgent — no screenshot file rotation needed."""
        pass

