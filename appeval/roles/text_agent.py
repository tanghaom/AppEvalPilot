#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/14
@File    : text_agent.py
@Desc    : Text-only OS Agent that uses accessibility tree / DOM tree instead of screenshots.
           Uses factory function + monkey-patch to avoid pydantic coercing subclass back to OSAgent.
"""
import copy
import re
import shutil
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.schema import AIMessage

from appeval.prompts.osagent import ActionPromptContext
from appeval.prompts.text_agent import TextPrompt, text_agent_system_prompt
from appeval.roles.osagent import OSAgent


def create_text_agent(**kwargs) -> OSAgent:
    """Factory: create an OSAgent instance configured for text-only mode.

    Pydantic coerces TextAgent subclass back to OSAgent during model validation,
    so we use a factory function + monkey-patch instead of subclassing.

    Returns an OSAgent with text-only methods patched onto the instance.
    """
    # Force disable all image-dependent features
    kwargs["use_ocr"] = False
    kwargs["use_icon_detect"] = False
    kwargs["use_icon_caption"] = False
    kwargs["use_som"] = False
    kwargs["use_tell_verifier"] = False
    kwargs["use_reflection"] = False
    kwargs.setdefault("use_chrome_debugger", False)
    kwargs.setdefault("think_history_images", 0)

    debug_screenshots = kwargs.pop("debug_screenshots", True)

    if not kwargs.get("system_prompt"):
        kwargs["system_prompt"] = text_agent_system_prompt

    agent = OSAgent(**kwargs)

    # Store text-mode config
    agent._text_mode = True
    agent._debug_screenshots = debug_screenshots

    # Replace prompt utils
    agent.prompt_utils = TextPrompt()

    # Monkey-patch core methods with text-only implementations
    agent._react = types.MethodType(_react_text, agent)
    agent._get_perception_infos = types.MethodType(_get_perception_infos_text, agent)
    agent._generate_initial_task_list = types.MethodType(_generate_initial_task_list_text, agent)
    agent._save_iteration_images = types.MethodType(_save_iteration_images_text, agent)
    agent._update_screenshot_files = types.MethodType(_update_screenshot_files_noop, agent)

    logger.info(f"TextAgent created (factory): a11y_mode={agent.a11y_mode}, "
                f"debug_screenshots={debug_screenshots}")
    return agent


# ================================================================
# Text-only method implementations (will be bound to OSAgent instance)
# ================================================================

async def _get_perception_infos_text(
    self, screenshot_file: str, screenshot_som_file: str
) -> Tuple[List[Dict[str, Any]], int, int, str]:
    """Get perception info from accessibility tree only. No screenshots for LLM."""
    if getattr(self, '_debug_screenshots', True):
        try:
            self.controller.get_screenshot(screenshot_file)
        except Exception as e:
            logger.debug(f"Debug screenshot failed (non-critical): {e}")

    perception_infos = []
    if self.extend_xml_infos and self.platform in ["Android", "Windows", "Linux"]:
        perception_infos = self.controller.get_screen_xml(self.location_info)

    width, height = 1920, 1080
    if getattr(self, '_debug_screenshots', True):
        try:
            from PIL import Image
            img = Image.open(screenshot_file)
            width, height = img.size
        except Exception:
            pass

    logger.info(f"TextAgent perception: {len(perception_infos)} elements from a11y tree")
    return perception_infos, width, height, ""


async def _think_text(self) -> bool:
    """Generate operation decisions using text-only LLM (no images)."""
    add_info = self.add_info

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

    system_msg = (
        self.system_prompt
        if self.system_prompt
        else "You are a helpful AI PC operating assistant. You operate web pages by reading their accessibility tree."
    )

    # KEY: images=[] — pure text LLM call, no screenshots
    output_action = await self.llm.aask(
        prompt_action,
        system_msgs=[system_msg],
        images=[],
        stream=False,
    )

    _parse_think_output(self, output_action)

    logger.info(
        f"\n\n######################## output_action (TextAgent):\n{output_action}\n"
        f"\n######################## output_action end\n\n"
    )
    logger.info(f"#### Assumption: {self.rc.assumption}")
    logger.info(f"#### Confidence: {self.rc.confidence}")

    return not self.rc.action.startswith("Stop")


def _parse_think_output(self, output_action: str) -> None:
    """Parse LLM output. Handles both Screen State (TextAgent) and Image Description (OSAgent)."""
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

    screen_state = _extract_between(
        output_action, "### Screen State ###", "### Reflection Thought ###", escape_newlines=True
    )
    if not screen_state:
        screen_state = _extract_between(
            output_action, "### Image Description ###", "### Reflection Thought ###", escape_newlines=True
        )
    self.rc.image_description = screen_state

    self.rc.reflection_thought = _extract_between(
        output_action, "### Reflection Thought ###", "### Thought ###", escape_newlines=True)
    self.rc.thought = _extract_between(
        output_action, "### Thought ###", "### Action ###", normalize=True)
    self.rc.action = _extract_between(
        output_action, "### Action ###", "### Operation ###")
    self.rc.summary = _extract_between(
        output_action, "### Operation ###", "### Task List ###", escape_newlines=True)
    self.rc.task_list = _extract_between(
        output_action, "### Task List ###", "### Assumption ###")
    self.rc.assumption = _extract_between(
        output_action, "### Assumption ###", "### Confidence ###", escape_newlines=True)

    confidence_str = _extract_between(output_action, "### Confidence ###")
    try:
        match = re.search(r"(\d+\.?\d*)", confidence_str)
        self.rc.confidence = max(0.0, min(1.0, float(match.group(1)))) if match else 0.0
    except (ValueError, AttributeError):
        self.rc.confidence = 0.0


async def _act_text(self) -> AIMessage:
    """Execute action — simplified for text-only mode. No TellVerifier, no screenshot management."""
    self.run_action_failed = False
    self.run_action_failed_exception = ""

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
    self.rc.last_perception_infos = copy.deepcopy(self.rc.perception_infos)

    self.rc.perception_infos, self.width, self.height, self.output_image_path = (
        await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)
    )

    if getattr(self, '_debug_screenshots', True):
        self._save_iteration_images(self.rc.iter)

    self.rc.thought_history.append(self.rc.thought)
    self.rc.summary_history.append(self.rc.summary)
    self.rc.action_history.append(self.rc.action)
    self.rc.assumption_history.append(self.rc.assumption)
    self.rc.confidence_history.append(self.rc.confidence)
    self.rc.memory.append(getattr(self.rc, "image_description", "") or "")
    self.rc.reflection_thought_history.append(self.rc.reflection_thought)

    if self.run_action_failed:
        self.rc.error_message = f"ERROR(run action code failed): {self.run_action_failed_exception}\\n"
        self.rc.error_flag = True
    else:
        self.rc.error_message = ""

    return AIMessage(content=self.rc.action, cause_by=Action)


async def _react_text(self) -> AIMessage:
    """Main react loop — text-only version."""
    self.rc.iter = 0
    rsp = AIMessage(content="No actions taken yet", cause_by=Action)

    while self.rc.iter < self.max_iters and not self._check_last_three_start_with_wait(
        self.rc.action_history
    ):
        self.rc.iter += 1
        logger.info(f"\n\n\n\n\n\n#### iter:{self.rc.iter} (TextAgent)\n\n")

        if self.rc.iter == 1:
            (
                self.rc.perception_infos, self.width, self.height, self.output_image_path,
            ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

            if getattr(self, '_debug_screenshots', True):
                self._save_iteration_images(0)

            self.rc.task_list = await self._generate_initial_task_list(
                self.instruction, self.screenshot_file, None
            )

        # Think (text-only, no images)
        has_todo = await _think_text(self)
        if not has_todo:
            rsp = AIMessage(content="TextAgent has finished all tasks", cause_by=Action)
            break

        # Act (text-only, no TellVerifier)
        logger.debug(f"{self._setting}: {self.rc.state=}, will do {self.rc.todo}")
        rsp = await _act_text(self)

        if self.rc.action.startswith("Tell"):
            logger.info("Tell action completed, exiting loop")
            break

    # Force Tell at max_iters
    if self.rc.iter >= self.max_iters and not (
        self.rc.action_history and self.rc.action_history[-1].startswith("Tell")
    ):
        logger.info(f"Reached max_iters ({self.max_iters}), forcing Tell action...")
        (
            self.rc.perception_infos, self.width, self.height, self.output_image_path,
        ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

        has_todo = await _think_text(self)
        if has_todo:
            rsp = await _act_text(self)
            if not self.rc.action.startswith("Tell"):
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


async def _generate_initial_task_list_text(
    self, instruction: str, screenshot_file: str = None, screenshot_som_file: str = None
) -> str:
    """Generate initial task list from element tree (no screenshots)."""
    elements_text = _format_elements(self.rc.perception_infos)

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
  * ...
"""

    system_msg = (
        self.system_prompt
        if self.system_prompt
        else "You are a helpful AI PC operating assistant that reads accessibility trees to understand web pages."
    )

    result = await self.llm.aask(prompt, system_msgs=[system_msg], images=[], stream=False)
    task_list = result.strip()
    logger.info(
        f"\n\n######################## Initial Task List (TextAgent):\n{task_list}\n"
        f"\n######################## End of Initial Task List\n\n"
    )
    return task_list


def _save_iteration_images_text(self, iter_num: int) -> None:
    """Save debug screenshots if enabled."""
    if not getattr(self, '_debug_screenshots', True):
        return
    try:
        origin_path = f"{self.save_img}/origin_{iter_num}.jpg"
        if Path(self.screenshot_file).exists():
            shutil.copy2(self.screenshot_file, origin_path)
    except Exception as e:
        logger.debug(f"Failed to save debug screenshot: {e}")


def _update_screenshot_files_noop(self) -> None:
    """No-op — no screenshot file rotation needed in text mode."""
    pass


def _format_elements(elements: List[Dict], max_elements: int = 200) -> str:
    """Format a11y tree elements as readable text."""
    if not elements:
        return "(No elements detected — the page may be loading or empty)"
    lines = [f"  [{el.get('coordinates', ())}] {el.get('text', '')}" for el in elements[:max_elements]]
    result = "\n".join(lines)
    if len(elements) > max_elements:
        result += f"\n  ... ({len(elements) - max_elements} more elements truncated)"
    return result


# Keep TextAgent class as alias for backward compatibility (isinstance checks in docs, etc.)
TextAgent = OSAgent
