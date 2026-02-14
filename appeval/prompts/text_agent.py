#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/14
@File    : text_agent.py
@Desc    : Text-only prompt templates for TextAgent (no screenshots, uses accessibility tree)
"""
from typing import Dict, List

from appeval.prompts.osagent import ActionPromptContext, BasePrompt


class TextPrompt(BasePrompt):
    """Prompt templates for text-only agent operating via accessibility tree / DOM tree.

    Core difference from PC_prompt:
    - No references to screenshots or images
    - Perception is based entirely on the structured element tree
    - Output format replaces "Image Description" with "Screen State"
    - LLM call uses text-only (no images parameter)
    """

    def __init__(self):
        super().__init__("PC")

        # ── Override: background template (no image references) ──
        self.background_template = (
            "The current web page has a viewport of {width}x{height} pixels. "
            "The user's instruction is: {instruction}.\n\n"
            "You will receive the page's **accessibility tree** — a structured list of UI elements "
            "with their roles, names, and coordinates. You do NOT have screenshots. "
            "Make all decisions based on the element tree below."
        )

        # ── Override: element info template (replaces screenshot_info_template) ──
        self.screenshot_info_template = """
### Page Elements (Accessibility Tree) ###
Below is the current page's accessibility tree extracted via Chrome DevTools Protocol.
Each entry contains: element coordinates and element description (name, control type, bounding rect).
{location_format}
{content_format}

{clickable_info}

Use this tree to understand the page layout, find interactive elements, and decide on actions.
**Important**: If an element is not in this tree, it is either not visible or not present on the current page."""

        # ── Override: hints (remove all image/screenshot references) ──
        self.hints = """
There are hints to help you complete the user's instructions. The hints are as follow:

**Critical Rules:**
- **NEVER repeat a failed action**: If an operation failed in the previous step, you MUST try a different approach. Repeating the same action will likely fail again.
- **Check for patterns**: If you see the same operation attempted 2+ times in history without success, stop and try a completely different strategy (e.g., use keyboard shortcuts instead of clicking, try a different element, or break down the task differently).
- **When stuck**: If multiple attempts fail, consider whether the current goal is achievable with available elements, or if you need to take prerequisite steps first.

**Element Interaction:**
- Use the coordinates from the accessibility tree to target elements precisely.
- When there is no direct element match, infer the likely element based on the task context and nearby elements.
- Sometimes both shortcuts and clicking can accomplish the same action; in such cases, prioritize using shortcuts.
- Perform only one click at a time. Do not skip steps; wait for the previous action to finish.
- Pay attention to the history to verify completion and avoid duplicate operations.

**Accessibility Tree Understanding:**
- The tree only shows elements currently in the DOM viewport. Use PageDown/scroll to reveal more elements.
- Elements with control_type like 'button', 'link', 'textField' are interactive.
- If an expected element is missing from the tree, it may be off-screen (try scrolling) or dynamically loaded (try waiting).
- Compare the current element tree with the previous step's tree to understand what changed after your action.
"""

        # ── Override: output format (Screen State instead of Image Description) ──
        self.output_format = """
Your output consists of the following eight parts. Please note that only one set of content should be output at a time, and do not repeat the output.

**IMPORTANT**: Each section title must be wrapped with `###` on both sides (e.g., `### Title ###`). Follow this exact format:

### Screen State ###
Based on the accessibility tree, describe the current page state concisely: what page is shown, what key elements are visible, and what interactive elements are available. Focus on elements relevant to the current task.

### Reflection Thought ###
Write a comprehensive analysis of the last operation's outcome in one paragraph. Your analysis must include:
- **What was expected vs. what actually happened**: Compare the intended outcome with the current element tree
- **Success evaluation**: Clearly state if the operation succeeded, partially succeeded, or failed, with specific evidence from the element tree (e.g., "a new 'Dashboard' heading appeared" or "the login form is still visible")
- **Failure analysis** (if applicable): Explain the root cause (e.g., wrong coordinates, element not interactive, timing issue)
- **Important observations**: Note any new elements, disappeared elements, error messages, or state changes

### Thought ###
Based on the reflection and history, plan your next action in one paragraph. Your thought process must include:
- **History review**: Check recent operations for any patterns of repeated failures with the same approach
- **Strategy validation**: If the last operation failed, explain why your new approach will be different
- **Element selection**: Identify the specific element(s) from the tree you plan to interact with, citing their coordinates
- **Clear reasoning**: State your logic before deciding on the next action

### Action ###
{action_options}

### Operation ###
This is a one sentence summary of this operation.

### Task List ###
* **[Completed Tasks]:** (List the tasks that have been successfully completed so far)
    * <Task 1 Description>
    * <Task 2 Description>
    ...
* **[Current Task]:** <Current Task Description>.
* **[Next Operation]:** (Describe the immediate next operation in detail, including what needs to be done)
    * <Step 1 Description>
    * <Step 2 Description>
    ...
* **[Remaining Tasks]:** (List the remaining high-level tasks that need to be completed to achieve the user's objective, excluding the current and next operation.)
    * <Task 1 Description>
    * <Task 2 Description>
    ...

### Assumption ###
Based on your analysis, state your assumption about whether the task can meet the expected result. Format: "The function can/cannot meet the expected result because [reason]"

### Confidence ###
Rate your confidence level from 0 to 1 based ONLY on the strength of available observable evidence.

Definition:
- 0.0–0.2: Pure speculation, no supporting evidence
- 0.3–0.5: Weak or indirect evidence
- 0.6–0.8: Partial but incomplete evidence
- 0.9–1.0: Strong, explicit, verifiable evidence

If no direct evidence is available, confidence must not exceed 0.3.

Output a single number only.

"""

        # ── Override: task requirements (same actions, but no screenshot references) ──
        self.task_requirements = """
In order to meet the user's requirements, you need to select one of the following operations to operate on the current page:
For certain items that require selection, such as font and font size, direct input is more efficient than scrolling through choices.
You must choose one of the actions below:

- Run (your code)
    You can use this action to run python code. Your code will run on the computer for controlling the mouse and keyboard. You are required to use `pyautogui` to perform the action grounded to the element coordinates from the accessibility tree. DO NOT use `pyautogui.locateCenterOnScreen`. DO NOT USE `pyautogui.screenshot()`. Return one line of python code to perform the action each time. When predicting multiple code statements, separate them with semicolons (;) within the same line.

    For actions:
    - Use `pyautogui.click(x, y)` for single clicks (use center coordinates from the element tree)
    - Use `pyautogui.doubleClick(x, y)` for double clicks
    - Use `pyautogui.rightClick(x, y)` for right clicks
    - Use `pyautogui.moveTo(x, y)` to move mouse
    - Use `pyautogui.dragTo(x, y)` to drag mouse
    - Use `pyautogui.scroll(amount)` to scroll, positive numbers scroll up, negative scroll down
    - Use `pyautogui.hotkey(key1, key2)` for keyboard shortcuts
    - Use `pyautogui.press('pagedown')` to scroll the page down and reveal more elements

    For text input:
    1. First click the target text field using its coordinates from the element tree
    2. Use `pyperclip.copy(text)` to copy the content
    3. Use `pyautogui.hotkey('ctrl', 'v')` to paste it
    4. Make sure the content of text is enclosed in triple quotes

    Each action must end with a `time.sleep(duration)` statement.
    - Simple clicks: 0.5-1 second
    - Text input: 1-2 seconds
    - Opening apps/pages: 5-10 seconds
    - Loading content: 2-5 seconds

    Example: Run (pyperclip.copy(\"\"\"hello\"\"\"); time.sleep(0.5); pyautogui.hotkey('ctrl', 'v'); time.sleep(1))

    Limit the execution to no more than 8 steps at a time to avoid errors.

- Tell (your answer)
    If you think the user's instruction has been fully satisfied, use this action to answer the user's question in English, and tell me the final answer. The final answer must be inside the brackets. Do not reuse this action to output the same response.

- Stop
    If all the operations to meet the user's requirements have been completed in ### History operation ###, use this operation to stop the whole process."""

    def _build_background(self, ctx: ActionPromptContext, device_type: str = "computer") -> str:
        """Build background information section (no image references)."""
        return self.background_template.format(
            width=ctx.width, height=ctx.height, instruction=ctx.instruction
        )

    def _build_screenshot_info(self, ctx: ActionPromptContext, source_desc: str = "") -> str:
        """Build element information section (replaces screenshot info)."""
        location_format = {
            "center": "Coordinates format: [x, y] — center point of the element.",
            "bbox": "Coordinates format: [x1, y1, x2, y2] — bounding box (top-left, bottom-right).",
        }.get(ctx.location_info, "Coordinates format: [x, y] — center point.")

        content_format = (
            "Each element shows: text (element name/label), control_type (button, link, textField, etc.), "
            "and rect (bounding rectangle in pixels)."
        )

        clickable_info = "\n".join(
            f"  {info['coordinates']}; {info['text']}"
            for info in ctx.clickable_infos
            if info.get("text", "") != "" and info.get("coordinates") != (0, 0)
        )

        if not clickable_info:
            clickable_info = "(No elements detected — the page may be loading or empty)"

        return self.screenshot_info_template.format(
            location_format=location_format,
            content_format=content_format,
            clickable_info=clickable_info,
        )

    def _build_assumption_prompt(self, ctx: ActionPromptContext) -> str:
        """Build assumption verification prompt (adapted for text-only)."""
        if ctx.is_first_step:
            return """
### Assumption Verification ###
I guess the function can meet the expected result but possibly wrong. Verify my assumption by analyzing the element tree and then step by step find the correct answer.
"""
        else:
            can_or_cannot = "can" if "can meet" in ctx.previous_assumption.lower() \
                or "cannot" not in ctx.previous_assumption.lower() else "cannot"
            return f"""
### Assumption Verification ###
Based on previous observation, I guess the function {can_or_cannot} ({ctx.previous_assumption}) meet the expected result but possibly wrong because I might have misinterpreted the element tree, lack complete signals as proof of full success or failure, or explored the page insufficiently. Verify my assumption and then step by step find the correct answer.
"""

    def get_action_prompt(self, ctx: ActionPromptContext) -> str:
        """Build the complete text-only action prompt."""
        background = self._build_background(ctx)
        element_info = self._build_screenshot_info(ctx)
        history_operations = self._build_history_operations(ctx)
        task_list = self._build_task_list(ctx)
        last_operation = self._build_last_operation(ctx)
        assumption_prompt = self._build_assumption_prompt(ctx)

        return self.prompt_template.format(
            background=background,
            screenshot_info=element_info,
            hints=self.hints,
            additional_info=ctx.add_info,
            history_operations=history_operations,
            task_list=task_list,
            last_operation=last_operation,
            task_requirements=self.task_requirements,
            assumption_prompt=assumption_prompt,
            output_format=self.output_format.format(
                action_options="Run () or Tell () or Stop. Only one action can be output at one time."
            ),
        )


# ── System prompt for text-only batch testing ──
text_agent_system_prompt = """
You are a professional and responsible web testing engineer. You operate web pages by reading their accessibility tree (structured element list) — you do NOT have screenshots.

Your workflow:
1. Read the element tree to understand the current page state
2. Plan and execute test operations using pyautogui with coordinates from the element tree
3. After each action, read the updated element tree to verify the outcome
4. Report test results using the Tell action

Key principles:
- Trust the element tree as your source of truth for page state
- If an element is not in the tree, it's not visible on screen — try scrolling (PageDown) to reveal it
- Compare element trees between steps to understand what changed
- You must test ALL test cases before reporting. Do not skip or fabricate results.

Reporting format:
{
    "0": {"result": "Pass", "evidence": "Element tree shows 'Dashboard' heading appeared after login"},
    "1": {"result": "Fail", "evidence": "After clicking submit, no success message element appeared in the tree"}
}
**Return only the result string. Do not include any additional text, markdown formatting, or code blocks.**
"""

