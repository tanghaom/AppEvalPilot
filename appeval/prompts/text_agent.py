#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/14
@File    : text_agent.py
@Desc    : Optimized text-only prompt templates for TextAgent v2.
           4-section output (Changes / Thought / Action / Operation).
           No screenshots; uses accessibility tree only.
           Does NOT affect OSAgent prompts (osagent.py is untouched).
"""
from typing import Dict, List

from appeval.prompts.osagent import ActionPromptContext, BasePrompt


class TextPrompt(BasePrompt):
    """Prompt templates for text-only agent operating via accessibility tree / DOM tree.

    v2 optimizations vs v1:
    - Output reduced from 8 sections → 4 (Changes, Thought, Action, Operation)
    - No "Screen State" (avoids re-describing the input element tree)
    - Reflection + Planning + Assumption + Confidence merged into single "Thought"
    - Task List generated once (initial), shown as read-only reference thereafter
    - History format: compact one-liner per step
    - Element diff computed programmatically and injected as context
    """

    def __init__(self):
        super().__init__("PC")

        # ── Simplified background ──
        self.background_template = (
            "You are testing a web page (viewport {width}×{height} px).\n"
            "Instruction: {instruction}\n\n"
            "You receive the page's **accessibility tree** — a structured element list "
            "with roles, names, and coordinates. You do NOT have screenshots. "
            "Decide all actions based on the element tree."
        )

        # ── Compact element list ──
        self.screenshot_info_template = """
### Page Elements (Accessibility Tree) ###
{location_format}
{clickable_info}
If an element is not listed, it is not visible or not present on the current page."""

        # ── Compact hints with early-termination + interaction verification rules ──
        self.hints = """
**Rules:**
- NEVER repeat a failed action with the same approach. If it failed, try differently.
- If 2+ attempts with the same strategy failed, switch to a fundamentally different method.
- Use coordinates from the element tree. If no exact match, infer from context and nearby elements.
- Prefer keyboard shortcuts over clicking when both work.
- One action at a time. Wait for the previous action to complete.
- The tree only shows elements in DOM viewport. Use PageDown/scroll to reveal more.
- Elements with control_type 'button', 'link', 'textField' are interactive.
- **Smart search**: If a target element is NOT found after scrolling, try `pyautogui.hotkey('ctrl', 'f')` to open browser search and type the keyword to locate it quickly. If Ctrl+F also finds nothing, the feature likely does not exist — report Fail immediately.
- **Interaction verification**: After modifying a value (input, toggle, select, edit), you MUST verify the change actually took effect by checking the element's current value/state in the NEXT element tree. Do NOT assume success just because the action ran without error. If the value in the element tree did not change, the modification FAILED — try a different approach (e.g. clear field first, use clipboard paste, or try a different input method).
"""

        # ── 4-section output format ──
        self.output_format = """
Output exactly four sections. Each title must use `### Title ###` format:

### Changes ###
In 1–3 sentences, describe ONLY what changed compared to the previous step.
Focus on: new elements that appeared, elements that disappeared, text/value changes.
If screenshots are provided, also note any visual changes (colors, layout, rendering, animations).
If this is the first step, write "Initial page load."
Do NOT re-describe the full page — only the delta.

### Thought ###
In one concise paragraph, combine reflection and planning:
1. Did the last action succeed or fail? Cite specific evidence from the current element tree (check actual element values/states, not just whether the action ran).
2. If you modified a value, verify it changed in the element tree. If unchanged, the edit FAILED.
3. If failed, what was the root cause and how will your next approach differ?
4. What will you do next and why? Cite the target element's coordinates.
5. If the target feature is clearly missing after thorough exploration, state so and plan to Tell Fail.

End with exactly these two lines:
`Status: completed [task1, task2] | current: <current task> | next: <next step>`
`Assumption: [can/cannot] meet expected result | Confidence: [0.0-1.0]`

### Action ###
{action_options}

### Operation ###
One-sentence summary of this operation.
"""

        # ── Compact task requirements ──
        self.task_requirements = """
Choose one action:

- Run (your code)
    Return ONE LINE of python code. When using multiple statements, separate them with semicolons (;) on the SAME line. NEVER use newlines inside the Run() parentheses.
    Use `pyautogui` with coordinates from the element tree:
    - `pyautogui.click(x, y)` — single click
    - `pyautogui.doubleClick(x, y)` — double click
    - `pyautogui.rightClick(x, y)` — right click
    - `pyautogui.moveTo(x, y)` — hover
    - `pyautogui.dragTo(x, y)` — drag
    - `pyautogui.scroll(amount)` — scroll (positive=up, negative=down)
    - `pyautogui.hotkey(key1, key2)` — keyboard shortcut
    - `pyautogui.press('pagedown')` — page down
    - For text input: click target field, then `pyperclip.copy(\"\"\"text\"\"\"); pyautogui.hotkey('ctrl', 'v')`
    - End with `time.sleep(duration)` (click: 0.5–1s, input: 1–2s, page load: 3–5s)
    - Max 8 statements per action.
    DO NOT use `pyautogui.locateCenterOnScreen` or `pyautogui.screenshot()`.
    Example: Run (pyautogui.click(558, 601); time.sleep(1))
    Example: Run (pyautogui.press('pagedown'); time.sleep(1))

- Tell (your answer)
    Report final test results. The answer must be inside the brackets.
    Do not reuse Tell to output the same response.

- Stop
    When all tasks in the History are completed."""

    # ──────────────────── Builder methods ────────────────────

    def _build_background(self, ctx: ActionPromptContext, device_type: str = "computer") -> str:
        return self.background_template.format(
            width=ctx.width, height=ctx.height, instruction=ctx.instruction
        )

    def _build_screenshot_info(self, ctx: ActionPromptContext, source_desc: str = "") -> str:
        location_format = {
            "center": "Coordinates: [x, y] — element center point.",
            "bbox": "Coordinates: [x1, y1, x2, y2] — bounding box.",
        }.get(ctx.location_info, "Coordinates: [x, y] — center point.")

        clickable_info = "\n".join(
            f"  {info['coordinates']}; {info['text']}"
            for info in ctx.clickable_infos
            if info.get("text", "") != "" and info.get("coordinates") != (0, 0)
        )
        if not clickable_info:
            clickable_info = "(No elements detected — page may be loading or empty)"

        return self.screenshot_info_template.format(
            location_format=location_format,
            clickable_info=clickable_info,
        )

    def _build_history_operations(self, ctx: ActionPromptContext) -> str:
        """Compact history: one line per step + optional result."""
        if len(ctx.action_history) == 0:
            return ""

        lines = []
        for i in range(len(ctx.action_history)):
            op = ctx.summary_history[i].replace("\n", " ").replace("\r", " ") if i < len(ctx.summary_history) else ""
            act = ctx.action_history[i].replace("\n", " ").replace("\r", " ")
            line = f"Step-{i+1}: {op} → {act}"

            # Reflection from next step reflects on this step's outcome
            ri = i + 1
            if ri < len(ctx.reflection_thought_history) and ctx.reflection_thought_history[ri]:
                ref = ctx.reflection_thought_history[ri].replace("\n", " ").replace("\r", " ")
                line += f"\n  Result: {ref[:300]}"
            lines.append(line)

        history_text = "\n".join(lines)
        return (
            "### History ###\n"
            "Completed operations (oldest → newest). Review for repeated failures.\n"
            f"{history_text}"
        )

    def _build_task_list(self, ctx: ActionPromptContext) -> str:
        """Show task list as read-only reference (not regenerated each step)."""
        if not ctx.task_list:
            return ""
        return (
            "### Task Plan (reference only — do NOT reproduce in output) ###\n"
            f"{ctx.task_list}"
        )

    def _build_assumption_prompt(self, ctx: ActionPromptContext) -> str:
        if ctx.is_first_step:
            return ""
        prev = ctx.previous_assumption or "unknown"
        return f"### Previous Assumption ###\n{prev}\nRe-evaluate based on current evidence.\n"

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
                action_options="Run () or Tell () or Stop. Only one action at a time."
            ),
        )


# ── Utility: compute element diff ──

def compute_element_diff(
    current_elements: List[Dict],
    previous_elements: List[Dict],
    max_show: int = 20,
) -> str:
    """Compute a compact textual diff between two element lists.

    Returns a markdown section string ready to inject into the prompt,
    or empty string if no previous elements (first step).
    """
    if not previous_elements:
        return ""

    curr_texts = {el.get("text", "") for el in current_elements if el.get("text")}
    prev_texts = {el.get("text", "") for el in previous_elements if el.get("text")}

    added = curr_texts - prev_texts
    removed = prev_texts - curr_texts

    if not added and not removed:
        return "\n### Element Diff (auto-computed) ###\nNo significant element changes since last step.\n"

    parts = ["### Element Diff (auto-computed) ###"]
    if added:
        sample = sorted(added)[:max_show]
        parts.append(f"**+New** ({len(added)}): {', '.join(sample)}")
    if removed:
        sample = sorted(removed)[:max_show]
        parts.append(f"**−Removed** ({len(removed)}): {', '.join(sample)}")

    return "\n" + "\n".join(parts) + "\n"


# ── System prompt for text-only batch testing (with optional visual supplement) ──
text_agent_system_prompt = """You are a web testing engineer. You operate web pages primarily by reading their accessibility tree.

Workflow:
1. Read the element tree to understand the page.
2. Execute test operations using pyautogui with coordinates from the tree.
3. After each action, read the updated element tree to verify the outcome.
4. If screenshots are provided, use them to verify visual aspects (colors, layout, rendering).
5. Report results via Tell.

Principles:
- The element tree is your primary source of truth for structure and coordinates.
- When screenshots are provided, use them to verify visual rendering that the element tree cannot capture (e.g., colors, layout, animations, Canvas/WebGL game content like scores, health bars, game objects).
- **Canvas/WebGL content**: Games and canvas-based apps render content as pixels. Scores, health bars, game objects, and UI elements are NOT in the accessibility tree. For canvas/game testing, rely on screenshots for verification.
- After modifying a value, always check the element tree to confirm the change took effect.
- If an element is missing, try scrolling (PageDown) before concluding it doesn't exist.
- Compare element trees between steps to detect changes.
- You must test ALL cases. Do not skip or fabricate results.
- If a feature is absent after full-page exploration, report Fail with evidence.

Result format:
{
    "0": {"result": "Pass", "evidence": "..."},
    "1": {"result": "Fail", "evidence": "..."}
}
**Return only the result string. No extra text, markdown, or code blocks.**
"""
