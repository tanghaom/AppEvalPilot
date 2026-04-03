#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/14
@File    : text_agent.py
@Desc    : Text-only OS Agent v2 — uses accessibility tree / DOM tree instead of screenshots.
           Factory function + monkey-patch to avoid Pydantic coercing subclass back to OSAgent.
           Optimized 4-section output: Changes / Thought / Action / Operation.
           v2.1: Visual Supplement — LLM auto-detects if a case needs visual info,
                 then attaches prev+curr screenshots for visual verification.
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
from metagpt.utils.common import encode_image

from appeval.prompts.osagent import ActionPromptContext
from appeval.prompts.text_agent import TextPrompt, compute_element_diff, is_tree_unchanged, text_agent_system_prompt
from appeval.roles.osagent import OSAgent


# ================================================================
# Visual Supplement: LLM auto-classification prompt
# ================================================================

_VISUAL_CHECK_PROMPT = """Given this test instruction, does verifying the result require SEEING the visual appearance of the page?

Visual verification is needed when the test checks:
- Colors, themes, or styling (e.g. "verify the button turns red")
- Animations or transitions (e.g. "check the animation plays")
- Images, thumbnails, or media rendering (e.g. "verify the 3D model displays")
- Layout, positioning, or responsive design (e.g. "check mobile layout")
- Font styles, syntax highlighting, or text rendering
- Visual effects (e.g. shadows, gradients, hover effects)
- Charts, graphs, or data visualization rendering
- **Canvas/WebGL game content** (e.g. score, health bar, game objects, game over screen).
  Games built with Canvas/WebGL render ALL content as pixels — scores, health, objects,
  and UI are NOT in the accessibility tree. Any test involving a game requires visual verification.
- Game state verification (e.g. "check score increases", "verify health decreases",
  "check game over screen shows final score") — these values are rendered on canvas,
  not as DOM text elements.
- **SVG / custom component interactions** where the target element is rendered visually
  but may lack accessibility attributes (e.g. SVG chart segments, icon-only buttons,
  CSS-styled divs, shadow DOM widgets, custom sliders or toggles).
- **Interactions where the target is described visually** rather than by a text label
  (e.g. "click the top-right icon", "drag the handle", "click the highlighted cell",
  "select the colored chip", "click the X button in the corner").

Visual verification is NOT needed when the test only checks:
- Functionality (click, input, navigation, form submission) on standard HTML pages
  where the target element has a clear text label in the accessibility tree
- Text content, labels, or values in standard DOM elements
- Element existence or absence in standard web pages
- Error messages or notifications (text-based)
- API responses or data correctness

IMPORTANT: If the page title or instruction mentions "game", "shooter", "breaker",
"puzzle", "canvas", "player", "score", "health", "level" in a gaming context,
answer YES — game content is rendered on canvas and invisible to accessibility tools.

Instruction: {instruction}

Answer only YES or NO."""


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
    kwargs["use_tell_verifier"] = True
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
    agent._visual_supplement = False  # Will be set at iter==1 by LLM auto-detection
    agent._prev_screenshot_path = ""  # Path to previous step's screenshot for visual diff

    # Replace prompt utils with optimized v2 TextPrompt
    agent.prompt_utils = TextPrompt()

    # Monkey-patch core methods with text-only implementations
    agent._react = types.MethodType(_react_text, agent)
    agent._get_perception_infos = types.MethodType(_get_perception_infos_text, agent)
    agent._generate_initial_task_list = types.MethodType(_generate_initial_task_list_text, agent)
    agent._save_iteration_images = types.MethodType(_save_iteration_images_text, agent)
    agent._update_screenshot_files = types.MethodType(_update_screenshot_files_noop, agent)

    logger.info(f"TextAgent v2 created (factory): a11y_mode={agent.a11y_mode}, "
                f"debug_screenshots={debug_screenshots}")
    return agent


# ================================================================
# Runtime Canvas/sparse tree detection
# ================================================================

# Browser chrome elements that are always present regardless of page content
_BROWSER_CHROME_KEYWORDS = {
    "back", "forward", "reload", "address and search bar", "bookmark",
    "chrome", "new tab", "search tabs", "close", "separator",
    "managed bookmarks", "saved tab groups", "tab groups",
    "hidden toolbar buttons", "all bookmarks", "menu containing",
    "infobar", "you are using an unsupported", "view site information",
}

# Game-related keywords in page content or title
_GAME_CONTENT_KEYWORDS = re.compile(
    r"game|shooter|breaker|canvas|player|score|health|level|paddle|brick|"
    r"enemy|bullet|ship|tetris|puzzle|match|arcade|maze|snake|pong",
    re.IGNORECASE,
)


def _is_sparse_a11y_tree(
    elements: List[Dict[str, Any]], instruction: str
) -> bool:
    """Detect if the a11y tree is too sparse to be useful (Canvas/WebGL app).

    Returns True if visual mode should be forced ON.

    Heuristics:
    1. Count "content elements" (excluding browser chrome like Back, Forward, address bar).
    2. If content elements ≤ 5 AND there's a canvas/game-like element, it's likely a Canvas app.
    3. If the page title or instruction contains game-related keywords, lower the threshold.
    """
    if not elements:
        return False

    content_elements = []
    has_canvas_hint = False
    page_title = ""

    for el in elements:
        text = (el.get("text") or "").strip().lower()
        ctrl = (el.get("control_type") or "").strip().lower()

        # Check for Canvas/game hints
        if "canvas" in text or "canvas" in ctrl:
            has_canvas_hint = True
        if ctrl in ("document web", "frame") and _GAME_CONTENT_KEYWORDS.search(text):
            has_canvas_hint = True
            page_title = text

        # Skip browser chrome elements
        is_chrome = False
        for kw in _BROWSER_CHROME_KEYWORDS:
            if kw in text:
                is_chrome = True
                break
        if ctrl in ("frame", "page tab", "tool bar", "separator", "panel", "alert"):
            is_chrome = True

        if not is_chrome and text:
            content_elements.append(el)

    # Check instruction for game keywords
    instruction_is_game = bool(_GAME_CONTENT_KEYWORDS.search(instruction))

    # Decision logic
    n_content = len(content_elements)

    if has_canvas_hint and n_content <= 10:
        logger.info(
            f"🎮 Sparse tree + Canvas detected: {n_content} content elements, "
            f"page='{page_title[:60]}' → forcing visual mode"
        )
        return True

    if instruction_is_game and n_content <= 5:
        logger.info(
            f"🎮 Sparse tree + game instruction: {n_content} content elements → forcing visual mode"
        )
        return True

    return False


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


# ── Visual Supplement: LLM auto-detection ──

async def _detect_visual_need(self, instruction: str) -> bool:
    """Ask the LLM whether this test case requires visual verification.

    Called once at iter==1. Result is cached for the rest of the run.
    Cost: ~50 tokens (one short yes/no call).
    """
    prompt = _VISUAL_CHECK_PROMPT.format(instruction=instruction)
    system_msg = "Answer only YES or NO."
    try:
        result = await self.llm.aask(prompt, system_msgs=[system_msg], images=[], stream=False)
        needs_visual = result.strip().upper().startswith("YES")
        logger.info(f"🔍 Visual supplement detection: {'YES → attaching screenshots' if needs_visual else 'NO → text-only mode'}")
        return needs_visual
    except Exception as e:
        logger.warning(f"Visual detection failed (defaulting to text-only): {e}")
        return False


# ── Think (v2.1: with element diff + optional visual supplement) ──

async def _think_text(self) -> bool:
    """Generate operation decisions using text-only LLM.

    v2.1: injects a programmatic element diff into add_info so the model
    only needs to interpret what changed, not re-describe the whole page.
    When _visual_supplement is True, attaches prev+curr screenshots.
    """
    # Build element diff
    prev_elements = getattr(self.rc, 'last_perception_infos', None) or []
    diff_text = compute_element_diff(self.rc.perception_infos, prev_elements)

    # Prepend diff to add_info
    add_info = self.add_info
    if diff_text:
        add_info = diff_text + "\n" + add_info

    # Visual supplement: inject hint into add_info when screenshots are attached
    visual_mode = getattr(self, '_visual_supplement', False)
    if visual_mode:
        visual_hint = (
            "\n### Visual Context ###\n"
            "Screenshots are provided for visual verification. "
        )
        if self.rc.iter > 1:
            visual_hint += (
                "Two images: BEFORE (previous step) and AFTER (current). "
                "Compare them for visual changes (colors, layout, rendering, animations)."
            )
        else:
            visual_hint += (
                "One image: current page state. "
                "Use it to verify visual aspects (colors, layout, rendering)."
            )
        visual_hint += "\nThe accessibility tree remains your primary source for element coordinates and structure.\n"
        add_info = visual_hint + "\n" + add_info

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
        f"\n\n######################## prompt_action (TextAgent v2.1):\n{prompt_action}\n"
        f"\n######################## prompt_action end\n\n"
    )

    system_msg = (
        self.system_prompt
        if self.system_prompt
        else "You are a helpful AI web testing assistant that reads accessibility trees."
    )

    # Build images list: empty for text-only, prev+curr for visual supplement
    images = []
    if visual_mode and getattr(self, '_debug_screenshots', True):
        try:
            # Previous step screenshot (for visual diff)
            prev_path = getattr(self, '_prev_screenshot_path', "")
            if prev_path and Path(prev_path).exists():
                images.append(encode_image(prev_path))
            # Current screenshot
            if Path(self.screenshot_file).exists():
                images.append(encode_image(self.screenshot_file))
            if images:
                logger.info(f"📸 Visual supplement: attaching {len(images)} screenshot(s)")
        except Exception as e:
            logger.warning(f"Failed to encode screenshots for visual supplement: {e}")
            images = []

    output_action = await self.llm.aask(
        prompt_action,
        system_msgs=[system_msg],
        images=images,
        stream=False,
    )

    _parse_think_output_v2(self, output_action)

    logger.info(
        f"\n\n######################## output_action (TextAgent v2.1):\n{output_action}\n"
        f"\n######################## output_action end\n\n"
    )
    logger.info(f"#### Assumption: {self.rc.assumption}")
    logger.info(f"#### Confidence: {self.rc.confidence}")
    if visual_mode:
        logger.info(f"📸 Visual mode: ON ({len(images)} images attached)")

    return not self.rc.action.startswith("Stop")


# ── Parser for 4-section output ──

def _parse_think_output_v2(self, output_action: str) -> None:
    """Parse the 4-section LLM output (Changes / Thought / Action / Operation).

    Also extracts embedded Assumption and Confidence from the Thought section.
    Populates the same rc.* fields as v1 for framework compatibility.
    """
    def _extract(text: str, start: str, end: str = None) -> str:
        if start not in text:
            return ""
        idx = text.find(start) + len(start)
        if end is not None:
            end_idx = text.find(end, idx)
            if end_idx == -1:
                return text[idx:].strip()
            return text[idx:end_idx].strip()
        return text[idx:].strip()

    # ── Extract 4 sections ──
    changes = _extract(output_action, "### Changes ###", "### Thought ###")
    thought = _extract(output_action, "### Thought ###", "### Action ###")
    action = _extract(output_action, "### Action ###", "### Operation ###")
    operation = _extract(output_action, "### Operation ###")

    # ── Fallback: try v1 section names for backward compat ──
    if not action:
        action = _extract(output_action, "### Action ###")
    if not thought and "### Reflection Thought ###" in output_action:
        # v1 fallback
        thought = _extract(output_action, "### Reflection Thought ###", "### Thought ###")
        thought += " " + _extract(output_action, "### Thought ###", "### Action ###")

    # ── Extract embedded Assumption + Confidence from Thought ──
    assumption = ""
    confidence = 0.0

    # Pattern: "Assumption: can/cannot meet expected result | Confidence: 0.X"
    assumption_match = re.search(
        r"Assumption:\s*(.*?)(?:\||$)", thought, re.IGNORECASE
    )
    if assumption_match:
        assumption = assumption_match.group(1).strip()

    confidence_match = re.search(
        r"Confidence:\s*(\d+\.?\d*)", thought, re.IGNORECASE
    )
    if confidence_match:
        try:
            confidence = max(0.0, min(1.0, float(confidence_match.group(1))))
        except (ValueError, AttributeError):
            confidence = 0.0

    # If no structured assumption found, try to infer from thought text
    if not assumption:
        if "cannot meet" in thought.lower():
            assumption = "cannot meet expected result"
        elif "can meet" in thought.lower():
            assumption = "can meet expected result"
        else:
            assumption = "uncertain"

    # ── Populate rc fields (same names as v1 for compatibility) ──
    self.rc.image_description = changes  # "Changes" replaces "Screen State"
    self.rc.reflection_thought = thought  # Thought includes reflection
    self.rc.thought = thought
    self.rc.action = action
    self.rc.summary = operation

    # Task list: NOT extracted from output — keep the initial task list unchanged
    # self.rc.task_list stays as-is

    self.rc.assumption = assumption
    self.rc.confidence = confidence


# ── Code-level guardrail helpers (FIX-C1/C2/C3) ──

_GAME_KEY_PATTERN = re.compile(
    r"pyautogui\.(?:press|hotkey)\(\s*['\"]?(left|right|up|down|space|w|a|s|d)['\"]?",
    re.IGNORECASE,
)
_TEXT_EDITING_COMBO = re.compile(
    r"pyautogui\.hotkey\([^)]*(?:ctrl|alt|command|shift)",
    re.IGNORECASE,
)


def _is_keyboard_game_action(action: str) -> bool:
    """True if action contains keyboard game inputs (arrows/WASD/space), not text-editing combos."""
    if _TEXT_EDITING_COMBO.search(action):
        return False
    return bool(_GAME_KEY_PATTERN.search(action))


def _has_canvas_element(elements: List[Dict[str, Any]]) -> bool:
    """True if the a11y tree contains a canvas-related element."""
    for el in elements:
        text = (el.get("text") or "").lower()
        ctrl = (el.get("control_type") or "").lower()
        if "canvas" in text or "canvas" in ctrl:
            return True
    return False


def _get_canvas_center(
    elements: List[Dict[str, Any]], width: int, height: int
) -> Tuple[int, int]:
    """Return center of the canvas element, or screen center as fallback."""
    for el in elements:
        text = (el.get("text") or "").lower()
        ctrl = (el.get("control_type") or "").lower()
        if "canvas" in text or "canvas" in ctrl:
            coords = el.get("coordinates", [])
            if len(coords) == 2:
                return coords[0], coords[1]
            elif len(coords) >= 4:
                return (coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2
    return width // 2, height // 2


# ── Act (text-only, simplified) ──

async def _act_text(self) -> AIMessage:
    """Execute action for text mode, including TellVerifier support when enabled."""
    self.run_action_failed = False
    self.run_action_failed_exception = ""

    # ================================================================
    # Code-level guardrails (FIX-C1 / FIX-C2 / FIX-C3)
    # ================================================================

    # FIX-C1+C4: Block early Tell — agent must have ≥3 Run actions before reporting
    if self.rc.action.startswith("Tell"):
        run_count = sum(1 for a in self.rc.action_history if a.startswith("Run"))
        if run_count < 3:
            logger.warning(
                f"🚫 [FIX-C1/C4] Early Tell blocked: {run_count} Run actions < 3 minimum"
            )
            self.rc.action = "Wait (Tell rejected — interact with the page more first)"
            self.rc.error_flag = True
            self.rc.error_message = (
                f"ACTION REJECTED: You tried to report results (Tell) but have only "
                f"executed {run_count} real interactions (Run). You MUST perform at "
                f"least 3 Run actions (click, type, scroll) before reporting. "
                f"Read the element tree, find target elements, and interact NOW. "
                f"Try different approaches if previous attempts didn't work."
            )

    # FIX-C3: Block 3+ consecutive identical Run actions
    if self.rc.action.startswith("Run") and len(self.rc.action_history) >= 2:
        if (
            self.rc.action_history[-1] == self.rc.action
            and self.rc.action_history[-2] == self.rc.action
        ):
            logger.warning(
                f"🔄 [FIX-C3] Blocked identical action repeated 3x: "
                f"{self.rc.action[:80]}"
            )
            self.rc.action = (
                "Wait (Same action repeated 3 times — try a different approach)"
            )
            self.rc.error_flag = True
            self.rc.error_message = (
                "ACTION REJECTED: You repeated the exact same action 3 times. "
                "This clearly isn't working. Try a DIFFERENT approach: "
                "different coordinates, different element, different method, "
                "or scroll/navigate first to find the right target."
            )

    # FIX-C2: Auto-focus canvas before keyboard game actions
    if self.rc.action.startswith("Run") and _is_keyboard_game_action(self.rc.action):
        if _has_canvas_element(self.rc.perception_infos):
            cx, cy = _get_canvas_center(
                self.rc.perception_infos, self.width, self.height
            )
            try:
                import pyautogui
                pyautogui.click(cx, cy)
                time.sleep(0.5)
                logger.info(
                    f"🎮 [FIX-C2] Auto-focused canvas at ({cx}, {cy}) "
                    f"before keyboard action"
                )
            except Exception as e:
                logger.warning(f"[FIX-C2] Canvas auto-focus failed: {e}")

    # ================================================================

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

    # Runtime visual fallback: consecutive action failures → a11y coords unreliable
    if not getattr(self, '_visual_supplement', False) and getattr(self, '_debug_screenshots', True):
        fail_count = getattr(self, '_action_fail_count', 0)
        if self.run_action_failed:
            self._action_fail_count = fail_count + 1
            if self._action_fail_count >= 2:
                self._visual_supplement = True
                logger.info("👁️ 2 consecutive action failures → enabling visual supplement")
                prev_origin = f"{self.save_img}/origin_{self.rc.iter - 1}.jpg"
                if Path(prev_origin).exists():
                    self._prev_screenshot_path = prev_origin
        else:
            self._action_fail_count = 0

    # Save previous elements for diff computation in next _think_text
    self.rc.last_perception_infos = copy.deepcopy(self.rc.perception_infos)

    # Save previous screenshot path for visual diff (before taking new screenshot)
    if getattr(self, '_visual_supplement', False) and getattr(self, '_debug_screenshots', True):
        prev_origin = f"{self.save_img}/origin_{self.rc.iter - 1}.jpg"
        if Path(prev_origin).exists():
            self._prev_screenshot_path = prev_origin

    self.rc.perception_infos, self.width, self.height, self.output_image_path = (
        await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)
    )

    if getattr(self, '_debug_screenshots', True):
        self._save_iteration_images(self.rc.iter)

    # Runtime visual fallback: if tree became sparse after action (e.g. game started),
    # auto-enable visual mode for remaining iterations
    if not getattr(self, '_visual_supplement', False) and getattr(self, '_debug_screenshots', True):
        if _is_sparse_a11y_tree(self.rc.perception_infos, self.instruction):
            self._visual_supplement = True
            # Also save the previous screenshot for visual diff
            prev_origin = f"{self.save_img}/origin_{self.rc.iter - 1}.jpg"
            if Path(prev_origin).exists():
                self._prev_screenshot_path = prev_origin

    # Runtime visual fallback: a11y tree unchanged for 2+ consecutive steps → agent stuck,
    # fall back to screenshot mode so the model can visually locate UI elements
    if not getattr(self, '_visual_supplement', False) and getattr(self, '_debug_screenshots', True):
        prev_els = getattr(self.rc, 'last_perception_infos', None) or []
        if is_tree_unchanged(prev_els, self.rc.perception_infos):
            no_change_count = getattr(self, '_no_change_count', 0) + 1
            self._no_change_count = no_change_count
            if no_change_count >= 2:
                self._visual_supplement = True
                logger.info(
                    f"👁️ {no_change_count} steps with no a11y tree change → enabling visual supplement"
                )
                prev_origin = f"{self.save_img}/origin_{self.rc.iter - 1}.jpg"
                if Path(prev_origin).exists():
                    self._prev_screenshot_path = prev_origin
        else:
            self._no_change_count = 0

    # Reuse OSAgent's Tell verification semantics so TextAgent can correct or retry
    # before the Tell action is committed into history.
    if self.use_tell_verifier and self.rc.action.startswith("Tell"):
        try:
            logger.info("Tell action detected, triggering verification...")
            verification_result = await self.tell_verifier.run(
                tell_content=self.rc.action,
                action_history=self.rc.action_history,
                reflection_history=self.rc.reflection_thought_history,
                screenshot_dir=self.save_img,
                current_iter=self.rc.iter,
                test_cases=getattr(self, 'instruction', ''),
            )

            if verification_result.has_action_error:
                logger.warning(
                    f"Tell action verification found ACTION ERROR ({verification_result.verification_status}), "
                    f"agent will retry with corrective guidance. Reasoning: {verification_result.reasoning}"
                )
                self.rc.error_flag = True
                corrective_guidance = verification_result.get_corrective_guidance() or ""
                action_error = verification_result.action_error
                error_type_desc = (
                    "Interaction Modality Mismatch (W4): Wrong interaction method used"
                    if action_error and action_error.error_type == "W4"
                    else "Mechanics & Focus Failure (W6): Basic operation mistake"
                )
                self.rc.error_message = (
                    f"ACTION ERROR - {error_type_desc}\\n"
                    f"Error Description: {action_error.error_description if action_error else verification_result.reasoning}\\n"
                    f"Required Action: {action_error.required_action if action_error else 'Review task requirements'}\\n"
                    f"Corrective Guidance: {corrective_guidance}\\n"
                    f"Please follow the corrective guidance above to retry the operation correctly."
                )
                self._action_error_detected = True
                self.rc.action = "Wait (Action error detected, retrying with corrective guidance)"
                logger.info(
                    f"Agent will continue with corrective guidance: {corrective_guidance[:200]}..."
                )
            elif verification_result.needs_correction:
                logger.warning(
                    f"Tell action verification found hallucination ({verification_result.verification_status}), "
                    f"correcting action. Reasoning: {verification_result.reasoning[:200]}..."
                )
                self.rc.action = verification_result.corrected_action
                self._action_error_detected = False
                logger.info(f"Corrected Tell action: {self.rc.action[:200]}...")
            else:
                logger.info(
                    f"Tell action verification passed: {verification_result.verification_status}"
                )
                self._action_error_detected = False
        except Exception as e:
            logger.error(
                f"Tell action verification failed with error: {str(e)}, using original action"
            )
            self._action_error_detected = False

    # Append to history lists (same fields as v1 for compatibility)
    self.rc.thought_history.append(self.rc.thought)
    self.rc.summary_history.append(self.rc.summary)
    self.rc.action_history.append(self.rc.action)
    self.rc.assumption_history.append(self.rc.assumption)
    self.rc.confidence_history.append(self.rc.confidence)
    self.rc.memory.append(self.rc.image_description or "")  # "Changes" text
    self.rc.reflection_thought_history.append(self.rc.reflection_thought)

    if self.run_action_failed:
        self.rc.error_message = f"ERROR(run action code failed): {self.run_action_failed_exception}\\n"
        self.rc.error_flag = True
    else:
        self.rc.error_message = ""

    return AIMessage(content=self.rc.action, cause_by=Action)


# ── Main react loop ──

async def _react_text(self) -> AIMessage:
    """Main react loop — text-only version v2."""
    self.rc.iter = 0
    rsp = AIMessage(content="No actions taken yet", cause_by=Action)

    while self.rc.iter < self.max_iters and not self._check_last_three_start_with_wait(
        self.rc.action_history
    ):
        self.rc.iter += 1
        logger.info(f"\n\n\n\n\n\n#### iter:{self.rc.iter} (TextAgent v2)\n\n")

        if self.rc.iter == 1:
            (
                self.rc.perception_infos, self.width, self.height, self.output_image_path,
            ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

            if getattr(self, '_debug_screenshots', True):
                self._save_iteration_images(0)

            # Visual Supplement: LLM auto-detects if visual verification is needed
            if getattr(self, '_debug_screenshots', True):
                self._visual_supplement = await _detect_visual_need(self, self.instruction)

                # Runtime fallback: if a11y tree is almost empty (Canvas/WebGL app),
                # force visual mode regardless of LLM detection
                if not self._visual_supplement:
                    self._visual_supplement = _is_sparse_a11y_tree(
                        self.rc.perception_infos, self.instruction
                    )

                # FIX-C5: Force visual mode when canvas element exists
                # Canvas apps (drawing tools, games) render content as pixels,
                # invisible to a11y tree — agent needs screenshots to see what happened
                if not self._visual_supplement:
                    if _has_canvas_element(self.rc.perception_infos):
                        self._visual_supplement = True
                        logger.info(
                            "🎨 [FIX-C5] Canvas element detected in a11y tree "
                            "→ forcing visual supplement mode"
                        )

            # Task list: generated ONCE, then kept as read-only reference
            self.rc.task_list = await self._generate_initial_task_list(
                self.instruction, self.screenshot_file, None
            )

        # Think (text-only with element diff)
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
        elif getattr(self, "_action_error_detected", False):
            logger.info(
                "Action error detected, continuing loop to retry with corrective guidance"
            )
            self._action_error_detected = False
            continue

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
                current_changes = self.rc.image_description or "Unknown state"
                self.rc.action = (
                    f"Tell (Reached maximum steps ({self.max_iters}). "
                    f"Task may be incomplete. Last changes: {current_changes[:200]})"
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


# ── Initial task list generation ──

async def _generate_initial_task_list_text(
    self, instruction: str, screenshot_file: str = None, screenshot_som_file: str = None
) -> str:
    """Generate initial task list from element tree (no screenshots)."""
    elements_text = _format_elements(self.rc.perception_infos)

    prompt = f"""Based on the instruction and current page elements, generate a concise task plan.

**Instruction:** {instruction}

**Current Page Elements (top entries):**
{elements_text}

Output format:
* **[Current Task]:** <first task>
* **[Remaining Tasks]:**
  * <task 2>
  * ...
"""

    system_msg = (
        self.system_prompt
        if self.system_prompt
        else "You are a helpful AI web testing assistant."
    )

    result = await self.llm.aask(prompt, system_msgs=[system_msg], images=[], stream=False)
    task_list = result.strip()
    logger.info(
        f"\n\n######################## Initial Task List (TextAgent v2):\n{task_list}\n"
        f"\n######################## End of Initial Task List\n\n"
    )
    return task_list


# ── Utility methods ──

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
        return "(No elements detected — page may be loading or empty)"
    lines = [f"  [{el.get('coordinates', ())}] {el.get('text', '')}" for el in elements[:max_elements]]
    result = "\n".join(lines)
    if len(elements) > max_elements:
        result += f"\n  ... ({len(elements) - max_elements} more elements truncated)"
    return result


# Keep TextAgent class as alias for backward compatibility (isinstance checks in docs, etc.)
TextAgent = OSAgent
