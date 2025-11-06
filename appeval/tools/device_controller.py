#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : device_controller.py
@Desc    : Device control utility class for operating Android and PC devices
"""

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pyautogui
import pyperclip
import uiautomator2 as u2
from metagpt.logs import logger
from pywinauto import Desktop
from pywinauto.controls.uiawrapper import UIAWrapper
from pywinauto.win32structures import RECT


class BaseController:
    """Base device controller class

    Provides common functionality for Android and PC controllers.
    """

    def get_screenshot(self, filepath: str = "./screenshot/screenshot.jpg") -> None:
        """Take a screenshot

        Args:
            filepath: Path to save the screenshot
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self._take_screenshot(filepath)
            logger.info(f"Screenshot saved to: {filepath}")
        except Exception as e:
            logger.error(f"Screenshot failed: {str(e)}")

    def _take_screenshot(self, filepath: str) -> None:
        """Implementation method for taking screenshots, to be implemented by subclasses"""
        raise NotImplementedError

    def run_action(self, action: str) -> None:
        """Execute action

        Args:
            action: Action description string
        """
        logger.info(f"Executing action: {action}")
        # Use list to maintain action order
        action_handlers = [
            ("Run", lambda x: hasattr(self, "_handle_run") and self._handle_run(x)),
            ("Tell", lambda x: hasattr(self, "_handle_tell") and self._handle_tell(x)),
        ]

        for action_type, handler in action_handlers:
            if action_type in action:
                handler(action)
                break

    def _handle_tell(self, action: str) -> None:
        """Handle 'Tell' action"""
        # Get text from action
        text = self._extract_code(action)
        logger.info(f"Handling 'Tell' action: {text}")

    def _extract_code(self, action: str) -> str:
        """Extract code from action string

        Args:
            action: Action string

        Returns:
            str: Extracted code
        """
        start = action.find("(")
        end = action.rfind(")")
        if start != -1 and end != -1 and end > start:
            code = action[start + 1 : end]
            return code.strip("```").replace("\n", "; ")
        return ""

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Check if text contains Chinese characters

        Args:
            text: Text to check

        Returns:
            bool: Whether text contains Chinese characters
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)


class AndroidController(BaseController):
    """Android device controller class

    Provides basic operations for Android devices, including clicking, swiping, input, etc.
    """

    def __init__(self):
        """Initialize Android controller"""
        try:
            self.device = u2.connect()  # Connect device
            u2.enable_pretty_logging()
            self.device.set_input_ime(False)  # Switch input method
        except Exception as e:
            logger.error(f"Failed to initialize Android controller: {str(e)}")
            raise

    def _take_screenshot(self, filepath: str) -> None:
        """Implement screenshot function for Android device"""
        self.device.screenshot(filepath)

    def get_screen_xml(self, location_info: str = "center") -> List[Dict]:
        """Get screen XML information

        Args:
            location_info: Location information format ('center' or 'bbox')

        Returns:
            List[Dict]: List containing element information
        """
        result = []
        screen_height = self.device.window_size()[1]
        xml = self.device.dump_hierarchy()
        root = ET.fromstring(xml)

        def get_element_text(element: ET.Element) -> str:
            """Recursively get element text"""
            if element.attrib.get("text"):
                return element.attrib.get("text")
            for child in element:
                text = get_element_text(child)
                if text:
                    return text
            return ""

        for elem in root.iter():
            elem_class = elem.attrib.get("class", "")
            clickable = elem.attrib.get("clickable", "false")
            focusable = elem.attrib.get("focusable", "false")
            elem_text = get_element_text(elem)
            elem_id = elem.attrib.get("resource-id", "")
            elem_desc = elem.attrib.get("content-desc", "")

            bounds = elem.attrib.get("bounds", "")
            if bounds:
                bounds = bounds.replace("][", ",").replace("[", "").replace("]", "")
                bounds = list(map(int, bounds.split(",")))

                if bounds and (bounds[3] - bounds[1]) > screen_height / 2:
                    continue

                if clickable == "true" or (
                    focusable == "true" and (elem_class == "android.widget.EditText" or elem_class == "android.widget.TextView")
                ):
                    center_x = int((bounds[0] + bounds[2]) / 2)
                    center_y = int((bounds[1] + bounds[3]) / 2)

                    result.append(
                        {
                            "coordinates": [center_x, center_y] if location_info == "center" else bounds,
                            "text": f"Class={elem_class}, Text={elem_text}, ID={elem_id}, Content-desc={elem_desc}, Bounds={bounds}",
                        }
                    )

        return result

    def get_all_packages(self) -> List[str]:
        """Get all installed app package names

        Returns:
            List[str]: List of package names
        """
        return self.device.app_list()

    def get_current_app_package(self) -> str:
        """Get current running app's package name

        Returns:
            str: Current app package name
        """
        return self.device.app_current()["package"]

    def open_app(self, package_name: str) -> bool:
        """Launch application

        Args:
            package_name: Application package name

        Returns:
            bool: Whether launch was successful
        """
        package_name = package_name.split(":")[-1].strip()
        try:
            installed_packages = self.get_all_packages()
            if package_name not in installed_packages:
                logger.error(f"App {package_name} is not installed")
                return False

            self.device.app_start(package_name)
            logger.info(f"Successfully launched app: {package_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to launch app: {str(e)}")
            return False

    def _handle_run(self, action: str) -> None:
        """Handle 'Run' action"""
        code = self._extract_code(action)
        code = code.replace("self.device.tap(", "self.device.click(")
        code = self._add_ime_control(code)
        logger.info(f"Executing code: {code}")
        exec(code)

    def _add_ime_control(self, code: str) -> str:
        """Add input method control to code

        Args:
            code: Original code

        Returns:
            str: Code with input method control added
        """
        matches = re.finditer(r'self\.device\.send_keys\("""(.*?)"""(?:, clear=True)?\);', code)
        modified_code = code
        offset = 0

        for match in matches:
            send_keys = match.group(0)
            new_send_keys = f"self.device.set_input_ime(True); time.sleep(0.5); {send_keys} time.sleep(0.5); self.device.set_input_ime(False);"

            start_index = match.start() + offset
            end_index = match.end() + offset

            modified_code = modified_code[:start_index] + new_send_keys + modified_code[end_index:]
            offset += len(new_send_keys) - len(send_keys)

        return modified_code


class PCController(BaseController):
    """PC device controller class

    Provides basic operations for Windows/Mac devices.
    """

    def __init__(
        self,
        search_keys: Tuple[str, str] = ("win", "s"),
        ctrl_key: str = "ctrl",
        pc_type: str = "windows",
        max_tokens: int = 1000,
    ):
        """Initialize PC controller

        Args:
            search_keys: Search shortcut keys
            ctrl_key: Control key
            pc_type: Operating system type
            max_tokens: Maximum token count for UI element text, defaults to 1000 tokens
        """
        try:
            self.search_keys = search_keys
            self.ctrl_key = ctrl_key
            self.pc_type = pc_type.lower()
            self.max_tokens = max_tokens
        except Exception as e:
            logger.error(f"Failed to initialize PC controller: {str(e)}")
            raise

    def _take_screenshot(self, filepath: str) -> None:
        """Implement screenshot function for PC device"""
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)

    def open_app(self, name: str) -> None:
        """Open application

        Args:
            name: Application name
        """
        logger.info(f"Opening application: {name}")
        pyautogui.hotkey(*self.search_keys)
        time.sleep(0.5)

        if self._contains_chinese(name):
            pyperclip.copy(name)
            pyautogui.hotkey(self.ctrl_key, "v")
        else:
            pyautogui.typewrite(name)

        time.sleep(1)
        pyautogui.press("enter")

    def get_screen_xml(self, location_info: str = "center") -> List[Dict]:
        """Get screen element information

        Args:
            location_info: Location information format ('center' or 'bbox')

        Returns:
            List[Dict]: List of element information
        """
        if self.pc_type == "mac":
            logger.warning("Mac OS not supported yet")
            return []
        t1 = time.time()
        try:
            # Get all visible non-taskbar windows
            windows = [w for w in Desktop(backend="uia").windows() if w.is_visible() and w.texts() and w.texts()[0] not in ["任务栏", "Taskbar", ""]]

            if not windows:
                logger.warning("No active window found")
                return []

            active_window = windows[0]  # Get first matching window
            visible_rect = active_window.rectangle()
            t2 = time.time()
            logger.info(f"Time taken to get screen element info: {t2 - t1} seconds")
            processor = WindowsElementProcessor(visible_rect, location_info, self.max_tokens)
            return processor.process_element(active_window)

        except Exception as e:
            logger.error(f"Failed to get screen element info: {str(e)}")
            return []

    def _handle_run(self, action: str) -> None:
        """Handle 'Run' action"""
        code = self._extract_code(action)
        logger.info(f"Executing code: {code}")
        exec(code)


class WindowsElementProcessor:
    """Windows UI element processor class

    Used for analyzing and processing UI elements in Windows windows.
    """

    def __init__(self, visible_rect: RECT, location_info: str = "center", max_tokens: int = 50, max_depth: int = 30, max_nodes: int = 5000):
        """Initialize Windows element processor

        Args:
            visible_rect (RECT): Visible area rectangle
            location_info (str): Location information format, can be 'center' or 'bbox'
            max_tokens (int): Maximum token count for text, defaults to 50 tokens
            max_depth (int): Maximum recursion depth to protect against extremely deep UI trees
            max_nodes (int): Maximum number of nodes to process to avoid huge traversals
        """
        self.visible_rect = visible_rect
        self.location_info = location_info
        self.max_tokens = max_tokens
        self.SPECIAL_CONTROL_TYPES = {"Hyperlink", "TabItem", "Button", "ComboBox", "ScrollBar", "Edit", "ToolBar"}
        # Guards to avoid excessive recursion / traversal
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._visited: Set[Tuple[int, ...]] = set()
        self._processed_nodes: int = 0

    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters

        Args:
            text: Text to check

        Returns:
            bool: Whether text contains Chinese characters
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _truncate_text(self, text: str) -> str:
        """Truncate text based on estimated token count

        Args:
            text (str): Original text

        Returns:
            str: Truncated text with ellipsis if too long
        """
        if not text:
            return text

        # Estimate token count: for English, roughly 1 word = 1 token
        # For Chinese, roughly 1 character = 1 token
        estimated_tokens = self._estimate_token_count(text)

        if estimated_tokens <= self.max_tokens:
            return text

        # Smart truncation for mixed Chinese-English text
        return self._smart_truncate(text)

    def _smart_truncate(self, text: str) -> str:
        """Smart truncation for mixed Chinese-English text

        Args:
            text (str): Input text to truncate

        Returns:
            str: Truncated text with proper handling of mixed content
        """
        # Split text into proper tokens (separate Chinese and English)
        tokens = self._tokenize_mixed_text(text)

        result_tokens = []
        current_token_count = 0

        for token in tokens:
            if token.isspace():
                # Always keep spaces if we haven't exceeded limit
                if current_token_count < self.max_tokens:
                    result_tokens.append(token)
                continue

            # Calculate tokens for this token
            token_count = self._calculate_token_count_for_unit(token)

            if current_token_count + token_count <= self.max_tokens:
                # Can fit the whole token
                result_tokens.append(token)
                current_token_count += token_count
            else:
                # Need to truncate this token
                remaining_tokens = self.max_tokens - current_token_count
                if remaining_tokens > 0:
                    truncated_token = self._truncate_token(token, remaining_tokens)
                    if truncated_token:
                        result_tokens.append(truncated_token)
                break

        result = "".join(result_tokens).rstrip()
        return result + "..." if result != text else result

    def _tokenize_mixed_text(self, text: str) -> list:
        """Tokenize mixed Chinese-English text properly

        Args:
            text (str): Input text to tokenize

        Returns:
            list: List of tokens where Chinese chars and English words are separated
        """
        import re

        # Pattern to match: Chinese characters, English words, or whitespace
        pattern = r"[\u4e00-\u9fff]|[a-zA-Z0-9]+|[^\u4e00-\u9fff\w\s]|\s+"
        tokens = re.findall(pattern, text)
        return tokens

    def _calculate_token_count_for_unit(self, token: str) -> int:
        """Calculate token count for a single unit (should always be 1 after proper tokenization)

        Args:
            token (str): Single token unit

        Returns:
            int: Token count (should be 1 for properly tokenized units)
        """
        if token.isspace():
            return 0  # Spaces don't count as tokens
        return 1  # Each properly tokenized unit counts as 1 token

    def _truncate_token(self, token: str, max_tokens: int) -> str:
        """Truncate a single token

        Args:
            token (str): Token to truncate
            max_tokens (int): Maximum tokens allowed

        Returns:
            str: Truncated token
        """
        if max_tokens <= 0:
            return ""

        if max_tokens >= 1:
            return token  # Single tokens are either kept whole or not at all
        else:
            return ""

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for given text

        Args:
            text (str): Input text

        Returns:
            int: Estimated token count
        """
        if not text:
            return 0

        # Use same tokenization logic as smart truncation for consistency
        tokens = self._tokenize_mixed_text(text)

        total_tokens = 0
        for token in tokens:
            total_tokens += self._calculate_token_count_for_unit(token)

        return total_tokens

    def _get_runtime_id(self, element: UIAWrapper) -> Optional[Tuple[int, ...]]:
        """Safely get a stable runtime_id for visited detection.

        Returns a tuple so it is hashable. May return None if not available.
        """
        try:
            runtime_id = getattr(element.element_info, "runtime_id", None)
            if runtime_id is None:
                return None
            # Some backends return list-like runtime ids
            return tuple(runtime_id)  # type: ignore[arg-type]
        except Exception:
            return None

    def process_element(self, element: UIAWrapper, depth: int = 0) -> List[Dict[str, Union[Tuple[int, ...], str]]]:
        """Process UI element with guards against deep/huge trees.

        - Applies depth limit and total node budget.
        - Uses a visited set keyed by runtime_id to avoid cycles / duplicates.
        - Wraps UIA calls in try/except to tolerate flaky elements.
        """
        results: List[Dict[str, Union[Tuple[int, ...], str]]] = []

        # Depth guard
        if depth > self.max_depth:
            return results

        # Node budget guard
        if self._processed_nodes >= self.max_nodes:
            return results

        # Visited guard (if runtime_id is available)
        rid = self._get_runtime_id(element)
        if rid is not None:
            if rid in self._visited:
                return results
            self._visited.add(rid)

        # Count this node toward the budget as soon as we visit it
        self._processed_nodes += 1
        if self._processed_nodes > self.max_nodes:
            return results

        # Read basic properties safely
        try:
            friendly = element.friendly_class_name()
        except Exception:
            friendly = ""

        if friendly == "TitleBar":
            return results

        # Getting rectangle can be expensive; try/except to avoid hard failures
        try:
            rect = element.rectangle()
        except Exception:
            rect = None  # type: ignore[assignment]

        try:
            control_type = element.element_info.control_type
        except Exception:
            control_type = ""

        try:
            text = element.window_text()
        except Exception:
            text = ""

        try:
            if rect is not None and rect.width() > 0 and rect.height() > 0 and self._is_element_visible(rect) and element.is_enabled():
                coordinates = self._calculate_coordinates(rect)
                rect_str = f"({rect.left}, {rect.top}, {rect.right}, {rect.bottom})"
                truncated_text = self._truncate_text(text)
                results.append(
                    {
                        "coordinates": coordinates,
                        "text": f"text:{truncated_text}; control_type:{control_type}; rect: {rect_str}",
                    }
                )
        except Exception:
            # Ignore elements that may throw due to UIA quirks
            pass

        # Recurse into children with protection
        try:
            for child in element.children():
                # Optional: filter duplicated Edit with same text as parent
                try:
                    child_text = child.window_text()
                    if child.element_info.control_type == "Edit" and child_text and child_text == text:
                        continue
                except Exception:
                    pass

                if self._processed_nodes >= self.max_nodes:
                    break

                child_results = self.process_element(child, depth + 1)
                if child_results:
                    results.extend(child_results)

                if self._processed_nodes >= self.max_nodes:
                    break
        except Exception:
            # If children() fails, skip this branch
            pass

        return results

    def _is_element_visible(self, element_rect: RECT) -> bool:
        """Check if element is visible

        Args:
            element_rect (RECT): Element's rectangle area

        Returns:
            bool: Returns True if element is in visible area, False otherwise
        """
        return not (
            element_rect.right < self.visible_rect.left
            or element_rect.left > self.visible_rect.right
            or element_rect.bottom < self.visible_rect.top
            or element_rect.top > self.visible_rect.bottom
        )

    def _calculate_coordinates(self, rect: RECT) -> Union[Tuple[int, int], Tuple[int, int, int, int]]:
        """Calculate element coordinates

        Args:
            rect (RECT): Element's rectangle area

        Returns:
            Union[Tuple[int, int], Tuple[int, int, int, int]]:
                Returns center point coordinates (x, y) if location_info is 'center'
                Returns bounding box coordinates (left, top, right, bottom) if location_info is 'bbox'
        """
        if self.location_info == "center":
            return ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
        return (rect.left, rect.top, rect.right, rect.bottom)


class ControllerTool:
    """Device control tool class

    Provides unified device control interface, supporting Android and PC devices.
    """

    def __init__(self, platform: str = "Android", **kwargs):
        self.controller = AndroidController() if platform == "Android" else PCController(**kwargs)

    def __getattr__(self, name):
        """Proxy all method calls to specific controller"""
        return getattr(self.controller, name)


def create_controller(platform: str = "Android", **kwargs) -> ControllerTool:
    """Create controller tool instance

    Args:
        platform: Platform type
        **kwargs: Other parameters

    Returns:
        ControllerTool: Controller tool instance

    Raises:
        ValueError: Raised when device type is invalid
    """
    if platform not in ["Android", "Windows"]:
        raise ValueError(f"Unsupported device type: {platform}")
    return ControllerTool(platform, **kwargs)
