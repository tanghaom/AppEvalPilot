#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : device_controller.py
@Desc    : Device control utility class for operating Android and PC devices
"""

import re
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pyautogui
import pyperclip
import uiautomator2 as u2

# from metagpt.logs import logger
from loguru import logger

# Windows-only imports guarded to allow Linux/Ubuntu usage
try:
    from pywinauto import Desktop
    from pywinauto.controls.uiawrapper import UIAWrapper
    from pywinauto.win32structures import RECT

    _HAS_PYWINAUTO = True
except Exception:  # pragma: no cover - absence on non-Windows
    Desktop = None  # type: ignore
    UIAWrapper = object  # type: ignore

    class RECT:  # type: ignore
        pass

    _HAS_PYWINAUTO = False

# Linux-only imports (AT-SPI) guarded to allow Windows usage
try:
    import pyatspi  # type: ignore

    _HAS_PYATSPI = True
except Exception:  # pragma: no cover - absence on non-Linux
    _HAS_PYATSPI = False


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
        if self.pc_type in ("linux", "ubuntu"):
            self._open_app_linux(name)
            return

        # Default to Windows behavior
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
        if self.pc_type in ("linux", "ubuntu"):
            if not _HAS_PYATSPI:
                logger.error("pyatspi is not available; please install 'at-spi2-core' and 'python3-pyatspi'.")
                return []
            t1 = time.time()
            try:
                processor = LinuxElementProcessor(location_info, self.max_tokens)
                elements = processor.collect_elements()
                t2 = time.time()
                logger.info(f"Time taken to get Linux screen element info: {t2 - t1} seconds")
                return elements
            except Exception as e:
                logger.error(f"Linux AT-SPI processing failed: {e}")
                return []
        t1 = time.time()
        try:
            if not _HAS_PYWINAUTO:
                logger.error("pywinauto is not available; Windows UI inspection is unavailable on this platform.")
                return []
            # Get all visible non-taskbar windows
            windows = [w for w in Desktop(backend="uia").windows() if w.is_visible() and w.texts() and w.texts()[0] not in ["Taskbar", "Taskbar", ""]]

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

    # -------------------- Linux/Ubuntu helpers --------------------
    def _open_app_linux(self, name: str) -> None:
        """Open application on Linux/Ubuntu.

        Notes:
            - Prefer passing an executable command (e.g., "firefox", "nautilus").
            - If the command is not found, we try "gtk-launch" as a best-effort.
        """
        try:
            cmd_list = shlex.split(name) if name and name.strip() else []
            if not cmd_list:
                logger.error("Empty application name provided for Linux open_app.")
                return

            executable = shutil.which(cmd_list[0])
            if executable:
                subprocess.Popen(cmd_list)
                return

            # Try gtk-launch with desktop id (may succeed for common apps)
            if shutil.which("gtk-launch"):
                try:
                    subprocess.Popen(["gtk-launch", cmd_list[0]])
                    return
                except Exception:
                    pass

            # Fallback: try xdg-open (works for URLs/files; limited for app names)
            if shutil.which("xdg-open"):
                try:
                    subprocess.Popen(["xdg-open", name])
                    return
                except Exception:
                    pass

            logger.error(f"Failed to open '{name}'. Ensure the command exists in PATH or provide a valid desktop id.")
        except Exception as e:
            logger.error(f"Linux open_app failed: {e}")


class LinuxElementProcessor:
    """Linux UI element processor based on AT-SPI (pyatspi).

    Traverse accessible tree and extract visible elements with geometry.
    """

    def __init__(self, location_info: str = "center", max_tokens: int = 1000):
        """Initialize Linux element processor.

        Args:
            location_info: 'center' to return center point, 'bbox' to return bounding box
            max_tokens: maximum token count for element text
        """
        self.location_info = location_info
        self.max_tokens = max_tokens
        self.max_nodes = 3000  # safety cap to avoid excessive traversal
        # Blacklist system UI applications and window managers to avoid desktop components
        self.system_app_blacklist = {
            # Desktop shells
            "gnome-shell",
            "gnome shell",
            # Window managers
            "xfwm4",  # Xfce window manager
            "xfdesktop",  # Xfce desktop manager
            "kwin",
            "kwin_x11",
            "kwin_wayland",  # KDE window manager
            "mutter",  # GNOME 3+ window manager
            "openbox",  # Openbox window manager
            "i3",  # i3 window manager
            "awesome",  # Awesome window manager
            "bspwm",  # bspwm window manager
            "compiz",  # Compiz window manager
            "marco",  # MATE window manager
            "metacity",  # Old GNOME window manager
            "fluxbox",  # Fluxbox window manager
            "enlightenment",  # Enlightenment window manager
            # Desktop panels and system UI
            "xfce4-panel",  # Xfce panel
            "plasma-desktop",  # KDE desktop
            "plasmashell",  # KDE shell
            "lxpanel",  # LXDE panel
            "mate-panel",  # MATE panel
        }
        # Roles that are typically useful for interaction
        self.interactive_roles = {
            # Common UI controls
            "button",
            "push button",
            "toggle button",
            "menu item",
            "menu",
            "combo box",
            "check box",
            "radio button",
            "entry",
            "text",
            "password text",
            "scroll bar",
            "slider",
            "spin button",
            "tab",
            "page tab",
            "toolbar",
            # Web/document elements (important for Firefox)
            "link",
            "hyperlink",
            "heading",
            "paragraph",
            "section",
            "article",
            "document web",
            "document frame",
            "embedded",
            "internal frame",
            # Lists and tables
            "list item",
            "list",
            "tree item",
            "tree",
            "table",
            "table cell",
            "cell",
            "row",
            "column header",
            # Additional interactive elements
            "image",
            "canvas",
            "label",
            "icon",
            "form",
            "panel",
            "layered pane",
        }

    def collect_elements(self) -> List[Dict]:
        """Collect elements from the active (foreground) window only.

        Returns:
            List of dicts with 'coordinates' and 'text'
        """
        elements: List[Dict] = []
        try:
            desktop = pyatspi.Registry.getDesktop(0)
        except Exception as e:
            logger.error(f"Failed to get desktop from AT-SPI: {e}")
            return elements

        # Get all available windows
        all_frames = list(self._iter_top_level_frames(desktop))

        # Print debug information: show all available windows
        logger.info("=== Window Debug Information ===")
        logger.info(f"Found {len(all_frames)} top-level windows")
        for idx, frame in enumerate(all_frames, 1):
            try:
                title = self._get_window_title(frame)
                app_name = self._get_application_name(frame)
                role = frame.getRoleName()
                states = self._get_state_set_safe(frame)

                # Collect state information
                state_info = []
                if states is not None:
                    if self._state_contains(states, "STATE_ACTIVE"):
                        state_info.append("ACTIVE")
                    if self._state_contains(states, "STATE_FOCUSED"):
                        state_info.append("FOCUSED")
                    if self._state_contains(states, "STATE_SHOWING"):
                        state_info.append("SHOWING")
                    if self._state_contains(states, "STATE_VISIBLE"):
                        state_info.append("VISIBLE")
                    if self._state_contains(states, "STATE_ICONIFIED"):
                        state_info.append("ICONIFIED")

                is_valid = self._is_valid_root(frame)
                is_valid_fallback = self._is_valid_root_fallback(frame)
                is_system = self._is_system_ui(frame)

                logger.info(f"  [{idx}] Title='{title}' | App='{app_name}' | Role={role}")
                logger.info(
                    f"       State=[{', '.join(state_info) if state_info else 'None'}] | "
                    f"Valid={is_valid} | Fallback={is_valid_fallback} | SystemUI={is_system}"
                )
            except Exception as e:
                logger.warning(f"  [{idx}] Unable to get window information: {e}")
        logger.info("===================")

        # Select active window or fallback to browser/first valid window
        root = None
        active_frame = self._get_active_frame(desktop)

        # Print active window detection results
        if active_frame is not None:
            logger.info(f"Active window detected: '{self._get_window_title(active_frame)}' (App: {self._get_application_name(active_frame)})")
            logger.info(f"  Is active window valid: {self._is_valid_root(active_frame)}")
        else:
            logger.warning("No active window detected, using fallback strategy")

        if active_frame is not None and self._is_valid_root(active_frame):
            root = active_frame
            logger.info(f"✓ Using active window: '{self._get_window_title(root)}'")
        else:
            # First try browser windows, then any valid window
            browser_apps = {"firefox", "chrome", "chromium", "brave", "edge", "safari", "opera"}
            logger.info("Trying to find browser windows...")
            for frame in all_frames:
                if self._is_valid_root_fallback(frame):
                    app_name = self._get_application_name(frame).lower()
                    logger.debug(f"  Checking window: '{self._get_window_title(frame)}' (App: {app_name})")
                    if any(browser in app_name for browser in browser_apps):
                        root = frame
                        logger.info(f"✓ Found browser window: '{self._get_window_title(root)}' (App: {app_name})")
                        break

            if root is None:
                logger.info("No browser window found, using first valid window...")
                for frame in all_frames:
                    if self._is_valid_root_fallback(frame):
                        root = frame
                        logger.info(f"✓ Using fallback window: '{self._get_window_title(root)}' (App: {self._get_application_name(root)})")
                        break

        if root is None:
            logger.error("No valid window found to extract elements from")
            logger.warning("Tip: Some applications require an accessibility client (like Orca) to be running")
            logger.warning("Start Orca with: orca &")
            return elements

        visible_bounds = self._get_extents_bounds(root)
        elements.extend(self._process_accessible(root, visible_bounds, 0))
        logger.info(f"Collected {len(elements)} elements from active window")
        return elements

    # ---------------- Internal helpers ----------------
    def _iter_top_level_frames(self, desktop) -> List[object]:
        """Iterate over all top-level window frames from all applications."""
        for i in range(desktop.childCount):
            app = desktop.getChildAtIndex(i)
            if app is None:
                continue
            for j in range(getattr(app, "childCount", 0)):
                win = app.getChildAtIndex(j)
                try:
                    if win:
                        role = win.getRoleName().lower()
                        if role in ("frame", "window"):
                            yield win
                except Exception:
                    continue

    def _get_active_frame(self, desktop):
        """Get the currently active/focused window frame.

        Tries multiple strategies to find the foreground window:
        1. Follow focus to find parent frame
        2. Find frame with ACTIVE state
        3. Find frame with FOCUSED state

        Returns the first valid match found.
        """
        # Try direct focus API first
        try:
            if hasattr(desktop, "focus"):
                focused = desktop.focus
            elif hasattr(desktop, "get_focus"):
                focused = desktop.get_focus()
            else:
                focused = None

            if focused is not None:
                # Walk up to the top-level frame
                acc = focused
                for _ in range(10):
                    try:
                        role = acc.getRoleName().lower()
                    except Exception:
                        break
                    if role in ("frame", "window"):
                        # Use only if the frame is a valid, visible, non-minimized, non-system UI window
                        if self._is_valid_root(acc):
                            return acc
                        break
                    parent = getattr(acc, "parent", None)
                    if parent is None or parent is acc:
                        break
                    acc = parent
        except Exception:
            pass

        # Fallback 1: find frame with ACTIVE state (most reliable for active window)
        for frame in self._iter_top_level_frames(desktop):
            try:
                states = self._get_state_set_safe(frame)
                if self._state_contains(states, "STATE_ACTIVE") and self._is_valid_root(frame):
                    return frame
            except Exception:
                continue

        # Fallback 2: find frame with FOCUSED state
        for frame in self._iter_top_level_frames(desktop):
            try:
                states = self._get_state_set_safe(frame)
                if self._state_contains(states, "STATE_FOCUSED") and self._is_valid_root(frame):
                    return frame
            except Exception:
                continue

        logger.warning("Could not find active frame")
        return None

    def _get_window_title(self, acc) -> str:
        """Get window title for debugging purposes."""
        try:
            return acc.name or "(untitled)"
        except Exception:
            return "(unknown)"

    # ------ root/window filtering helpers ------
    def _is_valid_root(self, acc) -> bool:
        """Check if accessible is a valid top-level window to traverse.

        Criteria:
        - Role is 'frame' or 'window'
        - Not minimized/iconified and not offscreen
        - Visible/showing
        - Not part of system UI applications (e.g., GNOME Shell)
        """
        try:
            role = (acc.getRoleName() or "").lower()
        except Exception:
            role = ""
        if role not in ("frame", "window"):
            return False

        states = self._get_state_set_safe(acc)

        # If we have state info, check it
        if states is not None:
            # Must not be minimized
            if self._state_contains(states, "STATE_ICONIFIED"):
                return False
            # Prefer showing/visible, but don't strictly require it (some apps don't set these)
            self._state_contains(states, "STATE_SHOWING")
            self._state_contains(states, "STATE_VISIBLE")
            # If neither showing nor visible, it might still be valid if no state info is reliable
            # So we don't filter it out here

        # Exclude system UI windows (e.g., GNOME Shell popups/status menus)
        if self._is_system_ui(acc):
            return False

        return True

    def _is_valid_root_fallback(self, acc) -> bool:
        """Relaxed root validation for fallback selection.

        Criteria:
        - Role is 'frame' or 'window'
        - Not minimized/iconified
        - Not a system UI window
        - Prefer showing/visible if state info is available (but not strictly required)
        """
        try:
            role = (acc.getRoleName() or "").lower()
        except Exception:
            role = ""
        if role not in ("frame", "window"):
            return False

        if self._is_system_ui(acc):
            return False

        states = self._get_state_set_safe(acc)

        if states is not None:
            if self._state_contains(states, "STATE_ICONIFIED"):
                return False
            # If it is showing/visible that's great; otherwise still allow as fallback
        return True

    def _is_system_ui(self, acc) -> bool:
        """Return True if the accessible belongs to a system UI application or window manager.

        This includes desktop shells, window managers, panels, and other desktop environment components.
        """
        app_name = self._get_application_name(acc)
        name_l = (app_name or "").lower().strip()

        # Check against blacklist (exact match or substring)
        for blacklisted in self.system_app_blacklist:
            if blacklisted in name_l or name_l == blacklisted:
                return True

        return False

    def _is_system_application(self, app) -> bool:
        """Return True if given application accessible is a system UI or window manager."""
        try:
            name = (app.name or "").lower().strip()
        except Exception:
            name = ""

        # Check against blacklist (exact match or substring)
        for blacklisted in self.system_app_blacklist:
            if blacklisted in name or name == blacklisted:
                return True

        return False

    def _get_application_name(self, acc) -> str:
        """Best-effort to retrieve application name for an accessible node."""
        # Try direct API
        try:
            app = acc.getApplication()
            if app is not None:
                try:
                    return app.name or ""
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: walk up to 'application' role
        try:
            ancestor = acc
            for _ in range(20):
                if ancestor is None:
                    break
                try:
                    role = (ancestor.getRoleName() or "").lower()
                except Exception:
                    role = ""
                if role == "application":
                    try:
                        return ancestor.name or ""
                    except Exception:
                        return ""
                ancestor = getattr(ancestor, "parent", None)
        except Exception:
            pass
        return ""

    def _process_accessible(self, acc, visible_bounds, depth: int = 0) -> List[Dict]:
        """Depth-first traversal limited to the active window's visible bounds.

        - Desktop coordinates only; nodes without valid geometry are skipped.
        - Only include elements that are actually SHOWING and VISIBLE (stricter filtering).
        - System UI windows are already filtered at root.
        - For browsers: skip inactive tab pages to avoid element mixing.

        Args:
            acc: Accessible object to process
            visible_bounds: Bounding box of visible area
            depth: Current recursion depth
        """
        results: List[Dict] = []
        if acc is None:
            return results

        # Safety depth/limit
        if depth > 30 or len(results) >= self.max_nodes:
            return results

        states = self._get_state_set_safe(acc)

        # Geometry (desktop coords only)
        rect = self._get_extents_bounds(acc)
        has_valid_rect = False
        coordinates = None
        rect_str = "(unknown)"

        try:
            role = acc.getRoleName()
        except Exception:
            role = "Unknown"
        try:
            name = acc.name or ""
        except Exception:
            name = ""

        if rect is not None:
            left, top, right, bottom = rect
            width = max(0, right - left)
            height = max(0, bottom - top)
            if width > 0 and height > 0 and self._is_within_bounds(rect, visible_bounds):
                has_valid_rect = True
                coordinates = (left + width // 2, top + height // 2) if self.location_info == "center" else (left, top, right, bottom)
                rect_str = f"({left}, {top}, {right}, {bottom})"

        # Inclusion criteria for visible window elements:
        # 1. Must have valid rect within bounds
        # 2. Must not be explicitly offscreen
        # 3. Should be showing/visible if state info is available
        # 4. Must be interactive role OR have meaningful name
        include_node = False
        role_l = (role or "").lower()

        # More balanced state check: filter out obviously invisible elements
        state_ok = False
        if states is None:
            # If no state info available, accept by default (trust geometry check)
            state_ok = True
        else:
            # Check visibility states
            is_showing = self._state_contains(states, "STATE_SHOWING")
            is_visible = self._state_contains(states, "STATE_VISIBLE")
            is_offscreen = self._state_contains(states, "STATE_OFFSCREEN")

            # Element should be showing OR visible (at least one)
            # Only filter out if explicitly offscreen
            if is_offscreen:
                state_ok = False
            elif is_showing or is_visible:
                # Explicitly showing or visible - accept
                state_ok = True
            else:
                # No explicit visibility state - check if it's at least sensitive/enabled
                # This handles cases where Firefox or other apps don't set SHOWING/VISIBLE
                is_enabled = self._state_contains(states, "STATE_ENABLED")
                is_sensitive = self._state_contains(states, "STATE_SENSITIVE")
                is_focusable = self._state_contains(states, "STATE_FOCUSABLE")
                state_ok = is_enabled or is_sensitive or is_focusable

        if has_valid_rect and state_ok and (role_l in self.interactive_roles or bool(name)):
            include_node = True

        if include_node and coordinates is not None:
            truncated = self._truncate_text(name)

            # Skip elements without text unless they are truly interactive controls
            # Layout containers (section, panel) without text are useless for automation
            always_useful_roles = {
                "button",
                "push button",
                "toggle button",
                "link",
                "hyperlink",
                "entry",
                "text",
                "password text",
                "check box",
                "radio button",
                "combo box",
                "list",
                "list item",
                "menu item",
                "menu",
                "tab",
                "page tab",
            }

            # Skip if: no text AND not in always-useful roles
            if not truncated and role_l not in always_useful_roles:
                # Skip layout containers and decorative elements without text
                pass
            else:
                results.append(
                    {
                        "coordinates": coordinates,
                        "text": f"text:{truncated}; control_type:{role}; rect: {rect_str}",
                    }
                )

        # Recurse into children
        try:
            child_count = getattr(acc, "childCount", 0)
            for i in range(child_count):
                child = acc.getChildAtIndex(i)

                # Skip inactive browser tabs/documents to avoid element mixing
                if child is not None and self._should_skip_inactive_tab(child):
                    continue

                results.extend(self._process_accessible(child, visible_bounds, depth + 1))
                if len(results) >= self.max_nodes:
                    break
        except Exception:
            pass

        return results

    def _should_skip_inactive_tab(self, acc) -> bool:
        """Check if this accessible is an inactive browser tab/document that should be skipped.

        In browsers (Firefox, Chrome, etc.), each tab is represented as a document.
        Only the active tab's document should be SHOWING/VISIBLE.
        This prevents mixing elements from multiple tabs.

        Args:
            acc: Accessible object to check

        Returns:
            True if this is an inactive tab that should be skipped
        """
        try:
            role = acc.getRoleName().lower()
        except Exception:
            role = ""

        try:
            getattr(acc, "name", None) or "(unnamed)"
        except Exception:
            pass

        # Check if this is a browser document/tab container
        # Common roles: "document web", "document frame", "page tab", "panel" (in some browsers)
        is_document_container = role in (
            "document web",
            "document frame",
            "page tab",
            "page tab list",  # Tab container in some browsers
            "panel",  # Some browsers use panel for tab content
        )

        if not is_document_container:
            return False

        # For document containers, check if they are actually visible
        states = self._get_state_set_safe(acc)
        if states is None:
            return False

        # Check visibility states
        is_showing = self._state_contains(states, "STATE_SHOWING")
        is_offscreen = self._state_contains(states, "STATE_OFFSCREEN")

        # Skip inactive browser tabs based on role-specific logic
        if role in ("document web", "document frame"):
            # For document containers: only SHOWING documents are active tabs
            should_skip = not is_showing or is_offscreen
        elif role == "panel":
            # Panels are used for various UI elements in browsers
            should_skip = not is_showing or is_offscreen
        elif role in ("page tab", "page tab list"):
            # Tab UI elements - don't skip
            should_skip = False
        else:
            # Other document containers
            is_visible = self._state_contains(states, "STATE_VISIBLE")
            should_skip = is_offscreen or (not is_showing and not is_visible)

        return should_skip

    def _get_extents_bounds(self, acc):
        """Return bounds in desktop coordinates. No Wayland/window fallback."""
        try:
            comp = acc.queryComponent()
            try:
                x, y, w, h = comp.getExtents(pyatspi.DESKTOP_COORDS)
            except TypeError:
                e = comp.getExtents(pyatspi.DESKTOP_COORDS)
                x, y, w, h = int(getattr(e, "x", 0)), int(getattr(e, "y", 0)), int(getattr(e, "width", 0)), int(getattr(e, "height", 0))
            if int(w) <= 0 or int(h) <= 0:
                return None
            return (int(x), int(y), int(x + w), int(y + h))
        except Exception:
            return None

    def _is_within_bounds(self, rect, bounds) -> bool:
        """Check if rect is within bounds.

        Args:
            rect: Element rectangle (left, top, right, bottom)
            bounds: Window bounds (left, top, right, bottom)

        Returns:
            True if rect is within or overlaps with bounds, False otherwise
        """
        if rect is None:
            return False  # No valid rect means not visible
        if bounds is None:
            return True  # No bounds constraint means accept all
        l, t, r, b = rect
        L, T, R, B = bounds
        # Check if rectangles overlap (element is at least partially visible)
        return not (r <= L or l >= R or b <= T or t >= B)

    def _get_state_set_safe(self, acc):
        """Safely get state set from accessible object.

        Args:
            acc: Accessible object

        Returns:
            StateSet object or None if failed
        """
        if acc is None:
            return None
        try:
            # pyatspi uses get_state_set() method (note the underscore)
            if hasattr(acc, "get_state_set"):
                state_set = acc.get_state_set()
                # state_set should be an Atspi.StateSet object
                return state_set
            elif hasattr(acc, "getState"):
                return acc.getState()
            else:
                # No state method available
                return None
        except Exception as e:
            logger.debug(f"Failed to get state set: {e}")
            return None

    def _state_contains(self, state_set, state_name: str) -> bool:
        """Check if state set contains a specific state.

        Args:
            state_set: StateSet object (can be None)
            state_name: State name like 'STATE_ACTIVE'

        Returns:
            True if state is contained, False otherwise
        """
        if state_set is None:
            return False
        try:
            # Get state constant from pyatspi (e.g., pyatspi.STATE_ACTIVE)
            state = getattr(pyatspi, state_name, None)
            if state is None:
                return False
            # StateSet.contains() method checks if state is in the set
            if hasattr(state_set, "contains"):
                return state_set.contains(state)
            else:
                # Fallback: check if state is in the state_set directly
                return state in state_set
        except Exception as e:
            logger.debug(f"Failed to check state {state_name}: {e}")
            return False

    # ---- text truncation for Linux ----
    def _truncate_text(self, text: str) -> str:
        if not text:
            return text
        if self._estimate_token_count(text) <= self.max_tokens:
            return text
        return self._smart_truncate(text)

    def _smart_truncate(self, text: str) -> str:
        tokens = self._tokenize_mixed_text(text)
        result_tokens: List[str] = []
        current = 0
        for token in tokens:
            if token.isspace():
                if current < self.max_tokens:
                    result_tokens.append(token)
                continue
            cost = 0 if token.isspace() else 1
            if current + cost <= self.max_tokens:
                result_tokens.append(token)
                current += cost
            else:
                break
        out = "".join(result_tokens).rstrip()
        return out + "..." if out != text else out

    def _tokenize_mixed_text(self, text: str) -> list:
        import re as _re

        pattern = r"[\u4e00-\u9fff]|[a-zA-Z0-9]+|[^\u4e00-\u9fff\w\s]|\s+"
        return _re.findall(pattern, text)

    def _estimate_token_count(self, text: str) -> int:
        return sum(0 if t.isspace() else 1 for t in self._tokenize_mixed_text(text))


class WindowsElementProcessor:
    """Windows UI element processor class

    Used for analyzing and processing UI elements in Windows windows.
    """

    def __init__(self, visible_rect: RECT, location_info: str = "center", max_tokens: int = 50):
        """Initialize Windows element processor

        Args:
            visible_rect (RECT): Visible area rectangle
            location_info (str): Location information format, can be 'center' or 'bbox'
            max_tokens (int): Maximum token count for text, defaults to 50 tokens
        """
        self.visible_rect = visible_rect
        self.location_info = location_info
        self.max_tokens = max_tokens
        self.SPECIAL_CONTROL_TYPES = {"Hyperlink", "TabItem", "Button", "ComboBox", "ScrollBar", "Edit", "ToolBar"}

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

    def process_element(self, element: UIAWrapper, depth: int = 0) -> List[Dict[str, Union[Tuple[int, ...], str]]]:
        """Process UI element

        Args:
            element (UIAWrapper): UI element to process
            depth (int): Recursion depth, defaults to 0

        Returns:
            List[Dict[str, Union[Tuple[int, ...], str]]]: List of element information, each containing coordinates and text
        """
        current_elements_info = []
        rect = element.rectangle()

        if element.friendly_class_name() == "TitleBar":
            return current_elements_info

        control_type = element.element_info.control_type
        text = element.window_text()

        if rect.width() > 0 and rect.height() > 0 and self._is_element_visible(rect):
            if element.is_enabled():
                coordinates = self._calculate_coordinates(rect)
                rect_str = f"({rect.left}, {rect.top}, {rect.right}, {rect.bottom})"
                # Truncate text to avoid overly long content affecting LLM inference
                truncated_text = self._truncate_text(text)
                current_elements_info.append(
                    {
                        "coordinates": coordinates,
                        "text": f"text:{truncated_text}; control_type:{control_type}; rect: {rect_str}",
                    }
                )

        for child in element.children():
            child_text = child.window_text()
            if not (child.element_info.control_type == "Edit" and child_text and child_text == text):
                current_elements_info.extend(self.process_element(child, depth + 1))

        return current_elements_info

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
        """Create controller by platform.

        Supported platforms: "Android", "Windows", "Linux", "Ubuntu".
        """
        if platform == "Android":
            self.controller = AndroidController()
        elif platform == "Windows":
            kwargs["pc_type"] = "windows"
            self.controller = PCController(**kwargs)
        elif platform in ("Linux", "Ubuntu"):
            kwargs["pc_type"] = "linux"
            self.controller = PCController(**kwargs)
        else:
            raise ValueError(f"Unsupported device type: {platform}")

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
    if platform not in ["Android", "Windows", "Linux", "Ubuntu"]:
        raise ValueError(f"Unsupported device type: {platform}")
    return ControllerTool(platform, **kwargs)
