#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@File    : window_utils.py
@Desc    : Window control and browser automation utilities
"""
import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import psutil
from metagpt.logs import logger

# Windows-only imports - pywinauto only works on Windows
try:
    if os.name == "nt":
        from pywinauto import Desktop
        from pywinauto.application import WindowSpecification
        _HAS_PYWINAUTO = True
    else:
        Desktop = None  # type: ignore
        WindowSpecification = None  # type: ignore
        _HAS_PYWINAUTO = False
except ImportError:
    Desktop = None  # type: ignore
    WindowSpecification = None  # type: ignore
    _HAS_PYWINAUTO = False

# Add CREATE_NO_WINDOW flag import
if os.name == "nt":  # Only import on Windows systems
    CREATE_NO_WINDOW = 0x08000000
else:
    CREATE_NO_WINDOW = 0  # Set to 0 (invalid value) on non-Windows systems


def _setup_chrome_preferences(user_data_dir: str) -> None:
    """Pre-configure Chrome preferences and system policies to suppress all permission dialogs."""
    # 1. Chrome Preferences (per-profile)
    default_dir = Path(user_data_dir) / "Default"
    default_dir.mkdir(parents=True, exist_ok=True)
    prefs_file = default_dir / "Preferences"

    prefs = {}
    if prefs_file.exists():
        try:
            with open(prefs_file, "r") as f:
                prefs = json.load(f)
        except Exception:
            prefs = {}

    # Auto-allow all permissions (1=allow)
    prefs.setdefault("profile", {})
    prefs["profile"]["default_content_setting_values"] = {
        "notifications": 1, "geolocation": 1,
        "media_stream_camera": 1, "media_stream_mic": 1,
        "midi_sysex": 1, "bluetooth_guard": 1,
        "usb_guard": 1, "serial_guard": 1, "hid_guard": 1,
        "idle_detection": 1, "window_placement": 1,
        "clipboard_read_write": 1, "local_fonts": 1,
        "sensors": 1, "automatic_downloads": 1,
        "insecure_private_network": 1,
        "private_network_request_settings": 1,  # Chrome 120+: "Access devices on local network"
    }
    # 禁用密码保存弹窗
    prefs.setdefault("credentials_enable_service", False)
    prefs.setdefault("credentials_enable_autosignin", False)
    prefs.setdefault("password_manager", {})
    prefs["password_manager"]["enabled"] = False
    prefs["password_manager"]["leak_detection"] = False

    prefs.setdefault("browser", {})
    prefs["browser"]["check_default_browser"] = False
    prefs.setdefault("distribution", {})
    prefs["distribution"]["skip_first_run_ui"] = True
    prefs["distribution"]["show_welcome_page"] = False
    prefs["distribution"]["suppress_first_run_default_browser_prompt"] = True

    # Explicitly allow private network access for all origins
    allow_all = {"setting": 1, "last_modified": "13300000000000000"}
    prefs.setdefault("profile", {}).setdefault("content_settings", {}).setdefault("exceptions", {})
    prefs["profile"]["content_settings"]["exceptions"]["insecure_private_network"] = {"*,*": allow_all}
    prefs["profile"]["content_settings"]["exceptions"]["private_network_request_settings"] = {"*,*": allow_all}

    # Fixed download directory for result verification (export/download checks)
    download_dir = Path(user_data_dir) / "Default" / "Downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    prefs.setdefault("download", {})["default_directory"] = str(download_dir)

    try:
        with open(prefs_file, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write Chrome preferences: {e}")

    # 2. Chrome Local State — disable Private Network Access permission prompt via feature flags
    local_state_file = Path(user_data_dir) / "Local State"
    local_state = {}
    if local_state_file.exists():
        try:
            with open(local_state_file, "r") as f:
                local_state = json.load(f)
        except Exception:
            local_state = {}
    local_state.setdefault("browser", {}).setdefault("enabled_labs_experiments", [])
    existing = local_state["browser"]["enabled_labs_experiments"]
    # Disable private network access related flags (@2 = Disabled)
    pna_flags = [
        "private-network-access-permission-prompt",
        "private-network-access-respect-preflight-results",
    ]
    existing = [e for e in existing if not any(e.startswith(f) for f in pna_flags)]
    for f in pna_flags:
        existing.append(f"{f}@2")
    local_state["browser"]["enabled_labs_experiments"] = existing
    try:
        with open(local_state_file, "w") as f:
            json.dump(local_state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write Chrome Local State: {e}")

    # 3. Chrome Enterprise Policy (system-wide)
    for policy_dir in [
        "/etc/opt/chrome/policies/managed",
        "/etc/chromium/policies/managed",
        "/etc/chromium-browser/policies/managed",
    ]:
        try:
            os.makedirs(policy_dir, exist_ok=True)
            policy = {
                "InsecurePrivateNetworkRequestsAllowed": True,
                "InsecurePrivateNetworkRequestsAllowedForUrls": ["*"],
                "PrivateNetworkAccessRestrictionsEnabled": False,
                "DefaultPrivateNetworkRequestSettings": 1,  # 1=Allow
                "DefaultNotificationsSetting": 1,
                "DefaultGeolocationSetting": 1,
            }
            policy_file = Path(policy_dir) / "appeval_policy.json"
            with open(policy_file, "w") as f:
                json.dump(policy, f, indent=2)
        except Exception:
            pass


def get_download_dir(user_data_dir: str) -> str:
    """Return the Chrome download directory for result verification.
    On Linux we set it in _setup_chrome_preferences to user_data_dir/Default/Downloads.
    On Windows we do not set it, so use system default Downloads.
    """
    if os.name == "nt" or not user_data_dir:
        return os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "Downloads")
    return str(Path(user_data_dir) / "Default" / "Downloads")


def list_new_files_since(directory: str, since_mtime: float) -> List[Path]:
    """List files in directory (non-recursive) with st_mtime >= since_mtime.
    Used to detect new downloads after an export/download action.
    """
    out: List[Path] = []
    try:
        p = Path(directory)
        if not p.is_dir():
            return out
        for f in p.iterdir():
            if f.is_file() and f.stat().st_mtime >= since_mtime:
                out.append(f)
    except Exception as e:
        logger.warning(f"list_new_files_since failed: {e}")
    return out


def match_name(window_name: List[str], patterns: List[str]) -> bool:
    """
    Check if any pattern matches the window name.

    Args:
        window_name: List of window text elements
        patterns: List of patterns to match against

    Returns:
        bool: True if any pattern matches, False otherwise
    """
    if not window_name:
        return False
    name = window_name[0]
    if isinstance(name, str):
        return any(pattern.lower() in name.lower() for pattern in patterns)


async def start_windows(
    target_url: str = "",
    app_path: str = "C:/Program Files/Google/Chrome/Application/chrome.exe",
    work_path: str = "",
    remote_debugging_port: int = 9222,
    user_data_dir: str = "",
) -> int:
    """
    Start browser with accessibility and remote debugging enabled or launch a batch file.

    Args:
        target_url: URL to open in browser, used if provided
        app_path: Path to browser executable, defaults to Chrome
        work_path: Path to batch file or executable to run (e.g., xxx/start.bat)
        remote_debugging_port: Chrome remote debugging port (default 9222)
        user_data_dir: Chrome user data directory for isolating profiles

    Returns:
        int: Process ID (PID) of the started process
    """
    # Pre-configure Chrome preferences & policies (Linux only)
    if user_data_dir and os.name != "nt":
        _setup_chrome_preferences(user_data_dir)

    if target_url:
        if os.name == "nt":
            # Windows
            app_path_obj = Path(app_path)
            if not app_path_obj.exists():
                raise FileNotFoundError(f"Browser executable not found at: {app_path_obj}")
            cmd = (
                f'"{app_path_obj}" --force-renderer-accessibility'
                f' --remote-debugging-port={remote_debugging_port}'
                f' --start-fullscreen {target_url}'
            )
        else:
            # Linux/Mac: find Chrome; prefer non-snap to avoid AppArmor/DBus errors (snap chromium blocks DBus)
            chrome_cmd = None
            candidate_names = [
                "google-chrome-stable", "google-chrome", "chromium-browser", "chromium", "chrome"
            ]
            for name in candidate_names:
                p = shutil.which(name)
                if p:
                    # Prefer non-snap: snap Chromium triggers "AppArmor policy prevents ... DBus.ListActivatableNames"
                    if "/snap/" not in p:
                        chrome_cmd = p
                        break
                    if chrome_cmd is None:
                        chrome_cmd = p  # keep as fallback if no non-snap found
            if not chrome_cmd:
                raise FileNotFoundError("Chrome/Chromium not found.")

            flags = [
                chrome_cmd,
                "--no-sandbox" if os.geteuid() == 0 else "",
                "--no-default-browser-check", "--no-first-run",
                "--force-renderer-accessibility",
                f"--remote-debugging-port={remote_debugging_port}",
                "--remote-allow-origins=*",
                f"--user-data-dir={user_data_dir}" if user_data_dir else "",
                # Xvfb rendering
                "--disable-gpu", "--disable-software-rasterizer", "--disable-dev-shm-usage",
                "--window-size=1920,1080", "--start-maximized",
                # Suppress prompts & dialogs
                "--disable-infobars", "--disable-component-update",
                "--disable-background-networking",
                "--disable-features=PrivateNetworkAccessPermissionPrompt",
                # 禁用密码保存弹窗
                "--password-store=basic",
                "--disable-sync",  # 减少 GCM 等后台注册，避免 DEPRECATED_ENDPOINT 刷屏
                "--disable-features=PrivateNetworkAccessPermissionPrompt,PrivateNetworkAccessNullIpAddress,DialMediaRouteProvider",
                # 避免 SSL 握手失败导致页面/请求不可用（测试环境）
                "--ignore-certificate-errors", "--ignore-ssl-errors",
                target_url,
            ]
            # 彻底抑制「Access other devices on your local network」弹窗（仅测试环境）
            # 需与 --user-data-dir 同时使用；仅当设置环境变量时启用
            if os.environ.get("APPEVAL_CHROME_DISABLE_WEB_SECURITY", "").strip().lower() in ("1", "true", "yes") and user_data_dir:
                flags.insert(-1, "--disable-web-security")
                flags.insert(-1, "--disable-site-isolation-trials")
            cmd = " ".join(f for f in flags if f)
    elif work_path:
        work_path = Path(work_path)
        if not work_path.exists():
            raise FileNotFoundError(f"Executable not found at: {work_path}")
        work_dir = work_path.parent
        logger.info(f"Working directory: {work_dir}")
        if work_dir:
            cmd = f'cd /d "{work_dir}" && "{work_path.name}"'
        else:
            cmd = f'"{work_path}"'
        logger.info(f"Command: {cmd}")
    else:
        raise ValueError("Either target_url or work_path must be provided")

    if os.name == "nt":
        process = subprocess.Popen(cmd, shell=True, creationflags=CREATE_NO_WINDOW)
    else:
        # 抑制 Chrome 的 stderr（SSL handshake、GCM DEPRECATED 等）避免刷屏
        process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return process.pid


async def kill_windows(target_names: List[str]) -> Optional[List]:
    """
    Find and close windows matching the target names.

    Args:
        target_names: List of window names to match and close

    Returns:
        Optional[List]: List of windows that couldn't be closed, or None if successful
    """
    if not _HAS_PYWINAUTO or os.name != "nt":
        # Linux/Unix: Use pkill to kill processes by name
        logger.debug(f"Using pkill to kill processes: {target_names}")
        for name in target_names:
            try:
                # Use pkill to kill processes by name (case-insensitive)
                cmd = f"pkill -f -i '{name}'"
                process = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                logger.info(f"Killed processes matching: {name}")
            except Exception as e:
                logger.error(f"Failed to kill process {name}: {str(e)}")
        return None

    try:
        desktop = Desktop(backend="uia")
        windows = desktop.windows()

        # Log all visible windows for debugging
        logger.debug("Visible windows:")
        for w in windows:
            if w.is_visible() and w.texts():
                logger.debug(f"Window: {w.texts()[0]}")

        # Find matching windows
        matching_windows = [w for w in windows if w.is_visible() and w.texts() and match_name(w.texts(), target_names)]

        if not matching_windows:
            logger.warning(f"No active windows found matching patterns: {target_names}")
            return []

        failed_windows = []
        for window in matching_windows:
            try:
                window_name = window.texts()[0] if window.texts() else "Unknown"
                logger.info(f"Attempting to close window: {window_name}")
                window.close()
                logger.success(f"Successfully closed window: {window_name}")
            except Exception as e:
                logger.error(f"Failed to close window {window_name}: {str(e)}")
                failed_windows.append(window)

        return failed_windows if failed_windows else None

    except Exception as e:
        logger.error(f"Error while killing windows: {str(e)}")
        return []


async def kill_process(pid: int) -> bool:
    """Terminate the specified process

    Args:
        pid: Process ID (PID) of the process to terminate

    Returns:
        bool: True if the process was terminated successfully, False otherwise
    """
    try:
        if os.name == "nt":  # Windows system
            # Use psutil to send termination signal instead of directly killing the process
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    # Use terminate() to send normal termination signal
                    child.terminate()
                    # Give the process some time to gracefully shut down
                    child.wait(timeout=5)
                except:
                    try:
                        # If timeout, force terminate
                        child.kill()
                    except Exception as e:
                        logger.error(f"Error killing child process: {str(e)}")

            try:
                # Do the same for parent process
                parent.terminate()
                parent.wait(timeout=5)
            except:
                try:
                    parent.kill()
                except Exception as e:
                    logger.error(f"Error killing parent process: {str(e)}")
        else:  # Linux/Unix system
            # First try to send SIGTERM signal
            cmd = f"kill -15 {pid}"
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            # Give the process some time to respond to SIGTERM
            await asyncio.sleep(5)

            # Check if the process still exists
            if psutil.pid_exists(pid):
                # If the process still exists, send SIGKILL
                cmd = f"kill -9 {pid}"
                process = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

        logger.info(f"Process {pid} terminated")
        return True
    except Exception as e:
        logger.error(f"Error terminating process: {str(e)}")
        logger.exception(e)
        return False
