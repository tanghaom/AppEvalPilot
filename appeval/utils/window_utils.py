#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@File    : window_utils.py
@Desc    : Window control and browser automation utilities
"""
import os
import subprocess
from typing import List, Optional

import psutil
from metagpt.logs import logger
from pywinauto import Desktop
from pywinauto.application import WindowSpecification


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
    target_url: str, app_path: str = "C:/Program Files/Google/Chrome/Application/chrome.exe"
) -> int:
    """
    Start browser with accessibility and remote debugging enabled.

    Args:
        target_url: URL to open in browser
        app_path: Path to browser executable, defaults to Chrome

    Returns:
        int: Process ID (PID) of the started browser process
    """
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Browser executable not found at: {app_path}")

    cmd = f'"{app_path}" --force-renderer-accessibility --remote-debugging-port=9222 {target_url}'

    process = subprocess.Popen(cmd)
    return process.pid


async def kill_windows(target_names: List[str]) -> Optional[List[WindowSpecification]]:
    """
    Find and close windows matching the target names.

    Args:
        target_names: List of window names to match and close

    Returns:
        Optional[List[WindowSpecification]]: List of windows that couldn't be closed, or None if successful
    """
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
            # Use psutil to ensure the process and its children are all terminated
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        else:  # Linux/Unix system
            cmd = f"kill {pid}"
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        logger.info(f"Process {pid} killed")
        return True
    except Exception as e:
        logger.error(f"Error killing process: {str(e)}")
        return False
