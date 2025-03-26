#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@File    : window_utils.py
@Desc    : Window control and browser automation utilities
"""
import asyncio
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import psutil
from metagpt.logs import logger
from pywinauto import Desktop
from pywinauto.application import WindowSpecification

# Add CREATE_NO_WINDOW flag import
if os.name == "nt":  # Only import on Windows systems
    CREATE_NO_WINDOW = 0x08000000
else:
    CREATE_NO_WINDOW = 0  # Set to 0 (invalid value) on non-Windows systems


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
    target_url: str = "", app_path: str = "C:/Program Files/Google/Chrome/Application/chrome.exe", work_path: str = ""
) -> int:
    """
    Start browser with accessibility and remote debugging enabled or launch a batch file.

    Args:
        target_url: URL to open in browser, used if provided
        app_path: Path to browser executable, defaults to Chrome
        work_path: Path to batch file or executable to run (e.g., xxx/start.bat)

    Returns:
        int: Process ID (PID) of the started process
    """
    if target_url:
        app_path = Path(app_path)
        if not app_path.exists():
            raise FileNotFoundError(f"Browser executable not found at: {app_path}")

        cmd = (
            f'"{app_path}" --force-renderer-accessibility --remote-debugging-port=9222 --start-fullscreen {target_url}'
        )
    elif work_path:
        work_path = Path(work_path)
        if not work_path.exists():
            raise FileNotFoundError(f"Executable not found at: {work_path}")

        # Get the directory containing the batch file
        work_dir = work_path.parent
        logger.info(f"Working directory: {work_dir}")
        # Change to the directory and then execute the batch file
        if work_dir:
            cmd = f'cd /d "{work_dir}" && "{work_path.name}"'
            logger.info(f"Command: {cmd}")
        else:
            cmd = f'"{work_path}"'
            logger.info(f"Command: {cmd}")
    else:
        raise ValueError("Either target_url or work_path must be provided")

    if os.name == "nt":  # Windows system
        process = subprocess.Popen(cmd, shell=True, creationflags=CREATE_NO_WINDOW)
    else:  # Non-Windows system
        process = subprocess.Popen(cmd, shell=True)
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
