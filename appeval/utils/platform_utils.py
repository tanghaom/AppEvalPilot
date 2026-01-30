#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/29
@File    : platform_utils.py
@Desc    : Platform-agnostic utilities that route to Windows or Linux implementations
"""
import os
from typing import List, Optional

from metagpt.logs import logger

# Detect platform
IS_WINDOWS = os.name == "nt"
IS_LINUX = os.name == "posix"

# Platform name for configuration
CURRENT_PLATFORM = "Windows" if IS_WINDOWS else "Linux"


async def start_browser(
    target_url: str = "",
    work_path: str = "",
    remote_debugging_port: int = 9222,
    **kwargs
) -> int:
    """
    Start browser with remote debugging enabled.
    Routes to platform-specific implementation.

    Args:
        target_url: URL to open in browser
        work_path: Path to executable/script to run
        remote_debugging_port: Chrome remote debugging port
        **kwargs: Additional platform-specific arguments

    Returns:
        int: Process ID (PID) of the started process
    """
    if IS_WINDOWS:
        from appeval.utils.window_utils import start_windows
        return await start_windows(target_url=target_url, work_path=work_path)
    else:
        from appeval.utils.linux_utils import start_browser as linux_start
        return await linux_start(
            target_url=target_url,
            work_path=work_path,
            remote_debugging_port=remote_debugging_port,
            **kwargs
        )


async def kill_browser(
    target_names: List[str] = None,
    port: int = 9222
) -> bool:
    """
    Kill browser processes.
    Routes to platform-specific implementation.

    Args:
        target_names: List of window/process names to match and close
        port: Remote debugging port (used on Linux)

    Returns:
        bool: True if successful
    """
    if IS_WINDOWS:
        from appeval.utils.window_utils import kill_windows
        if target_names is None:
            target_names = ["Chrome"]
        result = await kill_windows(target_names)
        return result is None or len(result) == 0
    else:
        from appeval.utils.linux_utils import kill_chrome_by_port
        return await kill_chrome_by_port(port)


async def kill_process(pid: int) -> bool:
    """
    Terminate a process by PID.
    Routes to platform-specific implementation.

    Args:
        pid: Process ID to terminate

    Returns:
        bool: True if successful
    """
    if IS_WINDOWS:
        from appeval.utils.window_utils import kill_process as win_kill
        return await win_kill(pid)
    else:
        from appeval.utils.linux_utils import kill_process as linux_kill
        return await linux_kill(pid)


async def cleanup_environment(
    is_web: bool = True,
    pid: Optional[int] = None,
    port: int = 9222
) -> None:
    """
    Clean up test environment.
    Routes to platform-specific implementation.

    Args:
        is_web: Whether testing web application
        pid: Optional browser process PID
        port: Remote debugging port
    """
    if IS_WINDOWS:
        from appeval.utils.window_utils import kill_windows, kill_process as win_kill
        processes = ["Chrome"] if is_web else ["Chrome", "cmd", "npm", "projectapp", "Edge"]
        await kill_windows(processes)
        if pid:
            await win_kill(pid)
    else:
        from appeval.utils.linux_utils import cleanup_environment as linux_cleanup
        await linux_cleanup(pid=pid, port=port)


def get_default_platform() -> str:
    """
    Get the default platform name for OSAgent configuration.

    Returns:
        str: "Windows" or "Linux"
    """
    return CURRENT_PLATFORM


logger.info(f"Platform detected: {CURRENT_PLATFORM}")
