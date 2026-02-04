#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/01/29
@File    : linux_utils.py
@Desc    : Linux-specific browser control and automation utilities
"""
import asyncio
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import psutil
from metagpt.logs import logger

# Try to import undetected_chromedriver
try:
    import undetected_chromedriver as uc
    # Disable UC for now due to Xvfb compatibility issues
    UC_AVAILABLE = False
    logger.info("undetected_chromedriver available but disabled (Xvfb compatibility)")
except ImportError:
    UC_AVAILABLE = False
    logger.warning("undetected_chromedriver not available, using standard Chrome")

# Alternative Chrome paths for Linux
LINUX_CHROME_PATHS = [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
    "/opt/google/chrome/chrome",
]


def find_chrome_path() -> Optional[str]:
    """
    Find Chrome executable path on Linux.

    Returns:
        Optional[str]: Path to Chrome executable, or None if not found
    """
    for path in LINUX_CHROME_PATHS:
        if Path(path).exists():
            logger.info(f"Found Chrome at: {path}")
            return path
    return None


# Global dict to store UC driver instances
_uc_drivers = {}


async def _start_browser_with_uc(
    target_url: str,
    remote_debugging_port: int,
    headless: bool,
    user_data_dir: str,
) -> int:
    """
    Start browser using undetected_chromedriver to bypass Cloudflare.
    
    Args:
        target_url: URL to open
        remote_debugging_port: Chrome debugging port
        headless: Whether to run headless
        user_data_dir: User data directory
        
    Returns:
        int: Process ID of the browser
    """
    import undetected_chromedriver as uc
    
    options = uc.ChromeOptions()
    
    # Basic options
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--window-position=0,0")
    options.add_argument("--force-renderer-accessibility")
    options.add_argument(f"--remote-debugging-port={remote_debugging_port}")
    
    if user_data_dir:
        options.add_argument(f"--user-data-dir={user_data_dir}")
    
    if headless:
        options.add_argument("--headless=new")
    
    logger.info(f"Starting undetected Chrome on port {remote_debugging_port}")
    
    try:
        # Create driver with version_main to match installed Chrome (144)
        driver = uc.Chrome(
            options=options,
            use_subprocess=True,
            version_main=144,  # Match installed Chrome version
        )
        
        # Navigate to URL
        driver.get(target_url)
        
        # Store driver reference to prevent garbage collection
        _uc_drivers[remote_debugging_port] = driver
        
        # Get the browser process PID
        pid = driver.browser_pid if hasattr(driver, 'browser_pid') else driver.service.process.pid
        
        logger.info(f"Undetected Chrome started with PID: {pid}")
        
        # Wait for page to load
        await asyncio.sleep(3)
        
        return pid
        
    except Exception as e:
        logger.error(f"Failed to start undetected Chrome: {e}")
        # Fallback to standard method
        raise


async def start_browser(
    target_url: str = "",
    chrome_path: str = None,
    work_path: str = "",
    remote_debugging_port: int = 9222,
    headless: bool = False,
    user_data_dir: str = None,
) -> int:
    """
    Start browser with accessibility and remote debugging enabled on Linux.
    Uses undetected_chromedriver to bypass Cloudflare detection.

    Args:
        target_url: URL to open in browser
        chrome_path: Path to Chrome executable (auto-detect if not provided)
        work_path: Path to shell script to run (e.g., start.sh)
        remote_debugging_port: Chrome remote debugging port (default: 9222)
        headless: Whether to run Chrome in headless mode
        user_data_dir: Custom user data directory for Chrome profile isolation

    Returns:
        int: Process ID (PID) of the started process
    """
    # Clean up existing Chrome processes on the same port
    await kill_chrome_by_port(remote_debugging_port)

    if target_url:
        # Try using undetected_chromedriver first
        if UC_AVAILABLE:
            return await _start_browser_with_uc(
                target_url, remote_debugging_port, headless, user_data_dir
            )
        
        # Fallback to standard Chrome
        # Find Chrome path
        if chrome_path is None:
            chrome_path = find_chrome_path()
        if chrome_path is None:
            raise FileNotFoundError(
                "Chrome not found. Please install Chrome or specify chrome_path. "
                f"Searched paths: {LINUX_CHROME_PATHS}"
            )

        # Build Chrome command with anti-detection flags
        cmd_parts = [
            chrome_path,
            "--force-renderer-accessibility",
            f"--remote-debugging-port={remote_debugging_port}",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-sandbox",           # Required for running in containers/isolated environments
            "--disable-gpu",          # Disable GPU acceleration for virtual displays
            "--disable-dev-shm-usage", # Overcome limited /dev/shm in containers
            "--window-position=0,0",  # Start window at top-left corner
            "--window-size=1920,1080", # Set exact window size
            # Anti-detection flags to bypass Cloudflare
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--disable-automation",
            "--disable-extensions",
            '--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]

        if headless:
            cmd_parts.append("--headless=new")

        if user_data_dir:
            cmd_parts.append(f"--user-data-dir={user_data_dir}")

        cmd_parts.append(target_url)
        cmd = " ".join(f'"{p}"' if " " in p else p for p in cmd_parts)

    elif work_path:
        work_path = Path(work_path)
        if not work_path.exists():
            raise FileNotFoundError(f"Executable not found at: {work_path}")

        work_dir = work_path.parent
        logger.info(f"Working directory: {work_dir}")
        cmd = f'cd "{work_dir}" && bash "{work_path.name}"'

    else:
        raise ValueError("Either target_url or work_path must be provided")

    logger.info(f"Starting browser with command: {cmd}")

    # Start process
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait a moment for Chrome to start
    await asyncio.sleep(2)

    logger.info(f"Browser started with PID: {process.pid}")
    return process.pid


async def kill_chrome_by_port(port: int = 9222) -> bool:
    """
    Kill Chrome processes listening on the specified debugging port.

    Args:
        port: The remote debugging port to check

    Returns:
        bool: True if any process was killed, False otherwise
    """
    killed = False
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                cmdline_str = ' '.join(cmdline)
                # Check if this is a Chrome process with our debugging port
                if (
                    'chrome' in proc.info.get('name', '').lower()
                    and f'--remote-debugging-port={port}' in cmdline_str
                ):
                    logger.info(f"Killing Chrome process {proc.info['pid']} on port {port}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error killing Chrome by port: {str(e)}")

    return killed


async def kill_chrome_by_name(process_names: List[str] = None) -> bool:
    """
    Kill Chrome/Chromium processes by process name.

    Args:
        process_names: List of process names to kill (default: common Chrome names)

    Returns:
        bool: True if any process was killed, False otherwise
    """
    if process_names is None:
        process_names = ["chrome", "chromium", "chromium-browser", "google-chrome"]

    killed = False
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc_name = proc.info.get('name', '').lower()
                if any(name.lower() in proc_name for name in process_names):
                    logger.info(f"Killing process: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error killing Chrome by name: {str(e)}")

    return killed


async def kill_process(pid: int) -> bool:
    """
    Terminate the specified process on Linux.

    Args:
        pid: Process ID (PID) of the process to terminate

    Returns:
        bool: True if the process was terminated successfully, False otherwise
    """
    try:
        if not psutil.pid_exists(pid):
            logger.warning(f"Process {pid} does not exist")
            return True

        # First try SIGTERM
        cmd = f"kill -15 {pid}"
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

        # Wait for graceful shutdown
        await asyncio.sleep(3)

        # If still exists, use SIGKILL
        if psutil.pid_exists(pid):
            cmd = f"kill -9 {pid}"
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

        logger.info(f"Process {pid} terminated")
        return True

    except Exception as e:
        logger.error(f"Error terminating process: {str(e)}")
        return False


async def cleanup_environment(pid: Optional[int] = None, port: int = 9222) -> None:
    """
    Clean up test environment on Linux.

    Args:
        pid: Optional PID of the browser process to kill
        port: Remote debugging port to clean up
    """
    # Kill by PID if provided
    if pid:
        await kill_process(pid)

    # Also kill any Chrome on the debugging port
    await kill_chrome_by_port(port)

    logger.info("Environment cleanup completed")
