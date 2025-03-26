#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for window_utils.py
"""
import asyncio
import os
import sys

# Add the project root to sys.path to make imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metagpt.logs import logger

from appeval.utils.window_utils import kill_process, kill_windows, start_windows


async def test_start_windows():
    """Test the start_windows function with both URL and work_path scenarios."""
    logger.info("Testing start_windows function...")

    try:
        # Test with work_path
        work_path = r"G:\torch\gpt_pilot\gpt_pilot_workspace_v1\1\start.bat"
        logger.info(f"Testing with work_path: {work_path}")
        pid = await start_windows(work_path=work_path)
        logger.info(f"Process started with PID: {pid}")

        # Give the process some time to start
        # await asyncio.sleep(3)

        # Clean up - kill the process
        # success = await kill_process(pid)
        # logger.info(f"Process cleanup successful: {success}")

        # Test with URL (commented out for now)
        # url = "https://www.example.com"
        # logger.info(f"Testing with URL: {url}")
        # pid = await start_windows(target_url=url)
        # logger.info(f"Browser started with PID: {pid}")
        # await asyncio.sleep(3)
        # success = await kill_process(pid)
        # logger.info(f"Browser cleanup successful: {success}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error in test_start_windows: {e}")

    logger.info("start_windows test completed")


async def test_kill_windows():
    """Test the kill_windows function."""
    logger.info("Testing kill_windows function...")

    # First, start a browser to test closing
    try:
        # Start a browser window
        # url = "https://www.example.com"
        # pid = await start_windows(target_url=url)
        # logger.info(f"Browser started with PID: {pid}")

        # Give the browser some time to open
        # await asyncio.sleep(3)

        # Now try to kill the window
        target_names = ["cmd", "Chrome", "npm", "projectapp", "Edge"]
        failed_windows = await kill_windows(target_names)

        if failed_windows is None:
            logger.info("All matching windows closed successfully")
        elif not failed_windows:
            logger.info("No matching windows found")
        else:
            logger.warning(f"Failed to close {len(failed_windows)} windows")

        # Clean up - kill the process just in case
        # await kill_process(pid)
    except Exception as e:
        logger.error(f"Error in test_kill_windows: {e}")

    logger.info("kill_windows test completed")


async def test_kill_process():
    """Test the kill_process function."""
    logger.info("Testing kill_process function...")

    try:
        # Start a process to test killing
        url = "https://www.example.com"
        pid = await start_windows(target_url=url)
        logger.info(f"Browser started with PID: {pid}")

        # Give the browser some time to open
        await asyncio.sleep(3)

        # Now try to kill the process
        success = await kill_process(pid)
        logger.info(f"Process {pid} killed successfully: {success}")
    except Exception as e:
        logger.error(f"Error in test_kill_process: {e}")

    logger.info("kill_process test completed")


async def main():
    """Run all tests."""
    logger.info("Starting window_utils tests...")

    # Test start_windows with work_path
    # await test_start_windows()
    # await asyncio.sleep(2)  # Add a pause between tests

    await test_kill_windows()
    # await asyncio.sleep(2)  # Add a pause between tests

    # await test_kill_process()

    logger.info("All window_utils tests completed")


if __name__ == "__main__":
    asyncio.run(main())
