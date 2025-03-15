#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for window_utils.py
"""
import os
import sys
import asyncio
from typing import List

# Add the project root to sys.path to make imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from appeval.utils.window_utils import (
    match_name,
    start_windows,
    kill_windows,
    kill_process
)
from metagpt.logs import logger

async def test_start_windows():
    """Test the start_windows function."""
    logger.info("Testing start_windows function...")
    
    try:
        # Test with default Chrome path and a test URL
        url = "https://www.example.com"
        pid = await start_windows(target_url=url)
        logger.info(f"Browser started with PID: {pid}")
        
        # Give the browser some time to open
        await asyncio.sleep(3)
        
        # Clean up - kill the process
        success = await kill_process(pid)
        logger.info(f"Process cleanup successful: {success}")
    except FileNotFoundError as e:
        logger.error(f"Browser executable not found: {e}")
    except Exception as e:
        logger.error(f"Error in test_start_windows: {e}")
    
    logger.info("start_windows test completed")


async def test_kill_windows():
    """Test the kill_windows function."""
    logger.info("Testing kill_windows function...")
    
    # First, start a browser to test closing
    try:
        # Start a browser window
        url = "https://www.example.com"
        pid = await start_windows(target_url=url)
        logger.info(f"Browser started with PID: {pid}")
        
        # Give the browser some time to open
        await asyncio.sleep(3)
        
        # Now try to kill the window
        target_names = ["Example Domain", "Chrome", "Google Chrome"]
        failed_windows = await kill_windows(target_names)
        
        if failed_windows is None:
            logger.info("All matching windows closed successfully")
        elif not failed_windows:
            logger.info("No matching windows found")
        else:
            logger.warning(f"Failed to close {len(failed_windows)} windows")
        
        # Clean up - kill the process just in case
        await kill_process(pid)
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
    
    # # Test the other functions that require more setup
    # await test_start_windows()
    # await asyncio.sleep(2)  # Add a pause between tests
    
    # await test_kill_windows()
    # await asyncio.sleep(2)  # Add a pause between tests
    
    await test_kill_process()
    
    logger.info("All window_utils tests completed")


if __name__ == "__main__":
    asyncio.run(main()) 