#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@Author  : tanghaoming
@File    : test_screen_info_extractor.py
@Desc    : Test script for ScreenInfoExtractor
"""
import asyncio

from metagpt.const import TEST_DATA_PATH

from appeval.actions.screen_info_extractor import ScreenInfoExtractor


async def main():
    """Test the ScreenInfoExtractor functionality"""
    # Create ScreenInfoExtractor instance
    extractor = ScreenInfoExtractor(platform="Android")

    # Test parameters
    screenshot_path = str(TEST_DATA_PATH / "screenshots" / "android.jpg")
    task_content = "Please summarize the main content shown on the interface."

    try:
        # Execute information extraction
        result = await extractor.run(task_content, screenshot_path)
        print("\nExtraction result:")
        print(result)

    except Exception as e:
        print(f"Execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
