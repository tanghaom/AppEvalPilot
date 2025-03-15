#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@Author  : tanghaoming
@File    : test_reflection.py
@Desc    : Test script for Reflection action
"""
import argparse
import asyncio

from metagpt.const import TEST_DATA_PATH
from metagpt.logs import logger
from PIL import Image

from appeval.actions.reflection import Reflection


async def test_reflection(
    instruction: str = "Open WeChat",
    last_screenshot_path: str = None,
    current_screenshot_path: str = None,
    platform: str = "Android",
):
    """Test the Reflection action with provided screenshots

    Args:
        instruction: User instruction for the test
        last_screenshot_path: Path to the previous screenshot
        current_screenshot_path: Path to the current screenshot
        platform: Platform type (Android/PC)
    """
    # Create Reflection instance
    reflection_action = Reflection(platform=platform)

    # Use default test images if not provided
    if not last_screenshot_path:
        last_screenshot_path = str(TEST_DATA_PATH / "screenshots" / "android.jpg")
    if not current_screenshot_path:
        current_screenshot_path = str(TEST_DATA_PATH / "screenshots" / "android.jpg")

    # Mock test data
    last_perception_infos = [{"coordinates": (10, 10), "text": "Previous perception info"}]
    perception_infos = [{"coordinates": (20, 20), "text": "Current perception info"}]
    summary = f"I need to {instruction}"
    action = f"Click on the {instruction} icon"
    add_info = ""

    # Get screenshot dimensions
    try:
        with Image.open(current_screenshot_path) as img:
            width, height = img.size
        logger.info(f"Screenshot dimensions: {width}x{height}")
    except Exception as e:
        logger.error(f"Failed to read screenshot dimensions: {str(e)}, using default size 1080x1920")
        width, height = 1080, 1920

    try:
        # Execute reflection
        logger.info("Running reflection analysis...")
        reflect, reflection_thought = await reflection_action.run(
            instruction,
            last_perception_infos,
            perception_infos,
            width,
            height,
            summary,
            action,
            add_info,
            last_screenshot_path,
            current_screenshot_path,
        )

        # Print results
        print(f"Test Instruction: {instruction}")
        print(f"Platform: {platform}")
        print("=" * 50)
        print("\nReflection result:")
        print(f"  {reflect}")
        print("\nReflection thought:")
        print(f"  {reflection_thought}")

        return reflect, reflection_thought

    except Exception as e:
        logger.error(f"Reflection execution failed: {str(e)}")
        return "ERROR", f"Execution failed: {str(e)}"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the Reflection action")
    parser.add_argument("--instruction", type=str, default="Open WeChat", help="User instruction for the test")
    parser.add_argument("--before", type=str, help="Path to the previous screenshot")
    parser.add_argument("--after", type=str, help="Path to the current screenshot")
    parser.add_argument("--platform", type=str, default="Android", choices=["Android", "PC"], help="Platform type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    asyncio.run(
        test_reflection(
            instruction=args.instruction,
            last_screenshot_path=args.before,
            current_screenshot_path=args.after,
            platform=args.platform,
        )
    )
