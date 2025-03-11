#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : test_icon_detect.py
@Desc    : Test script for icon detection and description tool
"""

import argparse
import asyncio

from metagpt.config2 import Config
from metagpt.const import TEST_DATA_PATH
from metagpt.llm import LLM

from appeval.tools.icon_detect import IconDetectTool


async def main(image_path=None, platform="Android"):
    """
    Test the icon detection and description functionality

    Args:
        image_path: Path to the image for testing
        platform: Device platform (Android or PC)
    """
    # Use default test image if not specified
    if not image_path:
        image_path = str(TEST_DATA_PATH / "screenshots" / "chrome.jpg")

    # Initialize LLM
    llm_config = Config.default()
    llm = LLM(llm_config.llm)

    # Initialize IconDetectTool class
    detector = IconDetectTool(llm)

    # Detection
    coordinates = detector.detect(image_path)
    print(f"Detected {len(coordinates)} icons with coordinates:")
    for i, coord in enumerate(coordinates, 1):
        print(f"  Icon {i}: {coord}")

    # Generate descriptions
    descriptions = await detector.caption(image_path, coordinates, platform)
    print("\nIcon descriptions:")
    for idx, desc in descriptions.items():
        print(f"  Icon {idx}: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test icon detection and description")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument(
        "--platform", type=str, default="Android", choices=["Android", "PC"], help="Device platform (Android or PC)"
    )

    args = parser.parse_args()

    asyncio.run(main(args.image, args.platform))
