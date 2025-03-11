#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : test_ocr.py
@Desc    : Test script for OCR tool functionality
"""

from metagpt.const import TEST_DATA_PATH

from appeval.tools.ocr import OCRTool, ocr_recognize


def main():
    # Test image path
    image_path = str(TEST_DATA_PATH / "screenshots" / "chrome.jpg")

    # Instantiate OCRTool class
    ocr_tool = OCRTool()

    # Test normal OCR recognition
    print("Performing normal OCR recognition...")
    texts, coordinates = ocr_tool.ocr(image_path)
    print("Recognized texts:", texts)
    print("Text coordinates:", coordinates)

    # Test split OCR recognition
    print("\nPerforming split OCR recognition...")
    split_texts, split_coordinates = ocr_tool.ocr(image_path, split=True)
    print("Split recognized texts:", split_texts)
    print("Split text coordinates:", split_coordinates)

    # Test shortcut function
    print("\nTesting shortcut function...")
    quick_texts, quick_coordinates = ocr_recognize(image_path)
    print("Shortcut function recognition results:", quick_texts)
    print("Shortcut function coordinates:", quick_coordinates)


if __name__ == "__main__":
    main()
