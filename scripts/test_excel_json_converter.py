#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : test_excel_json_converter.py
@Desc    : Test script for excel_json_converter
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from appeval.utils.excel_json_converter import update_project_excel_iters


def test_update_project_excel_iters():
    """Test the update_project_excel_iters function"""
    excel_path = "data/4d807943.xlsx"  # Path to your Excel file
    json_path = "data/4d807943_45eab1a_results.json"  # Path to your JSON file
    print(f"Reading from {json_path} and updating {excel_path}...")
    update_project_excel_iters(excel_path, json_path)
    return True


if __name__ == "__main__":
    print("Starting test for update_project_excel_iters function...")
    result = test_update_project_excel_iters()
    print(f"\nTest {'PASSED' if result else 'FAILED'}")
