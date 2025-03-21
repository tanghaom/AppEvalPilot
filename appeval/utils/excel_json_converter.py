#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@File    : excel_json_converter.py
@Desc    : Excel and JSON format conversion utilities
"""
import ast
import os
from typing import List

import pandas as pd
from metagpt.logs import logger
from metagpt.utils.common import read_json_file, write_json_file


def reset_json_results(json_file_path: str) -> None:
    """
    Reset the result and evidence fields to empty strings for all test cases in the JSON file.

    Args:
        json_file_path: Path to the JSON file to process
    """
    try:
        # Read the JSON file
        data = read_json_file(json_file_path)

        # Iterate through each project
        for project in data.values():
            if "test_cases" in project:
                # Iterate through each test case
                for case in project["test_cases"].values():
                    # Reset result and evidence to empty strings
                    case["result"] = ""
                    case["evidence"] = ""

        # Write back to the JSON file
        write_json_file(json_file_path, data, indent=4)

        logger.info(f"Successfully reset results and evidence in {json_file_path}")

    except Exception as e:
        logger.error(f"Error processing file {json_file_path}: {str(e)}")
        raise


def list_to_json(excel_file: str, json_file: str) -> None:
    """
    Convert data from Excel file to JSON format and save to JSON file.\n
    Input: app_name in Excel table, Auto Generated Test Cases, prod_url\n
    Output: Test json
    Args:
        excel_file (str): Path to Excel file.
        json_file (str): Path to output JSON file.
    """
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names
    output_data = {}

    for sheet_name in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheet_data = {}
        for index, row in df.iterrows():
            app_name = row["app_name"]
            url = row["prod_url"]
            if pd.isna(app_name):
                continue
            task_list_str = row["Auto Generated Test Cases"]

            try:
                task_list = ast.literal_eval(task_list_str)  # Convert string type list to list
                if not isinstance(task_list, list):
                    task_list = []
            except (ValueError, SyntaxError):
                task_list = []

            sheet_data[app_name] = {"url": f"{url}", "test_cases": {}}

            for i, task_desc in enumerate(task_list):
                sheet_data[app_name]["test_cases"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}

        output_data.update(sheet_data)

    write_json_file(json_file, output_data, indent=4)


def convert_json_to_excel(json_file_path: str, excel_file_path: str) -> None:
    """
    Read JSON file and convert it to Excel table.\n

    Args:
        json_file_path (str): Path to JSON file.
        excel_file_path (str): Path to output Excel file.
    """
    data = read_json_file(json_file_path)

    excel_data = []
    for task, details in data.items():
        for check_id, condition in details.get("test_cases", {}).items():
            excel_data.append(
                {
                    "project_name": task,
                    "case_name": "",
                    "case_desc": condition.get("case_desc", ""),
                    "result": condition.get("result", ""),
                    "evidence": condition.get("evidence", ""),
                    "auto_function_detection": "",
                    "console_logs": "",
                }
            )

    df = pd.DataFrame(excel_data)
    df.to_excel(excel_file_path, index=False)


def make_json_single(case_name: str, url: str, test_cases: List[str], json_path: str) -> None:
    """
    Convert a single test case to JSON format and save it

    Args:
        case_name: Test case name
        url: Test URL
        test_cases: Test case list
        json_path: Path to output JSON file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    data = {}
    data[case_name] = {"url": url, "iters": "", "test_cases": {}}
    for i, task_desc in enumerate(test_cases):
        data[case_name]["test_cases"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}
    write_json_file(json_path, data, indent=4)
