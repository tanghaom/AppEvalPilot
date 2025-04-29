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
    Input: app_name in Excel table, Auto Generated Test Cases, prod_url, work_path\n
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
            if pd.isna(app_name):
                continue
            task_list_str = row["Auto Generated Test Cases"]

            try:
                task_list = ast.literal_eval(task_list_str)  # Convert string type list to list
                if not isinstance(task_list, list):
                    task_list = []
            except (ValueError, SyntaxError):
                task_list = []

            # Initialize app entry with empty dict
            sheet_data[app_name] = {"test_cases": {}}

            # Check and add prod_url if available
            if "prod_url" in row and not pd.isna(row["prod_url"]):
                sheet_data[app_name]["url"] = f"{row['prod_url']}"

            # Check and add work_path if available
            if "work_path" in row and not pd.isna(row["work_path"]):
                sheet_data[app_name]["work_path"] = f"{row['work_path']}"

            # Ensure at least one of url or work_path is available
            if not ("url" in sheet_data[app_name] or "work_path" in sheet_data[app_name]):
                # Add empty url as fallback if neither field is present
                sheet_data[app_name]["url"] = ""

            for i, task_desc in enumerate(task_list):
                sheet_data[app_name]["test_cases"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}

        output_data.update(sheet_data)

    write_json_file(json_file, output_data, indent=4)


def mini_list_to_json(excel_file: str, json_file: str) -> None:
    """
    Convert data from Excel file to JSON format and save to JSON file.\n
    Input: app_name in Excel table, Auto Generated Test Cases, prod_url, work_path\n
    Output: Test json with separate entries for each category of test cases
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
            if pd.isna(app_name):
                continue
            task_list_str = row["Auto Generated Test Cases"]

            try:
                # Convert string to list of lists
                task_list = ast.literal_eval(task_list_str)
                if not isinstance(task_list, list):
                    task_list = []
                # Ensure each element is a list
                task_list = [item if isinstance(item, list) else [] for item in task_list]
            except (ValueError, SyntaxError):
                task_list = []

            # Create separate entries for each category
            for i, category_tasks in enumerate(task_list):
                category_app_name = f"{app_name}_{i}"
                # Initialize app entry with empty dict
                sheet_data[category_app_name] = {"test_cases": {}}

                # Check and add prod_url if available
                if "prod_url" in row and not pd.isna(row["prod_url"]):
                    sheet_data[category_app_name]["url"] = f"{row['prod_url']}"

                # Check and add work_path if available
                if "work_path" in row and not pd.isna(row["work_path"]):
                    sheet_data[category_app_name]["work_path"] = f"{row['work_path']}"

                # Ensure at least one of url or work_path is available
                if not ("url" in sheet_data[category_app_name] or "work_path" in sheet_data[category_app_name]):
                    # Add empty url as fallback if neither field is present
                    sheet_data[category_app_name]["url"] = ""

                # Add test cases for this category
                for j, task_desc in enumerate(category_tasks):
                    sheet_data[category_app_name]["test_cases"][str(j)] = {
                        "case_desc": task_desc,
                        "result": "",
                        "evidence": "",
                    }

        output_data.update(sheet_data)

    write_json_file(json_file, output_data, indent=4)


def mini_list_to_excel(json_file: str, excel_file: str) -> None:
    """
    Convert JSON file with categorized test cases back to Excel format.\n
    Merges test cases from different categories back into their original project names.

    Args:
        json_file (str): Path to JSON file.
        excel_file (str): Path to output Excel file.
    """
    data = read_json_file(json_file)
    excel_data = []

    # Group data by original project name (removing the category suffix)
    project_data = {}
    for app_name, details in data.items():
        # Extract original project name by removing the category suffix
        base_name = app_name.rsplit("_", 1)[0]
        if base_name not in project_data:
            project_data[base_name] = []

        # Add test cases for this category
        for case_id, case in details.get("test_cases", {}).items():
            project_data[base_name].append(
                {
                    "project_name": base_name,
                    "case_desc": case.get("case_desc", ""),
                    "result": case.get("result", ""),
                    "evidence": case.get("evidence", ""),
                }
            )

    # Flatten the data into a single list
    for cases in project_data.values():
        excel_data.extend(cases)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_data)
    df.to_excel(excel_file, index=False)


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
                    "case_desc": condition.get("case_desc", ""),
                    "result": condition.get("result", ""),
                    "evidence": condition.get("evidence", ""),
                }
            )

    df = pd.DataFrame(excel_data)
    df.to_excel(excel_file_path, index=False)


def make_json_single(case_name: str, url: str, test_cases: List[str], json_path: str, work_path: str) -> None:
    """
    Convert a single test case to JSON format and save it

    Args:
        case_name: Test case name
        url: Test URL
        test_cases: Test case list
        json_path: Path to output JSON file
        work_path: Test case work path
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    data = {}
    if url:
        data[case_name] = {"url": url, "iters": "", "test_cases": {}}
    else:
        data[case_name] = {"work_path": work_path, "iters": "", "test_cases": {}}
    for i, task_desc in enumerate(test_cases):
        data[case_name]["test_cases"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}
    write_json_file(json_path, data, indent=4)


def update_project_excel_iters(project_excel_path: str, json_file_path: str) -> None:
    """
    Update the project Excel file with iteration counts from the JSON results.

    The JSON structure is expected to have the following format:
    {
        "App Name 1": {
            "url": "https://example.com",
            "test_cases": { ... },
            "iters": 6  # Number of iterations for this app
        },
        "App Name 2": { ... }
    }

    Args:
        project_excel_path (str): Path to the Excel file to update
        json_file_path (str): Path to the JSON file containing app data with iteration counts
    """
    try:
        # Read JSON data
        json_data = read_json_file(json_file_path)

        # Read Excel file
        df = pd.read_excel(project_excel_path)

        # Add iters column if it doesn't exist
        if "iters" not in df.columns:
            df["iters"] = ""

        # Update iters for each app
        for index, row in df.iterrows():
            app_name = row["app_name"]
            if pd.isna(app_name):
                continue

            if app_name in json_data:
                # Update the iters value directly from the JSON structure
                df.at[index, "iters"] = json_data[app_name].get("iters", "")

        # Write back to Excel
        df.to_excel(project_excel_path, index=False)
        logger.info(f"Successfully updated iteration counts in {project_excel_path}")

    except Exception as e:
        logger.error(f"Error updating project Excel file {project_excel_path}: {str(e)}")
        raise


def make_work_path(excel_file: str, project_path: str) -> None:
    """
    Add work_path column to Excel file based on project path and id column.
    Each work_path will be constructed as: project_path/id/start.bat

    Args:
        excel_file (str): Path to Excel file
        project_path (str): Base project path
    """
    try:
        # Read Excel file
        df = pd.read_excel(excel_file)

        # Add work_path column if it doesn't exist
        if "work_path" not in df.columns:
            df["work_path"] = ""

        # Update work_path for each row
        for index, row in df.iterrows():
            if "id" in row and not pd.isna(row["id"]):
                # Construct work_path as project_path/id/start.bat
                work_path = os.path.join(project_path, str(row["id"]), "start.bat")
                df.at[index, "work_path"] = work_path

        # Save back to Excel
        df.to_excel(excel_file, index=False)
        logger.info(f"Successfully added work_path column to {excel_file}")

    except Exception as e:
        logger.error(f"Error processing Excel file {excel_file}: {str(e)}")
        raise
