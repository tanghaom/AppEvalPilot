import ast
import json
import os
import re
from typing import List, Optional

import pandas as pd
from metagpt.logs import logger


def clean_text(text: str) -> str:
    # Remove leading numbers and dots followed by spaces
    return re.sub(r"^\d+\.\s*", "", str(text))


def get_result_value(result_text: str) -> Optional[int]:
    # Process expected results, convert "Completed"/"Not completed"/"Not implemented" to 1/0/-1
    result_text = clean_text(result_text)
    if result_text == "Completed":
        return 1
    elif result_text == "Not completed":
        return 0
    elif result_text == "Not implemented":
        return -1
    return None


def excel_to_json(input_path: str) -> None:
    # Read Excel file
    df = pd.read_excel(input_path)

    # Get column names and corresponding indices
    columns = df.columns.tolist()
    test_result_col = columns[1]  # B列 - 测试用例的预期结果
    check_result_col = columns[4]  # E列 - 检验条件的预期结果

    # Initialize result dictionary
    result = {}
    current_test_case = {}
    current_checks = {}
    case_index = 0
    check_index = 0
    current_results = []

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        # Check if it's a new test case (check if the first column starts with a number and a dot)
        if str(row["test_cases"]).strip() and re.match(r"1\.", str(row["test_cases"])):
            # If there is a previous test case, save it
            if current_test_case:
                current_test_case["check_conditions"] = current_checks
                if current_results:
                    current_test_case["expected_result"] = f"Tell ({','.join(map(str, current_results))})"
                result[str(case_index)] = current_test_case
                case_index += 1

            # Create a new test case
            current_test_case = {
                "test_cases": "",
                "url": "None",
                "expected_result": "",
                "model_result": "None",
                "check_conditions": {},
            }
            current_checks = {}
            current_results = []
            check_index = 0

        # Process test case
        if not pd.isna(row["test_cases"]):
            test_step = str(row["test_cases"]).strip()
            if current_test_case.get("test_cases"):
                current_test_case["test_cases"] += f" {test_step}"
            else:
                current_test_case["test_cases"] = test_step

            # Process expected results in B column
            if not pd.isna(row[test_result_col]):
                result_value = get_result_value(row[test_result_col])
                if result_value is not None:
                    current_results.append(result_value)

        # Process check conditions
        if not pd.isna(row["check_conditions"]) and not pd.isna(row[check_result_col]):
            clean_condition = clean_text(row["check_conditions"])
            clean_result = clean_text(row[check_result_col])
            if clean_condition and clean_result:
                current_checks[str(check_index)] = {
                    "task_description": clean_condition,
                    "expected_result": "Tell (1)" if clean_result == "Yes" else "Tell (0)",
                    "model_result": "None",
                }
                check_index += 1

    # Save the last test case
    if current_test_case:
        current_test_case["check_conditions"] = current_checks
        if current_results:
            current_test_case["expected_result"] = f"Tell ({','.join(map(str, current_results))})"
        result[str(case_index)] = current_test_case

    # Write results to JSON file
    output_path = input_path.split(".")[0] + ".json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def reset_json_results(json_file_path: str) -> None:
    """
    Reset the result and evidence fields to empty strings for all test cases in the JSON file.

    Args:
        json_file_path: Path to the JSON file to process
    """
    try:
        # Read the JSON file
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Iterate through each project
        for project in data.values():
            if "test_cases" in project:
                # Iterate through each test case
                for case in project["test_cases"].values():
                    # Reset result and evidence to empty strings
                    case["result"] = ""
                    case["evidence"] = ""

        # Write back to the JSON file
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Successfully reset results and evidence in {json_file_path}")

    except Exception as e:
        logger.error(f"Error processing file {json_file_path}: {str(e)}")
        raise


def list_to_json(excel_file: str, json_file: str) -> None:
    """
    Convert data from Excel file to JSON format and save to JSON file.\n
    Input: Project name in Excel table, List(case_desc), prod_url, requirement\n
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
            test_case = row["case_name"] if "case_name" in row else row["project_name"]
            url = row["prod_url"]
            if pd.isna(test_case):
                continue
            task_list_str = row["auto_generated_test_cases"]

            try:
                task_list = ast.literal_eval(task_list_str)  # Convert string type list to list
                if not isinstance(task_list, list):
                    task_list = []
            except (ValueError, SyntaxError):
                task_list = []

            sheet_data[test_case] = {"url": f"{url}", "test_cases": {}}

            for i, task_desc in enumerate(task_list):
                sheet_data[test_case]["test_cases"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}

        output_data.update(sheet_data)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


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
    data[case_name] = {"url": url, "test_cases": {}}
    for i, task_desc in enumerate(test_cases):
        data[case_name]["test_cases"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
