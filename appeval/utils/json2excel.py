import json

import pandas as pd


def convert_json_to_excel(json_file_path: str, excel_file_path: str) -> None:
    """
    Read JSON file and convert it to Excel table.\n

    Args:
        json_file_path (str): Path to JSON file.
        excel_file_path (str): Path to output Excel file.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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


def convert_excel_to_json(excel_file_path: str, json_file_path: str) -> None:
    """
    Read Excel file and convert it to a specific JSON format, using test.json format.\n
    Excel table must have the following columns: project, original requirement, web_url\n

    Args:
        excel_file_path (str): Path to Excel file.
        json_file_path (str): Path to output JSON file.
    """
    df = pd.read_excel(excel_file_path)

    output_json_data = {}
    check_counters = {}  # Used to track the count of check conditions for each project

    for _, row in df.iterrows():
        task = row["project_name"]
        url = row["web_url"]
        if task not in output_json_data:
            output_json_data[task] = {
                "test_cases": f"{url}",
                "env_script": "chrome.bat",
                "original_requirement": row["original_requirement"] if not pd.isna(row["original_requirement"]) else "",
                "expected_result": "",
                "model_result": "",
                "console_logs": "",
                "check_conditions": {},
            }
            check_counters[task] = 0

        check_id = str(check_counters[task])
        # task_description = row["input"] if not pd.isna(row["input"]) else ""
        # model_result = row["osagent_output"] if not pd.isna(row["osagent_output"]) else ""
        # console_logs = row["console_logs"] if not pd.isna(row["console_logs"]) else []

        output_json_data[task]["check_conditions"][check_id] = {
            "task_description": "",
            "expected_result": "",
            "model_result": "",
            "console_logs": "",
        }

        check_counters[task] += 1

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=4, ensure_ascii=False)


def clear_test_results(file_path: str) -> None:
    """
    Clears model results and console logs from a test case JSON file.

    Args:
        file_path (str): Path to JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return

    for project, project_data in data.items():
        if "model_result" in project_data:
            project_data["model_result"] = ""
        if "console_logs" in project_data:
            project_data["console_logs"] = []
        if "check_conditions" in project_data:
            for case_key, case_data in project_data["check_conditions"].items():
                if "model_result" in case_data:
                    case_data["model_result"] = ""
                if "console_logs" in case_data:
                    case_data["console_logs"] = []

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Successfully cleared test results in {file_path}")
