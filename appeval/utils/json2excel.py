import json

import pandas as pd


def convert_json_to_excel(json_file_path, excel_file_path):
    """
    读取 JSON 文件并将其转换为 Excel 表格。\n

    Args:
        json_file_path (str): JSON 文件的路径。
        excel_file_path (str): 输出 Excel 文件的路径。
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    excel_data = []
    for task, details in data.items():
        for check_id, condition in details.get("测试用例", {}).items():
            excel_data.append(
                {
                    "项目名称": task,
                    "case_name": "",
                    "case_desc": condition.get("case_desc", ""),
                    "result": condition.get("result", ""),
                    "evidence": condition.get("evidence", ""),
                    "自动功能检测": "",
                    "console_logs": "",
                }
            )

    df = pd.DataFrame(excel_data)
    df.to_excel(excel_file_path, index=False)


def convert_excel_to_json(excel_file_path, json_file_path):
    """
    读取 Excel 文件并将其转换为特定格式的 JSON 文件, 使用test.json的格式。
    Excel表格中必须要有的列：项目 原始需求 web_url
    Args:
        excel_file_path (str): Excel 文件的路径。
        json_file_path (str): 输出 JSON 文件的路径。
    """
    df = pd.read_excel(excel_file_path)

    output_json_data = {}
    check_counters = {}  # 用于追踪每个项目的检验条件计数

    for _, row in df.iterrows():
        task = row["项目"]
        url = row["web_url"]
        if task not in output_json_data:
            output_json_data[task] = {
                "测试用例": f"{url}",
                "env_script": "chrome.bat",
                "原始需求": row["原始需求"] if not pd.isna(row["原始需求"]) else "",
                "预期结果": "",
                "模型结果": "",
                "console_logs": "",
                "检验条件": {},
            }
            check_counters[task] = 0

        check_id = str(check_counters[task])
        # task_description = row["输入"] if not pd.isna(row["输入"]) else ""
        # model_result = row["osagent输出"] if not pd.isna(row["osagent输出"]) else ""
        # console_logs = row["console_logs"] if not pd.isna(row["console_logs"]) else []

        output_json_data[task]["检验条件"][check_id] = {"任务描述": "", "预期结果": "", "模型结果": "", "console_logs": ""}

        check_counters[task] += 1

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=4, ensure_ascii=False)


def clear_test_results(file_path):
    """
    Clears model results and console logs from a test case JSON file.

    Args:
        file_path (str): The path to the JSON file.
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
        if "模型结果" in project_data:
            project_data["模型结果"] = ""
        if "console_logs" in project_data:
            project_data["console_logs"] = []
        if "检验条件" in project_data:
            for case_key, case_data in project_data["检验条件"].items():
                if "模型结果" in case_data:
                    case_data["模型结果"] = ""
                if "console_logs" in case_data:
                    case_data["console_logs"] = []

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Successfully cleared test results in {file_path}")


if __name__ == "__main__":
    file_name = "自动测试用例"
    json_file = f"data/{file_name}.json"
    excel_file = f"data/{file_name}.xlsx"
    convert_json_to_excel(json_file, excel_file)
    print(f"已将 {json_file} 转换为 {excel_file}")
