import ast
import json
import os
import re
from typing import List

import pandas as pd
from metagpt.logs import logger


def clean_text(text):
    # 去掉开头的数字和点以及空格
    return re.sub(r"^\d+\.\s*", "", str(text))


def get_result_value(result_text):
    # 处理预期结果,将"已完成"/"未完成"/"未实现"转换为1/0/-1
    result_text = clean_text(result_text)
    if result_text == "已完成":
        return 1
    elif result_text == "未完成":
        return 0
    elif result_text == "未实现":
        return -1
    return None


def excel_to_json(input_path):
    # 读取Excel文件
    df = pd.read_excel(input_path)

    # 获取列名和对应的索引
    columns = df.columns.tolist()
    test_result_col = columns[1]  # B列 - 测试用例的预期结果
    check_result_col = columns[4]  # E列 - 检验条件的预期结果

    # 初始化结果字典
    result = {}
    current_test_case = {}
    current_checks = {}
    case_index = 0
    check_index = 0
    current_results = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 检查是否为新的测试用例（通过检查第一列是否以数字和点开始）
        if str(row["测试用例"]).strip() and re.match(r"1\.", str(row["测试用例"])):
            # 如果存在前一个测试用例，保存它
            if current_test_case:
                current_test_case["检验条件"] = current_checks
                if current_results:
                    current_test_case["预期结果"] = f"Tell ({','.join(map(str, current_results))})"
                result[str(case_index)] = current_test_case
                case_index += 1

            # 创建新的测试用例
            current_test_case = {"测试用例": "", "url": "None", "预期结果": "", "模型结果": "None", "检验条件": {}}
            current_checks = {}
            current_results = []
            check_index = 0

        # 处理测试用例
        if not pd.isna(row["测试用例"]):
            test_step = str(row["测试用例"]).strip()
            if current_test_case.get("测试用例"):
                current_test_case["测试用例"] += f" {test_step}"
            else:
                current_test_case["测试用例"] = test_step

            # 处理B列的预期结果
            if not pd.isna(row[test_result_col]):
                result_value = get_result_value(row[test_result_col])
                if result_value is not None:
                    current_results.append(result_value)

        # 处理校验条件
        if not pd.isna(row["校验条件"]) and not pd.isna(row[check_result_col]):
            clean_condition = clean_text(row["校验条件"])
            clean_result = clean_text(row[check_result_col])
            if clean_condition and clean_result:
                current_checks[str(check_index)] = {
                    "任务描述": clean_condition,
                    "预期结果": "Tell (1)" if clean_result == "是" else "Tell (0)",
                    "模型结果": "None",
                }
                check_index += 1

    # 保存最后一个测试用例
    if current_test_case:
        current_test_case["检验条件"] = current_checks
        if current_results:
            current_test_case["预期结果"] = f"Tell ({','.join(map(str, current_results))})"
        result[str(case_index)] = current_test_case

    # 将结果写入JSON文件
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
            if "测试用例" in project:
                # Iterate through each test case
                for case in project["测试用例"].values():
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


def list_to_json(excel_file, json_file):
    """
    将 Excel 文件中的数据转换为 JSON 格式，并保存到 JSON 文件中。\n
    Input：Excel表格中的项目名称，List(case_desc)，prod_url，requirement\n
    Output：测试json
    Args:
        excel_file (str): Excel 文件的路径。
        json_file (str): 输出 JSON 文件的路径。
    """
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names
    output_data = {}

    for sheet_name in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheet_data = {}
        for index, row in df.iterrows():
            test_case = row["case_name"] if "case_name" in row else row["项目名称"]
            url = row["prod_url"]
            if pd.isna(test_case):
                continue
            task_list_str = row["自动生成测试用例"]

            try:
                task_list = ast.literal_eval(task_list_str)  # 将字符串类型的列表转换为list
                if not isinstance(task_list, list):
                    task_list = []
            except (ValueError, SyntaxError):
                task_list = []

            sheet_data[test_case] = {"url": f"{url}", "测试用例": {}}

            for i, task_desc in enumerate(task_list):
                sheet_data[test_case]["测试用例"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}

        output_data.update(sheet_data)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def make_json_single(case_name: str, url: str, test_cases: List[str], json_path: str) -> None:
    """
    将单个测试用例转换为JSON格式并保存

    Args:
        case_name: 测试用例名称
        url: 测试URL
        test_cases: 测试用例列表
        json_path: 输出JSON文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    data = {}
    data[case_name] = {"url": url, "测试用例": {}}
    for i, task_desc in enumerate(test_cases):
        data[case_name]["测试用例"][str(i)] = {"case_desc": task_desc, "result": "", "evidence": ""}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    try:
        excel_to_json("data/自动测试用例.xlsx")
        print("Excel文件已成功转换为JSON格式！")
    except Exception as e:
        print(f"转换过程中出现错误：{str(e)}")
