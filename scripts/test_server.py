#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/06
@Author  : tanghaoming
@File    : test_server.py
@Desc    : 测试服务器模块
"""

import json
import time

import requests

BASE_URL = "http://localhost:8888"


def wait_for_completion(task_id, timeout=300):
    """等待任务完成，超时时间默认5分钟"""
    start_time = time.time()
    while True:
        status_response = requests.get(f"{BASE_URL}/task_status/{task_id}")
        status_data = status_response.json()
        print("当前任务状态:", status_data)

        if status_data["status"] in ["completed", "failed"]:
            break

        if time.time() - start_time > timeout:
            print("任务等待超时！")
            return False

        time.sleep(2)
    return True


def test_url_task():
    """测试URL类型任务"""
    print("\n测试URL类型任务:")
    response = requests.post(
        f"{BASE_URL}/submit_task",
        files={"file": None},
        data={
            "params": json.dumps(
                {"type": "url", "case_name": "test_url", "url": "http://example.com", "user_requirement": "测试需求"}
            )
        },
    )
    print("提交任务响应:", response.json())

    task_id = response.json()["task_id"]

    # 等待任务完成
    if wait_for_completion(task_id):
        # 查询任务结果
        result_response = requests.get(f"{BASE_URL}/task_result/{task_id}")
        print("任务结果:", result_response.json())
    else:
        print("任务未能在规定时间内完成")


def test_python_app():
    """测试Python应用任务"""
    print("\n测试Python应用任务:")
    # 准备一个简单的zip文件
    with open("test_server_pythonapp.zip", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/submit_task",
            files={"file": ("test_server_pythonapp.zip", f, "application/zip")},
            data={
                "params": json.dumps(
                    {
                        "type": "python_app",
                        "case_name": "test_app",
                        "start_path": "main.py",
                        "user_requirement": "测试Python应用",
                    }
                )
            },
        )
    print("提交任务响应:", response.json())

    task_id = response.json()["task_id"]

    # 等待任务完成
    if wait_for_completion(task_id):
        # 查询任务结果
        result_response = requests.get(f"{BASE_URL}/task_result/{task_id}")
        print("任务结果:", result_response.json())
    else:
        print("任务未能在规定时间内完成")


def test_python_web():
    """测试Python Web应用任务"""
    print("\n测试Python Web应用任务:")
    with open("test_server_pythonweb.zip", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/submit_task",
            files={"file": ("test_server_pythonweb.zip", f, "application/zip")},
            data={
                "params": json.dumps(
                    {
                        "type": "python_web",
                        "case_name": "test_web",
                        "start_path": "main.py",
                        "user_requirement": "测试Web应用",
                    }
                )
            },
        )
    print("提交任务响应:", response.json())

    task_id = response.json()["task_id"]

    # 等待任务完成
    if wait_for_completion(task_id):
        # 查询任务结果
        result_response = requests.get(f"{BASE_URL}/task_result/{task_id}")
        print("任务结果:", result_response.json())
    else:
        print("任务未能在规定时间内完成")


if __name__ == "__main__":
    # 测试URL类型任务
    # test_url_task()

    # 测试Python应用任务
    # test_python_app()

    # 测试Python Web应用任务
    test_python_web()
