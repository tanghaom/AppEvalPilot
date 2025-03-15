#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/06
@Author  : tanghaoming
@File    : test_server.py
@Desc    : Test server module
"""

import json
import time

import requests

BASE_URL = "http://localhost:8888"


def wait_for_completion(task_id, timeout=300):
    """Wait for task completion, default timeout is 5 minutes"""
    start_time = time.time()
    while True:
        status_response = requests.get(f"{BASE_URL}/task_status/{task_id}")
        status_data = status_response.json()
        print("Current task status:", status_data)

        if status_data["status"] in ["completed", "failed"]:
            break

        if time.time() - start_time > timeout:
            print("Task waiting timeout!")
            return False

        time.sleep(2)
    return True


def test_url_task():
    """Test URL type task"""
    print("\nTesting URL type task:")
    response = requests.post(
        f"{BASE_URL}/submit_task",
        files={"file": None},
        data={
            "params": json.dumps(
                {
                    "type": "url",
                    "case_name": "test_url",
                    "url": "http://example.com",
                    "user_requirement": "original user requirement",
                }
            )
        },
    )
    print("Task submission response:", response.json())

    task_id = response.json()["task_id"]

    # Wait for task completion
    if wait_for_completion(task_id):
        # Query task result
        result_response = requests.get(f"{BASE_URL}/task_result/{task_id}")
        print("Task result:", result_response.json())
    else:
        print("Task did not complete within the specified time")


def test_python_app():
    """Test Python application task"""
    print("\nTesting Python application task:")
    # Prepare a simple zip file
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
                        "user_requirement": "Test Python application",
                    }
                )
            },
        )
    print("Task submission response:", response.json())

    task_id = response.json()["task_id"]

    # Wait for task completion
    if wait_for_completion(task_id):
        # Query task result
        result_response = requests.get(f"{BASE_URL}/task_result/{task_id}")
        print("Task result:", result_response.json())
    else:
        print("Task did not complete within the specified time")


def test_python_web():
    """Test Python Web application task"""
    print("\nTesting Python Web application task:")
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
                        "user_requirement": "Test Web application",
                    }
                )
            },
        )
    print("Task submission response:", response.json())

    task_id = response.json()["task_id"]

    # Wait for task completion
    if wait_for_completion(task_id):
        # Query task result
        result_response = requests.get(f"{BASE_URL}/task_result/{task_id}")
        print("Task result:", result_response.json())
    else:
        print("Task did not complete within the specified time")


if __name__ == "__main__":
    # Test URL type task
    # test_url_task()

    # Test Python application task
    # test_python_app()

    # Test Python Web application task
    test_python_web()
