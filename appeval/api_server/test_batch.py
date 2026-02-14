#!/usr/bin/env python3
"""测试 /batch 接口的 Python 示例"""
import requests
import json
import time

# 服务地址
API_URL = "http://127.0.0.1:8888/batch"
STATUS_URL = "http://127.0.0.1:8888/status"

# 构造请求数据
payload = {
    "tasks": [
        {
            "task_id": 1,
            "detail_id": 1001,
            "case_name": "示例应用测试",
            "prod_url": "https://example.com",
            "test_arry": [
                "打开网页，检查标题是否包含Example",
                "点击导航栏的About链接",
                "确认About页面加载成功"
            ]
        },
        {
            "task_id": 1,
            "detail_id": 1002,
            "case_name": "示例应用测试2",
            "prod_url": "https://example.com",
            "test_arry": [
                "测试搜索功能是否正常",
                "输入关键词并提交",
                "检查搜索结果页面"
            ]
        }
    ],
    "callback_url": "http://127.0.0.1:9999"
}

print("=" * 60)
print("测试 AppEval API - /batch 接口")
print("=" * 60)
print(f"\n发送请求到: {API_URL}")
print(f"请求体:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")

try:
    # 发送请求
    response = requests.post(API_URL, json=payload, timeout=10)
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}\n")
    
    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            print("✅ 任务提交成功！")
            print("\n任务已在后台执行，可以通过以下方式查看状态：")
            print(f"  - 查看所有任务: curl {STATUS_URL}")
            print(f"  - 查看单个任务: curl {STATUS_URL}/1001")
            print("\n等待5秒后查询任务状态...")
            time.sleep(5)
            
            # 查询状态
            status_resp = requests.get(STATUS_URL)
            print(f"\n当前任务状态:\n{json.dumps(status_resp.json(), indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 任务提交失败: {result.get('message')}")
    else:
        print(f"❌ 请求失败: HTTP {response.status_code}")
        print(f"详细信息: {response.text}")

except requests.exceptions.ConnectionError:
    print("❌ 连接失败！请确保服务已启动：")
    print("   python -m appeval.api_server.server --port 8888")
except Exception as e:
    print(f"❌ 发生错误: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("注意事项：")
print("1. 确保服务已启动")
print("2. 确保回调接收端已启动: python appeval/api_server/receive_callback.py 9999")
print("3. 任务完成后会自动回调到 callback_url")
print("=" * 60)
