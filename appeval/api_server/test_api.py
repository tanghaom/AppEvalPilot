#!/usr/bin/env python3
"""
测试 api_server 是否正常：请求 /start、查 /status、/health。
不依赖真实跑完任务，仅验证接口可调通；回调会发往对方真实地址（可用 --dry-run 仅打日志不请求）。
"""
import argparse
import json
import sys
import time

try:
    import httpx
except ImportError:
    print("请安装: pip install httpx")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="测试 AppEval API Server")
    parser.add_argument("--base", type=str, default="http://127.0.0.1:8888", help="服务地址，如 http://127.0.0.1:8888")
    parser.add_argument("--start", action="store_true", help="发送 POST /start 测试")
    parser.add_argument("--status", action="store_true", help="请求 GET /status")
    parser.add_argument("--health", action="store_true", help="请求 GET /health")
    parser.add_argument("--detail-id", type=int, default=999, help="测试用 detail_id（避免与真实任务冲突）")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    if not (args.start or args.status or args.health):
        args.health = True
        args.status = True
        args.start = True

    with httpx.Client(timeout=30) as client:
        if args.health:
            print(">>> GET /health")
            try:
                r = client.get(f"{base}/health")
                print(f"    状态码: {r.status_code}")
                print(f"    响应: {r.json()}")
            except Exception as e:
                print(f"    失败: {e}")
                return 1
            print()

        if args.status:
            print(">>> GET /status")
            try:
                r = client.get(f"{base}/status")
                print(f"    状态码: {r.status_code}")
                print(f"    响应: {r.json()}")
            except Exception as e:
                print(f"    失败: {e}")
            print()

        if args.start:
            body = {
                "task_id": 1,
                "detail_id": args.detail_id,
                "case_name": "TestCase",
                "test_arry": ["检查页面是否加载"],
                "prod_url": "https://www.example.com",
            }
            print(">>> POST /start")
            print(f"    Body: {json.dumps(body, ensure_ascii=False)}")
            try:
                r = client.post(f"{base}/start", json=body)
                print(f"    状态码: {r.status_code}")
                data = r.json()
                print(f"    响应: {data}")
                ok = data.get("data", {}).get("succes", data.get("data", {}).get("success", False))
                print(f"    $.data.succes/success: {ok}")
                if ok:
                    print("    已接单，任务在后台执行。可再次执行 GET /status 查看进度。")
            except Exception as e:
                print(f"    失败: {e}")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
