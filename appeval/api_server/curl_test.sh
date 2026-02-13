#!/bin/bash
# 测试 API 服务。请先在另一终端启动: python appeval/api_server/server.py --port 8888
# 关键: export no_proxy='*' 绕过 http_proxy，直接访问本机
export no_proxy='*'

BASE="${1:-http://127.0.0.1:8888}"

echo "=== 1. GET $BASE/health ==="
curl -s "$BASE/health" | python3 -m json.tool 2>/dev/null || curl -s "$BASE/health"
echo ""

echo "=== 2. POST $BASE/start ==="
curl -s -X POST "$BASE/start" \
  -H "Content-Type: application/json" \
  -d '{"task_id":1,"detail_id":999,"case_name":"Test","test_arry":["检查页面"],"prod_url":"https://www.example.com"}' \
  | python3 -m json.tool 2>/dev/null
echo ""

echo "=== 3. GET $BASE/status ==="
curl -s "$BASE/status" | python3 -m json.tool 2>/dev/null || curl -s "$BASE/status"
echo ""
