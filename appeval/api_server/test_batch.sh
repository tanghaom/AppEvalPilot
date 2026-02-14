#!/bin/bash
# 测试 /batch 接口的示例脚本

# 服务地址
API_URL="http://127.0.0.1:8888/batch"

# 构造请求体
curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
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
      "case_name": "示例应用测试",
      "prod_url": "https://example.com",
      "test_arry": [
        "测试搜索功能是否正常",
        "输入关键词并提交",
        "检查搜索结果页面"
      ]
    }
  ],
  "callback_url": "http://127.0.0.1:9999"
}'

echo -e "\n\n=== 说明 ==="
echo "1. 请确保服务已启动：python -m appeval.api_server.server --port 8888"
echo "2. 请确保回调接收端已启动：python appeval/api_server/receive_callback.py 9999"
echo "3. 任务会后台执行，完成后自动回调到 http://127.0.0.1:9999"
echo "4. 查看任务状态：curl http://127.0.0.1:8888/status"
