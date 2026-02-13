#!/bin/bash
# 从 simulate_20_tasks_from_test2.json 读取任务并调用 /batch（仅接单，结果走回调）
# 同一个终端里自动启动本地回调接收器并打印最终结果
# 用法:
#   no_proxy='*' bash appeval/api_server/simulate_send.sh
#   no_proxy='*' bash appeval/api_server/simulate_send.sh http://127.0.0.1:8888

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
JSON="$DIR/simulate_20_tasks_from_test2.json"
BASE="${1:-http://127.0.0.1:8888}"
CALLBACK_PORT=9999
RESULT_FILE="/tmp/batch_callback_result.json"

if [ ! -f "$JSON" ]; then
  echo "任务文件不存在: $JSON"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "需要 jq，请先安装（apt-get install jq）"
  exit 1
fi

count=$(jq -r 'length' "$JSON")
echo "从 $(basename "$JSON") 加载 $count 条任务，POST $BASE/batch（仅返回 data.success）"

callback_url="http://127.0.0.1:${CALLBACK_PORT}"
body=$(jq -c --arg cb "$callback_url" '{tasks: ., callback_url: $cb}' "$JSON")
timeout=$((count * 90))
[ "$timeout" -gt 1800 ] && timeout=1800
wait_timeout=$((timeout + 120))

rm -f "$RESULT_FILE"
# 回调接收器空闲超时设置得比任务总等待时间更长，避免任务还没跑完接收器先退出
# batch 一次性回调，收到 1 次就立即落盘退出，避免脚本继续空等
python "$DIR/receive_callback.py" "$CALLBACK_PORT" "$RESULT_FILE" "$wait_timeout" 1 >/tmp/receive_callback.log 2>&1 &
receiver_pid=$!
sleep 1
if ! kill -0 "$receiver_pid" 2>/dev/null; then
  echo "本地回调接收器启动失败，请检查端口 $CALLBACK_PORT 是否被占用"
  echo "日志:"
  cat /tmp/receive_callback.log
  exit 1
fi

resp=$(no_proxy='*' curl -s -X POST "$BASE/batch" \
  -H "Content-Type: application/json" \
  -d "$body" \
  --max-time "$timeout")

echo "完整响应:"
echo "$resp" | jq '.'

ok=$(echo "$resp" | jq -r '.data.success // false')
if [ "$ok" != "true" ]; then
  echo "接单失败，不等待回调"
  kill "$receiver_pid" 2>/dev/null || true
  exit 1
fi

echo ""
echo "等待回调结果（同一终端输出）..."
elapsed=0
while [ "$elapsed" -lt "$wait_timeout" ]; do
  if ! kill -0 "$receiver_pid" 2>/dev/null && [ ! -s "$RESULT_FILE" ]; then
    echo "回调接收器提前退出，日志:"
    cat /tmp/receive_callback.log
    exit 1
  fi
  if [ -s "$RESULT_FILE" ]; then
    break
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

if [ -s "$RESULT_FILE" ]; then
  echo "收到回调结果:"
  jq '.' "$RESULT_FILE"
else
  echo "等待回调超时（${wait_timeout}s）"
  echo "接收器日志:"
  cat /tmp/receive_callback.log
  kill "$receiver_pid" 2>/dev/null || true
  exit 1
fi

