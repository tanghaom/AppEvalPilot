#!/usr/bin/env python3
"""本地测试用：接收回调 POST，按文档返回 data.success=true。"""
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            if isinstance(data, list) and data:
                self.server.received_results.extend(data)
            else:
                self.server.received_results.append(data)
        except Exception:
            self.server.received_results.append({"raw": body.decode("utf-8", errors="replace")[:500]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data":{"success":true},"msg":"ok","code":0}')
        self.server.last_request_time = time.time()

    def log_message(self, *a):
        pass

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/batch_callback_result.json"
    idle_sec = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    max_requests = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.received_results = []
    server.last_request_time = 0.0
    server.socket.settimeout(1.0)
    print(
        f"本地回调接收: http://127.0.0.1:{port}，结果写入 {out_path}，"
        f"收到 {max_requests} 次或空闲 {idle_sec}s 后退出"
    )
    try:
        while True:
            try:
                server.handle_request()
            except (TimeoutError, OSError):
                pass
            if len(server.received_results) >= max_requests:
                break
            if server.received_results and (time.time() - server.last_request_time) > idle_sec:
                break
    finally:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(server.received_results, f, ensure_ascii=False, indent=2)
        print(f"已收到 {len(server.received_results)} 条回调，已写入 {out_path}")
        server.server_close()

if __name__ == "__main__":
    main()
