"""
AppEval API Server：接收外部测试任务 -> 异步执行 -> 回调结果。
配置与入口均在 appeval/api_server 下，不依赖 test2。
启动: python -m appeval.api_server [--port 8888] [--config config.yaml]
"""
from appeval.api_server.server import app, main

__all__ = ["app", "main"]
