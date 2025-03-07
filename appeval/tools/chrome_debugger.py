#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : chrome_debugger.py
@Desc    : Chrome调试工具类，用于监控浏览器控制台输出
"""

import asyncio
import json
import threading
from typing import Dict, List

import requests
import websockets
from metagpt.logs import logger


class ChromeDebugger:
    """Chrome调试工具类，用于监控浏览器控制台输出

    Attributes:
        debugger_url: Chrome调试器URL
        ws: WebSocket连接实例
        message_id: 消息ID计数器
        message_buffer: 消息缓冲区
        last_read_index: 最后读取的消息索引
        _running: 监控运行状态标志
        _thread: 监控线程
        _loop: 事件循环
    """

    def __init__(self, host: str = "localhost", port: int = 9222):
        """初始化Chrome调试工具

        Args:
            host: 调试主机地址
            port: 调试端口号
        """
        self.debugger_url = f"http://{host}:{port}"
        self.ws = None
        self.message_id = 0
        self.message_buffer = []
        self.last_read_index = 0
        self._running = False
        self._thread = None
        self._loop = None

    def get_pages(self) -> List[Dict]:
        """获取所有页面信息"""
        response = requests.get(f"{self.debugger_url}/json")
        return response.json()

    def get_ws_url(self) -> str:
        """获取WebSocket URL"""
        pages = self.get_pages()
        for page in pages:
            if page["type"] == "page" and not page["url"].startswith("devtools://"):
                return page["webSocketDebuggerUrl"]
        raise Exception("No suitable page found")

    async def connect(self):
        """建立WebSocket连接"""
        while self._running:  # 添加循环尝试连接
            if self.ws is not None:
                try:
                    await self.ws.ping()
                    return
                except Exception as e:
                    logger.error(f"WebSocket ping failed: {e}")
                    await self.ws.close()
                    self.ws = None

            try:
                ws_url = self.get_ws_url()
                logger.debug(f"Connecting to: {ws_url}")
                self.ws = await websockets.connect(ws_url)
                return  # 连接成功后返回
            except Exception as e:
                logger.error(f"Failed to establish WebSocket connection: {e}")
                await asyncio.sleep(3)  # 失败后等待3秒再重试
                if not self._running:  # 如果停止监控则退出重试
                    break

    async def send_command(self, method: str, params: Dict = None) -> Dict:
        """发送调试命令

        Args:
            method: 命令方法名
            params: 命令参数

        Returns:
            命令执行结果
        """
        if params is None:
            params = {}
        self.message_id += 1
        message = {"id": self.message_id, "method": method, "params": params}
        await self.ws.send(json.dumps(message))
        response = await self.ws.recv()
        return json.loads(response)

    async def enable_runtime(self):
        """启用运行时监控"""
        await self.send_command("Runtime.enable")
        await self.send_command("Log.enable")

    def _format_console_message(self, method: str, params: Dict) -> str:
        """格式化控制台消息

        Args:
            method: 消息类型
            params: 消息参数

        Returns:
            格式化后的消息
        """
        if method == "Runtime.consoleAPICalled":
            message_type = params["type"]
            args = params["args"]
            messages = []
            for arg in args:
                if "value" in arg:
                    messages.append(str(arg["value"]))
                elif "description" in arg:
                    messages.append(arg["description"])
            return f"Console {message_type}: {' '.join(messages)}" if messages else None

        elif method == "Runtime.exceptionThrown":
            exception = params["exceptionDetails"]
            if "exception" in exception:
                return f"Error: {exception['exception'].get('description', '')}"
            return f"Exception: {exception.get('text', '')}"

        elif method == "Log.entryAdded":
            entry = params["entry"]
            if not entry["url"].startswith("devtools://"):
                return f"Log {entry['level']}: {entry['text']}"

        return None

    async def _monitor_console(self):
        """监控控制台输出的核心逻辑"""
        is_first = True

        while self._running:
            try:
                if is_first:
                    is_first = False
                    await self.connect()
                    await self.enable_runtime()

                message = await self.ws.recv()
                data = json.loads(message)
                if "method" in data:
                    formatted_message = self._format_console_message(data["method"], data.get("params", {}))
                    if formatted_message:
                        logger.info(f"获取到浏览器console内容：{formatted_message}")
                        if formatted_message not in self.message_buffer:
                            self.message_buffer.append(formatted_message)

            except websockets.exceptions.ConnectionClosed:
                if self._running:
                    logger.warning("Connection closed. Waiting before reconnect...")
                    await asyncio.sleep(3)
                    try:
                        await self.connect()
                        await self.enable_runtime()
                    except Exception as e:
                        logger.error(f"Reconnection failed: {e}")
                        await asyncio.sleep(3)

            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(1)
                if not self._running:
                    break

    def _run_async_loop(self):
        """在新线程中运行事件循环"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._monitor_console())
        self._loop.close()

    def start_monitoring(self):
        """启动监控"""
        logger.info("启动浏览器监控")
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Monitoring already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop_monitoring(self):
        """停止监控"""
        logger.info("停止浏览器监控")
        self._running = False
        if self._thread and self._thread.is_alive():
            if self._loop and self.ws:
                # 添加超时处理
                async def close_ws():
                    try:
                        logger.info("开始关闭WebSocket连接")
                        # 使用transport直接关闭，不等待服务器响应
                        if self.ws and self.ws.transport:
                            self.ws.transport.close()
                            # 可选：强制中断连接
                            # self.ws.transport.abort()
                        logger.info("WebSocket连接已关闭")
                    except Exception as e:
                        logger.warning(f"Close websocket error: {e}")

                future = asyncio.run_coroutine_threadsafe(close_ws(), self._loop)
                try:
                    future.result(timeout=5)  # 设置超时
                except Exception as e:
                    logger.warning(f"关闭失败: {e}")
                    if self.ws and self.ws.transport:
                        self.ws.transport.abort()
                finally:
                    self.ws = None
                    logger.info("WebSocket连接已设置为None")

            self._thread.join()
            self._thread = None
            # 显式关闭事件循环
            if not self._loop.is_closed():
                self._loop.close()
            self._loop = None

    def get_new_messages(self) -> List[str]:
        """获取新消息并更新读取位置"""
        if self.last_read_index >= len(self.message_buffer):
            return []

        new_messages = self.message_buffer[self.last_read_index :]
        self.last_read_index = len(self.message_buffer)
        return new_messages

    def clear_buffer(self):
        """清除消息缓冲区"""
        self.message_buffer.clear()
        self.last_read_index = 0

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        self.start_monitoring()  # 在进入上下文时自动启动监控
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop_monitoring()


if __name__ == "__main__":
    """
    使用方法：
    1. 启动Chrome浏览器，添加调试参数：
       Windows: "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
       Mac: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222
       Linux: google-chrome --remote-debugging-port=9222

    2. 运行此脚本：
       - 同步方式：python chrome_debugger.py
       - 异步方式：python chrome_debugger.py --async
    """

    def usage_example_sync():
        """同步使用示例"""
        debugger = ChromeDebugger()
        debugger.start_monitoring()

        try:
            while True:
                # 每隔5秒获取一次新消息
                import time

                time.sleep(5)
                new_messages = debugger.get_new_messages()
                if new_messages:
                    print("\nNew messages:")
                    for msg in new_messages:
                        print(msg)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            debugger.stop_monitoring()

    async def usage_example_async():
        """异步使用示例"""
        async with ChromeDebugger() as debugger:  # 进入上下文时自动启动监控
            try:
                while True:
                    # 每隔5秒获取一次新消息
                    await asyncio.sleep(5)
                    new_messages = debugger.get_new_messages()
                    if new_messages:
                        print("\nNew messages:")
                        for msg in new_messages:
                            print(msg)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(usage_example_async())
    else:
        usage_example_sync()
