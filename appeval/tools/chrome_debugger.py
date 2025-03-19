#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : chrome_debugger.py
@Desc    : Chrome debugger tool class for monitoring browser console output
"""

import asyncio
import json
import threading
from typing import Dict, List

import requests
import websockets
from metagpt.logs import logger


class ChromeDebugger:
    """Chrome debugger tool class for monitoring browser console output

    Attributes:
        debugger_url: Chrome debugger URL
        ws: WebSocket connection instance
        message_id: Message ID counter
        message_buffer: Message buffer
        last_read_index: Last read message index
        _running: Monitoring running status flag
        _thread: Monitoring thread
        _loop: Event loop
    """

    def __init__(self, host: str = "localhost", port: int = 9222):
        """Initialize Chrome debugger

        Args:
            host: Debugging host address
            port: Debugging port number
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
        """Get all page information"""
        response = requests.get(f"{self.debugger_url}/json")
        return response.json()

    def get_ws_url(self) -> str:
        """Get WebSocket URL"""
        pages = self.get_pages()
        for page in pages:
            if page["type"] == "page" and not page["url"].startswith("devtools://"):
                return page["webSocketDebuggerUrl"]
        raise Exception("No suitable page found")

    async def connect(self):
        """Establish WebSocket connection"""
        while self._running:  # Add loop to try connecting
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
                return  # Return after successful connection
            except Exception as e:
                logger.error(f"Failed to establish WebSocket connection: {e}")
                await asyncio.sleep(3)  # Wait 3 seconds before retrying if failed
                if not self._running:  # Exit retry loop if monitoring stopped
                    break

    async def send_command(self, method: str, params: Dict = None) -> Dict:
        """Send debugging command

        Args:
            method: Command method name
            params: Command parameters

        Returns:
            Command execution result
        """
        if params is None:
            params = {}
        self.message_id += 1
        message = {"id": self.message_id, "method": method, "params": params}
        await self.ws.send(json.dumps(message))
        response = await self.ws.recv()
        return json.loads(response)

    async def enable_runtime(self):
        """Enable runtime monitoring"""
        await self.send_command("Runtime.enable")
        await self.send_command("Log.enable")

    def _format_console_message(self, method: str, params: Dict) -> str:
        """Format console message

        Args:
            method: Message type
            params: Message parameters

        Returns:
            Formatted message
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
        """Core logic for monitoring console output"""
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
                        logger.info(f"Obtained browser console content: {formatted_message}")
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
        """Run event loop in a new thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._monitor_console())
        self._loop.close()

    def start_monitoring(self):
        """Start monitoring"""
        logger.info("Starting browser monitoring")
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Monitoring already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("Stopping browser monitoring")
        self._running = False
        if self._thread and self._thread.is_alive():
            if self._loop and self.ws:
                # Add timeout handling
                async def close_ws():
                    try:
                        logger.info("Starting to close WebSocket connection")
                        # Use transport to close directly without waiting for server response
                        if self.ws and self.ws.transport:
                            self.ws.transport.close()
                            # Optional: Force interrupt connection
                            # self.ws.transport.abort()
                        logger.info("WebSocket connection closed")
                    except Exception as e:
                        logger.warning(f"Close websocket error: {e}")

                future = asyncio.run_coroutine_threadsafe(close_ws(), self._loop)
                try:
                    future.result(timeout=5)  # Set timeout
                except Exception as e:
                    logger.warning(f"Close failed: {e}")
                    if self.ws and self.ws.transport:
                        self.ws.transport.abort()
                finally:
                    self.ws = None
                    logger.info("WebSocket connection set to None")

            self._thread.join()
            self._thread = None
            # Explicitly close event loop
            if not self._loop.is_closed():
                self._loop.close()
            self._loop = None

    def get_new_messages(self) -> List[str]:
        """Get new messages and update read position"""
        if self.last_read_index >= len(self.message_buffer):
            return []

        new_messages = self.message_buffer[self.last_read_index :]
        self.last_read_index = len(self.message_buffer)
        return new_messages

    def clear_buffer(self):
        """Clear message buffer"""
        self.message_buffer.clear()
        self.last_read_index = 0

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        self.start_monitoring()  # Automatically start monitoring when entering context
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_monitoring()
