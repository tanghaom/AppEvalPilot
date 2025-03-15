#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@Author  : tanghaoming
@File    : test_chrome_debugger.py
@Desc    : Test script for Chrome debugger functionality
"""

import argparse
import asyncio
import time

from metagpt.logs import logger

from appeval.tools.chrome_debugger import ChromeDebugger


def test_sync_mode(duration: int = 30):
    """Test Chrome debugger in synchronous mode

    Args:
        duration: Test duration in seconds
    """
    logger.info("Starting Chrome debugger test in synchronous mode")
    logger.info("Make sure Chrome is running with --remote-debugging-port=9222")

    debugger = ChromeDebugger()
    debugger.start_monitoring()

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            # Check for new messages every 2 seconds
            time.sleep(2)
            new_messages = debugger.get_new_messages()
            if new_messages:
                logger.info(f"Received {len(new_messages)} new messages:")
                for msg in new_messages:
                    logger.info(f"  {msg}")
            else:
                logger.debug("No new messages")

        logger.info("Test completed successfully")
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
    finally:
        logger.info("Stopping Chrome debugger")
        debugger.stop_monitoring()


async def test_async_mode(duration: int = 30):
    """Test Chrome debugger in asynchronous mode

    Args:
        duration: Test duration in seconds
    """
    logger.info("Starting Chrome debugger test in asynchronous mode")
    logger.info("Make sure Chrome is running with --remote-debugging-port=9222")

    async with ChromeDebugger() as debugger:
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                # Check for new messages every 2 seconds
                await asyncio.sleep(2)
                new_messages = debugger.get_new_messages()
                if new_messages:
                    logger.info(f"Received {len(new_messages)} new messages:")
                    for msg in new_messages:
                        logger.info(f"  {msg}")
                else:
                    logger.debug("No new messages")

            logger.info("Test completed successfully")
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Test Chrome debugger functionality")
    parser.add_argument(
        "--mode",
        choices=["sync", "async"],
        default="sync",
        help="Test mode: synchronous or asynchronous (default: sync)",
    )
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds (default: 30)")

    args = parser.parse_args()

    if args.mode == "async":
        asyncio.run(test_async_mode(args.duration))
    else:
        test_sync_mode(args.duration)


if __name__ == "__main__":
    main()
