#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@Author  : tanghaoming
@File    : run_osagent.py
@Desc    : OSAgent demo script
"""
import argparse
import asyncio
from pathlib import Path

from appeval.roles.osagent import OSAgent


async def run_osagent(args):
    """Run OSAgent with given arguments"""
    print(f"Running OSAgent with platform: {args.platform}, instruction: {args.instruction}")

    # Initialize OSAgent with provided arguments
    agent = OSAgent(
        platform=args.platform,
        max_iters=args.max_iters,
        use_ocr=args.use_ocr,
        use_icon_detect=args.use_icon_detect,
        use_icon_caption=args.use_icon_caption,
        use_memory=args.use_memory,
        use_reflection=args.use_reflection,
        use_som=args.use_som,
        extend_xml_infos=args.extend_xml_infos,
        use_chrome_debugger=args.use_chrome_debugger,
        location_info=args.location_info,
        draw_text_box=args.draw_text_box,
        quad_split_ocr=args.quad_split_ocr,
        log_dirs=args.log_dirs,
        font_path=args.font_path,
        knowledge_base_path=args.knowledge_base_path,
        add_info=args.add_info,
    )

    # Run the agent with the provided instruction
    response = await agent.run(args.instruction)
    print(f"Agent response: {response.content}")

    # Get and print action history
    history = agent.get_action_history()
    print(f"Action history count: {len(history)}")

    # Print the first and last actions if available
    if history:
        print(f"First action: {history[0]['action']}")
        print(f"Last action: {history[-1]['action']}")

    # Get and print console logs if enabled
    if args.use_chrome_debugger:
        console_logs = agent.get_webbrowser_console_logs(steps=args.max_iters)
        print(f"Console logs count: {len(console_logs)}")


def main():
    """Parse arguments and run the test"""
    parser = argparse.ArgumentParser(description="Test OSAgent")

    # Basic configuration parameters
    parser.add_argument(
        "--platform", type=str, default="Windows", help="Operating system type (Windows, Mac, or Android)"
    )
    parser.add_argument("--max_iters", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("--instruction", type=str, default="Search Xiamen weather tomorrow", help="User instruction")

    # Feature switch parameters
    parser.add_argument("--use_ocr", type=int, default=0, help="Whether to use OCR")
    parser.add_argument("--use_icon_detect", type=int, default=0, help="Whether to use icon detection")
    parser.add_argument("--use_icon_caption", type=int, default=0, help="Whether to use icon caption")
    parser.add_argument("--use_memory", type=int, default=1, help="Whether to enable important content memory")
    parser.add_argument("--use_reflection", type=int, default=1, help="Whether to perform reflection")
    parser.add_argument("--use_som", type=int, default=0, help="Whether to draw visualization boxes on screenshots")
    parser.add_argument("--extend_xml_infos", type=int, default=1, help="Whether to get XML element information")
    parser.add_argument("--use_chrome_debugger", type=int, default=0, help="Whether to record browser console output")

    # Display and layout parameters
    parser.add_argument(
        "--location_info", type=str, default="center", help="Location information type (center or bbox)"
    )
    parser.add_argument("--draw_text_box", type=int, default=0, help="Whether to draw text boxes in visualization")
    parser.add_argument(
        "--quad_split_ocr", type=int, default=0, help="Whether to split image into four parts for OCR recognition"
    )

    # Path related parameters
    parser.add_argument("--log_dirs", type=str, default="workspace", help="Log directory")
    parser.add_argument(
        "--font_path",
        type=str,
        default=str(Path(__file__).parent.parent / "simhei.ttf"),
        help="Font path",
    )
    parser.add_argument(
        "--knowledge_base_path",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "knowledge"),
        help="Preset knowledge base file directory path",
    )

    # Other optional parameters
    parser.add_argument("--add_info", type=str, default="", help="Additional information to add to the prompt")

    args = parser.parse_args()

    # Set default add_info based on platform if not provided
    if args.add_info == "":
        if args.platform == "Windows":
            args.add_info = (
                "If you need to interact with elements outside of a web popup, such as calendar or time "
                "selection popups, make sure to close the popup first. If the content in a text box is "
                "entered incorrectly, use the select all and delete actions to clear it, then re-enter "
                "the correct information. To open a folder in File Explorer, please use a double-click."
            )
        elif args.platform == "Android":
            args.add_info = (
                "If you need to open an app, prioritize using the Open app (app name) action. If this fails, "
                "return to the home screen and click the app icon on the desktop. If you want to exit an app, "
                "return to the home screen. If there is a popup ad in the app, you should close the ad first. "
                "If you need to switch to another app, you should first return to the desktop. When summarizing "
                "content, comparing items, or performing cross-app actions, remember to leverage the content in memory."
            )

    # Run the test
    asyncio.run(run_osagent(args))


if __name__ == "__main__":
    main()
