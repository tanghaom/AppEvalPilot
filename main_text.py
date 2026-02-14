#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TextAgent entry point — text-only testing using accessibility tree / DOM tree.
No screenshots sent to LLM. Uses cheaper text-only models.

Usage:
    bash test_run_text.sh
    # or directly:
    python main_text.py --platform Linux --max_iters 10
"""
import asyncio
import argparse
import os
from pathlib import Path

from loguru import logger

# Set MetaGPT config root
os.environ['CONFIG_ROOT'] = '/root/.metagpt'

from appeval.roles.eval_runner import AppEvalRole


async def run_text_agent_test():
    """Run a test using TextAgent (text-only, a11y tree)."""
    try:
        json_file = "data/test_results_text.json"

        # Initialize AppEvalRole with agent_class="text_agent"
        appeval = AppEvalRole(
            json_file=json_file,
            agent_class="text_agent",       # 🆕 Use TextAgent instead of OSAgent
            a11y_mode="cdp",                # CDP mode: lightweight, only needs Chrome --remote-debugging-port
            remote_debugging_port=9333,     # Port for Chrome DevTools Protocol
            user_data_dir="/tmp/chrome_text_agent",  # Required for CDP remote debugging
            extend_xml_infos=True,
            max_iters=10,
            debug_screenshots=True,         # Save screenshots to disk for human review (not sent to LLM)
            # These are ignored by TextAgent but kept for compatibility:
            use_ocr=False,
            use_memory=False,
            use_reflection=False,
            use_chrome_debugger=False,
        )

        # Example test case
        test_cases = {
            "0": {
                "case_desc": "Verify successful login with valid username and password",
                "result": "",
                "evidence": "",
            },
            "1": {
                "case_desc": "Verify login fails with invalid username and valid password",
                "result": "",
                "evidence": "",
            },
        }

        url = "https://mgx.dev/"
        result, executability = await appeval.run_api(
            task_name="MGX_TextAgent",
            test_cases=test_cases,
            start_func=url,
            log_dir="work_dirs_text",
            max_retry_uncertain=0,  # No retry for quick test
        )

        logger.info(f"TextAgent test result: {result}")
        logger.info(f"Executability: {executability}")

    except Exception as e:
        logger.error(f"TextAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AppEval TextAgent Runner")
    parser.add_argument("--platform", type=str, default="Linux",
                        help="Platform: Windows, Linux, Mac")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Maximum iterations per test")
    parser.add_argument("--url", type=str, default=None,
                        help="Override target URL")
    parser.add_argument("--test_point", type=str, default=None,
                        help="Single test point description")
    args = parser.parse_args()

    if args.platform:
        os.environ['PLATFORM'] = args.platform

    await run_text_agent_test()


if __name__ == "__main__":
    asyncio.run(main())

