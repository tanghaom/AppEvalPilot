#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel Web Testing Runner for Linux

Run multiple web tests in parallel with isolated environments.
Each test runs in a separate process with its own Xvfb display and Chrome instance.

Usage:
    python main_parallel.py
"""
from loguru import logger

from appeval.utils.parallel_runner import run_parallel_tests


# ============================================================================
# CONFIGURATION - Modify these values as needed
# ============================================================================

# Maximum number of parallel workers
MAX_WORKERS = 20

# Starting Xvfb display number (:100, :101, ...)
BASE_DISPLAY = 100

# Starting Chrome debug port (9300, 9301, ...)
BASE_PORT = 9300

# Maximum iterations per test
MAX_ITERS = 20

# LLM configuration file
CONFIG_FILE = "config/config2.yaml"

# Output directory for results
OUTPUT_DIR = "work_dirs/parallel"


# ============================================================================
# TEST EXECUTION FUNCTIONS
# ============================================================================

def run_parallel_api_test():
    """
    Run parallel API test - similar to run_api_test() in main.py
    
    Define your test tasks here, then run them in parallel.
    """
    # Define test tasks - 20 tasks for parallel testing
    test_urls = [
        ("Baidu", "https://www.baidu.com"),
        ("GitHub", "https://github.com"),
        ("Bing", "https://www.bing.com"),
        ("Google", "https://www.google.com"),
        ("Wikipedia", "https://www.wikipedia.org"),
        ("StackOverflow", "https://stackoverflow.com"),
        ("YouTube", "https://www.youtube.com"),
        ("Twitter", "https://twitter.com"),
        ("LinkedIn", "https://www.linkedin.com"),
        ("Amazon", "https://www.amazon.com"),
        ("Apple", "https://www.apple.com"),
        ("Microsoft", "https://www.microsoft.com"),
        ("Netflix", "https://www.netflix.com"),
        ("Spotify", "https://www.spotify.com"),
        ("Discord", "https://discord.com"),
        ("Twitch", "https://www.twitch.tv"),
        ("Pinterest", "https://www.pinterest.com"),
        ("Quora", "https://www.quora.com"),
        ("Medium", "https://medium.com"),
        ("Notion", "https://www.notion.so"),
    ]
    
    # Common test cases for all sites
    common_test_cases = {
        "0": {
            "case_desc": "Verify page loads successfully with main content visible",
            "result": "",
            "evidence": ""
        },
        "1": {
            "case_desc": "Verify logo or branding is displayed",
            "result": "",
            "evidence": ""
        }
    }
    
    tasks = [
        {
            "task_name": name,
            "test_cases": common_test_cases.copy(),
            "start_func": url,
            "log_dir": f"{OUTPUT_DIR}/{name.lower()}"
        }
        for name, url in test_urls
    ]
    
    logger.info(f"Starting parallel execution of {len(tasks)} tasks")
    logger.info(f"Max workers: {MAX_WORKERS}")
    
    # Run tests in parallel
    results = run_parallel_tests(
        tasks=tasks,
        max_workers=MAX_WORKERS,
        base_display=BASE_DISPLAY,
        base_port=BASE_PORT,
        config_file=CONFIG_FILE,
        use_ocr=False,
        use_reflection=True,
        max_iters=MAX_ITERS
    )
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    
    for task_name, result, executability, error in results:
        if error:
            logger.error(f"[{task_name}] ERROR: {error}")
        else:
            logger.info(f"[{task_name}] Executability: {executability}")
            for case_id, case_result in result.items():
                status = case_result.get("result", "Unknown")
                evidence = case_result.get("evidence", "")
                logger.info(f"  Case {case_id}: {status} - {evidence[:100]}")
    
    return results


def run_single_parallel_test():
    """
    Run a single test in parallel mode (for testing the parallel infrastructure)
    """
    tasks = [
        {
            "task_name": "ExampleTest",
            "test_cases": {
                "0": {
                    "case_desc": "Verify page loads successfully",
                    "result": "",
                    "evidence": ""
                }
            },
            "start_func": "https://example.com",
            "log_dir": f"{OUTPUT_DIR}/example"
        }
    ]
    
    results = run_parallel_tests(
        tasks=tasks,
        max_workers=1,
        base_display=BASE_DISPLAY,
        base_port=BASE_PORT,
        config_file=CONFIG_FILE,
        max_iters=MAX_ITERS
    )
    
    for task_name, result, executability, error in results:
        if error:
            logger.error(f"Test failed: {error}")
        else:
            logger.info(f"Test result: {result}")
            logger.info(f"Executability: {executability}")
    
    return results


def main():
    """Main function"""
    # Run parallel API test (similar to run_api_test in main.py)
    logger.info("Starting parallel test execution...")
    run_parallel_api_test()
    
    # Or run single test:
    # run_single_parallel_test()


if __name__ == "__main__":
    main()
