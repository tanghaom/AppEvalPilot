import asyncio
import json
from pathlib import Path

from loguru import logger

from appeval.roles.eval_runner import AppEvalRole


async def run_batch_test():
    """Run batch test example"""
    try:
        # Set test related paths
        project_excel = "data/test_cases/test.xlsx"
        case_excel = "data/test_cases/test_results.xlsx"
        json_file = "data/test_cases/test_results.json"

        # Initialize automated test role
        appeval = AppEvalRole(
            json_file=json_file,
            use_ocr=True,
            quad_split_ocr=True,
            use_memory=False,
            use_reflection=True,
            use_chrome_debugger=True,
            extend_xml_infos=True,
        )

        # Execute batch test
        result = await appeval.run(project_excel_path=project_excel, case_excel_path=case_excel)
        result = json.loads(result.content)
        logger.info(f"Batch test execution result: {result}")

    except Exception as e:
        logger.error(f"Batch test execution failed: {str(e)}")


async def run_single_test():
    """Run single test case example"""
    try:
        # Set test parameters
        case_name = "MGX"
        url = "https://mgx.dev/"
        requirement = "Please help me create an MGX official website. The website should include the following: 1. Homepage 2. Dialog box 3. AppWorld 4. Contact information"
        json_path = f"data/{case_name}.json"

        # Initialize automated test role
        appeval = AppEvalRole(
            json_file=json_path,
            use_ocr=True,
            quad_split_ocr=True,
            use_memory=False,
            use_reflection=True,
            use_chrome_debugger=True,
            extend_xml_infos=True,
            log_dirs=f"work_dirs/{case_name}",
        )

        # Execute single test
        result = await appeval.run(case_name=case_name, url=url, user_requirement=requirement, json_path=json_path)
        result = json.loads(result.content)
        logger.info(f"Single test execution result: {result}")

    except Exception as e:
        logger.error(f"Single test execution failed: {str(e)}")
        logger.exception("Detailed error information")


async def main():
    """Main function"""
    # Create necessary directories
    Path("data/test_cases").mkdir(parents=True, exist_ok=True)

    # Run single test example
    logger.info("Starting to execute single test...")
    await run_single_test()

    # Run batch test example
    # logger.info("Starting to execute batch test...")
    # await run_batch_test()


if __name__ == "__main__":
    asyncio.run(main())
