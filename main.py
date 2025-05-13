import asyncio
import json
from pathlib import Path

from loguru import logger

from appeval.roles.eval_runner import AppEvalRole
from appeval.utils.excel_json_converter import make_work_path


async def run_batch_test():
    """Run batch test example"""
    try:
        # Set test related paths
        project_excel = r"data/test.xlsx"
        case_excel = r"data/test_results.xlsx"
        json_file = r"data/test_results.json"
        work_dir = r"work_dirs/test"
        # Make work path
        make_work_path(project_excel, work_dir)

        # Initialize automated test role
        appeval = AppEvalRole(
            json_file=json_file,
            use_ocr=False,
            quad_split_ocr=False,
            use_memory=False,
            use_reflection=True,
            use_chrome_debugger=False,
            extend_xml_infos=True,
            max_iters=20,
        )

        # Execute batch test
        # result = await appeval.run_batch(project_excel_path=project_excel, case_excel_path=case_excel)
        case_result = await appeval.run_mini_batch(project_excel_path=project_excel, case_excel_path=case_excel, generate_case_only=False)
        logger.info(f"Batch test execution result: {case_result}")

    except Exception as e:
        logger.error(f"Batch test execution failed: {str(e)}")

async def run_api_test():
    """Run batch test example"""
    try:
        json_file = r"data/test_results.json"
        # Initialize automated test role
        appeval = AppEvalRole(
            json_file=json_file,
            use_ocr=False,
            quad_split_ocr=False,
            use_memory=False,
            use_reflection=True,
            use_chrome_debugger=False,
            extend_xml_infos=True,
            max_iters=20,
        )
        # project_excel = r"data/test.xlsx"
        # case_excel = r"data/test_results.xlsx"
        # case_result = await appeval.run_mini_batch(project_excel_path=project_excel, case_excel_path=case_excel, generate_case_only=True)
        # logger.info(f"Batch test execution result: {case_result}")
        #其中url和work_path二者只存在一种
        case_result_example = {
            "1": {
                "task_name": "Example Task",
                "url": "https://mgx.dev/",
                "requirement": "Create a login page with username and password fields",
                "tag": "1",
                "test_cases": {
                    "0": {
                        "case_desc": "Verify successful login with valid username and password",
                        "result": "",
                        "evidence": ""
                    },
                    "1": {
                        "case_desc": "Verify login fails with invalid username and valid password",
                        "result": "",
                        "evidence": ""
                    },
                    "2": {
                        "case_desc": "Verify login fails with valid username and invalid password",
                        "result": "",
                        "evidence": ""
                    },
                    "3": {
                        "case_desc": "Verify login fails with empty username and valid password",
                        "result": "",
                        "evidence": ""
                    },
                    "4": {
                        "case_desc": "Verify login fails with valid username and empty password",
                        "result": "",
                        "evidence": ""
                    },
                    "5": {
                        "case_desc": "Verify login fails with empty username and empty password",
                        "result": "",
                        "evidence": ""
                    }
                }
            }
        }
        test_cases = case_result_example["1"]["test_cases"]
        result, executability = await appeval.run_api(task_name="MGX", test_cases=test_cases, start_func="http://localhost:7687", log_dir="work_dirs/MGX")
        # eval output format
        # {'0': {'result': 'Pass', 'evidence': 'All required login page UI elements are present and properly displayed: username/email input field at (1414, 750), password input field at (1414, 840), and sign in button at (1413, 984). The elements are clearly visible and positioned appropriately on the login form.'}, '1': {'result': 'Pass', 'evidence': "Successfully entered alphanumeric string 'Test123User' into the username field. The field accepted and displayed the input correctly without any restrictions or errors."}, '2': {'result': 'Pass', 'evidence': "The password field successfully masks input characters - when 'testpass123' was entered, it displays as bullet points/dots (•••••••••••) instead of plain text, providing proper password security."}, '3': {'result': 'Uncertain', 'evidence': 'Unable to verify special character acceptance in password field due to connection error with accounts.google.com (ERR_CONNECTION_CLOSED)'}}
        logger.info(f"Batch test execution result: {result}")
        logger.info(f"Executability: {executability}")

    except Exception as e:
        logger.error(f"Batch test execution failed: {str(e)}")


async def run_single_test(mode: str = "all"):
    """Run single test case example"""
    if mode == "single":
        try:
            # Set test parameters
            case_name = "MGX"
            url = "https://mgx.dev/"
            requirement = (
                "Please help me create an MGX official website. The website should include "
                "the following: 1. Homepage 2. Dialog box 3. AppWorld 4. Contact information"
            )
            json_path = f"data/{case_name}.json"

            # Initialize automated test role
            appeval = AppEvalRole(
                json_file=json_path,
                use_ocr=False,
                quad_split_ocr=False,
                use_memory=False,
                use_reflection=True,
                use_chrome_debugger=False,
                extend_xml_infos=True,
                log_dirs=f"work_dirs/{case_name}",
                max_iters=20,
            )

            # Execute single test
            result, executability = await appeval.run(case_name=case_name, url=url, user_requirement=requirement, json_path=json_path)
            logger.info(f"Single test execution result: {result}")
            logger.info(f"Executability: {executability}")

        except Exception as e:
            logger.error(f"Single test execution failed: {str(e)}")
            logger.exception("Detailed error information")
    elif mode == "api":
        try:
            # Set test parameters
            case_result_example = {
                "1": {
                    "task_name": "Example Task",
                    "url": "https://mgx.dev/",
                    "requirement": "Create a login page with username and password fields",
                    "tag": "1",
                    "test_cases": {
                        "0": {
                            "case_desc": "Verify successful login with valid username and password",
                            "result": "",
                            "evidence": ""
                        },
                        "1": {
                            "case_desc": "Verify login fails with invalid username and valid password",
                            "result": "",
                            "evidence": ""
                        },
                        "2": {
                            "case_desc": "Verify login fails with valid username and invalid password",
                            "result": "",
                            "evidence": ""
                        },
                        "3": {
                            "case_desc": "Verify login fails with empty username and valid password",
                            "result": "",
                            "evidence": ""
                        },
                        "4": {
                            "case_desc": "Verify login fails with valid username and empty password",
                            "result": "",
                            "evidence": ""
                        },
                        "5": {
                            "case_desc": "Verify login fails with empty username and empty password",
                            "result": "",
                            "evidence": ""
                        }
                    }
                }
            }
            test_cases = case_result_example["1"]["test_cases"]
            task_name = case_result_example["1"]["task_name"]
            url = case_result_example["1"]["url"]
            # Initialize automated test role
            json_path = f"data/{task_name}.json"
            appeval = AppEvalRole(
                json_file=json_path,
                use_ocr=False,
                quad_split_ocr=False,
                use_memory=False,
                use_reflection=True,
                use_chrome_debugger=False,
                extend_xml_infos=True,
                log_dirs=f"work_dirs/{task_name}",
                max_iters=20,
            )
            result, executability = await appeval.run_api(task_name=task_name, test_cases=test_cases, start_func=url, log_dir=f"work_dirs/{task_name}")
            logger.info(f"Batch test execution result: {result}")
            logger.info(f"Executability: {executability}")
        except Exception as e:
            logger.error(f"Single test execution failed: {str(e)}")
            logger.exception("Detailed error information")
    elif mode == "generate_case":
        try:
            # Set test parameters
            project_excel = r"data/test.xlsx"
            case_excel = r"data/test_results.xlsx"
            appeval = AppEvalRole(
                use_ocr=False,
                quad_split_ocr=False,
                use_memory=False,
                use_reflection=True,
                use_chrome_debugger=False,
                extend_xml_infos=True,
                log_dirs=f"work_dirs/{task_name}",
                max_iters=20,
            )
            case_result = await appeval.run_mini_batch(project_excel_path=project_excel, case_excel_path=case_excel, generate_case_only=True)
            logger.info(f"Batch test execution result: {case_result}")
        except Exception as e:
            logger.error(f"Single test execution failed: {str(e)}")
            logger.exception("Detailed error information")

async def main():
    """Main function"""
    # Run single test example
    # logger.info("Starting to execute single test...")
    # await run_single_test()

    # Run batch test example
    # logger.info("Starting to execute batch test...")
    # await run_batch_test()
    await run_api_test()


if __name__ == "__main__":
    asyncio.run(main())
