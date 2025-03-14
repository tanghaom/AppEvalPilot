#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@File    : test_runner.py
@Author  : bianyutong
@Desc    : Automated Testing Role
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from metagpt.actions import Action
from metagpt.roles.role import Role, RoleContext
from metagpt.schema import Message
from pydantic import ConfigDict, Field

from appeval.actions.case_generator import CaseGenerator
from appeval.prompts.test_runner import batch_check_prompt
from appeval.roles.osagent import OSAgent
from appeval.utils.excel_json_converter import (
    convert_json_to_excel,
    list_to_json,
    make_json_single,
)
from appeval.utils.window_utils import kill_process, start_windows


class AppEvalContext(RoleContext):
    """AppEval Runtime Context"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    json_file: str = ""
    env_process: Optional[Any] = None
    agent_params: Dict = Field(default_factory=dict)
    test_generator: Optional[CaseGenerator] = None


class AppEvalRole(Role):
    """Automated Testing Role"""

    name: str = "AppEvalRole"
    profile: str = "Automated Test Executor"
    goal: str = "Execute automated testing tasks"
    constraints: str = "Ensure accuracy and efficiency of test execution"

    rc: AppEvalContext = Field(default_factory=AppEvalContext)
    osagent: Optional[OSAgent] = None

    def __init__(self, json_file: str, **kwargs):
        super().__init__()
        self.rc.json_file = json_file

        # Initialize agent_params
        self.rc.agent_params = {
            "use_ocr": kwargs.get("use_ocr", True),
            "quad_split_ocr": kwargs.get("quad_split_ocr", True),
            "use_memory": kwargs.get("use_memory", True),
            "use_reflection": kwargs.get("use_reflection", True),
            "use_chrome_debugger": kwargs.get("use_chrome_debugger", True),
            "extend_xml_infos": kwargs.get("extend_xml_infos", True),
            "log_dirs": kwargs.get("log_dirs", "work_dirs"),
        }

        # Initialize CaseGenerator Action
        self.rc.test_generator = CaseGenerator()

        # Initialize OSAgent
        self._init_osagent(**kwargs)

    def _init_osagent(self, **kwargs) -> None:
        """Initialize OSAgent"""
        add_info = (
            "If you need to interact with elements outside of a web popup, such as calendar or time selection "
            "popups, make sure to close the popup first. If the content in a text box is entered incorrectly, "
            "use the select all and delete actions to clear it, then re-enter the correct information. "
            "To open a folder in File Explorer, please use a double-click. "
            "If there is a problem with opening the web page, please do not keep trying to "
            "refresh the page or click repeatedly. After an attempt, please proceed directly to the remaining tasks. "
            "Pay attention not to use shortcut keys to change the window size when testing on the web page. "
            "If it involves the display effect of a web page on mobile devices, you can open the developer mode of the web page by pressing F12, "
            "and then use the shortcut key Ctrl+Shift+M to switch to the mobile view. "
            "When testing game-related content, please pay close attention to judge whether the game functions are abnormal. "
            "If you find that no expected changes occur after certain operations, directly exit and mark this feature as negative."
            "Please use the Tell action to report the results of all test cases before executing Stop"
        )

        self.osagent = OSAgent(
            platform=kwargs.get("os_type", "Windows"),
            max_iters=40,
            use_ocr=self.rc.agent_params["use_ocr"],
            quad_split_ocr=self.rc.agent_params["quad_split_ocr"],
            use_icon_detect=False,
            use_icon_caption=True,
            use_memory=self.rc.agent_params["use_memory"],
            use_reflection=self.rc.agent_params["use_reflection"],
            use_som=False,
            extend_xml_infos=self.rc.agent_params["extend_xml_infos"],
            use_chrome_debugger=self.rc.agent_params["use_chrome_debugger"],
            location_info="center",
            draw_text_box=False,
            log_dirs=self.rc.agent_params["log_dirs"],
            add_info=add_info,
            system_prompt=batch_check_prompt,
        )

    async def execute_batch_check(self, task_id: str, task_id_case_number: int, check_list: dict) -> None:
        """Execute single verification condition"""
        logger.info(f"Start testing project {task_id}")
        logger.info(f"Setting log_dirs to: {self.osagent.log_dirs}")
        instruction = f"Please complete the following tasks，And after completion, use the Tell action to inform me of the results of all the test cases at once: {check_list}\n"
        await self.osagent.run(instruction)

        # Get action history
        action_history = self.osagent.rc.action_history
        task_list = self.osagent.rc.task_list
        memory = self.osagent.rc.memory

        await self.write_batch_res_to_json(task_id, task_id_case_number, action_history, task_list, memory)

    async def write_batch_res_to_json(
        self, task_id: str, task_id_case_number: int, action_history: List[str], task_list: str, memory: List[str]
    ) -> None:
        """Write verification results to json file"""
        try:
            content = action_history[-1]
            results_dict = None

            if content.startswith("Tell ("):
                start = content.find("(")
                end = content.rfind(")")
                if start != -1 and end != -1 and end > start:
                    answer = content[start + 1 : end]
                    try:
                        results_dict = eval(answer)
                    except:
                        start = answer.find("{")
                        end = answer.rfind("}")
                        if start != -1 and end != -1 and end > start:
                            try:
                                results_dict = eval(answer[start : end + 1])
                            except Exception as e:
                                logger.error(f"Result parsing failed: {str(e)}")

            if not results_dict or len(results_dict) != task_id_case_number:
                results_dict = await self.rc.test_generator.generate_results_dict(
                    action_history, task_list, memory, task_id_case_number
                )

            with open(self.rc.json_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                for key, value in results_dict.items():
                    data[task_id]["test_cases"][key].update({"result": value["result"], "evidence": value["evidence"]})
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()

        except Exception as e:
            logger.error(f"Failed to write verification results: {str(e)}")
            raise

    async def run_batch(self, project_excel_path: str = None, case_excel_path: str = None) -> None:
        """Run batch testing

        Complete testing process includes:
        1. Generate test cases from Excel
        2. Convert to JSON format
        3. Execute automated testing
        4. (Optional) Output results to Excel

        Args:
            project_excel_path: Project level Excel file path
            case_excel_path: Case level Excel file path (optional)
        """
        try:
            if project_excel_path:
                # 1. Generate automated test cases
                logger.info("Start generating automated test cases...")
                await self.rc.test_generator.process_excel_file(project_excel_path, "generate_cases")

                # 2. Convert to JSON format
                logger.info("Start converting to JSON format...")
                list_to_json(project_excel_path, self.rc.json_file)
            else:
                raise ValueError("project_excel_path must be provided for batch run if not using existing json file.")

            # 3. Execute automated testing
            logger.info("Start executing automated testing...")
            with open(self.rc.json_file, "r", encoding="utf-8") as f:
                test_cases = json.load(f)

            for task_id, task_info in test_cases.items():
                self.osagent.log_dirs = f"work_dirs/{task_id}"

                if "test_cases" in task_info:
                    if "url" in task_info:
                        pid = await start_windows(task_info["url"])
                    await asyncio.sleep(10)

                    task_id_case_number = len(test_cases[task_id]["test_cases"])
                    await self.execute_batch_check(task_id, task_id_case_number, task_info)
                    if "url" in task_info:
                        await kill_process(pid)

            # 4. Output results to Excel (if case_excel_path is provided)
            if case_excel_path:
                logger.info("Start generating result spreadsheet...")
                convert_json_to_excel(self.rc.json_file, case_excel_path)

            logger.info("Test process completed")

        except Exception as e:
            logger.error(f"Error occurred during test execution: {str(e)}")
            raise

    async def run_single(
        self, case_name: str, url: str, user_requirement: str, json_path: str = "data/temp.json"
    ) -> dict:
        """Execute single test case

        Args:
            case_name (str): Test case name
            url (str): Test target URL
            user_requirement (str): Test requirement description
            json_path (str, optional): Output JSON file path

        Returns:
            dict: Test result dictionary
        """
        try:
            # 1. Generate automated test cases
            logger.info(f"Start generating automated test cases for '{case_name}'...")
            test_cases = await self.rc.test_generator.generate_test_cases(user_requirement)

            # 2. Convert to JSON format
            logger.info("Start converting to JSON format...")
            make_json_single(case_name, url, test_cases, json_path)

            # 3. Execute automated testing
            logger.info("Start executing automated testing...")
            self.rc.json_file = json_path

            with open(self.rc.json_file, "r", encoding="utf-8") as f:
                test_cases = json.load(f)

            for task_id, task_info in test_cases.items():
                if "test_cases" in task_info:
                    if "url" in task_info:
                        pid = await start_windows(task_info["url"])
                    await asyncio.sleep(10)

                    task_id_case_number = len(test_cases[task_id]["test_cases"])
                    await self.execute_batch_check(task_id, task_id_case_number, task_info)
                    if "url" in task_info:
                        await kill_process(pid)

            # 4. Read results
            with open(self.rc.json_file, "r", encoding="utf-8") as f:
                result = json.load(f)

            logger.info(f"Test process completed for case '{case_name}'")
            return result

        except Exception as e:
            logger.error(f"Test process execution failed: {str(e)}")
            raise

    async def run(self, **kwargs) -> Message:
        """Run automated testing

        Supports two calling methods:
        1. Batch testing: run(project_excel_path="xxx.xlsx", case_excel_path="xxx.xlsx")
        2. Single test: run(case_name="xxx", url="xxx", user_requirement="xxx")

        Args:
            **kwargs: Parameters
                Batch testing:
                    - project_excel_path: Project level Excel file path
                    - case_excel_path: Case level Excel file path (optional)
                Single test:
                    - case_name: Test case name
                    - url: Test target URL
                    - user_requirement: Requirement description
                    - json_path: Output JSON file path (optional)
        """
        try:
            if kwargs.get("case_name") and kwargs.get("user_requirement"):
                # Single test scenario
                result = await self.run_single(
                    kwargs["case_name"],
                    kwargs["url"],
                    kwargs["user_requirement"],
                    kwargs.get("json_path", "data/temp.json"),
                )
                return Message(content=json.dumps(result), cause_by=Action)
            else:
                # Batch test scenario
                await self.run_batch(kwargs.get("project_excel_path"), kwargs.get("case_excel_path"))
                return Message(content=json.dumps("Test execution completed"), cause_by=Action)
        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            logger.exception("Detailed error information")
            return Message(content=json.dumps(f"Test execution failed: {str(e)}"), cause_by=Action)
