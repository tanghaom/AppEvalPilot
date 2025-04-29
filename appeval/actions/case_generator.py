#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/07
@File    : case_generator.py
@Desc    : Action for generating and validating test cases
"""
from enum import Enum
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from metagpt.actions.action import Action
from metagpt.config2 import Config
from metagpt.llm import LLM
from metagpt.logs import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from appeval.prompts.case_generator import CasePrompts


class OperationType(Enum):
    GENERATE_CASES = "generate_cases"
    MAKE_CASE_NAME = "make_case_name"
    CHECK_RESULTS = "check_results"
    GENERATE_CASES_MINI_BATCH = "generate_cases_mini_batch"

class CaseGenerator(Action):
    name: str = "CaseGenerator"
    desc: str = "Action for generating and validating test cases"

    def __init__(self, config_path: str = "config/config2.yaml"):
        super().__init__()
        self.config_path = Path(config_path)
        # Load configuration
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file).get("case_generator")
            self.config = Config.from_llm_config(config)
        logger.info(f"CaseGenerator Config: {self.config}")
        self.llm = LLM(self.config.llm)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"_inference_chat failed, {retry_state.attempt_number}th retry: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _inference_chat(self, content: str) -> str:
        """Use MetaGPT's aask method for chat inference

        Args:
            content: Input content

        Returns:
            str: Response content
        """
        try:
            response = await self.llm.aask(
                content,
                system_msgs=[CasePrompts.SYSTEM_MESSAGE],
                stream=False,
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    async def generate_test_cases(self, demand: str) -> List[str]:
        """Generate test cases based on requirements

        Args:
            demand: User requirement description

        Returns:
            List[str]: List of test cases
        """
        try:
            prompt = CasePrompts.GENERATE_CASES.format(demand=demand)
            logger.info(f"Original requirement: {demand}")
            # Call chat to generate test cases
            answer = await self._inference_chat(prompt)
            # Convert string to list
            test_cases = eval(answer)
            return test_cases

        except Exception as e:
            logger.error(f"Error occurred while generating test cases: {str(e)}")
            raise
    
    async def generate_test_cases_mini_batch(self, demand: str) -> List[str]:
        """Generate test cases based on requirements

        Args:
            demand: User requirement description

        Returns:
            List[List[str], List[str], ...]: List of test case lists, where each inner list contains related test cases for a category
        """
        try:
            prompt = CasePrompts.GENERATE_CASES_MINI_BATCH.format(demand=demand)
            logger.info(f"Original requirement: {demand}")
            # Call chat to generate test cases
            answer = await self._inference_chat(prompt)
            # Convert string to list
            test_cases = eval(answer)
            return test_cases

        except Exception as e:
            logger.error(f"Error occurred while generating test cases: {str(e)}")
            raise

    async def generate_case_name(self, case_desc: str) -> str:
        """Generate a short case_name for the test case

        Args:
            case_desc: Test case description

        Returns:
            str: Generated case_name
        """
        if not case_desc:
            return ""

        prompt = CasePrompts.GENERATE_CASE_NAME.format(case_desc=case_desc)
        case_name = await self._inference_chat(prompt)
        return case_name.strip()

    async def check_result(self, case_desc: str, model_output: str) -> str:
        """Check if test results meet expectations

        Args:
            case_desc: Test case description
            model_output: Model output result

        Returns:
            str: "Yes", "No" or "Uncertain"
        """
        if not case_desc or not model_output:
            return "Uncertain"

        prompt = CasePrompts.CHECK_RESULT.format(case_desc=case_desc, model_output=model_output)
        answer = await self._inference_chat(prompt)
        return answer.strip()

    async def generate_results_dict(
        self, action_history: List[str], task_list: str, memory: List[str], task_id_case_number: int
    ) -> Dict:
        """Generate result dictionary based on historical information

        Args:
            action_history: List of historical operation information
            task_list: Task list information
            memory: Memory history information
            task_id_case_number: Number of tasks

        Returns:
            Dict: Result dictionary
        """
        try:
            prompt = CasePrompts.GENERATE_RESULTS.format(
                action_history=action_history,
                task_list=task_list,
                memory=memory,
                task_id_case_number=task_id_case_number,
            )
            logger.info(f"History information length: {len(str(action_history))}")
            # Call chat to generate results
            answer = await self._inference_chat(prompt)
            logger.info(f"answer: {answer}")
            # Convert string to dictionary
            results = eval(answer)
            return results

        except Exception as e:
            logger.error(f"Error occurred while generating result dictionary: {str(e)}")
            raise

    async def process_excel_file(
        self, excel_path: str, operation: OperationType = OperationType.GENERATE_CASES
    ) -> None:
        """Process Excel file

        Args:
            excel_path: Excel file path
            operation: Operation type, supports GENERATE_CASES, MAKE_CASE_NAME, CHECK_RESULTS
        """
        excel_path = Path(excel_path)
        if not excel_path.exists():
            raise FileNotFoundError(f"File does not exist: {excel_path}")

        # Read Excel file
        df = pd.read_excel(excel_path)

        if df.empty:
            logger.warning("Excel file is empty")
            return

        if operation == OperationType.GENERATE_CASES:
            # Process each row to generate test cases
            for index, row in df.iterrows():
                ori_demand = str(row["requirement"])
                if not ori_demand:
                    continue

                test_cases = await self.generate_test_cases(ori_demand)
                df.at[index, "Auto Generated Test Cases"] = str(test_cases)
                # Save after processing each row
                df.to_excel(excel_path, index=False)
        elif operation == OperationType.GENERATE_CASES_MINI_BATCH:
            # Process each row to generate test cases
            for index, row in df.iterrows():
                ori_demand = str(row["requirement"])
                if not ori_demand:
                    continue
                test_cases = await self.generate_test_cases_mini_batch(ori_demand)
                df.at[index, "Auto Generated Test Cases"] = str(test_cases)
                # Save after processing each row
                df.to_excel(excel_path, index=False)
        elif operation == OperationType.MAKE_CASE_NAME:
            # Generate case_name for each test case
            for index, row in df.iterrows():
                task_desc = str(row["case_desc"])
                if not task_desc:
                    continue

                case_name = await self.generate_case_name(task_desc)
                df.at[index, "case_name"] = case_name
                df.to_excel(excel_path, index=False)

        elif operation == OperationType.CHECK_RESULTS:
            # Check each test result
            for index, row in df.iterrows():
                task_desc = str(row["case_desc"])
                model_output = str(row["os_output"])
                if not task_desc or not model_output:
                    continue

                result = await self.check_result(task_desc, model_output)
                df.at[index, "Auto Function Detection"] = result
                df.to_excel(excel_path, index=False)

        else:
            raise ValueError(f"Unsupported operation type: {operation}")

        logger.info(f"{operation.value} operation completed and saved")
