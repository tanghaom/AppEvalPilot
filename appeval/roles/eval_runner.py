#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/10/24
@File    : eval_runner.py
@Author  : tanghaoming
@Desc    : Refactored automated testing role for AppEval
"""
import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from metagpt.roles.role import Role, RoleContext
from metagpt.utils.common import read_json_file, write_json_file
from pydantic import ConfigDict, Field

from appeval.actions.case_generator import CaseGenerator, OperationType
from appeval.prompts.osagent import case_batch_check_system_prompt
from appeval.roles.osagent import OSAgent
from appeval.utils.excel_json_converter import (
    convert_json_to_excel,
    list_to_json,
    make_json_single,
    mini_list_to_excel,
    mini_list_to_json,
    update_project_excel_iters,
)
from appeval.utils.window_utils import kill_process, kill_windows, start_windows

# Constants for sleep times
SLEEP_AFTER_START_WEB = 10
SLEEP_AFTER_START_APP = 20
SLEEP_AFTER_CLEANUP = 5
SLEEP_BEFORE_EXECUTE = 30
SLEEP_BETWEEN_RETRIES = 5


class AppEvalContext(RoleContext):
    """AppEval Runtime Context"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    json_file: str = "data/default_results.json"
    env_process: Optional[Any] = None
    agent_params: Dict = Field(default_factory=dict)
    test_generator: Optional[CaseGenerator] = None
    test_cases: Optional[List[str]] = None


class AppEvalRole(Role):
    """Automated Testing Role"""

    name: str = "AppEvalRole"
    profile: str = "Automated Test Executor"
    goal: str = "Execute automated testing tasks"
    constraints: str = "Ensure accuracy and efficiency of test execution"

    rc: AppEvalContext = Field(default_factory=AppEvalContext)
    osagent: Optional[OSAgent] = None

    def __init__(self, json_file: str = "data/default_results.json", **kwargs):
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
            "use_tell_verifier": kwargs.get("use_tell_verifier", True),
            "log_dirs": kwargs.get("log_dirs", "work_dirs"),
            "max_iters": kwargs.get("max_iters", 20),
        }

        # Initialize CaseGenerator Action
        self.test_generator = CaseGenerator()

        # Initialize OSAgent
        self._init_osagent(**kwargs)

    def _init_osagent(self, **kwargs) -> None:
        """Initialize OSAgent"""
        add_info = """Before interacting with any web page, first browse the page completely from top to bottom by pressing Page Down to page through the content, so you get an overall understanding and can locate the required elements. If after a full scan you still cannot find the element, press Ctrl+F to search by visible keywords such as labels, button text, or field names. Clear the search and continue once the element is located.
If you need to interact with elements outside of a web popup, such as calendar or time selection popups, make sure to close the popup first. If the content in a text box is entered incorrectly, use the select all and delete actions to clear it, then re-enter the correct information.
To open a folder in File Explorer, please use a double-click.
If there is a problem with opening the web page, please do not keep trying to refresh the page or click repeatedly. After an attempt, please proceed directly to the remaining tasks.
Pay attention not to use shortcut keys to change the window size when testing on the web page.
If it involves the display effect of a web page on mobile devices, you can open the developer mode of the web page by pressing F12, and then use the shortcut key Ctrl+Shift+M to switch to the mobile view.
When testing game-related content, please pay close attention to judge whether the game functions are abnormal. If you find that no expected changes occur after certain operations, directly exit and mark this feature as negative.
Please use the Tell action to report the results of all test cases before executing Stop"""

        self.osagent = OSAgent(
            platform=kwargs.get("os_type", "Windows"),
            max_iters=self.rc.agent_params["max_iters"],
            use_ocr=self.rc.agent_params["use_ocr"],
            quad_split_ocr=self.rc.agent_params["quad_split_ocr"],
            use_icon_detect=False,
            use_icon_caption=True,
            use_memory=self.rc.agent_params["use_memory"],
            use_reflection=self.rc.agent_params["use_reflection"],
            use_som=False,
            extend_xml_infos=self.rc.agent_params["extend_xml_infos"],
            use_chrome_debugger=self.rc.agent_params["use_chrome_debugger"],
            use_tell_verifier=self.rc.agent_params["use_tell_verifier"],
            location_info="center",
            draw_text_box=False,
            log_dirs=self.rc.agent_params["log_dirs"],
            add_info=add_info,
            system_prompt=case_batch_check_system_prompt,
        )

    # ==================== Core Helper Methods ====================

    async def _start_environment(self, url: str = None, work_path: str = None) -> Optional[int]:
        """Start test environment (browser or application)"""
        if url:
            return await start_windows(target_url=url)
        if work_path:
            return await start_windows(work_path=work_path)
        return None

    async def _cleanup_environment(self, is_web: bool, pid: Optional[int] = None) -> None:
        """Clean up test environment"""
        processes = ["Chrome"] if is_web else [
            "Chrome", "cmd", "npm", "projectapp", "Edge"]
        await kill_windows(processes)
        if pid:
            await kill_process(pid)

    @staticmethod
    def _find_matching_key(key: Any, target_dict: dict) -> Optional[Any]:
        """Find matching key in target dict, handling both string and int key types.

        Args:
            key: The key to match (can be string or int)
            target_dict: The dictionary to search in

        Returns:
            The matching key if found, None otherwise
        """
        if key in target_dict:
            return key

        str_key = str(key)
        if str_key in target_dict:
            return str_key

        if str_key.isdigit():
            int_key = int(str_key)
            if int_key in target_dict:
                return int_key

        return None

    def _parse_results_from_tell(self, action_history: List[str]) -> Optional[dict]:
        """Parse test results from Tell action in action history

        Note: If use_tell_verifier is enabled, the Tell action content in action_history
        has already been verified and potentially corrected by TellVerifier in OSAgent._act().
        This ensures that hallucinated judgments (outcome hallucination, confirmation bias)
        are detected and corrected before being stored in history and parsed here.
        """
        if not action_history:
            return None

        content = action_history[-1]
        if not content.startswith("Tell ("):
            return None

        # Extract content between parentheses
        start_idx = content.find("(")
        end_idx = content.rfind(")")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None

        answer = content[start_idx + 1: end_idx]

        # Try direct eval
        try:
            return eval(answer)
        except Exception:
            pass

        # Try extracting dict from answer
        dict_start = answer.find("{")
        dict_end = answer.rfind("}")
        if dict_start != -1 and dict_end != -1 and dict_end > dict_start:
            try:
                return eval(answer[dict_start: dict_end + 1])
            except Exception as e:
                logger.error(f"Result parsing failed: {str(e)}")

        return None

    async def _execute_test_with_retry(
        self, task_id: str, task_id_case_number: int, check_list: dict, max_retries: int = 2
    ) -> tuple[List[str], str, List[str], str]:
        """Execute test with retry mechanism"""
        instruction = (
            "Please complete the following tasks，And after completion, use the Tell action to "
            f"inform me of the results of all the test cases at once: {check_list}\n"
        )

        for attempt in range(max_retries + 1):
            try:
                await self.osagent.run(instruction)
                return (self.osagent.rc.action_history, self.osagent.rc.task_list, self.osagent.rc.memory, self.osagent.rc.iter)
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed for task {task_id}, retrying... Error: {str(e)}")
                    await asyncio.sleep(SLEEP_BETWEEN_RETRIES)
                else:
                    logger.error(
                        f"All {max_retries + 1} attempts failed for task {task_id}. Error: {str(e)}")
                    raise

    async def _process_test_results(
        self,
        task_id: str,
        task_id_case_number: int,
        action_history: List[str],
        task_list: str,
        memory: List[str],
        iter_num: str,
        check_list: dict = None,
        return_dict: bool = False,
    ) -> Optional[dict]:
        """Process test results and either save to JSON or return as dict"""
        # Parse results from Tell action
        results_dict = self._parse_results_from_tell(action_history)

        # Generate results if parsing failed or incomplete
        if not results_dict or len(results_dict) != task_id_case_number:
            results_dict = await self.test_generator.generate_results_dict(action_history, task_list, memory, task_id_case_number, check_list)

        # Return dict for API mode
        if return_dict:
            return results_dict

        # Write to JSON for batch mode
        data = read_json_file(self.rc.json_file)
        data[task_id]["iters"] = iter_num
        for key, value in results_dict.items():
            matched_key = self._find_matching_key(
                key, data[task_id]["test_cases"])
            if matched_key is not None:
                data[task_id]["test_cases"][matched_key].update(
                    {"result": value.get("result", ""), "evidence": value.get("evidence", "")})
        write_json_file(self.rc.json_file, data, indent=4)
        return None

    async def execute_batch_check(self, task_id: str, task_id_case_number: int, check_list: dict) -> None:
        """Execute test and write results to JSON file"""
        logger.info(
            f"Start testing project {task_id}, log_dirs: {self.osagent.log_dirs}")

        try:
            action_history, task_list, memory, iter_num = await self._execute_test_with_retry(task_id, task_id_case_number, check_list)
            await self._process_test_results(task_id, task_id_case_number, action_history, task_list, memory, iter_num)
        except Exception as e:
            # Write failed result to JSON
            try:
                await self._process_test_results(task_id, task_id_case_number, ["Failed after all retries"], "Failed", [f"Error: {str(e)}"], "0")
            except Exception as write_error:
                logger.error(
                    f"Failed to write error result to JSON: {str(write_error)}")
                raise

    async def execute_api_check(self, task_id: str, task_id_case_number: int, check_list: dict) -> dict:
        """Execute test and return results as dictionary"""
        logger.info(
            f"Start testing project {task_id}, log_dirs: {self.osagent.log_dirs}")

        action_history, task_list, memory, iter_num = await self._execute_test_with_retry(task_id, task_id_case_number, check_list)
        return await self._process_test_results(
            task_id, task_id_case_number, action_history, task_list, memory, iter_num, check_list, return_dict=True
        )

    async def _execute_task_batch(self, test_cases: dict, max_retry_uncertain: int = 1, sequential_mode: bool = False) -> None:
        """Execute batch of test tasks with retry mechanism

        Args:
            test_cases: Dictionary of test cases to execute
            max_retry_uncertain: Maximum retries for uncertain cases
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
        """
        for task_id, task_info in test_cases.items():
            if "test_cases" not in task_info:
                continue

            start_func = (task_info.get("url") or task_info.get(
                "work_path") or "").strip()
            if not start_func:
                logger.warning(
                    f"No valid url or work_path for task {task_id}, skipping...")
                continue

            logger.info(f"Executing task: {task_id}")

            try:
                final_test_cases, _ = await self._run_test_with_retry(
                    task_name=task_id,
                    test_cases=task_info["test_cases"],
                    start_func=start_func,
                    log_dir="batch",
                    max_retry_uncertain=max_retry_uncertain,
                    save_to_file=False,
                    sequential_mode=sequential_mode,
                )
                task_info["test_cases"] = final_test_cases

            except Exception as e:
                logger.error(f"Failed to execute task {task_id}: {str(e)}")
                for case_id in task_info["test_cases"]:
                    task_info["test_cases"][case_id]["result"] = "Failed"
                    task_info["test_cases"][case_id]["evidence"] = f"Error: {str(e)}"

    async def _prepare_batch_test_cases(self, project_excel_path: str, operation_type: OperationType, converter_func) -> Optional[Any]:
        """Prepare test cases from Excel file"""
        if not project_excel_path:
            raise ValueError(
                "project_excel_path must be provided for batch run.")

        logger.info("Start generating automated test cases...")
        await self.test_generator.process_excel_file(project_excel_path, operation_type)

        logger.info("Start converting to JSON format...")
        return converter_func(project_excel_path, self.rc.json_file)

    async def _retry_uncertain_api_mode(self, task_name: str, uncertain_cases: dict, is_web: bool, start_func: str, retry_count: int) -> dict:
        """Retry uncertain cases in API mode"""
        uncertain_test_cases = uncertain_cases[task_name]["test_cases"]

        logger.info(f"Restarting environment for retry {retry_count}...")
        await self._start_environment(url=start_func if is_web else None, work_path=start_func if not is_web else None)
        await asyncio.sleep(SLEEP_BEFORE_EXECUTE)

        task_id_case_number = len(uncertain_test_cases)
        logger.info(
            f"Executing retry {retry_count} for {task_id_case_number} uncertain cases...")
        retry_result_dict = await self.execute_api_check(task_name, task_id_case_number, uncertain_test_cases)

        logger.info(f"Cleaning up environment after retry {retry_count}...")
        await self._cleanup_environment(is_web)
        await asyncio.sleep(SLEEP_AFTER_CLEANUP)

        # Construct and return retry result
        return {
            task_name: {
                "test_cases": {
                    key: {"result": value.get("result", ""), "evidence": value.get("evidence", "")} for key, value in retry_result_dict.items()
                }
            }
        }

    async def _retry_uncertain_single_mode(self, uncertain_cases: dict, json_path: str, result: dict, retry_count: int) -> dict:
        """Retry uncertain cases in Single mode"""
        retry_json_path = str(Path(json_path).parent /
                              f"{Path(json_path).stem}_retry_{retry_count}.json")
        write_json_file(retry_json_path, uncertain_cases, indent=4)
        self.rc.json_file = retry_json_path

        await self._execute_test_cases(uncertain_cases, log_dir_suffix=f"retry_{retry_count}")

        retry_result = read_json_file(retry_json_path)

        # Update original JSON file
        self.rc.json_file = json_path
        write_json_file(json_path, result, indent=4)

        # Clean up temporary file
        Path(retry_json_path).unlink(missing_ok=True)

        return retry_result

    async def _retry_uncertain_cases(
        self, result: dict, max_retry: int, task_name: str = None, start_func: str = None, json_path: str = None
    ) -> dict:
        """Retry uncertain cases (unified for both API and Single modes)"""
        retry_count = 0
        previous_uncertain_count = float("inf")
        original_log_dir = self.osagent.log_dirs
        is_api_mode = task_name is not None and start_func is not None
        is_web = (start_func.startswith("http://")
                  or start_func.startswith("https://")) if start_func else False

        while retry_count < max_retry:
            uncertain_cases = self._extract_uncertain_cases(result)
            should_retry, current_count = self._should_retry_uncertain(
                uncertain_cases, retry_count, max_retry, previous_uncertain_count)

            if not should_retry:
                break

            previous_uncertain_count = current_count
            self.osagent.log_dirs = f"{original_log_dir}/retry_{retry_count}"
            logger.info(
                f"Setting log_dirs to: {self.osagent.log_dirs} (retry)")

            # Execute retry based on mode
            if is_api_mode:
                retry_result = await self._retry_uncertain_api_mode(task_name, uncertain_cases, is_web, start_func, retry_count)
            else:
                retry_result = await self._retry_uncertain_single_mode(uncertain_cases, json_path, result, retry_count)

            result = self._merge_results(result, retry_result)
            self.osagent.log_dirs = original_log_dir
            retry_count += 1

        return result

    # ==================== Uncertain Cases Retry Methods ====================

    def _extract_uncertain_cases(self, test_result: dict) -> dict:
        """Extract test cases with uncertain results and clear result/evidence fields for retry"""
        uncertain_cases = {}

        for task_id, task_info in test_result.items():
            if "test_cases" not in task_info:
                continue

            uncertain_test_cases = {}
            for case_id, case_info in task_info["test_cases"].items():
                if case_info.get("result", "").lower() == "uncertain":
                    # Deep copy and clear result/evidence to treat as fresh test
                    clean_case = copy.deepcopy(case_info)
                    clean_case.pop("result", None)
                    clean_case.pop("evidence", None)
                    uncertain_test_cases[case_id] = clean_case

            if uncertain_test_cases:
                uncertain_cases[task_id] = copy.deepcopy(task_info)
                uncertain_cases[task_id]["test_cases"] = uncertain_test_cases

        return uncertain_cases

    def _merge_results(self, original_result: dict, retry_result: dict) -> dict:
        """Merge retry results into original results"""
        merged_result = copy.deepcopy(original_result)

        for task_id, task_info in retry_result.items():
            if task_id not in merged_result or "test_cases" not in task_info:
                continue

            for case_id, case_info in task_info["test_cases"].items():
                matched_key = self._find_matching_key(
                    case_id, merged_result[task_id]["test_cases"])
                if matched_key is not None:
                    merged_result[task_id]["test_cases"][matched_key]["result"] = case_info.get(
                        "result", "")
                    merged_result[task_id]["test_cases"][matched_key]["evidence"] = case_info.get(
                        "evidence", "")
                    logger.info(
                        f"Updated case {case_id} in task {task_id} with retry result: {case_info.get('result', '')}")

        return merged_result

    def _count_uncertain_cases(self, test_result: dict) -> int:
        """Count total number of uncertain cases"""
        return sum(len(task_info["test_cases"]) for task_info in test_result.values() if "test_cases" in task_info)

    def _should_retry_uncertain(self, uncertain_cases: dict, retry_count: int, max_retry: int, previous_count: int) -> tuple[bool, int]:
        """Determine if uncertain cases should be retried"""
        if not uncertain_cases:
            logger.info("No uncertain cases found, skipping retry")
            return False, 0

        if retry_count >= max_retry:
            return False, 0

        current_count = self._count_uncertain_cases(uncertain_cases)

        if current_count >= previous_count:
            logger.warning(
                f"No improvement in uncertain cases ({current_count} cases still uncertain), " f"stopping retry to avoid redundant testing"
            )
            return False, current_count

        logger.info(
            f"Found {current_count} uncertain cases, starting retry {retry_count + 1}/{max_retry}...")
        return True, current_count

    async def _execute_test_cases(self, test_cases: dict, log_dir_suffix: str = "") -> None:
        """Execute test cases with environment setup and cleanup"""
        for task_id, task_info in test_cases.items():
            if "test_cases" not in task_info:
                continue

            # Handle log directory
            original_log_dir = None
            if log_dir_suffix:
                original_log_dir = self.osagent.log_dirs
                self.osagent.log_dirs = f"{original_log_dir}/{log_dir_suffix}"
                logger.info(
                    f"Setting log_dirs to: {self.osagent.log_dirs} (retry)")

            # Start environment and wait
            is_web = "url" in task_info
            pid = await self._start_environment(url=task_info.get("url"), work_path=task_info.get("work_path"))
            await asyncio.sleep(SLEEP_AFTER_START_APP if not is_web else 0)
            await asyncio.sleep(SLEEP_AFTER_START_WEB)

            # Execute tests - pass only test_cases dict to maintain consistency
            await self.execute_batch_check(task_id, len(task_info["test_cases"]), task_info["test_cases"])

            # Restore and cleanup
            if original_log_dir:
                self.osagent.log_dirs = original_log_dir
            await self._cleanup_environment(is_web, pid)

    async def _run_test_with_retry(
        self,
        task_name: str,
        test_cases: dict,
        start_func: str,
        log_dir: str,
        max_retry_uncertain: int,
        save_to_file: bool = True,
        sequential_mode: bool = False,
    ) -> tuple[dict, bool]:
        """Core test execution logic with retry mechanism

        Args:
            task_name: Task identifier
            test_cases: Dictionary of test cases to execute
            start_func: URL or work path to start the environment
            log_dir: Directory for logs
            max_retry_uncertain: Maximum retries for uncertain cases
            save_to_file: Whether to save results to file
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
        """
        self.osagent.log_dirs = f"work_dirs/{log_dir}/{task_name}"
        is_web = start_func.startswith(
            "http://") or start_func.startswith("https://")

        # Start environment
        await self._start_environment(url=start_func if is_web else None, work_path=start_func if not is_web else None)
        await asyncio.sleep(SLEEP_BEFORE_EXECUTE)

        if sequential_mode:
            # Sequential mode: execute test cases one by one
            logger.info(
                f"Start executing automated testing in sequential mode ({len(test_cases)} cases)...")
            all_results = {}
            base_log_dir = self.osagent.log_dirs

            for idx, (case_id, case_info) in enumerate(test_cases.items(), 1):
                logger.info(
                    f"Executing test case {idx}/{len(test_cases)}: {case_id}")

                # Set case-specific log directory to avoid overwriting
                self.osagent.log_dirs = f"{base_log_dir}/{case_id}"

                # Create single case dict for execution
                single_case = {case_id: case_info}

                # Execute single test case
                result_dict = await self.execute_api_check(task_name, 1, single_case)

                # Merge result
                matched_key = self._find_matching_key(case_id, result_dict)
                if matched_key is not None:
                    all_results[case_id] = result_dict[matched_key]
                    test_cases[case_id].update(
                        {"result": result_dict[matched_key].get(
                            "result", ""), "evidence": result_dict[matched_key].get("evidence", "")}
                    )

                # Reset osagent state for next case (no browser cleanup)
                if idx < len(test_cases):
                    logger.info("Resetting osagent state for next case...")
                    self.osagent.rc.reset()

            # Restore base log directory
            self.osagent.log_dirs = base_log_dir
        else:
            # Batch mode: execute all test cases at once (original behavior)
            logger.info("Start executing automated testing...")
            result_dict = await self.execute_api_check(task_name, len(test_cases), test_cases)

            # Merge results
            for key, value in result_dict.items():
                matched_key = self._find_matching_key(key, test_cases)
                if matched_key is not None:
                    test_cases[matched_key].update({"result": value.get(
                        "result", ""), "evidence": value.get("evidence", "")})

        result = {task_name: {"test_cases": test_cases}}

        # Cleanup after initial test
        logger.info("Cleaning up environment after initial test run...")
        await self._cleanup_environment(is_web)
        await asyncio.sleep(SLEEP_AFTER_CLEANUP)

        # Retry uncertain cases
        result = await self._retry_uncertain_cases(result, max_retry_uncertain, task_name=task_name, start_func=start_func)

        final_test_cases = result[task_name]["test_cases"]

        # Save to file if needed
        if save_to_file:
            output_dir = Path("work_dirs") / log_dir / task_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{Path(task_name).name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"test_cases": final_test_cases},
                          f, indent=4, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")

        # Execute executability check
        executability = await self.test_generator.generate_executability(result, self.osagent.output_image_path)

        return final_test_cases, executability

    async def run_api(
        self,
        task_name: str,
        test_cases: dict,
        start_func: str,
        log_dir: str,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
    ) -> tuple[dict, bool]:
        """Run API testing with retry mechanism for uncertain results

        Args:
            task_name: Task identifier
            test_cases: Dictionary of test cases to execute
            start_func: URL or work path to start the environment
            log_dir: Directory for logs
            max_retry_uncertain: Maximum retries for uncertain cases
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
        """
        try:
            final_test_cases, executability = await self._run_test_with_retry(
                task_name=task_name,
                test_cases=test_cases,
                start_func=start_func,
                log_dir=log_dir,
                max_retry_uncertain=max_retry_uncertain,
                save_to_file=True,
                sequential_mode=sequential_mode,
            )
            logger.info("Test process completed")
            return final_test_cases, executability
        except Exception as e:
            logger.error(f"Error occurred during test execution: {str(e)}")
            await kill_windows(["Chrome", "cmd", "npm", "projectapp", "Edge"])
            raise

    async def run_single(
        self,
        case_name: str,
        url: str,
        work_path: str,
        user_requirement: str,
        json_path: str = "data/temp.json",
        use_json_only: bool = False,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
    ) -> tuple[dict, bool]:
        """Execute single test case with retry mechanism for uncertain results

        Args:
            case_name: Test case name
            url: Test target URL
            work_path: Work path for local application
            user_requirement: Requirement description
            json_path: Output JSON file path
            use_json_only: Whether to only use JSON files
            max_retry_uncertain: Maximum retries for uncertain cases
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).
        """
        # Generate test cases if needed
        if not use_json_only:
            logger.info(
                f"Start generating automated test cases for '{case_name}'...")
            generated_cases = await self.test_generator.generate_test_cases(user_requirement)
            logger.info("Start converting to JSON format...")
            make_json_single(case_name, url, generated_cases,
                             json_path, work_path)

        # Read and validate JSON
        self.rc.json_file = json_path
        test_data = read_json_file(json_path)

        if case_name not in test_data:
            raise ValueError(
                f"Case '{case_name}' not found in JSON file {json_path}")

        task_info = test_data[case_name]
        test_cases = task_info.get("test_cases", {})

        # Determine start function
        start_func = (task_info.get("url") or url or task_info.get(
            "work_path") or work_path or "").strip()
        if not start_func:
            raise ValueError(
                "No valid url or work_path provided for single test execution")

        # Execute tests with retry
        final_test_cases, executability = await self._run_test_with_retry(
            task_name=case_name,
            test_cases=test_cases,
            start_func=start_func,
            log_dir=f"single/{Path(json_path).stem}",
            max_retry_uncertain=max_retry_uncertain,
            save_to_file=False,
            sequential_mode=sequential_mode,
        )

        # Update and save results
        task_info["test_cases"] = final_test_cases
        test_data[case_name] = task_info
        write_json_file(json_path, test_data, indent=4)

        logger.info(f"Test process completed for case '{case_name}'")
        return test_data, executability

    async def run_batch(
        self,
        project_excel_path: str = None,
        case_excel_path: str = None,
        batch_mode: str = "standard",
        generate_case_only: bool = False,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
    ) -> Union[tuple[dict, bool], Any, None]:
        """Run batch testing (unified for both standard and mini modes)

        Complete testing process includes:
        1. Generate test cases from Excel
        2. Convert to JSON format
        3. Execute automated testing
        4. Retry uncertain cases (default enabled)
        5. (Optional) Output results to Excel
        6. (Optional) Only generate test cases

        Args:
            project_excel_path: Project level Excel file path
            case_excel_path: Case level Excel file path (optional)
            batch_mode: Batch mode - "standard" or "mini" (default: "standard")
            generate_case_only: Whether to only generate test cases (only for mini mode)
            max_retry_uncertain: Maximum retry times for uncertain cases (default: 1)
            sequential_mode: If True, execute test cases one by one without browser cleanup between cases,
                           only reset osagent state. If False, execute all test cases at once (default).

        Returns:
            - For standard mode: tuple[dict, bool] (result dict and executability)
            - For mini mode with generate_case_only: Case result
            - For mini mode without generate_case_only: None
        """
        try:
            # Select converters based on mode
            is_mini = batch_mode == "mini"
            operation_type = OperationType.GENERATE_CASES_MINI_BATCH if is_mini else OperationType.GENERATE_CASES
            json_converter = mini_list_to_json if is_mini else list_to_json
            excel_converter = mini_list_to_excel if is_mini else convert_json_to_excel

            # Prepare test cases
            case_result = await self._prepare_batch_test_cases(project_excel_path, operation_type, json_converter)

            # Return early if only generating cases (mini mode only)
            if generate_case_only:
                if not is_mini:
                    raise ValueError(
                        "generate_case_only is only supported for mini batch mode.")
                return case_result

            # Execute tests with retry support
            logger.info("Start executing automated testing...")
            test_cases = read_json_file(self.rc.json_file)
            await self._execute_task_batch(test_cases, max_retry_uncertain=max_retry_uncertain, sequential_mode=sequential_mode)
            write_json_file(self.rc.json_file, test_cases, indent=4)

            # Output results to Excel
            if case_excel_path:
                logger.info("Start generating result spreadsheet...")
                excel_converter(self.rc.json_file, case_excel_path)

            # Update project Excel with iteration counts
            logger.info("Updating project Excel with iteration counts...")
            update_project_excel_iters(project_excel_path, self.rc.json_file)

            logger.info("Test process completed")
            return ({}, None) if batch_mode == "standard" else None

        except Exception as e:
            logger.error(f"Error occurred during test execution: {str(e)}")
            await kill_windows(["Chrome", "cmd", "npm", "projectapp", "Edge"])
            raise

    async def run_mini_batch(
        self,
        project_excel_path: str = None,
        case_excel_path: str = None,
        generate_case_only: bool = False,
        max_retry_uncertain: int = 1,
        sequential_mode: bool = False,
    ) -> Optional[Any]:
        """Deprecated: Use run_batch(batch_mode='mini') instead"""
        return await self.run_batch(
            project_excel_path=project_excel_path,
            case_excel_path=case_excel_path,
            batch_mode="mini",
            generate_case_only=generate_case_only,
            max_retry_uncertain=max_retry_uncertain,
            sequential_mode=sequential_mode,
        )

    async def run(self, **kwargs) -> Union[tuple[dict, bool], dict, Exception]:
        """Run automated testing

        Supports two calling methods:
        1. Batch testing: run(project_excel_path="xxx.xlsx", case_excel_path="xxx.xlsx", use_json_only=False)
        2. Single test: run(case_name="xxx", url="xxx", user_requirement="xxx")

        Args:
            **kwargs: Parameters
                Batch testing:
                    - project_excel_path: Project level Excel file path
                    - case_excel_path: Case level Excel file path (optional)
                    - use_json_only: Whether to only use JSON files (optional)
                    - sequential_mode: If True, execute test cases one by one (optional)
                Single test:
                    - case_name: Test case name
                    - url: Test target URL
                    - user_requirement: Requirement description
                    - json_path: Output JSON file path (optional)
                    - sequential_mode: If True, execute test cases one by one (optional)
        """
        try:
            if kwargs.get("case_name") and kwargs.get("user_requirement"):
                # Single test scenario
                return await self.run_single(
                    case_name=kwargs["case_name"],
                    url=kwargs.get("url", ""),
                    work_path=kwargs.get("work_path", ""),
                    user_requirement=kwargs["user_requirement"],
                    json_path=kwargs.get("json_path", "data/temp.json"),
                    use_json_only=kwargs.get("use_json_only", False),
                    sequential_mode=kwargs.get("sequential_mode", False),
                )
            else:
                # Batch test scenario
                return await self.run_batch(
                    kwargs.get("project_excel_path"),
                    kwargs.get("case_excel_path"),
                    sequential_mode=kwargs.get("sequential_mode", False),
                )
        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            logger.exception("Detailed error information")
            return e
