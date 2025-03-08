#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/03/21
@File    : appeval.py
@Desc    : 自动化测试角色
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from metagpt.actions import Action
from metagpt.roles.role import Role
from metagpt.schema import Message
from pydantic import BaseModel, ConfigDict, Field

from appeval.actions.test_generator import TestGeneratorAction
from appeval.prompts.appeval import batch_check_prompt
from appeval.roles.osagent import OSAgent
from appeval.utils.excel2json import list_to_json, make_json_single
from appeval.utils.json2excel import convert_json_to_excel


class AppEvalContext(BaseModel):
    """AppEval运行时上下文"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    date_str: str = Field(default_factory=lambda: datetime.now().strftime("%m-%d"))
    json_file: str = ""
    env_process: Optional[Any] = None
    agent_params: Dict = Field(default_factory=dict)
    osagent: Optional[OSAgent] = None
    test_generator: Optional[TestGeneratorAction] = None


class AppEvalRole(Role):
    """自动化测试角色"""

    name: str = "AppEvalRole"
    profile: str = "自动化测试执行者"
    goal: str = "执行自动化测试任务"
    constraints: str = "确保测试执行的准确性和效率"

    rc: AppEvalContext = Field(default_factory=AppEvalContext)

    def __init__(self, json_file: str, **kwargs):
        super().__init__()
        self.rc.json_file = json_file

        # 初始化 agent_params
        self.rc.agent_params = {
            "use_ocr": kwargs.get("use_ocr", True),
            "quad_split_ocr": kwargs.get("quad_split_ocr", True),
            "use_memory": kwargs.get("use_memory", True),
            "use_reflection": kwargs.get("use_reflection", True),
            "use_chrome_debugger": kwargs.get("use_chrome_debugger", True),
            "extend_xml_infos": kwargs.get("extend_xml_infos", True),
            "log_dirs": f"work_dirs/{self.rc.date_str}",
        }

        # 初始化 TestGeneratorAction
        self.rc.test_generator = TestGeneratorAction()

        # 初始化 OSAgent
        self._init_osagent(**kwargs)

    def _init_osagent(self, **kwargs) -> None:
        """初始化 OSAgent"""
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

        self.rc.osagent = OSAgent(
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
            system_prompt=batch_check_prompt(),
        )

    @staticmethod
    async def _start_env(env_script: str, web_url: str) -> None:
        """异步启动环境"""
        if env_script:
            logger.info(f"启动环境: {env_script}")
            script_path = os.path.join(os.path.dirname(__file__), "..", "data", env_script)
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"环境脚本不存在: {script_path}")

            await asyncio.create_subprocess_shell(f"{script_path} {web_url}", env=os.environ)
            await asyncio.sleep(5)

    @staticmethod
    async def _kill_env() -> None:
        """异步关闭环境"""
        try:
            logger.info("关闭 Chrome 浏览器...")
            process = await asyncio.create_subprocess_exec(
                "taskkill", "/IM", "chrome.exe", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            logger.info(stdout.decode())
            logger.info(stderr.decode())
        except Exception as e:
            logger.error(f"关闭环境失败: {str(e)}")
            raise

    async def execute_batch_check(self, task_id: str, task_id_case_number: int, check_list: dict) -> None:
        """执行单个检验条件"""
        logger.info(f"开始测试项目{task_id}")
        instruction = f"Please complete the following tasks，And after completion, use the Tell action to inform me of the results of all the test cases at once: {check_list}\n"
        await self.rc.osagent.run(instruction)

        # 获取动作历史
        action_history = self.rc.osagent.rc.action_history
        task_list = self.rc.osagent.rc.task_list
        memory = self.rc.osagent.rc.memory

        self.write_batch_res_to_json(task_id, task_id_case_number, action_history, task_list, memory)

    async def write_batch_res_to_json(
        self, task_id: str, task_id_case_number: int, action_history: List[str], task_list: str, memory: List[str]
    ) -> None:
        """将检验结果写入json文件"""
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
                                logger.error(f"结果解析失败: {str(e)}")

            if not results_dict or len(results_dict) != task_id_case_number:
                # 使用 TestGeneratorAction 替代原来的 generate_results 函数
                results_dict = await self.rc.test_generator.generate_results_dict(
                    action_history, task_list, memory, task_id_case_number
                )

            with open(self.rc.json_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                for key, value in results_dict.items():
                    data[task_id]["测试用例"][key].update({"result": value["result"], "evidence": value["evidence"]})
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()

        except Exception as e:
            logger.error(f"写入检验结果失败: {str(e)}")
            raise

    async def run_batch(self, project_excel_path: str = None, case_excel_path: str = None) -> None:
        """运行批量测试

        完整的测试流程包括:
        1. 从Excel生成测试用例
        2. 转换为JSON格式
        3. 执行自动化测试
        4. (可选)输出结果到Excel

        Args:
            project_excel_path: 项目级别的Excel文件路径
            case_excel_path: 用例级别的Excel文件路径(可选)
        """
        try:
            if project_excel_path:
                # 1. 生成自动测试样例
                logger.info("开始生成自动测试样例...")
                await self.rc.test_generator.process_excel_file(project_excel_path, "generate_cases")

                # 2. 转换为JSON格式
                logger.info("开始转换为JSON格式...")
                list_to_json(project_excel_path, self.rc.json_file)
            else:
                raise ValueError("project_excel_path must be provided for batch run if not using existing json file.")

            # 3. 执行自动化测试
            logger.info("开始执行自动化测试...")
            with open(self.rc.json_file, "r", encoding="utf-8") as f:
                test_cases = json.load(f)

            for task_id, task_info in test_cases.items():
                self.rc.osagent.log_dirs = f"work_dirs/{self.rc.date_str}/{task_id}"

                if "测试用例" in task_info:
                    if "url" in task_info:
                        await self._start_env("chrome.bat", task_info["url"])
                    await asyncio.sleep(1)

                    task_id_case_number = len(test_cases[task_id]["测试用例"])
                    await self.execute_batch_check(task_id, task_id_case_number, task_info)
                    await self._kill_env()

            # 4. 输出结果到Excel(如果提供了case_excel_path)
            if case_excel_path:
                logger.info("开始生成结果表格...")
                convert_json_to_excel(self.rc.json_file, case_excel_path)

            logger.info("测试流程执行完成")

        except Exception as e:
            logger.error(f"执行测试时发生错误: {str(e)}")
            raise

    async def run_single(
        self, case_name: str, url: str, user_requirement: str, json_path: str = "data/temp.json"
    ) -> dict:
        """执行单个测试用例

        Args:
            case_name (str): 测试用例名称
            url (str): 测试目标URL
            user_requirement (str): 测试需求描述
            json_path (str, optional): 输出JSON文件路径

        Returns:
            dict: 测试结果字典
        """
        try:
            # 1. 生成自动测试样例
            logger.info(f"开始为用例 '{case_name}' 生成自动测试样例...")
            test_cases = await self.rc.test_generator.generate_test_cases(user_requirement)

            # 2. 转换为JSON格式
            logger.info("开始转换为JSON格式...")
            make_json_single(case_name, url, test_cases, json_path)

            # 3. 执行自动化测试
            logger.info("开始执行自动化测试...")
            self.rc.json_file = json_path
            result = await self.run_batch()

            # 4. 读取结果
            with open(self.rc.json_file, "r", encoding="utf-8") as f:
                result = json.load(f)

            logger.info(f"用例 '{case_name}' 测试流程执行完成")
            return result

        except Exception as e:
            logger.error(f"测试流程执行失败: {str(e)}")
            raise

    async def run(self, **kwargs) -> Message:
        """运行自动化测试

        支持两种调用方式:
        1. 批量测试: run(project_excel_path="xxx.xlsx", case_excel_path="xxx.xlsx")
        2. 单个测试: run(case_name="xxx", url="xxx", user_requirement="xxx")

        Args:
            **kwargs: 参数
                批量测试:
                    - project_excel_path: 项目级别Excel文件路径
                    - case_excel_path: 用例级别Excel文件路径(可选)
                单个测试:
                    - case_name: 测试用例名称
                    - url: 测试目标URL
                    - user_requirement: 需求描述
                    - json_path: 输出JSON文件路径(可选)
        """
        try:
            if kwargs.get("case_name") and kwargs.get("user_requirement"):
                # 单个测试场景
                result = await self.run_single(
                    kwargs["case_name"],
                    kwargs["url"],
                    kwargs["user_requirement"],
                    kwargs.get("json_path", "data/temp.json"),
                )
                return Message(content=str(result), cause_by=Action)
            else:
                # 批量测试场景
                await self.run_batch(kwargs.get("project_excel_path"), kwargs.get("case_excel_path"))
                return Message(content="测试执行完成", cause_by=Action)
        except Exception as e:
            logger.error(f"执行测试失败: {str(e)}")
            return Message(content=f"测试执行失败: {str(e)}", cause_by=Action)
