#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/07
@File    : test_generator.py
@Desc    : 用于生成和验证测试用例的Action
"""
import asyncio
import os
import warnings
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from loguru import logger
from metagpt.actions.action import Action
from metagpt.logs import logger
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

# 忽略警告
warnings.filterwarnings("ignore")


class TestGeneratorAction(Action):
    name: str = "TestGeneratorAction"
    desc: str = "用于生成和验证测试用例的Action"

    def __init__(self, config_path: str = "config/config2.yaml"):
        super().__init__()
        self.config_path = config_path
        # 读取配置
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file).get("appeval")
            self.model = config.get("model")
            self.base_url = config.get("base_url")
            self.api_key = config.get("api_key")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, log_level=logger.level("WARNING")),
        after=after_log(logger, log_level=logger.level("INFO")),
        reraise=True,
    )
    async def _inference_chat(self, content: str) -> Tuple[str, int]:
        """使用MetaGPT的aask方法进行聊天推理

        Args:
            content: 输入内容

        Returns:
            tuple: (响应内容, token使用量)
        """
        try:
            system_message = "你是一个专业的测试工程师，擅长生成和验证测试用例。"
            response = await self.llm.aask(
                content,
                system_msgs=[system_message],
                stream=False,
            )
            return response
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise

    async def generate_test_cases(self, demand: str) -> List[str]:
        """根据需求生成测试用例

        Args:
            demand: 用户需求描述

        Returns:
            List[str]: 测试用例列表
        """
        try:
            # 构造提示语
            prompt = (
                "你是一个专业的测试工程师。请根据以下用户对该网页的要求，生成一系列具体的测试用例。\n"
                "要求：\n"
                "1. 测试用例必须完全围绕用户要求生成，绝对不能遗漏任何用户的要求\n"
                "2. 请以Python列表形式返回所有测试用例\n"
                "3. 在生成测试用例时，既要考虑网页中有没有对应模块的显示，也要考虑对应模块的功能是否正常，你需要根据你的知识来生成检验网页功能的方法。\n"
                "4. 请你不要实现需要其他设备辅助验证的测试用例。\n"
                f"用户要求：{demand}\n"
                "请返回测试用例列表，格式为List(str)，不需要其他额外字符，后续要将该结果用eval函数转换。"
            )

            logger.info(f"原始需求: {demand}")
            # 调用chat生成测试用例
            answer = await self._inference_chat(prompt)
            # 将字符串转换为列表
            test_cases = eval(answer)
            return test_cases

        except Exception as e:
            logger.error(f"生成测试用例时发生错误: {str(e)}")
            raise

    async def generate_case_name(self, case_desc: str) -> str:
        """为测试用例生成简短的case_name

        Args:
            case_desc: 测试用例描述

        Returns:
            str: 生成的case_name
        """
        if not case_desc:
            return ""

        prompt = f"""请将以下测试用例描述凝练为一个简短的英文case_name(不超过5个词):
        {case_desc}
        只需返回凝练后的case_name,不需要加引号"""

        case_name = await self._inference_chat(prompt)
        return case_name.strip()

    async def check_result(self, case_desc: str, model_output: str) -> str:
        """检查测试结果是否符合预期

        Args:
            case_desc: 测试用例描述
            model_output: 模型输出结果

        Returns:
            str: "是"、"否"或"不确定"
        """
        if not case_desc or not model_output:
            return "不确定"

        # 构造提示语
        prompt = f"""下面内容中模型结果为ground truth，请你根据事实判断描述的用例是否成功实现，如果有证据表明其已实现就只输出是，否则就输出否。如果模型结果表明其无法确定结果就输出不确定:
        用例描述: {case_desc}
        模型结果: {model_output}
        
        只需回答"是","否","不确定" """

        answer = await self._inference_chat(prompt)
        return answer.strip()

    async def generate_results_dict(
        self, action_history: List[str], task_list: str, memory: List[str], task_id_case_number: int
    ) -> Dict:
        """根据历史信息生成结果字典

        Args:
            action_history: 历史操作信息列表
            task_list: 任务列表信息
            memory: 记忆历史信息
            task_id_case_number: 任务数量

        Returns:
            Dict: 结果字典
        """
        try:
            # 构造提示语
            prompt = (
                """
                你是一个专业的测试工程师。请根据以下历史信息，生成指定格式的结果字典。
                你需要综合所有历史信息，推断最后的测试结果，请注意不要遗漏任何可能的测试用例结果，其中你觉得没有给出结果的用例，请你使用Uncertain作为该用例的结果。
                Result Format:
                    {
                        "0": {"result": "Pass", "evidence": "The thumbnail click functionality is working correctly."},
                        "1": {"result": "Uncertain", "evidence": "Cannot verify price calculation accuracy as no pricing information is displayed"},
                        "2": {"result": "Fail", "evidence": "After fully browsing and exploring the web page, I did not find the message board."},
                    }
                """
                f"动作历史信息：{action_history}\n"
                f"任务列表信息：{task_list}\n"
                f"记忆历史信息：{memory}\n"
                f"结果字典中测试用例的个数：{task_id_case_number}"
                "请返回结果字典，不需要其他额外字符，后续要将该结果用eval函数转换。"
            )

            logger.info(f"历史信息长度: {len(str(action_history))}")
            # 调用chat生成结果
            answer = await self._inference_chat(prompt)
            # 将字符串转换为字典
            results = eval(answer)
            return results

        except Exception as e:
            logger.error(f"生成结果字典时发生错误: {str(e)}")
            raise

    async def process_excel_file(self, excel_path: str, operation: str = "generate_cases") -> None:
        """处理Excel文件

        Args:
            excel_path: Excel文件路径
            operation: 操作类型，支持"generate_cases"、"make_case_name"、"check_results"
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"文件不存在: {excel_path}")

        # 读取Excel文件
        df = pd.read_excel(excel_path)

        if df.empty:
            logger.warning("Excel文件为空")
            return

        if operation == "generate_cases":
            # 处理每行需求生成测试用例
            for index, row in df.iterrows():
                ori_demand = str(row["requirement"])
                if not ori_demand:
                    continue

                test_cases = await self.generate_test_cases(ori_demand)
                df.at[index, "自动生成测试用例"] = str(test_cases)
                # 每处理完一行就保存一次
                df.to_excel(excel_path, index=False)

        elif operation == "make_case_name":
            # 为每行测试用例生成case_name
            for index, row in df.iterrows():
                task_desc = str(row["case_desc"])
                if not task_desc:
                    continue

                case_name = await self.generate_case_name(task_desc)
                df.at[index, "case_name"] = case_name
                df.to_excel(excel_path, index=False)

        elif operation == "check_results":
            # 检查每行测试结果
            for index, row in df.iterrows():
                task_desc = str(row["case_desc"])
                model_output = str(row["os_output"])
                if not task_desc or not model_output:
                    continue

                result = await self.check_result(task_desc, model_output)
                df.at[index, "自动功能检测(去除不可靠用例)"] = result
                df.to_excel(excel_path, index=False)

        else:
            raise ValueError(f"不支持的操作类型: {operation}")

        logger.info(f"{operation} 操作完成并已保存")


# 测试代码
if __name__ == "__main__":

    async def main():
        # 创建TestGeneratorAction实例
        test_generator = TestGeneratorAction()

        # 测试生成测试用例
        demand = "请基于以下要求开发一个数学练习游戏：\n1. 游戏中随机生成加减乘除四种运算题目。\n2. 玩家需要在输入框中输入答案，游戏会判断答案的正误。\n3. 游戏需要记录玩家的得分，正确回答增加分数。\n4. 支持难度选择，难度等级决定题目的复杂程度。\n5. 游戏结束时，显示玩家的总得分和正确率"

        try:
            test_cases = await test_generator.generate_test_cases(demand)
            print("\n生成的测试用例:")
            for i, case in enumerate(test_cases):
                print(f"{i+1}. {case}")

            # 测试生成case_name
            case_name = await test_generator.generate_case_name(test_cases[0])
            print(f"\n测试用例名称: {case_name}")

        except Exception as e:
            print(f"执行失败: {str(e)}")

    asyncio.run(main())
