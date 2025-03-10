import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

from appeval.roles.appeval_role import AppEvalRole


async def run_batch_test():
    """运行批量测试示例"""
    try:
        # 设置测试相关路径
        project_excel = "data/test_cases/project_cases.xlsx"
        case_excel = "data/test_cases/test_results.xlsx"
        json_file = "data/test_cases/test_cases.json"

        # 初始化自动化测试角色
        appeval = AppEvalRole(
            json_file=json_file,
            use_ocr=True,
            quad_split_ocr=True,
            use_memory=True,
            use_reflection=True,
            use_chrome_debugger=True,
            extend_xml_infos=True,
        )

        # 执行批量测试
        result = await appeval.run(project_excel_path=project_excel, case_excel_path=case_excel)
        result = json.loads(result.content)
        logger.info(f"批量测试执行结果: {result}")

    except Exception as e:
        logger.error(f"批量测试执行失败: {str(e)}")


async def run_single_test():
    """运行单个测试用例示例"""
    try:
        # 设置测试参数
        case_name = "首页"
        url = "https://www.baidu.com"
        requirement = "中间有个输入框"
        json_path = "data/test_cases/single_test.json"

        # 初始化自动化测试角色
        appeval = AppEvalRole(
            json_file=json_path,
            use_ocr=True,
            quad_split_ocr=True,
            use_memory=True,
            use_reflection=True,
            use_chrome_debugger=True,
            extend_xml_infos=True,
        )

        # 执行单个测试
        result = await appeval.run(case_name=case_name, url=url, user_requirement=requirement, json_path=json_path)
        result = json.loads(result.content)
        logger.info(f"单个测试执行结果: {result}")

    except Exception as e:
        logger.error(f"单个测试执行失败: {str(e)}")
        logger.exception("详细错误信息")


async def main():
    """主函数"""
    # 创建必要的目录
    Path("data/test_cases").mkdir(parents=True, exist_ok=True)

    # 运行单个测试示例
    logger.info("开始执行单个测试...")
    await run_single_test()

    # 运行批量测试示例
    # logger.info("开始执行批量测试...")
    # await run_batch_test()


if __name__ == "__main__":
    asyncio.run(main())
