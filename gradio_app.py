#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11
@File    : gradio_app.py
@Author  : bianyutong
@Desc    : Gradio Web Interface for AppEval Testing Tool
"""
import asyncio
import json
import os
from pathlib import Path

import gradio as gr
from loguru import logger

from appeval.roles.eval_runner import AppEvalRole
from appeval.utils.excel_json_converter import make_json_single

# Global variables to control execution
stop_execution = False
current_test_task = None
current_appeval = None


def run_single_test_wrapper(case_name: str, url: str, requirement: str, test_cases_input: str = None) -> tuple:
    """Wrapper function for running a single test case in a new event loop.

    Args:
        case_name: Name of the test case
        url: Target URL to test
        requirement: Test requirements description
        test_cases_input: Optional pre-defined test cases

    Returns:
        tuple: (test_result, status_message, test_cases)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(run_single_test(case_name, url, requirement, test_cases_input))
        return result
    finally:
        loop.close()


def get_action_history() -> callable:
    """Get current action history from AppEval instance.

    Returns:
        callable: Function that returns action history string
    """

    def inner() -> str:
        global current_appeval
        if current_appeval and hasattr(current_appeval.osagent.rc, "action_history"):
            try:
                action_history = current_appeval.osagent.rc.action_history
                return "\n".join(action_history) if action_history else "No actions recorded yet"
            except Exception as e:
                return f"ERROR: {str(e)}"
        return "No actions recorded yet"

    return inner


def get_screenshot_image() -> callable:
    """Get current screenshot image from AppEval instance.

    Returns:
        callable: Function that returns screenshot image path
    """

    def inner() -> str:
        global current_appeval
        if current_appeval and hasattr(current_appeval.osagent, "output_image_path"):
            image_path = current_appeval.osagent.output_image_path
            if image_path and os.path.exists(image_path):
                try:
                    return image_path
                except Exception as e:
                    logger.error(f"Error loading screenshot image: {str(e)}")
        return None

    return inner


def get_task_list() -> callable:
    """Get current task list from AppEval instance.

    Returns:
        callable: Function that returns task list string
    """

    def inner() -> str:
        global current_appeval
        if current_appeval and hasattr(current_appeval.osagent, "rc"):
            task_list = current_appeval.osagent.rc.task_list
            return task_list if task_list else "No tasks recorded yet"
        return "No tasks recorded yet"

    return inner


def get_test_cases() -> callable:
    """Get current test cases from AppEval instance.

    Returns:
        callable: Function that returns test cases string
    """

    def inner() -> str:
        global current_appeval
        if current_appeval and hasattr(current_appeval.rc, "test_cases"):
            try:
                test_cases = current_appeval.rc.test_cases
                return "\n".join(test_cases) if test_cases else "No test cases generated yet"
            except Exception as e:
                return f"ERROR: {str(e)}"
        return "No test cases generated yet"

    return inner


async def run_single_test(case_name: str, url: str, requirement: str, test_cases_input: str = None) -> tuple:
    """Run a single test case and update the UI with results.

    Args:
        case_name: Name of the test case
        url: Target URL to test
        requirement: Test requirements description
        test_cases_input: Optional pre-defined test cases

    Returns:
        tuple: (formatted_result, status_message, test_cases)
    """
    global stop_execution, current_test_task, current_appeval
    stop_execution = False

    try:
        log_dir = Path(f"work_dirs/{case_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        json_path = f"data/{case_name}.json"

        current_appeval = AppEvalRole(
            json_file=json_path,
            use_ocr=False,
            quad_split_ocr=False,
            use_memory=False,
            use_reflection=True,
            use_chrome_debugger=True,
            extend_xml_infos=True,
            log_dirs=f"work_dirs/{case_name}",
        )

        should_generate = True
        if test_cases_input:
            cleaned_cases = [case.strip() for case in test_cases_input.split("\n") if case.strip()]
            if cleaned_cases and not any(x in test_cases_input for x in ["No test cases generated yet", "ERROR:"]):
                should_generate = False
                test_cases = cleaned_cases
                current_appeval.rc.test_cases = test_cases
                logger.info(f"User provided test cases: {test_cases}")
                make_json_single(case_name, url, test_cases, json_path)
                current_test_task = asyncio.create_task(
                    current_appeval.run(
                        case_name=case_name,
                        url=url,
                        user_requirement=requirement,
                        json_path=json_path,
                        use_json_only=True,
                    )
                )

        if should_generate:
            test_cases = await current_appeval.rc.test_generator.generate_test_cases(requirement)
            logger.info(f"Generated test cases: {test_cases}")
            current_appeval.rc.test_cases = test_cases
            make_json_single(case_name, url, test_cases, json_path)
            current_test_task = asyncio.create_task(
                current_appeval.run(
                    case_name=case_name,
                    url=url,
                    user_requirement=requirement,
                    json_path=json_path,
                    use_json_only=False,
                )
            )

        output_result = await current_test_task
        result_json = json.loads(output_result.content)
        formatted_result = json.dumps(result_json, indent=2)
        logger.info(f"Single test execution result: {result_json}")

        if should_generate:
            return (
                formatted_result,
                "Test completed successfully! Check the results below.",
                "\n".join(test_cases),
            )
        return (
            formatted_result,
            "Test completed successfully! Check the results below.",
            test_cases_input,
        )

    except asyncio.CancelledError:
        return (
            "Test execution was cancelled by user",
            "Test execution was cancelled by user",
            test_cases_input or "",
        )
    except Exception as e:
        logger.error(f"Single test execution failed: {str(e)}")
        logger.exception("Detailed error information")
        return (
            f"Test execution failed: {str(e)}",
            f"Test execution failed: {str(e)}",
            test_cases_input or "",
        )
    finally:
        current_test_task = None
        current_appeval = None


def stop_test() -> str:
    """Stop the current test execution.

    Returns:
        str: Status message
    """
    global stop_execution
    stop_execution = True
    return "Stopping test execution... Please wait."


def create_folders() -> None:
    """Create necessary folders for the application."""
    Path("data/test_cases").mkdir(parents=True, exist_ok=True)
    Path("work_dirs").mkdir(parents=True, exist_ok=True)


def create_ui() -> gr.Blocks:
    """Create the Gradio UI with components for test execution and result display.

    Returns:
        gr.Blocks: Gradio interface object
    """
    # 只设置背景颜色，不设置字体相关属性
    custom_theme = gr.themes.Base().set(
        # 蓝色系背景
        body_background_fill="#e6f2ff",  # 浅蓝色背景
        block_background_fill="#ffffff",  # 白色区块背景
        block_label_background_fill="#4682B4",  # 深蓝色标签背景 (Steel Blue)
        button_primary_background_fill="#ffb6c1",  # 浅粉色主按钮
        button_primary_background_fill_hover="#ff99aa",  # 粉色主按钮悬停
        button_secondary_background_fill="#fffacd",  # 浅黄色次按钮 (Lemon Chiffon)
        button_secondary_background_fill_hover="#fff68f",  # 黄色次按钮悬停
        border_color_accent="#4169E1",  # 皇家蓝边框
        button_large_text_size="1.2rem",  # 大按钮文本尺寸
        button_small_text_size="1.0rem",  # 小按钮文本尺寸
    )

    with gr.Blocks(title="AppEval Testing Tool", theme=custom_theme) as app:
        gr.Markdown(
            """
            # AppEval Testing Tool
            ### Automated Application Testing & Evaluation Platform
            
            This tool helps you run automated tests and evaluate applications 
            based on your requirements
            """
        )

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            case_name = gr.TextArea(
                                label="📋 Case Name",
                                placeholder="Enter test case name",
                                value="Professional Portfolio",
                                info="Unique identifier for this test case",
                                lines=5,
                            )
                            url = gr.TextArea(
                                label="🔗 Target URL",
                                placeholder="Enter target URL",
                                value="https://mgx.dev/app/pzo8wd",
                                info="The URL of the application to test",
                                lines=5,
                            )
                        with gr.Column():
                            requirement = gr.TextArea(
                                label="📝 Requirements",
                                placeholder="Enter test requirements",
                                value="""Please help me create a professional personal portfolio website...""",
                                lines=5,
                                info="Detailed description of what needs to be tested",
                            )
                            single_status = gr.TextArea(
                                label="🚦 Current Status",
                                interactive=False,
                                lines=5,
                                info="Status will be displayed here during test execution",
                            )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    test_cases = gr.Textbox(
                        label="📝 Test Cases",
                        lines=12,
                        max_lines=12,
                        value=get_test_cases(),
                        # value="",
                        every=2,
                    )
                with gr.Group():
                    gr.Textbox(
                        label="📋 Tasks",
                        interactive=False,
                        lines=12,
                        max_lines=12,
                        value=get_task_list(),
                        every=2,
                    )
                with gr.Group():
                    gr.Textbox(
                        label="📜 Actions",
                        interactive=False,
                        lines=4,
                        max_lines=4,
                        value=get_action_history(),
                        every=2,
                    )

            with gr.Column(scale=2):
                with gr.Group():
                    with gr.Row():
                        single_run_btn = gr.Button("Run Test", variant="primary", size="large")
                        single_stop_btn = gr.Button("Stop Test", variant="stop", size="large")

                with gr.Group():
                    gr.Markdown("### Live Screenshot")
                    gr.Image(
                        label="📸 Current Screenshot",
                        value=get_screenshot_image(),
                        every=3,
                        show_download_button=True,
                        height=724,
                    )

        single_run_btn.click(
            fn=run_single_test_wrapper,
            inputs=[case_name, url, requirement, test_cases],
            outputs=[single_status, single_status, test_cases],
        )

        single_stop_btn.click(fn=stop_test, outputs=[single_status])

    return app


if __name__ == "__main__":
    create_folders()
    ui = create_ui()
    ui.launch(share=False)
