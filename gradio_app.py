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
    custom_theme = gr.themes.Soft().set(
        body_background_fill="#eadbe7",
        block_background_fill="#ffffff",
        block_label_background_fill="#aeb9e2",
        block_title_text_color="#333333",
        button_primary_background_fill="#acc9e9",
        button_primary_background_fill_hover="#8fb3de",
        button_secondary_background_fill="#f2ddb3",
        button_secondary_background_fill_hover="#e6ca91",
        border_color_accent="#aeb9e2",
    )

    with gr.Blocks(title="AppEval Testing Tool", theme=custom_theme) as app:
        gr.Markdown(
            """
            <div class="header-container">
                <h1>✨ AppEval Testing Tool ✨</h1>
                <h3>🔍 Automated Application Testing & Evaluation Platform</h3>
                <p>This tool helps you run automated tests and evaluate applications 
                based on your requirements</p>
            </div>
            """
        )

        with gr.Row(elem_classes="config-container"):
            with gr.Column():
                with gr.Group(elem_classes="config-group"):
                    with gr.Row():
                        with gr.Column():
                            case_name = gr.TextArea(
                                label="📋 Case Name",
                                placeholder="Enter test case name",
                                value="Professional Portfolio",
                                info="Unique identifier for this test case",
                                elem_classes="input-field",
                                lines=5,
                            )
                            url = gr.TextArea(
                                label="🔗 Target URL",
                                placeholder="Enter target URL",
                                value="https://mgx.dev/app/pzo8wd",
                                info="The URL of the application to test",
                                elem_classes="input-field",
                                lines=5,
                            )
                        with gr.Column():
                            requirement = gr.TextArea(
                                label="📝 Requirements",
                                placeholder="Enter test requirements",
                                value="""Please help me create a professional personal portfolio website...""",
                                lines=5,
                                info="Detailed description of what needs to be tested",
                                elem_classes="input-field",
                            )
                            single_status = gr.TextArea(
                                label="🚦 Current Status", 
                                interactive=False, 
                                lines=5, 
                                info="Status will be displayed here during test execution",
                                elem_classes="status-box"
                            )

        with gr.Row(elem_classes="main-container"):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="monitor-group-large"):
                    gr.Markdown("""<div class="section-header"><i class="icon-test-cases"></i> Test Cases</div>""")
                    test_cases = gr.Textbox(
                        label="📝 Test Cases",
                        lines=10,
                        value=get_test_cases(),
                        every=2,
                        elem_classes="input-field",
                        )
                with gr.Group(elem_classes="monitor-group-large"):
                    gr.Markdown("""<div class="section-header"><i class="icon-tasks"></i> Task List</div>""")
                    gr.Textbox(
                        label="📋 Tasks",
                        interactive=False,
                        lines=10,
                        value=get_task_list(),
                        every=2,
                        elem_classes="task-box",
                    )
                with gr.Group(elem_classes="monitor-group-small"):
                    gr.Markdown("""<div class="section-header"><i class="icon-history"></i> Action History</div>""")
                    gr.Textbox(
                        label="📜 Actions",
                        interactive=False,
                        lines=4,
                        value=get_action_history(),
                        every=2,
                        elem_classes="history-box",
                    )

            with gr.Column(scale=2):
                with gr.Group(elem_classes="control-group"):
                    with gr.Row(elem_classes="button-container"):
                        single_run_btn = gr.Button(
                            "▶️ Run Test", variant="primary", size="large", elem_classes="action-button"
                        )
                        single_stop_btn = gr.Button(
                            "⏹️ Stop Test", variant="stop", size="large", elem_classes="action-button"
                        )
                    
                with gr.Group(elem_classes="monitor-group"):
                    gr.Markdown("""<div class="section-header"><i class="icon-screenshot"></i> Live Screenshot</div>""")
                    gr.Image(
                        label="📸 Current Screenshot",
                        value=get_screenshot_image(),
                        every=3,
                        elem_classes="screenshot-box",
                        show_download_button=True,
                        height=729,
                    )

        gr.Markdown(
            """
        <style>
        :root {
            --primary-color: #aeb9e2;
            --secondary-color: #eadbe7;
            --accent-color: #acc9e9;
            --highlight-color: #f2ddb3;
            --text-primary: #333333;
            --text-secondary: #666666;
            --shadow-color: rgba(0, 0, 0, 0.1);
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-image: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
            background-attachment: fixed;
            color: var(--text-primary);
        }
        
        .header-container {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
            border-radius: 12px;
            box-shadow: 0 4px 12px var(--shadow-color);
            margin-bottom: 2rem;
            color: white;
        }
        
        .header-container h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #ffffff, #f8f8f8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header-container h3 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            color: #f7f7f7;
        }
        
        .header-container p {
            font-size: 1rem;
            color: #f0f0f0;
        }
        
        .main-container, .monitoring-container, .screenshot-container {
            margin-bottom: 1.5rem;
        }
        
        .config-group, .monitor-group {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px var(--shadow-color);
            border: none;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
        }
        
        .config-group:hover, .monitor-group:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px var(--shadow-color);
        }
        
        .section-header {
            font-size: 1.3rem;
            font-weight: bold;
            color: var(--accent-color);
            margin-bottom: 1rem;
            border-bottom: 2px solid var(--highlight-color);
            padding-bottom: 0.5rem;
            display: flex;
            align-items: center;
        }
        
        .section-header i {
            margin-right: 0.5rem;
            font-size: 1.2rem;
        }
        
        .icon-gear:before { content: "⚙️"; }
        .icon-history:before { content: "🕒"; }
        .icon-tasks:before { content: "📋"; }
        .icon-test-cases:before { content: "🧪"; }
        .icon-screenshot:before { content: "📸"; }
        
        .history-box, .task-box, .status-box {
            background-color: #fcfcfc;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            box-shadow: inset 0 1px 3px var(--shadow-color);
            border: 1px solid #e0e0e0;
            height: 300px;
            overflow-y: auto;
        }
        
        .status-box {
            border-left: 4px solid var(--highlight-color);
        }
        
        .input-field {
            margin-bottom: 1rem;
        }
        
        .input-field textarea {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 0.8rem;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
            background-color: #fcfcfc;
            width: 100%;
            resize: none;
            height: 150px !important;
        }
        
        .input-field textarea:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(172, 201, 233, 0.3);
            outline: none;
        }
        
        .input-field label {
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .monitor-group-large {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px var(--shadow-color);
            border: none;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 250px;
            margin-bottom: 1rem;
        }
        
        .monitor-group-small {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px var(--shadow-color);
            border: none;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100px;
            margin-bottom: 1rem;
        }
        
        .control-group {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px var(--shadow-color);
            border: none;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 1rem;
        }
        
        .file-input {
            border: 2px dashed var(--primary-color);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            transition: background-color 0.2s;
        }
        
        .file-input:hover {
            background-color: rgba(174, 185, 226, 0.1);
        }
        
        .button-container {
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
            justify-content: center;
        }
        
        .action-button {
            border-radius: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 8px var(--shadow-color);
            padding: 0.8rem 2rem;
            min-width: 150px;
        }
        
        .action-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--shadow-color);
        }
        
        .screenshot-container {
            margin-top: 1.5rem;
        }
        
        .screenshot-box {
            background-color: #fcfcfc;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 8px var(--shadow-color);
            transition: transform 0.2s;
            overflow: hidden;
            height: 400px;
            object-fit: contain;
        }
        
        .screenshot-box:hover {
            transform: scale(1.01);
            box-shadow: 0 4px 12px var(--shadow-color);
        }
        </style>
        """
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
