import asyncio
import json
import os
import threading
import time
from pathlib import Path
import shutil

import gradio as gr
from loguru import logger
import numpy as np
from PIL import Image as PILImage

from appeval.roles.eval_runner import AppEvalRole
from datetime import datetime
# Global variables to control execution
stop_execution = False
current_test_task = None
current_appeval = None  # Add global variable to store AppEval instance

def current_time():
    def inner():
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        return f"欢迎使用,当前时间是: {current_time}"
    return inner

# Non-async wrapper functions that will be called directly by Gradio
def run_single_test_wrapper(case_name, url, requirement, json_file, result):
    """Wrapper function for running a single test case"""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async function in the new event loop
        return loop.run_until_complete(
            run_single_test(case_name, url, requirement, json_file, result)
        )
    finally:
        loop.close()

def get_action_history():
    """Get current action history from AppEval instance"""
    def inner():
        global current_appeval
        if current_appeval and hasattr(current_appeval.osagent, 'rc'):
            action_history = current_appeval.osagent.get_action_history()
            return "\n".join(action_history) if action_history else "No actions recorded yet"
        return "No actions recorded yet"
    return inner

def get_task_list():
    """Get current task list from AppEval instance"""
    def inner():
        global current_appeval
        if current_appeval and hasattr(current_appeval.osagent, 'rc'):
            task_list = current_appeval.osagent.rc.task_list
            return task_list if task_list else "No tasks recorded yet"
        return "No tasks recorded yet"
    return inner

async def run_single_test(case_name, url, requirement, json_file, result):
    """Run a single test case and update the UI with results"""
    global stop_execution, current_test_task, current_appeval
    stop_execution = False
    
    try:
        # Create case-specific directory for logs
        log_dir = Path(f"work_dirs/{case_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        # Save JSON file if uploaded
        json_path = f"data/{case_name}.json"
        # Initialize automated test role
        current_appeval = AppEvalRole(
            json_file=json_path,
            use_ocr=False,
            quad_split_ocr=False,
            use_memory=False,
            use_reflection=True,
            use_chrome_debugger=True,
            extend_xml_infos=True,
            log_dirs=f"work_dirs/{case_name}"
        )
        
        # Execute single test
        current_test_task = asyncio.create_task(
            current_appeval.run(case_name=case_name, url=url, user_requirement=requirement, json_path=json_path)
        )
        
        # Wait for test completion
        output_result = await current_test_task
        
        # Process result
        result_json = json.loads(output_result.content)
        formatted_result = json.dumps(result_json, indent=2)
        logger.info(f"Single test execution result: {result_json}")
        return formatted_result, "Test completed successfully! Check the results below."
    
    except asyncio.CancelledError:
        return "Test execution was cancelled by user", "Test execution was cancelled by user"
    except Exception as e:
        logger.error(f"Single test execution failed: {str(e)}")
        logger.exception("Detailed error information")
        return f"Test execution failed: {str(e)}", f"Test execution failed: {str(e)}"
    finally:
        current_test_task = None
        current_appeval = None

def stop_test():
    """Stop the current test execution"""
    global stop_execution
    stop_execution = True
    return "Stopping test execution... Please wait."

def create_folders():
    """Create necessary folders for the application"""
    Path("data/test_cases").mkdir(parents=True, exist_ok=True)
    Path("work_dirs").mkdir(parents=True, exist_ok=True)

def create_ui():
    """Create the Gradio UI with components for test execution and result display"""
    with gr.Blocks(title="AppEval Testing Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚀 AppEval Testing Tool
        ### Automated Application Testing and Evaluation Platform
        
        This tool helps you run automated tests and evaluate applications based on your requirements.
        """)
        
        # Current Time Display
        with gr.Row():
            time_display = gr.Textbox(
                label="Current Time",
                value=current_time(),
                every=1,
                interactive=False,
                elem_classes="time-display"
            )
        
        # Main Testing Interface
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### Test Configuration")
                    case_name = gr.Textbox(
                        label="Case Name",
                        placeholder="Enter test case name",
                        value="Professional Portfolio",
                        info="Unique identifier for this test case"
                    )
                    url = gr.Textbox(
                        label="URL",
                        placeholder="Enter target URL",
                        value="https://mgx.dev/app/pzo8wd",
                        info="The URL of the application to test"
                    )
                    requirement = gr.TextArea(
                        label="Requirement",
                        placeholder="Enter test requirements",
                        value="""Please help me create a professional personal portfolio website...""",
                        lines=5,
                        info="Detailed description of what needs to be tested"
                    )
                    json_file = gr.File(
                        label="JSON Config File (Optional)"
                    )
                
                with gr.Row():
                    single_run_btn = gr.Button(
                        "▶️ Run Test",
                        variant="primary",
                        size="large"
                    )
                    single_stop_btn = gr.Button(
                        "⏹️ Stop Test",
                        variant="stop",
                        size="large"
                    )
                
                single_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes="status-box"
                )
        # Monitoring Section
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Action History")
                    action_history = gr.Textbox(
                        label="Actions",
                        interactive=False,
                        lines=10,
                        value=get_action_history(),
                        every=2,
                        elem_classes="history-box"
                    )
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Task List")
                    task_list = gr.Textbox(
                        label="Tasks",
                        interactive=False,
                        lines=10,
                        value=get_task_list(),
                        every=2,
                        elem_classes="task-box"
                    )
        
        # Add custom CSS
        gr.Markdown("""
        <style>
        .time-display {
            text-align: center;
            font-size: 1.2em;
            color: #666;
            margin-bottom: 1em;
        }
        .status-box {
            background-color: #f5f5f5;
            border-radius: 4px;
            padding: 8px;
            margin-top: 1em;
        }
        .result-box, .history-box, .task-box {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 8px;
        }
        .group {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1em;
            margin-bottom: 1em;
        }
        </style>
        """)
        
        # Bind event handlers to UI components
        single_run_btn.click(
            fn=run_single_test_wrapper,
            inputs=[case_name, url, requirement, json_file],
            outputs=[single_status]
        )
        
        single_stop_btn.click(fn=stop_test, outputs=[single_status])
    
    return app

if __name__ == "__main__":
    # Create necessary folders
    create_folders()
    
    # Create and launch the UI
    ui = create_ui()
    ui.launch(share=False) 