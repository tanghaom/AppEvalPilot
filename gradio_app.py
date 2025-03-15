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
        if current_appeval and hasattr(current_appeval.osagent.rc, 'action_history'):
            try:
                action_history = current_appeval.osagent.rc.action_history
                return "\n".join(action_history) if action_history else "No actions recorded yet"
            except Exception as e:
                return "ERROR: " + str(e)
        return "No actions recorded yet"
    return inner

def get_screenshot_image():
    """Get current screenshot image from AppEval instance"""
    def inner():
        global current_appeval
        if current_appeval and hasattr(current_appeval.osagent, 'output_image_path'):
            image_path = current_appeval.osagent.output_image_path
            if image_path and os.path.exists(image_path):
                try:
                    return image_path
                except Exception as e:
                    logger.error(f"Error loading screenshot image: {str(e)}")
        return None
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
    # Custom theme with the specified color palette
    custom_theme = gr.themes.Soft().set(
        body_background_fill="#eadbe7",
        block_background_fill="#ffffff",
        block_label_background_fill="#aeb9e2",
        block_title_text_color="#333333",
        button_primary_background_fill="#acc9e9",
        button_primary_background_fill_hover="#8fb3de",
        button_secondary_background_fill="#f2ddb3",
        button_secondary_background_fill_hover="#e6ca91",
        border_color_accent="#aeb9e2"
    )
    
    with gr.Blocks(title="AppEval Testing Tool", theme=custom_theme) as app:
        gr.Markdown("""
        <div class="header-container">
            <h1>✨ AppEval Testing Tool ✨</h1>
            <h3>🔍 Automated Application Testing & Evaluation Platform</h3>
            <p>This tool helps you run automated tests and evaluate applications based on your requirements</p>
        </div>
        """)    
        # Main Testing Interface
        with gr.Row(elem_classes="main-container"):
            with gr.Column(scale=2):
                with gr.Group(elem_classes="config-group"):
                    gr.Markdown("""<div class="section-header"><i class="icon-gear"></i> Test Configuration</div>""")
                    case_name = gr.Textbox(
                        label="📋 Case Name",
                        placeholder="Enter test case name",
                        value="Professional Portfolio",
                        info="Unique identifier for this test case",
                        elem_classes="input-field"
                    )
                    url = gr.Textbox(
                        label="🔗 Target URL",
                        placeholder="Enter target URL",
                        value="https://mgx.dev/app/pzo8wd",
                        info="The URL of the application to test",
                        elem_classes="input-field"
                    )
                    requirement = gr.TextArea(
                        label="📝 Requirements",
                        placeholder="Enter test requirements",
                        value="""Please help me create a professional personal portfolio website...""",
                        lines=5,
                        info="Detailed description of what needs to be tested",
                        elem_classes="input-field"
                    )
                    json_file = gr.File(
                        label="📁 JSON Config File (Optional)",
                        elem_classes="file-input"
                    )
                
                with gr.Row(elem_classes="button-container"):
                    single_run_btn = gr.Button(
                        "▶️ Run Test",
                        variant="primary",
                        size="large",
                        elem_classes="action-button"
                    )
                    single_stop_btn = gr.Button(
                        "⏹️ Stop Test",
                        variant="stop",
                        size="large",
                        elem_classes="action-button"
                    )
                
                single_status = gr.Textbox(
                    label="🚦 Status",
                    interactive=False,
                    elem_classes="status-box"
                )
        
        # Monitoring Section
        with gr.Row(elem_classes="monitoring-container"):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="monitor-group"):
                    gr.Markdown("""<div class="section-header"><i class="icon-history"></i> Action History</div>""")
                    action_history = gr.Textbox(
                        label="📜 Actions",
                        interactive=False,
                        lines=10,
                        value=get_action_history(),
                        every=2,
                        elem_classes="history-box"
                    )
            with gr.Column(scale=1):
                with gr.Group(elem_classes="monitor-group"):
                    gr.Markdown("""<div class="section-header"><i class="icon-tasks"></i> Task List</div>""")
                    task_list = gr.Textbox(
                        label="📋 Tasks",
                        interactive=False,
                        lines=10,
                        value=get_task_list(),
                        every=2,
                        elem_classes="task-box"
                    )
            # Add Screenshot Column
            with gr.Column(scale=2):
                with gr.Group(elem_classes="monitor-group"):
                    gr.Markdown("""<div class="section-header"><i class="icon-screenshot"></i> Live Screenshot</div>""")
                    screenshot = gr.Image(
                        label="📸 Current Screenshot",
                        value=get_screenshot_image(),
                        every=3,  # Refresh every 3 seconds
                        elem_classes="screenshot-box",
                        show_download_button=True,
                        height=400
                    )
        
        # Add custom CSS
        gr.Markdown("""
        <style>
        /* Main color scheme variables */
        :root {
            --primary-color: #aeb9e2;
            --secondary-color: #eadbe7;
            --accent-color: #acc9e9;
            --highlight-color: #f2ddb3;
            --text-primary: #333333;
            --text-secondary: #666666;
            --shadow-color: rgba(0, 0, 0, 0.1);
        }
        
        /* Global styles */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-image: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
            background-attachment: fixed;
            color: var(--text-primary);
        }
        
        /* Header styles */
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
        
        /* Main containers */
        .main-container, .monitoring-container {
            margin-bottom: 1.5rem;
        }
        
        /* Group styling */
        .config-group, .monitor-group {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px var(--shadow-color);
            border: none;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .config-group:hover, .monitor-group:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px var(--shadow-color);
        }
        
        /* Section headers */
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
        
        /* Icons */
        .icon-gear:before { content: "⚙️"; }
        .icon-history:before { content: "🕒"; }
        .icon-tasks:before { content: "📋"; }
        .icon-screenshot:before { content: "📸"; }
        
        /* Form inputs */
        .input-field input, .input-field textarea {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 0.8rem;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
            background-color: #fcfcfc;
        }
        
        .input-field input:focus, .input-field textarea:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(172, 201, 233, 0.3);
            outline: none;
        }
        
        .input-field label {
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        /* File input */
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
        
        /* Button container */
        .button-container {
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        /* Buttons */
        .action-button {
            border-radius: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 8px var(--shadow-color);
        }
        
        .action-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--shadow-color);
        }
        
        /* Status box */
        .status-box {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            box-shadow: 0 2px 8px var(--shadow-color);
            border-left: 4px solid var(--highlight-color);
        }
        
        /* History and task boxes */
        .history-box, .task-box {
            background-color: #fcfcfc;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            box-shadow: inset 0 1px 3px var(--shadow-color);
            border: 1px solid #e0e0e0;
            max-height: 300px;
            overflow-y: auto;
        }
        
        /* Screenshot box */
        .screenshot-box {
            background-color: #fcfcfc;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 8px var(--shadow-color);
            transition: transform 0.2s;
            overflow: hidden;
        }
        
        .screenshot-box:hover {
            transform: scale(1.01);
            box-shadow: 0 4px 12px var(--shadow-color);
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