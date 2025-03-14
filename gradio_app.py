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

from appeval.roles.test_runner import AppEvalRole

# Global variables to control execution
stop_execution = False
current_test_task = None

class ScreenshotUpdater:
    """Class to handle screenshot updates to the Gradio interface
    
    Args:
        screenshot_gallery: Gradio Gallery component that displays screenshots in the UI
        case_name: Name of the test case, used for organizing screenshots
    """
    def __init__(self, screenshot_gallery, case_name=None):
        self.screenshot_gallery = screenshot_gallery  # Store reference to Gradio Gallery component
        self.running = False
        self.latest_screenshots = []
        self.thread = None
        self.case_name = case_name or "default"
        self.case_dir = None
    
    def start(self):
        """Start the screenshot updater thread"""
        self.running = True
        
        # Create case-specific directory for screenshots
        self.case_dir = Path(f"work_dirs/{self.case_name}")
        self.case_dir.mkdir(parents=True, exist_ok=True)
        
        # Start updater thread
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        
        # Log start
        logger.info(f"Screenshot updater started for case: {self.case_name} in directory: {self.case_dir}")
    
    def stop(self):
        """Stop the screenshot updater thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _update_loop(self):
        """Update loop to check for new screenshots"""
        while self.running:
            try:
                # Check for new screenshots in the case directory
                screenshots = []
                
                # Check case-specific directory for origin_*.jpg files
                if self.case_dir.exists():
                    # Look for origin_*.jpg files (screenshots taken during evaluation)
                    case_screenshots = list(self.case_dir.glob("origin_*.jpg"))
                    
                    if case_screenshots:
                        # Extract iteration numbers from filenames
                        def get_iter_number(filepath):
                            try:
                                # Extract iteration number from origin_{iter}.jpg format
                                filename = filepath.name
                                iter_num = int(filename.replace("origin_", "").replace(".jpg", ""))
                                return iter_num
                            except:
                                return -1  # For files that don't match the pattern
                        
                        # Sort by iteration number (ascending order: 0, 1, 2, ...)
                        case_screenshots.sort(key=get_iter_number)
                        
                        # Always show the most recent screenshots (up to 3)
                        # When new screenshots appear, they will be appended to the end after sorting
                        # and we'll take the last 3, which removes the oldest ones
                        max_screenshots = 3
                        recent_screenshots = case_screenshots[-max_screenshots:] if len(case_screenshots) > max_screenshots else case_screenshots
                        screenshots.extend(recent_screenshots)
                        
                        logger.info(f"Found {len(recent_screenshots)} screenshots in sliding window from {self.case_dir}")
                
                if screenshots:
                    # Check if these are different from our current set
                    current_paths = {str(ss) for ss in self.latest_screenshots}
                    new_paths = {str(ss) for ss in screenshots}
                    
                    if current_paths != new_paths:
                        # Load the actual images
                        loaded_images = []
                        for ss in screenshots:
                            try:
                                img = PILImage.open(ss)
                                loaded_images.append(img)
                            except Exception as e:
                                logger.error(f"Failed to load image {ss}: {e}")
                        
                        if loaded_images:
                            self.latest_screenshots = screenshots
                            # Convert PIL images to numpy arrays
                            numpy_images = [np.array(img) for img in loaded_images]
                            # Update the Gradio gallery component with the numpy arrays
                            self.screenshot_gallery.update(value=numpy_images)
                            logger.info(f"Updated screenshots gallery with {len(numpy_images)} images")
                        else:
                            logger.warning("No images were loaded to update the gallery.")
            except Exception as e:
                logger.error(f"Error updating screenshots: {str(e)}")
            
            time.sleep(1)  # Check every second


# Non-async wrapper functions that will be called directly by Gradio
def run_single_test_wrapper(case_name, url, requirement, json_file, screenshot_gallery, result_output):
    """Wrapper function for running a single test case"""
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async function in the new event loop
        return loop.run_until_complete(
            run_single_test(case_name, url, requirement, json_file, screenshot_gallery, result_output)
        )
    finally:
        loop.close()


async def run_single_test(case_name, url, requirement, json_file, screenshot_gallery, result_output):
    """Run a single test case and update the UI with results
    
    Args:
        case_name: Name of the test case
        url: Target URL to test
        requirement: Test requirements
        json_file: Optional JSON configuration file
        screenshot_gallery: Gradio Gallery component for displaying live screenshots
        result_output: Gradio TextArea component for displaying test results
    """
    global stop_execution, current_test_task
    stop_execution = False
    
    # Initialize screenshot updater with the Gradio gallery component
    updater = ScreenshotUpdater(screenshot_gallery, case_name=case_name)
    updater.start()  # Start monitoring for new screenshots
    await asyncio.sleep(10)  # Wait for initial screenshots to load
    
    try:
        # Create case-specific directory for screenshots
        screenshot_dir = Path(f"work_dirs/{case_name}")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        # Save JSON file if uploaded
        json_path = f"data/{case_name}.json"
        # Initialize automated test role
        appeval = AppEvalRole(
            json_file=json_path,
            use_ocr=False,
            quad_split_ocr=False,
            use_memory=False,
            use_reflection=True,
            use_chrome_debugger=True,
            extend_xml_infos=True,
            log_dirs=f"work_dirs/{case_name}"  # Set log_dirs to case-specific directory
        )
        
        # Execute single test
        current_test_task = asyncio.create_task(
            appeval.run(case_name=case_name, url=url, user_requirement=requirement, json_path=json_path)
        )
        
        # Set up a task to check for cancellation
        async def cancel_checker():
            while not current_test_task.done():
                if stop_execution:
                    current_test_task.cancel()
                    return
                await asyncio.sleep(0.5)
        
        # Start cancel checker
        cancel_check_task = asyncio.create_task(cancel_checker())
        
        # Wait for test completion
        result = await current_test_task
        
        # Process result
        result_json = json.loads(result.content)
        result_output.update(value=json.dumps(result_json, indent=2))
        logger.info(f"Single test execution result: {result_json}")
        return "Test completed successfully! Check the results below."
    
    except asyncio.CancelledError:
        result_output.update(value="Test execution was cancelled by user")
        return "Test execution was cancelled by user"
    except Exception as e:
        logger.error(f"Single test execution failed: {str(e)}")
        logger.exception("Detailed error information")
        result_output.update(value=f"Error: {str(e)}")
        return f"Test execution failed: {str(e)}"
    finally:
        updater.stop()
        current_test_task = None


def stop_test():
    """Stop the current test execution"""
    global stop_execution
    stop_execution = True
    return "Stopping test execution... Please wait."


def create_folders():
    """Create necessary folders for the application"""
    Path("data/test_cases").mkdir(parents=True, exist_ok=True)
    Path("work_dirs/screenshots").mkdir(parents=True, exist_ok=True)


def create_ui():
    """Create the Gradio UI with components for test execution and result display"""
    with gr.Blocks(title="AppEval Testing Tool") as app:
        gr.Markdown("# AppEval Testing Tool")
        gr.Markdown("Upload files and run tests for your application")
        
        # Single Test UI
        with gr.Row():
            with gr.Column(scale=2):
                case_name = gr.Textbox(label="Case Name", placeholder="Enter test case name", value="Professional Portfolio")
                url = gr.Textbox(label="URL", placeholder="Enter target URL", value="https://mgx.dev/app/pzo8wd")
                requirement = gr.TextArea(
                    label="Requirement", 
                    placeholder="Enter test requirements",
                    value="""Please help me create a professional personal portfolio website...""",
                    lines=5
                )
                json_file = gr.File(label="JSON Config File (Optional)")
                
                with gr.Row():
                    single_run_btn = gr.Button("Run Test", variant="primary")
                    single_stop_btn = gr.Button("Stop Test", variant="stop")
                
                single_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                # Create gallery component for displaying live screenshots
                single_screenshot = gr.Gallery(
                    label="Live Screenshots", 
                    interactive=False,
                    columns=3,
                    object_fit="contain",
                    height="auto"
                )
                single_result = gr.TextArea(label="Test Results", interactive=False, lines=10)
        
        # Bind event handlers to UI components
        single_run_btn.click(
            fn=run_single_test_wrapper,
            inputs=[case_name, url, requirement, json_file, single_screenshot, single_result],
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