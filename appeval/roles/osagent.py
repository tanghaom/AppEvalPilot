#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/10
@Author  : tanghaoming
@File    : osagent.py
@Desc    : Operating System Operation Assistant
"""
import copy
import json
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metagpt.actions.action import Action
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleContext
from metagpt.schema import AIMessage, Message
from metagpt.utils.common import encode_image
from PIL import Image, ImageDraw, ImageFont
from pydantic import ConfigDict, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from appeval.prompts.osagent import ActionPromptContext, Android_prompt, PC_prompt
from appeval.tools.chrome_debugger import ChromeDebugger
from appeval.tools.device_controller import ControllerTool
from appeval.tools.icon_detect import IconDetectTool
from appeval.tools.ocr import OCRTool
from appeval.em import EMManager, Evidence, OnlineEvidenceCollector

# 忽略所有警告
warnings.filterwarnings("ignore")


class OSAgentContext(RoleContext):
    """Runtime context for OSAgent"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thought: str = ""  # Current thinking content
    # Historical thinking records list
    thought_history: List[str] = Field(default_factory=list)
    # Historical operation summary list
    summary_history: List[str] = Field(default_factory=list)
    # Historical executed action list
    action_history: List[str] = Field(default_factory=list)
    reflection_thought_history: List[str] = Field(
        default_factory=list)  # Historical reflection records list
    reflection_thought: str = ""  # Current reflection content
    summary: str = ""  # Current operation summary
    image_description: str = ""  # Current image description extracted during thinking
    action: str = ""  # Current executed action
    task_list: str = ""  # Task list
    completed_requirements: str = ""  # Completed requirements
    # Important content memory list
    memory: List[str] = Field(default_factory=list)
    error_flag: bool = False  # Error flag
    error_message: str = ""  # Error message when action execution fails
    iter: int = 0  # Current iteration count
    # Current perception information list
    perception_infos: List[Dict] = Field(default_factory=list)
    last_perception_infos: List[Dict] = Field(
        default_factory=list)  # Previous perception information list
    width: int = 0  # Screen width
    height: int = 0  # Screen height
    webbrowser_console_logs: List[Any] = Field(
        default_factory=list)  # Browser console log list

    def reset(self) -> None:
        """Reset all states to initial values"""
        self.thought = ""
        self.thought_history = []
        self.summary_history = []
        self.action_history = []
        self.reflection_thought_history = []
        self.reflection_thought = ""
        self.summary = ""
        self.action = ""
        self.task_list = ""
        self.completed_requirements = ""
        self.memory = []
        self.error_flag = False
        self.error_message = ""
        self.iter = 0
        self.perception_infos = []
        self.last_perception_infos = []
        self.width = 0
        self.height = 0
        self.webbrowser_console_logs = []


class OSAgent(Role):
    """Operating System Agent class for executing automated tasks"""

    name: str = "OSAgent"
    profile: str = "OS Agent"
    goal: str = "Execute automated tasks"
    constraints: str = "Ensure task execution accuracy and efficiency"
    desc: str = "Operating System Agent class for executing automated tasks"

    rc: OSAgentContext = Field(default_factory=OSAgentContext)

    def __init__(
        self,
        # Basic configuration parameters
        platform: str = "Android",
        max_iters: int = 50,
        # Feature switch parameters
        use_ocr: bool = False,
        quad_split_ocr: bool = False,
        use_icon_detect: bool = False,
        use_icon_caption: bool = False,
        use_som: bool = False,
        extend_xml_infos: bool = True,
        use_chrome_debugger: bool = False,
        think_history_images: int = 3,
        # Display and layout parameters
        location_info: str = "center",
        draw_text_box: bool = False,
        # Path related parameters
        log_dirs: str = "workspace",
        font_path: str = str(Path(__file__).parent / "simhei.ttf"),
        knowledge_base_path: str = str(Path(__file__).parent),
        # Other optional parameters
        system_prompt: str = "",
        add_info: str = "",
        # Evidence collection parameters (for online learning)
        enable_evidence_collection: bool = True,
        evidence_output_dir: str = None,
        evidence_fallback_to_mllm: bool = False,
        # EM model parameters (for prediction and correction)
        enable_em_correction: bool = True,
        em_params_path: str = None,
        em_tau_agentfail: float = 0.7,
        em_tau_envfail: float = 0.7,
        **kwargs,
    ) -> None:
        """Initialize OSAgent.

        Args:
            platform (str): Operating system type (Windows, Mac, or Android).
            max_iters (int): Maximum number of iterations.
            use_ocr (bool): Whether to use OCR.
            quad_split_ocr (bool): Whether to split image into four parts for OCR recognition.
            use_icon_detect (bool): Whether to use icon detection.
            use_icon_caption (bool): Whether to use icon caption.
            use_som (bool): Whether to draw visualization boxes on screenshots.
            extend_xml_infos (bool): Whether to add XML element information.
            use_chrome_debugger (bool): Whether to record browser console output.
            location_info (str): Location information type (center or bbox).
            draw_text_box (bool): Whether to draw text boxes in visualization.
            log_dirs (str): Log directory
            font_path (str): Font path.
            knowledge_base_path (str): Preset knowledge base file directory path
            system_prompt (str): System prompt
            add_info (str): Additional information to add to the prompt
            think_history_images (int): Max number of screenshots (latest-first) to include during think
            enable_evidence_collection (bool): Whether to enable evidence collection for online learning.
            evidence_output_dir (str): Output directory for evidence files. If None, uses log_dirs/evidence.
            evidence_fallback_to_mllm (bool): Whether to fallback to MLLM when element tree matching fails.
            enable_em_correction (bool): Whether to enable EM model for prediction and correction when Tell action is detected.
            em_params_path (str): Path to EM model parameters file. If None, uses default path.
            em_tau_agentfail (float): Threshold for AgentFail in EM correction.
            em_tau_envfail (float): Threshold for EnvFail in EM correction.
        """
        super().__init__(**kwargs)

        # Save configuration parameters
        self._init_config(locals())

        # Initialize environment
        self._init_environment()

        # Initialize tools
        self._init_tools()

    def _init_config(self, params: dict) -> None:
        """Initialize configuration parameters"""
        # Filter out self and kwargs
        config_params = {k: v for k, v in params.items() if k not in [
            "self", "kwargs"]}
        for key, value in config_params.items():
            setattr(self, key, value)

        # Set default additional prompt information
        if not self.add_info:
            self.add_info = self._get_default_add_info()

    def _get_default_add_info(self) -> str:
        """Get default additional prompt information"""
        if self.platform == "Windows":
            return (
                "If you need to interact with elements outside of a web popup, such as calendar or time selection "
                "popups, make sure to close the popup first. If the content in a text box is entered incorrectly, "
                "use the select all and delete actions to clear it, then re-enter the correct information. "
                "To open a folder in File Explorer, please use a double-click. "
            )
        elif self.platform == "Android":
            return (
                "If you need to open an app, prioritize using the Open app (app name) action. If this fails, "
                "return to the home screen and click the app icon on the desktop. If you want to exit an app, "
                "return to the home screen. If there is a popup ad in the app, you should close the ad first. "
                "If you need to switch to another app, you should first return to the desktop. When summarizing "
                "content, comparing items, or performing cross-app actions, remember to leverage the content in memory. "
            )
        return ""

    def _init_environment(self) -> None:
        """Initialize runtime environment"""
        # Initialize paths
        self._get_timestamped_paths()

        # Initialize logs
        self._setup_logs()

        # Initialize operating system environment
        self._init_os_env()

    def _init_tools(self) -> None:
        """Initialize tool components"""
        # Initialize icon detection/caption tool
        if self.use_icon_detect or self.use_icon_caption:
            self.icon_tool = IconDetectTool(self.llm)

        # Initialize OCR tool
        if self.use_ocr:
            self.ocr_tool = OCRTool()

        # Initialize browser debugger
        if self.use_chrome_debugger:
            self.chrome_debugger = ChromeDebugger()

        # Initialize evidence collector for online learning
        self.evidence_collector = None
        if self.enable_evidence_collection:
            evidence_dir = self.evidence_output_dir or str(
                Path(self.log_dirs) / "evidence")
            self.evidence_collector = OnlineEvidenceCollector(
                output_dir=evidence_dir,
                enable_coordinate_analysis=True,
                llm=self.llm,
                fallback_to_mllm=self.evidence_fallback_to_mllm,
                project_name=self.name,
            )
            logger.info(
                f"Evidence collector initialized, output dir: {evidence_dir}")

        # Initialize EM manager for prediction and correction
        self.em_manager = None
        self.em_correction_result = None  # Store the latest EM correction result
        if self.enable_em_correction:
            try:
                self.em_manager = EMManager(
                    params_path=self.em_params_path,
                    enable_online_learning=True,
                    tau_agentfail=self.em_tau_agentfail,
                    tau_envfail=self.em_tau_envfail,
                )
                logger.info(
                    "EM manager initialized for prediction and correction")
            except Exception as e:
                logger.warning(f"Failed to initialize EM manager: {e}")

    def _get_timestamped_paths(self) -> None:
        """Update file paths with timestamps"""
        current_time = time.strftime("%Y%m%d%H%M")

        # Base paths
        log_dir = Path(self.log_dirs) / current_time
        self.save_info = str(log_dir / "info.txt")
        self.save_img = str(log_dir)

        # Screenshot related paths
        self.screenshot_dir = log_dir / "screenshot"
        self.screenshot_file = str(self.screenshot_dir / "screenshot.jpg")
        self.screenshot_som_file = str(
            self.screenshot_dir / "screenshot_som.png")
        self.last_screenshot_file = str(
            self.screenshot_dir / "last_screenshot.jpg")
        self.last_screenshot_som_file = str(
            self.screenshot_dir / "last_screenshot_som.png")

    def _init_os_env(self) -> None:
        """Initialize operating system environment.

        Initialize corresponding controller and prompt tools based on different platforms (Android/Windows/Mac).
        """
        platform_configs = {
            "Android": {"controller_args": {"platform": "Android"}, "prompt_class": Android_prompt},
            "Windows": {
                "controller_args": {
                    "platform": "Windows",
                    "search_keys": ["win", "s"],
                    "ctrl_key": "ctrl",
                    "pc_type": "Windows",
                },
                "prompt_class": PC_prompt,
            },
            "Mac": {
                "controller_args": {
                    "platform": "Mac",
                    "search_keys": ["command", "space"],
                    "ctrl_key": "command",
                    "pc_type": "Mac",
                },
                "prompt_class": PC_prompt,
            },
        }

        if self.platform not in platform_configs:
            raise ValueError(f"Unsupported platform: {self.platform}")

        config = platform_configs[self.platform]
        logger.info(f"Initializing controller: {config['controller_args']}")
        self.controller = ControllerTool(**config["controller_args"])
        self.prompt_utils = config["prompt_class"]()

    def _reset_state(self) -> None:
        """Reset state, clear previous records when running new tasks"""
        # Reset state in rc
        self.rc.reset()

        # Reset temporary files and directories
        self._get_timestamped_paths()

        # Reset other states
        self.run_action_failed = False
        self.run_action_failed_exception = ""

        if self.use_chrome_debugger:
            self.chrome_debugger.start_monitoring()

        # Reset evidence collector
        if self.evidence_collector:
            # Update output directory with new timestamp
            evidence_dir = self.evidence_output_dir or str(
                Path(self.log_dirs) / time.strftime("%Y%m%d%H%M") / "evidence")
            self.evidence_collector.output_dir = evidence_dir
            Path(evidence_dir).mkdir(parents=True, exist_ok=True)
            self.evidence_collector.clear_evidences()
            logger.info(
                f"Evidence collector reset, new output dir: {evidence_dir}")

        # Reset EM manager
        if self.em_manager:
            self.em_manager.clear_evidences()
            self.em_manager.set_case_id(self.name)
            self.em_correction_result = None
            logger.info("EM manager reset")

        # Recreate screenshot directory
        if self.screenshot_dir.exists():
            shutil.rmtree(self.screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logs(self) -> None:
        """Set up logging"""
        log_dir = Path(self.save_info).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Remove previously existing log handlers
        logger.remove()

        # Define log format
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | " "{level:<8} | " "{module}:{function}:{line} - " "{message}"

        # Add file log handler
        logger.add(
            self.save_info,
            level="DEBUG",
            format=log_format,
            mode="w",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        # Add console log handler
        logger.add(sys.stdout, level="DEBUG", format=log_format,
                   colorize=True, enqueue=True)

        logger.info(f"Initialized logging, log file: {self.save_info}")

    def _draw_bounding_boxes(self, image_path: str, coordinates: List[List[int]], output_path: str, font_path: str) -> None:
        """Draw numbered coordinate boxes on the image.

        Args:
            image_path (str): Image path.
            coordinates (list): List of coordinate boxes, each box is a list of four elements [x1, y1, x2, y2].
            output_path (str): Output image path.
            font_path (str): Font path.
        """
        # Open image and get dimensions
        image = Image.open(image_path)
        height = image.size[1]

        # Calculate drawing parameters
        line_width = int(height * 0.0025)
        font_size = int(height * 0.012)
        text_offset_x = line_width
        text_offset_y = int(height * 0.013)

        # Generate random colors for each bounding box
        colors = [tuple(random.randint(0, 255) for _ in range(3))
                  for _ in range(len(coordinates))]

        # Draw bounding boxes and numbers
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        for i, (coord, color) in enumerate(zip(coordinates, colors)):
            # Draw bounding box using RGB color directly
            draw.rectangle(coord, outline=color, width=line_width)

            # Calculate text position and draw number
            text_x = coord[0] + text_offset_x
            text_y = max(0, coord[1] - text_offset_y)
            draw.text((text_x, text_y), str(i + 1), fill=color, font=font)

        # Save result
        image.convert("RGB").save(output_path)

    def _save_iteration_images(self, iter_num: int) -> None:
        """Save original and annotated images for current iteration.

        Args:
            iter_num: Current iteration number
        """
        # Build file paths
        origin_path = f"{self.save_img}/origin_{iter_num}.jpg"
        draw_path = f"{self.save_img}/draw_{iter_num}.jpg"

        # Copy image files
        shutil.copy2(self.screenshot_file, origin_path)
        shutil.copy2(self.output_image_path, draw_path)

    def _update_screenshot_files(self) -> None:
        """Update screenshot files"""
        # Update normal screenshot
        last_screenshot = Path(self.last_screenshot_file)
        if last_screenshot.exists():
            last_screenshot.unlink()
        Path(self.screenshot_file).rename(last_screenshot)

        # Update SOM screenshot
        if self.use_som:
            last_screenshot_som = Path(self.last_screenshot_som_file)
            if last_screenshot_som.exists():
                last_screenshot_som.unlink()
            Path(self.screenshot_som_file).rename(last_screenshot_som)

    def _check_last_three_start_with_wait(self, string_list: List[str]) -> bool:
        """Check if the last three strings in the list start with "Wait".

        Args:
            string_list (list): List of strings.

        Returns:
            bool: Returns True if the last three strings start with "Wait", False otherwise.
        """
        if len(string_list) < 3:
            return False
        return all(s.startswith("Wait") for s in string_list[-3:])

    def _get_app_info(self) -> Optional[str]:
        """Get application auxiliary information from preset app_info.json file."""
        info_path = Path(self.knowledge_base_path) / "app_info.json"
        if not info_path.exists():
            return None
        app_info = json.loads(info_path.read_text(encoding="utf-8"))
        package_name = self.controller.get_current_app_package()
        if not package_name:
            return None
        return app_info.get(package_name, None)

    async def _get_perception_infos(self, screenshot_file: str, screenshot_som_file: str) -> Tuple[List[Dict[str, Any]], int, int, str]:
        """Get perception information, including OCR and icon detection.
        Args:
            screenshot_file (str): Screenshot file path.
            screenshot_som_file (str): Screenshot file path with visualization boxes.
        Returns:
            tuple: Tuple containing perception information list, image width, image height and output image path.
        """
        # Get screen screenshot
        self.controller.get_screenshot(screenshot_file)
        # Get screen screenshot width and height
        width, height = Image.open(screenshot_file).size

        # OCR processing
        text, text_coordinates = [], []
        if self.use_ocr:
            text, text_coordinates = self.ocr_tool.ocr(
                screenshot_file, split=self.quad_split_ocr)

        # Icon detection
        icon_coordinates = []
        if self.use_icon_detect:
            icon_coordinates = self.icon_tool.detect(screenshot_file)

        # Process output image
        output_image_path = screenshot_som_file
        if self.use_ocr and self.use_icon_detect and self.draw_text_box:
            rec_list = text_coordinates + icon_coordinates
            self._draw_bounding_boxes(screenshot_file, copy.deepcopy(
                rec_list), screenshot_som_file, self.font_path)
        elif self.use_icon_detect:
            self._draw_bounding_boxes(screenshot_file, copy.deepcopy(
                icon_coordinates), screenshot_som_file, self.font_path)
        else:
            output_image_path = screenshot_file

        # Build perception information
        mark_number = 0
        perception_infos = []

        # Add OCR text information
        if self.use_ocr:
            for i in range(len(text_coordinates)):
                mark_number += 1
                if self.use_som and self.draw_text_box:
                    perception_info = {
                        "text": f"mark number: {mark_number} text: {text[i]}",
                        "coordinates": text_coordinates[i],
                    }
                else:
                    perception_info = {
                        "text": f"text: {text[i]}", "coordinates": text_coordinates[i]}
                perception_infos.append(perception_info)

        # Add icon information
        if self.use_icon_detect:
            for i in range(len(icon_coordinates)):
                mark_number += 1
                if self.use_som:
                    perception_info = {
                        "text": f"mark number: {mark_number} icon", "coordinates": icon_coordinates[i]}
                else:
                    perception_info = {"text": "icon",
                                       "coordinates": icon_coordinates[i]}
                perception_infos.append(perception_info)

        # Icon description
        if self.use_icon_detect and self.use_icon_caption:
            icon_indices = [i for i in range(
                len(perception_infos)) if "icon" in perception_infos[i]["text"]]
            if icon_indices:
                icon_boxes = [perception_infos[i]["coordinates"]
                              for i in icon_indices]
                descriptions = await self.icon_tool.caption(screenshot_file, icon_boxes, platform=self.platform)

                # Add description to perception information
                for idx, desc_idx in enumerate(icon_indices):
                    if descriptions.get(idx + 1):
                        perception_infos[desc_idx]["text"] += ": " + \
                            descriptions[idx + 1].replace("\n", " ")

        # According to parameter modify coordinate information
        if self.location_info == "center":
            for i in range(len(perception_infos)):
                x1, y1, x2, y2 = perception_infos[i]["coordinates"]
                perception_infos[i]["coordinates"] = [
                    int((x1 + x2) / 2), int((y1 + y2) / 2)]
        elif self.location_info == "icon_center":
            for i in range(len(perception_infos)):
                if "icon" in perception_infos[i]["text"]:
                    x1, y1, x2, y2 = perception_infos[i]["coordinates"]
                    perception_infos[i]["coordinates"] = [
                        int((x1 + x2) / 2), int((y1 + y2) / 2)]

        # If extend_xml_infos is enabled, then get XML information
        if self.extend_xml_infos and self.platform in ["Android", "Windows"]:
            xml_results = self.controller.get_screen_xml(self.location_info)
            logger.debug(xml_results)
            perception_infos.extend(xml_results)

        return perception_infos, width, height, output_image_path

    def get_webbrowser_console_logs(self, steps: int = 100, expand: bool = True) -> List[Any]:
        """
        Get recent web browser console logs.
        Note: Only used for mgx automated web testing.
        Args:
            steps (int, optional): Number of logs to get, default is 1.
            expand (bool, optional): Whether to return expanded log list, default is True.
                If True, returns the most recent `steps` log list.
                If False, returns the most recent `steps` log dictionary list, containing corresponding operations and console output.
        Returns:
            list: Recent console log list or dictionary list.
        """
        if not self.rc.webbrowser_console_logs:
            return []  # If there is no log, directly return empty list
        if expand:
            # Filter empty list
            return [log for log in self.rc.webbrowser_console_logs[-steps:] if log]
        else:
            # Use zip to pair operation history and log
            outputs = [
                {"action": action, "console_output": log}
                for action, log in zip(self.rc.summary_history, self.rc.webbrowser_console_logs)
                if log  # Filter empty list
            ]
            return outputs[-steps:]

    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get action history, including thoughts, summaries, actions, memories and reflections.
        Returns:
            list: A list of dictionaries, each dictionary represents a historical record of an action step.
                  Each dictionary contains "thought", "summary", "action", "memory" and "reflection".
        """
        outputs = []
        # Use zip to pair corresponding elements of historical lists
        for i, (thought, summary, action) in enumerate(zip(self.rc.thought_history, self.rc.summary_history, self.rc.action_history)):
            output = {
                "thought": thought,
                "summary": summary,
                "action": action,
                "memory": self.rc.memory[i] if i < len(self.rc.memory) else "",
                "reflection": self.rc.reflection_thought_history[i] if i < len(self.rc.reflection_thought_history) else "",
            }
            outputs.append(output)
        return outputs

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Generate operation decision failed, {retry_state.attempt_number}th retry: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _think(self) -> bool:
        """Generate operation decisions"""
        # Add preset knowledge
        add_info = self.add_info
        # Add application information to prompt
        if self.platform == "Android":
            info = self._get_app_info()
            if info:
                add_info += " ".join(info) if isinstance(info, list) else info
            else:
                info = "No add_info"
            logger.info(f"\n\n\n\n\n\n#### add_info:{info}\n\n")
        else:
            logger.info(
                "Knowledge base currently only implemented for Android")

        # Generate action
        ctx = ActionPromptContext(
            instruction=self.instruction,
            clickable_infos=self.rc.perception_infos,
            width=self.width,
            height=self.height,
            thought_history=self.rc.thought_history,
            summary_history=self.rc.summary_history,
            action_history=self.rc.action_history,
            reflection_thought_history=self.rc.reflection_thought_history,
            last_summary=self.rc.summary,
            last_action=self.rc.action,
            reflection_thought=self.rc.reflection_thought,
            add_info=add_info,
            error_flag=self.rc.error_flag,
            error_message=self.rc.error_message,
            completed_content=self.rc.completed_requirements,
            memory=self.rc.memory,
            task_list=self.rc.task_list,
            use_som=self.use_som,
            location_info=self.location_info,
        )

        prompt_action = self.prompt_utils.get_action_prompt(ctx)
        logger.info(
            f"\n\n######################## prompt_action:\n{prompt_action}\n\n######################## prompt_action end\n\n\n\n")

        # Call LLM to generate decision with history images
        images = []
        # include previous frames up to think_history_images - 1 using saved origin/draw files
        try:
            if isinstance(self.think_history_images, int) and self.think_history_images > 1:
                # Skip the immediate previous frame (iter-1). Select up to (think_history_images - 1) most recent frames
                # from iter-2 backward, then append them in chronological order (old -> new).
                max_hist_frames = self.think_history_images - 1
                end = self.rc.iter - 2
                if end >= 0:
                    start = max(0, end - (max_hist_frames - 1))
                    for frame_num in range(start, end + 1):  # ascending: old -> new
                        origin_path = Path(self.save_img) / \
                            f"origin_{frame_num}.jpg"
                        draw_path = Path(self.save_img) / \
                            f"draw_{frame_num}.jpg"
                        if origin_path.exists():
                            images.append(encode_image(str(origin_path)))
                            # If SOM is enabled and annotated image exists, also include it for the same frame
                            if self.use_som and draw_path.exists():
                                images.append(encode_image(str(draw_path)))
        except Exception:
            pass
        # include latest image (with/without SOM)
        images.append(encode_image(self.screenshot_file))
        if self.use_som:
            images.append(encode_image(self.screenshot_som_file))

        # Use custom system prompt or default prompt
        system_msg = (
            self.system_prompt
            if self.system_prompt
            else f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant. You need to help me operate the device to complete the user's instruction."
        )

        output_action = await self.llm.aask(
            prompt_action,
            system_msgs=[system_msg],
            images=images,
            stream=False,
        )

        # Parse output
        # Safely parse LLM output sections. If any required marker is missing, return empty to avoid mis-parsing.
        def _extract_between(text, start, end=None, normalize=False, escape_newlines=False):
            if start not in text:
                return ""
            start_idx = text.find(start) + len(start)
            if end is not None:
                end_idx = text.find(end, start_idx)
                if end_idx == -1:
                    return ""
                content = text[start_idx:end_idx]
            else:
                content = text[start_idx:]
            content = content.strip()
            if escape_newlines:
                content = content.replace("\n", "\\n")
            if normalize:
                content = content.replace(":", "")
                # collapse multiple spaces
                content = re.sub(r"\s{2,}", " ", content)
            return content.strip()

        self.rc.image_description = _extract_between(
            output_action, "### Image Description ###", "### Reflection Thought ###", escape_newlines=True)
        self.rc.reflection_thought = _extract_between(
            output_action, "### Reflection Thought ###", "### Thought ###", escape_newlines=True)
        self.rc.thought = _extract_between(
            output_action, "### Thought ###", "### Action ###", normalize=True)
        self.rc.action = _extract_between(
            output_action, "### Action ###", "### Operation ###")
        self.rc.summary = _extract_between(
            output_action, "### Operation ###", "### Task List ###", escape_newlines=True)
        self.rc.task_list = _extract_between(
            output_action, "### Task List ###")

        logger.info(
            f"\n\n######################## output_action:\n{output_action}\n\n######################## output_action end\n\n\n\n")

        if self.rc.action.startswith("Stop"):
            return False
        else:
            return True

    async def _get_app_package_name(self, app_name: str) -> str:
        """Get application package name

        Args:
            app_name (str): Application name

        Returns:
            str: Application package name
        """
        package_list = self.controller.get_all_packages()

        # Read application mapping information
        map_path = Path(self.knowledge_base_path) / "app_mapping.json"
        app_mapping = ""
        if map_path.exists():
            app_mapping = map_path.read_text(encoding="utf-8").strip()
        else:
            logger.warning(
                f"{map_path} file does not exist, using default empty mapping")

        # Get package name
        prompt_package_name = self.prompt_utils.get_package_name_prompt(
            app_name=app_name, app_mapping=app_mapping, package_list=package_list)

        package_name = await self.llm.aask(
            prompt_package_name,
            system_msgs=[
                f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant."],
            stream=False,
        )

        return package_name.strip()

    async def _handle_open_app(self) -> None:
        """Handle open application action"""
        if self.platform == "Android":
            app_name = re.search(r"\((.*?)\)", self.rc.action).group(1)
            logger.debug(f"Opening Android app: {app_name}")

            package_name = await self._get_app_package_name(app_name)

            if not self.controller.open_app(package_name):
                self.rc.error_flag = True
                logger.error("Failed to start app via adb")
            else:
                time.sleep(10)

        elif self.platform == "Windows":
            app_name = self.rc.action.split("(")[-1].split(")")[0]
            logger.debug(f"Opening Windows app: {app_name}")
            self.controller.open_app(app_name)
            time.sleep(10)
        else:
            logger.error(
                f"Platform {self.platform} not supported for opening apps")

    async def _act(self) -> Message:
        """Execute action step"""
        if self.use_chrome_debugger:
            # Store browser logs from before action execution in previous action log. Note: Need a log for step 0 here since mgx web testing is not started by osagent
            self.rc.webbrowser_console_logs.append(
                self.chrome_debugger.get_new_messages())

        self.run_action_failed = False
        self.run_action_failed_exception = ""

        # Execute action
        if "Stop" in self.rc.action:
            # If it's a stop operation, end the loop
            return AIMessage(content=self.rc.action, cause_by=Action)
        elif "Open App" in self.rc.action:
            await self._handle_open_app()
        else:
            # Execute other actions
            try:
                if self.platform in ["Android", "Windows"]:
                    self.controller.run_action(self.rc.action)
                else:
                    logger.error("Currently only supports Android and Windows")
            except Exception as e:
                # For direct exit when using tell in automg
                if isinstance(e, SystemExit) and e.code == 0:
                    return AIMessage(content=self.rc.action, cause_by=Action)
                logger.error(f"run action failed: {e}")
                self.run_action_failed = True
                self.run_action_failed_exception = e

        time.sleep(0.5)
        # Save previous perception information and screenshot
        self.rc.last_perception_infos = copy.deepcopy(self.rc.perception_infos)

        # Update screenshot files
        self._update_screenshot_files()

        # Get new perception information
        self.rc.perception_infos, self.width, self.height, self.output_image_path = await self._get_perception_infos(
            self.screenshot_file, self.screenshot_som_file
        )

        # Save images
        self._save_iteration_images(self.rc.iter)

        # Update history records
        self.rc.thought_history.append(self.rc.thought)
        self.rc.summary_history.append(self.rc.summary)
        self.rc.action_history.append(self.rc.action)

        # Save memory: use image_description from think (merged request mode)
        self.rc.memory.append(getattr(self.rc, "image_description", "") or "")

        # Handle reflection: always persist the reflection from think (reflects on previous step)
        self.rc.reflection_thought_history.append(self.rc.reflection_thought)

        # Handle execution errors separately
        if self.run_action_failed:
            # Store error message for next iteration's prompt
            self.rc.error_message = f"ERROR(run action code filed): {self.run_action_failed_exception}\\n"
            self.rc.error_flag = True
        else:
            # Clear error message on successful execution
            self.rc.error_message = ""

        # Collect evidence for online learning
        if self.evidence_collector:
            await self._collect_step_evidence()

        # Clean up screenshots
        Path(self.last_screenshot_som_file if self.use_som else self.last_screenshot_file).unlink()

        return AIMessage(content=self.rc.action, cause_by=Action)

    async def _collect_step_evidence(self) -> None:
        """Collect evidence for the current step for online learning"""
        if not self.evidence_collector:
            return

        # Extract click coordinates from action
        click_coords = None
        if self.rc.action:
            click_match = re.search(
                r"pyautogui\.click\((\d+),\s*(\d+)\)", self.rc.action)
            if click_match:
                click_coords = (int(click_match.group(1)),
                                int(click_match.group(2)))

        # Get screenshot path for current iteration
        screenshot_path = f"{self.save_img}/origin_{self.rc.iter}.jpg"
        if not Path(screenshot_path).exists():
            screenshot_path = self.screenshot_file

        # Collect evidence
        evidence = self.evidence_collector.collect_evidence(
            iter_num=self.rc.iter,
            action_content=self.rc.action,
            operation_desc=self.rc.summary,
            click_coords=click_coords,
            thought=self.rc.thought,
            reflection_thought=self.rc.reflection_thought,
            perception_infos=self.rc.perception_infos,
            screenshot_path=screenshot_path,
            task_list=self.rc.task_list,
            instruction=getattr(self, "instruction", None),
            error_flag=self.rc.error_flag,
            error_message=self.rc.error_message if self.rc.error_flag else None,
            width=self.width,
            height=self.height,
        )

        logger.debug(
            f"Evidence collected for iter {self.rc.iter}: coordinate_match={evidence.coordinate_match}")

        # 将证据添加到 EM 管理器
        if self.em_manager:
            self.em_manager.add_evidence(
                gui_evidence=evidence.coordinate_match,
                agent_noresp=evidence.agent_noresp,
                test_case_id=self.evidence_collector.project_name if self.evidence_collector else self.name,
                iter_num=self.rc.iter,
            )

        # 如果检测到 Tell 动作，分析 agent_noresp 并进行 EM 预测和纠正
        if evidence.tell_evidence:
            agent_noresp = await self.evidence_collector.analyze_tell_action(evidence)
            logger.info(
                f"Tell action analyzed for iter {self.rc.iter}: agent_noresp={agent_noresp}")

            # 更新 EM 管理器中最新证据的 agent_noresp
            if self.em_manager and self.em_manager._evidence_rows:
                self.em_manager._evidence_rows[-1]["E4_noresp"] = agent_noresp if agent_noresp is not None else 0
                self.em_manager._evidence_rows[-1]["M_noresp"] = 0 if agent_noresp is not None else 1

            # 进行 EM 预测和纠正
            await self._perform_em_correction(evidence)

    async def _perform_em_correction(self, evidence: Evidence) -> None:
        """
        Perform EM prediction and correction when Tell action is detected.

        Args:
            evidence: The current evidence object containing Tell action info
        """
        if not self.em_manager:
            logger.debug("EM manager not available, skipping correction")
            return

        try:
            # Extract agent's original judgment from Tell action
            # Tell action typically contains the agent's judgment result
            agent_original = self._extract_agent_judgment(
                evidence.tell_evidence)

            if agent_original is None:
                logger.warning(
                    "Could not extract agent judgment from Tell evidence")
                return

            # Get EM prediction
            prediction = self.em_manager.predict()
            logger.info(
                f"EM Prediction: P_EnvFail={prediction['P_EnvFail']:.3f}, "
                f"P_AgentFail={prediction['P_AgentFail']:.3f} "
                f"(Retry={prediction['P_AgentRetryFail']:.3f}, "
                f"Reasoning={prediction['P_AgentReasoningFail']:.3f})"
            )

            # Perform correction
            correction_result = self.em_manager.correct_judgment(
                agent_original=agent_original)
            self.em_correction_result = correction_result

            # Log the correction result
            logger.info("EM Correction Result:")
            logger.info(
                f"  - Agent Original: {agent_original} ({'PASS' if agent_original == 1 else 'FAIL'})")
            logger.info(
                f"  - Corrected Label: {correction_result['corrected_label']} " f"({'PASS' if correction_result['corrected_label'] == 1 else 'FAIL'})"
            )
            logger.info(f"  - Action: {correction_result['action']}")
            logger.info(
                f"  - P(EnvFail): {correction_result['P_EnvFail']:.3f}")
            logger.info(
                f"  - P(AgentFail): {correction_result['P_AgentFail']:.3f}")

            # If correction differs from original, log prominently
            if agent_original != correction_result["corrected_label"]:
                logger.warning(
                    f"⚠️ EM CORRECTION APPLIED: {agent_original} → {correction_result['corrected_label']}")
                logger.warning(f"   Reason: {correction_result['action']}")

        except Exception as e:
            logger.error(f"Error performing EM correction: {e}")

    def _extract_agent_judgment(self, tell_evidence: str) -> Optional[int]:
        """
        Extract agent's judgment (PASS=1/FAIL=0) from Tell evidence.

        Args:
            tell_evidence: The Tell action evidence string

        Returns:
            1 for PASS, 0 for FAIL, None if cannot extract
        """
        if not tell_evidence:
            return None

        tell_lower = tell_evidence.lower()

        # Common patterns for PASS
        pass_patterns = [
            "pass",
            "passed",
            "success",
            "successful",
            "completed",
            "done",
            "finished",
            "yes",
            "correct",
            "verified",
            "requirement met",
            "task completed",
            "功能已实现",
            "测试通过",
            "已完成",
        ]

        # Common patterns for FAIL
        fail_patterns = [
            "fail",
            "failed",
            "error",
            "not implemented",
            "not found",
            "missing",
            "incorrect",
            "wrong",
            "no",
            "unable",
            "cannot",
            "couldn't",
            "requirement not met",
            "功能未实现",
            "测试失败",
            "未完成",
            "不正确",
        ]

        # Check for explicit pass/fail indicators
        for pattern in pass_patterns:
            if pattern in tell_lower:
                return 1

        for pattern in fail_patterns:
            if pattern in tell_lower:
                return 0

        # Default: if we can't determine, return None
        return None

    def get_em_correction_result(self) -> Optional[Dict]:
        """
        Get the latest EM correction result.

        Returns:
            Dict containing correction result, or None if not available
        """
        return self.em_correction_result

    def get_em_prediction(self) -> Optional[Dict]:
        """
        Get the current EM prediction.

        Returns:
            Dict containing prediction probabilities, or None if EM manager not available
        """
        if self.em_manager:
            return self.em_manager.predict()
        return None

    def get_evidences(self) -> List[Evidence]:
        """Get all collected evidences for online learning.

        Returns:
            List[Evidence]: List of evidence objects from all steps.
        """
        if self.evidence_collector:
            return self.evidence_collector.get_all_evidences()
        return []

    def get_evidences_for_em(self) -> List[Dict]:
        """Get evidences in dictionary format suitable for EM model.

        Returns:
            List[Dict]: List of evidence dictionaries ready for EM model training.
        """
        if self.evidence_collector:
            return self.evidence_collector.get_evidences_for_em()
        return []

    def get_latest_evidence(self) -> Optional[Evidence]:
        """Get the most recent evidence.

        Returns:
            Optional[Evidence]: The latest evidence object, or None if no evidence collected.
        """
        if self.evidence_collector:
            return self.evidence_collector.get_latest_evidence()
        return None

    def save_evidences(self, output_file: str = None) -> None:
        """Save all collected evidences to a JSONL file.

        Args:
            output_file: Output file path. If None, saves to default location.
        """
        if self.evidence_collector:
            self.evidence_collector.save_evidences(output_file)

    def get_evidence_summary(self) -> Dict:
        """Get summary statistics of collected evidences.

        Returns:
            Dict: Summary statistics including total steps, coordinate accuracy, etc.
        """
        if self.evidence_collector:
            return self.evidence_collector.get_summary_stats()
        return {}

    async def _generate_initial_task_list(self, instruction: str, screenshot_file: str, screenshot_som_file: str = None) -> str:
        """Generate initial task list for the first iteration.

        Args:
            instruction: User instruction
            screenshot_file: Path to the screenshot file
            screenshot_som_file: Path to the SOM screenshot file (if use_som is enabled)

        Returns:
            str: Generated initial task list
        """
        # Create the task list prompt
        initial_task_prompt = f"""
        Based on the following instruction, please generate an initial task list:
        {instruction}
        
        Please output the task list in the following format:
        * **[Completed Tasks]:** 
          * None
        * **[Current Task]:** <describe the first high-level task to execute>
        * **[Next Operation]:** 
          * <describe the first step in detail>
        * **[Remaining Tasks]:** (List the remaining high-level tasks that need to be completed to achieve the user's objective, excluding the current and next operation.)
          * <describe remaining high-level task 1>
          * <describe remaining high-level task 2>
          * ...
        """

        # Prepare images for LLM
        images = [encode_image(screenshot_file)]
        if self.use_som and screenshot_som_file:
            images.append(encode_image(screenshot_som_file))

        # Prepare system message
        system_msg = (
            self.system_prompt
            if self.system_prompt
            else f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant. You need to help me operate the device to complete the user's instruction."
        )

        # Generate task list using LLM
        initial_task_list = await self.llm.aask(
            initial_task_prompt,
            system_msgs=[system_msg],
            images=images,
            stream=False,
        )

        task_list = initial_task_list.strip()
        logger.info(
            f"\n\n######################## Initial Task List:\n{task_list}\n\n######################## End of Initial Task List\n\n\n\n")

        return task_list

    async def _react(self) -> Message:
        self.rc.iter = 0
        # will be overwritten after Role _act
        rsp = AIMessage(content="No actions taken yet", cause_by=Action)
        while self.rc.iter < self.max_iters and not self._check_last_three_start_with_wait(self.rc.action_history):
            self.rc.iter += 1

            logger.info(f"\n\n\n\n\n\n#### iter:{self.rc.iter}\n\n")

            # Get initial perception information
            if self.rc.iter == 1:
                (
                    self.rc.perception_infos,
                    self.width,
                    self.height,
                    self.output_image_path,
                ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

                # Save images (use 0 for the very first snapshot to avoid being overwritten after think/act)
                self._save_iteration_images(0)

                # Generate initial task list
                self.rc.task_list = await self._generate_initial_task_list(
                    self.instruction, self.screenshot_file, self.screenshot_som_file if self.use_som else None
                )

            # think
            has_todo = await self._think()
            if not has_todo:
                rsp = AIMessage(
                    content="OS Agent has finished all tasks", cause_by=Action)
                break
            # act
            logger.debug(
                f"{self._setting}: {self.rc.state=}, will do {self.rc.todo}")
            rsp = await self._act()

        if self.use_chrome_debugger:
            self.chrome_debugger.stop_monitoring()

        return rsp

    async def run(self, instruction: str) -> Message:
        """Run main loop.

        Args:
            instruction (str): User instruction.
        """
        self._reset_state()  # Reset state for each run
        self._setup_logs()  # Reset logs for each run
        self.instruction = instruction

        rsp = await self.react()

        # Save evidences at the end of the run
        if self.evidence_collector:
            self.save_evidences()
            summary = self.get_evidence_summary()
            logger.info(f"Evidence collection complete: {summary}")

        # Log final EM correction result if available
        if self.em_correction_result:
            logger.info("=" * 50)
            logger.info("FINAL EM CORRECTION RESULT:")
            logger.info(
                f"  Agent Original: {self.em_correction_result['agent_original']}")
            logger.info(
                f"  Corrected Label: {self.em_correction_result['corrected_label']}")
            logger.info(f"  Action: {self.em_correction_result['action']}")
            logger.info(
                f"  P(EnvFail): {self.em_correction_result['P_EnvFail']:.3f}")
            logger.info(
                f"  P(AgentFail): {self.em_correction_result['P_AgentFail']:.3f}")
            logger.info("=" * 50)

        return rsp
