#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/10
@Author  : tanghaoming
@File    : osagent.py
@Desc    : 操作系统操作助手
"""
import argparse
import asyncio
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

from appeval.actions.reflection import Reflection
from appeval.actions.screen_info_extractor import ScreenInfoExtractor
from appeval.prompts.osagent import ActionPromptContext, Android_prompt, PC_prompt
from appeval.tools.chrome_debugger import ChromeDebugger
from appeval.tools.device_controller import ControllerTool
from appeval.tools.icon_detect import IconDetectTool
from appeval.tools.ocr import OCRTool

# 忽略所有警告
warnings.filterwarnings("ignore")


class OSAgentContext(RoleContext):
    """OSAgent运行时上下文"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thought: str = ""  # 当前思考内容
    thought_history: List[str] = Field(default_factory=list)  # 历史思考记录列表
    summary_history: List[str] = Field(default_factory=list)  # 历史操作摘要列表
    action_history: List[str] = Field(default_factory=list)  # 历史执行动作列表
    reflection_thought_history: List[str] = Field(default_factory=list)  # 历史反思记录列表
    reflection_thought: str = ""  # 当前反思内容
    summary: str = ""  # 当前操作摘要
    action: str = ""  # 当前执行动作
    task_list: str = ""  # 任务列表
    completed_requirements: str = ""  # 已完成的需求
    memory: List[str] = Field(default_factory=list)  # 重要内容记忆列表
    error_flag: bool = False  # 错误标志
    iter: int = 0  # 当前迭代次数
    perception_infos: List[Dict] = Field(default_factory=list)  # 当前感知信息列表
    last_perception_infos: List[Dict] = Field(default_factory=list)  # 上一次感知信息列表
    width: int = 0  # 屏幕宽度
    height: int = 0  # 屏幕高度
    webbrowser_console_logs: List[Any] = Field(default_factory=list)  # 浏览器控制台日志列表

    def reset(self) -> None:
        """重置所有状态为初始值"""
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
        self.iter = 0
        self.perception_infos = []
        self.last_perception_infos = []
        self.width = 0
        self.height = 0
        self.webbrowser_console_logs = []


class OSAgent(Role):
    """操作系统代理类,用于执行自动化任务"""

    name: str = "OSAgent"
    profile: str = "OS Agent"
    goal: str = "执行自动化任务"
    constraints: str = "确保任务执行的准确性和效率"
    desc: str = "操作系统代理类,用于执行自动化任务"

    rc: OSAgentContext = Field(default_factory=OSAgentContext)

    def __init__(
        self,
        # 基础配置参数
        platform: str = "Android",
        max_iters: int = 50,
        # 功能开关参数
        use_ocr: bool = True,
        quad_split_ocr: bool = False,
        use_icon_detect: bool = True,
        use_icon_caption: bool = True,
        use_memory: bool = True,
        use_reflection: bool = True,
        use_som: bool = False,
        extend_xml_infos: bool = True,
        use_chrome_debugger: bool = False,
        # 显示和布局参数
        location_info: str = "center",
        draw_text_box: bool = False,
        # 路径相关参数
        log_dirs: str = "workspace",
        font_path: str = str(Path(__file__).parent / "simhei.ttf"),
        knowledge_base_path: str = str(Path(__file__).parent),
        # 其他可选参数
        system_prompt: str = "",
        add_info: str = "",
        **kwargs,
    ) -> None:
        """初始化 OSAgent。

        Args:
            platform (str): 操作系统类型 (Windows, Mac, or Android)。
            max_iters (int): 最大迭代次数。
            use_ocr (bool): 是否使用OCR。
            quad_split_ocr (bool): 是否将图像分割成四部分进行ocr识别。
            use_icon_detect (bool): 是否使用图标检测。
            use_icon_caption (bool): 是否使用图标描述。
            use_memory (bool): 是否开启重要内容记忆。
            use_reflection (bool): 是否进行反思。
            use_som (bool): 是否在截图上绘制可视化框。
            extend_xml_infos (bool): 是否增加获取xml元素信息。
            use_chrome_debugger (bool): 是否记录浏览器控制台的输出。
            location_info (str): 位置信息类型 (center or bbox)。
            draw_text_box (bool): 是否在可视化框中绘制文本框。
            log_dirs (str): 日志存放目录
            font_path (str): 字体路径。
            knowledge_base_path (str): 预设知识库文件所在目录路径
            system_prompt (str): 系统提示词
            add_info (str): 添加到提示中的额外信息
        """
        super().__init__(**kwargs)

        # 保存配置参数
        self._init_config(locals())

        # 初始化环境
        self._init_environment()

        # 初始化工具
        self._init_tools()

    def _init_config(self, params: dict) -> None:
        """初始化配置参数"""
        # 过滤掉 self 和 kwargs
        config_params = {k: v for k, v in params.items() if k not in ["self", "kwargs"]}
        for key, value in config_params.items():
            setattr(self, key, value)

        # 设置默认的额外提示信息
        if not self.add_info:
            self.add_info = self._get_default_add_info()

    def _get_default_add_info(self) -> str:
        """获取默认的额外提示信息"""
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
        """初始化运行环境"""
        # 初始化路径
        self._get_timestamped_paths()

        # 初始化日志
        self._setup_logs()

        # 初始化操作系统环境
        self._init_os_env()

    def _init_tools(self) -> None:
        """初始化工具组件"""
        # 初始化信息提取器
        self.info_extractor = ScreenInfoExtractor(platform=self.platform)

        # 初始化反思器
        self.reflection_action = Reflection(platform=self.platform)

        # 初始化图标检测/描述工具
        if self.use_icon_detect or self.use_icon_caption:
            self.icon_tool = IconDetectTool(self.llm)

        # 初始化OCR工具
        if self.use_ocr:
            self.ocr_tool = OCRTool()

        # 初始化浏览器调试器
        if self.use_chrome_debugger:
            self.chrome_debugger = ChromeDebugger()

    def _get_timestamped_paths(self) -> None:
        """更新带时间戳的文件路径"""
        current_time = time.strftime("%Y%m%d%H%M")

        # 基础路径
        log_dir = Path(self.log_dirs) / current_time
        self.save_info = str(log_dir / "info.txt")
        self.save_img = str(log_dir)

        # 截图相关路径
        self.screenshot_dir = log_dir / "screenshot"
        self.screenshot_file = str(self.screenshot_dir / "screenshot.jpg")
        self.screenshot_som_file = str(self.screenshot_dir / "screenshot_som.png")
        self.last_screenshot_file = str(self.screenshot_dir / "last_screenshot.jpg")
        self.last_screenshot_som_file = str(self.screenshot_dir / "last_screenshot_som.png")

    def _init_os_env(self) -> None:
        """初始化操作系统环境。

        根据不同平台(Android/Windows/Mac)初始化相应的控制器和提示工具。
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
        logger.info(f"初始化控制器: {config['controller_args']}")
        self.controller = ControllerTool(**config["controller_args"])
        self.prompt_utils = config["prompt_class"]()

    def _reset_state(self) -> None:
        """重置状态，用于每次run新的任务时清空之前的记录"""
        # 重置 rc 中的状态
        self.rc.reset()

        # 重置临时文件和目录
        self._get_timestamped_paths()

        # 重置其他状态
        self.run_action_failed = False
        self.run_action_failed_exception = ""

        if self.use_chrome_debugger:
            self.chrome_debugger.start_monitoring()

        # 重新创建截图目录
        if self.screenshot_dir.exists():
            shutil.rmtree(self.screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logs(self) -> None:
        """设置日志记录"""
        log_dir = Path(self.save_info).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # 移除以前可能存在的日志处理器
        logger.remove()

        # 定义日志格式
        log_format = "{time:YYYY-MM-DD HH:mm:ss} | " "{level:<8} | " "{module}:{function}:{line} - " "{message}"

        # 添加文件日志处理器
        logger.add(
            self.save_info,
            level="DEBUG",
            format=log_format,
            mode="w",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

        # 添加控制台日志处理器
        logger.add(sys.stdout, level="DEBUG", format=log_format, colorize=True, enqueue=True)

        logger.info(f"初始化日志记录, 日志文件: {self.save_info}")

    def _convert_cmyk_to_rgb(self, c: int, m: int, y: int, k: int) -> tuple[int, int, int]:
        """将 CMYK 颜色值转换为 RGB 颜色值。

        Args:
            c (int): 青色值 (0-255)
            m (int): 洋红色值 (0-255)
            y (int): 黄色值 (0-255)
            k (int): 黑色值 (0-255)

        Returns:
            tuple[int, int, int]: RGB颜色值 (r, g, b)
        """
        r = int(255 * (1.0 - c / 255) * (1.0 - k / 255))
        g = int(255 * (1.0 - m / 255) * (1.0 - k / 255))
        b = int(255 * (1.0 - y / 255) * (1.0 - k / 255))
        return r, g, b

    def _draw_bounding_boxes(
        self, image_path: str, coordinates: List[List[int]], output_path: str, font_path: str
    ) -> None:
        """在图像上绘制带有编号的坐标框。

        Args:
            image_path (str): 图像路径。
            coordinates (list): 坐标框列表，每个坐标框是一个包含四个元素的列表 [x1, y1, x2, y2]。
            output_path (str): 输出图像路径。
            font_path (str): 字体路径。
        """
        # 打开图像并获取尺寸
        image = Image.open(image_path)
        height = image.size[1]

        # 计算绘制参数
        line_width = int(height * 0.0025)
        font_size = int(height * 0.012)
        text_offset_x = line_width
        text_offset_y = int(height * 0.013)

        # 为每个边界框生成随机颜色
        colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(coordinates))]

        # 绘制边界框和编号
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        for i, (coord, color) in enumerate(zip(coordinates, colors)):
            # 直接使用RGB颜色绘制边界框
            draw.rectangle(coord, outline=color, width=line_width)

            # 计算文本位置并绘制编号
            text_x = coord[0] + text_offset_x
            text_y = max(0, coord[1] - text_offset_y)
            draw.text((text_x, text_y), str(i + 1), fill=color, font=font)

        # 保存结果
        image.convert("RGB").save(output_path)

    def _save_iteration_images(self, iter_num: int) -> None:
        """保存当前迭代的原始图片和标注图片。

        Args:
            iter_num: 当前迭代次数
        """
        # 构建文件路径
        origin_path = f"{self.save_img}/origin_{iter_num}.jpg"
        draw_path = f"{self.save_img}/draw_{iter_num}.jpg"

        # 复制图片文件
        shutil.copy2(self.screenshot_file, origin_path)
        shutil.copy2(self.output_image_path, draw_path)

    def _update_screenshot_files(self) -> None:
        """更新截图文件"""
        # 更新普通截图
        last_screenshot = Path(self.last_screenshot_file)
        if last_screenshot.exists():
            last_screenshot.unlink()
        Path(self.screenshot_file).rename(last_screenshot)

        # 更新 SOM 截图
        if self.use_som:
            last_screenshot_som = Path(self.last_screenshot_som_file)
            if last_screenshot_som.exists():
                last_screenshot_som.unlink()
            Path(self.screenshot_som_file).rename(last_screenshot_som)

    def _check_last_three_start_with_wait(self, string_list: List[str]) -> bool:
        """检查列表最后三个字符串是否都以 "Wait" 开头。

        Args:
            string_list (list): 字符串列表。

        Returns:
            bool: 如果最后三个字符串都以 "Wait" 开头，则返回 True，否则返回 False。
        """
        if len(string_list) < 3:
            return False
        return all(s.startswith("Wait") for s in string_list[-3:])

    def _get_app_info(self) -> Optional[str]:
        """从预设的app_info.json文件中获取应用辅助信息。"""
        info_path = Path(self.knowledge_base_path) / "app_info.json"
        if not info_path.exists():
            return None
        app_info = json.loads(info_path.read_text(encoding="utf-8"))
        package_name = self.controller.get_current_app_package()
        if not package_name:
            return None
        return app_info.get(package_name, None)

    async def _get_perception_infos(
        self, screenshot_file: str, screenshot_som_file: str
    ) -> Tuple[List[Dict[str, Any]], int, int, str]:
        """获取感知信息，包括OCR和图标检测。
        Args:
            screenshot_file (str): 截图文件路径。
            screenshot_som_file (str): 带有可视化框的截图文件路径。
        Returns:
            tuple: 包含感知信息列表、图像宽度、图像高度和输出图像路径的元组。
        """
        # 获取屏幕截图
        self.controller.get_screenshot(screenshot_file)
        # 获取屏幕截图的宽度和高度
        width, height = Image.open(screenshot_file).size

        # OCR处理
        text, text_coordinates = [], []
        if self.use_ocr:
            text, text_coordinates = self.ocr_tool.ocr(screenshot_file, split=self.quad_split_ocr)

        # 图标检测
        icon_coordinates = []
        if self.use_icon_detect:
            icon_coordinates = self.icon_tool.detect(screenshot_file)

        # 处理输出图像
        output_image_path = screenshot_som_file
        if self.use_ocr and self.use_icon_detect and self.draw_text_box:
            rec_list = text_coordinates + icon_coordinates
            self._draw_bounding_boxes(screenshot_file, copy.deepcopy(rec_list), screenshot_som_file, self.font_path)
        elif self.use_icon_detect:
            self._draw_bounding_boxes(
                screenshot_file, copy.deepcopy(icon_coordinates), screenshot_som_file, self.font_path
            )
        else:
            output_image_path = screenshot_file

        # 构建感知信息
        mark_number = 0
        perception_infos = []

        # 添加OCR文本信息
        if self.use_ocr:
            for i in range(len(text_coordinates)):
                mark_number += 1
                if self.use_som and self.draw_text_box:
                    perception_info = {
                        "text": f"mark number: {mark_number} text: {text[i]}",
                        "coordinates": text_coordinates[i],
                    }
                else:
                    perception_info = {"text": f"text: {text[i]}", "coordinates": text_coordinates[i]}
                perception_infos.append(perception_info)

        # 添加图标信息
        if self.use_icon_detect:
            for i in range(len(icon_coordinates)):
                mark_number += 1
                if self.use_som:
                    perception_info = {"text": f"mark number: {mark_number} icon", "coordinates": icon_coordinates[i]}
                else:
                    perception_info = {"text": "icon", "coordinates": icon_coordinates[i]}
                perception_infos.append(perception_info)

        # 图标描述
        if self.use_icon_detect and self.use_icon_caption:
            icon_indices = [i for i in range(len(perception_infos)) if "icon" in perception_infos[i]["text"]]
            if icon_indices:
                icon_boxes = [perception_infos[i]["coordinates"] for i in icon_indices]
                descriptions = await self.icon_tool.caption(screenshot_file, icon_boxes, platform=self.platform)

                # 将描述添加到感知信息中
                for idx, desc_idx in enumerate(icon_indices):
                    if descriptions.get(idx + 1):
                        perception_infos[desc_idx]["text"] += ": " + descriptions[idx + 1].replace("\n", " ")

        # 根据参数修改坐标信息
        if self.location_info == "center":
            for i in range(len(perception_infos)):
                x1, y1, x2, y2 = perception_infos[i]["coordinates"]
                perception_infos[i]["coordinates"] = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        elif self.location_info == "icon_center":
            for i in range(len(perception_infos)):
                if "icon" in perception_infos[i]["text"]:
                    x1, y1, x2, y2 = perception_infos[i]["coordinates"]
                    perception_infos[i]["coordinates"] = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

        # 如果启用 extend_xml_infos，则添加 XML 信息
        if self.extend_xml_infos and self.platform in ["Android", "Windows"]:
            xml_results = self.controller.get_screen_xml(self.location_info)
            logger.debug(xml_results)
            perception_infos.extend(xml_results)

        return perception_infos, width, height, output_image_path

    def get_webbrowser_console_logs(self, steps: int = 100, expand: bool = True) -> List[Any]:
        """
        获取最近的网页浏览器控制台日志。
        注意：仅用于mgx自动测试网页。
        Args:
            steps (int, optional): 要获取的日志数量，默认为 1。
            expand (bool, optional): 是否返回展开的日志列表，默认为 True。
                如果为 True，则返回最近的 `steps` 个日志列表。
                如果为 False，则返回最近的 `steps` 个日志字典列表，包含对应的操作和控制台输出。
        Returns:
            list: 最近的控制台日志列表或字典列表。
        """
        if not self.rc.webbrowser_console_logs:
            return []  # 如果没有日志，直接返回空列表
        if expand:
            return [log for log in self.rc.webbrowser_console_logs[-steps:] if log]  # 过滤空列表
        else:
            # 使用 zip 将操作历史和日志对应起来
            outputs = [
                {"action": action, "console_output": log}
                for action, log in zip(self.rc.summary_history, self.rc.webbrowser_console_logs)
                if log  # 过滤空列表
            ]
            return outputs[-steps:]

    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        获取行动历史记录，包括思考、总结、行动、可选的记忆和反思。
        Returns:
            list: 一个包含字典的列表，每个字典代表一个行动步骤的历史记录。
                  每个字典包含 "thought" (思考), "summary" (总结), "action" (行动),
                  以及可选的 "memory" (记忆) 和 "reflection" (反思)。
        """
        outputs = []
        # 使用 zip 将三个历史列表的对应元素打包成元组，并使用 enumerate 获取索引
        for i, (thought, summary, action) in enumerate(
            zip(self.rc.thought_history, self.rc.summary_history, self.rc.action_history)
        ):
            output = {"thought": thought, "summary": summary, "action": action}  # 当前步骤的思考  # 当前步骤的总结  # 当前步骤的行动
            # 如果启用了记忆开关，则添加记忆信息
            if self.use_memory:
                output["memory"] = self.rc.memory[i]
            # 如果启用了反思开关，则添加反思信息
            if self.use_reflection:
                output["reflection"] = self.rc.reflection_thought_history[i]
            outputs.append(output)  # 将当前步骤的信息添加到输出列表中
        return outputs

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"信息提取失败，第{retry_state.attempt_number}次重试: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _async_memory_task(self, insight: str, screenshot_file: str) -> str:
        """异步执行信息提取任务

        Args:
            insight (str): 需要关注的内容/任务描述
            screenshot_file (str): 截图文件路径

        Returns:
            str: 提取的重要内容
        """
        if not self.use_memory:
            return ""

        return await self.info_extractor.run(insight, screenshot_file)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"生成操作决策失败，第{retry_state.attempt_number}次重试: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _think(self) -> bool:
        """生成操作决策"""
        # 添加预设知识
        add_info = self.add_info
        # 添加应用信息到提示
        if self.platform == "Android":
            info = self._get_app_info()
            if info:
                add_info += " ".join(info) if isinstance(info, list) else info
            else:
                info = "没有相关知识库"
            if info == "":
                info = "无需add_info"
            logger.info(f"\n\n\n\n\n\n#### add_info:{info}\n\n")
        else:
            logger.info("目前只实现了安卓的知识库")

        # 生成action
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
            completed_content=self.rc.completed_requirements,
            memory=self.rc.memory,
            task_list=self.rc.task_list,
            use_som=self.use_som,
            location_info=self.location_info,
        )

        prompt_action = self.prompt_utils.get_action_prompt(ctx)
        logger.info(
            f"\n\n######################## prompt_action:\n{prompt_action}\n\n######################## prompt_action end\n\n\n\n"
        )

        # 调用LLM生成决策
        images = [encode_image(self.screenshot_file)]
        if self.use_som:
            images.append(encode_image(self.screenshot_som_file))

        # 使用自定义系统提示词或默认提示词
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

        # 解析输出
        self.rc.thought = (
            output_action.split("### Thought ###")[-1]
            .split("### Action ###")[0]
            .replace("\n", " ")
            .replace(":", "")
            .replace("  ", " ")
            .strip()
        )
        self.rc.action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].strip()
        self.rc.summary = (
            output_action.split("### Operation ###")[-1].split("### Task List ###")[0].strip().replace("\n", "\\n")
        )
        self.rc.task_list = output_action.split("### Task List ###")[-1].strip()

        logger.info(
            f"\n\n######################## output_action:\n{output_action}\n\n######################## output_action end\n\n\n\n"
        )

        if "Stop" in self.rc.action:
            return False
        else:
            return True

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"执行反思失败，第{retry_state.attempt_number}次重试: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _reflection(
        self,
        instruction: str,
        last_perception_infos: list,
        perception_infos: list,
        width: int,
        height: int,
        summary: str,
        action: str,
        add_info: str,
        last_screenshot_file: str,
        screenshot_file: str,
    ) -> Tuple[str, str]:
        """执行反思任务

        Args:
            instruction (str): 用户指令
            last_perception_infos (list): 上一次的感知信息
            perception_infos (list): 当前的感知信息
            width (int): 截图宽度
            height (int): 截图高度
            summary (str): 操作总结
            action (str): 执行的动作
            add_info (str): 额外信息
            last_screenshot_file (str): 上一次截图路径
            screenshot_file (str): 当前截图路径

        Returns:
            tuple: (反思结果, 反思内容)
        """
        if not self.use_reflection:
            return "", ""

        if "Tell" in action or "Wait" in action:
            return "", "When using the Tell or Wait action, there is no need to do reflection."

        reflect, reflection_thought = await self.reflection_action.run(
            instruction,
            last_perception_infos,
            perception_infos,
            width,
            height,
            summary,
            action,
            add_info,
            last_screenshot_file,
            screenshot_file,
        )

        return reflect, reflection_thought

    async def _get_app_package_name(self, app_name: str) -> str:
        """获取应用包名

        Args:
            app_name (str): 应用名称

        Returns:
            str: 应用包名
        """
        package_list = self.controller.get_all_packages()

        # 读取应用映射信息
        map_path = Path(self.knowledge_base_path) / "app_mapping.json"
        app_mapping = ""
        if map_path.exists():
            app_mapping = map_path.read_text(encoding="utf-8").strip()
        else:
            logger.warning(f"{map_path} 文件不存在，使用默认空映射")

        # 获取包名
        prompt_package_name = self.prompt_utils.get_package_name_prompt(
            app_name=app_name, app_mapping=app_mapping, package_list=package_list
        )

        package_name = await self.llm.aask(
            prompt_package_name,
            system_msgs=[
                f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant."
            ],
            stream=False,
        )

        return package_name.strip()

    async def _handle_open_app(self) -> None:
        """处理打开应用的动作"""
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
            logger.error(f"Platform {self.platform} not supported for opening apps")

    async def _act(self) -> Message:
        """执行行动步骤"""
        if self.use_chrome_debugger:
            # 执行动作前，把浏览器日志存到上一次的动作日志里。注意：这里需要存在一个第0步的日志，因为mgx网页测试的时候不是osagent去启动的
            self.rc.webbrowser_console_logs.append(self.chrome_debugger.get_new_messages())

        self.run_action_failed = False
        self.run_action_failed_exception = ""

        # 执行动作
        if "Stop" in self.rc.action:
            # 如果是停止操作，则结束循环
            return AIMessage(content=self.rc.action, cause_by=Action)
        elif "Open App" in self.rc.action:
            await self._handle_open_app()
        else:
            # 执行其他动作
            try:
                if self.platform in ["Android", "Windows"]:
                    self.controller.run_action(self.rc.action)
                else:
                    logger.error("目前只支持Android和Windows")
            except Exception as e:
                # 用于automg使用tell的时候能直接退出
                if isinstance(e, SystemExit) and e.code == 0:
                    return AIMessage(content=self.rc.action, cause_by=Action)
                logger.error(f"run action failed: {e}")
                self.run_action_failed = True
                self.run_action_failed_exception = e

        time.sleep(0.5)
        # 保存上一次的感知信息和截图
        self.rc.last_perception_infos = copy.deepcopy(self.rc.perception_infos)

        # 更新截图文件
        self._update_screenshot_files()

        # 获取新的感知信息
        self.rc.perception_infos, self.width, self.height, self.output_image_path = await self._get_perception_infos(
            self.screenshot_file, self.screenshot_som_file
        )

        # 保存图片
        self._save_iteration_images(self.rc.iter)

        # 异步执行记忆任务
        memory_task = None
        if self.use_memory:
            memory_task = asyncio.create_task(self._async_memory_task(self.instruction, self.screenshot_file))

        # 更新历史记录
        self.rc.thought_history.append(self.rc.thought)
        self.rc.summary_history.append(self.rc.summary)
        self.rc.action_history.append(self.rc.action)

        if self.run_action_failed:
            self.rc.reflection_thought_history.append(
                f"ERROR(run action code filed): {self.run_action_failed_exception}\\n "
            )
            self.rc.error_flag = True

        elif self.use_reflection:
            # 执行反思
            reflect, self.rc.reflection_thought = await self._reflection(
                self.instruction,
                self.rc.last_perception_infos,
                self.rc.perception_infos,
                self.width,
                self.height,
                self.rc.summary,
                self.rc.action,
                self.add_info,
                self.last_screenshot_file,
                self.screenshot_file,
            )
            self.rc.reflection_thought_history.append(self.rc.reflection_thought)
            if reflect == "CORRECT":
                self.rc.error_flag = False
            elif reflect == "ERROR":
                self.rc.error_flag = True

        # 清理截图
        if self.use_som:
            Path(self.last_screenshot_som_file).unlink()
        Path(self.last_screenshot_file).unlink()

        # 等待记忆任务完成并保存结果
        if memory_task:
            memory_content = await memory_task
            self.rc.memory.append(memory_content)

        return AIMessage(content=self.rc.action, cause_by=Action)

    async def _react(self) -> Message:
        self.rc.iter = 0
        rsp = AIMessage(content="No actions taken yet", cause_by=Action)  # will be overwritten after Role _act
        while self.rc.iter < self.max_iters and not self._check_last_three_start_with_wait(self.rc.action_history):
            self.rc.iter += 1

            logger.info(f"\n\n\n\n\n\n#### iter:{self.rc.iter}\n\n")

            # 获取初始感知信息
            if self.rc.iter == 1:
                (
                    self.rc.perception_infos,
                    self.width,
                    self.height,
                    self.output_image_path,
                ) = await self._get_perception_infos(self.screenshot_file, self.screenshot_som_file)

                # 保存图片
                self._save_iteration_images(self.rc.iter)

            # think
            has_todo = await self._think()
            if not has_todo:
                rsp = AIMessage(content="OS Agent has finished all tasks", cause_by=Action)
                break
            # act
            logger.debug(f"{self._setting}: {self.rc.state=}, will do {self.rc.todo}")
            rsp = await self._act()

        if self.use_chrome_debugger:
            self.chrome_debugger.stop_monitoring()

        return rsp

    async def run(self, instruction: str) -> Message:
        """运行主循环。

        Args:

            instruction (str): 用户指令。
        """
        self._reset_state()  # 在每次run的时候重置状态
        self._setup_logs()  # 每次run都重新设置日志
        self.instruction = instruction

        rsp = await self.react()
        return rsp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OS Agent")
    # 基础配置参数
    parser.add_argument("--platform", type=str, default="Windows", help="操作系统类型 (Windows, Mac, or Android)")
    parser.add_argument("--max_iters", type=int, default=50, help="最大迭代次数")
    parser.add_argument("--instruction", type=str, default="搜索厦门明天天气", help="用户指令")

    # 功能开关参数
    parser.add_argument("--use_ocr", type=int, default=0, help="是否使用OCR")
    parser.add_argument("--use_icon_detect", type=int, default=0, help="是否使用图标检测")
    parser.add_argument("--use_icon_caption", type=int, default=0, help="是否使用图标描述")
    parser.add_argument("--use_memory", type=int, default=1, help="是否开启重要内容记忆")
    parser.add_argument("--use_reflection", type=int, default=1, help="是否进行反思")
    parser.add_argument("--use_som", type=int, default=0, help="是否在截图上绘制可视化框")
    parser.add_argument("--extend_xml_infos", type=int, default=1, help="是否增加获取xml元素信息")
    parser.add_argument("--use_chrome_debugger", type=int, default=0, help="是否记录浏览器控制台的输出")

    # 显示和布局参数
    parser.add_argument("--location_info", type=str, default="center", help="位置信息类型 (center or bbox)")
    parser.add_argument("--draw_text_box", type=int, default=0, help="是否在可视化框中绘制文本框")
    parser.add_argument("--quad_split_ocr", type=int, default=0, help="是否将图像分割成四部分进行OCR识别")

    # 路径相关参数
    parser.add_argument("--log_dirs", type=str, default="workspace", help="日志存放目录")
    parser.add_argument("--font_path", type=str, default=str(Path(__file__).parent / "simhei.ttf"), help="字体路径")
    parser.add_argument("--knowledge_base_path", type=str, default=str(Path(__file__).parent), help="预设知识库文件所在目录路径")

    # 其他可选参数
    parser.add_argument("--add_info", type=str, default="", help="添加到提示中的额外信息")
    args = parser.parse_args()
    if args.add_info == "":
        if args.platform == "Windows":
            args.add_info = "If you need to interact with elements outside of a web popup, such as calendar or time selection popups, make sure to close the popup first. If the content in a text box is entered incorrectly, use the select all and delete actions to clear it, then re-enter the correct information. To open a folder in File Explorer, please use a double-click. "
        elif args.platform == "Android":
            args.add_info = "If you need to open an app, prioritize using the Open app (app name) action. If this fails, return to the home screen and click the app icon on the desktop. If you want to exit an app, return to the home screen. If there is a popup ad in the app, you should close the ad first. If you need to switch to another app, you should first return to the desktop. When summarizing content, comparing items, or performing cross-app actions, remember to leverage the content in memory. "

    async def main():
        agent = OSAgent(
            platform=args.platform,
            max_iters=args.max_iters,
            use_ocr=args.use_ocr,
            use_icon_detect=args.use_icon_detect,
            use_icon_caption=args.use_icon_caption,
            use_memory=args.use_memory,
            use_reflection=args.use_reflection,
            use_som=args.use_som,
            extend_xml_infos=args.extend_xml_infos,
            use_chrome_debugger=args.use_chrome_debugger,
            location_info=args.location_info,
            draw_text_box=args.draw_text_box,
            quad_split_ocr=args.quad_split_ocr,
            log_dirs=args.log_dirs,
            font_path=args.font_path,
            knowledge_base_path=args.knowledge_base_path,
        )
        await agent.run(args.instruction)
        history = agent.get_action_history()
        print(f"history: {history}")
        if args.use_chrome_debugger:
            console_logs = agent.get_webbrowser_console_logs(steps=args.max_iters)
            print(f"console_logs: {console_logs}")

    asyncio.run(main())
