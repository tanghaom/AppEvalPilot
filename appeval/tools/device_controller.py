#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : device_controller.py
@Desc    : 设备控制工具类，用于操作Android和PC设备
"""

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pyautogui
import pyperclip
import uiautomator2 as u2
from metagpt.logs import logger
from pywinauto import Desktop
from pywinauto.controls.uiawrapper import UIAWrapper
from pywinauto.win32structures import RECT


class BaseController:
    """基础设备控制器类

    为Android和PC控制器提供通用功能。
    """

    def get_screenshot(self, filepath: str = "./screenshot/screenshot.jpg") -> None:
        """获取屏幕截图

        Args:
            filepath: 截图保存路径
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self._take_screenshot(filepath)
            logger.info(f"截图已保存至: {filepath}")
        except Exception as e:
            logger.error(f"截图失败: {str(e)}")

    def _take_screenshot(self, filepath: str) -> None:
        """实际执行截图的方法，由子类实现"""
        raise NotImplementedError

    def run_action(self, action: str) -> None:
        """执行动作

        Args:
            action: 动作描述字符串
        """
        logger.info(f"执行动作: {action}")
        # 使用列表保持动作顺序
        action_handlers = [
            ("Run", lambda x: hasattr(self, "_handle_run") and self._handle_run(x)),
            ("Tell", lambda x: hasattr(self, "_handle_tell") and self._handle_tell(x)),
        ]

        for action_type, handler in action_handlers:
            if action_type in action:
                handler(action)
                break

    def _handle_tell(self, action: str) -> None:
        """处理'Tell'动作"""
        # 获取action中的text
        text = self._extract_code(action)
        logger.info(f"处理'Tell'动作: {text}")

    def _extract_code(self, action: str) -> str:
        """从动作字符串中提取代码

        Args:
            action: 动作字符串

        Returns:
            str: 提取的代码
        """
        start = action.find("(")
        end = action.rfind(")")
        if start != -1 and end != -1 and end > start:
            code = action[start + 1 : end]
            return code.strip("```").replace("\n", "; ")
        return ""

    def _parse_coordinates(self, action: str) -> Tuple[int, int]:
        """解析坐标信息

        Args:
            action: 动作字符串

        Returns:
            Tuple[int, int]: 坐标元组
        """
        coords = action.split("(")[-1].split(")")[0].split(", ")
        return int(coords[0]), int(coords[1])

    def _parse_swipe_coordinates(self, action: str) -> Tuple[int, int, int, int]:
        """解析滑动坐标信息

        Args:
            action: 动作字符串

        Returns:
            Tuple[int, int, int, int]: 起点和终点坐标
        """
        coord1 = action.split("Swipe (")[-1].split("), (")[0].split(", ")
        coord2 = action.split("), (")[-1].split(")")[0].split(", ")
        return (int(coord1[0]), int(coord1[1]), int(coord2[0]), int(coord2[1]))

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """检查文本是否包含中文

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否包含中文
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)


class AndroidController(BaseController):
    """Android设备控制器类

    提供Android设备的基本操作功能，包括点击、滑动、输入等。
    """

    def __init__(self):
        """初始化Android控制器"""
        try:
            self.device = u2.connect()  # 连接设备
            u2.enable_pretty_logging()
            self.device.set_input_ime(False)  # 切换输入法
        except Exception as e:
            logger.error(f"初始化Android控制器失败: {str(e)}")
            raise

    def _take_screenshot(self, filepath: str) -> None:
        """实现Android设备的截图功能"""
        self.device.screenshot(filepath)

    def get_screen_xml(self, location_info: str = "center") -> List[Dict]:
        """获取屏幕XML信息

        Args:
            location_info: 位置信息格式('center'或'bbox')

        Returns:
            List[Dict]: 包含元素信息的列表
        """
        result = []
        screen_height = self.device.window_size()[1]
        xml = self.device.dump_hierarchy()
        root = ET.fromstring(xml)

        def get_element_text(element: ET.Element) -> str:
            """递归获取元素文本"""
            if element.attrib.get("text"):
                return element.attrib.get("text")
            for child in element:
                text = get_element_text(child)
                if text:
                    return text
            return ""

        for elem in root.iter():
            elem_class = elem.attrib.get("class", "")
            clickable = elem.attrib.get("clickable", "false")
            focusable = elem.attrib.get("focusable", "false")
            elem_text = get_element_text(elem)
            elem_id = elem.attrib.get("resource-id", "")
            elem_desc = elem.attrib.get("content-desc", "")

            bounds = elem.attrib.get("bounds", "")
            if bounds:
                bounds = bounds.replace("][", ",").replace("[", "").replace("]", "")
                bounds = list(map(int, bounds.split(",")))

                if bounds and (bounds[3] - bounds[1]) > screen_height / 2:
                    continue

                if clickable == "true" or (
                    focusable == "true"
                    and (elem_class == "android.widget.EditText" or elem_class == "android.widget.TextView")
                ):
                    center_x = int((bounds[0] + bounds[2]) / 2)
                    center_y = int((bounds[1] + bounds[3]) / 2)

                    result.append(
                        {
                            "coordinates": [center_x, center_y] if location_info == "center" else bounds,
                            "text": f"Class={elem_class}, Text={elem_text}, ID={elem_id}, Content-desc={elem_desc}, Bounds={bounds}",
                        }
                    )

        return result

    def get_all_packages(self) -> List[str]:
        """获取所有已安装应用包名

        Returns:
            List[str]: 包名列表
        """
        return self.device.app_list()

    def get_current_app_package(self) -> str:
        """获取当前运行应用的包名

        Returns:
            str: 当前应用包名
        """
        return self.device.app_current()["package"]

    def open_app(self, package_name: str) -> bool:
        """启动应用

        Args:
            package_name: 应用包名

        Returns:
            bool: 启动是否成功
        """
        package_name = package_name.split(":")[-1].strip()
        try:
            installed_packages = self.get_all_packages()
            if package_name not in installed_packages:
                logger.error(f"应用 {package_name} 未安装")
                return False

            self.device.app_start(package_name)
            logger.info(f"成功启动应用: {package_name}")
            return True

        except Exception as e:
            logger.error(f"启动应用失败: {str(e)}")
            return False

    def _handle_run(self, action: str) -> None:
        """处理'Run'动作"""
        code = self._extract_code(action)
        code = code.replace("self.device.tap(", "self.device.click(")
        code = self._add_ime_control(code)
        logger.info(f"执行代码: {code}")
        exec(code)

    def _add_ime_control(self, code: str) -> str:
        """为代码添加输入法控制

        Args:
            code: 原始代码

        Returns:
            str: 添加输入法控制后的代码
        """
        matches = re.finditer(r'self\.device\.send_keys\("""(.*?)"""(?:, clear=True)?\);', code)
        modified_code = code
        offset = 0

        for match in matches:
            send_keys = match.group(0)
            new_send_keys = f"self.device.set_input_ime(True); time.sleep(0.5); {send_keys} time.sleep(0.5); self.device.set_input_ime(False);"

            start_index = match.start() + offset
            end_index = match.end() + offset

            modified_code = modified_code[:start_index] + new_send_keys + modified_code[end_index:]
            offset += len(new_send_keys) - len(send_keys)

        return modified_code


class PCController(BaseController):
    """PC设备控制器类

    提供Windows/Mac设备的基本操作功能。
    """

    def __init__(
        self,
        search_keys: Tuple[str, str] = ("win", "s"),
        ctrl_key: str = "ctrl",
        pc_type: str = "windows",
    ):
        """初始化PC控制器

        Args:
            search_keys: 搜索快捷键
            ctrl_key: 控制键
            pc_type: 操作系统类型
        """
        try:
            self.search_keys = search_keys
            self.ctrl_key = ctrl_key
            self.pc_type = pc_type.lower()
        except Exception as e:
            logger.error(f"初始化PC控制器失败: {str(e)}")
            raise

    def _take_screenshot(self, filepath: str) -> None:
        """实现PC设备的截图功能"""
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)

    def open_app(self, name: str) -> None:
        """打开应用

        Args:
            name: 应用名称
        """
        logger.info(f"打开应用: {name}")
        pyautogui.hotkey(*self.search_keys)
        time.sleep(0.5)

        if self._contains_chinese(name):
            pyperclip.copy(name)
            pyautogui.hotkey(self.ctrl_key, "v")
        else:
            pyautogui.typewrite(name)

        time.sleep(1)
        pyautogui.press("enter")

    def get_screen_xml(self, location_info: str = "center") -> List[Dict]:
        """获取屏幕元素信息

        Args:
            location_info: 位置信息格式('center'或'bbox')

        Returns:
            List[Dict]: 元素信息列表
        """
        if self.pc_type == "mac":
            logger.warning("Mac OS暂未支持")
            return []
        t1 = time.time()
        try:
            # 获取所有可见的非任务栏窗口
            windows = [
                w
                for w in Desktop(backend="uia").windows()
                if w.is_visible() and w.texts() and w.texts()[0] not in ["任务栏", ""]
            ]

            if not windows:
                logger.warning("未找到活动窗口")
                return []

            active_window = windows[0]  # 获取第一个符合条件的窗口
            visible_rect = active_window.rectangle()
            t2 = time.time()
            logger.info(f"获取屏幕元素信息时间: {t2 - t1} 秒")
            processor = WindowsElementProcessor(visible_rect, location_info)
            return processor.process_element(active_window)

        except Exception as e:
            logger.error(f"获取屏幕元素信息失败: {str(e)}")
            return []

    def _handle_run(self, action: str) -> None:
        """处理'Run'动作"""
        code = self._extract_code(action)
        logger.info(f"执行代码: {code}")
        exec(code)


class WindowsElementProcessor:
    """Windows UI元素处理器类

    用于分析和处理Windows窗口中的UI元素。
    """

    def __init__(self, visible_rect: RECT, location_info: str = "center"):
        """初始化Windows元素处理器

        Args:
            visible_rect (RECT): 可见区域的矩形范围
            location_info (str): 位置信息格式，可选值为 'center' 或 'bbox'
        """
        self.visible_rect = visible_rect
        self.location_info = location_info
        self.SPECIAL_CONTROL_TYPES = {"Hyperlink", "TabItem", "Button", "ComboBox", "ScrollBar", "Edit", "ToolBar"}

    def process_element(self, element: UIAWrapper, depth: int = 0) -> List[Dict[str, Union[Tuple[int, ...], str]]]:
        """处理UI元素

        Args:
            element (UIAWrapper): 要处理的UI元素
            depth (int): 递归深度，默认为0

        Returns:
            List[Dict[str, Union[Tuple[int, ...], str]]]: 元素信息列表，每个元素包含坐标和文本信息
        """
        current_elements_info = []
        rect = element.rectangle()

        if element.friendly_class_name() == "TitleBar":
            return current_elements_info

        control_type = element.element_info.control_type
        text = element.window_text()

        if rect.width() > 0 and rect.height() > 0 and self._is_element_visible(rect):
            if element.is_enabled():
                coordinates = self._calculate_coordinates(rect)
                rect_str = f"({rect.left}, {rect.top}, {rect.right}, {rect.bottom})"
                current_elements_info.append(
                    {"coordinates": coordinates, "text": f"text:{text}; control_type:{control_type}; rect: {rect_str}"}
                )

        for child in element.children():
            child_text = child.window_text()
            if not (child.element_info.control_type == "Edit" and child_text and child_text == text):
                current_elements_info.extend(self.process_element(child, depth + 1))

        return current_elements_info

    def _is_element_visible(self, element_rect: RECT) -> bool:
        """检查元素是否可见

        Args:
            element_rect (RECT): 元素的矩形区域

        Returns:
            bool: 如果元素在可见区域内返回True，否则返回False
        """
        return not (
            element_rect.right < self.visible_rect.left
            or element_rect.left > self.visible_rect.right
            or element_rect.bottom < self.visible_rect.top
            or element_rect.top > self.visible_rect.bottom
        )

    def _calculate_coordinates(self, rect: RECT) -> Union[Tuple[int, int], Tuple[int, int, int, int]]:
        """计算元素坐标

        Args:
            rect (RECT): 元素的矩形区域

        Returns:
            Union[Tuple[int, int], Tuple[int, int, int, int]]:
                如果location_info为'center'则返回中心点坐标(x, y)
                如果location_info为'bbox'则返回边界框坐标(left, top, right, bottom)
        """
        if self.location_info == "center":
            return ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
        return (rect.left, rect.top, rect.right, rect.bottom)


class ControllerTool:
    """设备控制工具类

    提供统一的设备控制接口，支持Android和PC设备。
    """

    def __init__(self, platform: str = "Android", **kwargs):
        self.controller = AndroidController() if platform == "Android" else PCController(**kwargs)

    def __getattr__(self, name):
        """代理所有方法调用到具体控制器"""
        return getattr(self.controller, name)


def create_controller(platform: str = "Android", **kwargs) -> ControllerTool:
    """创建控制器工具实例

    Args:
        platform: 平台类型
        **kwargs: 其他参数

    Returns:
        ControllerTool: 控制器工具实例

    Raises:
        ValueError: 设备类型无效时抛出
    """
    if platform not in ["Android", "Windows"]:
        raise ValueError(f"不支持的设备类型: {platform}")
    return ControllerTool(platform, **kwargs)


if __name__ == "__main__":
    # 测试代码
    def test_android():
        controller = create_controller("Android")
        controller.get_screenshot()
        elements = controller.get_screen_xml()
        print("Android元素信息:", elements)

    def test_pc():
        controller = create_controller("Windows", search_keys=("win", "s"), ctrl_key="ctrl")
        controller.get_screenshot()
        elements = controller.get_screen_xml()
        print("PC元素信息:", elements)

    # test_android()
    # test_pc()
