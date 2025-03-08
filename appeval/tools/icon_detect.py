#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : icon_detect.py
@Desc    : 图标检测和描述工具类
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Union

import requests
from metagpt.const import DATA_PATH, TEST_DATA_PATH
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.utils.common import encode_image
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from ultralytics import YOLO

# 类型别名
BoundingBox = List[int]  # [x1, y1, x2, y2]
Coordinates = List[BoundingBox]


class IconDetector:
    """图标检测工具类"""

    MODEL_URL = "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt"
    MODEL_PATH = DATA_PATH / "omniparser_icon_detect.pt"

    def __init__(self):
        """初始化检测器"""
        self.model = self._init_model()

    def _init_model(self) -> YOLO:
        """初始化并返回YOLO模型"""
        if not self.MODEL_PATH.exists():
            self._download_model()
        return YOLO(str(self.MODEL_PATH))

    def _download_model(self):
        """下载模型文件"""
        logger.info("正在下载OmniParser图标检测模型...")
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        try:
            with requests.get(self.MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(self.MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("OmniParser模型下载完成")
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            raise

    @staticmethod
    def _calculate_area(box: BoundingBox) -> float:
        """计算边界框面积
        Args:
            box: 边界框坐标 [x1, y1, x2, y2]
        Returns:
            float: 边界框的面积
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def _calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """计算两个边界框的交并比(IoU)
        Args:
            box1: 第一个边界框坐标 [x1, y1, x2, y2]
            box2: 第二个边界框坐标 [x1, y1, x2, y2]
        Returns:
            float: 两个边界框的IoU值
        """
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1, area2 = IconDetector._calculate_area(box1), IconDetector._calculate_area(box2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _filter_boxes(
        self, boxes: Coordinates, image_size: Tuple[int, int], iou_threshold: float = 0.5, area_threshold: float = 0.05
    ) -> Coordinates:
        """过滤和合并重叠的边界框
        Args:
            boxes: 边界框列表
            image_size: 图片尺寸 (宽,高)
            iou_threshold: IoU阈值
            area_threshold: 面积阈值(占比)
        Returns:
            List[List[int]]: 过滤后的边界框列表
        """
        if not boxes:
            return []

        img_area = image_size[0] * image_size[1]
        max_area = img_area * area_threshold

        # 按面积从大到小排序并过滤过大的框
        valid_boxes = [box for box in sorted(boxes, key=self._calculate_area) if self._calculate_area(box) <= max_area]

        filtered_boxes = []
        for box in valid_boxes:
            # 如果当前框与已保留的框IoU都小于阈值，则保留
            if not any(self._calculate_iou(box, existing_box) > iou_threshold for existing_box in filtered_boxes):
                filtered_boxes.append(box)

        return filtered_boxes

    def detect(
        self, image_path: Union[str, Path], conf_threshold: float = 0.25, iou_threshold: float = 0.3
    ) -> Coordinates:
        """检测图片中的图标
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        Returns:
            List[List[int]]: 检测到的图标坐标列表
        """
        try:
            image = Image.open(image_path)
            predictions = (
                self.model.predict(source=image, conf=conf_threshold, iou=iou_threshold)[0]
                .boxes.xyxy.cpu()
                .int()
                .tolist()
            )

            return self._filter_boxes(predictions, image.size)
        except Exception as e:
            logger.error(f"图标检测失败: {e}")
            return []


class IconDetectTool:
    """图标检测&描述工具

    提供完整的图标检测和描述功能，包括模型加载、图片处理和结果生成。
    """

    CAPTION_PROMPT_TEMPLATE = """You are a GUI automation assistant. This is an icon from a {device_type} interface.
Please describe the visual characteristics of this icon focusing on:
1. Primary shape (e.g., square, circle, custom shape)
2. Dominant colors
3. Key visual elements inside (symbols, text, patterns)
Requirements:
- Provide a concise, single-sentence description
- Emphasize the most distinctive features
- Order features by importance
- Focus on visual elements that would help identify this icon"""

    def __init__(self, llm: BaseLLM):
        """初始化图标检测工具

        Args:
            llm: 语言模型实例，用于生成图标描述
        """
        self.llm = llm
        self.icon_detector = IconDetector()

    def detect(self, image_path: Union[str, Path]) -> Coordinates:
        """检测图片中的图标位置

        Args:
            image_path: 图片路径
            split: 是否使用分割检测策略

        Returns:
            图标坐标列表
        """
        try:
            return self.icon_detector.detect(image_path)
        except Exception as e:
            logger.error(f"图标检测失败: {str(e)}")
            return []

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"图标{retry_state.args[-1]+1}描述生成失败，第{retry_state.attempt_number}次重试: {str(retry_state.outcome.exception())}"
        ),
    )
    async def _process_single_icon(self, image: Image.Image, box: List[int], prompt: str, idx: int) -> Tuple[int, str]:
        """处理单个图标的异步函数"""
        height = image.size[1]
        icon_img = image.crop(box)
        icon_width, icon_height = icon_img.size

        # 过滤过大的图标
        if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * image.size[0] * height:
            return idx, "None"

        description = await self.llm.aask(
            prompt,
            system_msgs="You are a helpful assistant that describes icons.",
            images=[encode_image(icon_img)],
            stream=False,
        )
        return idx, description

    async def caption(
        self, image_path: Union[str, Path], coordinates: Coordinates, platform: str = "Android"
    ) -> Dict[int, str]:
        """为检测到的图标生成描述
        Args:
            image_path: 图片路径
            coordinates: 图标坐标列表
            platform: 操作系统类型, 默认是Android, 也可以是PC
        Returns:
            Dict[int, str]: 图标索引到描述的映射
        """
        if not coordinates:
            return {}

        try:
            image = Image.open(image_path)
            prompt = self._get_caption_prompt(platform)

            # 创建信号量限制并发
            sem = asyncio.Semaphore(8)

            async def process_with_semaphore(idx: int, box: List[int]) -> Tuple[int, str]:
                async with sem:
                    return await self._process_single_icon(image, box, prompt, idx)

            # 并发处理所有图标
            tasks = [process_with_semaphore(i, box) for i, box in enumerate(coordinates)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 过滤掉异常结果，只保留成功的结果
            descriptions = {}
            for i, result in enumerate(results):
                if not isinstance(result, Exception) and isinstance(result, tuple):
                    idx, desc = result
                    descriptions[idx + 1] = desc
                else:
                    logger.warning(f"图标{i+1}描述生成失败: {result}")

            return descriptions

        except Exception as e:
            logger.error(f"生成图标描述失败: {e}")
            return {}

    def _get_caption_prompt(self, platform: str) -> str:
        """生成图标描述提示语"""
        device_type = "mobile phone" if platform == "Android" else "PC"
        return self.CAPTION_PROMPT_TEMPLATE.format(device_type=device_type)


def detect_icons(image_path: Union[str, Path], llm: BaseLLM = None) -> Coordinates:
    """图标检测快捷函数

    Args:
        image_path: 图片路径
        split: 是否使用分割检测策略

    Returns:
        图标坐标列表
    """
    detector = IconDetectTool(llm)
    return detector.detect(image_path)


async def caption_icons(
    image_path: Union[str, Path], coordinates: Coordinates, llm: BaseLLM = None, platform: str = "Android"
) -> Dict[int, str]:
    """图标描述快捷函数

    Args:
        image_path: 图片路径
        coordinates: 图标坐标列表
        platform: 操作系统类型, 默认是Android, 也可以是PC

    Returns:
        图标描述字典
    """
    detector = IconDetectTool(llm)
    return await detector.caption(image_path, coordinates, platform)


if __name__ == "__main__":
    from metagpt.config2 import Config

    async def main():
        image_path = str(TEST_DATA_PATH / "screenshots" / "chrome.jpg")
        llm_config = Config.default()
        llm = LLM(llm_config.llm)

        # 实例化 IconDetectTool 类
        detector = IconDetectTool(llm)

        # 检测
        coordinates = detector.detect(image_path)
        print("检测到的图标坐标:", coordinates, len(coordinates))

        # 生成描述
        descriptions = await detector.caption(image_path, coordinates)
        print("图标描述:", descriptions)

    asyncio.run(main())
