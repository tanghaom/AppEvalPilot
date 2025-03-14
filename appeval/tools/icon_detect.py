#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : icon_detect.py
@Desc    : Icon detection and description tool class
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Union

import requests
from metagpt.const import DATA_PATH
from metagpt.logs import logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.utils.common import encode_image
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

try:
    from ultralytics import YOLO

    _has_ultralytics = True
except ImportError:
    YOLO = None
    _has_ultralytics = False

# Type aliases
BoundingBox = List[int]  # [x1, y1, x2, y2]
Coordinates = List[BoundingBox]


def _check_ultralytics() -> bool:
    """Check if ultralytics package is installed"""
    if not _has_ultralytics:
        logger.warning("Warning: ultralytics package is not installed, icon detection function is unavailable.")
        logger.warning("Please use 'pip install appeval[ultra]' to install the required dependencies.")
        return False
    return True


class IconDetector:
    """Icon detection tool class"""

    MODEL_URL = "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt"
    MODEL_PATH = DATA_PATH / "omniparser_icon_detect.pt"

    def __init__(self, model_path=None):
        if not _check_ultralytics():
            self.model = None
            return

        self.model = self._init_model()

    def _init_model(self) -> YOLO:
        """Initialize and return YOLO model"""
        if not self.MODEL_PATH.exists():
            self._download_model()
        return YOLO(str(self.MODEL_PATH))

    def _download_model(self):
        """Download model file"""
        logger.info("Downloading OmniParser icon detection model...")
        logger.info("If the original download fails, will automatically try from mirror sites")
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Main URL and backup mirror URL
        urls = [
            self.MODEL_URL,
            self.MODEL_URL.replace("https://huggingface.co", "https://hf-mirror.com"),
            "https://gitee.com/hf-models/OmniParser-v2.0/raw/main/icon_detect/model.pt",
        ]

        for url in urls:
            try:
                logger.info(f"Trying to download model from: {url}")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.MODEL_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logger.info("OmniParser model download completed")
                return  # Download successful, exit function
            except Exception as e:
                logger.warning(f"Model download from {url} failed: {e}")
                continue  # Try next URL

        logger.error("All download attempts failed")
        raise Exception("Failed to download model from all sources")

    @staticmethod
    def _calculate_area(box: BoundingBox) -> float:
        """Calculate bounding box area

        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]

        Returns:
            float: Area of the bounding box
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def _calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes

        Args:
            box1: First bounding box coordinates [x1, y1, x2, y2]
            box2: Second bounding box coordinates [x1, y1, x2, y2]

        Returns:
            float: IoU value of the two boxes
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
        """Filter and merge overlapping bounding boxes

        Args:
            boxes: List of bounding boxes
            image_size: Image dimensions (width, height)
            iou_threshold: IoU threshold
            area_threshold: Area threshold (ratio)

        Returns:
            List[List[int]]: Filtered list of bounding boxes
        """
        if not boxes:
            return []

        img_area = image_size[0] * image_size[1]
        max_area = img_area * area_threshold

        # Sort boxes by area from largest to smallest and filter out large boxes
        valid_boxes = [box for box in sorted(boxes, key=self._calculate_area) if self._calculate_area(box) <= max_area]

        filtered_boxes = []
        for box in valid_boxes:
            # If current box IoU with all retained boxes is less than threshold, retain
            if not any(self._calculate_iou(box, existing_box) > iou_threshold for existing_box in filtered_boxes):
                filtered_boxes.append(box)

        return filtered_boxes

    def detect(
        self, image_path: Union[str, Path], conf_threshold: float = 0.25, iou_threshold: float = 0.3
    ) -> Coordinates:
        if not _check_ultralytics():
            return []

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
            logger.error(f"Icon detection failed: {e}")
            return []


class IconDetectTool:
    """Icon Detection & Description Tool

    Provides complete icon detection and description functionality, including model loading,
    image processing and result generation.
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
        """Initialize icon detection tool

        Args:
            llm: Language model instance, used to generate icon description
        """
        self.llm = llm
        self.icon_detector = IconDetector()

    def detect(self, image_path: Union[str, Path]) -> Coordinates:
        """Detect icon locations in the image

        Args:
            image_path: Image path

        Returns:
            List[List[int]]: List of icon coordinates
        """
        try:
            return self.icon_detector.detect(image_path)
        except Exception as e:
            logger.error(f"Icon detection failed: {str(e)}")
            return []

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Icon {retry_state.args[-1]+1} description generation failed, attempt {retry_state.attempt_number}: {str(retry_state.outcome.exception())}"
        ),
    )
    async def _process_single_icon(self, image: Image.Image, box: List[int], prompt: str, idx: int) -> Tuple[int, str]:
        """Process single icon asynchronously

        Args:
            image: Image object - The full screenshot image where the icon is located.
            box: Icon coordinates - Bounding box [x1, y1, x2, y2] of the detected icon in the image.
            prompt: Icon description prompt - Prompt text used to guide the language model in describing the icon.
            idx: Icon index - Index of the icon in the list of detected icons, used for tracking and referencing.

        Returns:
            Tuple[int, str]: Icon index and description
        """
        height = image.size[1]
        icon_img = image.crop(box)
        icon_width, icon_height = icon_img.size

        # Filter out large icons
        if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * image.size[0] * height:
            return idx, "None"

        description = await self.llm.aask(
            prompt,
            system_msgs=["You are a helpful assistant that describes icons."],
            images=[encode_image(icon_img)],
            stream=False,
        )
        return idx, description

    async def caption(
        self, image_path: Union[str, Path], coordinates: Coordinates, platform: str = "Android"
    ) -> Dict[int, str]:
        """Generate descriptions for detected icons

        Args:
            image_path: Image path
            coordinates: List of icon coordinates
            platform: Operating system type, default is Android, can also be PC

        Returns:
            Dict[int, str]: Mapping from icon index to description
        """
        if not coordinates:
            return {}

        try:
            image = Image.open(image_path)
            prompt = self._get_caption_prompt(platform)

            # Create semaphore to limit concurrency
            sem = asyncio.Semaphore(8)

            async def process_with_semaphore(idx: int, box: List[int]) -> Tuple[int, str]:
                async with sem:
                    return await self._process_single_icon(image, box, prompt, idx)

            # Concurrent processing of all icons
            tasks = [process_with_semaphore(i, box) for i, box in enumerate(coordinates)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exception results, only keep successful results
            descriptions = {}
            for i, result in enumerate(results):
                if not isinstance(result, Exception) and isinstance(result, tuple):
                    idx, desc = result
                    descriptions[idx + 1] = desc
                else:
                    logger.warning(f"Icon {i+1} description generation failed: {result}")

            return descriptions

        except Exception as e:
            logger.error(f"Icon description generation failed: {e}")
            return {}

    def _get_caption_prompt(self, platform: str) -> str:
        """Generate icon description prompt

        Args:
            platform: Operating system type

        Returns:
            str: Icon description prompt
        """
        device_type = "mobile phone" if platform == "Android" else "PC"
        return self.CAPTION_PROMPT_TEMPLATE.format(device_type=device_type)


def detect_icons(image_path: Union[str, Path], llm: BaseLLM = None) -> Coordinates:
    """Icon detection shortcut function

    Args:
        image_path: Image path

    Returns:
        List[List[int]]: List of icon coordinates
    """
    detector = IconDetectTool(llm)
    return detector.detect(image_path)


async def caption_icons(
    image_path: Union[str, Path], coordinates: Coordinates, llm: BaseLLM = None, platform: str = "Android"
) -> Dict[int, str]:
    """Icon description shortcut function

    Args:
        image_path: Image path
        coordinates: List of icon coordinates
        platform: Operating system type, default is Android, can also be PC

    Returns:
        Dict[int, str]: Dictionary of icon descriptions
    """
    detector = IconDetectTool(llm)
    return await detector.caption(image_path, coordinates, platform)
