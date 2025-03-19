#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : ocr.py
@Desc    : OCR tool class, provides OCR text recognition functionality
"""

from typing import Generator, List, Tuple, Union

import cv2
import numpy as np
from metagpt.logs import logger

try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    _has_modelscope = True
except ImportError:
    _has_modelscope = False
    logger.warning("Warning: modelscope package is not installed, OCR function is unavailable.")
    logger.warning("Please use 'pip install appeval[ultra]' to install the required dependencies.")


class OCRTool:
    def __init__(self):
        """Initialize OCR tool class"""
        if not _has_modelscope:
            logger.warning("Warning: modelscope package is not installed, OCR function is unavailable.")
            logger.warning("Please use 'pip install appeval[ultra]' to install the required dependencies.")
            self.detection_model = None
            self.recognition_model = None
            return

        # Initialize text detection and recognition models
        self.detection_model = pipeline(Tasks.ocr_detection, model="iic/cv_resnet18_ocr-detection-db-line-level_damo")
        self.recognition_model = pipeline(
            Tasks.ocr_recognition, model="iic/cv_convnextTiny_ocr-recognition-document_damo"
        )

    @staticmethod
    def _order_point(coor: np.ndarray) -> np.ndarray:
        """Arrange coordinate points order, ensure points are arranged clockwise

        Args:
            coor: Array of 4 coordinate points, shape (4,2) or (8,)

        Returns:
            sort_points: Arranged coordinate points, shape (4,2)
        """
        arr = coor.reshape([4, 2])
        centroid = np.mean(arr, axis=0)
        theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
        sort_points = arr[np.argsort(theta)]

        # Ensure the first point is on the left side of the centroid
        if sort_points[0][0] > centroid[0]:
            sort_points = np.roll(sort_points, -1, axis=0)

        return sort_points.astype("float32")

    @staticmethod
    def _crop_image(img: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Crop and rectify skewed text regions

        Args:
            img: Original image
            position: Coordinates of four vertices, shape (4,2)

        Returns:
            np.ndarray: Cropped and rectified image
        """

        def get_width_height(pts):
            """Calculate width and height of rectified image"""
            width = np.linalg.norm(pts[1] - pts[0])
            height = np.linalg.norm(pts[2] - pts[0])
            return int(width), int(height)

        # Ensure input is numpy array
        pts = np.float32(position)

        # Sort points by x coordinate
        x_sorted = pts[pts[:, 0].argsort()]

        # Get the left and right two points
        left = x_sorted[:2]
        right = x_sorted[2:]

        # Sort left and right groups of points by y coordinate
        left = left[left[:, 1].argsort()]
        right = right[right[:, 1].argsort()]

        # Build source and destination points
        src_pts = np.float32([left[0], right[0], left[1], right[1]])
        width, height = get_width_height(src_pts)

        dst_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

        # Calculate perspective transformation matrix and apply
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(img, matrix, (width, height))

        return result

    def _read_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """Load image data

        Args:
            image_input: Image input, can be image file path or loaded image array

        Returns:
            np.ndarray: Loaded image array

        Raises:
            ValueError: When input type is incorrect
        """
        if isinstance(image_input, str):
            return cv2.imdecode(np.fromfile(image_input, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("image_input must be a file path or numpy array")

    def _ocr_on_image(self, image: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        """Core OCR processing logic

        Args:
            image: Image array

        Returns:
            texts: List of recognized texts
            coordinates: List of text box coordinates [[x1,y1,x2,y2],...]
        """
        text_data = []
        coordinates = []

        try:
            # Detect text regions
            det_result = self.detection_model(image)
            polygons = det_result.get("polygons") if isinstance(det_result, dict) else det_result

            # Process each detected text region
            for polygon in polygons:
                try:
                    # Apply perspective transform to text region
                    pts = self._order_point(polygon)
                    text_region = self._crop_image(image, pts)

                    # Recognize text
                    text = self.recognition_model(text_region)["text"][0]

                    # Extract bounding box coordinates
                    box = pts.reshape(-1).astype(int).tolist()
                    bbox = [box[0], box[1], box[4], box[5]]  # Top-left and bottom-right coordinates

                    text_data.append(text)
                    coordinates.append(bbox)

                except Exception as e:
                    logger.warning(f"Text region processing failed: {e}")
                    continue

            return text_data, coordinates

        except Exception as e:
            logger.warning(f"OCR processing failed: {e}")
            return [], []

    def ocr(self, image_input: Union[str, np.ndarray], split: bool = False) -> Tuple[List[str], List[List[int]]]:
        """Perform OCR recognition

        Args:
            image_input: Image path or image array
            split: Whether to split process, default False

        Returns:
            texts: List of recognized texts
            coordinates: List of text box coordinates [[x1,y1,x2,y2],...]
        """
        if not _has_modelscope:
            logger.warning("Warning: modelscope package is not installed, OCR function is unavailable.")
            logger.warning("Please use 'pip install appeval[ultra]' to install the required dependencies.")
            return [], []

        image_full = self._read_image(image_input)
        if not split:
            return self._ocr_on_image(image_full)
        else:
            return self._split_ocr(image_full)

    def _get_split_regions(self, height: int, width: int) -> Generator[Tuple[int, int, int, int], None, None]:
        """Generate image split regions, divide image into 2x2 grid

        Args:
            height: Image height
            width: Image width

        Yields:
            Tuple[int, int, int, int]: Coordinates of sub-region (x1, y1, x2, y2), including small overlap
        """
        # Calculate grid size and overlap
        grid_w = width // 2
        grid_h = height // 2
        overlap = int(height * 0.0025)  # Add 0.25% overlap region

        # Generate coordinates for 2x2 grid
        for row in range(2):
            for col in range(2):
                x1 = max(0, col * grid_w - overlap)
                y1 = max(0, row * grid_h - overlap)
                x2 = min(width, (col + 1) * grid_w + overlap)
                y2 = min(height, (row + 1) * grid_h + overlap)
                yield (x1, y1, x2, y2)

    def _split_ocr(self, image: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        """Split image into multiple sub-regions for OCR processing

        Args:
            image: Input image array

        Returns:
            Tuple[List[str], List[List[int]]]:
                - List of recognized texts
                - List of text box coordinates
        """
        try:
            height, width = image.shape[:2]
            texts_all = []
            coords_all = []

            # Process each sub-region
            for x1, y1, x2, y2 in self._get_split_regions(height, width):
                # Extract and process sub-region
                sub_img = image[y1:y2, x1:x2]
                sub_texts, sub_coords = self._ocr_on_image(sub_img)

                # Adjust coordinates to original image position
                for coord in sub_coords:
                    coord[0] += x1  # Top-left x
                    coord[1] += y1  # Top-left y
                    coord[2] += x1  # Bottom-right x
                    coord[3] += y1  # Bottom-right y

                texts_all.extend(sub_texts)
                coords_all.extend(sub_coords)

            # Merge detection results from overlapping regions
            return self._merge_boxes_and_texts(texts_all, coords_all)

        except Exception as e:
            logger.error(f"Split OCR processing failed: {e}")
            return [], []

    def _merge_boxes_and_texts(
        self, texts: List[str], boxes: List[List[int]], iou_threshold: float = 0
    ) -> Tuple[List[str], List[List[int]]]:
        """Merge overlapping text boxes and their corresponding texts

        Args:
            texts: List of texts
            boxes: List of bounding box coordinates, each in format [x1,y1,x2,y2]
            iou_threshold: IOU threshold for determining overlap, default 0

        Returns:
            Tuple[List[str], List[List[int]]]:
                - List of merged texts
                - List of merged bounding box coordinates
        """
        if not boxes:
            return [], []

        boxes = np.array(boxes)
        merged_boxes = []
        merged_texts = []
        used = np.zeros(len(boxes), dtype=bool)

        for i, box_i in enumerate(boxes):
            if used[i]:
                continue

            # Initialize merged box range
            x_min, y_min, x_max, y_max = box_i
            merged_text = texts[i]
            overlapping_indices = [i]

            # Find all boxes that overlap with the current box
            for j, box_j in enumerate(boxes):
                if i != j and not used[j] and self._bbox_iou(box_i, box_j) > iou_threshold:
                    overlapping_indices.append(j)

            # Sort by vertical position to ensure correct text order
            overlapping_indices.sort(key=lambda idx: (boxes[idx][1] + boxes[idx][3]) / 2)

            # Merge all overlapping boxes and texts
            for idx in overlapping_indices[1:]:  # Skip the first (already processed)
                box_j = boxes[idx]
                x_min = min(x_min, box_j[0])
                y_min = min(y_min, box_j[1])
                x_max = max(x_max, box_j[2])
                y_max = max(y_max, box_j[3])
                merged_text += texts[idx]
                used[idx] = True

            merged_boxes.append([x_min, y_min, x_max, y_max])
            merged_texts.append(merged_text)
            used[i] = True

        return merged_texts, merged_boxes

    def _bbox_iou(self, boxA: List[int], boxB: List[int]) -> float:
        """Calculate IOU between two boxes

        Args:
            boxA: Box A coordinates [x1,y1,x2,y2]
            boxB: Box B coordinates [x1,y1,x2,y2]

        Returns:
            float: IOU value
        """
        # Calculate intersection area
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate areas of two boxes
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Calculate IOU
        return inter_area / float(boxA_area + boxB_area - inter_area)


# Export function
def ocr_recognize(image_path: str, split: bool = False) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """OCR text recognition interface

    Args:
        image_path: Image path
        split: Whether to split process, default False

    Returns:
        texts: List of recognized texts
        coordinates: List of text box coordinates [(x1,y1,x2,y2),...]
    """
    ocr_tool = OCRTool()
    return ocr_tool.ocr(image_path, split)
