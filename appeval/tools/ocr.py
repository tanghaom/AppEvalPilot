#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/11
@Author  : tanghaoming
@File    : ocr.py
@Desc    : OCR工具类，提供OCR文本识别功能
"""

from typing import Generator, List, Tuple, Union

import cv2
import numpy as np
from metagpt.const import TEST_DATA_PATH
from metagpt.logs import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class OCRTool:
    def __init__(self):
        """初始化OCR工具类"""
        # 初始化文本检测和识别模型
        self.detection_model = pipeline(Tasks.ocr_detection, model="iic/cv_resnet18_ocr-detection-db-line-level_damo")
        self.recognition_model = pipeline(
            Tasks.ocr_recognition, model="iic/cv_convnextTiny_ocr-recognition-document_damo"
        )

    @staticmethod
    def _order_point(coor: np.ndarray) -> np.ndarray:
        """整理坐标点顺序，确保坐标点按照顺时针方向排列

        Args:
            coor: 4个坐标点的数组，形状为(4,2)或(8,)

        Returns:
            sort_points: 整理后的坐标点，形状为(4,2)
        """
        arr = coor.reshape([4, 2])
        centroid = np.mean(arr, axis=0)
        theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
        sort_points = arr[np.argsort(theta)]

        # 确保第一个点在质心左侧
        if sort_points[0][0] > centroid[0]:
            sort_points = np.roll(sort_points, -1, axis=0)

        return sort_points.astype("float32")

    @staticmethod
    def _crop_image(img: np.ndarray, position: np.ndarray) -> np.ndarray:
        """裁剪并矫正倾斜的文本区域

        Args:
            img: 原始图片
            position: 四个顶点的坐标，形状为(4,2)

        Returns:
            np.ndarray: 裁剪并矫正后的图片
        """

        def get_width_height(pts):
            """计算矫正后图片的宽度和高度"""
            width = np.linalg.norm(pts[1] - pts[0])
            height = np.linalg.norm(pts[2] - pts[0])
            return int(width), int(height)

        # 确保输入是numpy数组
        pts = np.float32(position)

        # 对坐标点按照x坐标排序
        x_sorted = pts[pts[:, 0].argsort()]

        # 分别获取左侧和右侧的两个点
        left = x_sorted[:2]
        right = x_sorted[2:]

        # 按y坐标排序左右两组点
        left = left[left[:, 1].argsort()]
        right = right[right[:, 1].argsort()]

        # 构建源点和目标点
        src_pts = np.float32([left[0], right[0], left[1], right[1]])
        width, height = get_width_height(src_pts)

        dst_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

        # 计算透视变换矩阵并应用
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(img, matrix, (width, height))

        return result

    def _read_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        """加载图像数据

        Args:
            image_input: 图像输入，可以是图片文件路径或已加载的图像数组

        Returns:
            np.ndarray: 加载后的图像数组

        Raises:
            ValueError: 当输入类型不正确时抛出
        """
        if isinstance(image_input, str):
            return cv2.imdecode(np.fromfile(image_input, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("image_input 必须是文件路径或 numpy 数组")

    def _ocr_on_image(self, image: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        """OCR核心处理逻辑

        Args:
            image: 图像数组

        Returns:
            texts: 识别出的文本列表
            coordinates: 文本框坐标列表 [[x1,y1,x2,y2],...]
        """
        text_data = []
        coordinates = []

        try:
            # 检测文本区域
            det_result = self.detection_model(image)
            polygons = det_result.get("polygons") if isinstance(det_result, dict) else det_result

            # 处理每个检测到的文本区域
            for polygon in polygons:
                try:
                    # 对文本区域进行透视变换
                    pts = self._order_point(polygon)
                    text_region = self._crop_image(image, pts)

                    # 识别文本
                    text = self.recognition_model(text_region)["text"][0]

                    # 提取边界框坐标
                    box = pts.reshape(-1).astype(int).tolist()
                    bbox = [box[0], box[1], box[4], box[5]]  # 左上和右下角点坐标

                    text_data.append(text)
                    coordinates.append(bbox)

                except Exception as e:
                    logger.warning(f"文本区域处理失败: {e}")
                    continue

            return text_data, coordinates

        except Exception as e:
            logger.warning(f"OCR处理失败: {e}")
            return [], []

    def ocr(self, image_input: Union[str, np.ndarray], split: bool = False) -> Tuple[List[str], List[List[int]]]:
        """执行OCR识别

        Args:
            image_input: 图片路径或者图像数组
            split: 是否分割处理,默认False

        Returns:
            texts: 识别出的文本列表
            coordinates: 文本框坐标列表 [[x1,y1,x2,y2],...]
        """
        image_full = self._read_image(image_input)
        if not split:
            return self._ocr_on_image(image_full)
        else:
            return self._split_ocr(image_full)

    def _get_split_regions(self, height: int, width: int) -> Generator[Tuple[int, int, int, int], None, None]:
        """生成图像分割区域，将图像分为2x2的网格

        Args:
            height: 图像高度
            width: 图像宽度

        Yields:
            Tuple[int, int, int, int]: 子区域的坐标 (x1, y1, x2, y2)，包含少量重叠边界
        """
        # 计算网格大小和重叠边界
        grid_w = width // 2
        grid_h = height // 2
        overlap = int(height * 0.0025)  # 添加0.25%的重叠区域

        # 生成2x2网格的坐标
        for row in range(2):
            for col in range(2):
                x1 = max(0, col * grid_w - overlap)
                y1 = max(0, row * grid_h - overlap)
                x2 = min(width, (col + 1) * grid_w + overlap)
                y2 = min(height, (row + 1) * grid_h + overlap)
                yield (x1, y1, x2, y2)

    def _split_ocr(self, image: np.ndarray) -> Tuple[List[str], List[List[int]]]:
        """将图片分割成多个子区域进行OCR处理

        Args:
            image: 输入图像数组

        Returns:
            Tuple[List[str], List[List[int]]]:
                - 识别出的文本列表
                - 文本框坐标列表
        """
        try:
            height, width = image.shape[:2]
            texts_all = []
            coords_all = []

            # 处理每个子区域
            for x1, y1, x2, y2 in self._get_split_regions(height, width):
                # 提取并处理子区域
                sub_img = image[y1:y2, x1:x2]
                sub_texts, sub_coords = self._ocr_on_image(sub_img)

                # 调整坐标到原图位置
                for coord in sub_coords:
                    coord[0] += x1  # 左上角x
                    coord[1] += y1  # 左上角y
                    coord[2] += x1  # 右下角x
                    coord[3] += y1  # 右下角y

                texts_all.extend(sub_texts)
                coords_all.extend(sub_coords)

            # 合并重叠区域的检测结果
            return self._merge_boxes_and_texts(texts_all, coords_all)

        except Exception as e:
            logger.error(f"分割OCR处理失败: {e}")
            return [], []

    def _merge_boxes_and_texts(
        self, texts: List[str], boxes: List[List[int]], iou_threshold: float = 0
    ) -> Tuple[List[str], List[List[int]]]:
        """合并重叠的文本框及其对应的文本

        Args:
            texts: 文本列表
            boxes: 边界框坐标列表，每个坐标格式为[x1,y1,x2,y2]
            iou_threshold: 判定重叠的IOU阈值，默认为0

        Returns:
            Tuple[List[str], List[List[int]]]:
                - 合并后的文本列表
                - 合并后的边界框坐标列表
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

            # 初始化合并框的范围
            x_min, y_min, x_max, y_max = box_i
            merged_text = texts[i]
            overlapping_indices = [i]

            # 查找所有与当前框重叠的框
            for j, box_j in enumerate(boxes):
                if i != j and not used[j] and self._bbox_iou(box_i, box_j) > iou_threshold:
                    overlapping_indices.append(j)

            # 按垂直位置排序，确保文本顺序正确
            overlapping_indices.sort(key=lambda idx: (boxes[idx][1] + boxes[idx][3]) / 2)

            # 合并所有重叠的框和文本
            for idx in overlapping_indices[1:]:  # 跳过第一个(已经处理过)
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
        """计算两个框的IOU

        Args:
            boxA: 框A的坐标 [x1,y1,x2,y2]
            boxB: 框B的坐标 [x1,y1,x2,y2]

        Returns:
            iou: IOU值
        """
        # 计算交集区域
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算两个框的面积
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # 计算IOU
        return inter_area / float(boxA_area + boxB_area - inter_area)


# 导出函数
def ocr_recognize(image_path: str, split: bool = False) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """OCR文本识别接口

    Args:
        image_path: 图片路径
        split: 是否分割处理,默认False

    Returns:
        texts: 识别出的文本列表
        coordinates: 文本框坐标列表 [(x1,y1,x2,y2),...]
    """
    ocr_tool = OCRTool()
    return ocr_tool.ocr(image_path, split)


if __name__ == "__main__":

    def main():
        # 测试图片路径
        image_path = str(TEST_DATA_PATH / "screenshots" / "chrome.jpg")

        # 实例化 OCRTool 类
        ocr_tool = OCRTool()

        # 测试普通OCR识别
        print("正在进行普通OCR识别...")
        texts, coordinates = ocr_tool.ocr(image_path)
        print("识别到的文本:", texts)
        print("文本坐标:", coordinates)

        # 测试分割OCR识别
        print("\n正在进行分割OCR识别...")
        split_texts, split_coordinates = ocr_tool.ocr(image_path, split=True)
        print("分割识别到的文本:", split_texts)
        print("分割文本坐标:", split_coordinates)

        # 测试快捷函数
        print("\n测试快捷函数...")
        quick_texts, quick_coordinates = ocr_recognize(image_path)
        print("快捷函数识别结果:", quick_texts)
        print("快捷函数坐标:", quick_coordinates)

    main()
