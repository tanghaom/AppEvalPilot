from typing import List, Union

import numpy as np
import pandas as pd


def read_excel_scores(excel_path: str) -> tuple[list, list, list]:
    """
    从Excel表格中读取评分数据

    参数:
        excel_path: Excel文件路径

    返回:
        tuple: 包含三个列表 (agent_eval_score, human_eval_score, human_score)
    """

    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        # 获取三列数据并转换为列表
        agent_avail_score = df["agent_avail_score"].tolist()
        agent_avail_score_test = df["agent_avail_score_test(去除不可靠用例)"].tolist()
        human_score = df["human_score"].tolist()

        return agent_avail_score, agent_avail_score_test, human_score

    except Exception as e:
        raise Exception(f"读取Excel文件失败: {str(e)}")


def calculate_correlation(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> float:
    """
    计算两个变量之间的皮尔逊相关系数

    参数:
        x: 第一个变量的数值列表或numpy数组
        y: 第二个变量的数值列表或numpy数组

    返回:
        float: 相关系数，范围在[0, 1]之间
        - 1表示完全相关
        - 0表示无相关性
    """
    # 转换为numpy数组以便计算
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # 检查输入数据的有效性
    if len(x) != len(y):
        raise ValueError("输入的两个变量长度必须相同")

    if len(x) < 2:
        raise ValueError("输入数据至少需要两个样本点")

    # 计算相关系数
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    # 避免除以零
    if denominator == 0:
        return 0

    correlation = numerator / denominator
    # 将相关系数归一化到0~1之间
    normalized_correlation = (correlation + 1) / 2
    return normalized_correlation


def calculate_spearman_correlation(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> float:
    """
    计算两个变量之间的斯皮尔曼等级相关系数

    参数:
        x: 第一个变量的数值列表或numpy数组
        y: 第二个变量的数值列表或numpy数组

    返回:
        float: 相关系数，范围在[0, 1]之间
        - 1表示完全相关
        - 0表示无相关性
    """
    # 转换为numpy数组
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # 检查输入数据的有效性
    if len(x) != len(y):
        raise ValueError("输入的两个变量长度必须相同")

    if len(x) < 2:
        raise ValueError("输入数据至少需要两个样本点")

    # 计算排名
    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))

    # 处理相同值的情况
    def adjust_ranks(ranks):
        unique_values, value_counts = np.unique(ranks, return_counts=True)
        for value, count in zip(unique_values, value_counts):
            if count > 1:
                mask = ranks == value
                ranks[mask] = np.mean(np.where(mask)[0])
        return ranks

    x_ranks = adjust_ranks(x_ranks)
    y_ranks = adjust_ranks(y_ranks)

    # 计算斯皮尔曼相关系数
    n = len(x)
    numerator = 6 * np.sum((x_ranks - y_ranks) ** 2)
    denominator = n * (n**2 - 1)

    # 避免除以零
    if denominator == 0:
        return 0

    spearman_corr = 1 - (numerator / denominator)
    # 将相关系数归一化到0~1之间
    normalized_correlation = (spearman_corr + 1) / 2
    return normalized_correlation


# 使用示例
if __name__ == "__main__":
    # 从Excel文件读取数据
    excel_path = "data/自动测试用例.xlsx"
    agent_avail_score, agent_avail_score_test, human_score = read_excel_scores(excel_path)

    # 计算相关系数
    pearson_corr = calculate_correlation(agent_avail_score_test, human_score)
    spearman_corr = calculate_spearman_correlation(agent_avail_score_test, human_score)

    print(f"皮尔逊相关系数: {pearson_corr:.4f}")
    print(f"斯皮尔曼相关系数: {spearman_corr:.4f}")
