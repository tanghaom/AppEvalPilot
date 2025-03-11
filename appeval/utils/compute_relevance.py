from typing import List, Union

import numpy as np
import pandas as pd


def read_excel_scores(excel_path: str) -> tuple[list, list, list]:
    """
    Read scoring data from Excel file

    Args:
        excel_path: Path to Excel file

    Returns:
        tuple: Contains three lists (agent_eval_score, human_eval_score, human_score)
    """

    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        # Get three columns of data and convert to lists
        agent_avail_score = df["agent_avail_score"].tolist()
        agent_avail_score_test = df["agent_avail_score_test(remove unreliable cases)"].tolist()
        human_score = df["human_score"].tolist()

        return agent_avail_score, agent_avail_score_test, human_score

    except Exception as e:
        raise Exception(f"Failed to read Excel file: {str(e)}")


def calculate_correlation(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Pearson correlation coefficient between two variables

    Args:
        x: List or numpy array of values for the first variable
        y: List or numpy array of values for the second variable

    Returns:
        float: Correlation coefficient, range [0, 1]
        - 1 indicates perfect correlation
        - 0 indicates no correlation
    """
    # Convert to numpy arrays for calculation
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Check validity of input data
    if len(x) != len(y):
        raise ValueError("The two input variables must have the same length")

    if len(x) < 2:
        raise ValueError("Input data requires at least two sample points")

    # Calculate correlation coefficient
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    # Avoid division by zero
    if denominator == 0:
        return 0

    correlation = numerator / denominator
    # Normalize correlation coefficient to range 0~1
    normalized_correlation = (correlation + 1) / 2
    return normalized_correlation


def calculate_spearman_correlation(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]) -> float:
    """
    Calculate Spearman rank correlation coefficient between two variables

    Args:
        x: List or numpy array of values for the first variable
        y: List or numpy array of values for the second variable

    Returns:
        float: Correlation coefficient, range [0, 1]
        - 1 indicates perfect correlation
        - 0 indicates no correlation
    """
    # Convert to numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Check validity of input data
    if len(x) != len(y):
        raise ValueError("The two input variables must have the same length")

    if len(x) < 2:
        raise ValueError("Input data requires at least two sample points")

    # Calculate ranks
    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))

    # Handle ties
    def adjust_ranks(ranks: np.ndarray) -> np.ndarray:
        unique_values, value_counts = np.unique(ranks, return_counts=True)
        for value, count in zip(unique_values, value_counts):
            if count > 1:
                mask = ranks == value
                ranks[mask] = np.mean(np.where(mask)[0])
        return ranks

    x_ranks = adjust_ranks(x_ranks)
    y_ranks = adjust_ranks(y_ranks)

    # Calculate Spearman correlation coefficient
    n = len(x)
    numerator = 6 * np.sum((x_ranks - y_ranks) ** 2)
    denominator = n * (n**2 - 1)

    # Avoid division by zero
    if denominator == 0:
        return 0

    spearman_corr = 1 - (numerator / denominator)
    # Normalize correlation coefficient to range 0~1
    normalized_correlation = (spearman_corr + 1) / 2
    return normalized_correlation
