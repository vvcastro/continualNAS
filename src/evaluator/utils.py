from typing import List


def metric_smoothing(values: List[float], wsize: int) -> List[float]:
    """
    Smooth the metric values using a moving average.

    Args:
    values (List[float]): List of metric values to smooth.
    wsize (int): The window size for the moving average.

    Returns:
    List[float]: Smoothed metric values.
    """
    cumsum = [0] + [sum(values[: i + 1]) for i in range(len(values))]
    smoothed = [
        (cumsum[i + wsize] - cumsum[i]) / wsize for i in range(len(values) - wsize + 1)
    ]
    return smoothed
