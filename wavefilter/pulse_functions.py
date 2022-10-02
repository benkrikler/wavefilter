from typing import Any

import numpy as np


def simple(x: Any, tau: float = 10, x0: int = 0) -> Any:
    xx = x - x0
    y = xx * np.exp(-xx / tau) / tau / tau
    y[x < x0] = 0
    return y


def bimodal(x: Any, tau1: float = 10, tau2: float = 30, x0: int = 0) -> Any:
    y1 = simple(x, tau1, x0)
    y2 = simple(x, tau2, x0)
    return y1 - y2


def white_noise(x: Any, std: float) -> Any:
    if isinstance(x, np.ndarray):
        x = len(x)
    return np.random.normal(0, std, x)
