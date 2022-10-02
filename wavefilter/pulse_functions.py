import numpy as np
import numpy.typing as npt


def simple(x: npt.ArrayLike, tau: float = 10, x0: int = 0) -> npt.NDArray[float]:
    xx = x - x0
    y = xx * np.exp(-xx / tau) / tau / tau
    y[x < x0] = 0
    return y


def bimodal(x: npt.ArrayLike, tau1: float = 10, tau2: float = 30, x0: int = 0) -> npt.NDArray[float]:
    y1 = simple(x, tau1, x0)
    y2 = simple(x, tau2, x0)
    return y1 - y2


def white_noise(x: npt.ArrayLike, std: float) -> npt.NDArray[float]:
    if isinstance(x, np.ndarray):
        x = len(x)
    return np.random.normal(0, std, x)
