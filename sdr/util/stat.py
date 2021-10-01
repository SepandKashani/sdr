import numpy as np
import scipy.stats as ss

"""
Statistic routines.
"""


def error_rate(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the error rate between two arrays.

    Parameters
    ----------
    x: np.ndarray
    y: np.ndarray

    Returns
    -------
    r: float
        Ratio of number of elements in x/y that differ.
        x/y are assumed to have compatible shapes, i.e. broadcasting possible.
    """
    r = np.sum(~np.isclose(x, y)) / max(x.size, y.size)
    return r


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) between two arrays.

    Parameters
    ----------
    x: np.ndarray
    y: np.ndarray

    Returns
    -------
    err: float
        MSE = mean(|x-y|**2)
        x/y are assumed to have compatible shapes, i.e. broadcasting possible.
    """
    err = np.mean(np.abs(x - y) ** 2)
    return err


def qfunc(x: np.ndarray) -> np.ndarray:
    r"""
    Q function.

    Parameters
    ----------
    x: np.ndarray[float]

    Returns
    -------
    y: np.ndarray[float]
        Q(x) = P(X >= x), where X ~ N(0, 1).
    """
    y = 1 - ss.norm.cdf(x)
    return y
