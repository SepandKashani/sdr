import numpy as np

"""
Data transformation helpers.
"""


def int2bit(x: np.ndarray, N_bit: int) -> np.ndarray:
    """
    Convert unsigned integers to binary representation.

    Parameters
    ----------
    x: np.ndarray[int/uint]
    N_bit: int
        Number of bits to use to represent `x`.

    Returns
    -------
    y: np.ndarray[bool]
        (*x.shape, N_bit) binary representation, with most-significant bit on the left.
    """
    assert np.all(x >= 0)
    x = x.astype(np.dtype(">u8"))  # 64-bit unsigned integers (big-endian)

    N_bit_min = max(1, np.ceil(np.log2(x.max() + 1)))
    assert N_bit_min <= N_bit <= 8 * x.dtype.itemsize

    mask = (2 ** np.arange(N_bit, dtype=x.dtype))[::-1]
    y = (x.reshape((*x.shape, 1)) & mask) > 0
    return y


def bit2int(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """
    Convert binary digits to unsigned integers.

    Parameters
    ----------
    x: np.ndarray[bool]
        (..., N_bit, ...) bits. Big-endian representation is assumed.
    axis: int
        Dimension along which binary numbers are stored.
    keepdims: bool
        If `True`, output has same number of dimensions as `x`.

    Returns
    -------
    y: np.ndarray[uint]
        (..., 1, ...) or (..., ...) unsigned integers.
    """
    assert -x.ndim <= axis < x.ndim

    dtype = np.dtype(">u8")
    scale = 2 ** np.arange(x.shape[axis], dtype=dtype)[::-1]

    sh = [1] * x.ndim
    sh[axis] = -1

    y = np.sum(x.astype(dtype) * scale.reshape(sh), axis=axis, keepdims=keepdims)
    return y
