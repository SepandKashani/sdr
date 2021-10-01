import numpy as np

"""
Discrete to Continuous symbol mappers.

TX architecture:
    discrete symbols -> [mapper] -> continuous symbols -> [modulator] -> continuous signal

RX architecture:
    continuous signal -> [modulator] -> continuous symbols -> [mapper] -> discrete symbols
                         [    +    ]
                         [sym_sync ]
"""


def qamMap(M: int) -> np.ndarray:
    """
    M-ary Quadrature Amplitude Modulation codebook.

    Parameters
    ----------
    M: int
        Codebook cardinality.
        Must be of the form (2j)**2, j >= 1.

    Returns
    -------
    C: np.ndarray[complex]
        (M,) QAM codebook.
    """
    # Implement me
    pass


def pskMap(M: int) -> np.ndarray:
    """
    M-ary Phase-Shifted Key Modulation codebook.

    Parameters
    ----------
    M: int
        Codebook cardinality.

    Returns
    -------
    C: np.ndarray[float/complex]
        (M,) PSK codebook.
    """
    # Implement me
    pass


def encoder(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Map integers to symbols.

    Parameters
    ----------
    x: np.ndarray[int]
        (*sh_x,) values in {0, ..., M-1}.
    C: np.ndarray[float/complex]
        (M,) codebook.

    Returns
    -------
    y: np.ndarray[float/complex]
        (*sh_x,) codewords.
    """
    # Implement me
    pass


def decoder(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Map symbols to codeword-indices via minimum-distance decoding.

    Parameters
    ----------
    x: np.ndarray[float/complex]
        (*sh_x,) noisy symbols.
    C: np.ndarray[float/complex]
        (M,) codebook.

    Returns
    -------
    y: np.ndarray[int]
        (*sh_x,) values in {0, ..., M-1}.
    """
    # Implement me
    pass
