import abc
import math

import numpy as np
import numpy.fft as npf
import scipy.linalg as spl

"""
Discrete to Continuous symbol mappers.

TX architecture:
    discrete symbols -> [mapper] -> continuous symbols -> [modulator] -> continuous signal

RX architecture:
    continuous signal -> [modulator] -> continuous symbols -> [mapper] -> discrete symbols
                         [    +    ]
                         [sym_sync ]
"""


class Mapper:
    """
    Map discrete symbols to continuous symbols which can be modulated by an analog channel.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[discrete alphabet]

        Returns
        -------
        y: np.ndarray[continuous alphabet]
        """
        pass

    @abc.abstractmethod
    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[continuous alphabet]

        Returns
        -------
        y: np.ndarray[discrete alphabet]
        """
        pass


class PAM(Mapper):
    r"""
    M-ary Pulse Amplitude Modulation.

    A = {0, ..., M-1}
    C = {c_{k}, k \in A}
    c_{k} = (k - \frac{M - 1}{2}) * d
    """

    def __init__(self, M: int, d: int):
        """
        Parameters
        ----------
        M: int
            Codebook cardinality.
            Must be even-valued.
        d: float
            Inter-codeword spacing.
        """
        assert (M >= 2) and (M % 2 == 0)
        self._M = M

        assert d > 0
        self._d = d

        # Codeword dictionary
        self._C = (np.arange(M) - (M - 1) / 2) * d

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[A]

        Returns
        -------
        y: np.ndarray[C]
        """
        assert np.isin(x, np.arange(self._M)).all()
        return (y := self._C[x])

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[float/complex]

        Returns
        -------
        y: np.ndarray[A]
        """
        dist = np.abs(x.reshape((*x.shape, 1)) - self._C)
        return (y := np.argmin(dist, axis=-1))

    @property
    def codebook(self) -> np.ndarray:
        """
        Returns
        -------
        C: np.ndarray[float]
            (M,) codewords.
        """
        return self._C


class QAM(Mapper):
    r"""
    M-ary Quadrature Amplitude Modulation.

    A = {0, ..., M-1}
    C = {c_{k}, k \in A}
    c_{k} = a_{l} + j * b_{m} \in \bC, with k <--> (l, m) a bijection.
    a_{l} = (l - \frac{\sqrt{M} - 1}{2}) * d
    b_{m} = (m - \frac{\sqrt{M} - 1}{2}) * d
    """

    def __init__(self, M: int, d: int):
        """
        Parameters
        ----------
        M: int
            Codebook cardinality.
            Must be of the form (2j)**2, j >= 1.
        d: float
            Inter-codeword spacing (per dimension).
        """
        j = math.sqrt(M) / 2
        assert (j >= 1) and j.is_integer()
        self._M = M

        assert d > 0
        self._d = d

        # Codeword dictionary
        C_1d = PAM(math.sqrt(M), d).codebook
        self._C = np.reshape(C_1d.reshape((1, -1)) + 1j * C_1d.reshape((-1, 1)), -1)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[A]

        Returns
        -------
        y: np.ndarray[C]
        """
        assert np.isin(x, np.arange(self._M)).all()
        return (y := self._C[x])

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[float/complex]

        Returns
        -------
        y: np.ndarray[A]
        """
        dist = np.abs(x.reshape((*x.shape, 1)) - self._C)
        return (y := np.argmin(dist, axis=-1))


class PSK(Mapper):
    r"""
    M-ary Phase-Shifted Key Modulation.

    A = {0, ..., M-1}
    C = {c_{k}, k \in A}
    c_{k} = d * \exp(j 2 \pi k / M)
    """

    def __init__(self, M: int, d: int):
        """
        Parameters
        ----------
        M: int
            Codebook cardinality.
        d: float
            Codebook radius.
        """
        assert M >= 2
        self._M = M

        assert d > 0
        self._d = d

        # Codeword dictionary
        self._C = d * np.exp(1j * 2 * np.pi * np.arange(M) / M)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[A]

        Returns
        -------
        y: np.ndarray[C]
        """
        assert np.isin(x, np.arange(self._M)).all()
        return (y := self._C[x])

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[float/complex]

        Returns
        -------
        y: np.ndarray[A]
        """
        dist = np.abs(x.reshape((*x.shape, 1)) - self._C)
        return (y := np.argmin(dist, axis=-1))


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
    C = QAM(M=M, d=1)._C
    return C


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
    C = PSK(M=M, d=1)._C
    return C


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
    M = C.size
    assert np.isin(x, np.arange(M)).all()
    return C[x]


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
    dist = np.abs(x.reshape((*x.shape, 1)) - C)
    return (y := np.argmin(dist, axis=-1))
