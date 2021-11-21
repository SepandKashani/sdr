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


class OFDM(Mapper):
    """
    Orthogonal Frequency Division Multiplexing Modulation.

    As the notion of time is important in OFDM, encode/decode methods must specify the `axis`
    parameter.
    """

    def __init__(
        self,
        N_prefix: int,
        mask_ch: np.ndarray,
        x_train: np.ndarray,
    ):
        """
        Parameters
        ----------
        N_prefix: int
            Length of the circular prefix >= 0.
        mask_ch: np.ndarray[bool]
            (N_channel,) channel mask. Used to turn off individual channels. (I.e.: do not transmit
            data on inactive channels.)
        x_train: np.ndarray[continuous alphabet]
            (N_channel,) training symbols known to TX/RX. Used to estimate channel properties.
        """
        assert N_prefix >= 0
        self._N_prefix = N_prefix

        assert (mask_ch.ndim == 1) and (mask_ch.dtype.kind == "b")
        self._mask_ch = mask_ch.copy()
        self._N_channel = mask_ch.size
        self._N_active_ch = np.sum(mask_ch)
        assert self._N_active_ch > 0

        assert x_train.shape == (self._N_channel,)
        self._x_train = x_train.copy()

    def encode(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        OFDM-encode data stream(s).

        Parameters
        ----------
        x: np.ndarray[continuous alphabet]
            (..., N, ...) symbols. N must be divisible by the number of active channels.
        axis: int
            Dimension of `x` along which lie data streams.

        Returns
        -------
        y: np.ndarray[complex]
            (..., Q, ...) symbols, where Q is a multiple of (N_channel + N_prefix).
            Training symbols `x_train` are encoded at the start of `y`.
        """
        assert -x.ndim <= axis < x.ndim
        axis += 0 if (axis >= 0) else x.ndim

        sh_x, N = x.shape, x.shape[axis]
        assert N % self._N_active_ch == 0, "N must be divisible by the number of active channels."

        # reshape x to (..., -1, N_active_ch, ...)
        sh_xA = sh_x[:axis] + (-1, self._N_active_ch) + sh_x[(axis + 1) :]
        xA = x.reshape(sh_xA)

        # Add (active) training symbols. We need to do this in 2 steps due to the way np.concatenate
        # works:
        #
        # 1: reshape x_train to have same dimensionality as xA;
        sh_tr = [1] * xA.ndim
        sh_tr[axis + 1] = self._N_active_ch
        x_train = self._x_train[self._mask_ch].reshape(sh_tr)
        # 2: broadcast x_train to have same shape as xA, except at axis-th dimension where it
        #    should be 1.
        sh_tr = list(xA.shape)
        sh_tr[axis] = 1
        x_train = np.broadcast_to(x_train, sh_tr)
        xAT = np.concatenate([x_train, xA], axis=axis)

        # expand xA to (..., -1, N_channel, ...) by filling active channels only.
        sh_xB = sh_x[:axis] + (xAT.shape[axis], self._N_channel) + sh_x[(axis + 1) :]
        xB = np.zeros(sh_xB, dtype=x.dtype)
        idx = [slice(None)] * xB.ndim
        idx[axis + 1] = self._mask_ch
        xB[tuple(idx)] = xAT

        y = npf.ifft(xB, axis=axis + 1, norm="ortho")

        # insert circular prefix
        if self._N_prefix > 0:
            idx = [slice(None)] * xB.ndim
            idx[axis + 1] = slice(-self._N_prefix, None)
            y = np.concatenate([y[tuple(idx)], y], axis=axis + 1)

        # reshape y to (..., Q, ...)
        sh_y = list(x.shape)
        sh_y[axis] = -1
        y = y.reshape(sh_y)
        return y

    def decode(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        OFDM-decode data-stream(s).

        This method does not perform channel equalization: the responsibility is left to the user
        during post-processing.

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N, ...) symbols. N must be divisible by (N_channel + N_prefix).
        axis: int
            Dimension of `x` along which lie data streams.

        Returns
        -------
        y: np.ndarray[complex]
            (..., Q, N_active_ch, ...) symbols, where Q is the number of decoded OFDM
            (block-)symbols.
        """
        assert -x.ndim <= axis < x.ndim
        axis += 0 if (axis >= 0) else x.ndim

        sh_x, N = x.shape, x.shape[axis]
        assert (
            N % (self._N_channel + self._N_prefix) == 0
        ), "N must be divisible by (N_channel + N_prefix)."

        # reshape x to (..., -1, N_channel + N_prefix, ...)
        sh_xA = sh_x[:axis] + (-1, self._N_channel + self._N_prefix) + sh_x[(axis + 1) :]
        xA = x.reshape(sh_xA)

        # Drop circular prefix
        idx = [slice(None)] * xA.ndim
        idx[axis + 1] = slice(self._N_prefix, None)
        xB = xA[tuple(idx)]

        y = npf.fft(xB, axis=axis + 1, norm="ortho")

        # Drop symbols in inactive channels
        idx = [slice(None)] * xB.ndim
        idx[axis + 1] = self._mask_ch
        y = y[tuple(idx)]
        return y

    def channel(self, h: np.ndarray) -> np.ndarray:
        """
        Compute the frequency-domain channel coefficients for a given impulse response.

        Parameters
        ----------
        h: np.ndarray[float/complex]
            (N_tap,) channel impulse response.

        Returns
        -------
        H: np.ndarray[complex]
            (N_channel,) channel coefficients (frequency-domain).
            Coefficients of inactive channels are set to 0.
        """
        assert h.ndim == 1
        assert (
            self._N_prefix >= h.size - 1
        ), "Impulse response is incompatible with this OFDM transceiver."

        H = npf.fft(h, n=self._N_channel)
        H[~self._mask_ch] = 0
        return H

    def channel_LS(self, y_train: np.ndarray) -> np.ndarray:
        """
        Least-Squares (LS) estimation of frequency-domain channel coefficients.

        Parameters
        ----------
        y_train: np.ndarray[float/complex]
            (N_channel,) or (N_active_ch,) observed training symbols.

        Returns
        -------
        H: np.ndarray[complex]
            (N_channel,) channel coefficients (frequency-domain).
            Coefficients of inactive channels are set to 0.
        """
        assert y_train.ndim == 1
        N = y_train.size

        if N == self._N_channel:
            y_train = y_train[self._mask_ch]
        elif N == self._N_active_ch:
            pass
        else:
            raise ValueError("Parameter[y_train]: expected (N_channel,) or (N_active_ch,) array.")

        H = np.zeros((self._N_channel,), dtype=complex)
        H[self._mask_ch] = y_train / self._x_train[self._mask_ch]
        return H


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


def ofdm_tx_frame(
    N_prefix: int,
    x_train: np.ndarray,
    mask_ch: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Generate OFDM frame(s).

    Parameters
    ----------
    N_prefix: int
        Length of the circular prefix >= 0.
    x_train: np.ndarray[float/complex]
        (N_channel,) training symbols known to TX/RX. Used to estimate channel properties.
    mask_ch: np.ndarray[bool]
        (N_channel,) channel mask. Used to turn off individual channels. (I.e.: do not transmit data
        on inactive channels.)
    x: np.ndarray[float/complex]
        (N_data,) symbols. It will be padded with zeros if not a multiple of useable carriers.

    Returns
    -------
    y: np.ndarray[float/complex]
        (Q,) serialized OFDM symbols, with the training sequence transmitted in the first OFDM
        block.
    """
    mapper = OFDM(N_prefix, mask_ch, x_train)

    _, r = divmod(x.size, mapper._N_active_ch)
    x = np.pad(x, (0, mapper._N_active_ch - r))

    y = mapper.encode(x)
    return y


def ofdm_rx_frame(
    N_prefix: int,
    mask_ch: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Decode OFDM frame(s), without channel equalization.

    Parameters
    ----------
    N_prefix: int
        Length of the circular prefix >= 0.
    mask_ch: np.ndarray[bool]
        (N_channel,) channel mask. Used to turn off individual channels. (I.e.: no data received on
        inactive channels.)
    x: np.ndarray[float/complex]
        (N_data,) symbols. Any trailing partial OFDM frame will be discarded.

    Returns
    -------
    y: np.ndarray[complex]
        (Q, N_active_ch) symbols, where Q is the number of decoded OFDM (block-)symbols.
    """
    N_channel = mask_ch.size

    # x_train doesn't matter here since we don't do equalization (yet) -> choose anything.
    mapper = OFDM(N_prefix, mask_ch, x_train=np.zeros(N_channel))

    q, _ = divmod(x.size, N_channel + N_prefix)
    x = x[: (q * (N_channel + N_prefix))]

    y = mapper.decode(x)
    return y


def ofdm_channel(
    N_prefix: int,
    mask_ch: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """
    Compute the frequency-domain channel coefficients for a given impulse response.

    Parameters
    ----------
    N_prefix: int
        Length of the circular prefix >= 0. (Needed to determine if `h` is acceptable.)
    mask_ch: np.ndarray[bool]
        (N_channel,) channel mask. Used to turn off individual channels. (I.e.: no data on inactive
        channels.)
    h: np.ndarray[float/complex]
        (N_tap,) channel impulse response.

    Returns
    -------
    H: np.ndarray[complex]
        (N_active_ch,) channel coefficients (frequency-domain).
    """
    N_channel = mask_ch.size

    # x_train doesn't matter here -> choose anything.
    mapper = OFDM(N_prefix, mask_ch, x_train=np.zeros(N_channel))

    H = mapper.channel(h)[mask_ch]
    return H
