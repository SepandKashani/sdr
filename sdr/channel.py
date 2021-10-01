import abc

import numpy as np
import scipy.signal as sps

"""
Channel models (not necessarily analog-domain).

signal -> [channel] -> signal
"""


class Channel:
    """
    Channel interface.
    """

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply channel effect on input(s).

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N_sample, ...) input signal.
        axis: int
            Dimension along which signal(s) are stored.

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N, ...) output signal.
        """
        assert -x.ndim <= axis < x.ndim
        pass


class DelayChannel(Channel):
    """
    Delays signals which pass through it.
    """

    def __init__(self, sample_rate: int):
        """
        Parameters
        ----------
        sample_rate: int
            Input sample rate. [sample/s]
        """
        assert sample_rate > 0
        self._sample_rate = sample_rate

    def __call__(self, x: np.ndarray, T: float, axis: int = -1) -> np.ndarray:
        """
        Delay input by `T` [s]

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N_sample, ...) input signal.
            Sample rate = `sample_rate`
        T: float
            Channel delay [s].
        axis: int
            Dimension along which signal(s) are stored.

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N, ...) output signal.
        """
        super().__call__(x, axis)
        assert T >= 0

        sh = list(x.shape)
        sh[axis] = int(np.ceil(T * self._sample_rate))
        # If the number of samples required to implement a delay T is integer-valued, then one can
        # just add zeros at the start of the signal.
        # If the number of samples required to implement a delay T is a real-valued, then one must
        # filter the signal with a suitable sinc-like kernel.
        # Note: the sinc-like kernel collapses to a Dirac if T is integer-valued.
        # Since the sampling rate is often high, we choose to always round T to the closest
        # integer-valued sample for simplicity.

        return (y := np.concatenate([np.zeros(sh, x.dtype), x], axis=axis))


class AWGNChannel(Channel):
    """
    Additive White Gaussian Noise channel.
    """

    def __init__(self, var: float):
        """
        Parameters
        ----------
        sample_rate: int
            Input sample rate. [sample/s]
        var: float
            Noise variance.
        """
        assert var >= 0
        self._var = var

        self._rng = np.random.default_rng()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Corrupt input(s) with AWG noise.

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N_sample, ...) input signal.

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N_sample, ...) output signal.
            If `x` is    real-valued, then y[k] ~  N(x[k], var)
            If `x` is complex-valued, then y[k] ~ CN(x[k], var)
        """
        noise = lambda std: self._rng.normal(loc=0, scale=std, size=x.shape)

        dk = x.dtype.kind
        if dk in {"i", "f"}:  # int/float
            std = np.sqrt(self._var)
            n = noise(std)
        elif dk == "c":  # complex
            std = np.sqrt(0.5 * self._var)
            n = noise(std) + 1j * noise(std)
        else:
            raise ValueError(f"Parameter[x]: unsupported input type: {dk}.")

        return (y := x + n)


class FIRChannel(Channel):
    """
    Finite Impulse Response channel.

    Outputs are convolved with a provided channel filter.
    """

    def __init__(self, filter: np.ndarray):
        """
        Parameters
        ----------
        filter: np.ndarray[float/complex]
            (N_tap,) FIR taps.
        """
        assert filter.ndim == 1
        self._filter = filter.copy()

    def __call__(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Filter input(s).

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N, ...) input signal(s).
        axis: int
            Dimension along which signal(s) are stored.

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N + N_tap - 1, ...) output signal(s).

            y[N_tap-1] is the first output where the filter fully-overlaps with the input.
        """
        super().__call__(x, axis)

        y = sps.upfirdn(self._filter, x, axis=axis)
        return y


class MIMOChannel(Channel):
    r"""
    Multi-Input-Multi-Output channel.

    Y = H X + N, where X \in \bC^{N_tx}
                       Y \in \bC^{N_rx}
                       H[i,j] ~ CN(0, 1)
                       N ~ CN(0, var * I_{N_rx})
    """

    def __init__(self, N_tx: int, N_rx: int, var: float):
        """
        Parameters
        ----------
        N_tx: int
            Number of transmitter elements.
        N_rx: int
            Number of receiver elements.
        var: float
            Noise variance.
        """
        assert min(N_tx, N_rx) > 0
        assert var >= 0

        rng = np.random.default_rng()
        params = dict(loc=0, scale=np.sqrt(0.5), size=(N_rx, N_tx))
        self._H = rng.normal(**params) + 1j * rng.normal(**params)
        self._awgn = AWGNChannel(var)

    def __call__(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Generate noisy RX signals from TX inputs.

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N_tx, ...) TX input(s)
        axis: int
            Dimension along which lie input signals.

        Returns
        -------
        y: np.ndarray[complex]
            (..., N_rx, ...) RX signal(s).
        """
        assert -x.ndim <= axis < x.ndim

        Hx = np.tensordot(self._H, x, axes=((1,), (axis,)))  # shape: (N_rx, ...)
        Hx = np.moveaxis(Hx, 0, axis)  # shape: (..., N_rx, ...)

        n = self._awgn(Hx.astype(complex))  # to force complex-valued noise addition.
        return (y := Hx + n)

    @property
    def channel(self) -> np.ndarray:
        """
        Returns
        -------
        H : np.ndarray[complex]
            (N_rx, N_tx) channel matrix.
        """
        return self._H

    @property
    def var(self) -> float:
        """
        Returns
        -------
        v: float
            Noise variance.
        """
        return self._awgn._var
