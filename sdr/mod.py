import numpy as np
import scipy.signal as sps

"""
Continuous signal sampling and interpolation.

TX architecture:
    discrete symbols -> [mapper] -> continuous symbols -> [modulator] -> continuous signal

RX architecture:
    continuous signal -> [modulator] -> continuous symbols -> [mapper] -> discrete symbols
                         [    +    ]
                         [sym_sync ]
"""


class UpDownConverter:
    """
    Baseband <> Passband modulator.

    Note: the oscillator's last phase offset is retained between calls. Therefore it is possible to
    up/down-convert a stream block-wise. (If not using `post_filter` in `down()`.)

    Caveat: don't use the same object for up-conversion and down-conversion as oscillator is
    up/down-agnostic.
    """

    def __init__(self, sample_rate: int):
        assert sample_rate > 0
        self._sample_rate = sample_rate
        self._offset = 1  # \phi

    def up(self, x: np.ndarray, fc: float, axis: int = -1) -> np.ndarray:
        r"""
        Up-convert signal(s) `x` to center frequency `fc`.

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N_sample, ...) continous signal(s) to up-convert.
            Sample rate = `sample_rate`
        fc: float
            Target center frequency [Hz].
        axis: int
            Dimension along which to up-convert. (Default: -1)

        Returns
        -------
        y: np.ndarray[float]
            (..., N_sample, ...) up-converted continuous signal(s).
            Sample rate = `sample_rate`

            y(t) = \sqrt{2} \Re{ x(t) \exp(j 2 \pi f_{c} t + \phi) }
        """
        assert fc > 0
        assert -x.ndim <= axis < x.ndim

        t = np.arange(x.shape[axis] + 1) / self._sample_rate  # +1 to compute offset
        mod = np.exp(1j * (2 * np.pi * fc) * t) * self._offset
        mod, self._offset = np.sqrt(2) * mod[:-1], mod[-1]

        sh = [1] * x.ndim
        sh[axis] = -1

        return (y := np.real(x * mod.reshape(sh)))

    def down(self, x: np.ndarray, fc: float, post_filter: np.ndarray, axis: int = -1) -> np.ndarray:
        r"""
        Down-convert signal(s) `x` from center frequency `fc` to baseband.

        Parameters
        ----------
        x: np.ndarray[float]
            (..., N_sample, ...) continuous signal(s) to down-convert.
            Sample rate = `sample_rate`
        fc: float
            Current center frequency [Hz].
        post_filter: np.ndarray[float/complex]
            (N_tap,) low-pass filtering kernel.
            Sample rate = `sample_rate`
        axis:int
            Dimension along which to down-convert. (Default: -1)

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N_sample, ...) down-converted continuous signal(s).
            Sample rate = `sample_rate`

            y(t) = { \sqrt{2} x(t) \exp(-j 2 \pi f_{c} t + \phi) } \conv LP
        """
        assert fc > 0
        assert post_filter.ndim == 1
        assert -x.ndim <= axis < x.ndim

        t = np.arange(x.shape[axis] + 1) / self._sample_rate  # +1 to compute offset
        mod = np.exp(-1j * (2 * np.pi * fc) * t) * self._offset
        mod, self._offset = np.sqrt(2) * mod[:-1], mod[-1]

        sh = [1] * x.ndim
        sh[axis] = -1

        return (y := sps.upfirdn(post_filter, x * mod.reshape(sh), axis=axis))


def symbol2samples(x: np.ndarray, pulse: np.ndarray, usf: int) -> np.ndarray:
    r"""
    Modulate a discrete symbol stream to line samples.

    Parameters
    ----------
    x: np.ndarray[float/complex]
        (N_sym,) codewords
    pulse: np.ndarray[float/complex]
        (N_p,) sampled pulse
    usf: int
        Upsampling factor

    Returns
    -------
    y: np.ndarray[float/complex]
        (N_sample,) interpolated signal y(t) = \sum_{n} x[n] f(t - n T)
        N_sample should be >= usf * N_sym.
    """
    # Implement me
    pass


def sufficientStatistics(x: np.ndarray, pulse: np.ndarray, usf: int, delay: int = 0) -> np.ndarray:
    """
    Filter + sample an analog signal.

    Parameters
    ----------
    x: np.ndarray[float/complex]
        (N_sample,) line samples.
    pulse: np.ndarray[float/complex]
        (N_p,) pulse used to map codewords to line samples.
    usf: int
        Upsampling factor
    delay: int
        Number of samples to drop prior to downsampling. (Default: 0)

    Returns
    -------
    y: np.ndarray[float/complex]
        (N_sym,) sampled codewords.
    """
    # Implement me
    pass
