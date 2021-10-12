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


class Modulator:
    """
    Modulate continuous symbols onto/out-of an analog channel.
    """

    def __init__(self, sample_rate: int, symbol_rate: int):
        """
        Parameters
        ----------
        sample_rate: int
            Sampling rate [sample/s].
        symbol_rate: int
            Symbol rate [symbol/s].
        """
        assert sample_rate > symbol_rate > 0
        self._sample_rate = sample_rate
        self._symbol_rate = symbol_rate

    def tx_interpolate(self, x: np.ndarray, filter: np.ndarray, axis: int = -1) -> np.ndarray:
        r"""
        Upsample and interpolate symbol stream(s) `x`.

        Parameters
        ----------
        x: np.ndarray[continuous alphabet]
            (..., N_sym, ...) symbol stream.
            Symbol rate = `symbol_rate`
        filter: np.ndarray[float/complex]
            (N_tap,) interpolation filter.
            Sample rate = `sample_rate`
        axis: int
            Dimension along which to filter. (Default: -1)

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N, ...) interpolated signal(s)  y(t) = \sum_{n} x[n] f(t - n T)
            Sample rate = `sample_rate`
        """
        assert filter.ndim == 1
        assert -x.ndim <= axis < x.ndim

        upsample_factor = int(self._sample_rate / self._symbol_rate)
        return (y := sps.upfirdn(filter, x, up=upsample_factor, axis=axis))

    def rx_prefilter(self, x: np.ndarray, filter: np.ndarray, axis: int = -1) -> np.ndarray:
        r"""
        Filter samples of analog signal(s).

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N_sample, ...) analog signal(s). [sampled]
            Sample rate = `sample_rate`
        filter: np.ndarray[float/complex]
            (N_tap,) filtering kernel.
            Sample rate = `sample_rate`
        axis: int
            Dimension along which to filter. (Default: -1)

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N, ...) filtered signal(s) y(t) = x(t) \conv f(t)
            Sample rate = `sample_rate`

            y[N_tap-1] is the first output where the filter fully-overlaps with the input.
        """
        assert filter.ndim == 1
        assert -x.ndim <= axis < x.ndim

        return (y := sps.upfirdn(filter, x, axis=axis))

    def rx_sample(self, x: np.ndarray, delay: int = 0, axis: int = -1) -> np.ndarray:
        """
        Sample analog signal(s).
        This method should be called after `rx_prefilter()`.

        Parameters
        ----------
        x: np.ndarray[float/complex]
            (..., N_sample, ...) signal(s) to sample.
            Sample rate = `sample_rate`
        delay: int
            Number of samples to drop at the start of `x`. (Default: 0)
        axis: int
            Dimension along which to sample. (Default: -1)

        Returns
        -------
        y: np.ndarray[float/complex]
            (..., N, ...) sampled signal(s).
            Sample rate = `symbol_rate`
        """
        assert delay >= 0
        assert -x.ndim <= axis < x.ndim

        select = [slice(None)] * x.ndim
        step = int(self._sample_rate / self._symbol_rate)
        select[axis] = slice(delay, None, step)
        return (y := x[tuple(select)])


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


def locate(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Find offset in `x` where `y` is most-likely located.

    In essence, find the offset in `x` where |<x, y>| is maximal.

    Parameters
    ----------
    x: np.ndarray[float/complex]
        (..., N_x, ...) input signal(s).
    y: np.ndarray[float/complex]
        (N_y,) pattern samples.
    axis: int
        Dimension of `x` along which to search for `y`.

    Returns
    -------
    idx: np.ndarray[int]
        (..., ...) offsets where `y` is most-likely located.
    """
    assert y.ndim == 1
    assert -x.ndim <= axis < x.ndim

    z = sps.upfirdn(y[::-1], x, axis=axis)

    # x/y do not fully-overlap during the convolution at the start of z: drop samples.
    s = [slice(None)] * x.ndim
    s[axis] = slice(len(y) - 1, None, None)
    z = z[tuple(s)]

    idx = np.argmax(np.abs(z), axis=axis)
    return idx


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
    modder = Modulator(sample_rate=usf, symbol_rate=1)
    y = modder.tx_interpolate(x, filter=pulse)
    return y


def sufficientStatistics(
    x: np.ndarray,
    pulse: np.ndarray,
    usf: int,
    delay: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter + sample an analog signal after matched-filtering.

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
        (N_y,) raw matched-filter output.
    z: np.ndarray[float/complex]
        (N_sym,) downsampled (and delayed) matched-filter output.
    """
    modder = Modulator(sample_rate=usf, symbol_rate=1)
    y = modder.rx_prefilter(x, filter=pulse[::-1].conj())
    z = modder.rx_sample(y, delay=delay)
    return y, z


def estimateDelay(x: np.ndarray, y: np.ndarray):
    """
    Find offset in `x` where `y` is most-likely located.

    In essence, find the offset in `x` where |<x, y>| is maximal.

    Parameters
    ----------
    x: np.ndarray[float/complex]
        (N_x,) input signal(s).
    y: np.ndarray[float/complex]
        (N_y,) pattern samples.

    Returns
    -------
    idx: int
        Offset w.r.t `x` where `y` is most-likely located.
    """
    idx = locate(x, y, axis=0)
    return idx
