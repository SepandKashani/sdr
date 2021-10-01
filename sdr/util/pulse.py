import abc

import numpy as np
import numpy.linalg as npl

"""
Pulse generation and sampling.
"""


class Pulse:
    r"""
    Analog pulse \psi: \bR -> \bC

    All pulses are normalized such that \norm{\psi}_{2} = 1.
    """

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x: np.ndarray[float]
            Evaluation points.

        Returns
        -------
        y: np.ndarray[float/complex]
            Sample values.
        """
        raise NotImplementedError


class Sinc(Pulse):
    r"""
    Sinc pulse.

    \psi(x) = \sqrt{1 / T} \sinc(t / T)
    \sinc(x) = \sin(\pi x) / (\pi x)

    \psi is orthonormal to its T-spaced shifts.
    """

    def __init__(self, T: float):
        assert T > 0
        self._T = T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.sinc(x / self._T) / np.sqrt(self._T)
        return y


class Rectangle(Pulse):
    r"""
    Rectangular pulse.

    \psi(x) = \sqrt{1 / T}    if |x| <= T/2
            = 0               otherwise

    \psi is orthonormal to its T-spaced shifts.
    """

    def __init__(self, T: float):
        assert T > 0
        self._T = T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(x.shape, dtype=float)
        y[np.abs(x) <= self._T / 2] = np.sqrt(1 / self._T)
        return y


class Triangle(Pulse):
    r"""
    Triangular pulse.

    \psi(x) = \sqrt{\frac{3}{4 T}} (1 - |2x/T|)   if |x| <= T/2
            = 0                                   otherwise

    \psi is orthonormal to its T-spaced shifts.
    """

    def __init__(self, T: float):
        assert T > 0
        self._T = T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.fmax(1 - np.abs(2 * x / self._T), 0) * np.sqrt(0.75 / self._T)
        return y


class RRC(Pulse):
    r"""
    Root-Raised Cosine Pulse.

    \psi_{\cF}(f) = \sqrt(T)                                                                  if |f| <= \frac{1 - \beta}{2 * T}
                    \sqrt(T) \cos[ \frac{\pi T}{2 \beta} {\abs{f} - \frac{1 - \beta}{2 T}} ]  if \frac{1 - \beta}{2 T} <= |f| <= \frac{1 + \beta}{2 T}
                    0                                                                         otherwise

    \beta \in [0, 1]

    \psi is orthonormal to its T-spaced shifts.
    """

    def __init__(self, T: float, beta: float):
        assert T > 0
        self._T = T

        assert 0 <= beta <= 1
        self._beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.isclose(self._beta, 0):
            y = np.sinc(x / self._T) / np.sqrt(self._T)
        else:
            y = np.zeros(x.shape, dtype=np.float)
            mask = np.isclose(np.abs(x), 0.25 * self._T / self._beta)

            y[mask] = (self._beta / (np.pi * np.sqrt(2 * self._T))) * (
                (np.pi + 2) * np.sin(0.25 * np.pi / self._beta)
                + (np.pi - 2) * np.cos(0.25 * np.pi / self._beta)
            )

            y[~mask] = (
                (4 * self._beta / (np.pi * np.sqrt(self._T)))
                * (
                    np.cos((1 + self._beta) * np.pi * x[~mask] / self._T)
                    + (0.25 * np.pi * (1 - self._beta) / self._beta)
                    * np.sinc((1 - self._beta) * x[~mask] / self._T)
                )
                / (1 - (4 * self._beta * x[~mask] / self._T) ** 2)
            )
        return y


def sample(pulse: Pulse, start: float, stop: float, sample_rate: int) -> np.ndarray:
    r"""
    Sample a pulse \psi(x) at points x \in ]start, stop[.

    The sample at x=0 is always present in the output.

    Parameters
    ----------
    pulse: Pulse
        Pulse to sample.
    start: float
        Sampling interval starting point < 0.
    stop: float
        Sampling interval ending point > 0.
    sample_rate: int
        Signal sampling rate.

    Returns
    -------
    t: np.ndarray[float]
        Sample points.
    y: np.ndarray[float/complex]
        Sampled pulse values.
    """
    assert start < 0 < stop
    assert sample_rate > 0

    t = np.r_[
        -np.arange(0, -start, step=1 / sample_rate)[1:][::-1],
        0,
        np.arange(0, stop, step=1 / sample_rate)[1:],
    ]
    return t, (y := pulse(t))


def rcosdesign(beta: float, span: int, sps: int) -> np.ndarray:
    """
    Generate samples of an RRC pulse.

    Parameters
    ----------
    beta: float
    span: int
        Total number of symbols spanned by pulse. (Must be odd-valued.)
    sps: int
        Number of samples per symbol.

    Returns
    -------
    p: np.ndarray[float]
        (N_span * sps,) pulse coefficients, normalized to unit energy.
    """
    _, p = sample(
        pulse=RRC(T=1, beta=beta),
        start=-(span - 1),
        stop=span - 1,
        sample_rate=sps,
    )
    p /= npl.norm(p, ord=2)
    return p
