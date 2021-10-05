import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as npf

"""
Plotting utilities.
"""


def t_plot(
    sig: np.ndarray,
    sample_rate: int,
    name: str = "sig",
    title: str = "",
    ax: axes.Axes = None,
) -> axes.Axes:
    """
    Plot time-domain signal.

    Parameters
    ----------
    sig: np.ndarray[float]
        (N_sample,) time-domain signal.
    sample_rate: int
        Sampling rate of `sig`. [sample/s]
    name: str
        Signal name. (Optional)
    title: str
        Plot title. (Optional)
    ax: matplotlib.axes.Axes
        Canvas to draw on. (Optional)

    Returns
    -------
    ax: matplotlib.axes.Axes
        Time-domain plot.
    """
    assert sig.ndim == 1
    assert sample_rate > 0

    if ax is None:
        _, ax = plt.subplots()

    N_sample = sig.size
    t = np.arange(N_sample) / sample_rate

    ax.plot(t, sig)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(f"{name}(t)")
    ax.set_title(title)

    return ax


def f_plot(
    sig: np.ndarray,
    sample_rate: int,
    name: str = "sig",
    title: str = "",
    ax: axes.Axes = None,
) -> axes.Axes:
    """
    Plot signal magnitude response.

    Parameters
    ----------
    sig: np.ndarray[float]
        (N_sample,) time-domain signal.
    sample_rate: int
        Sampling rate of `sig`. [sample/s]
    name: str
        Signal name. (Optional)
    title: str
        Plot title. (Optional)
    ax: matplotlib.axes.Axes
        Canvas to draw on. (Optional)

    Returns
    -------
    ax: matplotlib.axes.Axes
        Magnitude response plot.
    """
    assert sig.ndim == 1
    assert sample_rate > 0

    if ax is None:
        _, ax = plt.subplots()

    N_sample = sig.size
    f = npf.fftfreq(N_sample, 1 / sample_rate)
    sf = np.abs(npf.fft(sig))

    ax.plot(npf.fftshift(f), npf.fftshift(sf))
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel(f"|{name}_{{F}}(f)|")
    ax.set_title(title)

    return ax


def tf_plot(sig: np.ndarray, sample_rate: int, name: str, title: str) -> figure.Figure:
    """
    Plot time-domain signal and its magnitude response.

    Parameters
    ----------
    sig: np.ndarray[float]
        (N_sample,) time-domain signal.
    sample_rate: int
        Sampling rate of `sig`. [sample/s]
    name: str
        Signal name.
    title: str
        Figure title.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Time-domain and magnitude response figure.
    """
    fig, ax = plt.subplots(2)
    t_plot(sig, sample_rate, name=name, ax=ax[0])
    f_plot(sig, sample_rate, name=name, ax=ax[1])
    fig.suptitle(title)
    return fig


def eye_plot(
    sig: np.ndarray,
    sample_rate: int,
    symbol_rate: int,
    delay: int = 0,
    ax: axes.Axes = None,
) -> axes.Axes:
    r"""
    Plot eye-diagram of a pulse train.

    Parameters
    ----------
    sig: np.ndarray[float]
        (N_sample,) signal of the form s(t) = \sum_{k} s_{k} \psi(t - kT), for some pulse \psi: \bR -> \bR.
    sample_rate: int
        Sampling rate of `sig`. [sample/s]
    symbol_rate: int
        Symbol rate of `sig`. [symbol/s]
    delay: int
        Index of first pulse peak. (Required to correctly label the X-axis.)
    ax: matplotlib.axes.Axes
        Canvas to draw on. (Optional)

    Returns
    -------
    ax: matplotlib.axes.Axes
        Eye-diagram
    """
    # Implement me
    pass
