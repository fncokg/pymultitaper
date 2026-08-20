from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray


# A helper function to convert cupy arrays to numpy arrays for plotting
def _as_np(x):
    try:
        return x.get()
    except AttributeError:
        return x


def plot_spectrogram(
    times: NDArray,
    freqs: NDArray,
    psd: NDArray,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> tuple:
    """
    Plot the spectrogram.

    Note: Convert the spectrogram to dB scale (set `db_scale` to `True` in the spectrogram functions, or convert it manually) before plotting, otherwise the plot may not be very informative.

    Args:
        times (n_frames,): Time points of each frame
        freqs (n_freqs,): Frequency points of the spectrogram
        psd (n_freqs,n_frames): PSD spectrogram
        ax (Optional[plt.Axes], optional): The Axes object to plot the spectrogram. If `None`, a new figure will be created. Defaults to None.
        **kwargs: Additional arguments to `ax.pcolormesh`

    Returns:
        fig (plt.Figure): The figure object
        ax (plt.Axes): The Axes object

    Examples:
        >>> f,ax = plt.subplots(1,1)
        >>> plot_spectrogram(times,freqs,psd,ax=ax,cmap="viridis")
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    times, freqs, psd = map(_as_np, [times, freqs, psd])
    mesh = ax.pcolormesh(times, freqs, psd, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax)
    return fig, ax


def plot_spectrum(
    times: NDArray,
    freqs: NDArray,
    psd: NDArray,
    time: float,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> tuple:
    """

    Plot the spectrum at a specific time point.

    Args:
        times (n_frames,): Time points of each frame
        freqs (n_freqs,): Frequency points of the spectrogram
        psd (n_freqs,n_frames): PSD spectrogram
        time (float): The time point to plot the spectrum
        ax (Optional[plt.Axes], optional): The Axes object to plot the spectrum. If `None`, a new figure will be created. Defaults to None.
        **kwargs: Additional arguments to `ax.plot`

    Returns:
        fig (plt.Figure): The figure object
        ax (plt.Axes): The Axes object

    Examples:
        >>> f,ax = plt.subplots(1,1)
        >>> plot_spectrum(times,freqs,psd,time=0.7,ax=ax)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    times, freqs, psd = map(_as_np, [times, freqs, psd])
    idx = np.argmin(np.abs(times - time))
    ax.plot(freqs, psd[:, idx], **kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"Spectrum at time {time}s")
    return fig, ax
