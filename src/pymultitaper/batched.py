from typing import Literal, Optional
from functools import wraps

import numpy as np
from numpy.typing import NDArray

from .backend import ArrayBackend
from .spectral import spectrogram, multitaper_spectrogram


def batch_signals(
    signal_list: list,
    padding_strategy: Literal["max", "max_length"] = "max",
    max_length: Optional[int] = None,
    return_mask: bool = False,
) -> NDArray:
    """
    Convert a list of 1D signals into a padded 2D array.

    The computation backend (CPU or GPU) is automatically determined by the
    type of the input arrays.

    Args:
        signal_list (list): A list of 1D NumPy or CuPy arrays.
        padding_strategy (str): The padding strategy. Options are:

            - ``"max"``: Pad all signals to the length of the longest signal.
            - ``"max_length"``: Pad or truncate all signals to ``max_length``.

        max_length (int, optional): The target length when using
            ``"max_length"``. Required if ``padding_strategy`` is
            ``"max_length"``.
        return_mask (bool, optional): Whether to also return a boolean mask
            indicating valid, non-padded entries. Defaults to ``False``.

    Returns:
        NDArray: A 2D array of shape ``(n_signals, max_len)``.
        NDArray, optional: A boolean mask of the same shape when
            ``return_mask`` is ``True``.

    Examples:
        >>> signals = [np.arange(3), np.arange(5)]
        >>> batch_signals(signals).shape
        (2, 5)
    """

    if padding_strategy == "max":
        max_len = max(signal.shape[0] for signal in signal_list)
    elif padding_strategy == "max_length":
        if max_length is None:
            raise ValueError(
                "`max_length` must be specified when using 'max_length' padding strategy."
            )
        max_len = max_length
    else:
        raise ValueError(f"Unsupported padding strategy: {padding_strategy}")

    # Determine the backend based on the first signal
    backend = ArrayBackend.like(signal_list[0])

    # Create an empty array with the appropriate shape and backend
    batched_array = backend.xp.zeros(
        (len(signal_list), max_len), dtype=signal_list[0].dtype
    )
    mask = backend.xp.zeros((len(signal_list), max_len), dtype=backend.xp.bool)

    for i, signal in enumerate(signal_list):
        length = min(signal.shape[0], max_len)
        batched_array[i, :length] = signal[:length]
        mask[i, :length] = 1

    if return_mask:
        return batched_array, mask
    return batched_array


def frame_masking(mask, fs, time_step, window_length=None, boundary_pad=False):
    """
    Compute which spectrogram frames are valid for each padded signal.

    Args:
        mask (NDArray): A boolean mask with shape ``(n_signals, n_samples)``.
        fs (float): Sampling frequency.
        time_step (float): Time step between spectrogram frames in seconds.
        window_length (float, optional): Window length in seconds. If
            ``None``, ``time_step`` is used. Defaults to ``None``.
        boundary_pad (bool, optional): Whether the original signals were
            boundary padded before framing. Defaults to ``False``.

    Returns:
        NDArray: A boolean array of shape ``(n_signals, n_frames)`` indicating
        which frames are valid.
    """
    window_length = window_length if window_length is not None else time_step
    n_tstep = int(time_step * fs)
    n_winlen = int(window_length * fs)
    backend = ArrayBackend.like(mask)
    if boundary_pad:
        n_pad = int(n_winlen / 2) + 1
        pad_width = [(0, 0)] * mask.ndim
        pad_width[-1] = (n_pad, n_pad)
        mask = backend.xp.pad(
            mask, pad_width=pad_width, mode="constant", constant_values=False
        )
    frames = backend.xp.lib.stride_tricks.sliding_window_view(
        mask, n_winlen, writeable=False, axis=-1
    )[..., ::n_tstep, :]
    if boundary_pad:
        frame_mask = backend.xp.any(frames, axis=-1)
    else:
        frame_mask = backend.xp.all(frames, axis=-1)
    return frame_mask


def _batched(func):
    """
    Wrap a spectrogram function so it accepts a list of signals.

    The wrapped function is evaluated on a padded batch first, then the
    per-signal valid frames are selected and returned as Python lists.
    """

    @wraps(func)
    def wrapper(
        signal_list: list,
        fs: float,
        time_step: float,
        window_length: Optional[float] = None,
        detrend: Literal["off", "linear", "constant"] = "off",
        boundary_pad: bool = False,
        **kwargs,
    ):
        batched_array, mask = batch_signals(signal_list, return_mask=True)
        freqs, times, spec = func(
            batched_array,
            fs=fs,
            time_step=time_step,
            window_length=window_length,
            detrend=detrend,
            boundary_pad=boundary_pad,
            **kwargs,
        )
        frame_mask = frame_masking(
            mask, fs, time_step, window_length=window_length, boundary_pad=boundary_pad
        )
        spec_list = []
        time_list = []
        for i in range(len(signal_list)):
            spec_list.append(spec[i][:, frame_mask[i]])
            time_list.append(times[frame_mask[i]])
        return freqs, time_list, spec_list

    return wrapper


@_batched
def batched_spectrogram(*args, **kwargs):
    """
    Compute spectrograms for a list of signals with automatic batching. This is useful for speeding up computations when processing multiple signals, especially on GPUs.

    This function accepts a list of 1D signals of varying lengths and pads them
    to a common size for efficient batch processing. Spectrogram computation is
    otherwise identical to :func:`spectrogram`, and results correspond to each
    input signal's valid frames.

    The computation backend (CPU or GPU) is automatically determined by the type
    of the input arrays.

    Args:
        signal_list (list): A list of 1D NumPy or CuPy arrays of any length.
        fs (float): Sampling frequency.
        time_step (float): Time step between frames in seconds.
        window_length (float, optional): Window length in seconds. If ``None``,
            defaults to ``time_step``.
        window_shape (Union[str, tuple], optional): Window function shape.
            Defaults to ``"hamming"``.
        freq_range (list, optional): Frequency range ``[fmin, fmax]``. If
            ``None``, defaults to ``[0, fs/2]``.
        detrend (str, optional): Detrending method: ``"constant"``,
            ``"linear"``, or ``"off"``. Defaults to ``"off"`` (no detrending).
        nfft (int, optional): Number of FFT points.
        db_scale (bool, optional): Whether to scale to dB. Defaults to
            ``True``.
        p_ref (float, optional): Reference pressure for dB conversion.
            Defaults to ``2e-5`` Pa.
        boundary_pad (bool, optional): Whether to pad data at boundaries.
            Defaults to ``False``.

    Returns:
        freqs (NDArray): (n_freqs,) Frequency points.
        time_list (list): List of time arrays, one per input signal.
        spec_list (list): List of PSD arrays ``(n_freqs, n_frames_i)``, one
            per input signal, with invalid/padded frames removed.

    Examples:
        >>> from pymultitaper import batched_spectrogram
        >>> signals = [np.random.randn(200), np.random.randn(150)]
        >>> freqs, times, specs = batched_spectrogram(signals, fs=1000, time_step=0.01)
        >>> len(specs), len(times)
        (2, 2)
    """
    return spectrogram(*args, **kwargs)


@_batched
def batched_multitaper_spectrogram(*args, **kwargs):
    """
    Compute multitaper spectrograms for a list of signals with automatic batching. This is useful for speeding up computations when processing multiple signals, especially on GPUs.

    This function accepts a list of 1D signals of varying lengths and pads them
    to a common size for efficient batch processing. Spectrogram computation is
    otherwise identical to :func:`multitaper_spectrogram`, and results correspond
    to each input signal's valid frames.

    The computation backend (CPU or GPU) is automatically determined by the type
    of the input arrays.

    Args:
        signal_list (list): A list of 1D NumPy or CuPy arrays of any length.
        fs (float): Sampling frequency.
        time_step (float): Time step between frames in seconds.
        window_length (float, optional): Window length in seconds. If ``None``,
            defaults to ``time_step``.
        NW (float, optional): Multitaper time-bandwidth parameter. Defaults
            to ``4.0``.
        n_tapers (int, optional): Number of tapers. If ``None``, defaults to
            ``floor(2*NW - 1)``.
        freq_range (list, optional): Frequency range ``[fmin, fmax]``. If
            ``None``, defaults to ``[0, fs/2]``.
        weight_type (str, optional): Taper weighting: ``"unity"`` or ``"eig"``.
            Defaults to ``"unity"``.
        detrend (str, optional): Detrending method: ``"constant"``,
            ``"linear"``, or ``"off"``. Defaults to ``"off"`` (no detrending).
        nfft (int, optional): Number of FFT points.
        db_scale (bool, optional): Whether to scale to dB. Defaults to
            ``True``.
        p_ref (float, optional): Reference pressure for dB conversion.
            Defaults to ``2e-5`` Pa.
        boundary_pad (bool, optional): Whether to pad data at boundaries.
            Defaults to ``False``.

    Returns:
        freqs (NDArray): (n_freqs,) Frequency points.
        time_list (list): List of time arrays, one per input signal.
        psd_list (list): List of multitaper PSD arrays ``(n_freqs, n_frames_i)``,
            one per input signal, with invalid/padded frames removed.

    Examples:
        >>> from pymultitaper import batched_multitaper_spectrogram
        >>> signals = [np.random.randn(200), np.random.randn(150)]
        >>> freqs, times, specs = batched_multitaper_spectrogram(signals, fs=1000, time_step=0.01, NW=4)
        >>> len(specs), len(times)
        (2, 2)
    """
    return multitaper_spectrogram(*args, **kwargs)
