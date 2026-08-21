from typing import Literal, Optional
from functools import wraps

from .backend import ArrayBackend
from .spectral import spectrogram, multitaper_spectrogram


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
        lens = [signal.shape[0] for signal in signal_list]
        max_len = max(lens)

        backend = ArrayBackend.like(signal_list[0])
        batched_array = backend.xp.zeros(
            (len(signal_list), max_len), dtype=signal_list[0].dtype
        )
        for i, (signal, length) in enumerate(zip(signal_list, lens)):
            batched_array[i, :length] = signal

        freqs, times, spec = func(
            batched_array,
            fs=fs,
            time_step=time_step,
            window_length=window_length,
            detrend=detrend,
            boundary_pad=boundary_pad,
            **kwargs,
        )

        spec_list = []
        time_list = []
        for i, length in enumerate(lens):
            spec_list.append(spec[i, :, :length])
            time_list.append(times[:length])
        return freqs, time_list, spec_list

    return wrapper


@_batched
def batched_spectrogram(*args, **kwargs):
    """
    Compute spectrograms for a list of signals with automatic batching. This may be useful for speeding up computations when processing multiple signals, especially on GPUs.

    This function accepts a list of 1D signals of varying lengths and pads them to a common size for efficient batch processing.  Spectrogram computation is otherwise identical to [`spectrogram`][src.pymultitaper.spectral.spectrogram], and results correspond to each input signal's valid frames.

    The computation backend (CPU or GPU) is automatically determined by the type of the input arrays.

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
    Compute multitaper spectrograms for a list of signals with automatic batching. This may be useful for speeding up computations when processing multiple signals, especially on GPUs.

    This function accepts a list of 1D signals of varying lengths and pads them to a common size for efficient batch processing. Spectrogram computation is otherwise identical to [`multitaper_spectrogram`][src.pymultitaper.spectral.multitaper_spectrogram], and results correspond to each input signal's valid frames.

    The computation backend (CPU or GPU) is automatically determined by the type of the input arrays.

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
