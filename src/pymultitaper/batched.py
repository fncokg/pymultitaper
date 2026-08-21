from typing import Literal, Optional
from functools import wraps
import numpy as np
from .backend import ArrayBackend
from .spectral import spectrogram, multitaper_spectrogram


def _batched_call(signal_list, lens, backend, func, kwargs):
    max_len = max(lens)
    batched_array = backend.xp.zeros(
        (len(signal_list), max_len), dtype=signal_list[0].dtype
    )
    for i, (signal, length) in enumerate(zip(signal_list, lens)):
        batched_array[i, :length] = signal

    freqs, times, spec = func(
        batched_array,
        **kwargs,
    )
    return freqs, times, spec


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
        mode: Literal["auto", "loop", "batched", "chunk_batched"] = "auto",
        chunk_size: Optional[int] = None,
        boundary_pad: bool = False,
        **kwargs,
    ):
        backend = ArrayBackend.like(signal_list[0])
        n_signals = len(signal_list)
        kwargs.update(
            dict(
                fs=fs,
                time_step=time_step,
                window_length=window_length,
                boundary_pad=boundary_pad,
            )
        )
        if mode == "auto":
            mode = "chunk_batched" if backend.backend == "cupy" else "loop"
        if mode in ["batched", "chunk_batched"]:
            nts = int(time_step * fs)
            nwl = int(window_length * fs) if window_length is not None else nts
            npad = int(nwl / 2) + 1
            lengths_offset = npad * 2 if boundary_pad else 0
            get_nframes = (
                lambda length: (length + lengths_offset - nwl + 1 + nts - 1) // nts
            )
            if mode == "batched":
                lens = [signal.shape[0] for signal in signal_list]
                freqs, times, spec = _batched_call(
                    signal_list, lens, backend, func, kwargs
                )
                spec_list = []
                time_list = []
                for i, length in enumerate(lens):
                    nframes = get_nframes(length)
                    spec_list.append(spec[i, :, :nframes])
                    time_list.append(times[:nframes])
            elif mode == "chunk_batched":
                chunk_size = chunk_size or n_signals // 10
                sort_ids = np.argsort([signal.shape[0] for signal in signal_list])
                spec_list = [None] * n_signals
                time_list = [None] * n_signals
                for chunk_id in range(0, n_signals, chunk_size):
                    ids = sort_ids[chunk_id : chunk_id + chunk_size]
                    chunk = [signal_list[i] for i in ids]
                    lens = [signal.shape[0] for signal in chunk]
                    freqs, times, spec = _batched_call(
                        chunk, lens, backend, func, kwargs
                    )

                    for i, length in enumerate(lens):
                        nframes = get_nframes(length)
                        spec_list[ids[i]] = spec[i, :, :nframes]
                        time_list[ids[i]] = times[:nframes]
        elif mode == "loop":
            freqs = None
            spec_list = []
            time_list = []
            for signal in signal_list:
                freqs, times, spec = func(
                    signal,
                    **kwargs,
                )
                spec_list.append(spec)
                time_list.append(times)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'loop' or 'batched'.")
        return freqs, time_list, spec_list

    return wrapper


@_batched
def batched_spectrogram(*args, **kwargs):
    """
        Compute spectrograms for a list of signals with selectable execution mode.

        This function accepts a list of 1D signals of varying lengths and supports
        four execution strategies through ``mode``:
        - ``"auto"``: select ``"loop"`` for NumPy or ``"chunk_batched"`` for CuPy.
        - ``"loop"``: compute each signal independently.
        - ``"batched"``: pad all signals to a common length and run one batched call.
        - ``"chunk_batched"``: split signals into chunks of similar
            lengths, and batch-process each chunk to minimize padding overhead. Chunk size can be controlled with the ``chunk_size`` argument, which defaults to ``n_signals // 10``.

        Spectrogram computation is otherwise identical to
        [`spectrogram`][src.pymultitaper.spectral.spectrogram], and returned results
        correspond to each input signal's valid frames.

        The computation backend (CPU or GPU) is determined by the input array type.

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
        mode (Literal["auto", "loop", "batched", "chunk_batched"], optional):
            Execution mode for list processing. ``"auto"`` selects automatically
            based on backend. ``"loop"`` runs per-signal computation.
            ``"batched"`` pads all signals to a common length. ``"chunk_batched"``
            splits into chunks for memory-efficient batch processing. Defaults to ``"auto"``.
        chunk_size (int, optional): Number of signals per chunk when using
            ``mode="chunk_batched"``. If ``None``, defaults to ``n_signals // 10``.

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
        Compute multitaper spectrograms for a list of signals with selectable
        execution mode.

        This function accepts a list of 1D signals of varying lengths and supports
        four execution strategies through ``mode``:
        - ``"auto"``: automatically select ``"loop"`` for NumPy or ``"chunk_batched"`` for CuPy.
        - ``"loop"``: compute each signal independently.
        - ``"batched"``: pad all signals to a common length and run one batched call.
        - ``"chunk_batched"``: split signals into chunks of similar
            lengths, and batch-process each chunk to minimize padding overhead. Chunk size can be controlled with the ``chunk_size`` argument, which defaults to ``n_signals // 10``.

        Spectrogram computation is otherwise identical to
        [`multitaper_spectrogram`][src.pymultitaper.spectral.multitaper_spectrogram],
        and returned results correspond to each input signal's valid frames.

        The computation backend (CPU or GPU) is determined by the input array type.

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
        mode (Literal["auto", "loop", "batched", "chunk_batched"], optional):
            Execution mode for list processing. ``"auto"`` selects automatically
            based on backend. ``"loop"`` runs per-signal computation.
            ``"batched"`` pads all signals to a common length. ``"chunk_batched"``
            splits into chunks for memory-efficient batch processing. Defaults to ``"auto"``.
        chunk_size (int, optional): Number of signals per chunk when using
            ``mode="chunk_batched"``. If ``None``, defaults to ``n_signals // 10``.

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
