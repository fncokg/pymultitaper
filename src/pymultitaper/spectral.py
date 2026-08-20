from typing import Literal, Tuple, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .backend import ArrayBackend, backend_like


@backend_like
def _get_dpss_windows(n_winlen, NW, n_tapers, weight_type="unity", backend=None):
    tapers, eigns = backend.signal.windows.dpss(
        n_winlen, NW, n_tapers, return_ratios=True
    )
    if weight_type == "unity":
        weights = backend.xp.ones(n_tapers) / n_tapers
    elif weight_type == "eig":
        weights = eigns / n_tapers
    else:
        raise ValueError(
            f"weight_type must be one of ['unity','eig'], got {weight_type}"
        )
    # (n_winlen,n_tapers)
    tapers = tapers.T
    return tapers, weights


@backend_like
def _get_1d_window(window_shape, n_winlen, backend=None):
    win_arr = backend.signal.get_window(window_shape, n_winlen)
    weights = backend.xp.ones(1)
    # (n_winlen,1)
    win_arr = win_arr[:, None]
    return win_arr, weights


def _spectrogram(
    data: NDArray,
    fs: float,
    time_step: float,
    win: NDArray,
    weights: NDArray,
    freq_range: list,
    detrend: Literal["constant", "linear", "off"],
    nfft: Optional[int] = None,
    db_scale: bool = True,
    p_ref: float = 2e-5,
    boundary_pad: bool = False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Core implementation of spectrogram (PSD) calculation.

    The spectrogram is calculated when multiple window array and their weights are given. The result is the weighted sum of the PSDs of each windowed frame (when the weights sum to 1, it is the weighted average PSD). The computation backend (CPU or GPU) is automatically determined by the type of the input arrays (numpy or cupy).

    Args:
        data (NDArray): (..., n_samples) Input data. Can be either a numpy array or a cupy array. The backend (CPU or GPU) will be automatically determined based on the type of the input data.
        fs (float): Sampling frequency
        time_step (float): Time step between frames in seconds
        win (NDArray): (n_winlen,n_wins) Window arrays from different window functions. Should be of the same backend as `data`.
        weights (NDArray): (n_wins,) Weights for each window array. Should be of the same backend as `data`.
        freq_range (list): [fmin,fmax] Frequency range to keep in the spectrogram, if `None`, [0,fs/2] is used
        detrend (str): {'constant','linear','off'} Detrend method
        nfft (int): The number of FFT points, if `None`, will be set to the smallest power of 2 that is larger than the window length
        db_scale (bool): Whether to scale the PSD in dB
        p_ref (float): When db_scale is True, the reference pressure level in Pa
        boundary_pad (bool, optional): Whether to pad the data with zeros at the beginning and end. This is useful when the data is not evenly divisible by the window length and time step. By default `False`.

            - If `True`, the data will be padded with zeros at the beginning and end, so that the first frame is centered on the first sample of data, and all samples are included in (at least) one frame.
            - If `False`, the first frame is centered at `window_length/2` seconds after the first sample, and samples after `n_frames*time_step+window_length` seconds are ignored.

    Returns:
        freqs (n_freqs,): Frequency points of the spectrogram
        times (n_frames,): Time points of each frame
        psd (..., n_freqs, n_frames): PSD spectrogram
    """
    # Prepare arguments
    # win: (n_winlen,n_wins)
    backend = ArrayBackend.like(data)
    if ArrayBackend.like(win) != backend or ArrayBackend.like(weights) != backend:
        raise ValueError(
            "The backend of win and weights must be the same as that of data"
        )
    n_winlen = win.shape[0]
    n_tstep = int(time_step * fs)
    if freq_range is None:
        freq_range = [0, fs / 2]
    fmin, fmax = freq_range

    if boundary_pad:
        n_pad = int(n_winlen / 2) + 1
        # pad only the last dimension
        # the following is equivalent to `np.pad(data,pad_width={-1: (n_pad, n_pad)},...)`, but this feature (`pad_width` as a dict) is not supported in cupy
        pad_width = [(0, 0)] * data.ndim
        pad_width[-1] = (n_pad, n_pad)
        data = backend.xp.pad(
            data, pad_width=pad_width, mode="constant", constant_values=0
        )
    # Step 1: Frame the data
    # (n_frames,n_winlen)
    frames = backend.xp.lib.stride_tricks.sliding_window_view(
        data, n_winlen, writeable=False, axis=-1
    )[..., ::n_tstep, :]
    n_frames = frames.shape[-2]
    # Step 2: Detrend (if necessary)
    if detrend != "off":
        frames = backend.signal.detrend(frames, axis=-1, type=detrend)
    # Step 3: Windowing
    # broadcasting: (...,n_frames,n_winlen,n_wins) = (...,n_frames,n_winlen,1) * (n_winlen,n_wins)
    wined_frames = frames[..., None] * win

    # Step 4: FFT
    nfft = (
        2 ** int(backend.xp.ceil(backend.xp.log2(n_winlen))) if nfft is None else nfft
    )
    # (...,n_frames,nfft,n_wins)
    # zero-padding is automatically done in `fft.rfft`
    fft_data = backend.fft.rfft(wined_frames, n=nfft, axis=-2)

    # Step 5: Calculate frequencies and time points
    raw_freqs = backend.fft.rfftfreq(nfft, 1 / fs)
    freqs_idx = backend.xp.where((raw_freqs >= fmin) & (raw_freqs <= fmax))[0]
    freqs = raw_freqs[freqs_idx]
    if boundary_pad:
        times = backend.xp.arange(0, n_frames) * time_step
    else:
        times = backend.xp.arange(0, n_frames) * time_step + n_winlen / 2 / fs

    # Note: we filter out the frequencies with frequency range, therefore implicitly filter out the negative frequencies
    fft_data = fft_data[..., freqs_idx, :]

    # Step 6: Calculate PSD and average over window types
    # (n_wins,) We need to scale the PSD by the sum of the square of the window and fs
    _scale = 1 / (fs * backend.xp.sum(win**2, axis=0))
    scale = _scale * weights
    psd_data = fft_data.real**2 + fft_data.imag**2
    psd_data = backend.xp.dot(psd_data, scale)
    psd_data *= 2
    if fmin == 0:
        psd_data[:, 0] /= 2
    if fmax == fs / 2 and nfft % 2 == 0:
        # if nfft is even, the Nyquist frequency is exactly at the middle of the spectrum and has no duplicate
        psd_data[:, -1] /= 2
    if db_scale:
        psd_data = 10 * backend.xp.log10(psd_data / p_ref**2)
    # (...,nfft,n_frames)
    psd_data = backend.xp.swapaxes(psd_data, -1, -2)
    return freqs, times, psd_data


def multitaper_spectrogram(
    data: NDArray,
    fs: float,
    time_step: float,
    window_length: Optional[float] = None,
    NW: float = 4.0,
    n_tapers: Optional[int] = None,
    freq_range: Optional[list] = None,
    weight_type: Literal["unity", "eig"] = "unity",
    detrend: Literal["constant", "linear", "off"] = "constant",
    nfft: Optional[int] = None,
    db_scale: bool = True,
    p_ref: float = 2e-5,
    boundary_pad: bool = False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute the multitaper PSD of the input data.

    The computation backend (CPU or GPU) is automatically determined by the type of the input arrays (numpy or cupy).

    Args:
        data (NDArray): (..., n_samples) Input data:

            - Can be of any shape **(including 1d array)**, as long as the last dimension is the time dimension (n_samples). The spectrogram computation will be applied to the last dimension, and the other dimensions will be treated as batch dimensions and preserved in the output.
            - Can be either a numpy array or a cupy array. The backend (CPU or GPU) will be automatically determined based on the type of the input data.

        fs (float): Sampling frequency
        time_step (float): Time step between frames in seconds
        window_length (float, optional): Window length in seconds. If `None`, will be set to the same as `time_step`. Defaults to None.
        NW (float, optional): NW value, see notes for details. Defaults to 4.0.
        n_tapers (Optional[int], optional): The max number of tapers, if `None`, will be set to NW*2-1. Defaults to None.
        freq_range (Optional[list], optional): The desired frequency range. If `None`, will be set to [0, fs/2]. Defaults to None.
        weight_type (Literal["unity","eig"], optional): The type of weights among tapers. Defaults to "unity".
        detrend (Literal["constant","linear","off"], optional): Whether and how to detrend the signal. Defaults to "constant".
        nfft (Optional[int], optional): The number of FFT points. If `None`, will be set to the smallest power of 2 that is larger than the window length. Defaults to None.
        db_scale (bool, optional): Whether convert the result to db scale, i.e. 10log10(psd/p_ref**2). Defaults to True.
        p_ref (float, optional): If `db_scale` is `True`, the `p_ref` value used in the dB conversion. Defaults to 2e-5.
        boundary_pad (bool, optional): Whether to pad the data with zeros at the beginning and end. This is useful when the data is not evenly divisible by the window length and time step. By default `False`.

            - If `True`, the data will be padded with zeros at the beginning and end, so that the first frame is centered on the first sample of data, and all samples are included in (at least) one frame.
            - If `False`, the first frame is centered at `window_length/2` seconds after the first sample, and samples after `n_frames*time_step+window_length` seconds are ignored.

    Notes:
        The value of 2W is the regularization bandwidth. Typically, we choose W to be a small multiple of the fundamental frequency 1/(N*dt) (where N is the number of samples in the data), i.e. W=i/(N*dt). The value of the parameter `NW` here is in fact the value of i (when dt is seen as 1). There's a trade-off between frequency resolution and variance reduction: A larger `NW` will reduce the variance of the PSD estimate, but also reduce the frequency resolution.

    Returns:
        freqs (NDArray): (n_freqs,) Frequency points of the spectrogram.
        times (NDArray): (n_frames,) Time points of each frame.
        psd (NDArray): (..., n_freqs, n_frames) PSD spectrogram. The shape of the output PSD is the same as the input data, except that the last dimension (time) is replaced by the frequency and time dimensions.

    Examples:
        >>> freqs,times,psd = multitaper_spectrogram(data,fs,time_step=0.001,window_length=0.005,NW=4)
    """
    # (nfft,n_frames)
    if n_tapers is None:
        # Note: NW may be a float number
        # We DONOT need a cupy float here
        n_tapers = np.floor(2 * NW - 1).astype(int)
    window_length = time_step if window_length is None else window_length
    n_winlen = int(window_length * fs)
    tapers, weights = _get_dpss_windows(n_winlen, NW, n_tapers, weight_type, like=data)
    return _spectrogram(
        data=data,
        fs=fs,
        time_step=time_step,
        win=tapers,
        weights=weights,
        freq_range=freq_range,
        detrend=detrend,
        nfft=nfft,
        db_scale=db_scale,
        p_ref=p_ref,
        boundary_pad=boundary_pad,
    )


def spectrogram(
    data: NDArray,
    fs: float,
    time_step: float,
    window_length: Optional[float] = None,
    window_shape: Union[str, tuple] = "hamming",
    freq_range: Optional[list] = None,
    detrend: Literal["constant", "linear", "off"] = "constant",
    nfft: Optional[int] = None,
    db_scale: bool = True,
    p_ref: float = 2e-5,
    boundary_pad: bool = False,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute the ordinary (single-taper) PSD of the input data.

    This is similar to `scipy.signal.spectrogram` except that it supports also GPU computation. The computation backend (CPU or GPU) is automatically determined by the type of the input arrays (numpy or cupy).

    Args:
        data (NDArray): (..., n_samples) Input data:

            - Can be of any shape **(including 1d array)**, as long as the last dimension is the time dimension (n_samples). The spectrogram computation will be applied to the last dimension, and the other dimensions will be treated as batch dimensions and preserved in the output.
            - Can be either a numpy array or a cupy array. The backend (CPU or GPU) will be automatically determined based on the type of the input data.

        fs (float): Sampling frequency
        time_step (float): Time step between frames in seconds
        window_length (float, optional): Window length in seconds. If `None`, will be set to the same as `time_step`. Defaults to None.
        window_shape (Union[str,tuple], optional): The shape of the window function. Defaults to "hamming".
        freq_range (Optional[list], optional): The desired frequency range. If `None`, will be set to [0, fs/2]. Defaults to None.
        detrend (Literal["constant","linear","off"], optional): Whether and how to detrend the signal. Defaults to "constant".
        nfft (Optional[int], optional): The number of FFT points. If `None`, will be set to the smallest power of 2 that is larger than the window length. Defaults to None.
        db_scale (bool, optional): Whether convert the result to db scale, i.e. 10log10(psd/p_ref**2). Defaults to True.
        p_ref (float, optional): If `db_scale` is `True`, the `p_ref` value used in the dB conversion. Defaults to 2e-5.
        boundary_pad (bool, optional): Whether to pad the data with zeros at the beginning and end. This is useful when the data is not evenly divisible by the window length and time step. By default `False`.

            - If `True`, the data will be padded with zeros at the beginning and end, so that the first frame is centered on the first sample of data, and all samples are included in (at least) one frame.
            - If `False`, the first frame is centered at `window_length/2` seconds after the first sample, and samples after `n_frames*time_step+window_length` seconds are ignored.

    Returns:
        freqs (NDArray): (n_freqs,) Frequency points of the spectrogram.
        times (NDArray): (n_frames,) Time points of each frame.
        psd (NDArray): (..., n_freqs, n_frames) PSD spectrogram. The shape of the output PSD is the same as the input data, except that the last dimension (time) is replaced by the frequency and time dimensions.

    Examples:
        >>> freqs,times,psd = spectrogram(data,fs,time_step=0.001,window_length=0.005)
    """
    window_length = time_step if window_length is None else window_length
    n_winlen = int(window_length * fs)
    win, weights = _get_1d_window(window_shape, n_winlen, like=data)
    return _spectrogram(
        data=data,
        fs=fs,
        time_step=time_step,
        win=win,
        weights=weights,
        freq_range=freq_range,
        detrend=detrend,
        nfft=nfft,
        db_scale=db_scale,
        p_ref=p_ref,
        boundary_pad=boundary_pad,
    )
