from typing import Literal, Tuple, Optional, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from scipy import signal
from scipy import fft


def _get_dpss_windows(n_winlen,NW,n_tapers,weight_type="unity"):
    tapers,eigns = signal.windows.dpss(n_winlen,NW,n_tapers,return_ratios=True)
    if weight_type == "unity":
        weights = np.ones(n_tapers) / n_tapers
    elif weight_type == "eig":
        weights = eigns / n_tapers
    else:
        raise ValueError(f"weight_type must be one of ['unity','eig'], got {weight_type}")
    # (n_winlen,n_tapers)
    tapers = tapers.T
    return tapers,weights

def _get_1d_window(window_shape,n_winlen):
    win_arr = signal.get_window(window_shape,n_winlen)
    weights = np.ones(1)
    # (n_winlen,1)
    win_arr = win_arr[:,None]
    return win_arr,weights

def _spectrogram(data:NDArray,fs:int,time_step:float,win:NDArray,weights:NDArray,freq_range:list,detrend:Literal["constant","linear","off"],nfft:Optional[int]=None,db_scale:bool=True,p_ref:float=2e-5) -> Tuple[NDArray,NDArray,NDArray]:
    """
    Core implementation of spectrogram (PSD) calculation.

    The spectrogram is calculated when multiple window array and their weights are given. The result is the weighted sum of the PSDs of each windowed frame (when the weights sum to 1, it is the weighted average PSD).

    Args:
        data (NDArray): (n_samples,)
        fs (int): Sampling frequency
        time_step (float): Time step between frames in seconds
        win (NDArray): (n_winlen,n_wins) Window arrays from different window functions
        weights (NDArray): (n_wins,) Weights for each window array
        freq_range (list): [fmin,fmax] Frequency range to keep in the spectrogram, if `None`, [0,fs/2] is used
        detrend (str): {'constant','linear','off'} Detrend method
        nfft (int): The number of FFT points, if `None`, will be set to the smallest power of 2 that is larger than the window length
        db_scale (bool): Whether to scale the PSD in dB
        p_ref (float): When db_scale is True, the reference pressure level in Pa

    Returns:
        times (n_frames,): Time points of each frame
        freqs (n_freqs,): Frequency points of the spectrogram
        psd (n_freqs,n_frames): PSD spectrogram
    """
    # Prepare arguments
    # win: (n_winlen,n_wins)
    n_winlen = win.shape[0]
    n_tstep = int(time_step*fs)
    if freq_range is None:
        freq_range = [0,fs/2]
    fmin,fmax = freq_range

    # Step 1: Frame the data
    # (n_frames,n_winlen)
    frames = np.lib.stride_tricks.sliding_window_view(data,n_winlen,writeable=False)[::n_tstep]
    n_frames = frames.shape[0]

    # Step 2: Detrend (if necessary)
    if detrend != "off":
        frames = signal.detrend(frames,axis=1,type=detrend)
    
    # Step 3: Windowing
    # (n_frames,n_winlen,n_wins) = (n_frames,n_winlen,1) * (1,n_winlen,n_wins)
    wined_frames = frames[...,None] * win[None,...]

    # Step 4: FFT
    nfft = 2**int(np.ceil(np.log2(n_winlen))) if nfft is None else nfft
    # (n_frames,nfft,n_wins)
    # zero-padding is automatically done in `fft.rfft`
    fft_data = fft.rfft(wined_frames,n=nfft,axis=1)

    # Step 5: Calculate frequencies and time points
    raw_freqs = fft.rfftfreq(nfft,1/fs)
    freqs_idx = np.where((raw_freqs >= fmin) & (raw_freqs <= fmax))[0]
    freqs = raw_freqs[freqs_idx]
    times = np.arange(0,n_frames) * time_step
    # Note: we filter out the frequencies with frequency range, therefore implicitly filter out the negative frequencies
    fft_data = fft_data[:,freqs_idx,:]
    
    # Step 6: Calculate PSD and average over window types
    # (n_wins,) We need to scale the PSD by the sum of the square of the window and fs
    _scale = 1 / (fs * np.sum(win**2,axis=0))
    scale = _scale * weights
    psd_data = fft_data.real**2 + fft_data.imag**2
    psd_data = np.dot(psd_data,scale)
    psd_data *= 2
    if fmin == 0:
        psd_data[:,0] /= 2
    if fmax == fs/2 and nfft % 2 == 1:
        # if nfft is odd, the Nyquist frequency is exactly at the middle of the spectrum and has no duplicate
        psd_data[:,-1] /= 2
    if db_scale:
        psd_data = 10*np.log10(psd_data/p_ref**2)
    # (nfft,n_frames)
    psd_data = psd_data.T
    return times,freqs,psd_data


def multitaper_spectrogram(data:NDArray,fs:int,time_step:float,window_length:float,NW:float=4.0,n_tapers:Optional[int]=None,freq_range:Optional[list]=None,weight_type:Literal["unity","eig"]="unity",detrend:Literal["constant","linear","off"]="constant",nfft:Optional[int]=None,db_scale:bool=True,p_ref:float=2e-5)-> Tuple[NDArray,NDArray,NDArray]:
    """
    Compute the multitaper PSD of the input data.

    Args:
        data (NDArray): (n_samples,) Input data
        fs (int): Sampling frequency
        time_step (float): Time step between frames in seconds
        window_length (float): Window length in seconds
        NW (float, optional): NW value, see notes for details. Defaults to 4.0.
        n_tapers (Optional[int], optional): The max number of tapers, if `None`, will be set to NW*2+1. Defaults to None.
        freq_range (Optional[list], optional): The desired frequency range. If `None`, will be set to [0, fs/2]. Defaults to None.
        weight_type (Literal["unity","eig"], optional): The type of weights among tapers. Defaults to "unity".
        detrend (Literal["constant","linear","off"], optional): Whether and how to detrend the signal. Defaults to "constant".
        nfft (Optional[int], optional): The number of FFT points. If `None`, will be set to the smallest power of 2 that is larger than the window length. Defaults to None.
        db_scale (bool, optional): Whether convert the result to db scale, i.e. 10log10(psd/p_ref**2). Defaults to True.
        p_ref (float, optional): If `db_scale` is `True`, the `p_ref` value used in the dB conversion. Defaults to 2e-5.
    
    Notes:
        The value of 2W is the regularization bandwidth. Typically, we choose W to be a small multiple of the fundamental frequency 1/(N*dt) (where N is the number of samples in the data), i.e. W=i/(N*dt). The value of the parameter `NW` here is in fact the value of i (when dt is seen as 1). There's a trade-off between frequency resolution and variance reduction: A larger `NW` will reduce the variance of the PSD estimate, but also reduce the frequency resolution. 

    Returns:
        times (NDArray): (n_frames,) Time points of each frame
        freqs (NDArray): (n_freqs,) Frequency points of the spectrogram
        psd (NDArray): (n_freqs,n_frames) PSD spectrogram
    
    Examples:
        >>> times,freqs,psd = multitaper_spectrogram(data,fs,time_step=0.001,window_length=0.005,NW=4)
    """
    # (nfft,n_frames)
    if n_tapers is None:
        # Note: NW may be a float number
        n_tapers = np.floor(2*NW-1).astype(int)
    n_winlen = int(window_length*fs)
    tapers,weights = _get_dpss_windows(n_winlen,NW,n_tapers,weight_type)
    return _spectrogram(data=data,fs=fs,time_step=time_step,win=tapers,weights=weights,freq_range=freq_range,detrend=detrend,nfft=nfft,db_scale=db_scale,p_ref=p_ref)

def spectrogram(data:NDArray,fs:int,time_step:float,window_length:float,window_shape:Union[str,tuple]="hamming",freq_range:Optional[list]=None,detrend:Literal["constant","linear","off"]="constant",nfft:Optional[int]=None,db_scale:bool=True,p_ref:float=2e-5)-> Tuple[NDArray,NDArray,NDArray]:
    """
    Compute the ordinary (single-taper) PSD of the input data. This is similar to `scipy.signal.spectrogram`.

    Args:
        data (NDArray): (n_samples,) Input data
        fs (int): Sampling frequency
        time_step (float): Time step between frames in seconds
        window_length (float): Window length in seconds
        window_shape (Union[str,tuple], optional): The shape of the window function. Defaults to "hamming".
        freq_range (Optional[list], optional): The desired frequency range. If `None`, will be set to [0, fs/2]. Defaults to None.
        detrend (Literal["constant","linear","off"], optional): Whether and how to detrend the signal. Defaults to "constant".
        nfft (Optional[int], optional): The number of FFT points. If `None`, will be set to the smallest power of 2 that is larger than the window length. Defaults to None.
        db_scale (bool, optional): Whether convert the result to db scale, i.e. 10log10(psd/p_ref**2). Defaults to True.
        p_ref (float, optional): If `db_scale` is `True`, the `p_ref` value used in the dB conversion. Defaults to 2e-5.

    Returns:
        times (NDArray): (n_frames,) Time points of each frame
        freqs (NDArray): (n_freqs,) Frequency points of the spectrogram
        psd (NDArray): (n_freqs,n_frames) PSD spectrogram
    
    Examples:
        >>> times,freqs,psd = spectrogram(data,fs,time_step=0.001,window_length=0.005)
    """
    
    n_winlen = int(window_length*fs)
    win,weights = _get_1d_window(window_shape,n_winlen)
    return _spectrogram(data=data,fs=fs,time_step=time_step,win=win,weights=weights,freq_range=freq_range,detrend=detrend,nfft=nfft,db_scale=db_scale,p_ref=p_ref)

def plot_spectrogram(times:NDArray,freqs:NDArray,psd:NDArray,ax:Optional[plt.Axes]=None,**kwargs)-> tuple:
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
        fig,ax = plt.subplots()
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(times,freqs,psd,**kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh,ax=ax)
    return fig,ax

def plot_spectrum(times:NDArray,freqs:NDArray,psd:NDArray,time:float,ax:Optional[plt.Axes]=None,**kwargs)-> tuple:
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
        fig,ax = plt.subplots()
    else:
        fig = ax.figure
    idx = np.argmin(np.abs(times-time))
    ax.plot(freqs,psd[:,idx],**kwargs)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"Spectrum at time {time}s")
    return fig,ax