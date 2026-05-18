import numpy as np
import pytest

from pymultitaper import spectrogram,multitaper_spectrogram


def test_check_nframe(xp):
    # convert results from xp backend to numpy for comparison
    def to_np(a):
        return xp.asnumpy(a) if hasattr(xp, "asnumpy") else a
    
    fs = 8000
    duration_sec = 2
    n_samples = duration_sec*fs
    data = xp.random.normal(size=n_samples)
    
    for _ in range(10):
        # make sure ts and wl are multiple of sampling period
        ts = xp.random.randint(10,1000)/fs
        wl = xp.random.randint(10,1000)/fs
        desired_times = np.arange(0,duration_sec,float(to_np(ts)))
        _, times, _ = spectrogram(data,fs=fs,time_step=float(to_np(ts)),window_length=float(to_np(wl)),db_scale=False,boundary_pad=True)

        _,times_mt,_ = multitaper_spectrogram(
            data,
            fs=fs,
            time_step=float(to_np(ts)),
            window_length=float(to_np(wl)),
            db_scale=False,
            boundary_pad=True,
        )
        
        times = to_np(times)
        times_mt = to_np(times_mt)
        
        assert times.shape == desired_times.shape
        assert np.allclose(times, desired_times)
        assert times_mt.shape == desired_times.shape
        assert np.allclose(times_mt, desired_times)
    
