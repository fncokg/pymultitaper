import numpy as np
import pytest

from pymultitaper import spectrogram,multitaper_spectrogram


@pytest.fixture(autouse=True)
def deterministic_random(monkeypatch):
    monkeypatch.setattr("numpy.random", np.random.RandomState(9876))


def test_check_nframe():
    fs = 8000
    duration_sec = 2
    n_samples = duration_sec*fs
    data = np.random.normal(size=n_samples)
    
    for _ in range(10):
        # make sure ts and wl are multiple of sampling period
        ts = np.random.randint(10,1000)/fs
        wl = np.random.randint(10,1000)/fs
        desired_times = np.arange(0,duration_sec,ts)
        _, times, _ = spectrogram(data,
                                                fs=fs,
                                                time_step=ts,
                                                window_length=wl,
                                                db_scale=False,
                                                boundary_pad=True,
                                                )

        _,times_mt,_ = multitaper_spectrogram(
            data,
            fs=fs,
            time_step=ts,
            window_length=wl,
            db_scale=False,
            boundary_pad=True,
        )
        
        
        
        assert times.shape == desired_times.shape
        assert np.allclose(times, desired_times)
        assert times_mt.shape == desired_times.shape
        assert np.allclose(times_mt, desired_times)
    
