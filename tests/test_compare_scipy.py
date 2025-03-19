import numpy as np
import pytest
from scipy import signal

from pymultitaper import spectrogram


@pytest.fixture(autouse=True)
def deterministic_random(monkeypatch):
    monkeypatch.setattr("numpy.random", np.random.RandomState(9876))


@pytest.mark.parametrize("window_length_multiplier", [1, 5, 25])
def test_compare_scipy(window_length_multiplier):
    # white noise signal, 2s @8kHz
    fs = 8000
    x = np.linspace(0, 2, 2*fs)
    data = np.random.normal(size=len(x))
    # 10ms frame length
    ts = 0.01
    wl = ts * window_length_multiplier

    st_times, st_freqs, st_spec = spectrogram(data,
                                              fs=fs,
                                              time_step=ts,
                                              window_length=wl,
                                              db_scale=False,
                                              )
    n_ts = int(ts * fs)
    n_wl = int(wl * fs)
    sc_freqs, sc_times, sc_spec = signal.spectrogram(data,
                                                     fs=fs,
                                                     nperseg=n_wl,
                                                     nfft=2**int(np.ceil(np.log2(n_wl))),
                                                     noverlap=n_wl - n_ts,
                                                     window="hamming",
                                                     )

    assert st_freqs.shape == sc_freqs.shape
    assert st_times.shape == sc_times.shape
    assert st_spec.shape == st_spec.shape

    assert np.allclose(st_times, sc_times - sc_times[0])
    assert np.allclose(st_freqs, sc_freqs)
    assert np.allclose(st_spec, sc_spec)
