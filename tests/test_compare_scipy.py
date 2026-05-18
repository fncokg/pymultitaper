import numpy as np
import pytest
from scipy import signal

from pymultitaper import spectrogram


@pytest.mark.parametrize("window_length_multiplier", [1, 5, 25])
def test_compare_scipy(window_length_multiplier, xp):

    # convert results from xp backend to numpy for comparison
    def to_np(a):
        return xp.asnumpy(a) if hasattr(xp, "asnumpy") else a
    
    # white noise signal, 2s @8kHz
    fs = 8000
    x = xp.linspace(0, 2, 2 * fs)
    data = xp.random.normal(size=len(x))
    # 10ms frame length
    ts = 0.01
    wl = ts * window_length_multiplier

    st_freqs, st_times, st_spec = spectrogram(
        data,
        fs=fs,
        time_step=ts,
        window_length=wl,
        db_scale=False,
        boundary_pad=False,
    )

    # compute scipy reference with numpy arrays
    n_ts = int(ts * fs)
    n_wl = int(wl * fs)
    np_data = to_np(data)
    sc_freqs, sc_times, sc_spec = signal.spectrogram(
        np_data,
        fs=fs,
        nperseg=n_wl,
        nfft=2 ** int(np.ceil(np.log2(n_wl))),
        noverlap=n_wl - n_ts,
        window="hamming",
    )

    st_freqs = to_np(st_freqs)
    st_times = to_np(st_times)
    st_spec = to_np(st_spec)

    assert st_freqs.shape == sc_freqs.shape
    assert st_times.shape == sc_times.shape
    assert st_spec.shape == sc_spec.shape

    assert np.allclose(st_times, sc_times)
    assert np.allclose(st_freqs, sc_freqs)
    assert np.allclose(st_spec, sc_spec)
