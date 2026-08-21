import numpy as np
import pytest
from scipy import signal

from pymultitaper import spectrogram
from conftest import to_np, ts_wl_gen


@pytest.mark.parametrize("detrend", ["off", "constant", "linear"])
@pytest.mark.parametrize("time_step, window_length", ts_wl_gen())
def test_compare_scipy(detrend, time_step, window_length, sig):
    """Compare pymultitaper spectrogram with scipy.signal.spectrogram."""
    data, fs, xp = sig

    st_freqs, st_times, st_spec = spectrogram(
        data,
        fs=fs,
        time_step=time_step,
        window_length=window_length,
        db_scale=False,
        detrend=detrend,
        boundary_pad=False,
    )

    # compute scipy reference with numpy arrays
    n_ts = int(time_step * fs)
    n_wl = int(window_length * fs)
    np_data = to_np(data)
    sc_freqs, sc_times, sc_spec = signal.spectrogram(
        np_data,
        fs=fs,
        nperseg=n_wl,
        nfft=2 ** int(np.ceil(np.log2(n_wl))),
        noverlap=n_wl - n_ts,
        window="hamming",
        detrend=False if detrend == "off" else detrend,
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
