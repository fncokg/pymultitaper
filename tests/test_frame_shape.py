import numpy as np
import pytest

from pymultitaper import spectrogram, multitaper_spectrogram
from conftest import to_np, TEST_PARAMS


def nts_nwl_gen(count):
    """Generate random time step and window length values."""
    rng = np.random.default_rng(42)
    nts_values = rng.integers(10, 1000, size=count).tolist()
    nwl_values = rng.integers(10, 1000, size=count).tolist()
    params = []
    for nts, nwl in zip(nts_values, nwl_values):
        params.append(pytest.param(nts, nwl, id=f"nts_{nts}_nwl_{nwl}"))
    return params


@pytest.mark.parametrize("nts, nwl", nts_nwl_gen(10))
def test_check_nframe(sig, nts, nwl):
    """Test that frame times match expected values for various time steps."""

    data, fs, xp = sig
    duration_sec = data.shape[0] / fs

    ts = nts / fs
    wl = nwl / fs
    desired_times = np.arange(0, duration_sec, ts)
    _, times, _ = spectrogram(
        data,
        fs=fs,
        time_step=ts,
        window_length=wl,
        db_scale=False,
        boundary_pad=True,
    )

    _, times_mt, _ = multitaper_spectrogram(
        data,
        fs=fs,
        time_step=ts,
        window_length=wl,
        db_scale=False,
        boundary_pad=True,
    )

    times = to_np(times)
    times_mt = to_np(times_mt)

    assert times.shape == desired_times.shape
    assert np.allclose(times, desired_times)
    assert times_mt.shape == desired_times.shape
    assert np.allclose(times_mt, desired_times)
