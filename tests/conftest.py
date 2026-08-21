import importlib
import pytest


def _get_backends():
    backends = ["numpy"]
    try:
        import cupy as cp

        backends.append("cupy")
    except Exception:
        pass
    return backends


@pytest.fixture(params=_get_backends(), ids=lambda v: v)
def xp(request):
    if request.param == "cupy":
        import cupy as xp
    else:
        import numpy as xp
    return xp


def to_np(a):
    """Convert array from xp backend (numpy/cupy) to numpy for comparison."""
    try:
        return a.get()
    except AttributeError:
        return a


# ============================================================================
# Signal generation parameters and fixtures
# ============================================================================


@pytest.fixture(params=[8000, 16000], ids=lambda v: f"fs_{v}")
def sample_rate(request):
    """Parametrized sampling frequency fixture."""
    return request.param


def ts_wl_gen():
    params = []
    for ts, wl in [(0.05, 0.02), (0.1, 0.1), (0.2, 0.15)]:
        params.append(pytest.param(ts, wl, id=f"ts_{ts}_wl_{wl}"))
    return params


# Default test parameters
TEST_PARAMS = {"default_duration": 2, "default_signal_count": 10}  # 2 seconds


def white_noise_signal_gen(xp, fs, duration):
    rng = xp.random.default_rng(42)
    n_samples = int(duration * fs)
    return rng.standard_normal(size=n_samples)


@pytest.fixture
def sig(xp, sample_rate):
    return (
        white_noise_signal_gen(xp, sample_rate, TEST_PARAMS["default_duration"]),
        sample_rate,
        xp,
    )


@pytest.fixture
def sig_list(xp, sample_rate):
    rng = xp.random.default_rng(42)
    durations = rng.integers(10, 30, size=TEST_PARAMS["default_signal_count"]) / 10
    return (
        [white_noise_signal_gen(xp, sample_rate, duration) for duration in durations],
        sample_rate,
        xp,
    )
