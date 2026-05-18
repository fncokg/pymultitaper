import importlib
import pytest


def _get_backends():
    backends = ["numpy"]
    try:
        import cupy as cp  # type: ignore

        backends.append("cupy")
    except Exception:
        pass
    return backends


@pytest.fixture(params=_get_backends(), ids=lambda v: v)
def xp(request):
    if request.param == "cupy":
        import cupy as xp  # type: ignore
    else:
        import numpy as xp
    return xp


@pytest.fixture(autouse=True)
def deterministic_random(monkeypatch, xp):
    try:
        rs = xp.random.RandomState(9876)
    except Exception:
        import numpy as _np

        rs = _np.random.RandomState(9876)
    monkeypatch.setattr(xp, "random", rs, raising=False)
