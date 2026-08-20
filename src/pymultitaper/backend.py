from typing import Literal
import warnings

# CPU backend
import numpy as np
import scipy.signal as sci_signal
import scipy.fft as sci_fft

# GPU backend
try:
    import cupy as cp

    # `import cupyx.scipy.signal` raises a FutureWarning.
    # See [Issue #8718](https://github.com/cupy/cupy/issues/8718) of `cupy` for details.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import cupyx.scipy.signal as cp_signal

    import cupyx.scipy.fft as cp_fft

    def _cp_dpss_windows(*args, **kwargs):
        # cupyx does not have dpss implementation
        # currently, we compute the dpss windows on CPU and transfer to GPU, which is not very efficient, but should be fine for most use cases since the number of tapers is usually small
        tapers, eigns = sci_signal.windows.dpss(*args, **kwargs)
        return cp.asarray(tapers), cp.asarray(eigns)

    cp_signal.windows.dpss = _cp_dpss_windows
except ImportError:
    cp = None
    cp_signal = None
    cp_fft = None


class ArrayBackend:
    def __init__(self, backend: Literal["numpy", "cupy"]):
        self.backend = backend
        if backend == "numpy":
            self.xp = np
            self.signal = sci_signal
            self.fft = sci_fft
        elif backend == "cupy":
            if cp is None:
                raise ImportError(
                    "cupy is not installed, so the GPU backend is unavailable"
                )
            self.xp = cp
            self.signal = cp_signal
            self.fft = cp_fft
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def like(arr):
        if isinstance(arr, np.ndarray):
            return ArrayBackend("numpy")
        elif cp is not None and isinstance(arr, cp.ndarray):
            return ArrayBackend("cupy")
        else:
            raise ValueError(f"Unsupported array type: {type(arr)}")

    def __eq__(self, other):
        return self.backend == other.backend


# A decorator letting the function accept an additional `like` argument to specify the backend based on the input array type.
def backend_like(func):
    def wrapper(*args, like=None, **kwargs):
        backend = ArrayBackend.like(like) if like is not None else ArrayBackend("numpy")
        return func(*args, backend=backend, **kwargs)

    return wrapper
