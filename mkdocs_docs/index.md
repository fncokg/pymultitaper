# Introduction

`pymultitaper` is a fast and easy-to-use small package for multitaper spectrogram/spectrum calculation on both CPU and GPU, as well as oridnary (single-taper) spectrogram calculation.

# Installation

Install via pip:

```bash
pip install pymultitaper
```

# Usage

```python
>>> from pymultitaper import multitaper_spectrogram, plot_spectrogram
>>> fs = 1000
>>> data = np.random.randn(2000)
>>> freqs,times,psd = multitaper_spectrogram(
...     data, fs,time_step=0.001,window_length=0.005,NW=4
... )
>>> fig,ax = plot_spectrogram(times,freqs,psd,cmap="viridis")
```

Multidimensional inputs are supported. The last dimension is treated as time and the output preserves the leading dimensions:

```python
>>> import numpy as np
>>> from pymultitaper import spectrogram
>>> fs = 1000
>>> data = np.random.randn(8, 2000)
>>> freqs, times, psd = spectrogram(data, fs, time_step=0.01)
>>> psd.shape  # (8, n_freqs, n_frames)
```

# GPU Support

GPU usage is automatic: **CuPy arrays in, CuPy arrays out**. Just pass a CuPy array into the spectrogram functions, and `pymultitaper` will use the GPU backend.

```python
>>> import cupy as cp
>>> from pymultitaper import multitaper_spectrogram, batched_multitaper_spectrogram
>>> data_gpu = cp.random.randn(2000)
>>> fs = 1000
>>> freqs, times, psd = multitaper_spectrogram(
...     data_gpu, fs, time_step=0.001, window_length=0.005, NW=4
... )
```

When processing multiple signals (possibly of varying lengths), you can try the batched version of the spectrogram functions:

```python
>>> signal_list = [cp.asarray(data[:200]), cp.asarray(data[:160]), cp.asarray(data[:240])]
>>> freqs, time_list, psd_list = batched_multitaper_spectrogram(signal_list, fs, time_step=0.01, mode="chunk_batched")
```

You can control execution with `mode="auto" | "loop" | "batched" | "chunk_batched"` (default: `"auto"`):
- `auto`: uses `chunk_batched` on GPU and `loop` on CPU.
- `loop`: compute each signal independently.
- `batched`: pad all signals to a common length and run one batched call.
- `chunk_batched`: split signals into chunks of similar lengths, and batch-process each chunk to minimize padding overhead. Chunk size can be controlled with the `chunk_size` argument, which defaults to `n_signals // 10`.

**NOTE: GPU support requires `CuPy`, which is not included in the `pymultitaper` package**. For installation instructions, see the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html).

# Algorithm Consistency

The correctness of the algorithms is verified by automated tests:

- **`spectrogram` vs. SciPy**: Results of `pymultitaper.spectrogram` are compared against `scipy.signal.spectrogram` across various window sizes and detrending methods.
- **`multitaper_spectrogram` vs. MATLAB**: Results of `pymultitaper.multitaper_spectrogram` are validated against MATLAB's `pmtm` implementation.
- **`batched_xxx` consistency**: Batched variants produce identical results to their non-batched counterparts.

See the test code in the [tests/](https://github.com/fncokg/pymultitaper/tree/master/tests) folder, including [matlab_test.m](https://github.com/fncokg/pymultitaper/blob/master/tests/matlab_test.m) for the MATLAB reference implementation.

# Examples

![Comparions of multitaper spectrograms](https://github.com/fncokg/pymultitaper/blob/master/spectrogram.jpg?raw=true)

![Comparions of multitaper spectrums](https://github.com/fncokg/pymultitaper/blob/master/spectrum.jpg?raw=true)