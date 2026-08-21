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

For a list of short signals, use the batched helpers:

```python
>>> from pymultitaper import batched_spectrogram
>>> signal_list = [np.random.randn(200), np.random.randn(160), np.random.randn(240)]
>>> fs = 1000
>>> freqs, time_list, spec_list = batched_spectrogram(signal_list, fs, time_step=0.01)
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

The batched helpers are especially useful on GPU because they reduce repeated small-kernel overhead:

```python
>>> signal_list = [cp.asarray(data[:200]), cp.asarray(data[:160]), cp.asarray(data[:240])]
>>> freqs, time_list, psd_list = batched_multitaper_spectrogram(signal_list, fs, time_step=0.01)
```

`batched_multitaper_spectrogram` and `batched_spectrogram` accept a list of 1D signals of varying lengths and **pads them to a common size for efficient batch processing**.

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