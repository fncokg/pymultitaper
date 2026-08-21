# Introduction

`pymultitaper` is a fast and easy-to-use small package for multitaper spectrogram/spectrum calculation, as well as oridnary (single-taper) spectrogram calculation.

Main updates:

- `spectrogram` and `multitaper_spectrogram` accept arbitrary leading dimensions and keep them in the output.
- `batched_spectrogram` and `batched_multitaper_spectrogram` speed up batch processing, especially for many short signals and GPU workloads.
- The test suite covers batched consistency, `spectrogram` vs. SciPy, and `multitaper_spectrogram` vs. MATLAB.


# Installation

Install via pip:

```
pip install pymultitaper
```

# Usage

```python
>>> from pymultitaper import multitaper_spectrogram, plot_spectrogram
>>> from scipy.io import wavfile
>>> fs, data = wavfile.read('test.wav')
>>> freqs,times,psd = multitaper_spectrogram(
...     data, fs,time_step=0.001,window_length=0.005,NW=4
... )
>>> fig,ax = plot_spectrogram(times,freqs,psd,cmap="viridis")
```

Multidimensional inputs are supported. The last dimension is treated as time and preserved in the output:

```python
>>> import numpy as np
>>> fs = 1000
>>> data = np.random.randn(8, 2000)
>>> from pymultitaper import spectrogram
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

**GPU support requires `CuPy`**. For installation instructions, see the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html).

GPU usage is automatic: if you pass a CuPy array into the spectrogram functions, `pymultitaper` will use the GPU backend. NumPy arrays still use the CPU backend.

```python
>>> import cupy as cp
>>> from pymultitaper import multitaper_spectrogram, batched_multitaper_spectrogram
>>> from scipy.io import wavfile
>>> fs, data = wavfile.read("test.wav")
>>> data_gpu = cp.asarray(data)
>>> freqs, times, psd = multitaper_spectrogram(
...     data_gpu, fs, time_step=0.001, window_length=0.005, NW=4
... )
>>> psd.shape
(len(freqs), len(times))
```

The returned arrays stay on the GPU until you explicitly move them back to NumPy, for example with `cp.asnumpy(...)`, before plotting.

The batched helpers are especially useful on GPU because they reduce repeated small-kernel overhead:

```python
>>> signal_list = [cp.asarray(data[:200]), cp.asarray(data[:160]), cp.asarray(data[:240])]
>>> freqs, time_list, psd_list = batched_multitaper_spectrogram(signal_list, fs, time_step=0.01)
```

# Algorithm Consistency

The correctness of the algorithms is verified by automated tests:

- **`spectrogram` vs. SciPy**: Results of `pymultitaper.spectrogram` are compared against `scipy.signal.spectrogram` across various window sizes and detrending methods.
- **`multitaper_spectrogram` vs. MATLAB**: Results of `pymultitaper.multitaper_spectrogram` are validated against MATLAB's `pmtm` implementation.
- **`batched_xxx` consistency**: Batched variants produce identical results to their non-batched counterparts.

See the test code in the [tests/](https://github.com/fncokg/pymultitaper/tree/master/tests) folder, including [matlab_test.m](https://github.com/fncokg/pymultitaper/blob/master/tests/matlab_test.m) for the MATLAB reference implementation.

# Examples

![Comparions of multitaper spectrograms](https://github.com/fncokg/pymultitaper/blob/master/spectrogram.jpg?raw=true)

![Comparions of multitaper spectrums](https://github.com/fncokg/pymultitaper/blob/master/spectrum.jpg?raw=true)