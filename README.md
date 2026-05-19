<p align="center">
  <img src="https://github.com/fncokg/pymultitaper/blob/master/spectrogram.jpg?raw=true" />
</p>

# pymultitaper


`pymultitaper` is a fast and easy-to-use small package for multitaper spectrogram/spectrum calculation, as well as oridnary (single-taper) spectrogram calculation.


# Installation

Install via pip:

```
pip install pymultitaper
```

GPU computation is supported in a pre-release version. To install the GPU-enabled version, use:

```bash
pip install --pre pymultitaper
```

**GPU support requires `CuPy`**. For installation instructions, see the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html).

*Note: `Cupy` is an optional dependency. If you do not have `CuPy` installed, `pymultitaper` will still work using the CPU backend.*

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

# GPU Support

GPU usage is automatic: if you pass a CuPy array into the spectrogram functions, `pymultitaper` will use the GPU backend. NumPy arrays still use the CPU backend.

```python
>>> import cupy as cp
>>> from pymultitaper import multitaper_spectrogram
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

# Documentation

Refer to [pymultitaper documentation](https://fncokg.github.io/pymultitaper/) for more details.

# Examples

![Comparions of multitaper spectrograms](https://github.com/fncokg/pymultitaper/blob/master/spectrogram.jpg)

![Comparions of multitaper spectrums](https://github.com/fncokg/pymultitaper/blob/master/spectrum.jpg)