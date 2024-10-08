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

# Usage

```python
>>> from pymultitaper import multitaper_spectrogram, plot_spectrogram
>>> from scipy.io import wavfile
>>> fs, data = wavfile.read('test.wav')
>>> times,freqs,psd = multitaper_spectrogram(
...     data, fs,time_step=0.001,window_length=0.005,NW=4
... )
>>> fig,ax = plot_spectrogram(times,freqs,psd,cmap="viridis")
```

# Documentation

Refer to [pymultitaper documentation](https://fncokg.github.io/pymultitaper/) for more details.

# Examples

![Comparions of multitaper spectrograms](https://github.com/fncokg/pymultitaper/blob/master/spectrogram.jpg)

![Comparions of multitaper spectrums](https://github.com/fncokg/pymultitaper/blob/master/spectrum.jpg)