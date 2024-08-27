
# pymultitaper

`pymultitaper` is a fast and easy-to-use small package for multitaper spectrogram/spectrum calculation, as well as oridnary (single-taper) spectrogram calculation.

# Installation

Install via pip:

```
pip install pymultitaper
```

# Quickstart

```python3
>>> from pymultitaper import multitaper_spectrogram as mt_sp, plot_spectrogram
>>> from scipy.io import wavfile
>>> fs, data = wavfile.read('test.wav')
>>> mt_times,mt_freqs,mt_spec = mt_sp(data, fs,time_step=0.001,window_length=0.005,NW=4)
>>> fig,ax = plot_spectrogram(mt_times,mt_freqs,mt_spec,cmap="viridis")
```

# Documentation

