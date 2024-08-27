import sys
sys.path.append(r"D:\程序项目\PythonLibs\pymultitaper\src")
from pymultitaper import multitaper_spectrogram as mt_sp, spectrogram as sp, plot_spectrogram, plot_spectrum
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Load the data
fs, data = wavfile.read('examples/test.wav')
mt_times4,mt_freqs4,mt_spec4 = mt_sp(data, fs,time_step=0.001,window_length=0.005,NW=4)
mt_times2,mt_freqs2,mt_spec2 = mt_sp(data, fs,time_step=0.001,window_length=0.005,NW=2)
st_times,st_freqs,st_spec = sp(data, fs,time_step=0.001,window_length=0.005)

# Plot the spectrogram
f,ax = plt.subplots(3,1,figsize=(16,12))
plot_spectrogram(st_times,st_freqs,st_spec,ax=ax[0],cmap="viridis")
ax[0].set_title('Single-Taper Spectrogram')
plot_spectrogram(mt_times2,mt_freqs2,mt_spec2,ax=ax[1],cmap="viridis")
ax[1].set_title('Multitaper Spectrogram (NW=2)')
plot_spectrogram(mt_times4,mt_freqs4,mt_spec4,ax=ax[2],cmap="viridis")
ax[2].set_title('Multitaper Spectrogram (NW=4)')
f.tight_layout()
f.savefig('spectrogram.jpg',dpi=300)

# Plot the spectrum
f,ax = plt.subplots(1,1,figsize=(10,5))
plot_spectrum(st_times,st_freqs,st_spec,time=0.7,ax=ax,label="Single-Taper",linestyle='--')
plot_spectrum(mt_times2,mt_freqs2,mt_spec2,time=0.7,ax=ax,label="Multitaper (NW=2)")
plot_spectrum(mt_times4,mt_freqs4,mt_spec4,time=0.7,ax=ax,label="Multitaper (NW=4)")
ax.legend()
ax.set_title('Spectrum')
f.tight_layout()
f.savefig('spectrum.jpg',dpi=300)
