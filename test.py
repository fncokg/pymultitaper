# %%
import numpy as np

# %%
# MATLAB test parameters
fs = 8000
duration_sec = 2
n_samples = fs * duration_sec
time_step = 0.05
window_length = time_step
nts = int(time_step * fs)
nwl = nts
nframes = n_samples // nwl
NW = 4
# %%
matlab_in = np.fromfile("tests/data/matlab_in.bin", dtype=np.float64)
matlab_out = np.fromfile("tests/data/matlab_out.bin", dtype=np.float64).reshape(
    -1, nframes, order="F"
)
half_idx = matlab_out.shape[0] // 2
matlab_out_eig = matlab_out[:half_idx, :]
matlab_out_uni = matlab_out[half_idx:, :]
# %%
from pymultitaper import multitaper_spectrogram

_, _, spec_eig = multitaper_spectrogram(
    matlab_in,
    fs=fs,
    time_step=time_step,
    window_length=window_length,
    NW=NW,
    db_scale=False,
    boundary_pad=False,
    detrend="off",
    weight_type="eig",
)

_, _, spec_uni = multitaper_spectrogram(
    matlab_in,
    fs=fs,
    time_step=time_step,
    window_length=window_length,
    NW=NW,
    db_scale=False,
    boundary_pad=False,
    detrend="off",
    weight_type="unity",
)
# %%
np.allclose(spec_eig, matlab_out_eig)
np.allclose(spec_uni, matlab_out_uni)
