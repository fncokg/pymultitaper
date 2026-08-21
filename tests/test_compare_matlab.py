import numpy as np
from pymultitaper import multitaper_spectrogram


def test_compare_matlab():
    # MATLAB test parameters
    fs = 8000
    duration_sec = 2
    time_step = 0.05
    window_length = time_step
    NW = 4

    kwargs = dict(
        fs=fs,
        time_step=time_step,
        window_length=window_length,
        NW=NW,
        db_scale=False,
        boundary_pad=False,
        detrend="off",
    )

    n_samples = fs * duration_sec
    nframes = n_samples // int(window_length * fs)

    matlab_in = np.fromfile("tests/data/matlab_in.bin", dtype=np.float64)
    matlab_out = np.fromfile("tests/data/matlab_out.bin", dtype=np.float64).reshape(
        -1, nframes, order="F"
    )
    half_idx = matlab_out.shape[0] // 2
    matlab_out_eig = matlab_out[:half_idx, :]
    matlab_out_uni = matlab_out[half_idx:, :]

    _, _, spec_eig = multitaper_spectrogram(matlab_in, weight_type="eig", **kwargs)

    _, _, spec_uni = multitaper_spectrogram(matlab_in, weight_type="unity", **kwargs)

    assert np.allclose(spec_eig, matlab_out_eig)
    assert np.allclose(spec_uni, matlab_out_uni)
