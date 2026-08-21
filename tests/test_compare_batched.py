import pytest

from conftest import ts_wl_gen

from pymultitaper import (
    batched_spectrogram,
    spectrogram,
    batched_multitaper_spectrogram,
    multitaper_spectrogram,
)


@pytest.mark.parametrize(
    "func_pair",
    [
        (spectrogram, batched_spectrogram),
        (multitaper_spectrogram, batched_multitaper_spectrogram),
    ],
    ids=lambda v: f"{v[0].__name__}_vs_{v[1].__name__}",
)
@pytest.mark.parametrize("detrend", ["off", "constant", "linear"])
@pytest.mark.parametrize(
    "boundary_pad", [True, False], ids=lambda v: "padded" if v else ""
)
@pytest.mark.parametrize("time_step, window_length", ts_wl_gen())
@pytest.mark.parametrize(
    "mode", ["loop", "batched", "chunk_batched"], ids=lambda v: f"mode_{v}"
)
def test_compare_batched_spectrogram(
    func_pair,
    detrend,
    boundary_pad,
    time_step,
    window_length,
    mode,
    sig_list,
):
    """Test that batched_spectrogram matches individual spectrogram calls."""
    # Generate white noise signal list
    data_list, fs, xp = sig_list

    kwargs = dict(
        fs=fs,
        time_step=time_step,
        window_length=window_length,
        db_scale=False,
        detrend=detrend,
        boundary_pad=boundary_pad,
    )

    single_func, batch_func = func_pair

    b_freqs, b_time_list, b_spec_list = batch_func(data_list, mode=mode, **kwargs)

    time_list, spec_list = [], []
    for data in data_list:
        freqs, times, spec = single_func(data, **kwargs)
        time_list.append(times)
        spec_list.append(spec)

    assert xp.allclose(b_freqs, freqs)
    for t1, t2 in zip(b_time_list, time_list):
        assert xp.allclose(t1, t2)
    for s1, s2 in zip(b_spec_list, spec_list):
        assert xp.allclose(s1, s2)
