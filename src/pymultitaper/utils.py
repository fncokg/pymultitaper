from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
from .backend import ArrayBackend


# TODO: we need the mask array of framed results, not of the raw signal.
def batch_signals(
    signal_list: list,
    padding_strategy: Literal["max", "max_length"] = "max",
    max_length: Optional[int] = None,
    return_mask: bool = False,
) -> NDArray:
    """
    Convert a list of 1D signals into a 2D array with padding.

    Args:
        signal_list (list): A list of 1D numpy or cupy arrays.
        padding_strategy (str): The strategy for padding. Options are:

            - "max": Pad all signals to the length of the longest signal in the list.
            - "max_length": Pad all signals to the specified `max_length`. If a signal is longer than `max_length`, it will be truncated.

        max_length (int, optional): The maximum length to pad/truncate signals to when using the "max_length" strategy. Required if `padding_strategy` is "max_length".
        return_mask (bool): If True, also return a boolean mask indicating the valid (non-padded) entries in the output array.

    Returns:
        NDArray: A 2D array where each row corresponds to a signal from `signal_list`, padded with zeros as necessary.
    """

    if padding_strategy == "max":
        max_len = max(signal.shape[0] for signal in signal_list)
    elif padding_strategy == "max_length":
        if max_length is None:
            raise ValueError(
                "`max_length` must be specified when using 'max_length' padding strategy."
            )
        max_len = max_length
    else:
        raise ValueError(f"Unsupported padding strategy: {padding_strategy}")

    # Determine the backend based on the first signal
    backend = ArrayBackend.like(signal_list[0])

    # Create an empty array with the appropriate shape and backend
    batched_array = backend.xp.zeros(
        (len(signal_list), max_len), dtype=signal_list[0].dtype
    )
    mask = backend.xp.zeros((len(signal_list), max_len), dtype=backend.xp.int8)

    for i, signal in enumerate(signal_list):
        length = min(signal.shape[0], max_len)
        batched_array[i, :length] = signal[:length]
        mask[i, :length] = 1

    if return_mask:
        return batched_array, mask
    return batched_array


def frame_masking(mask, fs, time_step, window_length=None, boundary_pad=False):
    window_length = window_length if window_length is not None else time_step
    n_tstep = int(time_step * fs)
    n_winlen = int(window_length * fs)
    backend = ArrayBackend.like(mask)
    if boundary_pad:
        n_pad = int(n_winlen / 2) + 1
        pad_width = [(0, 0)] * mask.ndim
        pad_width[-1] = (n_pad, n_pad)
        mask = backend.xp.pad(
            mask, pad_width=pad_width, mode="constant", constant_values=False
        )
    frames = backend.xp.lib.stride_tricks.sliding_window_view(
        mask, n_winlen, writeable=False, axis=-1
    )[..., ::n_tstep, :]
    frame_mask = backend.xp.empty_like(frames, dtype=backend.xp.int8)
    if boundary_pad:
        backend.xp.any(frames == 1, axis=-1, out=frame_mask)
    else:
        backend.xp.all(frames == 1, axis=-1, out=frame_mask)
    return frame_mask
