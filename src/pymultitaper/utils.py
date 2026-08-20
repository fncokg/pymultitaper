from typing import Literal, Optional
import numpy as np
from numpy.typing import NDArray
from .backend import ArrayBackend


def batched_signal(
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

    Returns:
        NDArray: A 2D array where each row corresponds to a signal from `signal_list`, padded with zeros as necessary.
    """
    backend = ArrayBackend.like(signal_list[0])
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
    mask = backend.xp.zeros((len(signal_list), max_len), dtype=bool)

    for i, signal in enumerate(signal_list):
        length = min(signal.shape[0], max_len)
        batched_array[i, :length] = signal[:length]
        mask[i, :length] = True

    if return_mask:
        return batched_array, mask
    return batched_array
