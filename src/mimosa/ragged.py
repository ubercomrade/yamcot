from typing import List

import numpy as np


class RaggedData:
    """
    Class for storing ragged (variable-length) arrays.

    Uses a flattened representation (data + offsets) for memory efficiency and fast access.
    This structure is particularly useful for storing sequences of different lengths
    without padding, saving memory and computation time.
    """

    def __init__(self, data: np.ndarray, offsets: np.ndarray):
        """Initialize the RaggedData object."""
        self.data = data
        self.offsets = offsets

    def get_length(self, i: int) -> int:
        """Return the length of the i-th sequence."""
        return self.offsets[i + 1] - self.offsets[i]

    def get_slice(self, i: int) -> np.ndarray:
        """Return a slice of data for the i-th sequence (view)."""
        return self.data[self.offsets[i] : self.offsets[i + 1]]

    def total_elements(self) -> int:
        """Return the total number of elements across all sequences."""
        return self.data.size

    @property
    def num_sequences(self) -> int:
        """Return the number of sequences."""
        return self.offsets.size - 1


def ragged_from_list(data_list: List[np.ndarray], dtype=None) -> RaggedData:
    """Create RaggedData from a list of numpy arrays."""
    if len(data_list) == 0:
        return RaggedData(np.empty(0, dtype=dtype if dtype else np.float32), np.zeros(1, dtype=np.int64))

    if dtype is None:
        dtype = data_list[0].dtype

    n = len(data_list)
    lengths = np.empty(n, dtype=np.int64)
    for i in range(n):
        lengths[i] = len(data_list[i])

    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    total_size = offsets[-1]
    data = np.empty(total_size, dtype=dtype)

    for i in range(n):
        data[offsets[i] : offsets[i + 1]] = data_list[i]

    return RaggedData(data, offsets)
