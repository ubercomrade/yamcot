from __future__ import annotations

from typing import List

import numpy as np


def _default_pad_value(dtype: np.dtype) -> int | float:
    """Return a dtype-compatible default padding value."""
    return 0.0 if np.issubdtype(dtype, np.floating) else 0


class MatrixData:
    """Matrix-backed variable-length data container."""

    def __init__(self, matrix: np.ndarray, lengths: np.ndarray, pad_value: int | float | None = None):
        values = np.asarray(matrix)
        lengths_arr = np.asarray(lengths)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.ndim != 2:
            raise ValueError("matrix must be a two-dimensional array")

        lengths_arr = np.asarray(lengths_arr, dtype=np.int64)
        if lengths_arr.ndim != 1:
            raise ValueError("lengths must be a one-dimensional array")
        if values.shape[0] != lengths_arr.size:
            raise ValueError("matrix rows must match the number of lengths")
        if np.any(lengths_arr < 0):
            raise ValueError("lengths must be non-negative")
        if values.shape[1] and np.any(lengths_arr > values.shape[1]):
            raise ValueError("row lengths cannot exceed matrix width")

        self.matrix = np.ascontiguousarray(values)
        self.lengths = np.ascontiguousarray(lengths_arr, dtype=np.int64)
        self.pad_value = _default_pad_value(self.matrix.dtype) if pad_value is None else pad_value

    def get_length(self, i: int) -> int:
        """Return the length of the i-th row."""
        return int(self.lengths[i])

    def get_slice(self, i: int) -> np.ndarray:
        """Return the valid prefix of the i-th row."""
        length = int(self.lengths[i])
        return self.matrix[i, :length]

    def total_elements(self) -> int:
        """Return the number of valid elements across all rows."""
        return int(np.sum(self.lengths, dtype=np.int64))

    @property
    def num_sequences(self) -> int:
        """Return the number of rows."""
        return int(self.lengths.size)

    @property
    def width(self) -> int:
        """Return the padded matrix width."""
        return int(self.matrix.shape[1]) if self.matrix.ndim == 2 else 0

    def valid_mask(self) -> np.ndarray:
        """Return a boolean mask of valid elements."""
        if self.width == 0 or self.num_sequences == 0:
            return np.zeros(self.matrix.shape, dtype=bool)
        return np.arange(self.width, dtype=np.int64)[None, :] < self.lengths[:, None]

    def flatten_valid(self) -> np.ndarray:
        """Flatten valid values row by row."""
        if self.num_sequences == 0 or self.width == 0:
            return np.empty(0, dtype=self.matrix.dtype)
        return self.matrix[self.valid_mask()]

    def astype(self, dtype: np.dtype) -> "MatrixData":
        """Return a copy with a new dtype."""
        return MatrixData(self.matrix.astype(dtype, copy=False), self.lengths.copy(), pad_value=self.pad_value)


def matrix_from_list(data_list: List[np.ndarray], dtype=None, pad_value: int | float | None = None) -> MatrixData:
    """Create MatrixData from a list of one-dimensional arrays."""
    if len(data_list) == 0:
        target_dtype = np.dtype(dtype if dtype is not None else np.float32)
        empty = np.empty((0, 0), dtype=target_dtype)
        return MatrixData(empty, np.zeros(0, dtype=np.int64), pad_value=_default_pad_value(target_dtype))

    if dtype is None:
        dtype = data_list[0].dtype

    dtype = np.dtype(dtype)
    lengths = np.array([len(row) for row in data_list], dtype=np.int64)
    width = int(lengths.max(initial=0))
    fill = _default_pad_value(dtype) if pad_value is None else pad_value
    matrix = np.full((len(data_list), width), fill, dtype=dtype)

    for i, row in enumerate(data_list):
        row_array = np.asarray(row, dtype=dtype)
        row_length = row_array.shape[0]
        if row_length:
            matrix[i, :row_length] = row_array

    return MatrixData(matrix, lengths, pad_value=fill)
