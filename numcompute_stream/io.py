"""
Load CSV files using numpy.loadtxt or numpy.genfromtxt
Support custom delimiters (e.g., comma, tab)
Handle missing values (skip or fill with a placeholder value)
Return data as NumPy arrays
"""

import numpy as np
import os
import csv


def load_csv(
    filename: str,
    dtype: tuple | None = None,
    delimiter: str = ',',
    skip_header: bool = True
) -> np.ndarray:
    """
    Load data from a text file with a specified delimiter.

    Parameters
    ----------
    filename : str
        Full path to the file to be loaded.
    dtype : tuple or None
        Possible data types representing the loaded data.
    delimiter : str, optional
        String used to split fields in the file. Default is ','.

    Returns
    -------
    np.ndarray
        2D array of shape (m, n) for tabular data, or
        1D array of shape (n,) for single-column data.

    Raises
    ------
    FileNotFoundError
        If the specified file could not be found.
    FileExistsError
        If the specified file exists but is empty.

    Complexity
    ----------
    Time Complexity: O(1) import-time symbol binding.
    Space Complexity: O(1) references only, no data allocation.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"'{filename}' cannot be found")

    if os.path.getsize(filename) <= 0:
        raise FileExistsError(f"{filename} is empty")
    else:
        try:
            return _load_csv_with_genfromtxt(
                filename, delimiter, dtype, skip_header=skip_header)
        except ValueError:
            return _clean_and_load_csv(
                filename, delimiter, dtype, skip_header=skip_header)


def _load_csv_with_genfromtxt(
        filename, delimiter, dtype, skip_header) -> np.ndarray:
    try:
        return np.genfromtxt(
            fname=filename, delimiter=delimiter, dtype=dtype,
            filling_values=0, skip_header=skip_header)
    except IOError:
        return np.array([])


def _clean_and_load_csv(filename, delimiter, dtype, skip_header) -> list[str]:
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        lines = list(map(lambda line: "*".join(line), list(reader)))
        if skip_header:
            del lines[0]
        return np.genfromtxt(
            fname=lines, delimiter='*', dtype=dtype)
