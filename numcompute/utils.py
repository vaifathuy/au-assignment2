import numpy as np


def validate_numeric_array(data: np.ndarray):
    """
    Validate that an array is numeric and contains no problematic values.

    Parameters
    ----------
    data : np.ndarray
        Array to validate.

    Raises
    ------
    ValueError
        If ``data`` contains ``None`` values.
        If ``data`` contains NaN values.
        If ``data`` contains infinite values.
        If ``data`` contains complex values.
    TypeError
        If ``data`` is not a numeric array.

    Complexity
    ----------
    Time Complexity: O(n) where n is the total number of elements in the array.
    Space Complexity: O(1)
    """

    if (data == None).any():  # noqa: E711
        raise ValueError(
            "input contains None(s). Fix these then try again."
        )

    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("input must be a numeric array.")

    if np.isnan(data).any():
        raise ValueError(
            "input contains NaN(s). Use the `SimpleImputer` then try again."
        )

    if np.isinf(data).any():
        raise ValueError(
            "input contains infinite values. Fix these then try again."
        )

    if np.iscomplexobj(data):
        raise ValueError(
            "input contains complex values. Fix these then try again."
        )
