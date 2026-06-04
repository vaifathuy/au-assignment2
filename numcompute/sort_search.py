import numpy as np


def quickselect(
    data: np.ndarray,
    k: int,
    largest: bool = False
) -> int | float:
    """
    Select the k-th smallest (or largest) element using partial sorting.

    Uses ``np.partition`` to place the target element in its sorted position
    without fully sorting the array.

    Parameters
    ----------
    data : np.ndarray
        1D input array of numeric values, shape (n,).
    k : int
        Zero-based rank of the element to select.
        ``k=0`` returns the smallest element when ``largest=False``.
    largest : bool, optional
        If ``True``, selects the k-th largest element instead.
        Default is ``False``.

    Returns
    -------
    int or float
        The k-th smallest or k-th largest value from the array.

    Raises
    ------
    ValueError
        If ``data`` is not 1D.
        If ``data`` is empty.
    IndexError
        If ``k`` is out of bounds.

    Complexity
    ----------
    Time Complexity: O(n) average; O(n log n) worst case.
    Space Complexity: O(n) due to the internal copy made by ``np.partition``.

    Notes
    -----
    The operation is not stable; relative order
    of equal elements is not preserved.
    """

    data = np.asarray(data)

    n = data.size

    if data.ndim != 1:
        raise ValueError("Input data must be 1D")
    if n == 0:
        raise ValueError("Input array is empty")

    if k < 0 or k >= len(data):
        raise IndexError(f"k={k} is out of bounds")

    if largest:
        k = n - 1 - k

    return np.partition(data, k)[k]


def topk(
    data: np.ndarray,
    k: int,
    largest: bool = True,
    return_indices: bool = True
) -> np.ndarray:
    """
    Return the top-k elements (largest or smallest) in sorted order.

    Parameters
    ----------
    data : np.ndarray
        1D input array, shape (n,).
    k : int
        Number of elements to return. Clamped to ``len(data)`` if larger.
    largest : bool, optional
        If ``True`` (default), returns the k largest elements
            in descending order.
        If ``False``, returns the k smallest elements in ascending order.
    return_indices : bool, optional
        If ``True`` (default), returns indices into ``data``.
        If ``False``, returns the element values directly.

    Returns
    -------
    np.ndarray
        Indices or values of the top-k elements.

    Raises
    ------
    ValueError
        If ``data`` is not 1D.

    Complexity
    ----------
    Time Complexity: O(n + k log k)
    Space Complexity: O(k)

    Notes
    -----
    Ordering among duplicate elements is not stable due to ``np.argpartition``.
    """
    arr = np.asarray(data)

    if arr.ndim != 1:
        raise ValueError('data must be a 1D array.')

    if k <= 0:
        return np.array([], dtype=int)

    k = min(k, len(arr))

    # Isolate top-k partition (O(n))
    if largest:
        idx = np.argpartition(arr, -k)[-k:]
    else:
        idx = np.argpartition(arr, k - 1)[:k]

    # Sort the subset ascending (O(k log k))
    idx = idx[np.argsort(arr[idx], kind='stable')]

    # Reverse if we want largest (descending)
    if largest:
        idx = idx[::-1]

    return idx if return_indices else arr[idx]


def binary_search(sorted_array: list | np.ndarray,
                  x: int | float) -> tuple[int, bool]:
    """
    Search for a value in a sorted array using binary search.

    Parameters
    ----------
    sorted_array : list or np.ndarray
        1D sorted array of shape (n,).
    x : int or float
        Value to search for.

    Returns
    -------
    insertion_index : int
        Position at which ``x`` would be inserted to maintain sort order.
        Range: [0, n].
    exists : bool
        ``True`` if ``x`` is present in ``sorted_array``, ``False`` otherwise.

    Complexity
    ----------
    Time Complexity: O(log n), delegated to ``np.searchsorted``.
    Space Complexity: O(1).
    """
    sorted_array = np.asarray(sorted_array)
    idx = np.searchsorted(sorted_array, x)
    exists = bool(idx < len(sorted_array) and sorted_array[idx] == x)
    return idx, exists


def stable_sort(data: np.ndarray, axis: int = 0,
                order: str | list[str] = None) -> np.ndarray:
    """
    Return a stable sorted copy of an array.

    Parameters
    ----------
    data : np.ndarray
        Input array to sort. Can be 1D or n-dimensional.
    axis : int, optional
        Axis along which to sort. Default is 0 (column-wise).
    order : str or list of str, optional
        Field(s) to sort by when ``data`` is a structured array.

    Returns
    -------
    np.ndarray
        Sorted array with the same shape as ``data``.

    Raises
    ------
    np.exceptions.AxisError
        If ``axis`` is out of bounds for the array.

    Complexity
    ----------
    Time Complexity: O(n log n).
    Space Complexity: O(n) for the output array.
    """
    return np.sort(data, axis=axis, kind='stable', order=order)


def multi_key_sort(data: np.ndarray, keys: list[int],
                   ascending: bool | list[bool] = True) -> np.ndarray:
    """
    Sort a 2D array by multiple columns in priority order.

    Applies stable argsort iteratively in reverse key order so that the first
    key in ``keys`` has the highest sort priority.

    Parameters
    ----------
    data : np.ndarray
        2D input array of shape (n_rows, n_cols) where each row is a record.
    keys : list of int
        Column indices to sort by, ordered from highest to lowest priority.
    ascending : bool or list of bool, optional
        Sort direction per key. ``True`` for ascending,
            ``False`` for descending.
        A single ``bool`` applies to all keys. Default is ``True``.

    Returns
    -------
    np.ndarray
        2D array sorted by the specified keys, shape (n_rows, n_cols).

    Raises
    ------
    ValueError
        If ``data`` is not a 2D array.
    IndexError
        If any key index is out of bounds for the column count.

    Complexity
    ----------
    Time Complexity:
        O(k * n log n), where n is the number of rows and k the number of keys.
    Space Complexity: O(n) for the index array.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError('data must be a 2D array.')

    if not isinstance(keys, list):
        keys = [keys]

    if keys and (max(keys) >= data.shape[1]):
        raise IndexError(f"key index {max(keys)} is out of bounds")

    if not isinstance(ascending, list):
        ascending_list = [ascending] * len(keys)
    else:
        ascending_list = ascending

    for key, asc in reversed(list(zip(keys, ascending_list))):
        order = 1 if asc else -1
        data = data[np.argsort(data[:, key] * order, kind='stable')]

    return data
