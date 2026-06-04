import numpy as np


def rank(data: np.ndarray, method: str):
    """
    Rank values column-wise using one of three supported strategies.

    Parameters
    ----------
    data : np.ndarray
        2D input array of shape (m, n) to be ranked.
    method : {'average', 'dense', 'ordinal'}
        Ranking strategy.

        - ``'average'``: tied values receive the mean of their would-be ranks.
        - ``'dense'``: tied values receive the same rank; no gaps in ranking.
        - ``'ordinal'``: tied values receive consecutive distinct ranks.

    Returns
    -------
    np.ndarray
        Array of ranks with the same shape as ``data``.

    Raises
    ------
    ValueError
        If ``data`` is not an ``np.ndarray``.
        If ``data`` is not a 2D array.
        If ``method`` is not a supported strategy.

    Complexity
    ----------
    Time Complexity: O(m log n * n) due to sorting each column.
    Space Complexity: O(m * n) for the output array of ranks.
    Where m is the number of rows and n is the number of columns.
    """
    if type(data) is not np.ndarray:
        raise ValueError("Data is not a type of ndarray[]")

    if data.ndim != 2:
        raise ValueError("Data must be a 2D array")

    if method == 'average':
        return np.apply_along_axis(_average_rank, axis=0, arr=data)
    elif method == 'dense':
        return np.apply_along_axis(_dense_rank, axis=0, arr=data)
    elif method == 'ordinal':
        return np.apply_along_axis(_ordinal_rank, axis=0, arr=data)
    else:
        raise ValueError("Ranking method not supported")


def _average_rank(x):

    sorted_indices = x.argsort()
    ranks = np.empty_like(sorted_indices, dtype=float)
    ranks[sorted_indices] = np.arange(1, len(x) + 1)

    # handle ties - duplicate values
    values, inverse, counts = np.unique(
        x, return_counts=True, return_inverse=True)
    dup_groups = np.where(counts > 1)[0]

    for g in dup_groups:
        indices = np.where(inverse == g)[0]
        rank_avg = sum(ranks[indices]) / len(indices)
        # Replace rank_avg to duplicate positions
        ranks[indices] = rank_avg
    return ranks


def _dense_rank(x):
    unique_vals = np.unique(x)
    rank_map = {val: i + 1 for i, val in enumerate(unique_vals)}
    out = np.array([rank_map[el] for el in x])
    return out


def _ordinal_rank(x):
    return np.argsort(np.argsort(x)) + 1


def percentile(data: np.ndarray, q: int, interpolation: str = 'linear'):
    """
    Calculate percentile values column-wise.

    Computes the position corresponding to quantile ``q`` and applies the
    selected interpolation strategy when that position falls between two
    data points.

    Parameters
    ----------
    data : np.ndarray
        2D input array of shape (m, n).
    q : int
        Percentile as a whole number in the range (0, 100], e.g. 25, 50, 90.
    interpolation : {'linear', 'lower', 'upper', 'midpoint'}, optional
        Strategy for values falling between two points.

        - ``'linear'``: weighted average proportional to distance from bounds.
        - ``'lower'``: use the lower bound value.
        - ``'upper'``: use the upper bound value.
        - ``'midpoint'``: average of the upper and lower bound values.

        Default is 'linear'.

    Returns
    -------
    np.ndarray
        Percentile value for each column, shape (n,).

    Raises
    ------
    ValueError
        If ``data`` is not an ``np.ndarray``.
        If ``data`` is not 2D.
        If ``data`` contains NaN or infinite values.
        If ``q`` is not a whole number.
        If ``q`` is not greater than 0.
        If ``interpolation`` is not a supported strategy.

    Time Complexity: O(m log m * n)
    Space Complexity: O(m) for
    """
    if type(data) is not np.ndarray:
        raise ValueError("data is not a type of ndarray[]")

    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    if np.isnan(data).any():
        raise ValueError("data contains NaN value(s)")

    if np.isinf(data).any():
        raise ValueError("data contains infinite value(s)")

    if not isinstance(q, int):
        raise TypeError("q must be passed as a whole number")

    if q <= 0:
        raise ValueError("q must be greater than 0")

    return np.apply_along_axis(
        lambda x: __percentile_1d(x, q, interpolation),
        axis=0,
        arr=data
    )


def __percentile_1d(x, q, interpolation):
    data = np.sort(np.asarray(x))
    n = len(x)
    pos = (q / 100) * (n - 1)
    lower_bound = int(np.floor(pos))
    upper_bound = int(np.ceil(pos))

    if interpolation == 'linear':
        if lower_bound == upper_bound:
            return data[lower_bound]
        weight = pos - lower_bound
        return data[lower_bound] * (1 - weight) + data[upper_bound] * weight
    elif interpolation == 'lower':
        return data[lower_bound]
    elif interpolation == 'upper':
        return data[upper_bound]
    elif interpolation == 'midpoint':
        return (data[lower_bound] + data[upper_bound]) / 2
    else:
        raise ValueError("interpolation strategy not supported")
