import numpy as np
from typing import Self
from numcompute.utils import validate_numeric_array


class Statistics:
    def __init__(self):
        self.reset()

    @property
    def mean(self):
        return self._mean

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def count(self):
        return self._count

    @property
    def M2(self):
        return self._M2

    def add(self, x : int | float) -> Self:
        """
        Update all running statistics with a new observed value.

        Uses Welford's online algorithm to update mean and total deviation
        in a single pass, avoiding a second loop over the data.

        Parameters
        ----------
        x : int or float
            The new value to incorporate into the running statistics.

        Raises
        ------
        ValueError
            If ``x`` is ``None``.
        TypeError
            If ``x`` is not numeric.

        Complexity
        ----------
        Time Complexity:
            O(1)
        Space Complexity:
            O(1)
        """
        if x is None:
            raise ValueError("Please pass a numeric value.")

        if not isinstance(x, (int, float, np.number)):
            raise TypeError("Please pass an integer or float value.")

        return self.update_stats(np.array([[x]], dtype=float))

    def median(
        self,
        data: np.ndarray,
        axis: int | tuple = None,
        overwrite_input: bool = False,
        keepdims: bool = False
    ) -> np.ndarray:
        """
        Calculate the median of an array.

        Parameters
        ----------
        data : np.ndarray
            Input array of numeric values.
        axis : int, tuple of int, or None, optional
            Axis along which to compute the median.
            ``None`` (default) computes over the entire array.
        overwrite_input : bool, optional
            If ``True``, the input array may be modified to save memory.
            Default is ``False``.
        keepdims : bool, optional
            If ``True``, the reduced axis is retained as a dimension of size 1.
            Default is ``False``.

        Returns
        -------
        np.ndarray
            Median value(s) of the input array.

        Raises
        ------
        ValueError
            If ``data`` is ``None``.
            If ``data`` is not numeric, contains NaNs, infinite values,
            or complex numbers.

        Complexity
        ----------
        Time Complexity: O(n log n)
        Space Complexity: O(1).
        """
        if data is None:
            raise ValueError("Please pass a value")
        validate_numeric_array(data)
        return np.median(data, axis=axis,
                         overwrite_input=overwrite_input, keepdims=keepdims)

    def reset(self) -> Self:
        """
        Clear all accumulated running statistics.

        Returns
        -------
        Statistics
            The reset statistics instance.
        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        self._count = 0
        self._mean = None
        self._M2 = None
        self._min = None
        self._max = None

        return self

    def update_stats(self, X_chunk: np.ndarray) -> Self:
        """
        Incrementally update running statistics from a newly received chunk.

        Uses Welford's online algorithm to update the per-feature running mean
        and accumulated squared deviation without storing previously processed
        chunks. Per-feature minimum and maximum values are updated alongside
        the running statistics.

        Parameters
        ----------
        X_chunk : np.ndarray
            Incoming numeric data of shape (m, n), where m is the number of
            rows in the chunk and n is the number of features.

        Returns
        -------
        Statistics
            The updated statistics instance.

        Raises
        ------
        ValueError
            If ``X_chunk`` is not 2D.
            If ``X_chunk`` contains no rows.
            If ``X_chunk`` has a different number of features than previously
            processed chunks.
            If ``X_chunk`` contains ``None``, NaN, infinite, or complex values.
        TypeError
            If ``X_chunk`` is not a numeric array.

        Complexity
        ----------
        Time Complexity:
            O(m * n), where m is the number of rows and n is the number of
            features.
        Space Complexity:
            O(m * n) in the worst case for converting the incoming chunk to
            float64. The retained running state requires O(n) space.
        """
        X_chunk = np.asarray(X_chunk)

        if X_chunk.ndim != 2:
            raise ValueError(
                "X_chunk must have 2 dimensions. Reshape your array first."
            )

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk must contain at least one row.")

        validate_numeric_array(X_chunk)

        # Working with decimal value is more appropriate -> casting is required
        X_chunk = X_chunk.astype("float64", copy=False)

        n_features = X_chunk.shape[1]

        if self._mean is None:
            self._mean = np.zeros(n_features, dtype=float)
            self._M2 = np.zeros(n_features, dtype=float)
            self._min = np.full(n_features, np.inf, dtype=float)
            self._max = np.full(n_features, -np.inf, dtype=float)
        elif n_features != len(self._mean):
            raise ValueError(
                "Input chunk has a different number of features than "
                "previously processed chunks."
            )

        for row in X_chunk:
            self._count += 1
            self._min = np.minimum(self._min, row)
            self._max = np.maximum(self._max, row)

            delta_1 = row - self._mean
            self._mean += delta_1 / self._count
            delta_2 = row - self._mean

            self._M2 += delta_1 * delta_2

        return self

    def variance(self, ddof: int = 0) -> np.ndarray:
        """
        Return the per-feature variance from the accumulated statistics.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom. Use ``0`` for population variance and
            ``1`` for sample variance. Default is 0.

        Returns
        -------
        np.ndarray
            Per-feature variance values.

        Raises
        ------
        ValueError
            If no values have been accumulated.
            If ``ddof`` is negative.
            If ``count - ddof`` is not greater than zero.

        Complexity
        ----------
        Time Complexity:
            O(n), where n is the number of features.
        Space Complexity:
            O(n) for the returned array.
        """
        if self._count == 0 or self._M2 is None:
            raise ValueError("No statistics have been accumulated yet.")

        if ddof < 0:
            raise ValueError("`ddof` must be non-negative.")

        denominator = self._count - ddof

        if denominator <= 0 :
            raise ValueError(
                "Variance cannot be calculated because `count - ddof` "
                "must be greater than zero."
            )

        return self._M2 / denominator

    def std_dev(self, ddof: int = 0) -> np.ndarray:
        """
        Return the per-feature standard deviation.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom passed to ``variance()``.
            Use ``0`` for population standard deviation and ``1`` for sample
            standard deviation. Default is 0.

        Returns
        -------
        np.ndarray
            Per-feature standard deviation values.

        Raises
        ------
        ValueError
            If variance cannot be calculated for the requested ``ddof``.

        Complexity
        ----------
        Time Complexity:
            O(n), where n is the number of features.
        Space Complexity:
            O(n) for the returned array.
        """
        return np.sqrt(self.variance(ddof=ddof))


def histogram(
    a: np.ndarray,
    bins: int | np.ndarray | str = 10,
    range: tuple = None,
    density: bool = None,
    weights: np.ndarray = None
) -> np.ndarray:
    """
    Compute a histogram of the data.

    Parameters
    ----------
    a : np.ndarray
        Input array of values to bin.
    bins : int, np.ndarray, or str, optional
        Number of equal-width bins (int), explicit bin edges (array), or a
        string estimator such as ``'auto'``, ``'sturges'``, ``'fd'``,
        ``'scott'``, or ``'sqrt'``. Default is 10.
    range : tuple of float, optional
        Lower and upper bounds of the bins as ``(min, max)``.
        Values outside this range are ignored.
    density : bool, optional
        If ``True``, the result is the probability
        density function for each bin.
        Default is ``False``.
    weights : np.ndarray, optional
        Array of weights, same shape as ``a``, assigning each value a custom
        contribution instead of counting as 1.

    Returns
    -------
    hist : np.ndarray
        Frequency (or density) count for each bin.
    bin_edges : np.ndarray
        Bin edge values of length ``len(hist) + 1``.

    Raises
    ------
    ValueError
        If ``a`` is not numeric, contains NaNs,
        infinite values, or complex numbers.

    Complexity
    ----------
    Time Complexity: O(n) int bins, O(n log k) array bins
    Space Complexity: O(k) where k is the number of bins.
    """

    validate_numeric_array(a)
    if isinstance(bins, np.ndarray):
        validate_numeric_array(bins)
    if weights is not None:
        validate_numeric_array(weights)
    return np.histogram(a, bins=bins, range=range,
                        density=density, weights=weights)


def quantile(
    a: np.ndarray,
    q: np.ndarray,
    axis: int | tuple = None,
    out: np.ndarray = None,
    overwrite_input: bool = None,
    method: str = "linear",
    keepdims: bool = False,
    weights: np.ndarray = None
) -> np.ndarray:
    """
    Compute quantiles of an array.

    Parameters
    ----------
    a : np.ndarray
        Input array of real numbers.
    q : float or np.ndarray
        Quantile(s) to compute, in the range [0, 1].
    axis : int, tuple of int, or None, optional
        Axis along which to compute quantiles. ``None`` (default) flattens
        the array before computing.
    out : np.ndarray, optional
        Output array with the same shape as the expected result.
    overwrite_input : bool, optional
        If ``True``, the input array ``a`` may be modified to save memory.
    method : str, optional
        Interpolation method when the quantile falls between data points.
        Supported values: ``'inverted_cdf'``, ``'averaged_inverted_cdf'``,
        ``'closest_observation'``, ``'lower'``, ``'higher'``, ``'nearest'``,
        ``'linear'`` (default), ``'midpoint'``, ``'weibull'``,
        ``'median_unbiased'``, ``'normal_unbiased'``.
    keepdims : bool, optional
        If ``True``, the reduced axes are retained as dimensions of size 1.
        Default is ``False``.
    weights : np.ndarray, optional
        Weights associated with the values in ``a``. Only supported by
        certain methods (e.g. ``'inverted_cdf'``).

    Returns
    -------
    np.ndarray
        Scalar if ``q`` is a single float; array otherwise.

    Raises
    ------
    ValueError
        If ``a`` is not numeric, contains NaNs,
        infinite values, or complex numbers.

    Complexity
    ----------
    Time Complexity: O(n), where n is the number of elements in ``a``.
    Space Complexity: O(1).
    """
    validate_numeric_array(a)
    if isinstance(q, np.ndarray):
        validate_numeric_array(q)
    if out is not None:
        validate_numeric_array(out)
    if weights is not None:
        validate_numeric_array(weights)
    return np.quantile(
        a, q, axis=axis, out=out, overwrite_input=overwrite_input,
        method=method, keepdims=keepdims, weights=weights
    )
