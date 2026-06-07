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


# Existing one-off histogram function
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


# Existing one-off quantile function
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


# Streaming Histogram
class Histogram:
    """
    Incrementally accumulate histogram counts from incoming numeric chunks.

    It retains fixed bin edges and adds the counts from each newly received
    chunk to its previously accumulated counts.
    Fixed edges are necessary for streaming updates because each bin must
    continue to represent the same numeric interval across all chunks.

    Examples
    --------
    Explicit bin edges:

    >>> hist = Histogram(bins=np.array([0, 10, 20, 30]))
    >>> hist.update_stats(np.array([2, 7, 15]))
    >>> hist.update_stats(np.array([12, 25]))
    >>> hist.counts
    array([2, 2, 1])

    Number of bins with a fixed range:

    >>> hist = Histogram(bins=3, range=(0, 30))
    >>> hist.update_stats(np.array([2, 7, 15, 12, 25]))
    >>> hist.counts
    array([2, 2, 1])
    """

    def __init__(
        self,
        bins: int | np.ndarray | list,
        range: tuple[float, float] | None = None,
        density: bool = False
    ):
        """
        Initialize a stateful histogram.

        Parameters
        ----------
        bins : int, np.ndarray, or list
            Histogram bin configuration.

            If an integer is supplied, ``range`` must also be provided.
            For example, ``bins=3`` and ``range=(0, 30)`` creates the edges
            ``[0, 10, 20, 30]``.

            If an array-like value is supplied, it must contain explicit,
            strictly increasing bin edges. For example,
            ``[0, 10, 20, 30]`` creates three bins.
        range : tuple of float, optional
            Lower and upper bounds used to construct equal-width bins when
            ``bins`` is an integer. Must not be supplied when explicit bin
            edges are used.
        density : bool, optional
            Whether ``result()`` returns probability densities rather than
            raw counts by default. Raw counts are always retained internally.
            Default is False.

        Raises
        ------
        ValueError
            If ``bins`` is not a positive integer.
            If integer ``bins`` is supplied without ``range``.
            If explicit bin edges are not 1D.
            If explicit bin edges contain fewer than two values.
            If explicit bin edges are not strictly increasing.
            If ``range`` does not contain exactly two values.
            If the upper range boundary is not greater than the lower
            boundary.
            If bin edges or range values contain ``None``, NaN, infinite,
            or complex values.
        TypeError
            If ``bins`` is not an integer or array-like collection of
            numeric edges.
            If ``density`` is not a boolean.
        """
        if not isinstance(density, bool):
            raise TypeError("density must be a boolean.")

        self.density = density
        self._bin_edges = self._create_bin_edges(
            bins=bins,
            range=range
        )

        self.reset()

    @staticmethod
    def _create_bin_edges(
        bins: int | np.ndarray | list,
        range: tuple[float, float] | None
    ) -> np.ndarray:
        if isinstance(bins, bool):
            raise TypeError(
                "bins must be a positive integer or a 1D array-like "
                "collection of numeric edges."
            )

        if isinstance(bins, (int, np.integer)):
            if bins <= 0:
                raise ValueError("bins must be greater than zero.")

            if range is None:
                raise ValueError(
                    "range must be provided when bins is an integer "
                    "because streaming histograms require fixed edges."
                )

            lower, upper = Histogram._validate_range(range)

            return np.linspace(
                lower,
                upper,
                num=bins + 1,
                dtype=float
            )

        if range is not None:
            raise ValueError(
                "range must not be supplied when bins contains explicit "
                "bin edges."
            )

        edges = np.asarray(bins)

        if edges.ndim != 1:
            raise ValueError("Explicit histogram bin edges must be 1D.")

        if edges.size < 2:
            raise ValueError(
                "Explicit histogram bin edges must contain at least "
                "two values."
            )

        validate_numeric_array(edges)

        edges = edges.astype("float64", copy=False)

        if not np.all(np.diff(edges) > 0):
            raise ValueError(
                "Explicit histogram bin edges must be strictly increasing."
            )

        # Return the copy to avoid future alter from external code
        return edges.copy()

    @staticmethod
    def _validate_range(
        range: tuple[float, float]
    ) -> tuple[float, float]:
        """
        Validate the fixed numeric range used to construct equal-width bins.
        """
        range_values = np.asarray(range)

        if range_values.shape != (2,):
            raise ValueError(
                "range must contain exactly two values: "
                "(lower_bound, upper_bound)."
            )

        validate_numeric_array(range_values)

        lower = float(range_values[0])
        upper = float(range_values[1])

        if upper <= lower:
            raise ValueError(
                "The upper histogram range boundary must be greater than "
                "the lower boundary."
            )

        return lower, upper

    def update_stats(
        self,
        a: np.ndarray,
        weights: np.ndarray | None = None
    ) -> Self:
        """
        Incrementally add the histogram counts from a newly received chunk.

        The chunk may have any shape. Values are flattened internally, which
        matches the behaviour of ``np.histogram()`` and the existing
        standalone ``histogram()`` wrapper.

        Values outside the configured bin edges are rejected rather than
        silently ignored. This ensures that the accumulated histogram remains
        consistent with the number of processed observations.

        Parameters
        ----------
        a : np.ndarray
            Incoming numeric data chunk.
        weights : np.ndarray, optional
            Contribution weight for each value. Must have the same shape as
            ``a``. When omitted, every value contributes 1.

        Returns
        -------
        Histogram
            The updated histogram instance.

        Raises
        ------
        ValueError
            If ``a`` is empty.
            If ``a`` contains ``None``, NaN, infinite, or complex values.
            If any value falls outside the configured histogram range.
            If ``weights`` does not have the same shape as ``a``.
            If ``weights`` contains ``None``, NaN, infinite, or complex
            values.
        TypeError
            If ``a`` or ``weights`` is not numeric.

        Complexity
        ----------
        Time Complexity:
            O(n), where n is the number of values in the incoming chunk.
        Space Complexity:
            O(n + k), where k is the number of histogram bins.
        """
        values = np.asarray(a)

        if values.size == 0:
            raise ValueError("Input chunk must contain at least one value.")

        validate_numeric_array(values)

        values = values.astype("float64", copy=False).ravel()

        flattened_weights = None

        if weights is not None:
            weights_array = np.asarray(weights)
            if weights_array.shape != np.asarray(a).shape:
                raise ValueError(
                    "weights must have the same shape as the input chunk."
                )

            validate_numeric_array(weights_array)

            flattened_weights = weights_array.astype(
                "float64", copy=False
            ).ravel()

        lower_edge = self._bin_edges[0]
        upper_edge = self._bin_edges[-1]

        if (
            np.any(values < lower_edge)
            or np.any(values > upper_edge)
        ):
            raise ValueError(
                "Input chunk contains values outside the configured "
                "histogram range."
            )

        chunk_counts, _ = np.histogram(
            values,
            bins=self._bin_edges,
            weights=flattened_weights,
            density=False
        )

        # Store float values internally so weighted and unweighted chunks
        # can both be accumulated safely.
        self._counts += chunk_counts.astype("float64", copy=False)
        self._n_observations += values.size

        if flattened_weights is not None:
            self._has_weights = True

        return self

    def result(
        self,
        density: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the accumulated histogram and bin edges.

        Parameters
        ----------
        density : bool, optional
            Whether to return probability densities rather than raw counts.

        Returns
        -------
        counts_or_density : np.ndarray
            Raw accumulated bin counts or probability densities.
        bin_edges : np.ndarray
            Fixed histogram bin edges.

        Raises
        ------
        TypeError
            If ``density`` is not a boolean or ``None``.
        ValueError
            If density is requested before any values have been accumulated.
            If accumulated weights sum to zero.

        Complexity
        ----------
        Time Complexity:
            O(k), where k is the number of bins.
        Space Complexity:
            O(k) for returned copies.
        """
        if density is None:
            density = self.density

        if not isinstance(density, bool):
            raise TypeError("density must be a boolean or None.")

        if not density:
            return self.counts, self.bin_edges
        else:
            total = np.sum(self._counts)

            if total <= 0:
                raise ValueError(
                    "Density cannot be calculated because the accumulated "
                    "histogram weight must be greater than zero."
                )

            bin_widths = np.diff(self._bin_edges)
            densities = self._counts / (total * bin_widths)
            return densities, self.bin_edges

    def reset(self) -> Self:
        """
        Clear accumulated histogram counts yet preserved fixed bin edges.

        Returns
        -------
        Histogram
            The reset histogram instance.

        Complexity
        ----------
        Time Complexity:
            O(k), where k is the number of bins.
        Space Complexity:
            O(k) for the reset count array.
        """
        self._counts = np.zeros(
            len(self._bin_edges) - 1,
            dtype=float
        )

        self._n_observations = 0
        self._has_weights = False

        return self

    @property
    def counts(self) -> np.ndarray:
        """
        Return a copy of the accumulated bin counts.

        Unweighted counts are returned as integers. Weighted counts are
        returned as floating-point values.
        """
        if self._has_weights:
            return self._counts.copy()

        return self._counts.astype(int)

    @property
    def bin_edges(self) -> np.ndarray:
        """
        Return a copy of the fixed bin edges.
        """
        return self._bin_edges.copy()

    @property
    def count(self) -> int:
        """
        Return the number of processed observations.
        """
        return self._n_observations
