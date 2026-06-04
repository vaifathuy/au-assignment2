import numpy as np
from numcompute.utils import validate_numeric_array

"""
The Statistics class helps find common statistics like mean, median, min, max
and standard deviation. It uses Welford Algorithm which helps calculate mean,
min, max and standard deviation in one pass. Traditional code required two
passes for finding standard deviation, one for finding the mean, second for
finding the difference between the individual points and the mean. After this
we finally divide this value by length of the dataset to find the variance.
Then we take the square root to find the standard deviation. Here is the
sample code:

data = [2,3,4]
mean = 0
#add all the data points
for i in data:
    mean += i
#divide by total data points
mean = mean / len(data)

meanDifSum = 0
for i in data:
    meanDifSum += pow((i - mean),2)
#find the variance
var = meanDifSum / len(data) #population variance
sd = pow(var, 0.5)

The disadvantage of the above code is that it requires 2 passes (2 for loops)
which is expensive. Moreover, we also store the mean in memory which can be
very large for large datasets and is only required in a small part in the
formula. We're also storing the whole dataset in memory which is expensive.

To overcome this, we use Welford's algorithm. This algo is used when we've a
large dataset and we want to calculate "running sd". The above code required
2 passes, here we calculate it with a single pass. The sd is calculated as
we're traversing the large dataset. It looks like this:

n++; //our index
delta = value - mean
mean = mean + delta / n
delta2 = value - mean
total_dev = delta * delta2

After running through the data, we find the variance and then the sd:

variance = total_dev / (n - 1)
sd = pow(variance, 0.5)

Advantage: Earlier we had to run 2 for loops (O(N)). Now it is O(1) inside
the main loop.

Finding mean, median, min, max: mean is already found out. min and max can
be calculated inside the for loop by doing cur_min=min(cur_min, val) and
cur_max=max(cur_max, val). For calculating the median, we use numpy.median().

>>> stat = Statistics()
        stat.add(5)
        stat.add(8)
        stat.add(3)
        stat.add(5)
        stat.add(2)
        stat.add(6)
>>> stat.std_dev()
>>> stat.mean
>>> stat.min
>>> stat.max
"""


class Statistics:
    def __init__(self):
        self._count = 0
        self._mean = 0
        self._total_dev = 0
        self._min = float('inf')
        self._max = float('-inf')

    def add(self, x : int | float):
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
            If ``x`` is ``None`` or not an ``int`` or ``float``.

        Complexity
        ----------
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if x is None or not isinstance(x, (int, float)):
            raise ValueError("Please pass an integer or float value")
        self._count += 1

        self._min = min(self._min, x)
        self._max = max(self._max, x)

        delta1 = x - self._mean

        self._mean = self._mean + delta1 / self._count

        delta2 = x - self._mean

        self._total_dev = delta1 * delta2

    def std_dev(self) -> float:
        """
        Compute the sample standard deviation from the accumulated data.

        Returns
        -------
        float
            Sample standard deviation (ddof=1).

        Complexity
        ----------
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        var = self._total_dev / (self._count - 1)
        return pow(var, 0.5)

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
