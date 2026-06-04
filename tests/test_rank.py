from numcompute.rank import rank, percentile
import numpy as np
from numpy.testing import assert_array_equal, assert_raises_regex
import pytest


def test_invalid_data_input():
    X = [[1, 2], [2, 2], [2, 3]]
    with pytest.raises(ValueError, match=r"Data is not a type of ndarray\[\]"):
        rank(X, method='average')


def test_unsupported_method():
    X = np.array([[1, 2], [2, 2], [2, 3]])
    with pytest.raises(ValueError, match="Ranking method not supported"):
        rank(X, method='min')


def test_average_rank():
    X = np.array([
        [10, 30],
        [40, 20],
        [10, 20],
        [50, 30],
        [40, 20],
    ])
    exp_X = np.array([
        [1.5, 4.5],
        [3.5, 2.0],
        [1.5, 2.0],
        [5.0, 4.5],
        [3.5, 2.0],
    ])
    assert_array_equal(rank(X, method='average'), exp_X)


def test_dense_rank():
    X = np.array([
        [10, 30],
        [40, 20],
        [10, 20],
        [50, 30],
        [40, 20],
    ])
    exp_X = np.array([
        [1, 2],
        [2, 1],
        [1, 1],
        [3, 2],
        [2, 1],
    ])
    assert_array_equal(rank(X, method='dense'), exp_X)


def test_ordinal_rank():
    X = np.array([
        [10, 30],
        [40, 20],
        [10, 20],
        [50, 30],
        [40, 20],
    ])
    exp_X = np.array([
        [1, 4],
        [3, 1],
        [2, 2],
        [5, 5],
        [4, 3],
    ])
    assert_array_equal(rank(X, method='ordinal'), exp_X)


def test_invalid_dimension_data():
    X = np.ones((2, ))
    with assert_raises_regex(
            ValueError, "Data must be a 2D array"):
        rank(X, method='average')


def test_percentile_invalid_data_input():
    X = [[1, 2], [2, 2], [2, 3]]
    with pytest.raises(
            ValueError, match=r"data is not a type of ndarray\[\]"):
        percentile(X, 10)

    one_d_X = np.array([1, 2, 3, 4, 5])
    with pytest.raises(
            ValueError, match=r"data must be a 2D array"):
        percentile(one_d_X, 10)


def test_percentile_NaN_data_input():
    X = np.array([[1, 2], [2, 2], [2, np.nan]])
    with pytest.raises(
            ValueError, match=r"data contains NaN value\(s\)"):
        percentile(X, 10)


def test_percentile_inf_data_input():
    X = np.array([[1, 2], [2, 2], [2, np.inf]])
    with pytest.raises(
            ValueError, match=r"data contains infinite value\(s\)"):
        percentile(X, 10)


def test_percentile_invalid_quantile_input():
    X = np.array([[1, 2], [2, 2], [2, 3]])
    with pytest.raises(
            TypeError,
            match=r"q must be passed as a whole number"):
        percentile(X, 0.1)

    with pytest.raises(
            ValueError,
            match=r"q must be greater than 0"):
        percentile(X, -1)


def test_percentile_linear_strategy():
    X = np.array([
        [10, 80, 90],
        [70, 20, 60],
        [40, 50, 30]
    ])
    exp_X = np.array([31.0, 41.0, 51.0])

    assert_array_equal(
        percentile(X, q=35),
        exp_X
    )


def test_percentile_lower_strategy():
    X = np.array([
        [10, 80, 90],
        [70, 20, 60],
        [40, 50, 30]
    ])
    exp_X = np.array([10, 20, 30])

    assert_array_equal(
        percentile(X, q=35, interpolation='lower'),
        exp_X
    )


def test_percentile_upper_strategy():
    X = np.array([
        [10, 80, 90],
        [70, 20, 60],
        [40, 50, 30]
    ])
    exp_X = np.array([40, 50, 60])

    assert_array_equal(
        percentile(X, q=35, interpolation='upper'),
        exp_X
    )


def test_percentile_midpoint_strategy():
    X = np.array([
        [10, 80, 90],
        [70, 20, 60],
        [40, 50, 30]
    ])
    exp_X = np.array([25, 35, 45])

    assert_array_equal(
        percentile(X, q=35, interpolation='midpoint'),
        exp_X
    )
