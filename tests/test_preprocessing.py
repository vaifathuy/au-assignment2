from numcompute.preprocessing import (
    SimpleImputer,
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
)
import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
    assert_allclose,
    assert_equal,
)
import math
import pytest


class TestSimpleImputer:

    def test_does_nothing_if_no_nans(self):
        imputer = SimpleImputer(fill_value=5)
        X = np.arange(6)
        transformed_X = imputer.transform(X)
        assert_array_equal(transformed_X, X)

    def test_does_not_modify_origin_array(self):
        imputer = SimpleImputer(fill_value=5)
        X = np.array([1, float('nan'), 4, 6, np.nan, math.nan, 10])
        original_X = X.copy()
        transformed_X = imputer.transform(X)
        assert_array_equal(X, original_X)
        assert_raises(
            AssertionError,
            assert_array_equal,
            transformed_X, original_X
        )

    def test_replaces_nans_transform(self):
        imputer = SimpleImputer(fill_value=5)
        X = np.array([1, float('nan'), 4, 6, np.nan, math.nan, 10])
        expected_X = np.array([1, 5, 4, 6, 5, 5, 10])
        transformed_X = imputer.transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_replaces_nans_fit_transform(self):
        imputer = SimpleImputer(fill_value=5)
        X = np.array([1, float('nan'), 4, 6, np.nan, math.nan, 10])
        expected_X = np.array([1, 5, 4, 6, 5, 5, 10])
        transformed_X = imputer.fit_transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_replaces_nans_not_nones(self):
        imputer = SimpleImputer(fill_value=5, replace_none=False)
        X = np.array([1, float('nan'), 4, 6, None, math.nan, 10])
        assert_raises_regex(
            ValueError,
            "Can't process `nan`s while `None`s exist. "
            "Retry with `replace_none` set `True`.",
            imputer.fit_transform,
            X
        )

    def test_replaces_nans_and_nones(self):
        imputer = SimpleImputer(fill_value=5)
        X = np.array([1, float('nan'), 4, 6, None, math.nan, 10])
        expected_X = np.array([1, 5, 4, 6, 5, 5, 10])
        transformed_X = imputer.fit_transform(X)
        assert_array_equal(transformed_X, expected_X)


class TestMinMaxScaler:

    def test_raises_if_transform_before_fit(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]])
        assert_raises(ValueError, scaler.transform, X)

    def test_raises_invalid_min_max(self):
        with pytest.raises(ValueError, match="max must be greater than min."):
            MinMaxScaler(min=10, max=-10)

    def test_raises_invalid_dimensions(self):
        scaler = MinMaxScaler()
        # Should fail on 1D array
        X = np.ones((3,))
        assert_raises(ValueError, scaler.fit, X)
        # Should fail on 3D array
        X = np.ones((2, 2, 2))
        assert_raises(ValueError, scaler.fit, X)

    def test_does_not_modify_origin_array(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]])
        original_X = X.copy()
        scaler.fit(X)
        transformed_X = scaler.transform(X)
        assert_array_equal(X, original_X)
        assert_raises(
            AssertionError,
            assert_array_equal,
            transformed_X, original_X
        )

    def test_transforms_simple_case(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]])
        expected_X = np.array([[0, 0], [1, 1]])
        scaler.fit(X)
        transformed_X = scaler.transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_transforms_supplied_min_max(self):
        scaler = MinMaxScaler(min=-10, max=10)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        expected_X = np.array([[-10, -10], [0, 0], [10, 10]])
        scaler.fit(X)
        transformed_X = scaler.transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_fit_transform_shape_mismatch(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)
        Y = np.array([[1, 2, 3], [4, 5, 6]])
        assert_raises(ValueError, scaler.transform, Y)

    def test_no_variance_in_feature(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [1, 4], [1, 6]])
        # Column index 0 has no variance, so all values should be 0.5.
        # Column index 1 should be scaled
        expected_X = np.array([[0.5, 0], [0.5, 0.5], [0.5, 1]])
        transformed_X = scaler.fit_transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_input_datatype(self):
        scaler = MinMaxScaler()
        X = np.array([["a", "b"], ["c", "d"]])
        assert_raises(TypeError, scaler.fit, X)
        scaler.fit(np.array([[1, 2], [3, 4]]))
        assert_raises(TypeError, scaler.transform, X)

    def test_large_range_raises_error(self):
        scaler = MinMaxScaler()
        X = np.array([[-1e308], [1e308]])
        assert_raises_regex(
            ValueError,
            r"Feature value range exceeds float64 limits. "
            r"Scale your data before fitting.",
            scaler.fit,
            X
        )

    def test_bad_input_data(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, np.nan]])
        assert_raises(ValueError, scaler.fit, X)
        X = np.array([[1, 2], [3, math.inf]])
        assert_raises(ValueError, scaler.fit, X)
        X = np.array([[1, 2], [3, None]])
        assert_raises(ValueError, scaler.fit, X)
        X = np.array([[1, 2], [3, 4j]])
        assert_raises(ValueError, scaler.fit, X)
        # Now test transform.
        scaler.fit(np.array([[1, 2], [3, 4]]))
        X = np.array([[1, 2], [3, np.nan]])
        assert_raises(ValueError, scaler.transform, X)
        X = np.array([[1, 2], [3, math.inf]])
        assert_raises(ValueError, scaler.transform, X)
        X = np.array([[1, 2], [3, None]])
        assert_raises(ValueError, scaler.transform, X)
        X = np.array([[1, 2], [3, 4j]])
        assert_raises(ValueError, scaler.transform, X)

    def test_simple_partial_fit(self):
        scaler = MinMaxScaler()
        X = np.array([[2, 10], [6, 20], [1, 30], [8, 15]])
        scaler.partial_fit(X[:2])
        scaler.partial_fit(X[2:])
        assert_equal(scaler._feature_mins, [1, 10])
        assert_equal(scaler._feature_maxs, [8, 30])
        assert_equal(scaler._feature_ranges, [7, 20])

    def test_simple_partial_fit_transform(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]])
        expected_X = np.array([[0, 0], [1, 1]])
        scaler.partial_fit(X[:1])
        scaler.partial_fit(X[1:])
        transformed_X = scaler.transform(X)
        assert_array_equal(transformed_X, expected_X)


class TestStandardScaler:

    def test_simple_batch_fit_transform(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        expected_X = np.array([[-1.22474487, -1.22474487],
                               [0, 0],
                               [1.22474487, 1.22474487]])
        transformed_X = scaler.fit_transform(X)
        assert_allclose(transformed_X, expected_X)

    def test_raises_if_transform_before_fit(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]])
        assert_raises(ValueError, scaler.transform, X)

    def test_does_not_modify_origin_array(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]])
        original_X = X.copy()
        scaler.fit(X)
        transformed_X = scaler.transform(X)
        assert_array_equal(X, original_X)
        assert_raises(
            AssertionError,
            assert_array_equal,
            transformed_X, original_X
        )

    def test_fit_transform_shape_mismatch(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)
        Y = np.array([[1, 2, 3], [4, 5, 6]])
        assert_raises(ValueError, scaler.transform, Y)

    def test_no_variance_in_feature(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [1, 4], [1, 6]])
        # Column index 0 has no variance, so all values should be 0.0.
        # Column index 1 should be scaled
        expected_X = np.array(
            [[0.0, -1.22474487], [0.0, 0.0], [0.0, 1.22474487]]
        )
        transformed_X = scaler.fit_transform(X)
        assert_allclose(transformed_X, expected_X)

    def test_non_numeric_data(self):
        scaler = StandardScaler()
        X = np.array([["a", "b"], ["c", "d"]])
        assert_raises(TypeError, scaler.fit, X)
        scaler.fit(np.array([[1, 2], [3, 4]]))
        assert_raises(TypeError, scaler.transform, X)

    def test_bad_data(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, np.nan]])
        assert_raises(ValueError, scaler.fit, X)
        X = np.array([[1, 2], [3, math.inf]])
        assert_raises(ValueError, scaler.fit, X)
        X = np.array([[1, 2], [3, None]])
        assert_raises(ValueError, scaler.fit, X)
        X = np.array([[1, 2], [3, 4j]])
        assert_raises(ValueError, scaler.fit, X)
        # Now test transform.
        scaler.fit(np.array([[1, 2], [3, 4]]))
        X = np.array([[1, 2], [3, np.nan]])
        assert_raises(ValueError, scaler.transform, X)
        X = np.array([[1, 2], [3, math.inf]])
        assert_raises(ValueError, scaler.transform, X)
        X = np.array([[1, 2], [3, None]])
        assert_raises(ValueError, scaler.transform, X)
        X = np.array([[1, 2], [3, 4j]])
        assert_raises(ValueError, scaler.transform, X)

    def test_simple_partial_fit(self):
        scaler = StandardScaler()
        X = np.array([[2, 10], [6, 20], [10, 30]])
        scaler.partial_fit(X[:2])
        scaler.partial_fit(X[2:])
        assert_equal(scaler._count, 3)
        assert_allclose(scaler._mean, [6, 20])
        assert_allclose(scaler._M2, [32, 200])
        assert_allclose(scaler._std, [3.26598632, 8.16496581])

    def test_simple_partial_fit_transform(self):
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.partial_fit(X[:2])
        scaler.partial_fit(X[2:])
        expected_X = np.array([[-1.22474487, -1.22474487],
                               [0, 0],
                               [1.22474487, 1.22474487]])
        transformed_X = scaler.transform(X)
        assert_allclose(transformed_X, expected_X)


class TestOneHotEncoder:

    def test_simple_case_str(self):
        scaler = OneHotEncoder()
        X = np.array([["red"], ["blue"], ["red"], ["green"], ["orange"]])
        expected_X = np.array([[0, 0, 0, 1],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0]])
        transformed_X = scaler.fit_transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_simple_case_int(self):
        scaler = OneHotEncoder()
        X = np.array([[3], [1], [2], [3]])
        expected_X = np.array([[0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        transformed_X = scaler.fit_transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_train_test_split(self):
        scaler = OneHotEncoder()
        X_train = np.array([["blue"], ["red"], ["green"], ["orange"]])
        X_test = np.array([["blue"], ["blue"], ["blue"], ["orange"]])
        expected_X_train = np.array([[1, 0, 0, 0],
                                     [0, 0, 0, 1],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0]])
        expected_X_test = np.array([[1, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 1, 0]])
        transformed_X_train = scaler.fit_transform(X_train)
        transformed_X_test = scaler.transform(X_test)

        assert_array_equal(transformed_X_train, expected_X_train)
        assert_array_equal(transformed_X_test, expected_X_test)

    def test_missing_value_ignore(self):
        scaler = OneHotEncoder()
        X_train = np.array([["blue"], ["red"], ["green"], ["orange"]])
        X_test = np.array([["blue"], ["blue"], ["blue"], ["silver"]])
        expected_X_test = np.array([[1, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0]])
        scaler.fit(X_train)
        transformed_X_test = scaler.transform(X_test)

        assert_array_equal(transformed_X_test, expected_X_test)

    def test_missing_value_error(self):
        scaler = OneHotEncoder(handle_unknown="error")
        X_train = np.array([["blue"], ["red"], ["green"], ["orange"]])
        X_test = np.array([["blue"], ["blue"], ["blue"], ["silver"]])
        scaler.fit(X_train)
        assert_raises_regex(
            ValueError,
            r"Column 0 contains unseen values \[\'silver\'\] at row "
            r"indexes \[3\]",
            scaler.transform,
            X_test
        )

    def test_multiple_features(self):
        scaler = OneHotEncoder()
        X = np.array([["blue", "small"],
                      ["red", "medium"],
                      ["green", "large"],
                      ["orange", "small"]])
        # First 4 columns are for colour, last 3 columns are for size.
        expected_X = np.array([[1, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 1, 0, 1, 0],
                               [0, 1, 0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0, 0, 1]])
        transformed_X = scaler.fit_transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_contains_none(self):
        scaler = OneHotEncoder()
        X = np.array([[None], [1], [2], [3]])
        assert_raises_regex(
            ValueError,
            "X contains None values. Handle these then try again.",
            scaler.fit,
            X
        )

    def test_fit_bad_input(self):
        scaler = OneHotEncoder()
        X = np.array([[math.inf], [1], [2], [3]])
        assert_raises_regex(
            ValueError,
            r"input contains infinite values. Fix these then try again.",
            scaler.fit,
            X
        )

    def test_transform_bad_input(self):
        scaler = OneHotEncoder()
        X_train = np.array([[0], [1], [2], [3]])
        scaler.fit(X_train)
        X_test = np.array([[np.nan], [1], [2], [3]])
        assert_raises_regex(
            ValueError,
            r"input contains NaN\(s\)\. Use the \`SimpleImputer\` "
            r"then try again\.",
            scaler.transform,
            X_test
        )
