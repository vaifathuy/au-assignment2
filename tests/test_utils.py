import numpy as np
import math
from numpy.testing import assert_raises_regex
from numcompute.utils import validate_numeric_array


class TestValidateNumericArray:
    def test_string_array(self):
        arr = np.array(["a", "b", "c"])
        assert_raises_regex(
            TypeError,
            r"input must be a numeric array.",
            validate_numeric_array,
            data=arr
        )

    def test_valid_array(self):
        # Should execute without error.
        arr = np.array([1, 2, 4])
        validate_numeric_array(arr)

    def test_contains_inf(self):
        arr = np.array([1, 3, math.inf])
        assert_raises_regex(
            ValueError,
            r"input contains infinite values. Fix these then try again.",
            validate_numeric_array,
            data=arr
        )

    def test_contains_nan(self):
        arr = np.array([1, 2, np.nan])
        assert_raises_regex(
            ValueError,
            r"input contains NaN\(s\). Use the `SimpleImputer` "
            "then try again.",
            validate_numeric_array,
            data=arr
        )

    def test_contains_complex_number(self):
        arr = np.array([4j])
        assert_raises_regex(
            ValueError,
            r"input contains complex values. Fix these then try again.",
            validate_numeric_array,
            data=arr
        )

    def test_contains_none(self):
        arr = np.array([1, 2, None])
        assert_raises_regex(
            ValueError,
            r"input contains None\(s\). Fix these then try again.",
            validate_numeric_array,
            data=arr
        )
