from abc import ABC, abstractmethod
from typing import Self
import numpy as np
from numcompute.utils import validate_numeric_array


class _BasePreprocessor(ABC):
    """Abstract base class for all preprocessors. Defines and enforces the
    API for `fit()`, `transform()` and `fit_transform()`. fit() methods
    take an optional y argument to allow for interoperability with the
    scikit-learn library.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y=None) -> Self:
        """This method must be implemented by any subclass."""
        pass

    # @abstractmethod
    # def partial_fit(self, X: np.ndarray, y=None) -> Self:
    #     """This method must be implemented by any subclass."""
    #     pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """This method must be implemented by any subclass."""
        return X

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit the preprocessor and transform the data in a single call.

        Should only be called on training data, as it updates the
        preprocessor's internal state.

        Parameters
        ----------
        X : np.ndarray
            Data to fit and transform. Source array is not modified.
        y : np.ndarray, optional
            Unused in built-in preprocessors. Present for API compliance.

        Returns
        -------
        np.ndarray
            Preprocessed copy of the input array.

        Complexity
        ----------
        Time Complexity:
            O(fit cost + transform cost), delegated to
            the subclass implementation.
        Space Complexity:
            O(m * n) for the transformed copy returned by the subclass.
        """
        self.fit(X, y)
        return self.transform(X)


class SimpleImputer(_BasePreprocessor):
    """A simple imputer to replace any NaN values in an ndarray with a
    constant value. It handles
    various types of nan, such as `float('nan')`, `math.nan`, and `np.nan`.

    Attributes
    ----------
    fill_value : a value used to replace NaN
    """

    def __init__(
            self,
            fill_value: float = 0.0,
            replace_nan: bool = True,
            replace_none: bool = True
    ):
        if not replace_nan and not replace_none:
            raise ValueError("Must have at least one of replace_nan or "
                             "replace_none set to True.")
        self.fill_value = fill_value
        self.replace_nan = replace_nan
        self.replace_none = replace_none

    def fit(self, X: np.ndarray, y=None) -> Self:
        """
        No-op fit. SimpleImputer has no parameters to learn.

        Parameters
        ----------
        X : np.ndarray
            Ignored. Present for API compliance.
        y : np.ndarray, optional
            Ignored. Present for API compliance.

        Returns
        -------
        SimpleImputer
            The unchanged instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace ``None`` and/or ``NaN`` values with ``fill_value``.

        Always returns a copy; the source array is never modified.
        If the array contains no missing values, a plain copy is returned.

        Parameters
        ----------
        X : np.ndarray
            Input array to impute.

        Returns
        -------
        np.ndarray
            Imputed copy of the input array.

        Raises
        ------
        ValueError
            If ``replace_nan=True`` but unfilled ``None`` values remain
            in the array.

        Complexity
        ----------
        Time Complexity:
            O(m * n) to scan the full array for None and NaN values.
        Space Complexity:
            O(m * n) for the copied array.
        """
        new_X = np.asarray(X).copy()

        if self.replace_none:
            none_mask = new_X == None  # noqa: E711
            if none_mask.any():
                new_X[none_mask] = self.fill_value
                new_X = new_X.astype(float)

        if self.replace_nan:
            # Can't process `nan`s if the array still contains `None`s
            none_mask = new_X == None  # noqa: E711
            if none_mask.any():
                raise ValueError("Can't process `nan`s while `None`s exist. "
                                 "Retry with `replace_none` set `True`.")
            new_X[np.isnan(new_X)] = self.fill_value

        return new_X


class MinMaxScaler(_BasePreprocessor):
    """Scale the values in the `ndarray` to a common range (Default 0-1).

    Attributes
    ----------
    min (float) : the minimuim of the resulting value set.
    max (float) : the maximuim of the resulting value set.
    """

    def __init__(self, min: float = 0.0, max: float = 1.0):

        if max <= min:
            raise ValueError(
                "max must be greater than min."
            )

        self.min = min
        self.max = max

        # attributes to fit
        self._feature_mins = None
        self._feature_ranges = None

    def fit(self, X: np.ndarray, y=None) -> Self:
        """
        Compute and store the per-feature minimum and value range from
        training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n). Must be 2D and contain no
            NaNs, infinities, or complex numbers.
        y : np.ndarray, optional
            Ignored. Present for API compliance.

        Returns
        -------
        MinMaxScaler
            The fitted instance.

        Raises
        ------
        ValueError
            If ``X`` is not 2D.
            If ``X`` contains None, NaN, infinite, or complex values.
            If any feature's value range exceeds float64 limits.

        Complexity
        ----------
        Time Complexity:
            O(m * n) for computing per-feature min and range across all rows.
        Space Complexity:
            O(n) to store the per-feature min and range arrays.
        """
        if X.ndim != 2:
            raise ValueError(
                "X must have 2 dimensions. Reshape your array first"
            )

        validate_numeric_array(data=X)

        # Cast to float64 before computing ptp to avoid silent integer overflow
        # (e.g. int8 range of [-128, 127] wraps to -1 in signed arithmetic).
        new_arr = X.astype('float64')
        self._feature_mins = np.min(new_arr, axis=0)
        self._feature_ranges = np.ptp(new_arr, axis=0)

        if np.isinf(self._feature_ranges).any():
            raise ValueError(
                "Feature value range exceeds float64 limits. "
                "Scale your data before fitting."
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features to the [min, max] range set at initialisation.

        Features with zero variance are set to the midpoint of the target
        range to avoid division by zero.

        Parameters
        ----------
        X : np.ndarray
            Data to scale of shape (m, n).

        Returns
        -------
        np.ndarray
            Scaled copy of the input array.

        Raises
        ------
        ValueError
            If ``fit()`` has not been called.
            If the number of features in ``X`` does not match the fitted array.

        Complexity
        ----------
        Time Complexity:
            O(m * n) for the element-wise scaling operations.
        Space Complexity:
            O(m * n) for the scaled copy of the input array.
        """

        NO_VARIANCE_VALUE = 0.5

        validate_numeric_array(data=X)

        if self._feature_mins is None or self._feature_ranges is None:
            raise ValueError("You must call fit() before transform()")

        if X.shape[1] != len(self._feature_mins):
            raise ValueError(
                "Input array has a different number of features "
                "than the fitted array."
            )

        # Initially scale to the range 0-1. Constant features will be set to
        # the NO_VARIANCE_VALUE (0.5) to avoid division by zero.
        features_to_scale = self._feature_ranges != 0

        scaled_X = np.full(X.shape, NO_VARIANCE_VALUE)
        scaled_X[:, features_to_scale] = \
            (X[:, features_to_scale] - self._feature_mins[features_to_scale]) \
            / self._feature_ranges[features_to_scale]

        # Scale to the provided min-max range
        scaled_X = scaled_X * (self.max - self.min) + self.min

        return scaled_X


class StandardScaler(_BasePreprocessor):
    """A simple standard scaler to standardize the features of an ndarray.
    It scales the features to have a mean of 0 and a standard deviation of 1.

    Attributes
    ----------
    _mean : The mean value for each feature in the training data.
    _std : The standard deviation for each feature in the training data.
           We use the population standard deviation over the sample
           standard deviation as this is consistent with both numpy and
           scikit-learn's approach and would have a negligible impact on
           the end result.
    """

    def __init__(self):
        self._count = 0
        self._mean = None
        self._std = None
        self._M2 = None

    def fit(self, X: np.ndarray, y=None) -> Self:
        """
        Compute and store the per-feature mean and standard deviation.
        It is designed to perform batch-processing, so it reset the interal 
        states.

        Uses the population standard deviation (ddof=0), consistent with
        NumPy and scikit-learn defaults.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n).
        y : np.ndarray, optional
            Ignored. Present for API compliance.

        Returns
        -------
        StandardScaler
            The fitted instance.
        """

        self._count = 0
        self._mean = None
        self._M2 = None
        self._std = None

        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None) -> Self:
        """
        Incrementally update the running mean and standard deviation.

        Uses Welford's online algorithm to update statistics one row at a time
        without storing previously observed chunks.

        Parameters
        ----------
        X : np.ndarray
            Incoming data chunk of shape (n_rows, n_features).
        y : np.ndarray, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        StandardScaler
            The updated scaler instance.
        """

        if X.ndim != 2:
            raise ValueError(
                "X must have 2 dimensions. Reshape your array first"
            )

        validate_numeric_array(data=X)

        X = X.astype("float64")
        n_features = X.shape[1]

        if self._mean is None:
            self._mean = np.zeros(n_features, dtype=float)
            self._M2 = np.zeros(n_features, dtype=float)
        elif n_features != len(self._mean):
            ValueError(
                "Input array has a different number of features than "
                "the previously fitted array."
            )

        # Apply Welford's update
        for row in X:
            self._count += 1
            delta_1 = row - self._mean
            self._mean += delta_1 / self._count
            delta_2 = row - self._mean
            self._M2 += delta_1 * delta_2

        self._std = np.sqrt(self._M2 / self._count)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.

        Features with zero standard deviation are left as 0.0 to avoid
        division by zero.

        Parameters
        ----------
        X : np.ndarray
            Data to standardize of shape (m, n).

        Returns
        -------
        np.ndarray
            Standardized copy of the input array.

        Raises
        ------
        ValueError
            If ``fit()`` has not been called.
            If the number of features in ``X`` does not match the fitted array.

        Complexity
        ----------
        Time Complexity:
            O(m * n) for the element-wise standardization operations.
        Space Complexity:
            O(m * n) for the standardized copy of the input array.
        """
        if self._mean is None or self._std is None:
            raise ValueError("The StandardScaler instance is not fitted yet. "
                             "Call 'fit()' with training data before using "
                             "this method.")

        if X.shape[1] != len(self._mean):
            raise ValueError(
                "Input array has a different number of features "
                "than the fitted array."
            )

        validate_numeric_array(data=X)

        # We only scale features that have a non-zero standard deviation to
        # avoid division by zero. Features with a standard deviation of 0
        # will be 0.0.
        features_to_scale = self._std != 0

        scaled_X = np.zeros(X.shape)
        scaled_X[:, features_to_scale] = \
            (X[:, features_to_scale] - self._mean[features_to_scale]) \
            / self._std[features_to_scale]

        return scaled_X


class OneHotEncoder(_BasePreprocessor):
    """Encode categorical features as a one-hot numeric array. The encoding
    is case-sensitive and will treat different capitalizations of the same
    word as different categories.

    Attributes
    ----------
    handle_unknown : str
        How to handle unknown categories in the input data duiring
        transformation. Options are "ignore" (default) or "error". If
        "ignore", unknown categories will have all 0 for that feature.
        If "error", a ValueError will be raised if unknown categories are
        encountered.

    _categories : list of arrays
        The unique categories for each feature, determined during fitting.
        The list stores a sorted array of unique values for each feature,
        in the order the features appear in the input array.

    """

    def __init__(self, handle_unknown: str = "ignore"):
        self.handle_unknown = handle_unknown
        self._categories = None

    def fit(self, X: np.ndarray, y=None) -> Self:
        """
        Identify and store the unique categories for each feature.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n). Must be 2D.
        y : np.ndarray, optional
            Ignored. Present for API compliance.

        Returns
        -------
        OneHotEncoder
            The fitted instance.

        Raises
        ------
        ValueError
            If ``X`` is not 2D or contains ``None`` values.

        Complexity
        ----------
        Time Complexity:
            O(m * n * log(m)) — ``np.unique`` sorts each column of m rows
            across n columns.
        Space Complexity:
            O(total unique categories across all features) to store
            ``_categories``.
        """
        if np.any(X == None): # noqa
            raise ValueError(
                "X contains None values. Handle these then try again."
            )

        if np.issubdtype(X.dtype, np.number):
            validate_numeric_array(data=X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self._categories = [np.unique(X[:, i]) for i in range(X.shape[1])]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Encode categorical features as a one-hot numeric array.

        Each feature with k unique categories is expanded into k binary
        columns. Column order matches the sorted unique values identified
        during ``fit()``. Unknown categories are either zeroed out or raise
        an error, depending on ``handle_unknown``.

        Parameters
        ----------
        X : np.ndarray
            Categorical data of shape (m, n).

        Returns
        -------
        np.ndarray
            One-hot encoded array of shape (m, total unique categories
            across all features).

        Raises
        ------
        ValueError
            If ``fit()`` has not been called.
            If the number of features in ``X`` does not match the fitted array.
            If unknown categories are encountered and
            ``handle_unknown='error'``.

        Complexity
        ----------
        Time Complexity:
            O(m * total unique categories across all features) for constructing
            and populating the one-hot blocks.
        Space Complexity:
            O(m * total unique categories across all features) for the
            one-hot encoded output array.
        """

        if np.any(X is None):
            raise ValueError(
                "X contains None values. Handle these then try again."
            )

        if np.issubdtype(X.dtype, np.number):
            validate_numeric_array(data=X)

        if self._categories is None:
            raise ValueError("The OneHotEncoder instance is not fitted yet. "
                             "Call 'fit' with training data before using "
                             "this method.")

        if X.shape[1] != len(self._categories):
            raise ValueError(
                "Input array has a different number of features "
                "than the training array."
            )

        blocks = []
        for i, category_values in enumerate(self._categories):
            feature_col = X[:, i]
            # get the category_values index for every row in this feature_col
            indices = np.searchsorted(category_values, feature_col)
            # stability: if a value in feature_col is larger than all values in
            # category_values, searchsorted will return an index that is out
            # of bounds.
            np.clip(indices, 0, len(category_values) - 1, out=indices)

            # create the one-hot block for this column
            block = np.zeros((X.shape[0], len(category_values)), dtype=int)
            block[np.arange(X.shape[0]), indices] = 1

            # searchsorted above will not handle unknown categories, so we
            # check for these after and handle as per `handle_unknown``
            missing_mask = ~np.isin(feature_col, category_values)
            if missing_mask.any():
                bad_indexes = np.where(missing_mask)[0]
                bad_values = feature_col[bad_indexes]
                if self.handle_unknown == "error":
                    raise ValueError(f"Column {i} contains unseen "
                                     f"values {bad_values} at row indexes "
                                     f"{bad_indexes}")
                else:
                    block[missing_mask, :] = 0

            blocks.append(block)
        return np.hstack(blocks)
