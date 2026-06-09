from abc import ABC, abstractmethod
from typing import Self
import numpy as np
from numcompute_stream.utils import validate_numeric_array


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

    @abstractmethod
    def partial_fit(self, X: np.ndarray, y=None) -> Self:
        """This method must be implemented by any subclass."""
        pass

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
            replace_none: bool = True,
            strategy: str = "constant"
    ):
        if not replace_nan and not replace_none:
            raise ValueError("Must have at least one of replace_nan or "
                             "replace_none set to True.")

        if strategy not in {"constant", "mean"}:
            raise ValueError(
                "strategy must be either 'constant' or 'mean'."
            )

        self.fill_value = fill_value
        self.replace_nan = replace_nan
        self.replace_none = replace_none
        self.strategy = strategy

        # attributes to fit
        self._feature_sums = None
        self._valid_counts = None
        self._statistics = None

    def fit(self, X: np.ndarray, y=None) -> Self:
        """
        Reset the imputer and learn replacement statistics from the supplied
        training dataset.

        Any previously learned state is discarded before fitting. When
        ``strategy='constant'``, no values need to be estimated and the method
        only validates the input shape. When ``strategy='mean'``, the method
        learns the per-feature mean from valid observations while ignoring
        missing values.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n), where m is the number of rows and
            n is the number of features.
        y : np.ndarray, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        SimpleImputer
            The fitted imputer instance.

        Raises
        ------
        ValueError
            If ``X`` is not 2D.
            If ``X`` contains no rows.
        TypeError
            If ``strategy='mean'`` and non-missing values cannot be converted
            to numeric values.

        Complexity
        ----------
        Time Complexity:
            O(m * n) when ``strategy='mean'``.
            O(1) when ``strategy='constant'`` after shape validation.
        Space Complexity:
            O(m * n) in the worst case for a temporary numeric copy.
            The retained fitted state requires O(n) space.
        """
        self._feature_sums = None
        self._valid_counts = None
        self._statistics = None

        return self.partial_fit(X, y)

    def partial_fit(self, X: np.ndarray, y=None) -> Self:
        """
        Incrementally update per-feature missing-value estimates using a newly
        received data chunk.

        Previously learned statistics are preserved. When ``strategy='mean'``,
        valid values are accumulated per feature while ``None`` and NaN values
        are ignored. The learned replacement statistics are updated after each
        chunk. When ``strategy='constant'``, no statistics are learned.

        Parameters
        ----------
        X : np.ndarray
            Incoming data chunk of shape (m, n), where m is the number of rows
            in the chunk and n is the number of features.
        y : np.ndarray, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        SimpleImputer
            The updated imputer instance.

        Raises
        ------
        ValueError
            If ``X`` is not 2D.
            If ``X`` contains no rows.
            If ``X`` has a different number of features than previously
            processed chunks.
        TypeError
            If ``strategy='mean'`` and non-missing values cannot be converted
            to numeric values.

        Complexity
        ----------
        Time Complexity:
            O(m * n) when ``strategy='mean'`` because the incoming chunk must
            be scanned for missing values and accumulated by feature.
            O(1) when ``strategy='constant'`` after shape validation.
        Space Complexity:
            O(m * n) in the worst case for a temporary numeric copy.
            The retained fitted state requires O(n) space.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must have 2 dimensions. Reshape your array first."
            )

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one row.")

        n_features = X.shape[1]

        if (
            self._statistics is not None
            and n_features != len(self._statistics)
        ):
            raise ValueError(
                "Input array has a different number of features than "
                "the previously fitted array."
            )

        # Constant replacement does not require learned statistics.
        if self.strategy == "constant":
            return self

        numeric_X = X.astype(object, copy=True)
        none_mask = numeric_X == None  # noqa: E711
        numeric_X[none_mask] = np.nan

        try:
            numeric_X = numeric_X.astype("float64")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Mean imputation requires numeric non-missing values."
            ) from exc

        valid_mask = ~np.isnan(numeric_X)

        chunk_sums = np.nansum(numeric_X, axis=0)
        chunk_valid_counts = np.sum(valid_mask, axis=0)

        if self._feature_sums is None:
            self._feature_sums = np.zeros(n_features, dtype=float)
            self._valid_counts = np.zeros(n_features, dtype=int)

        self._feature_sums += chunk_sums
        self._valid_counts += chunk_valid_counts

        self._statistics = np.divide(
            self._feature_sums,
            self._valid_counts,
            out=np.full(n_features, self.fill_value, dtype=float),
            where=self._valid_counts != 0
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace missing values using the configured imputation strategy.

        When ``strategy='constant'``, every selected missing value is
        replaced with ``fill_value``. When ``strategy='mean'``, missing
        values are replaced using the learned per-feature means.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (m, n).

        Returns
        -------
        np.ndarray
            Imputed copy of the input array.

        Raises
        ------
        ValueError
            If ``X`` is not 2D.
            If ``strategy='mean'`` but the imputer has not been fitted.
            If ``X`` has a different number of features than the fitted data.
            If unhandled ``None`` values prevent NaN processing.
        TypeError
            If ``strategy='mean'`` and input values cannot be converted to
            numeric values.

        Complexity
        ----------
        Time Complexity:
            O(m * n) to scan and replace missing values.
        Space Complexity:
            O(m * n) for the returned copy.
        """
        new_X = np.asarray(X).copy()

        if new_X.ndim != 2:
            raise ValueError(
                "X must have 2 dimensions. Reshape your array first."
            )

        if self.strategy == "mean":
            if self._statistics is None:
                raise ValueError(
                    "The SimpleImputer instance is not fitted yet. "
                    "Call 'fit()' or 'partial_fit()' before using "
                    "mean imputation."
                )

            if new_X.shape[1] != len(self._statistics):
                raise ValueError(
                    "Input array has a different number of features "
                    "than the fitted array."
                )

            fill_values = self._statistics

        else:
            fill_values = np.full(
                new_X.shape[1],
                self.fill_value,
                dtype=float
            )

        for feature_index in range(new_X.shape[1]):
            column = new_X[: , feature_index]

            if self.replace_none:
                none_mask = column == None  # noqa: E711
                if none_mask.any():
                    column[none_mask] = fill_values[feature_index]

            if self.replace_nan:
                # Can't process `nan`s if the array still contains `None`s
                none_mask = column == None  # noqa: E711
                if none_mask.any():
                    raise ValueError(
                        "Can't process `nan`s while `None`s exist. "
                        "Retry with `replace_none` set `True`."
                    )

                # Cast the column so that checking .isnan() work correctly
                try:
                    numeric_column = column.astype("float64")
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "NaN replacement requires numeric values."
                    ) from exc

                nan_mask = np.isnan(numeric_column)

                if nan_mask.any():
                    column[nan_mask] = fill_values[feature_index]

            new_X[:, feature_index] = column

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
        self._feature_maxs = None
        self._feature_ranges = None

    def fit(self, X: np.ndarray, y=None) -> Self:
        """
        Compute and store the per-feature minimum and value range from
        training data.

        Any previously learned state is discarded before fitting. This makes
        ``fit()`` suitable for traditional batch learning. To preserve the
        existing state and incorporate an additional data chunk, use
        ``partial_fit()`` instead.

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
            If ``X`` contains no rows.
            If ``X`` contains None, NaN, infinite, or complex values.
            If any feature's value range exceeds float64 limits.
        TypeError:
            If ``X`` is not a numeric array.

        Complexity
        ----------
        Time Complexity:
            O(m * n) for calculating per-feature minima and maxima.
        Space Complexity:
            O(m * n) in the worst case for converting ``X`` to float64.
            The retained fitted state requires O(n) space.
        """
        self._feature_mins = None
        self._feature_maxs = None
        self._feature_ranges = None

        return self.partial_fit(X, y)

    def partial_fit(self, X: np.ndarray, y=None) -> Self:
        """
        Incrementally Update the per-feature minimum, maximum, and range.

        Parameters
        ----------
        X : np.ndarray
            Incoming data chunk of shape (n_rows, n_features).
        y : np.ndarray, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        MinMaxScaler
            The updated scaler instance.

        Raises
        ------
        ValueError
            If ``X`` is not 2D.
            If ``X`` contains no rows.
            If ``X`` contains None, NaN, infinite, or complex values.
            If any feature's value range exceeds float64 limits.
        TypeError:
            If ``X`` is not a numeric array.

        Complexity
        ----------
        Time Complexity:
            O(m * n) for calculating the incoming chunk's per-feature minima
            and maxima.
        Space Complexity:
            O(m * n) in the worst case for converting ``X`` to float64.
            The retained fitted state requires O(n) space.
        """

        if X.ndim != 2:
            raise ValueError(
                "X must have 2 dimensions. Reshape your array first."
            )

        validate_numeric_array(data=X)

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one row.")

        X = X.astype("float64")
        n_features = X.shape[1]

        chunck_mins = np.min(X, axis=0)
        chunck_maxs = np.max(X, axis=0)

        if self._feature_mins is None:
            self._feature_mins = chunck_mins
            self._feature_maxs = chunck_maxs
        elif n_features != len(self._feature_mins):
            raise ValueError(
                "Input array has a different number of features than "
                "the previously fitted array."
            )
        else:
            self._feature_mins = np.minimum(self._feature_mins, chunck_mins)
            self._feature_maxs = np.maximum(self._feature_maxs, chunck_maxs)

        self._feature_ranges = self._feature_maxs - self._feature_mins

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

        if (
            self._feature_mins is None
            or self._feature_maxs is None
            or self._feature_ranges is None
        ):
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
            raise ValueError(
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
        if handle_unknown not in {"ignore", "error"}:
            raise ValueError(
                "handle_unknown must be either 'ignore' or 'error'."
            )

        self.handle_unknown = handle_unknown
        self._categories = None

    def fit(self, X: np.ndarray, y=None) -> Self:
        """
        Identify and store the unique categories for each feature.
        It is designed to perform batch-processing, so it resets the interal
        states.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n). Must be 2D and must not
            contain ``None`` values.
        y : np.ndarray, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        OneHotEncoder
            The fitted instance.

        Raises
        ------
        ValueError
            If ``X`` is not 2D.
            If ``X`` contains no rows.
            If ``X`` contains ``None`` values.
            If numeric ``X`` contains NaN, infinite, or complex values.
        TypeError
            If numeric ``X`` is not a valid numeric array.

        Complexity
        ----------
        Time Complexity:
            O(n * m log m) for identifying unique categories in each feature.
        Space Complexity:
            O(c) for retaining the unique categories, where c is the total
            number of unique categories across all features.
        """
        # if np.any(X == None): # noqa
        #     raise ValueError(
        #         "X contains None values. Handle these then try again."
        #     )

        # if np.issubdtype(X.dtype, np.number):
        #     validate_numeric_array(data=X)

        # if X.ndim != 2:
        #     raise ValueError("X must be a 2D array")

        # self._categories = [np.unique(X[:, i]) for i in range(X.shape[1])]

        self._categories = None

        return self.partial_fit(X, y)

    def partial_fit(self, X: np.ndarray, y=None) -> Self:
        """
        Incrementally identify and store the unique categories
        for each feature.

        Previously learned category positions are preserved. Categories that
        have not been observed before are appended to the end of the relevant
        feature's category array. Existing category indexes never shift.

        Parameters
        ----------
        X : np.ndarray
            Incoming data chunk of shape (m, n), where m is the number of rows
            and n is the number of categorical features. Must be 2D and must
            not contain ``None`` values.
        y : np.ndarray, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        OneHotEncoder
            The updated instance.

        Raises
        ------
        ValueError
            If ``X`` is not 2D.
            If ``X`` contains no rows.
            If ``X`` contains ``None`` values.
            If ``X`` has a different number of features than previously
            processed chunks.
            If numeric ``X`` contains NaN, infinite, or complex values.
        TypeError
            If numeric ``X`` is not a valid numeric array.

        Complexity
        ----------
        Time Complexity:
            O(n * m log m + u * c), where n is the number of features,
            m is the number of rows in the new chunk, u is the number of unique
            incoming categories, and c is the number of previously learned
            categories.
        Space Complexity:
            O(c + u) for retaining existing and newly discovered categories.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one row.")

        if (X == None).any():  # noqa: E711
            raise ValueError(
                "X contains None values. Handle these then try again."
            )

        if np.issubdtype(X.dtype, np.number):
            validate_numeric_array(data=X)

        n_features = X.shape[1]

        # First chunk: initialize categories.
        if self._categories is None:
            self._categories = [
                np.unique(X[:, i])
                for i in range(n_features)
            ]

            return self

        if n_features != len(self._categories):
            raise ValueError(
                "Input array has a different number of features than "
                "the previously fitted array."
            )

        for feature_index in range(n_features):
            existing_categories = self._categories[feature_index]

            incoming_categories = np.unique(X[:, feature_index])

            unseen_categories = incoming_categories[
                ~np.isin(incoming_categories, existing_categories)
            ]

            if unseen_categories.size > 0:
                self._categories[feature_index] = np.concatenate([
                    existing_categories,
                    unseen_categories
                ])

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
            If ``X`` is not 2D.
            If ``X`` contains ``None`` values.
            If ``fit()`` or ``partial_fit()`` has not been called.
            If ``X`` has a different number of features than the fitted data.
            If unknown categories are encountered while
            ``handle_unknown='error'``.
            If numeric ``X`` contains NaN, infinite, or complex values.
        TypeError
            If numeric ``X`` is not a valid numeric array.

        Complexity
        ----------
        Time Complexity:
            O(m * total unique categories across all features) for constructing
            and populating the one-hot blocks.
        Space Complexity:
            O(m * total unique categories across all features) for the
            one-hot encoded output array.
        """
        X = np.asarray(X)

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
        for feature_index, category_values in enumerate(self._categories):
            feature_col = X[:, feature_index]
            # compare every row value against every learned category
            matches = feature_col[:, None] == category_values[None, :]

            unknown_mask = ~np.any(matches, axis=1)

            if unknown_mask.any() and self.handle_unknown == "error":
                bad_indexes = np.where(unknown_mask)[0]
                bad_values = feature_col[bad_indexes]

                raise ValueError(
                    f"Column {feature_index} contains unseen "
                    f"values {bad_values} at row indexes "
                    f"{bad_indexes}."
                )

            # Rows with unknown values naturally remain all zeros.
            blocks.append(matches.astype(int))

        return np.hstack(blocks)
