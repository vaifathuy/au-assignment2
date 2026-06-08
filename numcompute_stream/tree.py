from typing import Self

import numpy as np

from .utils import validate_numeric_array


class _Node:
    """
    Internal node used by ``DecisionTreeClassifier``.

    A leaf node stores observations and a predicted class. An internal node
    stores a feature index, a threshold, and two child nodes.
    """

    def __init__(self, depth: int):
        self.depth = depth

        self.prediction = None

        self.feature_index = None
        self.threshold = None

        self.left = None
        self.right = None

        self.X_samples = None
        self.y_samples = None

    @property
    def is_leaf(self) -> bool:
        """
        Return whether the node has no child nodes.
        """
        return (
            self.left is None
            and self.right is None
        )


class DecisionTreeClassifier:
    """
    A Gini-based classification decision tree with incremental online growth.

    ``fit()`` resets the tree and trains it from one complete dataset.

    ``partial_fit()`` preserves existing branches, routes incoming rows into
    their corresponding leaves, and allows eligible leaves to split when
    sufficient observations have accumulated.

    Examples
    --------
    Batch training:

    >>> tree = DecisionTreeClassifier(max_depth=2)
    >>> tree.fit(
    ...     np.array([[2], [3], [8], [9]]),
    ...     np.array([0, 0, 1, 1])
    ... )
    >>> tree.predict(np.array([[2], [9]]))
    array([0, 1])

    Incremental training:

    >>> tree = DecisionTreeClassifier(
    ...     max_depth=2,
    ...     min_samples_split=4
    ... )
    >>> tree.partial_fit(
    ...     np.array([[2], [3]]),
    ...     np.array([0, 0])
    ... )
    >>> tree.partial_fit(
    ...     np.array([[8], [9]]),
    ...     np.array([1, 1])
    ... )
    >>> tree.predict(np.array([[2], [9]]))
    array([0, 1])
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        random_state: int | None = None
    ):
        """
        Initialize an empty decision tree.

        Parameters
        ----------
        max_depth : int or None, optional
            Maximum depth of the tree. If ``None``, growth continues until
            another stopping condition is met. Default is ``None``.
        min_samples_split : int, optional
            Minimum number of observations required before a leaf can split.
            Default is 2.
        max_features : int or None, optional
            Number of randomly selected feature columns considered when
            searching for the best split. If ``None``, every feature is
            considered. Default is ``None``.
        random_state : int or None, optional
            Seed used when randomly selecting features.

        Raises
        ------
        TypeError
            If an argument has an unsupported type.
        ValueError
            If an integer argument is outside its valid range.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) before fitting.
        """
        if (
            max_depth is not None
            and type(max_depth) is not int
        ):
            raise TypeError("max_depth must be an integer or None.")

        if (
            max_depth is not None
            and max_depth < 0
        ):
            raise ValueError("max_depth must be non-negative.")

        if type(min_samples_split) is not int:
            raise TypeError("min_samples_split must be an integer.")

        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")

        if (
            max_features is not None
            and type(max_features) is not int
        ):
            raise TypeError("max_features must be an integer or None.")

        if (
            max_features is not None
            and max_features <= 0
        ):
            raise ValueError("max_features must be greater than zero.")

        if (
            random_state is not None
            and type(random_state) is not int
        ):
            raise TypeError("random_state must be an integer or None.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.reset()

    @property
    def classes(self) -> np.ndarray:
        """
        Return a copy of the class labels observed during training.
        """
        return np.asarray(self._classes).copy()

    @property
    def n_features(self) -> int | None:
        """
        Return the number of feature columns observed during training.
        """
        return self._n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Reset the tree and train it from a complete dataset.

        Parameters
        ----------
        X : np.ndarray
            Numeric training features of shape (m, n).
        y : np.ndarray
            Class labels containing m observations.

        Returns
        -------
        DecisionTreeClassifier
            The fitted tree instance.

        Complexity
        ----------
        Time Complexity:
            Depends on the number of evaluated split candidates.
        Space Complexity:
            O(m * n) for observations retained by leaves.
        """
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Self:
        """
        Incrementally grow the tree from an incoming data chunk.

        Parameters
        ----------
        X_chunk : np.ndarray
            Numeric feature chunk of shape (m, n).
        y_chunk : np.ndarray
            Class labels for the incoming chunk.

        Returns
        -------
        DecisionTreeClassifier
            The updated tree instance.

        Raises
        ------
        ValueError
            If arrays are empty, contain invalid values, have incompatible
            shapes, or use a different feature count from previous chunks.
        TypeError
            If ``X_chunk`` is not numeric.

        Complexity
        ----------
        Time Complexity:
            O(m * d) for routing m rows through a tree of depth d, plus
            the cost of evaluating eligible leaf splits.
        Space Complexity:
            O(m * n) additional retained memory in the worst case.
        """
        X_chunk, y_chunk = self._validate_X_y(X_chunk, y_chunk)

        if self._root is None:
            self._n_features = X_chunk.shape[1]

            self._validate_max_features_for_input(self._n_features)

            self._append_new_classes(y_chunk)

            self._root = self._create_leaf(
                X=X_chunk,
                y=y_chunk,
                depth=0
            )

            self._grow_leaf(
                self._root
            )

            return self

        if X_chunk.shape[1] != self._n_features:
            raise ValueError(
                "Input chunk has a different number of features than "
                "the fitted tree."
            )

        self._append_new_classes(
            y_chunk
        )

        leaf_updates = {}

        for row, label in zip(
            X_chunk,
            y_chunk
        ):
            leaf = self._find_leaf(
                row
            )

            if leaf not in leaf_updates:
                leaf_updates[leaf] = {
                    "rows": [],
                    "labels": []
                }

            leaf_updates[leaf]["rows"].append(
                row
            )

            leaf_updates[leaf]["labels"].append(
                label
            )

        for leaf, updates in leaf_updates.items():
            new_X = np.asarray(
                updates["rows"],
                dtype=float
            )

            new_y = np.asarray(
                updates["labels"]
            )

            self._append_samples(
                leaf=leaf,
                X=new_X,
                y=new_y
            )

            self._grow_leaf(
                leaf
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input rows.

        Parameters
        ----------
        X : np.ndarray
            Numeric features of shape (m, n).

        Returns
        -------
        np.ndarray
            Predicted class label for each row.

        Raises
        ------
        ValueError
            If the classifier has not been fitted.
            If the input feature count differs from the fitted tree.
        TypeError
            If ``X`` is not numeric.

        Complexity
        ----------
        Time Complexity:
            O(m * d), where m is the number of rows and d is tree depth.
        Space Complexity:
            O(m) for the returned predictions.
        """
        if self._root is None:
            raise ValueError("The DecisionTreeClassifier is not fitted yet.")

        X = self._validate_X(X)

        if X.shape[1] != self._n_features:
            raise ValueError(
                "Input array has a different number of features than "
                "the fitted tree."
            )

        predictions = [
            self._find_leaf(row).prediction
            for row in X
        ]

        return np.asarray(predictions)

    def reset(self) -> Self:
        """
        Clear all learned tree state.

        Returns
        -------
        DecisionTreeClassifier
            The reset tree instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) after previous nodes are released.
        """
        self._root = None
        self._classes = []
        self._n_features = None

        self._rng = np.random.default_rng(self.random_state)

        return self

    def _create_leaf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> _Node:
        """
        Create a leaf node containing its current observations.
        """
        leaf = _Node(depth=depth)
        leaf.X_samples = X.copy()
        leaf.y_samples = y.copy()
        leaf.prediction = self._majority_class(y)
        return leaf

    def _append_samples(
        self,
        leaf: _Node,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Append newly received observations to an existing leaf.
        """
        leaf.X_samples = np.vstack([
            leaf.X_samples,
            X
        ])

        leaf.y_samples = np.concatenate([
            leaf.y_samples,
            y
        ])

        leaf.prediction = self._majority_class(
            leaf.y_samples
        )

    def _grow_leaf(self, leaf: _Node) -> None:
        """
        Split an eligible leaf and recursively grow its child leaves.
        """
        if not leaf.is_leaf:
            return

        if not self._can_split(
            leaf
        ):
            return

        split = self._best_split(
            leaf.X_samples,
            leaf.y_samples
        )

        if split is None:
            return

        (
            feature_index,
            threshold,
            left_mask,
            right_mask
        ) = split

        leaf.feature_index = feature_index
        leaf.threshold = threshold

        leaf.left = self._create_leaf(
            X=leaf.X_samples[left_mask],
            y=leaf.y_samples[left_mask],
            depth=leaf.depth + 1
        )

        leaf.right = self._create_leaf(
            X=leaf.X_samples[right_mask],
            y=leaf.y_samples[right_mask],
            depth=leaf.depth + 1
        )

        # Internal nodes no longer need their observations.
        leaf.X_samples = None
        leaf.y_samples = None

        self._grow_leaf(leaf.left)
        self._grow_leaf(leaf.right)

    def _can_split(self, leaf: _Node) -> bool:
        """
        Return whether a leaf satisfies the splitting conditions.
        """
        if (
            self.max_depth is not None
            and leaf.depth >= self.max_depth
        ):
            return False

        if leaf.y_samples.size < self.min_samples_split:
            return False

        if np.unique(leaf.y_samples).size < 2:
            return False

        return True

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[
        int,
        float,
        np.ndarray,
        np.ndarray
    ] | None:
        """
        Return the split with the greatest Gini impurity reduction.
        """
        parent_impurity = self._gini(y)

        best_gain = 0.0
        best_split = None

        feature_indexes = self._feature_indexes(X.shape[1])

        for feature_index in feature_indexes:
            feature_values = X[:, feature_index]

            unique_values = np.unique(feature_values)

            if unique_values.size < 2:
                continue

            thresholds = (
                unique_values[:-1]
                + unique_values[1:]
            ) / 2

            for threshold in thresholds:
                left_mask = (feature_values <= threshold)
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                weighted_impurity = (
                    left_y.size / y.size
                    * self._gini(left_y)
                    + right_y.size / y.size
                    * self._gini(right_y)
                )

                gain = (parent_impurity - weighted_impurity)

                if gain > best_gain:
                    best_gain = gain

                    best_split = (
                        int(feature_index),
                        float(threshold),
                        left_mask,
                        right_mask
                    )

        return best_split

    def _feature_indexes(
        self,
        n_features: int
    ) -> np.ndarray:
        """
        Return feature indexes considered when searching for a split.
        """
        if (
            self.max_features is None
            or self.max_features == n_features
        ):
            return np.arange(n_features)

        return self._rng.choice(
            n_features,
            size=self.max_features,
            replace=False
        )

    def _gini(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for class labels.

        A value of 0.0 represents a pure node.
        """
        _, counts = np.unique(y, return_counts=True)

        probabilities = (counts / counts.sum())

        return float(
            1 - np.sum(probabilities ** 2)
        )

    def _majority_class(self, y: np.ndarray):
        """
        Return the most frequently occurring class label.
        """
        classes, counts = np.unique(y, return_counts=True)

        return classes[np.argmax(counts)]

    def _find_leaf(
        self,
        row: np.ndarray
    ) -> _Node:
        """
        Route one observation through the tree and return its leaf.
        """
        current = self._root

        while not current.is_leaf:
            if (
                row[current.feature_index]
                <= current.threshold
            ):
                current = current.left
            else:
                current = current.right

        return current

    def _append_new_classes(
        self,
        y: np.ndarray
    ) -> None:
        """
        Append newly observed labels in first-seen order.
        """
        for label in y:
            if label not in self._classes:
                self._classes.append(
                    label
                )

    def _validate_max_features_for_input(
        self,
        n_features: int
    ) -> None:
        """
        Reject ``max_features`` values larger than the input width.
        """
        if (
            self.max_features is not None
            and self.max_features > n_features
        ):
            raise ValueError(
                "max_features must not exceed the number of input features."
            )

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and return a float64 two-dimensional feature array.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must have 2 dimensions. Reshape your array first."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "X must contain at least one row."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature."
            )

        validate_numeric_array(X)
        return X.astype("float64", copy=False)

    def _validate_X_y(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray
    ]:
        """
        Validate feature and target arrays.
        """
        X = self._validate_X(X)

        if y is None:
            raise ValueError("y must be provided.")

        y = np.asarray(y).ravel()

        if y.size == 0:
            raise ValueError(
                "y must contain at least one value."
            )

        if y.size != X.shape[0]:
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if (y == None).any():  # noqa: E711
            raise ValueError(
                "y must not contain None values."
            )

        if np.issubdtype(y.dtype, np.number):
            if np.iscomplexobj(y):
                raise ValueError(
                    "y must not contain complex values."
                )

            if np.isnan(y).any():
                raise ValueError(
                    "y must not contain NaN values."
                )

            if np.isinf(y).any():
                raise ValueError(
                    "y must not contain infinite values."
                )

        return X, y
