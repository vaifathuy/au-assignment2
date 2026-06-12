import numpy as np

from typing import Self
from .tree import DecisionTreeClassifier
from .utils import validate_numeric_array


class EnsembleClassifier:
    """
    A streaming bagging ensemble built from multiple decision trees.

    Each decision tree receives a randomly sampled version of every incoming
    chunk. Sampling is performed with replacement, allowing the trees to learn
    slightly different patterns from the same stream.

    Predictions are combined using majority voting.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | None = None,
        random_state: int | None = None
    ):
        """
        Initialize an empty ensemble of decision trees.

        Parameters
        ----------
        n_estimators : int, optional
            Number of decision trees created by the ensemble.
            Default is 10.
        max_depth : int or None, optional
            Maximum depth passed to each tree.
            Default is None.
        min_samples_split : int, optional
            Minimum number of samples required before a tree leaf can split.
            Default is 2.
        max_features : int or None, optional
            Number of randomly selected features considered by each tree when
            searching for a split. If None, every feature is considered.
            Default is None.
        random_state : int or None, optional
            Seed used for reproducible bootstrap sampling and tree seeds.

        Raises
        ------
        TypeError
            If ``n_estimators`` or ``random_state`` has an unsupported type.
        ValueError
            If ``n_estimators`` is not greater than zero.

        Complexity
        ----------
        Time Complexity:
            O(n_estimators) to create the trees.
        Space Complexity:
            O(n_estimators) before training.
        """
        if type(n_estimators) is not int:
            raise TypeError("n_estimators must be an integer.")

        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")

        if (
            random_state is not None
            and type(random_state) is not int
        ):
            raise TypeError("random_state must be an integer or None.")

        self.n_estimators = n_estimators

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self.random_state = random_state

        self.reset()

    @property
    def trees(self) -> list[DecisionTreeClassifier]:
        """
        Return a copy of the list containing the decision trees.
        """
        return self._trees.copy()

    @property
    def classes(self) -> np.ndarray:
        """
        Return a copy of the labels observed during training.
        """
        return np.asarray(
            self._classes
        ).copy()

    @property
    def count(self) -> int:
        """
        Return the total number of original observations processed,
        excluding duplicated rows created during bootstrap.
        """
        return self._count

    @property
    def n_features(self) -> int | None:
        """
        Return the number of feature columns observed during training.
        """
        return self._n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Reset the ensemble and train it from one complete dataset.

        Parameters
        ----------
        X : np.ndarray
            Numeric training features of shape (m, n).
        y : np.ndarray
            Class labels containing m observations.

        Returns
        -------
        EnsembleClassifier
            The fitted ensemble instance.

        Complexity
        ----------
        Time Complexity:
            Depends on the training cost of all decision trees.
        Space Complexity:
            Depends on the samples retained by all decision trees.
        """
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Self:
        """
        Incrementally update every decision tree from an incoming chunk.

        Parameters
        ----------
        X_chunk : np.ndarray
            Numeric feature chunk of shape (m, n).
        y_chunk : np.ndarray
            Labels corresponding to the incoming feature rows.

        Returns
        -------
        EnsembleClassifier
            The updated ensemble instance.

        Raises
        ------
        ValueError
            If arrays are empty, contain invalid values, have incompatible
            shapes, or use a different number of features from earlier
            chunks.
        TypeError
            If ``X_chunk`` is not numeric.

        Complexity
        ----------
        Time Complexity:
            O(n_estimators * m), excluding the tree-growth cost, where m is
            the number of rows in the incoming chunk.
        Space Complexity:
            O(m * n) for each temporary bootstrap sample, excluding the
            samples retained by the trees.
        """
        X_chunk = np.asarray(X_chunk)

        if X_chunk.ndim != 2:
            raise ValueError(
                "X_chunk must have 2 dimensions. Reshape your array first."
            )

        if X_chunk.shape[0] == 0:
            raise ValueError("X_chunk must contain at least one row.")

        if X_chunk.shape[1] == 0:
            raise ValueError("X_chunk must contain at least one feature.")

        validate_numeric_array(X_chunk)

        X_chunk = X_chunk.astype("float64", copy=False)

        if y_chunk is None:
            raise ValueError("y_chunk must be provided.")

        y_chunk = np.asarray(y_chunk).ravel()

        if y_chunk.size == 0:
            raise ValueError("y_chunk must contain at least one value.")

        if y_chunk.size != X_chunk.shape[0]:
            raise ValueError(
                "X_chunk and y_chunk must contain the same number of rows."
            )

        for label in y_chunk:
            if label is None:
                raise ValueError("y_chunk must not contain None values.")

            if isinstance(label, (complex, np.complexfloating)):
                raise ValueError(
                    "y_chunk must not contain complex values."
                )

            if isinstance(label, (float, np.floating)):
                if (
                    np.isnan(label)
                    or np.isinf(label)
                ):
                    raise ValueError(
                        "y_chunk must not contain NaN or infinite values."
                    )

        if self._n_features is None:
            self._n_features = X_chunk.shape[1]

        elif X_chunk.shape[1] != self._n_features:
            raise ValueError(
                "Input chunk has a different number of features than "
                "previous chunks."
            )

        # Append new classes
        for label in y_chunk:
            if label not in self._classes:
                self._classes.append(label)

        for tree in self._trees:
            sample_indexes = self._rng.choice(
                X_chunk.shape[0],
                size=X_chunk.shape[0],
                replace=True
            )

            tree.partial_fit(
                X_chunk[sample_indexes],
                y_chunk[sample_indexes]
            )

        self._count += X_chunk.shape[0]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using majority voting across all decision trees.

        Parameters
        ----------
        X : np.ndarray
            Numeric features of shape (m, n).

        Returns
        -------
        np.ndarray
            Predicted label for each row.

        Raises
        ------
        ValueError
            If the ensemble has not been fitted.

        Complexity
        ----------
        Time Complexity:
            O(n_estimators * m * d), where m is the number of rows and d is
            the typical tree depth.
        Space Complexity:
            O(n_estimators * m) for tree predictions.
        """
        if self._count == 0:
            raise ValueError("The EnsembleClassifier is not fitted yet.")

        tree_predictions = np.array([
            tree.predict(X)
            for tree in self._trees
        ])

        final_predictions = []

        for row_predictions in tree_predictions.T:
            labels, counts = np.unique(
                row_predictions,
                return_counts=True
            )

            majority_label = labels[
                np.argmax(counts)
            ]

            final_predictions.append(majority_label)

        return np.asarray(final_predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy for the supplied dataset.

        Parameters
        ----------
        X : np.ndarray
            Numeric features of shape (m, n).
        y : np.ndarray
            Expected labels.

        Returns
        -------
        float
            Fraction of correctly predicted rows.

        Raises
        ------
        ValueError
            If X and y do not contain the same number of rows.

        Complexity
        ----------
        Time Complexity:
            Dominated by ``predict()``.
        Space Complexity:
            O(m) for predictions.
        """
        predictions = self.predict(X)
        y = np.asarray(y).ravel()

        if y.size != predictions.size:
            raise ValueError("X and y must contain the same number of rows.")

        return float(np.mean(predictions == y))

    def reset(self) -> Self:
        """
        Recreate an empty ensemble of decision trees.

        Returns
        -------
        EnsembleClassifier
            The reset ensemble instance.

        Complexity
        ----------
        Time Complexity:
            O(n_estimators).
        Space Complexity:
            O(n_estimators).
        """
        self._rng = np.random.default_rng(self.random_state)

        self._trees = []

        for _ in range(self.n_estimators):
            tree_random_state = int(
                self._rng.integers(
                    0,
                    2**32 - 1
                )
            )

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_random_state
            )

            self._trees.append(tree)

        self._classes = []
        self._count = 0
        self._n_features = None

        return self
