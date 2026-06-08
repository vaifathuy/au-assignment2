import numpy as np


class Pipeline():
    """A simple implementation of an ML pipeline that contains the steps
    to perform in order. Each step is either a preprocessor (with fit()
    and transform()) or a model (with fit() and predict()).

    The pipeline will perform the steps in order, calling the appropriate
    method(s) for each step, passing the output of one step as the input
    to the next. If the pipline is called to fit_predict() or predict(),
    it will call the relevant transform() methods of any preprocessor in
    the pipeline.

    Attributes
    ----------
    steps : a list of tuples containing the names and instances of the
        steps in the pipeline
    """
    def __init__(self, steps: list[tuple[str, object]]):
        if not steps or len(steps) == 0:
            raise ValueError("Pipeline must have at least one step.")
        self.steps = steps

    def fit(self, X, y=None) -> 'Pipeline':
        """
        Fit the pipeline to the data.

        Calls ``fit()`` on each step in order. For preprocessors (steps with
        both ``fit()`` and ``transform()``), transforms ``X`` before passing
        it to the next step.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n).
        y : np.ndarray, optional
            Target labels. Passed to each step's ``fit()`` if accepted.

        Returns
        -------
        Pipeline
            The fitted pipeline instance.

        Complexity
        ----------
        Time Complexity:
            O(sum of each step's fit cost +
              sum of each step's transform cost).
        Space Complexity:
            O(m * n) for the transformed X held in memory between steps.
        """
        for _, step in self.steps:
            if hasattr(step, "fit"):
                step.fit(X, y)  # type: ignore[attr-defined]
                if hasattr(step, "transform"):
                    X = step.transform(X)  # type: ignore[attr-defined]
        return self

    def partial_fit(self, X, y=None) -> 'Pipeline':
        """
        Incrementally update the pipeline from a newly received data chunk.

        Each transformer is updated using ``partial_fit()`` and then applied
        using ``transform()`` before the chunk is passed to the next step.
        The final model receives the fully transformed chunk.

        Parameters
        ----------
        X : np.ndarray
            Incoming feature chunk of shape (m, n).
        y : np.ndarray, optional
            Target values for the incoming chunk.

        Returns
        -------
        Pipeline
            The updated pipeline instance.

        Raises
        ------
        ValueError
            If a step does not provide a ``partial_fit()`` method.

        Complexity
        ----------
        Time Complexity:
            O(sum of each step's partial-fit cost +
            sum of each transformer's transform cost).
        Space Complexity:
            O(m * n) for transformed chunks held between steps.
        """
        for name, step in self.steps:
            if hasattr(step, "partial_fit"):
                step.partial_fit(X, y)
                if hasattr(step, "transform"):
                    X = step.transform(X)
            else:
                raise ValueError(
                    f"Step '{name}' does not support partial_fit()."
                )
        return self

    def predict(self, X) -> np.ndarray:
        """
        Transform data through all preprocessors
        then predict with the final model.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (m, n).

        Returns
        -------
        np.ndarray
            Predicted labels.

        Raises
        ------
        ValueError
            If the final step does not have a ``predict()`` method.

        Complexity
        ----------
        Time Complexity:
            O(sum of each preprocessor's transform cost +
            final model's predict cost).
        Space Complexity:
            O(m * n) for the transformed X held in memory between steps.
        """

        if not hasattr(self.steps[-1][1], "predict"):
            raise ValueError("The final step in the pipeline must be a model "
                             "with a predict() method.")

        y = np.ndarray([])
        for _, step in self.steps:
            if hasattr(step, "transform"):
                X = step.transform(X)  # type: ignore[attr-defined]
            if hasattr(step, "predict"):
                y = step.predict(X)  # type: ignore[attr-defined]
                break
        return y

    def fit_predict(self, X, y=None) -> np.ndarray:
        """
        Fit all steps and predict in a single call.

        For each step, calls ``fit_transform()`` if available, otherwise
        ``fit()`` + ``transform()`` for preprocessors, or ``fit()`` +
        ``predict()`` for the final model.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n).
        y : np.ndarray, optional
            Target labels passed to each step's ``fit()`` if accepted.

        Returns
        -------
        np.ndarray
            Predicted labels from the final model step.

        Raises
        ------
        ValueError
            If the final step does not have a ``predict()`` method.

        Complexity
        ----------
        Time Complexity:
            O(sum of each step's fit cost +
            sum of each preprocessor's transform
            cost + final model's predict cost).
        Space Complexity:
            O(m * n) for the transformed X held in memory between steps.
        """

        if not hasattr(self.steps[-1][1], "predict"):
            raise ValueError("The final step in the pipeline must be a model "
                             "with a predict() method.")

        y = np.ndarray([])
        for _, step in self.steps:
            if hasattr(step, "fit_transform"):
                X = step.fit_transform(X, y)  # type: ignore[attr-defined]
            elif hasattr(step, "fit") and hasattr(step, "transform"):
                step.fit(X, y)  # type: ignore[attr-defined]
                X = step.transform(X)  # type: ignore[attr-defined]
            elif hasattr(step, "fit_predict"):
                y = step.fit_predict(X, y)  # type: ignore[attr-defined]
                break
            elif hasattr(step, "fit") and hasattr(step, "predict"):
                step.fit(X, y)  # type: ignore[attr-defined]
                y = step.predict(X)  # type: ignore[attr-defined]
                break
        return y

    def fit_transform(self, X, y=None) -> np.ndarray:
        """
        Fit all steps and transform the data in a single call.

        Calls ``fit_transform()`` on each step if available, otherwise
        ``fit()`` + ``transform()``. Steps with only ``fit()`` and
        ``predict()`` are fitted but not transformed.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n).
        y : np.ndarray, optional
            Target labels passed to each step's ``fit()`` if accepted.

        Returns
        -------
        np.ndarray
            The fully transformed data.

        Complexity
        ----------
        Time Complexity:
            O(sum of each step's fit cost + sum of each step's transform cost).
        Space Complexity:
            O(m * n) for the transformed X held in memory between steps.
        """
        for _, step in self.steps:
            if hasattr(step, "fit_transform"):
                X = step.fit_transform(X, y)  # type: ignore[attr-defined]
            elif hasattr(step, "fit") and hasattr(step, "transform"):
                step.fit(X, y)  # type: ignore[attr-defined]
                X = step.transform(X)  # type: ignore[attr-defined]
            elif hasattr(step, "fit") and hasattr(step, "predict"):
                # This case is a bit odd, but we'll allow it for completeness.
                # If a step has fit() and predict() but not transform(), we'll
                # call fit() but not transform().
                step.fit(X, y)  # type: ignore[attr-defined]
        return X


class FeatureUnion():
    """
    Combines multiple preprocessing transformers in parallel and concatenates
    their outputs. Each transformer is fitted independently on the same input
    data, then during transform, all transformers are applied in parallel and
    their outputs are concatenated horizontally (column-wise).

    Attributes
    ----------
    transformers : list of (name, transformer) tuples
        The transformers to be applied in parallel. Each transformer must have
        fit() and transform() methods compatible with _BasePreprocessor API.
    """

    def __init__(self, transformers):
        """
        Initialize FeatureUnion with a list of (name, transformer) tuples.

        Args
        ----------
        transformers : list of tuples
        List of (name, transformer) pairs where name is a string identifier
        and transformer is a preprocessing object with fit() and transform()
        methods.
        """
        self.transformers = transformers

    def fit(self, X, y=None):
        """
        Fit each transformer independently on the same input data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (m, n).
        y : np.ndarray, optional
            Target labels passed to each transformer's ``fit()`` if accepted.

        Returns
        -------
        FeatureUnion
            The fitted ``FeatureUnion`` instance.

        Complexity
        ----------
        Time Complexity:
            O(sum of each transformer's fit cost).
        Space Complexity:
            O(1) — no data is stored beyond
            each transformer's own fitted state.
        """
        for _, transformer in self.transformers:
            transformer.fit(X, y)
        return self

    def partial_fit(self, X, y=None) -> "FeatureUnion":
        """
        Incrementally update every transformer using the same incoming chunk.

        Parameters
        ----------
        X : np.ndarray
            Incoming feature chunk.
        y : np.ndarray, optional
            Target values passed to each transformer.

        Returns
        -------
        FeatureUnion
            The updated feature-union instance.

        Raises
        ------
        ValueError
            If a transformer does not provide ``partial_fit()``.

        Complexity
        ----------
        Time Complexity:
            O(sum of each transformer's partial-fit cost).
        Space Complexity:
            O(1), excluding retained transformer state.
        """
        # Loop over each transformer.
        # Verify it supports partial_fit().
        # Call transformer.partial_fit(X, y).
        # Return self.
        for name, transformer in self.transformers:
            if hasattr(transformer, "partial_fit"):
                transformer.partial_fit(X, y)
            else:
                raise ValueError(
                    f"Step '{name}' does not support partial_fit()."
                )

        return self

    def transform(self, X):
        """
        Apply all transformers in parallel and
        concatenate their outputs column-wise.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (m, n).

        Returns
        -------
        np.ndarray
            Horizontally stacked output from all transformers.

        Raises
        ------
        ValueError
            If any transformer has not yet been fitted.

        Complexity
        ----------
        Time Complexity:
            O(sum of each transformer's transform cost).
        Space Complexity:
            O(m * total output columns) for the hstacked result, where total
            output columns is the sum of
            each transformer's output column count.
        """
        transformed_features = []
        for _, transformer in self.transformers:
            transformed = transformer.transform(X)
            if transformed.ndim == 1:
                transformed = transformed.reshape(-1, 1)
            transformed_features.append(transformed)

        return np.hstack(transformed_features)
