import tracemalloc
from typing import Self

import numpy as np

from .metrics import Accuracy, accuracy


class StreamTrainer:
    """
    Manage incremental model training, scoring, and logging over data chunks.

    The trainer accepts either a direct model or a pipeline. The supplied
    object must provide:

        partial_fit(X_chunk, y_chunk)
        predict(X_chunk)

    ``fit_chunk()`` incrementally updates the model from one incoming chunk.

    ``score_chunk()`` predicts one chunk without fitting it, calculates the
    chunk accuracy, updates cumulative accuracy, and records a log entry.

    Notes
    -----
    For realistic online evaluation, score a newly received chunk before
    fitting the model on that same chunk:

        trainer.score_chunk(X_chunk, y_chunk)
        trainer.fit_chunk(X_chunk, y_chunk)

    The first chunk normally needs to be fitted without scoring because the
    model has not learned from any earlier observations yet.

    Memory footprint is measured using ``tracemalloc``. The logged values
    represent Python memory allocations traced within the current process.
    They are useful for comparing memory growth over time, but they do not
    represent the complete operating-system memory usage of the program.

    Examples
    --------
    >>> from numcompute_stream.ensemble import EnsembleClassifier
    >>>
    >>> model = EnsembleClassifier(
    ...     n_estimators=5,
    ...     max_depth=3,
    ...     random_state=42
    ... )
    >>>
    >>> trainer = StreamTrainer(model)
    >>>
    >>> trainer.fit_chunk(
    ...     np.array([[1], [2]]),
    ...     np.array([0, 0])
    ... )
    StreamTrainer(...)
    >>>
    >>> trainer.score_chunk(
    ...     np.array([[8], [9]]),
    ...     np.array([1, 1])
    ... )
    0.0
    >>>
    >>> trainer.fit_chunk(
    ...     np.array([[8], [9]]),
    ...     np.array([1, 1])
    ... )
    StreamTrainer(...)
    """

    def __init__(self, model):
        """
        Initialize a stream trainer.

        Parameters
        ----------
        model : object
            Incremental model or pipeline. It must provide ``partial_fit()``
            and ``predict()`` methods.

        Raises
        ------
        ValueError
            If ``model`` does not provide ``partial_fit()``.
            If ``model`` does not provide ``predict()``.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1), excluding the managed model.
        """
        if not hasattr(model, "partial_fit"):
            raise ValueError("The model must support partial_fit().")

        if not hasattr(model, "predict"):
            raise ValueError("The model must support predict().")

        self.model = model

        if not tracemalloc.is_tracing():
            tracemalloc.start()

        self._accuracy_metric = Accuracy()
        self._logs = []
        self._fit_chunk_count = 0
        self._score_chunk_count = 0

    @property
    def logs(self) -> list[dict]:
        """
        Return a copy of the recorded scoring logs.

        Returns
        -------
        list of dict
            Per-chunk scoring records. Each record contains:

            - ``chunk``: score-chunk sequence number.
            - ``samples``: number of scored observations.
            - ``chunk_accuracy``: accuracy for the latest chunk only.
            - ``cumulative_accuracy``: accuracy across all scored chunks.
            - ``memory_bytes``: current traced Python-memory allocation.
            - ``peak_memory_bytes``: peak traced Python-memory allocation.

        Complexity
        ----------
        Time Complexity:
            O(n), where n is the number of log entries.
        Space Complexity:
            O(n) for the copied log entries.
        """
        return [log.copy() for log in self._logs]

    @property
    def cumulative_accuracy(self) -> float | None:
        """
        Return accuracy across all scored chunks.

        Returns
        -------
        float or None
            Cumulative classification accuracy. Returns ``None`` if no chunks
            have been scored yet.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        if self._accuracy_metric.count == 0:
            return None

        return self._accuracy_metric.result()

    @property
    def fit_chunk_count(self) -> int:
        """
        Return the number of chunks used to update the model.
        """
        return self._fit_chunk_count

    @property
    def score_chunk_count(self) -> int:
        """
        Return the number of evaluated chunks.
        """
        return self._score_chunk_count

    def fit_chunk(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Incrementally update the model from one incoming chunk.

        Parameters
        ----------
        X : np.ndarray
            Incoming feature chunk.
        y : np.ndarray
            Target labels corresponding to the incoming feature rows.

        Returns
        -------
        StreamTrainer
            The updated trainer instance.
        """
        self.model.partial_fit(X, y)
        self._fit_chunk_count += 1
        return self

    def score_chunk(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate one chunk and update cumulative accuracy.

        Parameters
        ----------
        X : np.ndarray
            Feature chunk to evaluate.
        y : np.ndarray
            Expected labels for the chunk.

        Returns
        -------
        float
            Accuracy for the evaluated chunk only.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
            If the supplied labels and predictions are empty or have
            incompatible lengths.

        Complexity
        ----------
        Time Complexity:
            O(p + m), where p is the managed model's prediction cost and m is
            the number of predictions in the chunk.
        Space Complexity:
            O(m) for predictions and temporary comparisons.
        """
        y_pred = self.model.predict(X)
        chunk_accuracy = accuracy(y, y_pred)

        # Update chunked-based metric statisticcs
        self._accuracy_metric.update_stats(y, y_pred)

        self._score_chunk_count += 1

        current_memory, peak_memory = tracemalloc.get_traced_memory()

        self._logs.append({
            "chunk": self._score_chunk_count,
            "samples": int(
                np.asarray(
                    y
                ).size
            ),
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": (
                self._accuracy_metric.result()
            ),
            "memory_bytes": current_memory,
            "peak_memory_bytes": peak_memory
        })

        return chunk_accuracy

    def reset(self) -> Self:
        """
        Clear scoring metrics and logs while preserving the fitted model.

        The model is intentionally not reset. This allows users to
        restart monitoring without discarding learned model state.

        Returns
        -------
        StreamTrainer
            The reset trainer instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) after previous logs are released.
        """
        self._accuracy_metric.reset()
        self._logs = []
        self._fit_chunk_count = 0
        self._score_chunk_count = 0

        return self

    def __repr__(self) -> str:
        """
        Return a readable trainer representation.
        """
        return (
            "StreamTrainer("
            f"model={type(self.model).__name__}, "
            f"fit_chunks={self._fit_chunk_count}, "
            f"score_chunks={self._score_chunk_count}"
            ")"
        )
