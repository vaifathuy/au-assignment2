import numpy as np
from typing import Self


def _validate_and_flatten(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert inputs to 1D NumPy arrays and validate shape compatibility.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels. Any shape; flattened internally.
    y_pred : array-like
        Predicted labels. Any shape; flattened internally.

    Returns
    -------
    y_true : np.ndarray
        Flattened ground truth array of shape (n,).
    y_pred : np.ndarray
        Flattened predicted array of shape (n,).

    Raises
    ------
    ValueError
        If either input array is empty.
        If the arrays have different shapes after flattening.

    Complexity
    ----------
    Time Complexity: 0(n)
    Space Complexity: O(n)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same length.")

    return y_true, y_pred


def _multiclass_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class TP, FP, FN, and support using broadcasting.

    Parameters
    ----------
    y_true : np.ndarray
        Flattened ground truth labels of shape (n,).
    y_pred : np.ndarray
        Flattened predicted labels of shape (n,).

    Returns
    -------
    tp : np.ndarray
        True positives per class, shape (k,).
    fp : np.ndarray
        False positives per class, shape (k,).
    fn : np.ndarray
        False negatives per class, shape (k,).
    supports : np.ndarray
        Number of true instances per class, shape (k,).
    classes : np.ndarray
        Sorted unique class labels, shape (k,).

    Complexity
    ----------
    Time Complexity: O(n * k) due to broadcasting and summation.
    Space Complexity: O(k) for the output arrays
    where n is the number of samples and k is the number of unique classes.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))

    # (k, n) boolean matrices via broadcasting:
    # true_mat[i, j] = True if y_true[j] == classes[i]
    # pred_mat[i, j] = True if y_pred[j] == classes[i]
    true_mat = (y_true[None, :] == classes[:, None])
    pred_mat = (y_pred[None, :] == classes[:, None])

    # Reduce (k × n) indicator matrices into per-class statistics.
    # Each row corresponds to one class.
    #
    # tp[i] = count of samples where:
    #   y_true == classes[i] AND y_pred == classes[i]
    #
    # fp[i] = count of samples where:
    #   y_true != classes[i] AND y_pred == classes[i]
    #
    # fn[i] = count of samples where:
    #   y_true == classes[i] AND y_pred != classes[i]
    #
    # supports[i] = total number of samples with true label = classes[i]
    #              (class frequency in ground truth)
    #
    # axis=1 collapses the sample dimension (n), leaving per-class totals.
    tp = (true_mat & pred_mat).sum(axis=1)
    fp = (~true_mat & pred_mat).sum(axis=1)
    fn = (true_mat & ~pred_mat).sum(axis=1)
    supports = true_mat.sum(axis=1)

    return tp, fp, fn, supports, classes


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy: (TP + TN) / total.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n,) or (n, 1).
    y_pred : np.ndarray
        Predicted labels of shape (n,) or (n, 1).

    Returns
    -------
    float
        Accuracy score in the range [0, 1].

    Raises
    ------
    ValueError
        If arrays have different lengths or are empty.

    Complexity
    ----------
    Time Complexity: O(n) where n is the number of samples.
    Space Complexity: O(n) for the intermediate boolean array.
    """
    y_true, y_pred = _validate_and_flatten(y_true, y_pred)

    correct = (y_true == y_pred)
    return float(np.mean(correct))


def precision(y_true: np.ndarray,
              y_pred: np.ndarray,
              average: str = "binary") -> float:
    """
    Calculate precision: TP / (TP + FP).

    For binary classification, returns precision for the positive class
    (label = 1). For multi-class, precision is computed per class via a
    one-vs-rest approach and then aggregated by ``average``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n,) or (n, 1).
    y_pred : np.ndarray
        Predicted labels of shape (n,) or (n, 1).
    average : {'binary', 'macro', 'weighted', 'micro'}, optional
        Aggregation strategy for multi-class precision. Default is 'binary'.

    Returns
    -------
    float
        Precision score.

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.
        If ``average`` is not a supported option.

    Complexity
    ----------
    Time Complexity: O(n * k) due to broadcasting and summation.
    Space Complexity: O(k) for the output arrays
    where n is the number of samples and k is the number of unique classes.
    """
    y_true, y_pred = _validate_and_flatten(y_true, y_pred)

    if average == "binary":
        pos_label = 1
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))

    tp, fp, _, supports, _ = _multiclass_stats(y_true, y_pred)

    denom = tp + fp
    precisions = np.where(denom == 0, 0.0, tp / denom)

    if average == "macro":
        return float(np.mean(precisions))

    if average == "weighted":
        return float(np.sum(precisions * supports) / np.sum(supports))

    if average == "micro":
        tp_total = np.sum(y_true == y_pred)
        fp_total = np.sum(y_true != y_pred)
        return float(tp_total / (tp_total + fp_total))

    raise ValueError("Invalid average type")


def recall(y_true: np.ndarray,
           y_pred: np.ndarray,
           average: str = "binary") -> float:
    """
    Calculate recall: TP / (TP + FN).

    For binary classification, returns recall for the positive class
    (label = 1). For multi-class, recall is computed per class via a
    one-vs-rest approach and then aggregated by ``average``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n,) or (n, 1).
    y_pred : np.ndarray
        Predicted labels of shape (n,) or (n, 1).
    average : {'binary', 'macro', 'weighted', 'micro'}, optional
        Aggregation strategy for multi-class recall. Default is 'binary'.

    Returns
    -------
    float
        Recall score.

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.
        If ``average`` is not a supported option.

    Complexity
    ----------
    Time Complexity: O(n * k) due to broadcasting and summation.
    Space Complexity: O(k) for the output arrays
    where n is the number of samples and k is the number of unique classes.
    """
    y_true, y_pred = _validate_and_flatten(y_true, y_pred)

    if average == "binary":
        pos_label = 1

        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        if tp + fn == 0:
            return 0.0

        return float(tp / (tp + fn))

    tp, _, fn, supports, _ = _multiclass_stats(y_true, y_pred)

    denom = tp + fn
    recalls = np.where(denom == 0, 0.0, tp / denom)

    if average == "macro":
        return float(np.mean(recalls))

    if average == "weighted":
        return float(np.sum(recalls * supports) / np.sum(supports))

    if average == "micro":
        tp_total = np.sum(y_true == y_pred)
        fn_total = np.sum(y_true != y_pred)
        return float(tp_total / (tp_total + fn_total))

    raise ValueError("Invalid average type")


def f1(y_true: np.ndarray,
       y_pred: np.ndarray,
       average: str = "binary") -> float:
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall).

    For binary classification, computes F1 for the positive class (label = 1).
    For multi-class, F1 is computed per class via a one-vs-rest approach and
    then aggregated by ``average``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n,) or (n, 1).
    y_pred : np.ndarray
        Predicted labels of shape (n,) or (n, 1).
    average : {'binary', 'macro', 'weighted', 'micro'}, optional
        Aggregation strategy for multi-class F1. Default is 'binary'.

    Returns
    -------
    float
        F1 score.

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.
        If ``average`` is not a supported option.

    Complexity
    ----------
    Time Complexity: O(n * k) due to broadcasting and summation.
    Space Complexity: O(k) for the output arrays
    where n is the number of samples and k is the number of unique classes.
    """
    y_true, y_pred = _validate_and_flatten(y_true, y_pred)

    if average == "binary":
        p = precision(y_true, y_pred, average="binary")
        r = recall(y_true, y_pred, average="binary")

        if p + r == 0:
            return 0.0

        return float(2 * p * r / (p + r))

    tp, fp, fn, supports, _ = _multiclass_stats(y_true, y_pred)

    p = np.where(tp + fp == 0, 0.0, tp / (tp + fp))
    r = np.where(tp + fn == 0, 0.0, tp / (tp + fn))

    denom = p + r
    f1_scores = np.where(denom == 0, 0.0, 2 * p * r / denom)

    if average == "macro":
        return float(np.mean(f1_scores))

    if average == "weighted":
        return float(np.sum(f1_scores * supports) / np.sum(supports))

    if average == "micro":
        tp_total = np.sum(y_true == y_pred)
        fp_total = np.sum(y_true != y_pred)
        fn_total = fp_total

        if tp_total == 0:
            return 0.0

        return float(2 * tp_total / (2 * tp_total + fp_total + fn_total))

    raise ValueError("Invalid average type")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix for a classification task.

    Entry C[i, j] counts samples with true label ``classes[i]`` and predicted
    label ``classes[j]``. Rows correspond to true labels; columns to predicted.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n,) or (n, 1).
    y_pred : np.ndarray
        Predicted labels of shape (n,) or (n, 1).

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (k, k), where k is
        the number of unique classes.

    Raises
    ------
    ValueError
        If inputs are empty or have mismatched lengths.

    Complexity
    ----------
    Time Complexity: O(n * k) due to broadcasting and summation.
    Space Complexity: O(k^2) for the confusion matrix output
    where n is the number of samples and k is the number of unique classes.
    """
    y_true, y_pred = _validate_and_flatten(y_true, y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    k = len(classes)
    cm = np.zeros((k, k), dtype=int)

    if k == 1:
        cm[0, 0] = y_true.size
        return cm

    true_idx = np.searchsorted(classes, y_true)
    pred_idx = np.searchsorted(classes, y_pred)
    np.add.at(cm, (true_idx, pred_idx), 1)

    return cm


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values of shape (n,) or (n, 1).
    y_pred : np.ndarray
        Predicted values of shape (n,) or (n, 1).

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have mismatched shapes.
        If either input array is empty.

    Complexity
    ----------
    Time Complexity: O(n), where n is the total number of elements.
    Space Complexity: O(n) for the intermediate difference array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} \
                         != y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")

    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true: np.ndarray,
              y_scores: np.ndarray,
              pos_label: int = 1
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
Compute Receiver Operating Characteristic (ROC) curve points.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels. Any shape; flattened internally.
    y_scores : np.ndarray
        Continuous prediction scores or probabilities,
        same shape as ``y_true``.
    pos_label : int, optional
        Label of the positive class. Default is 1.

    Returns
    -------
    fpr : np.ndarray
        False Positive Rate at each threshold, shape (t,).
    tpr : np.ndarray
        True Positive Rate at each threshold, shape (t,).
    thresholds : np.ndarray
        Score thresholds used to compute each ROC point, shape (t,).

    Raises
    ------
    ValueError
        If either input array is empty.
        If ``y_true`` and ``y_scores`` have different shapes.
        If ``y_true`` contains more than two unique classes.
        If only one class is present in ``y_true``.

    Notes
    -----
    Time Complexity: O(n log n), dominated by sorting the score array.
    Space Complexity: O(n) for the sorted indices and intermediate arrays.
    """

    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    if y_true.size == 0 or y_scores.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true.shape != y_scores.shape:
        raise ValueError("Arrays must have the same length.")

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 2:
        raise ValueError("roc_curve only supports binary classification.")

    # Convert labels into boolean
    is_pos = (y_true == pos_label)

    # Denominators for normalization
    n_pos = is_pos.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both classes.")

    # Sort scores from high to low.
    # As threshold decreases ROC curve moves
    # from (0,0) — nothing predicted positive
    # toward (1,1) — everything predicted positive.
    desc_idx = np.argsort(y_scores)[::-1]

    # Reorder arrays with the same index so
    # score stays paired with its label.
    y_scores_s = y_scores[desc_idx]
    is_pos_s = is_pos[desc_idx]

    # cumulative counts
    tps = np.cumsum(is_pos_s)
    fps = np.cumsum(~is_pos_s)

    # find indices where score value changes
    distinct_indices = np.where(np.diff(y_scores_s))[0]

    # append last index so the final score block is included as a threshold
    threshold_idxs = np.r_[distinct_indices, len(y_scores_s) - 1]

    # construct ROC curve points at valid thresholds, including origin (0,0)
    tps = np.r_[0, tps[threshold_idxs]]
    fps = np.r_[0, fps[threshold_idxs]]
    thresholds = np.r_[np.inf, y_scores_s[threshold_idxs]]

    # normalize
    fpr = fps / n_neg
    tpr = tps / n_pos

    return fpr, tpr, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates of the curve (e.g., FPR), shape (n,).
    y : np.ndarray
        y-coordinates of the curve (e.g., TPR), shape (n,).

    Returns
    -------
    float
        Area under the curve.

    Raises
    ------
    ValueError
        If either input array is empty.
        If ``x`` and ``y`` have different shapes.

    Complexity
    ----------
    Time Complexity: O(n log n), dominated by sorting the x array.
    Space Complexity: O(n) for the sorted indices and intermediate arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if x.shape != y.shape:
        raise ValueError("Arrays must have the same length.")

    # Sort by x (FPR)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Trapezoidal integration
    return float(np.trapezoid(y, x))


# Streaming Support
def _classification_stats_from_matrix(
    matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate per-class classification statistics from a confusion matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix of shape (k, k), where rows represent actual
        classes and columns represent predicted classes.

    Returns
    -------
    true_positives : np.ndarray
        Correct predictions for each class.
    false_positives : np.ndarray
        Incorrect predictions assigned to each class.
    false_negatives : np.ndarray
        Actual observations missed for each class.
    supports : np.ndarray
        Number of actual observations for each class.

    Complexity
    ----------
    Time Complexity:
        O(k²), where k is the number of classes.
    Space Complexity:
        O(k) for the returned arrays.
    """
    true_positives = np.diag(matrix)
    predicted_counts = np.sum(matrix, axis=0)
    supports = np.sum(matrix, axis=1)
    false_positives = predicted_counts - true_positives
    false_negatives = supports - true_positives

    return (true_positives, false_positives, false_negatives, supports)


def _safe_divide(
    numerators: np.ndarray,
    denominators: np.ndarray
) -> np.ndarray:
    """
    Divide corresponding values safely, returning 0.0 when a denominator
    is zero.
    """
    if numerators.shape != denominators.shape:
        raise ValueError(
            "numerators and denominators must have the same shape."
        )

    results = []

    for numerator, denominator in zip(
        numerators,
        denominators
    ):
        if denominator == 0:
            results.append(0.0)
        else:
            results.append(numerator / denominator)

    return np.array(results, dtype=float)


def _aggregate_class_scores(
    scores: np.ndarray,
    matrix: np.ndarray,
    classes: np.ndarray,
    average: str,
    pos_label
) -> float:
    """
    Aggregate per-class metric scores using the requested averaging method.
    """
    if average == "binary":
        matching_indexes = np.where(classes == pos_label)[0]

        if matching_indexes.size == 0:
            return 0.0
        positive_index = matching_indexes[0]
        return float(scores[positive_index])

    if average == "macro":
        return float(np.mean(scores))

    if average == "weighted":
        supports = np.sum(matrix, axis=1)
        return float(np.sum(scores * supports) / np.sum(supports))

    if average == "micro":
        correct_predictions = np.sum(np.diag(matrix))
        total_predictions = np.sum(matrix)
        return float(correct_predictions / total_predictions)


class Accuracy:
    """
    Incrementally calculate classification accuracy from prediction chunks.

    It measures the proportion of predictions that match their
    corresponding true labels.

    Examples
    --------
    >>> metric = Accuracy()
    >>> metric.update(
    ...     np.array([1, 0, 1]),
    ...     np.array([1, 0, 0])
    ... )
    >>> metric.update(
    ...     np.array([0, 1]),
    ...     np.array([0, 1])
    ... )
    >>> metric.result()
    0.8
    """
    def __init__(self):
        self.reset()

    @property
    def correct_count(self) -> int:
        """
        Return the accumulated number of correct predictions.
        """
        return self._correct_count

    @property
    def count(self) -> int:
        """
        Return the accumulated number of processed predictions.
        """
        return self._total_count

    def reset(self) -> Self:
        """
        Clear all accumulated accuracy statistics.

        Returns
        -------
        Accuracy
            The reset metric instance.

        Complexity
        ----------
        Time Complexity:
            O(1)
        Space Complexity:
            O(1)
        """
        self._correct_count = 0
        self._total_count = 0
        return self

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> Self:
        """
        Incrementally update accuracy statistics from a prediction chunk.

        Previously accumulated statistics are preserved. The method counts
        matching labels in the incoming chunk and adds them to the retained
        totals.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels for the incoming chunk. Any shape is accepted
            because the values are flattened internally.
        y_pred : np.ndarray
            Predicted labels for the incoming chunk. Must contain the same
            number of values as ``y_true``.

        Returns
        -------
        Accuracy
            The updated metric instance.

        Raises
        ------
        ValueError
            If either input is empty.
            If the flattened arrays have different lengths.

        Complexity
        ----------
        Time Complexity:
            O(m), where m is the number of predictions in the incoming chunk.
        Space Complexity:
            O(m) for the temporary flattened arrays and comparison result.
            Retained metric state requires O(1) space.
        """
        y_true, y_pred = _validate_and_flatten(
            y_true,
            y_pred
        )

        correcte_preds = np.sum(y_true == y_pred)
        self._correct_count += int(correcte_preds)
        self._total_count += y_true.size
        return self

    def result(self) -> float:
        """
        Return the accumulated accuracy score.

        Returns
        -------
        float
            Accuracy score in the range [0, 1].

        Raises
        ------
        ValueError
            If no prediction chunks have been processed.

        Complexity
        ----------
        Time Complexity:
            O(1)
        Space Complexity:
            O(1)
        """
        if self._total_count == 0:
            raise ValueError(
                "No predictions have been accumulated yet."
            )

        return self._correct_count / self._total_count


class MSE:
    """
    Incrementally calculate mean squared error from prediction chunks.

    It measures the average squared difference between
    predicted and true values:

        MSE = sum((y_true - y_pred) ** 2) / number of observations

    Examples
    --------
    >>> metric = MSE()
    >>> metric.update(
    ...     np.array([1, 2, 3]),
    ...     np.array([1, 4, 2])
    ... )
    >>> metric.result()
    1.6666666666666667
    """

    def __init__(self):
        self.reset()

    @property
    def squared_error_sum(self) -> float:
        """
        Return the accumulated sum of squared errors.
        """
        return self._squared_error_sum

    @property
    def count(self) -> int:
        """
        Return the total number of processed values.
        """
        return self._count

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> "MSE":
        """
        Incrementally update the accumulated squared error.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth numeric values for the incoming chunk.
        y_pred : np.ndarray
            Predicted numeric values for the incoming chunk. Must have the
            same shape as ``y_true``.

        Returns
        -------
        MSE
            The updated metric instance.

        Raises
        ------
        ValueError
            If the input arrays have different shapes.
            If either input array is empty.
        TypeError
            If subtraction cannot be performed because values are not numeric.

        Complexity
        ----------
        Time Complexity:
            O(m), where m is the number of values in the incoming chunk.
        Space Complexity:
            O(m) for the temporary squared-error array.
            Retained metric state requires O(1) space.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} "
                f"!= y_pred {y_pred.shape}"
            )

        if y_true.size == 0:
            raise ValueError(
                "Input arrays must not be empty."
            )

        try:
            squared_errors = (y_true - y_pred) ** 2
        except TypeError as exc:
            raise TypeError(
                "Input arrays must contain numeric values."
            ) from exc

        self._squared_error_sum += float(np.sum(squared_errors))
        self._count += y_true.size

        return self

    def result(self) -> float:
        """
        Return the accumulated mean squared error.

        Returns
        -------
        float
            Mean squared error across all processed chunks.

        Raises
        ------
        ValueError
            If no prediction chunks have been processed.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        if self._count == 0:
            raise ValueError(
                "No predictions have been accumulated yet."
            )

        return self._squared_error_sum / self._count

    def reset(self) -> "MSE":
        """
        Clear all accumulated squared-error statistics.

        Returns
        -------
        MSE
            The reset metric instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        self._squared_error_sum = 0.0
        self._count = 0

        return self


class ConfusionMatrix:
    """
    Incrementally accumulate a confusion matrix from prediction chunks.

    Examples
    --------
    >>> metric = ConfusionMatrix()
    >>> metric.update(
    ...     np.array([0, 0, 1, 1]),
    ...     np.array([0, 1, 1, 1])
    ... )
    >>> metric.result()
    array([[1, 1],
           [0, 2]])

    >>> metric.update(
    ...     np.array([2, 1, 2]),
    ...     np.array([2, 2, 1])
    ... )
    >>> metric.classes
    array([0, 1, 2])
    >>> metric.result()
    array([[1, 1, 0],
           [0, 2, 1],
           [0, 1, 1]])
    """

    def __init__(self):
        self.reset()

    @property
    def classes(self) -> np.ndarray:
        """
        Return a copy of the known class labels in stable column order.
        """
        return np.asarray(self._classes).copy()

    @property
    def count(self) -> int:
        """
        Return the total number of processed predictions.
        """
        return self._count

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> Self:
        """
        Incrementally update the confusion matrix from a prediction chunk.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels for the incoming chunk. Any shape is accepted
            because values are flattened internally.
        y_pred : np.ndarray
            Predicted labels for the incoming chunk. Must contain the same
            number of values as ``y_true``.

        Returns
        -------
        ConfusionMatrix
            The updated instance.

        Raises
        ------
        ValueError
            If either input array is empty.
            If the flattened arrays have different lengths.
        TypeError
            If a class label is not hashable.

        Complexity
        ----------
        Time Complexity:
            O(m) in the usual case, where m is the number of predictions
            in the incoming chunk.

            If new classes appear, matrix expansion requires O(k²), where
            k is the updated total number of known classes.
        Space Complexity:
            O(k²) for the retained confusion matrix.
        """
        y_true, y_pred = _validate_and_flatten(y_true, y_pred)

        # append new classes
        old_class_count = len(self._classes)

        for label in np.concatenate([y_true, y_pred]):
            if label not in self._class_to_index:
                self._class_to_index[label] = len(self._classes)
                self._classes.append(label)

        new_class_count = len(self._classes)

        if new_class_count != old_class_count:
            expanded_matrix = np.zeros(
                (new_class_count, new_class_count),
                dtype=int
            )

            expanded_matrix[:old_class_count, :old_class_count] = self._matrix
            self._matrix = expanded_matrix

        true_indexes = np.array(
            [self._class_to_index[label] for label in y_true],
            dtype=int
        )
        pred_indexes = np.array(
            [self._class_to_index[label] for label in y_pred],
            dtype=int
        )

        # Increment every true-label / predicted-label pair.
        np.add.at(self._matrix, (true_indexes, pred_indexes), 1)

        self._count += y_true.size

        return self

    def result(self) -> np.ndarray:
        """
        Return a copy of the accumulated confusion matrix.

        Returns
        -------
        np.ndarray
            Confusion matrix of shape (k, k), where k is the number of
            observed classes.

        Raises
        ------
        ValueError
            If no predictions have been accumulated.

        Complexity
        ----------
        Time Complexity:
            O(k²) for returning a defensive copy.
        Space Complexity:
            O(k²) for the returned array.
        """
        if self._count == 0:
            raise ValueError(
                "No predictions have been accumulated yet."
            )

        return self._matrix.copy()

    def reset(self) -> Self:
        """
        Clear all accumulated classes and matrix counts.

        Returns
        -------
        ConfusionMatrix
            The reset instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        self._classes = []
        self._class_to_index = {}
        self._matrix = np.zeros((0, 0), dtype=int)
        self._count = 0

        return self


class Precision:
    """
    Incrementally calculate precision from prediction chunks.

    The class reuses ``ConfusionMatrix`` to retain class-to-class counts as
    prediction chunks arrive.

    Examples
    --------
    Binary precision:

    >>> metric = Precision()
    >>> metric.update(
    ...     np.array([1, 0, 1, 1]),
    ...     np.array([1, 1, 0, 1])
    ... )
    >>> metric.result()
    0.6666666666666666

    Multiclass macro precision:

    >>> metric = Precision(average="macro")
    >>> metric.update(
    ...     np.array([0, 1, 2, 0]),
    ...     np.array([0, 2, 2, 0])
    ... )
    >>> metric.result()
    0.5
    """

    def __init__(
        self,
        average: str = "binary",
        pos_label=1
    ):
        """
        Initialize an empty streaming precision tracker.

        Parameters
        ----------
        average : {'binary', 'macro', 'weighted', 'micro'}, optional
            Determines how precision is calculated.

            - ``'binary'`` returns precision for ``pos_label``.
            - ``'macro'`` returns the unweighted mean of per-class precision.
            - ``'weighted'`` weights per-class precision by class support.
            - ``'micro'`` calculates precision from the total number of
              correct and incorrect predictions.

            Default is ``'binary'``.
        pos_label : optional
            Class label treated as positive when ``average='binary'``.
            Default is 1.

        Raises
        ------
        ValueError
            If ``average`` is not supported.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) before prediction chunks are processed.
        """
        if average not in {
            "binary",
            "macro",
            "weighted",
            "micro"
        }:
            raise ValueError(
                "Invalid average type. Use 'binary', 'macro', "
                "'weighted', or 'micro'."
            )

        self.average = average
        self.pos_label = pos_label
        self._confusion_matrix = ConfusionMatrix()

    @property
    def classes(self) -> np.ndarray:
        """
        Return a copy of the known class labels.
        """
        return self._confusion_matrix.classes

    @property
    def count(self) -> int:
        """
        Return the total number of processed predictions.
        """
        return self._confusion_matrix.count

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> Self:
        """
        Incrementally update precision statistics from a prediction chunk.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels for the incoming chunk.
        y_pred : np.ndarray
            Predicted labels for the incoming chunk. Must contain the same
            number of values as ``y_true``.

        Returns
        -------
        Precision
            The updated instance.

        Raises
        ------
        ValueError
            If either input array is empty.
            If the flattened arrays have different lengths.

        Complexity
        ----------
        Time Complexity:
            O(m) in the usual case, where m is the number of predictions
            in the incoming chunk.

            If new classes appear, matrix expansion requires O(k²), where
            k is the updated number of known classes.
        Space Complexity:
            O(k²) for the retained confusion matrix.
        """
        self._confusion_matrix.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """
        Return the accumulated precision score.
        """
        matrix = self._confusion_matrix.result()
        classes = self._confusion_matrix.classes
        precisions = self._precision_from_matrix(matrix)

        return _aggregate_class_scores(
            scores=precisions,
            matrix=matrix,
            classes=classes,
            average=self.average,
            pos_label=self.pos_label
        )

    def reset(self) -> Self:
        """
        Clear all accumulated precision statistics.

        Returns
        -------
        Precision
            The reset instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        self._confusion_matrix.reset()

        return self

    @staticmethod
    def _precision_from_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Calculate precision separately for each known class.
        """
        (
            true_positives,
            false_positives,
            _,
            _
        ) = _classification_stats_from_matrix(matrix)

        return _safe_divide(
            true_positives,
            true_positives + false_positives
        )


class Recall:
    """
    Incrementally calculate recall from prediction chunks.

    The class reuses ``ConfusionMatrix`` to retain class-to-class counts as
    prediction chunks arrive.

    Examples
    --------
    Binary recall:

    >>> metric = Recall()
    >>> metric.update(
    ...     np.array([1, 0, 1, 1]),
    ...     np.array([1, 1, 0, 1])
    ... )
    >>> metric.result()
    0.6666666666666666

    Multiclass macro recall:

    >>> metric = Recall(average="macro")
    >>> metric.update(
    ...     np.array([0, 1, 2, 0]),
    ...     np.array([0, 2, 2, 0])
    ... )
    >>> metric.result()
    0.6666666666666666
    """

    def __init__(
        self, average: str = "binary",
        pos_label=1
    ):
        """
        Initialize an empty streaming recall tracker.

        Parameters
        ----------
        average : {'binary', 'macro', 'weighted', 'micro'}, optional
            Determines how recall is calculated.

            - ``'binary'`` returns recall for ``pos_label``.
            - ``'macro'`` returns the unweighted mean of per-class recall.
            - ``'weighted'`` weights per-class recall by class support.
            - ``'micro'`` calculates recall from the total number of
              correct and incorrect predictions.

            Default is ``'binary'``.
        pos_label : optional
            Class label treated as positive when ``average='binary'``.
            Default is 1.

        Raises
        ------
        ValueError
            If ``average`` is not supported.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) before prediction chunks are processed.
        """
        if average not in {
            "binary",
            "macro",
            "weighted",
            "micro"
        }:
            raise ValueError(
                "Invalid average type. Use 'binary', 'macro', "
                "'weighted', or 'micro'."
            )

        self.average = average
        self.pos_label = pos_label
        self._confusion_matrix = ConfusionMatrix()

    @property
    def classes(self) -> np.ndarray:
        """
        Return a copy of the known class labels.
        """
        return self._confusion_matrix.classes

    @property
    def count(self) -> int:
        """
        Return the total number of processed predictions.
        """
        return self._confusion_matrix.count

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> Self:
        """
        Incrementally update recall statistics from a prediction chunk.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels for the incoming chunk.
        y_pred : np.ndarray
            Predicted labels for the incoming chunk. Must contain the same
            number of values as ``y_true``.

        Returns
        -------
        Recall
            The updated instance.

        Raises
        ------
        ValueError
            If either input array is empty.
            If the flattened arrays have different lengths.

        Complexity
        ----------
        Time Complexity:
            O(m) in the usual case, where m is the number of predictions
            in the incoming chunk.

            If new classes appear, matrix expansion requires O(k²), where
            k is the updated number of known classes.
        Space Complexity:
            O(k²) for the retained confusion matrix.
        """
        self._confusion_matrix.update(y_true, y_pred)

        return self

    def result(self) -> float:
        """
        Return the accumulated recall score.
        """
        matrix = self._confusion_matrix.result()
        classes = self._confusion_matrix.classes
        recalls = self._recall_from_matrix(matrix)

        return _aggregate_class_scores(
            scores=recalls,
            matrix=matrix,
            classes=classes,
            average=self.average,
            pos_label=self.pos_label
        )

    def reset(self) -> Self:
        """
        Clear all accumulated recall statistics.

        Returns
        -------
        Recall
            The reset instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1).
        """
        self._confusion_matrix.reset()

        return self

    @staticmethod
    def _recall_from_matrix(
        matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate recall separately for each known class.
        """
        (
            true_positives,
            _,
            false_negatives,
            _
        ) = _classification_stats_from_matrix(
            matrix
        )

        return _safe_divide(
            true_positives,
            true_positives + false_negatives
        )


class F1:
    """
    Incrementally calculate the F1 score from prediction chunks.

    The class reuses ``ConfusionMatrix`` to retain class-to-class counts.
    """

    def __init__(
        self,
        average: str = "binary",
        pos_label=1
    ):
        """
        Initialize an empty streaming F1 tracker.

        Parameters
        ----------
        average : {'binary', 'macro', 'weighted', 'micro'}, optional
            Determines how F1 is calculated.
        pos_label : optional
            Class label treated as positive when ``average='binary'``.
            Default is 1.

        Raises
        ------
        ValueError
            If ``average`` is not supported.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) before prediction chunks are processed.
        """
        if average not in {
            "binary",
            "macro",
            "weighted",
            "micro"
        }:
            raise ValueError(
                "Invalid average type. Use 'binary', 'macro', "
                "'weighted', or 'micro'."
            )

        self.average = average
        self.pos_label = pos_label
        self._confusion_matrix = ConfusionMatrix()

    @property
    def classes(self) -> np.ndarray:
        """
        Return a copy of the known class labels.
        """
        return self._confusion_matrix.classes

    @property
    def count(self) -> int:
        """
        Return the total number of processed predictions.
        """
        return self._confusion_matrix.count

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> Self:
        """
        Incrementally update F1 statistics from a prediction chunk.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels for the incoming chunk.
        y_pred : np.ndarray
            Predicted labels for the incoming chunk.

        Returns
        -------
        F1
            The updated instance.
        """
        self._confusion_matrix.update(
            y_true,
            y_pred
        )

        return self

    def result(self) -> float:
        """
        Return the accumulated F1 score.

        Returns
        -------
        float
            F1 score calculated using the configured averaging method.

        Raises
        ------
        ValueError
            If no predictions have been accumulated.
        """
        matrix = self._confusion_matrix.result()
        classes = self._confusion_matrix.classes
        precisions = Precision._precision_from_matrix(matrix)
        recalls = Recall._recall_from_matrix(matrix)
        f1_scores = self._f1_from_precision_and_recall(precisions, recalls)

        return _aggregate_class_scores(
            scores=f1_scores,
            matrix=matrix,
            classes=classes,
            average=self.average,
            pos_label=self.pos_label
        )

    @staticmethod
    def _f1_from_precision_and_recall(
        precisions: np.ndarray,
        recalls: np.ndarray
    ) -> np.ndarray:
        """
        Calculate per-class F1 scores from precision and recall values.

        Parameters
        ----------
        precisions : np.ndarray
            Per-class precision values.
        recalls : np.ndarray
            Per-class recall values.

        Returns
        -------
        np.ndarray
            Per-class F1 values.
        """
        f1_scores = []

        for precision, recall in zip(
            precisions,
            recalls
        ):
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(
                    2 * precision * recall
                    / (precision + recall)
                )

        return np.array(f1_scores, dtype=float)

    def reset(self) -> Self:
        """
        Clear all accumulated F1 statistics.

        Returns
        -------
        F1
            The reset instance.
        """
        self._confusion_matrix.reset()

        return self


class ROCAUC:
    """
    Incrementally retain labels and prediction scores for exact ROC-AUC
    calculation.

    Examples
    --------
    >>> metric = ROCAUC()
    >>> metric.update(
    ...     np.array([0, 1]),
    ...     np.array([0.10, 0.90])
    ... )
    >>> metric.update(
    ...     np.array([1, 0]),
    ...     np.array([0.60, 0.40])
    ... )
    >>> metric.result()
    1.0
    """

    def __init__(
        self,
        pos_label=1
    ):
        """
        Initialize an empty streaming ROC-AUC tracker.

        Parameters
        ----------
        pos_label : optional
            Label treated as the positive class. Default is 1.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) before prediction chunks are processed.
        """
        self.pos_label = pos_label
        self.reset()

    @property
    def count(self) -> int:
        """
        Return the total number of retained observations.
        """
        return self._count

    @property
    def y_true(self) -> np.ndarray:
        """
        Return a copy of all retained ground-truth labels.

        Returns
        -------
        np.ndarray
            Flattened array containing all retained labels.

        Complexity
        ----------
        Time Complexity:
            O(n), where n is the total number of retained observations.
        Space Complexity:
            O(n) for the returned array.
        """
        if self._count == 0:
            return np.array([])

        return np.concatenate(
            self._y_true_chunks
        ).copy()

    @property
    def y_scores(self) -> np.ndarray:
        """
        Return a copy of all retained prediction scores.

        Returns
        -------
        np.ndarray
            Flattened array containing all retained scores.

        Complexity
        ----------
        Time Complexity:
            O(n), where n is the total number of retained observations.
        Space Complexity:
            O(n) for the returned array.
        """
        if self._count == 0:
            return np.array(
                [],
                dtype=float
            )

        return np.concatenate(
            self._y_score_chunks
        ).copy()

    def update(self, y_true: np.ndarray, y_scores: np.ndarray) -> Self:
        """
        Incrementally retain labels and prediction scores from a new chunk.

        Previously received chunks are preserved. Input arrays are flattened
        internally and copied so later changes to the caller's arrays cannot
        modify the retained state.

        A chunk may contain only one class. Both positive and negative classes
        are required only when ``roc_curve()`` or ``result()`` is called.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels for the incoming chunk.
        y_scores : np.ndarray
            Continuous prediction scores or probabilities for the incoming
            chunk. Must contain the same number of values as ``y_true``.

        Returns
        -------
        ROCAUC
            The updated instance.

        Raises
        ------
        ValueError
            If either array is empty.
            If the flattened arrays have different lengths.
            If ``y_scores`` contains NaN or infinite values.
        TypeError
            If ``y_scores`` cannot be converted to numeric values.

        Complexity
        ----------
        Time Complexity:
            O(m), where m is the number of values in the incoming chunk.
        Space Complexity:
            O(m) additional retained memory for the incoming chunk.
            Total retained memory becomes O(n), where n is the total number
            of observations received so far.
        """
        y_true, y_scores = _validate_and_flatten(y_true, y_scores)

        try:
            y_scores = y_scores.astype("float64")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "y_scores must contain numeric values."
            ) from exc

        if (
            np.isnan(y_scores).any()
            or np.isinf(y_scores).any()
        ):
            raise ValueError(
                "y_scores must not contain NaN or infinite values."
            )

        self._y_true_chunks.append(y_true.copy())
        self._y_score_chunks.append(y_scores.copy())
        self._count += y_true.size

        return self

    def roc_curve(
        self
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]:
        """
        Calculate the exact ROC curve from all retained observations.

        Returns
        -------
        fpr : np.ndarray
            False-positive rates at each threshold.
        tpr : np.ndarray
            True-positive rates at each threshold.
        thresholds : np.ndarray
            Score thresholds used to calculate the ROC points.

        Raises
        ------
        ValueError
            If no observations have been accumulated.
            If more than two classes are present.
            If the retained labels do not include both positive and negative
            observations.

        Complexity
        ----------
        Time Complexity:
            O(n log n), dominated by sorting prediction scores.
        Space Complexity:
            O(n) for the combined arrays and ROC calculations.
        """
        if self._count == 0:
            raise ValueError(
                "No predictions have been accumulated yet."
            )

        return roc_curve(
            self.y_true,
            self.y_scores,
            pos_label=self.pos_label
        )

    def result(self) -> float:
        """
        Return the exact accumulated ROC-AUC score.

        Returns
        -------
        float
            Area under the ROC curve.

        Raises
        ------
        ValueError
            If the ROC curve cannot be calculated from the retained
            observations.

        Complexity
        ----------
        Time Complexity:
            O(n log n), dominated by ROC-curve calculation.
        Space Complexity:
            O(n) for the combined arrays and ROC calculations.
        """
        fpr, tpr, _ = self.roc_curve()

        return auc(fpr, tpr)

    def reset(self) -> Self:
        """
        Clear all retained labels and prediction scores.

        Returns
        -------
        ROCAUC
            The reset instance.

        Complexity
        ----------
        Time Complexity:
            O(1).
        Space Complexity:
            O(1) after previously retained chunks are released.
        """
        self._y_true_chunks = []
        self._y_score_chunks = []
        self._count = 0

        return self
