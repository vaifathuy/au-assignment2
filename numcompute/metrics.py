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
class Accuracy:
    """
    Incrementally calculate classification accuracy from prediction chunks.

    Accuracy measures the proportion of predictions that match their
    corresponding true labels:

        accuracy = number of correct predictions / total predictions

    Unlike the standalone ``accuracy()`` function, this class is stateful.
    It preserves the accumulated number of correct predictions and total
    observations as new chunks arrive.

    Examples
    --------
    >>> metric = Accuracy()
    >>> metric.update_stats(
    ...     np.array([1, 0, 1]),
    ...     np.array([1, 0, 0])
    ... )
    >>> metric.update_stats(
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

    def update_stats(self, y_true: np.ndarray, y_pred: np.ndarray) -> Self:
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
