import numpy as np
import numpy.testing as npt
from numcompute.metrics import (
    Accuracy,
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    mse,
    auc,
    roc_curve
)


class TestAccuracy:
    def test_accuracy_basic(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])

        expected = 0.8
        result = accuracy(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_accuracy_all_correct(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        expected = 1
        result = accuracy(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_accuracy_all_wrong(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        expected = 0.0
        result = accuracy(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_accuracy_multiclass(self):
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 2, 2, 1])

        expected = 0.75
        result = accuracy(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_accuracy_2d_inputs(self):
        y_true = np.array([[1], [0], [1]])
        y_pred = np.array([[1], [0], [0]])

        expected = 0.666666666
        result = accuracy(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_accuracy_empty_both(self):
        npt.assert_raises(ValueError, accuracy, np.array([]), np.array([]))

    def test_accuracy_one_empty(self):
        npt.assert_raises(ValueError, accuracy, np.array([]), np.array([1]))

    def test_accuracy_length_mismatch(self):
        npt.assert_raises(
            ValueError,
            accuracy,
            np.array([1, 0, 1]),
            np.array([1])
        )

    def test_accuracy_2d_and_1d_equivalent(self):
        y_true = np.array([[1], [0], [1]])
        y_pred = np.array([1, 0, 1])

        expected = 1.0
        result = accuracy(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_accuracy_shape_mismatch_after_ravel(self):
        npt.assert_raises(
            ValueError,
            accuracy,
            np.array([[1], [0], [1]]),
            np.array([1, 0])
        )


class TestAccuracyStream:
    def test_update_stats_single_chunk(self):
        metric = Accuracy()
        metric.update_stats(
            np.array([1, 0, 1, 1]),
            np.array([1, 0, 0, 1])
        )
        npt.assert_equal(metric.correct_count, 3)
        npt.assert_equal(metric.count, 4)
        npt.assert_allclose(
            metric.result(),
            0.75
        )

    def test_update_stats_accumulates_multiple_chunks(self):
        metric = Accuracy()
        metric.update_stats(
            np.array([1, 0, 1]),
            np.array([1, 0, 0])
        )
        npt.assert_equal(metric.correct_count, 2)
        npt.assert_equal(metric.count, 3)

        metric.update_stats(
            np.array([0, 1]),
            np.array([0, 1])
        )
        npt.assert_equal(metric.correct_count, 4)
        npt.assert_equal(metric.count, 5)

        npt.assert_allclose(
            metric.result(),
            0.8
        )

    def test_supports_2d_inputs(self):
        metric = Accuracy()
        metric.update_stats(
            np.array([
                [1],
                [0],
                [1]
            ]),
            np.array([
                [1],
                [0],
                [0]
            ])
        )
        npt.assert_equal(metric.correct_count, 2)
        npt.assert_equal(metric.count, 3)
        npt.assert_allclose(
            metric.result(),
            0.6666666667
        )

    def test_reset_clears_accumulated_statistics(self):
        metric = Accuracy()

        metric.update_stats(
            np.array([1, 0, 1]),
            np.array([1, 0, 0])
        )

        metric.reset()

        npt.assert_equal(metric.correct_count, 0)
        npt.assert_equal(metric.count, 0)


class TestPrecisionBinary:
    def test_precision_basic(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])

        expected = 1.0
        result = precision(y_true, y_pred)

        npt.assert_almost_equal(result, expected)

    def test_precision_all_correct_positive(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])

        expected = 1.0
        result = precision(y_true, y_pred)

        npt.assert_almost_equal(result, expected)

    def test_precision_no_predicted_positive(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0, 0, 0])

        expected = 0.0
        result = precision(y_true, y_pred)

        npt.assert_almost_equal(result, expected)

    def test_precision_all_false_positive(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])

        expected = 0.0
        result = precision(y_true, y_pred)

        npt.assert_almost_equal(result, expected)

    def test_precision_mixed_case(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

        expected = 0.6666666666
        result = precision(y_true, y_pred)

        npt.assert_almost_equal(result, expected)

    def test_precision_2d_inputs(self):
        y_true = np.array([[1], [0], [1]])
        y_pred = np.array([[1], [1], [1]])

        expected = 0.666666666
        result = precision(y_true, y_pred)

        npt.assert_almost_equal(result, expected)

    def test_precision_empty(self):
        npt.assert_raises(ValueError, precision, [], [])

    def test_precision_length_mismatch(self):
        npt.assert_raises(ValueError, precision, [1, 0], [1])


class TestPrecisionMulticlass:
    def test_precision_macro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.38888888888888884
        result = precision(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_precision_weighted(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.38888888888888884
        result = precision(y_true, y_pred, average="weighted")

        npt.assert_allclose(result, expected)

    def test_precision_micro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.5
        result = precision(y_true, y_pred, average="micro")

        npt.assert_allclose(result, expected)

    def test_precision_multiclass_imbalanced_weighted(self):
        y_true = np.array([0, 0, 0, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2])

        expected = 0.9
        result = precision(y_true, y_pred, average="weighted")

        npt.assert_allclose(result, expected)

    def test_precision_multiclass_zero_division(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 0, 2])

        expected = 0.5
        result = precision(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_precision_multiclass_2d_inputs(self):
        y_true = np.array([[0], [1], [2], [1]])
        y_pred = np.array([[0], [2], [2], [1]])

        expected = 0.8333333333333334
        result = precision(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_precision_multiclass_invalid_average(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 2, 1])

        npt.assert_raises(ValueError, precision, y_true, y_pred,
                          average="multiple")


class TestRecallBinary:
    def test_recall_basic(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])

        expected = 0.666666666
        result = recall(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_recall_all_correct(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        expected = 1.0
        result = recall(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_recall_all_wrong(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        expected = 0.0
        result = recall(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_recall_no_positive_pred(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        expected = 0.0
        result = recall(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_recall_empty_arrays(self):
        npt.assert_raises(
            ValueError,
            recall,
            np.array([]),
            np.array([])
        )

    def test_recall_length_mismatch(self):
        npt.assert_raises(
            ValueError,
            recall,
            np.array([1, 0, 1]),
            np.array([1, 0])
        )

    def test_recall_binary_zero_division(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 1, 0])

        result = recall(y_true, y_pred, average="binary")

        assert result == 0.0


class TestRecallMulticlass:
    def test_recall_macro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.5
        result = recall(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_recall_weighted(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.5
        result = recall(y_true, y_pred, average="weighted")

        npt.assert_allclose(result, expected)

    def test_recall_micro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.5
        result = recall(y_true, y_pred, average="micro")

        npt.assert_allclose(result, expected)

    def test_recall_multiclass_imbalanced(self):
        y_true = np.array([0, 0, 0, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2])

        expected = 0.8
        result = recall(y_true, y_pred, average="weighted")

        npt.assert_allclose(result, expected)

    def test_recall_multiclass_zero_division(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 0, 2])

        expected = 0.166666666666
        result = recall(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_recall_2d_inputs(self):
        y_true = np.array([[1], [0], [1], [0]])
        y_pred = np.array([[1], [1], [1], [0]])

        expected = 0.75
        result = recall(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_recall_invalid_average(self):
        npt.assert_raises(
            ValueError,
            recall,
            np.array([1, 0, 1]),
            np.array([1, 0, 1]),
            average="average"
        )


class TestF1Binary:
    def test_f1_basic(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 1])

        expected = 0.8
        result = f1(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_f1_all_correct(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])

        expected = 1.0
        result = f1(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_f1_all_wrong(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])

        expected = 0.0
        result = f1(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_f1_zero_division(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        result = f1(y_true, y_pred)
        assert result == 0.0

    def test_f1_2d_inputs(self):
        y_true = np.array([[1], [0], [1]])
        y_pred = np.array([[1], [1], [1]])

        expected = 0.8
        result = f1(y_true, y_pred)

        npt.assert_allclose(result, expected)

    def test_f1_empty(self):
        npt.assert_raises(
            ValueError,
            f1,
            np.array([]),
            np.array([])
        )

    def test_f1_shape_mismatch(self):
        y_true = np.array([1, 0, 1, 1])
        y_pred = np.array([1, 0, 0])
        npt.assert_raises(
            ValueError,
            f1,
            y_true,
            y_pred
        )


class TestF1Multiclass:
    def test_f1_macro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.433333333
        result = f1(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_f1_weighted(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.433333333
        result = f1(y_true, y_pred, average="weighted")

        npt.assert_allclose(result, expected)

    def test_f1_micro(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = 0.5
        result = f1(y_true, y_pred, average="micro")

        npt.assert_allclose(result, expected)

    def test_f1_multiclass_imbalanced(self):
        y_true = np.array([0, 0, 0, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2])

        expected = 0.8133333333333332
        result = f1(y_true, y_pred, average="weighted")

        npt.assert_allclose(result, expected)

    def test_f1_multiclass_zero_division(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 0, 2])

        expected = 0.5555555555555555
        result = f1(y_true, y_pred, average="macro")

        npt.assert_allclose(result, expected)

    def test_f1_invalid_average(self):
        npt.assert_raises(
            ValueError,
            f1,
            np.array([1, 0, 1]),
            np.array([1, 0, 1]),
            average="average"
        )

    def test_f1_micro_tp_zero(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([0, 0, 0])

        expected = 0.0
        result = f1(y_true, y_pred, average="micro")

        npt.assert_allclose(result, expected)


class TestConfusionMatrix:
    def test_confusion_matrix_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        expected = np.array([[1, 1], [0, 2]])
        result = confusion_matrix(y_true, y_pred)

        npt.assert_array_equal(result, expected)

    def test_confusion_matrix_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])

        expected = np.array([[2, 0, 0], [1, 0, 1], [0, 1, 1]])
        result = confusion_matrix(y_true, y_pred)

        npt.assert_array_equal(result, expected)

    def test_confusion_matrix_all_correct(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = confusion_matrix(y_true, y_pred)

        npt.assert_array_equal(result, expected)

    def test_confusion_matrix_all_wrong(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])

        expected = np.array([[0, 3], [0, 0]])
        result = confusion_matrix(y_true, y_pred)

        npt.assert_array_equal(result, expected)

    def test_confusion_matrix_missing_class_in_true(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 1, 1])

        expected = np.array([[1, 2], [0, 0]])
        result = confusion_matrix(y_true, y_pred)

        npt.assert_array_equal(result, expected)

    def test_confusion_matrix_2d_inputs(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array([[0], [2], [1]])

        expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        result = confusion_matrix(y_true, y_pred)

        npt.assert_array_equal(result, expected)

    def test_confusion_matrix_empty(self):
        npt.assert_raises(ValueError,
                          confusion_matrix, np.array([]), np.array([]))

    def test_confusion_matrix_length_mismatch(self):
        npt.assert_raises(
            ValueError,
            confusion_matrix,
            np.array([0, 1]),
            np.array([0])
        )

    def test_confusion_matrix_single_class(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        cm = confusion_matrix(y_true, y_pred)

        expected = np.array([[4]])

        assert cm.shape == (1, 1)
        assert np.array_equal(cm, expected)


class TestMSE:
    def test_mse_zero(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert mse(y_true, y_pred) == 0.0

    def test_mse_basic(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([3, 2, 1])
        expected = ((2**2 + 0 + 2**2) / 3)
        assert mse(y_true, y_pred) == expected

    def test_mse_empty_raises(self):
        np.testing.assert_raises_regex(
            ValueError,
            r"Input arrays must not be empty",
            mse,
            np.array([]),
            np.array([])
        )

    def test_mse_length_mismatch_raises(self):
        np.testing.assert_raises_regex(
            ValueError,
            r"Shape mismatch",
            mse,
            np.array([1, 2, 3]),
            np.array([0])
        )


class TestROCCurve:
    def test_roc_basic(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fpr_s = np.array([0., 0., 0.5, 0.5, 1.])
        tpr_s = np.array([0., 0.5, 0.5, 1., 1.])

        npt.assert_allclose(fpr, fpr_s)
        npt.assert_allclose(tpr, tpr_s)

    def test_roc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])

        fpr, tpr, _ = roc_curve(y_true, y_score)

        # perfect classifier → TPR hits 1 before FPR increases
        assert tpr[-1] == 1.0

    def test_roc_worst(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])

        fpr, tpr, _ = roc_curve(y_true, y_score)

        # worst classifier → behaves like inverted
        assert tpr[0] <= tpr[-1]

    def test_roc_2d_inputs(self):
        y_true = np.array([[0], [1], [1], [1]])
        y_score = np.array([[0.1], [0.9], [0.2], [0.8]])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        fpr_s = np.array([0., 0., 0., 1.])
        tpr_s = np.array([0., 0.33333333, 1., 1.])

        # bounds
        assert np.all((fpr >= 0) & (fpr <= 1))
        assert np.all((tpr >= 0) & (tpr <= 1))

        # monotonicity
        assert np.all(np.diff(fpr) >= 0)
        assert np.all(np.diff(tpr) >= 0)

        # endpoints
        assert fpr[0] == 0.0 and tpr[0] == 0.0
        assert fpr[-1] == 1.0 and tpr[-1] == 1.0

        # curve equivalence via AUC (representation-independent)
        auc = np.trapezoid(tpr, fpr)
        auc_s = np.trapezoid(tpr_s, fpr_s)
        npt.assert_allclose(auc, auc_s)

    def test_roc_empty(self):
        npt.assert_raises(ValueError, roc_curve, np.array([]), np.array([]))

    def test_roc_length_mismatch(self):
        npt.assert_raises(
            ValueError,
            roc_curve,
            np.array([0, 1]),
            np.array([0.1])
        )

    def test_roc_non_binary(self):
        npt.assert_raises(
            ValueError,
            roc_curve,
            np.array([0, 1, 2]),
            np.array([0.1, 0.5, 0.9])
        )

    def test_roc_requires_both_classes(self):
        # only positives
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])

        npt.assert_raises(
            ValueError,
            roc_curve,
            y_true,
            y_score
        )

        # only negatives
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])

        npt.assert_raises(
            ValueError,
            roc_curve,
            y_true,
            y_score
        )

    def test_roc_threshold_correctness(self):
        y_true = np.array([0, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.2])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)

        # thresholds must be derived from actual scores (plus inf)
        finite_thresholds = thresholds[1:]
        unique_scores = np.sort(np.unique(y_score))[::-1]

        assert np.all(np.isin(finite_thresholds, unique_scores))

        # thresholds must be strictly decreasing
        assert np.all(np.diff(thresholds[1:]) < 0)

        # each threshold must reproduce correct ROC step
        for i, thr in enumerate(thresholds):
            y_pred = (y_score >= thr)

            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))

            assert np.isclose(tp / P, tpr[i])
            assert np.isclose(fp / N, fpr[i])


class TestAUC:
    def test_auc_basic(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.75, 1.0])

        expected = 0.625
        result = auc(x, y)

        npt.assert_allclose(result, expected)

    def test_auc_unsorted_input(self):
        x = np.array([1.0, 0.0, 0.5])
        y = np.array([1.0, 0.0, 0.75])

        expected = 0.625
        result = auc(x, y)

        npt.assert_allclose(result, expected)

    def test_auc_perfect_roc(self):
        # perfect ROC → AUC = 1
        x = np.array([0.0, 0.0, 1.0])
        y = np.array([0.0, 1.0, 1.0])

        result = auc(x, y)
        assert result == 1.0

    def test_auc_random_curve(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = auc(x, y)
        assert result == 0.5

    def test_auc_empty(self):
        npt.assert_raises(ValueError, auc, np.array([]), np.array([]))

    def test_auc_length_mismatch(self):
        npt.assert_raises(
            ValueError,
            auc,
            np.array([0, 1]),
            np.array([0])
        )
