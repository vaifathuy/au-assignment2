import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numpy.testing import assert_equal, assert_raises_regex

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

# Set non-interactive mode to matplotlib
matplotlib.use("Agg")


# The fixture to close charts after each test:
@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# Testing
class TestPlotMetricOverTime:

    def test_valid_metric_values(self):
        result = plot_metric_over_time(
            metric_values=[
                0.50,
                0.65,
                0.80
            ],
            title="Accuracy over Time",
            ylabel="Accuracy",
            show=False
        )

        assert_equal(
            result,
            None
        )

    def test_rejects_empty_values(self):
        assert_raises_regex(
            ValueError,
            r"metric_values must contain at least one value\.",
            plot_metric_over_time,
            metric_values=[],
            title="Accuracy",
            ylabel="Accuracy",
            show=False
        )


class TestCompareModels:

    def test_valid_model_metrics(self):
        result = compare_models(
            metric1=[
                0.50,
                0.65,
                0.80
            ],
            metric2=[
                0.55,
                0.70,
                0.85
            ],
            labels=[
                "Decision Tree",
                "Ensemble"
            ],
            show=False
        )

        assert_equal(
            result,
            None
        )

    def test_rejects_different_metric_shapes(self):
        assert_raises_regex(
            ValueError,
            r"Both metric arrays must have the same shape\.",
            compare_models,
            metric1=[
                0.50,
                0.65
            ],
            metric2=[
                0.55,
                0.70,
                0.85
            ],
            labels=[
                "Decision Tree",
                "Ensemble"
            ],
            show=False
        )

    def test_rejects_invalid_number_of_labels(self):
        assert_raises_regex(
            ValueError,
            r"labels must contain exactly two model names\.",
            compare_models,
            metric1=[
                0.50,
                0.65
            ],
            metric2=[
                0.55,
                0.70
            ],
            labels=[
                "Only One Label"
            ],
            show=False
        )

    def test_rejects_empty_values(self):
        assert_raises_regex(
            ValueError,
            r"Both metric arrays must contain at least one value\.",
            compare_models,
            metric1=[],
            metric2=[],
            labels=[
                "Decision Tree",
                "Ensemble"
            ],
            show=False
        )


class TestPlotPredictionsVsGroundTruth:

    def test_valid_predictions(self):
        result = plot_predictions_vs_ground_truth(
            y_true=np.array([
                0,
                0,
                1,
                1
            ]),
            y_pred=np.array([
                0,
                1,
                1,
                1
            ]),
            show=False
        )

        assert_equal(result, None)

    def test_saves_chart_to_file(
        self,
        tmp_path
    ):
        save_path = (
            tmp_path
            / "predictions.png"
        )

        plot_predictions_vs_ground_truth(
            y_true=np.array([
                0,
                1
            ]),
            y_pred=np.array([
                0,
                1
            ]),
            save_path=str(save_path),
            show=False
        )

        assert_equal(
            save_path.exists(),
            True
        )

    def test_rejects_different_lengths(self):
        assert_raises_regex(
            ValueError,
            r"Arrays must have the same length\.",
            plot_predictions_vs_ground_truth,
            y_true=np.array([
                0,
                1,
                1
            ]),
            y_pred=np.array([
                0,
                1
            ]),
            show=False
        )

    def test_rejects_empty_arrays(self):
        assert_raises_regex(
            ValueError,
            r"Input arrays must not be empty\.",
            plot_predictions_vs_ground_truth,
            y_true=np.array([]),
            y_pred=np.array([]),
            show=False
        )
