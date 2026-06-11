import numpy as np
import matplotlib.pyplot as plt


def _finalize_plot(figure, save_path: str | None, show: bool) -> None:
    if save_path is not None:
        figure.savefig(save_path)

    if show:
        plt.show()


def plot_metric_over_time(
    metric_values: list | np.ndarray,
    title: str,
    ylabel: str,
    save_path: str | None = None,
    show: bool = True
):
    """
    Plot a metric across chunks.

    Parameters
    ----------
    metric_values : list or np.ndarray
        Metric value recorded for each processed chunk.
    title : str
        Chart title.
    ylabel : str
        Label displayed on the vertical axis.
    save_path : str or None, optional
        File path used to save the chart. If None, the figure is not saved.
        Default is None.
    show : bool, optional
        Whether to display the chart. Default is True.

    Raises
    ------
    ValueError
        If ``metric_values`` is empty.

    Complexity
    ----------
    Time Complexity:
        O(n), where n is the number of metric values.
    Space Complexity:
        O(n) for the chunk-number array.
    """

    metric_values = np.asarray(metric_values)

    if metric_values.size == 0:
        raise ValueError("metric_values must contain at least one value.")

    chunk_numbers = np.arange(1, metric_values.size + 1)

    # Plotting
    figure, axes = plt.subplots()

    axes.plot(
        chunk_numbers,
        metric_values,
        marker="o"
    )

    axes.set_title(title)
    axes.set_xlabel("Chunk")
    axes.set_ylabel(ylabel)
    axes.grid()

    _finalize_plot(figure, save_path, show)


def compare_models(
    metric1: list | np.ndarray,
    metric2: list | np.ndarray,
    labels: tuple[str, str] | list[str],
    title: str = "Model Comparison",
    ylabel: str = "Metric Value",
    save_path: str | None = None,
    show: bool = True
):
    """
    Compare metric values from two models across chunks.

    Parameters
    ----------
    metric1 : list or np.ndarray
        Metric values produced by the first model.
    metric2 : list or np.ndarray
        Metric values produced by the second model.
    labels : tuple or list of str
        Two labels identifying the plotted models.
    title : str, optional
        Chart title.
    ylabel : str, optional
        Label displayed on the vertical axis.
    save_path : str or None, optional
        File path used to save the chart.
    show : bool, optional
        Whether to display the chart. Default is True.

    Raises
    ------
    ValueError
        If either metric array is empty.
        If the metric arrays have different lengths.
        If exactly two labels are not provided.

    Complexity
    ----------
    Time Complexity:
        O(n), where n is the number of metric values.
    Space Complexity:
        O(n) for the generated chunk-number array.
    """
    metric1 = np.asarray(metric1)
    metric2 = np.asarray(metric2)

    if (metric1.size == 0 or metric2.size == 0):
        raise ValueError(
            "Both metric arrays must contain at least one value."
        )

    if metric1.shape != metric2.shape:
        raise ValueError("Both metric arrays must have the same shape.")

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two model names.")

    chunk_numbers = np.arange(1, metric1.size + 1)

    # Plotting
    figure, axes = plt.subplots()
    axes.plot(chunk_numbers, metric1, marker="o", label=labels[0])
    axes.plot(chunk_numbers, metric2, marker="o", label=labels[1])
    axes.set_title(title)
    axes.set_xlabel("Chunk")
    axes.set_ylabel(ylabel)
    axes.legend()
    axes.grid()

    _finalize_plot(figure, save_path, show)


def plot_predictions_vs_ground_truth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Ground Truth",
    save_path: str | None = None,
    show: bool = True
):
    """
    Plot predicted values against their ground-truth values.

    Parameters
    ----------
    y_true : np.ndarray
        Expected values.
    y_pred : np.ndarray
        Predicted values.
    title : str, optional
        Chart title.
    save_path : str or None, optional
        File path used to save the chart.
    show : bool, optional
        Whether to display the chart. Default is True.

    Raises
    ------
    ValueError
        If either input is empty.
        If the flattened arrays have different lengths.

    Complexity
    ----------
    Time Complexity:
        O(n), where n is the number of observations.
    Space Complexity:
        O(n) for flattened arrays and observation indexes.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if (
        y_true.size == 0
        or y_pred.size == 0
    ):
        raise ValueError(
            "Input arrays must not be empty."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Arrays must have the same length."
        )

    observation_numbers = np.arange(1, y_true.size + 1)

    # Plotting
    figure, axes = plt.subplots()

    axes.plot(
        observation_numbers,
        y_true,
        marker="o",
        label="Ground Truth"
    )

    axes.plot(
        observation_numbers,
        y_pred,
        marker="o",
        label="Prediction"
    )

    axes.set_title(title)
    axes.set_xlabel("Observation")
    axes.set_ylabel("Value")
    axes.legend()
    axes.grid()

    _finalize_plot(figure, save_path, show)
