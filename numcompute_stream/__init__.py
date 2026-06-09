from . import io, metrics, optim, pipeline, preprocessing, sort_search
from . import utils
from .io import load_csv
from .metrics import (
    accuracy, Accuracy,
    confusion_matrix, ConfusionMatrix,
    f1, F1,
    mse, MSE,
    precision, Precision,
    recall, Recall,
    auc, roc_curve, ROCAUC
)
from .optim import grad, jacobian
from .pipeline import Pipeline, FeatureUnion
from .preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from .tree import DecisionTreeClassifier
from .ensemble import EnsembleClassifier
from .stream import StreamTrainer
from .rank import rank, percentile
from .sort_search import (
    binary_search,
    multi_key_sort,
    quickselect,
    stable_sort,
    topk,
)

from .visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth
)

from .stats import (
    Statistics,
    Histogram,
    Quantile,
    histogram,
    quantile,
)

__all__ = [
    "io",
    "load_csv",
    "metrics",
    "accuracy",
    "Accuracy",
    "confusion_matrix",
    "ConfusionMatrix",
    "f1",
    "F1",
    "mse",
    "MSE",
    "precision",
    "Precision",
    "recall",
    "Recall",
    "ROCAUC",
    "optim",
    "grad",
    "stats",
    "jacobian",
    "pipeline",
    "Pipeline",
    "FeatureUnion",
    "preprocessing",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "StandardScaler",
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "StreamTrainer",
    "percentile",
    "rank",
    "sort_search",
    "binary_search",
    "multi_key_sort",
    "quickselect",
    "stable_sort",
    "topk",
    "utils",
    "histogram",
    "quantile",
    "Statistics",
    "Histogram",
    "Quantile",
    "auc",
    "roc_curve",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth"
]
