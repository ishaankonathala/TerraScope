"""Evaluation metrics, benchmarking, and visualization exports."""

from ml.evaluation.benchmark import (
    benchmark_models,
    build_comparison_table,
    evaluate_model,
    load_experiment_summaries,
)
from ml.evaluation.metrics import ChangeDetectionMetrics, MetricAccumulator, compute_binary_metrics
from ml.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_prediction_samples,
    plot_training_curves,
)

__all__ = [
    "ChangeDetectionMetrics",
    "MetricAccumulator",
    "benchmark_models",
    "build_comparison_table",
    "compute_binary_metrics",
    "evaluate_model",
    "load_experiment_summaries",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "plot_prediction_samples",
    "plot_training_curves",
]

