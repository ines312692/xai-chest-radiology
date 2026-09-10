"""Reporting layer: the figures of the thesis, each regenerable on its own."""

from .figures import (cam_comparison_grid, cam_metric_barplots, dataset_preview,
                      training_report)

__all__ = ["cam_comparison_grid", "cam_metric_barplots", "dataset_preview",
           "training_report"]
