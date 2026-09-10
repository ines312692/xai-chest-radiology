"""Explanation layer: saliency production and the two scoring protocols."""

from .cam import (METHOD_NAMES, SLOW_METHODS, build_cam_engines, compute_heatmap,
                  load_heatmap, method_names, overlay, peak_zone, save_heatmap)
from .evaluation import (deletion_insertion, ground_truth_mask, localization_scores,
                         summarise)

__all__ = ["METHOD_NAMES", "SLOW_METHODS", "build_cam_engines", "compute_heatmap",
           "load_heatmap", "method_names", "overlay", "peak_zone", "save_heatmap",
           "deletion_insertion", "ground_truth_mask", "localization_scores",
           "summarise"]
