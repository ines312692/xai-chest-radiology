"""Production of the saliency maps.

Five variants are compared, following the plan of the project and the five method
comparison of the arXiv 2025 paper of the state of the art table. They all hook the
same layer and differ only in how the activations of that layer are weighted, which is
what makes the comparison meaningful: any difference between two maps comes from the
method and not from where it looked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..config import ExplainConfig
from ..models import target_layer
from ..utils import get_logger

logger = get_logger(__name__)

# Human readable names, used by the figures and the summary table
METHOD_NAMES: Dict[str, str] = {
    "gradcam": "Grad-CAM",
    "gradcampp": "Grad-CAM++",
    "eigencam": "Eigen-CAM",
    "layercam": "LayerCAM",
    "scorecam": "Score-CAM",
}

# Score-CAM masks the image once per activation map instead of using a gradient, so it
# needs hundreds of forward passes and is by far the slowest of the five.
SLOW_METHODS = ("scorecam",)


def build_cam_engines(model: nn.Module, architecture: str, config: ExplainConfig,
                      methods: Iterable[str] | None = None) -> Dict[str, object]:
    """Instantiate one engine per requested method, all hooked on the same layer."""
    from pytorch_grad_cam import (EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM,
                                  ScoreCAM)

    registry = {"gradcam": GradCAM, "gradcampp": GradCAMPlusPlus,
                "eigencam": EigenCAM, "layercam": LayerCAM, "scorecam": ScoreCAM}
    layers = [target_layer(model, architecture)]

    engines: Dict[str, object] = {}
    for name in (methods if methods is not None else config.methods):
        if name not in registry:
            raise ValueError(f"Unknown saliency method {name}, "
                             f"expected one of {list(registry)}")
        engine = registry[name](model=model, target_layers=layers)
        if name in SLOW_METHODS:
            engine.batch_size = config.scorecam_batch_size
        engines[name] = engine

    logger.info("Saliency engines ready: %s", list(engines))
    return engines


def compute_heatmap(engine, tensor: torch.Tensor, class_index: int) -> np.ndarray:
    """Return the saliency map of one image for one class, in the range zero to one."""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    return engine(input_tensor=tensor,
                  targets=[ClassifierOutputTarget(int(class_index))])[0]


def overlay(display: np.ndarray, heatmap: np.ndarray,
            image_weight: float = 0.5) -> np.ndarray:
    """Blend one map over its study, for the figures and the qualitative reading."""
    from pytorch_grad_cam.utils.image import show_cam_on_image

    return show_cam_on_image(display, heatmap, use_rgb=True, image_weight=image_weight)


def save_heatmap(heatmap: np.ndarray, display: np.ndarray, directory: str | Path,
                 patient_id: str, method: str) -> None:
    """Store the raw map and its overlay under one naming convention.

    The npy array is what the quantitative protocols read, the png is what the figures
    show. The name carries the patient and the method, so the scoring step never has to
    keep an index of what it produced.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{patient_id}__{method}.npy", heatmap)
    Image.fromarray(overlay(display, heatmap)).save(
        directory / f"{patient_id}__{method}.png")


def load_heatmap(directory: str | Path, patient_id: str, method: str) -> np.ndarray:
    """Read back one stored map."""
    return np.load(Path(directory) / f"{patient_id}__{method}.npy")


def peak_zone(heatmap: np.ndarray) -> str:
    """Name the coarse anatomical zone of the maximum.

    Image left is the patient right side, following the radiological convention. The
    grid is a plain division in thirds and knows nothing of anatomy, so a peak falling
    outside the lung fields is still named as a lung zone. Reading this label together
    with the map is therefore necessary, and replacing the grid by a lung mask is a
    known improvement to make.
    """
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    height, width = heatmap.shape
    vertical = ("upper" if y < height / 3
                else "middle" if y < 2 * height / 3 else "lower")
    side = "right" if x < width / 2 else "left"
    return f"{side} {vertical} lung zone"


def method_names(methods: Iterable[str]) -> List[str]:
    """Human readable names in the order given."""
    return [METHOD_NAMES.get(name, name) for name in methods]
