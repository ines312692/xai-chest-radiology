"""Backbone construction, target layer selection and checkpoint loading.

Two backbones are supported and they differ in exactly two places, the name of their
classification head and the name of their last convolutional block. Both differences
are handled here, so no other module contains a test on the architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models as tv_models

from ..config import ModelConfig
from ..utils import get_logger

logger = get_logger(__name__)

SUPPORTED = ("resnet50", "densenet121")


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate one backbone with a fresh head of the right size.

    ResNet50 is the baseline of the MDPI Information 2025 paper, which makes our numbers
    directly comparable to its reported accuracy of 0.90 and AUC of 0.93. DenseNet121 is
    the reference of chest radiography since CheXNet and the classifier of the Nature
    Machine Intelligence 2022 saliency benchmark, with three times fewer parameters.
    """
    if config.architecture == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if config.pretrained else None
        model = tv_models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    elif config.architecture == "densenet121":
        weights = (tv_models.DenseNet121_Weights.IMAGENET1K_V1
                   if config.pretrained else None)
        model = tv_models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, config.num_classes)
    else:
        raise ValueError(f"Unsupported architecture {config.architecture}, "
                         f"expected one of {SUPPORTED}")

    millions = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("%s built, %.1f million parameters, pretrained %s",
                config.architecture, millions, config.pretrained)
    return model


def target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return the deepest convolutional block, the layer every saliency method hooks.

    The original Grad-CAM paper recommends the last convolutional layer because it holds
    the deepest spatial semantics. It is layer4 for ResNet and the last dense block for
    DenseNet.
    """
    if architecture == "resnet50":
        return model.layer4[-1]
    if architecture == "densenet121":
        return model.features[-1]
    raise ValueError(f"Unsupported architecture {architecture}")


def load_checkpoint(model: nn.Module, path: str | Path,
                    device: torch.device) -> nn.Module:
    """Load trained weights and put the model in evaluation mode."""
    state: Dict[str, torch.Tensor] = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info("Checkpoint loaded from %s", path)
    return model
