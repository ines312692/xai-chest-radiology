"""Scoring of the saliency maps, by two independent protocols.

Localisation compares a map to expert boxes, following the Nature Machine Intelligence
2022 benchmark, with mIoU as primary metric and the hit rate, also called pointing
game, as secondary metric. Faithfulness compares a map to the behaviour of the model
itself, following the RISE paper of Petsiuk et al. BMVC 2018, and needs no annotation.

Reading them jointly is the point. A method can agree with the radiologist and not
reflect what the network actually used, or the reverse, and that dissociation is the
result the project is after.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def ground_truth_mask(boxes: pd.DataFrame, patient_id: str,
                      original_shape: Tuple[int, int], image_size: int) -> np.ndarray:
    """Union of the expert boxes of one patient, rescaled to the network resolution.

    Several boxes are merged into one mask rather than scored separately, because a
    study with two opacities has one correct answer, not two competing ones.
    """
    height, width = original_shape
    mask = np.zeros((image_size, image_size), dtype=bool)
    for _, box in boxes[boxes["patientId"] == patient_id].iterrows():
        x1 = int(box["x"] / width * image_size)
        y1 = int(box["y"] / height * image_size)
        x2 = int((box["x"] + box["width"]) / width * image_size)
        y2 = int((box["y"] + box["height"]) / height * image_size)
        mask[y1:y2, x1:x2] = True
    return mask


def localization_scores(heatmap: np.ndarray, mask: np.ndarray,
                        binarization_ratio: float = 0.5) -> Dict[str, float]:
    """Intersection over union and hit rate of one map against one mask.

    The map is binarised by keeping the pixels above a fraction of its maximum, a
    relative threshold rather than an absolute one because the scale of a map is not
    comparable between methods. The hit is whether the single most activated pixel
    falls inside the expert region, which is what the pointing game measures.
    """
    predicted = heatmap >= binarization_ratio * heatmap.max()
    intersection = np.logical_and(predicted, mask).sum()
    union = np.logical_or(predicted, mask).sum()
    peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return {"iou": float(intersection / union) if union > 0 else 0.0,
            "hit": int(bool(mask[peak]))}


def deletion_insertion(model: nn.Module, tensor: torch.Tensor, heatmap: np.ndarray,
                       class_index: int, device: torch.device, steps: int = 20,
                       blur_kernel: int = 25) -> Dict[str, float]:
    """Area under the deletion and insertion curves for one map.

    Deletion removes the most highlighted pixels step by step and a faithful map makes
    the probability collapse quickly, so a low area is better. Insertion restores those
    same pixels onto a blurred study and a faithful map makes the probability recover
    quickly, so a high area is better. The blurred image is used as the neutral
    baseline rather than black, since a black region is itself a strong signal on a
    radiograph.
    """
    order = np.argsort(-heatmap.flatten())
    n_pixels = order.size
    base = tensor.clone()
    blurred = F.avg_pool2d(base, kernel_size=blur_kernel, stride=1,
                           padding=blur_kernel // 2)

    deletion_probs, insertion_probs = [], []
    for step in range(steps + 1):
        k = int(n_pixels * step / steps)
        mask = torch.zeros(n_pixels, device=device)
        if k > 0:
            mask[torch.from_numpy(order[:k].copy()).to(device)] = 1.0
        mask = mask.view(1, 1, *heatmap.shape)
        with torch.no_grad():
            deleted = torch.softmax(model(base * (1 - mask)), dim=1)[0, class_index]
            inserted = torch.softmax(model(blurred * (1 - mask) + base * mask),
                                     dim=1)[0, class_index]
        deletion_probs.append(float(deleted))
        insertion_probs.append(float(inserted))

    return {"deletion_auc": float(np.mean(deletion_probs)),
            "insertion_auc": float(np.mean(insertion_probs))}


def summarise(localization: pd.DataFrame, faithfulness: pd.DataFrame) -> pd.DataFrame:
    """Join the two protocols into the ranking table of the study.

    The faithfulness column is insertion minus deletion, one number that says how much
    the highlighted pixels really drive the decision. Sorting on mIoU rather than on
    faithfulness is deliberate: it puts agreement with the radiologist first and lets
    the reader see when the two criteria disagree.
    """
    scores = localization.groupby("method")[["iou", "hit"]].mean()
    scores = scores.join(
        faithfulness.groupby("method")[["deletion_auc", "insertion_auc"]].mean())
    scores["faithfulness"] = scores["insertion_auc"] - scores["deletion_auc"]
    return scores.round(4).sort_values("iou", ascending=False)
