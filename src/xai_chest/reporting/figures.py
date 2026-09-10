"""The figures of the thesis.

Every figure is produced by one function that takes data and a destination and returns
the path it wrote, so a script never contains plotting code and a figure can be
regenerated alone from a saved CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # the scripts run headless, on Kaggle as on a server
import matplotlib.patches as patches            # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402

BLUE, GREEN, RED = "#2E75B6", "#70AD47", "#C0504D"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def _save(fig, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return destination


def training_report(history: pd.DataFrame, labels: np.ndarray,
                    probabilities: np.ndarray, metrics: Dict[str, float],
                    destination: str | Path) -> Path:
    """Loss curves, confusion matrix and ROC curve on one row.

    The three together answer the three questions a jury asks about a training run: did
    it converge without overfitting, what kind of errors does it make, and how well
    does it separate the classes independently of the threshold.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].set_title("Loss curves")

    predictions = (probabilities >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        labels, predictions, display_labels=CLASS_NAMES, cmap="Blues",
        colorbar=False, ax=axes[1])
    axes[1].set_title("Confusion matrix on the test set")

    false_positive, true_positive, _ = roc_curve(labels, probabilities)
    axes[2].plot(false_positive, true_positive, color=BLUE,
                 label=f"AUC {metrics['auc']:.3f}")
    axes[2].plot([0, 1], [0, 1], linestyle=":", color="gray")
    axes[2].set_xlabel("false positive rate")
    axes[2].set_ylabel("true positive rate")
    axes[2].legend()
    axes[2].set_title("ROC curve")

    return _save(fig, destination)


def dataset_preview(samples: Sequence[dict], boxes: pd.DataFrame,
                    destination: str | Path) -> Path:
    """A few studies per class with their expert boxes.

    This is the visual check that DICOM files decode correctly and that the boxes point
    at the opacities, before any training consumes them.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for row, label in enumerate(CLASS_NAMES):
        chosen = [s for s in samples if s["label"] == label][:3]
        for column, sample in enumerate(chosen):
            axis = axes[row, column]
            axis.imshow(sample["array"], cmap="gray")
            for _, box in boxes[boxes["patientId"] == sample["patientId"]].iterrows():
                axis.add_patch(patches.Rectangle(
                    (box["x"], box["y"]), box["width"], box["height"],
                    linewidth=2, edgecolor="red", facecolor="none"))
            axis.set_title(label, fontsize=10)
            axis.axis("off")
    return _save(fig, destination)


def cam_metric_barplots(summary: pd.DataFrame, destination: str | Path) -> Path:
    """The two protocols side by side, localisation left and faithfulness right."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    summary[["iou", "hit"]].plot(kind="bar", ax=axes[0], color=[BLUE, GREEN])
    axes[0].set_title("Localisation against expert boxes")
    axes[0].tick_params(axis="x", rotation=15)
    summary[["deletion_auc", "insertion_auc"]].plot(kind="bar", ax=axes[1],
                                                    color=[RED, BLUE])
    axes[1].set_title("Faithfulness by perturbation")
    axes[1].tick_params(axis="x", rotation=15)
    return _save(fig, destination)


def cam_comparison_grid(examples: List[dict], methods: Sequence[str],
                        titles: Sequence[str], heatmap_dir: str | Path,
                        boxes: pd.DataFrame, image_size: int,
                        destination: str | Path) -> Path:
    """One row per confusion category, the study then every method.

    Four situations to inspect. On the true positive the map should cover the red
    boxes. On the true negative a diffuse map is expected since no lesion drives the
    decision. On the false positive the map shows what was mistaken for an opacity. On
    the false negative it separates two failure modes, looking at the right region and
    missing the lesion, or looking somewhere else entirely.
    """
    from PIL import Image

    heatmap_dir = Path(heatmap_dir)
    rows = len(examples)
    fig, axes = plt.subplots(rows, len(methods) + 1,
                             figsize=(3 * (len(methods) + 1), 3.2 * rows))
    axes = np.atleast_2d(axes)

    for row, example in enumerate(examples):
        height, width = example["shape"]
        axes[row, 0].imshow(example["display"])
        for _, box in boxes[boxes["patientId"] == example["patientId"]].iterrows():
            axes[row, 0].add_patch(patches.Rectangle(
                (box["x"] / width * image_size, box["y"] / height * image_size),
                box["width"] / width * image_size, box["height"] / height * image_size,
                linewidth=2, edgecolor="red", facecolor="none"))
        axes[row, 0].set_ylabel(example["category"].replace("_", " "), fontsize=10)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("Study with expert boxes", fontsize=10)

        for column, (method, title) in enumerate(zip(methods, titles), start=1):
            path = heatmap_dir / f"{example['patientId']}__{method}.png"
            axes[row, column].imshow(Image.open(path))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title, fontsize=11)

    return _save(fig, destination)
