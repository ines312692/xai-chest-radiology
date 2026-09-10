"""From the raw label files to a leak free train validation test partition.

Two decisions are implemented here and both are defended in the state of the art table
of the project. The negative class keeps only the Normal group, because the third group
of the detailed class file holds other pathologies without pneumonia and would teach the
model that an effusion is normal. The split is made on patient identifiers rather than
on images, which corrects the limitation acknowledged by the PLOS ONE 2024 paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import DataConfig
from ..utils import get_logger

logger = get_logger(__name__)

SPLIT_COLUMNS = ["patientId", "filepath", "class", "label", "target"]


def build_classification_table(data_root: str | Path, config: DataConfig) -> pd.DataFrame:
    """Merge the two label files into one row per usable image.

    The label file has one row per bounding box, so a patient with several opacities
    appears several times and is deduplicated. Only the images actually present on disk
    are kept, which makes the table safe to use with a partial download.
    """
    data_root = Path(data_root)
    image_dir = data_root / "stage_2_train_images"

    labels = pd.read_csv(data_root / "stage_2_train_labels.csv")
    classes = pd.read_csv(data_root / "stage_2_detailed_class_info.csv")

    table = labels.merge(classes, on="patientId", how="left")
    table = table.drop_duplicates("patientId")
    table = table[table["class"].isin(config.kept_classes)].reset_index(drop=True)

    table["label"] = np.where(table["class"] == config.positive_class,
                              "PNEUMONIA", "NORMAL")
    table["target"] = (table["label"] == "PNEUMONIA").astype(int)
    table["filepath"] = table["patientId"].apply(
        lambda patient: str(image_dir / f"{patient}.dcm"))
    table = table[table["filepath"].apply(lambda p: Path(p).exists())]
    table = table.reset_index(drop=True)

    logger.info("Usable images: %d, of which pneumonia %d",
                len(table), int(table["target"].sum()))
    return table


def patient_level_split(table: pd.DataFrame, config: DataConfig,
                        seed: int) -> Dict[str, pd.DataFrame]:
    """Split on patient identifiers, stratified on the class.

    A patient never appears in two subsets, which is asserted rather than assumed: an
    image level split would leak the same patient into training and test and inflate
    every metric of the study.
    """
    patients = table["patientId"].to_numpy()
    strata = table["target"].to_numpy()

    holdout = config.val_size + config.test_size
    train_patients, temp_patients, _, temp_strata = train_test_split(
        patients, strata, test_size=holdout, stratify=strata, random_state=seed)
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=config.test_size / holdout,
        stratify=temp_strata, random_state=seed)

    splits = {
        "train": table[table["patientId"].isin(train_patients)].reset_index(drop=True),
        "val": table[table["patientId"].isin(val_patients)].reset_index(drop=True),
        "test": table[table["patientId"].isin(test_patients)].reset_index(drop=True),
    }

    assert set(splits["train"].patientId).isdisjoint(splits["val"].patientId)
    assert set(splits["train"].patientId).isdisjoint(splits["test"].patientId)
    assert set(splits["val"].patientId).isdisjoint(splits["test"].patientId)

    for name, frame in splits.items():
        logger.info("%-5s %6d images, pneumonia ratio %.3f",
                    name, len(frame), frame["target"].mean())
    return splits


def export_expert_boxes(data_root: str | Path, test_split: pd.DataFrame,
                        destination: str | Path) -> pd.DataFrame:
    """Keep every expert box of the test images, one row per box.

    These boxes are the ground truth of the localisation protocol. They are exported at
    preprocessing time so the explanation step never needs to open the training data,
    and they stay in the original pixel space, rescaled only when a map is scored.
    """
    labels = pd.read_csv(Path(data_root) / "stage_2_train_labels.csv")
    boxes = labels[labels["patientId"].isin(set(test_split["patientId"]))]
    boxes = boxes.dropna(subset=["x"]).reset_index(drop=True)
    boxes.to_csv(destination, index=False)
    logger.info("Expert boxes of the test split: %d", len(boxes))
    return boxes


def save_splits(splits: Dict[str, pd.DataFrame], paths) -> None:
    """Freeze the partition on disk before any training.

    Written once and read by every later step, so the training, the evaluation and the
    saliency study provably share the same patients.
    """
    for name, destination in (("train", paths.train_csv), ("val", paths.val_csv),
                              ("test", paths.test_csv)):
        splits[name][SPLIT_COLUMNS].to_csv(destination, index=False)
    logger.info("Splits written to %s", paths.root)


def load_splits(paths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read back the frozen partition."""
    return (pd.read_csv(paths.train_csv), pd.read_csv(paths.val_csv),
            pd.read_csv(paths.test_csv))
