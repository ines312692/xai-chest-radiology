"""Unit tests of the parts that are easy to get silently wrong.

These tests use synthetic arrays and never touch the dataset, so they run in a second
on any machine. What they protect is the logic a reviewer would question: that the
split really is leak free, that a saliency map is scored the way the protocol says,
and that a faithful map scores better than a random one.

    python -m pytest tests -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_chest.config import DataConfig, ExperimentConfig
from xai_chest.data.splits import patient_level_split
from xai_chest.explain.cam import peak_zone
from xai_chest.explain.evaluation import (ground_truth_mask, localization_scores,
                                          summarise)
from xai_chest.training.metrics import classification_metrics, confusion_category


@pytest.fixture
def table() -> pd.DataFrame:
    """Two hundred patients, thirty percent of them positive."""
    rng = np.random.default_rng(0)
    targets = (rng.random(200) < 0.3).astype(int)
    return pd.DataFrame({"patientId": [f"p{i:03d}" for i in range(200)],
                         "target": targets})


def test_split_is_patient_disjoint(table):
    splits = patient_level_split(table, DataConfig(), seed=42)
    train, val, test = (set(splits[k]["patientId"]) for k in ("train", "val", "test"))
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == len(table)


def test_split_is_deterministic(table):
    first = patient_level_split(table, DataConfig(), seed=42)["test"]["patientId"]
    second = patient_level_split(table, DataConfig(), seed=42)["test"]["patientId"]
    assert list(first) == list(second), "the same seed must give the same partition"


def test_split_keeps_the_class_ratio(table):
    splits = patient_level_split(table, DataConfig(), seed=42)
    reference = table["target"].mean()
    for name, frame in splits.items():
        assert abs(frame["target"].mean() - reference) < 0.05, name


def test_ground_truth_mask_is_rescaled():
    boxes = pd.DataFrame([{"patientId": "p1", "x": 512, "y": 512,
                           "width": 256, "height": 256}])
    mask = ground_truth_mask(boxes, "p1", (1024, 1024), 224)
    assert mask.shape == (224, 224)
    assert mask[130, 130] and not mask[10, 10]
    assert abs(mask.sum() - 56 * 56) < 200


def test_perfect_map_scores_one():
    mask = np.zeros((224, 224), dtype=bool)
    mask[50:100, 50:100] = True
    heatmap = mask.astype(np.float32)
    scores = localization_scores(heatmap, mask)
    assert scores["iou"] == pytest.approx(1.0)
    assert scores["hit"] == 1


def test_map_pointing_elsewhere_scores_zero():
    mask = np.zeros((224, 224), dtype=bool)
    mask[50:100, 50:100] = True
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[150:200, 150:200] = 1.0
    scores = localization_scores(heatmap, mask)
    assert scores["iou"] == pytest.approx(0.0)
    assert scores["hit"] == 0


def test_summary_ranks_on_localization_and_computes_faithfulness():
    localization = pd.DataFrame([{"method": "a", "iou": 0.30, "hit": 1},
                                 {"method": "b", "iou": 0.10, "hit": 0}])
    faithfulness = pd.DataFrame([{"method": "a", "deletion_auc": 0.60,
                                  "insertion_auc": 0.90},
                                 {"method": "b", "deletion_auc": 0.80,
                                  "insertion_auc": 0.85}])
    summary = summarise(localization, faithfulness)
    assert list(summary.index) == ["a", "b"]
    assert summary.loc["a", "faithfulness"] == pytest.approx(0.30)


def test_peak_zone_follows_the_radiological_convention():
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[190, 30] = 1.0        # bottom left of the image, patient right side
    assert peak_zone(heatmap) == "right lower lung zone"


def test_confusion_categories():
    assert confusion_category(1, 1) == "true_positive"
    assert confusion_category(0, 0) == "true_negative"
    assert confusion_category(0, 1) == "false_positive"
    assert confusion_category(1, 0) == "false_negative"


def test_classification_metrics_on_a_perfect_classifier():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.10, 0.90, 0.99])
    metrics = classification_metrics(labels, probabilities)
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_config_roundtrip(tmp_path):
    config = ExperimentConfig(name="unit", output_dir=str(tmp_path))
    destination = tmp_path / "config.yaml"
    config.save(destination)
    reloaded = ExperimentConfig.from_yaml(destination)
    assert reloaded.to_dict() == config.to_dict()


def test_output_paths_carry_the_experiment_name(tmp_path):
    config = ExperimentConfig(name="densenet121_rsna", output_dir=str(tmp_path))
    assert config.paths.checkpoint.name == "densenet121_rsna_best.pth"
    assert config.paths.train_csv.name == "rsna_train.csv"
