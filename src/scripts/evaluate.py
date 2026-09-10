"""Step three. Score the best checkpoint once on the untouched test split.

    python scripts/evaluate.py --config configs/resnet50_rsna.yaml

Writes the metric file, the per image predictions and the evaluation figure. The test
split is read here and nowhere else in the training path, which is what makes the
reported numbers an honest estimate rather than a tuned one.
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from xai_chest.config import ExperimentConfig
from xai_chest.data import build_loader, load_splits
from xai_chest.models import build_model, load_checkpoint
from xai_chest.reporting import training_report
from xai_chest.training import DECISION_THRESHOLD, classification_metrics, predict
from xai_chest.utils import get_device, get_logger, set_seed

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--output-dir", default=None, help="override the output folder")
    parser.add_argument("--checkpoint", default=None,
                        help="override the checkpoint, for an external model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    paths = config.paths.ensure()
    set_seed(config.seed)
    device = get_device()

    _, _, test_df = load_splits(paths)
    test_loader = build_loader(test_df, config.data, training=False)

    model = build_model(config.model)
    model = load_checkpoint(model, args.checkpoint or paths.checkpoint, device)

    labels, probabilities = predict(model, test_loader, device)
    metrics = classification_metrics(labels, probabilities)
    for name, value in metrics.items():
        logger.info("%-9s %.4f", name, value)

    with open(paths.metrics_json, "w", encoding="utf-8") as handle:
        json.dump({k: round(v, 4) for k, v in metrics.items()}, handle, indent=2)

    predictions = test_df[["patientId", "label", "target"]].copy()
    predictions["prob_pneumonia"] = probabilities
    predictions["prediction"] = (probabilities >= DECISION_THRESHOLD).astype(int)
    predictions.to_csv(paths.predictions_csv, index=False)

    if paths.history_csv.exists():
        figure = training_report(pd.read_csv(paths.history_csv), labels, probabilities,
                                 metrics, paths.figures / f"{config.name}_evaluation.png")
        logger.info("Evaluation figure written to %s", figure)
    else:
        logger.warning("No history file, the evaluation figure is skipped")

    logger.info("Evaluation done. Metrics %s, predictions %s",
                paths.metrics_json, paths.predictions_csv)


if __name__ == "__main__":
    main()
