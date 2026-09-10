"""Step two. Fine tune one backbone on the frozen split.

    python scripts/train.py --config configs/resnet50_rsna.yaml

Reads the CSV files produced by preprocess.py, writes the checkpoint of best validation
AUC and the epoch history. Changing the backbone means changing the configuration file,
never this script, which is what keeps the two runs comparable.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from xai_chest.config import ExperimentConfig
from xai_chest.data import build_loader, load_splits
from xai_chest.models import build_model
from xai_chest.training import fit
from xai_chest.utils import get_device, get_logger, set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--output-dir", default=None, help="override the output folder")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the epoch budget, useful for a smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.model.epochs = args.epochs

    paths = config.paths.ensure()
    set_seed(config.seed)
    device = get_device()

    train_df, val_df, _ = load_splits(paths)
    train_loader = build_loader(train_df, config.data, training=True)
    val_loader = build_loader(val_df, config.data, training=False)
    logger.info("Batches per training epoch: %d", len(train_loader))

    model = build_model(config.model).to(device)
    history = fit(model, train_loader, val_loader, train_df, config.model, device,
                  paths.checkpoint)

    pd.DataFrame(history).to_csv(paths.history_csv, index=False)
    config.save(paths.resolved_config)
    logger.info("Training done. Checkpoint %s, history %s",
                paths.checkpoint, paths.history_csv)


if __name__ == "__main__":
    main()
