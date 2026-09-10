"""Step one. Build the classification table and freeze the patient level split.

    python scripts/preprocess.py --config configs/resnet50_rsna.yaml

Writes rsna_train.csv, rsna_val.csv, rsna_test.csv and rsna_bboxes_eval.csv in the
output folder, plus a preview figure of a few studies with their expert boxes. The
split depends only on the seed and the proportions, so running this once and reusing
its CSV files is what guarantees that both backbones are compared on the same patients.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from xai_chest.config import ExperimentConfig
from xai_chest.data import (build_classification_table, export_expert_boxes,
                            patient_level_split, read_dicom, save_splits)
from xai_chest.reporting import dataset_preview
from xai_chest.utils import get_logger, resolve_data_root, set_seed

logger = get_logger("preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--data-root", default=None,
                        help="RSNA folder, resolved by search when omitted")
    parser.add_argument("--output-dir", default=None, help="override the output folder")
    parser.add_argument("--no-figure", action="store_true",
                        help="skip the preview figure, which decodes six DICOM files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.data_root:
        config.data.root = args.data_root

    paths = config.paths.ensure()
    set_seed(config.seed)

    data_root = resolve_data_root(config.data.root)
    table = build_classification_table(data_root, config.data)
    splits = patient_level_split(table, config.data, config.seed)
    save_splits(splits, paths)
    boxes = export_expert_boxes(data_root, splits["test"], paths.boxes_csv)

    if not args.no_figure:
        samples = []
        for label in ("NORMAL", "PNEUMONIA"):
            subset = splits["train"][splits["train"]["label"] == label]
            for _, row in subset.sample(3, random_state=config.seed).iterrows():
                samples.append({"label": label, "patientId": row["patientId"],
                                "array": read_dicom(row["filepath"])})
        all_boxes = pd.read_csv(data_root / "stage_2_train_labels.csv").dropna(
            subset=["x"])
        figure = dataset_preview(samples, all_boxes,
                                 paths.figures / "dataset_preview.png")
        logger.info("Preview figure written to %s", figure)

    config.save(paths.resolved_config)
    logger.info("Preprocessing done. Expert boxes: %d. Outputs in %s",
                len(boxes), paths.root)


if __name__ == "__main__":
    main()
