"""Step four. Produce the saliency maps of the study set.

    python scripts/explain.py --config configs/resnet50_rsna.yaml

The study set has two parts. The localisation part holds correctly classified pneumonia
images that carry expert boxes, since mIoU and hit rate are only defined where a ground
truth exists, following the Nature Machine Intelligence 2022 protocol. The qualitative
part adds examples of the four confusion cells, because the map of an error is what
reveals the failure mode.

Every map is written twice, as a raw array for the scoring step and as an overlay for
the figures.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from xai_chest.config import ExperimentConfig
from xai_chest.data import build_transforms, load_study
from xai_chest.explain import build_cam_engines, compute_heatmap, save_heatmap
from xai_chest.models import build_model, load_checkpoint
from xai_chest.training import confusion_category
from xai_chest.utils import get_device, get_logger, set_seed

logger = get_logger("explain")

CATEGORIES = ["true_positive", "true_negative", "false_positive", "false_negative"]


def build_study_set(predictions: pd.DataFrame, boxes: pd.DataFrame,
                    config: ExperimentConfig) -> pd.DataFrame:
    """Draw the localisation subset and the qualitative subset, then merge them."""
    predictions = predictions.copy()
    predictions["category"] = predictions.apply(
        lambda row: confusion_category(int(row["target"]), int(row["prediction"])),
        axis=1)
    logger.info("Confusion cells:\n%s", predictions["category"].value_counts())

    boxed = set(boxes["patientId"])
    pool = predictions[(predictions["category"] == "true_positive")
                       & (predictions["patientId"].isin(boxed))]
    localization = pool.sample(min(config.explain.localization_samples, len(pool)),
                               random_state=config.seed)
    localization = localization.reset_index(drop=True)

    qualitative = pd.concat([
        predictions[predictions["category"] == category].sample(
            min(config.explain.qualitative_samples_per_category,
                len(predictions[predictions["category"] == category])),
            random_state=config.seed)
        for category in CATEGORIES if (predictions["category"] == category).any()
    ]).reset_index(drop=True)

    study = pd.concat([localization, qualitative])
    study = study.drop_duplicates("patientId").reset_index(drop=True)
    study["in_localization_set"] = study["patientId"].isin(localization["patientId"])
    logger.info("Localisation subset %d, study set %d", len(localization), len(study))
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--output-dir", default=None, help="override the output folder")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="subset of the configured saliency methods")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    paths = config.paths.ensure()
    set_seed(config.seed)
    device = get_device()

    predictions = pd.read_csv(paths.predictions_csv)
    test_split = pd.read_csv(paths.test_csv)[["patientId", "filepath"]]
    predictions = predictions.merge(test_split, on="patientId", how="left")
    boxes = pd.read_csv(paths.boxes_csv)

    study = build_study_set(predictions, boxes, config)
    study.to_csv(paths.root / f"{config.name}_study_set.csv", index=False)

    model = build_model(config.model)
    model = load_checkpoint(model, paths.checkpoint, device)
    engines = build_cam_engines(model, config.model.architecture, config.explain,
                                args.methods)
    transform = build_transforms(config.data, training=False)

    for index, row in study.iterrows():
        image, display, _ = load_study(row["filepath"], config.data.image_size)
        tensor = transform(image).unsqueeze(0).to(device)
        for method, engine in engines.items():
            heatmap = compute_heatmap(engine, tensor, int(row["prediction"]))
            save_heatmap(heatmap, display, paths.heatmaps, row["patientId"], method)
        if (index + 1) % 10 == 0:
            logger.info("processed %d of %d studies", index + 1, len(study))

    logger.info("Saliency maps written to %s", paths.heatmaps)


if __name__ == "__main__":
    main()
