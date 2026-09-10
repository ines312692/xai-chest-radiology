"""Step five. Score the saliency maps and rank the methods.

    python scripts/benchmark_cams.py --config configs/resnet50_rsna.yaml

Runs the two protocols on the maps produced by explain.py, writes the per image scores,
the summary table and the two figures of the thesis. This is the numerical answer to
the research question on the clinical relevance of the attention maps.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from xai_chest.config import ExperimentConfig
from xai_chest.data import build_transforms, load_study
from xai_chest.explain import (deletion_insertion, ground_truth_mask, load_heatmap,
                               localization_scores, method_names, summarise)
from xai_chest.models import build_model, load_checkpoint
from xai_chest.reporting import cam_comparison_grid, cam_metric_barplots
from xai_chest.utils import get_device, get_logger, set_seed

logger = get_logger("benchmark_cams")

CATEGORIES = ["true_positive", "true_negative", "false_positive", "false_negative"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--output-dir", default=None, help="override the output folder")
    parser.add_argument("--skip-faithfulness", action="store_true",
                        help="localisation only, when the GPU budget is short")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir

    paths = config.paths.ensure()
    set_seed(config.seed)
    device = get_device()

    study = pd.read_csv(paths.root / f"{config.name}_study_set.csv")
    boxes = pd.read_csv(paths.boxes_csv)
    methods = config.explain.methods
    localization_set = study[study["in_localization_set"]].reset_index(drop=True)

    model = build_model(config.model)
    model = load_checkpoint(model, paths.checkpoint, device)
    transform = build_transforms(config.data, training=False)

    localization_rows, faithfulness_rows = [], []
    for index, row in localization_set.iterrows():
        image, _, shape = load_study(row["filepath"], config.data.image_size)
        tensor = transform(image).unsqueeze(0).to(device)
        mask = ground_truth_mask(boxes, row["patientId"], shape,
                                 config.data.image_size)

        for method in methods:
            heatmap = load_heatmap(paths.heatmaps, row["patientId"], method)
            scores = localization_scores(heatmap, mask,
                                         config.explain.binarization_ratio)
            localization_rows.append({"patientId": row["patientId"],
                                      "method": method, **scores})

            if not args.skip_faithfulness:
                faith = deletion_insertion(
                    model, tensor, heatmap, int(row["prediction"]), device,
                    steps=config.explain.perturbation_steps,
                    blur_kernel=config.explain.blur_kernel)
                faithfulness_rows.append({"patientId": row["patientId"],
                                          "method": method, **faith})

        if (index + 1) % 10 == 0:
            logger.info("scored %d of %d studies", index + 1, len(localization_set))

    localization = pd.DataFrame(localization_rows)
    localization.to_csv(paths.localization_csv, index=False)
    logger.info("Localisation:\n%s",
                localization.groupby("method")[["iou", "hit"]].mean().round(4))

    if args.skip_faithfulness:
        logger.warning("Faithfulness skipped, no summary table is produced")
        return

    faithfulness = pd.DataFrame(faithfulness_rows)
    faithfulness.to_csv(paths.faithfulness_csv, index=False)

    summary = summarise(localization, faithfulness)
    summary.to_csv(paths.cam_summary_csv)
    logger.info("Summary:\n%s", summary)

    cam_metric_barplots(summary, paths.figures / f"{config.name}_cam_metrics.png")

    examples = []
    for category in CATEGORIES:
        pool = study[study["category"] == category]
        if pool.empty:
            continue
        row = pool.iloc[0]
        _, display, shape = load_study(row["filepath"], config.data.image_size)
        examples.append({"patientId": row["patientId"], "category": category,
                         "display": display, "shape": shape})
    if examples:
        figure = cam_comparison_grid(examples, methods, method_names(methods),
                                     paths.heatmaps, boxes, config.data.image_size,
                                     paths.figures / f"{config.name}_cam_grid.png")
        logger.info("Qualitative figure written to %s", figure)

    logger.info("Benchmark done. Summary table %s", paths.cam_summary_csv)


if __name__ == "__main__":
    main()
