# xai-chest

Library form of the pneumonia detection and saliency evaluation pipeline of the
project *Multimodal Explainable AI for Chest Radiology*, Master 2 ISI.

The notebooks under `../notebooks` remain the record of what was run on Kaggle and what
it produced. This package is the same pipeline written as reusable modules, so that a
step can be read, tested and re run alone, and so that changing the backbone is a line
in a configuration file rather than a copy of a notebook.

## Layout

```
src/
├── configs/                     one YAML file per experiment
│   ├── resnet50_rsna.yaml
│   └── densenet121_rsna.yaml
├── scripts/                     the five entry points, in pipeline order
│   ├── preprocess.py            table, patient level split, expert boxes
│   ├── train.py                 fine tuning with early stopping
│   ├── evaluate.py              metrics and figures on the test split
│   ├── explain.py               saliency maps of the study set
│   └── benchmark_cams.py        localisation, faithfulness, ranking
├── tests/                       unit tests of the logic that must not drift
└── xai_chest/                   the library
    ├── config.py                what an experiment is, and where its files live
    ├── utils.py                 seed, logging, device, dataset discovery
    ├── data/                    DICOM decoding, split, dataset, loaders
    ├── models/                  backbone factory and saliency target layer
    ├── training/                fine tuning loop and the metric set
    ├── explain/                 saliency production and the two scoring protocols
    └── reporting/               the figures of the thesis
```

Two rules keep the structure honest. A module never reads a configuration file or
parses arguments, it receives values, which is what makes it testable. A script never
contains method logic, it wires modules together, which is what keeps the two
experiments identical except for their configuration.

## Install

```bash
cd src
pip install -e .            # or: pip install -r requirements.txt
```

On Kaggle nothing needs installing beyond the saliency library, since the image already
carries the rest:

```python
!pip install grad-cam --quiet
!cp -r /kaggle/input/<your-code-dataset>/src /kaggle/working/src
%cd /kaggle/working/src
```

## Run one experiment

```bash
python scripts/preprocess.py     --config configs/resnet50_rsna.yaml
python scripts/train.py          --config configs/resnet50_rsna.yaml
python scripts/evaluate.py       --config configs/resnet50_rsna.yaml
python scripts/explain.py        --config configs/resnet50_rsna.yaml
python scripts/benchmark_cams.py --config configs/resnet50_rsna.yaml
```

Every script accepts `--output-dir` to redirect its artefacts, and `--config` is the
only required argument. `train.py` accepts `--epochs` for a smoke test,
`explain.py` accepts `--methods` to restrict the saliency methods, and
`benchmark_cams.py` accepts `--skip-faithfulness` when the GPU budget is short.

The dataset folder is resolved by searching for `stage_2_train_labels.csv`, first under
the Kaggle input mount and then under the current directory, so the same command works
on Kaggle and on a laptop. Pass `--data-root` to be explicit.

## Compare the two backbones

The comparison is controlled by construction: same seed, same split logic, same
optimisation, one difference.

```bash
python scripts/preprocess.py --config configs/densenet121_rsna.yaml
python scripts/train.py      --config configs/densenet121_rsna.yaml
python scripts/evaluate.py   --config configs/densenet121_rsna.yaml
```

Point both configurations at the same `output_dir` to share one frozen split, or leave
them separate: with the same seed and proportions the partition is identical, which the
test suite verifies.

## What each step writes

| Step | Artefacts |
|---|---|
| `preprocess` | `rsna_train.csv`, `rsna_val.csv`, `rsna_test.csv`, `rsna_bboxes_eval.csv`, preview figure |
| `train` | `<name>_best.pth`, `<name>_history.csv`, resolved configuration |
| `evaluate` | `<name>_metrics.json`, `<name>_test_predictions.csv`, evaluation figure |
| `explain` | `heatmaps/<patient>__<method>.npy` and `.png`, `<name>_study_set.csv` |
| `benchmark_cams` | `<name>_cam_localization.csv`, `<name>_cam_faithfulness.csv`, `<name>_cam_summary.csv`, two figures |

Paths are never built by concatenation in a script. They all come from `OutputPaths` in
`config.py`, so renaming an artefact is a one line change and no two scripts can
disagree about where a file lives.

## Methodological choices, and where they come from

Each of these is defended in the state of the art table of the project and is
implemented in exactly one place.

| Choice | Where | Reference |
|---|---|---|
| Normal versus Lung Opacity only, third group excluded | `data/splits.py` | binary task of MDPI Information 2025 and PLOS ONE 2024 |
| Patient level split, 70 10 20, stratified, asserted disjoint | `data/splits.py` | corrects the limitation acknowledged by PLOS ONE 2024 |
| Resize 224, ImageNet normalisation, rotation 15, flip, zoom 0.8 to 1.2 | `data/datasets.py` | augmentation parameters of MDPI Information 2025 |
| Inverse frequency class weights | `training/engine.py` | same motivation as the positive weight of Scientific Reports 2025 |
| Adam at 1e-4, ReduceLROnPlateau, early stopping on validation AUC | `training/engine.py` | fine tuning value of the Nature Machine Intelligence 2022 benchmark |
| Accuracy, precision, recall, F1, AUC | `training/metrics.py` | protocol of PLOS ONE 2024 |
| Last convolutional block as saliency target | `models/factory.py` | Selvaraju et al. ICCV 2017 |
| mIoU and hit rate against expert boxes, on true positives | `explain/evaluation.py` | Nature Machine Intelligence 2022 benchmark |
| Deletion and insertion on a blurred baseline | `explain/evaluation.py` | RISE, Petsiuk et al. BMVC 2018 |

## Tests

```bash
python -m pytest tests -q
```

Twelve tests on synthetic arrays, no dataset needed, one second to run. They cover what
a reviewer would question: that the split is leak free, deterministic and stratified,
that a box is rescaled correctly, that a map covering the lesion scores one and a map
pointing elsewhere scores zero, that the summary ranks and computes faithfulness as
stated, and that a configuration survives a save and reload.

## Known limitation

`explain/cam.peak_zone` divides the image into thirds with no lung mask, so a peak
falling on a shoulder is still named as a lung zone. The label is therefore indicative
and must be read together with the map. Replacing the grid by a lung mask is the next
improvement, and the case where the peak falls outside the mask is itself a useful
indicator for the research question on the clinical relevance of the maps.
