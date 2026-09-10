"""Experiment configuration.

An experiment is fully described by one YAML file, so a run can be reproduced from the
file alone and two runs can be compared by diffing their configurations. The defaults
written here are the values used by the notebooks of the project, each justified in the
state of the art table, so a configuration file only needs to state what it changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Where the data is and how it becomes a classification table."""

    # Root of the RSNA competition folder. When left empty it is resolved by searching
    # for stage_2_train_labels.csv, which keeps the code valid on Kaggle where the mount
    # name changes between sessions.
    root: Optional[str] = None

    # The two unambiguous groups of the detailed class file. The third group, No Lung
    # Opacity Not Normal, holds other pathologies without pneumonia and is excluded so
    # the negative class does not teach the model that an effusion is normal.
    kept_classes: List[str] = field(default_factory=lambda: ["Normal", "Lung Opacity"])
    positive_class: str = "Lung Opacity"

    # Proportions of the patient level split, the values of the Scientific Reports 2025
    # comparison and of the MedVAG paper.
    val_size: float = 0.10
    test_size: float = 0.20

    image_size: int = 224
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation of the MDPI Information 2025 paper, training split only
    rotation_degrees: float = 15.0
    horizontal_flip_probability: float = 0.5
    zoom_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    batch_size: int = 32
    num_workers: int = 2


@dataclass
class ModelConfig:
    """Which backbone is fine tuned and how it is optimised."""

    architecture: str = "resnet50"          # resnet50 or densenet121
    pretrained: bool = True
    num_classes: int = 2

    learning_rate: float = 1e-4             # fine tuning value of the NMI 2022 benchmark
    epochs: int = 50
    patience: int = 8                       # early stopping on the validation AUC
    scheduler_factor: float = 0.1
    scheduler_patience: int = 2
    class_weighting: bool = True            # inverse frequency, for the class imbalance


@dataclass
class ExplainConfig:
    """Which saliency methods are produced and how they are scored."""

    methods: List[str] = field(
        default_factory=lambda: ["gradcam", "gradcampp", "eigencam", "layercam",
                                 "scorecam"])

    # Localisation protocol of the Nature Machine Intelligence 2022 benchmark
    localization_samples: int = 50
    binarization_ratio: float = 0.5         # keep the pixels above half of the maximum

    # Faithfulness protocol of the RISE paper, Petsiuk et al. BMVC 2018
    perturbation_steps: int = 20
    blur_kernel: int = 25

    qualitative_samples_per_category: int = 5
    scorecam_batch_size: int = 16


@dataclass
class ExperimentConfig:
    """One complete experiment, the unit that scripts consume."""

    name: str = "resnet50_rsna"
    seed: int = 42
    output_dir: str = "outputs/resnet50_rsna"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Read a configuration file, the unspecified fields keeping their default."""
        with open(path, "r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        sections = {"data": DataConfig, "model": ModelConfig, "explain": ExplainConfig}
        kwargs: Dict[str, Any] = {k: v for k, v in raw.items() if k not in sections}
        for key, section_cls in sections.items():
            kwargs[key] = section_cls(**(raw.get(key) or {}))
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the resolved configuration next to the results of the run.

        A run is only reproducible if the values actually used are stored with its
        outputs, defaults included, rather than only the file the user edited.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    @property
    def paths(self) -> "OutputPaths":
        return OutputPaths(Path(self.output_dir), self.name)


@dataclass
class OutputPaths:
    """Every file a run reads or writes, named in one place.

    Scripts never build a path by concatenation, so renaming an artefact is a one line
    change here and no script can disagree with another about where a file lives.
    """

    root: Path
    name: str

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def ensure(self) -> "OutputPaths":
        for directory in (self.root, self.heatmaps, self.figures):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    # Splits produced by the preprocessing step, shared by every backbone
    @property
    def train_csv(self) -> Path: return self.root / "rsna_train.csv"

    @property
    def val_csv(self) -> Path: return self.root / "rsna_val.csv"

    @property
    def test_csv(self) -> Path: return self.root / "rsna_test.csv"

    @property
    def boxes_csv(self) -> Path: return self.root / "rsna_bboxes_eval.csv"

    # Produced by the training and evaluation steps, named after the experiment
    @property
    def checkpoint(self) -> Path: return self.root / f"{self.name}_best.pth"

    @property
    def history_csv(self) -> Path: return self.root / f"{self.name}_history.csv"

    @property
    def metrics_json(self) -> Path: return self.root / f"{self.name}_metrics.json"

    @property
    def predictions_csv(self) -> Path:
        return self.root / f"{self.name}_test_predictions.csv"

    @property
    def resolved_config(self) -> Path: return self.root / f"{self.name}_config.yaml"

    # Produced by the explanation steps
    @property
    def heatmaps(self) -> Path: return self.root / "heatmaps"

    @property
    def figures(self) -> Path: return self.root / "figures"

    @property
    def localization_csv(self) -> Path:
        return self.root / f"{self.name}_cam_localization.csv"

    @property
    def faithfulness_csv(self) -> Path:
        return self.root / f"{self.name}_cam_faithfulness.csv"

    @property
    def cam_summary_csv(self) -> Path: return self.root / f"{self.name}_cam_summary.csv"
