"""Torch dataset, transformations and loaders.

The augmentation policy reproduces the parameters of the MDPI Information 2025 paper of
the state of the art table, and applies to the training split only, since augmenting a
validation or test image would change what is being measured.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from .dicom import read_dicom
from PIL import Image


def build_transforms(config: DataConfig, training: bool) -> T.Compose:
    """Return the transformation pipeline of one split.

    Resizing to the configured side matches the resolution the backbones were
    pretrained at, and the ImageNet channel statistics are imposed by those pretrained
    weights rather than chosen.
    """
    steps = [T.Resize((config.image_size, config.image_size))]
    if training:
        steps += [
            T.RandomRotation(config.rotation_degrees),
            T.RandomHorizontalFlip(p=config.horizontal_flip_probability),
            T.RandomAffine(degrees=0, scale=tuple(config.zoom_range)),
        ]
    steps += [T.ToTensor(), T.Normalize(config.imagenet_mean, config.imagenet_std)]
    return T.Compose(steps)


class RsnaDataset(Dataset):
    """One row of a split table becomes one tensor and one target.

    The DICOM is decoded on access rather than cached, because the full dataset does not
    fit in the memory of a Kaggle session, and the grayscale array is replicated to
    three channels for the RGB backbones.
    """

    def __init__(self, frame: pd.DataFrame, transform: T.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.fromarray(read_dicom(row["filepath"])).convert("RGB")
        target = torch.tensor(int(row["target"]), dtype=torch.long)
        return self.transform(image), target


def build_loader(frame: pd.DataFrame, config: DataConfig, training: bool) -> DataLoader:
    """Wrap one split into a loader, shuffled for training only."""
    dataset = RsnaDataset(frame, build_transforms(config, training))
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                      num_workers=config.num_workers, pin_memory=True)
