"""Cross cutting helpers: reproducibility, logging, device and dataset discovery."""
from __future__ import annotations

import glob
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints once, whatever the number of imports."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
    return logging.getLogger(name)


logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Fix every random generator the pipeline uses.

    Reproducibility is an explicit deliverable of the project, and the seed also makes
    the backbone comparison fair: with the same seed and the same split logic, every
    backbone sees exactly the same patients in each subset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info("Seed fixed at %d", seed)


def get_device() -> torch.device:
    """The GPU when there is one, the CPU otherwise."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    return device


def resolve_data_root(explicit: Optional[str] = None) -> Path:
    """Locate the RSNA competition folder.

    An explicit path wins. Otherwise the folder is found by searching for its label
    file, first under the Kaggle input mount whose name changes between sessions, then
    under the current directory for a local run. Searching rather than hardcoding is
    what keeps the same code valid on Kaggle and on a laptop.
    """
    if explicit:
        root = Path(explicit)
        if not (root / "stage_2_train_labels.csv").exists():
            raise FileNotFoundError(f"No stage_2_train_labels.csv under {root}")
        return root

    for pattern in ("/kaggle/input/**/stage_2_train_labels.csv",
                    "**/stage_2_train_labels.csv"):
        matches = glob.glob(pattern, recursive=True)
        if matches:
            root = Path(matches[0]).parent
            logger.info("Dataset root resolved to %s", root)
            return root

    raise FileNotFoundError(
        "stage_2_train_labels.csv was not found. Attach the RSNA competition on Kaggle, "
        "or pass the folder with --data-root.")


def find_file(filename: str, search_roots=("/kaggle/input", ".")) -> Path:
    """Find one file by name under the usual roots, for artefacts of a previous step."""
    for root in search_roots:
        matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError(f"{filename} was not found under {search_roots}")
