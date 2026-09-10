"""Fine tuning loop and inference pass.

Model selection monitors the validation AUC and keeps the best checkpoint, the practice
of the PLOS ONE 2024 and Ensemble-CAM papers. Early stopping is what makes a fifty
epoch budget honest on a Kaggle quota: with fourteen thousand images convergence
arrives within a few epochs and the run stops itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..config import ModelConfig
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class EpochResult:
    """What one pass over a loader produced."""

    loss: float
    labels: np.ndarray
    probabilities: np.ndarray

    @property
    def auc(self) -> float:
        return float(roc_auc_score(self.labels, self.probabilities))


def class_weights(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Inverse frequency weights, to compensate the imbalance of the dataset.

    Same motivation as the positive weight of the Scientific Reports 2025 paper: without
    it the majority class dominates the loss and recall on pneumonia suffers, which is
    the error that matters clinically.
    """
    counts = frame["target"].value_counts().sort_index().to_numpy()
    weights = len(frame) / (2.0 * counts)
    logger.info("Class weights: %s", np.round(weights, 3))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, optimizer: Optional[torch.optim.Optimizer] = None
              ) -> EpochResult:
    """One pass over a loader, training when an optimizer is given.

    The probability of the positive class is collected for every image, since AUC and
    every downstream analysis need scores rather than hard decisions.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_labels, all_probabilities = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            all_labels.extend(targets.detach().cpu().numpy())
            all_probabilities.extend(
                torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())

    return EpochResult(total_loss / len(loader.dataset),
                       np.asarray(all_labels), np.asarray(all_probabilities))


def build_optimisation(model: nn.Module, config: ModelConfig, device: torch.device,
                       train_frame: pd.DataFrame):
    """Loss, optimizer and scheduler, with the values defended in the notebooks.

    Adam at 1e-4 is the fine tuning value of the Nature Machine Intelligence 2022
    benchmark, lower than the 1e-3 of MDPI because that value suits training a new head
    only while the whole network is fine tuned here. ReduceLROnPlateau on the validation
    AUC follows PLOS ONE 2024 and Ensemble-CAM.
    """
    weights = class_weights(train_frame, device) if config.class_weighting else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.scheduler_factor,
        patience=config.scheduler_patience)
    return criterion, optimizer, scheduler


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        train_frame: pd.DataFrame, config: ModelConfig, device: torch.device,
        checkpoint_path: str | Path) -> List[dict]:
    """Fine tune the backbone and keep the checkpoint of best validation AUC.

    Returns the epoch history, which the report step turns into the convergence figure
    of the thesis.
    """
    criterion, optimizer, scheduler = build_optimisation(model, config, device,
                                                         train_frame)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    best_auc, epochs_without_improvement, history = 0.0, 0, []
    for epoch in range(1, config.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device)
        val_auc = val_result.auc
        scheduler.step(val_auc)

        history.append({"epoch": epoch, "train_loss": train_result.loss,
                        "val_loss": val_result.loss, "val_auc": val_auc})
        logger.info("epoch %2d  train_loss %.4f  val_loss %.4f  val_auc %.4f",
                    epoch, train_result.loss, val_result.loss, val_auc)

        if val_auc > best_auc:
            best_auc, epochs_without_improvement = val_auc, 0
            torch.save(model.state_dict(), str(checkpoint_path))
            logger.info("new best checkpoint saved, validation AUC %.4f", val_auc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info("early stopping after %d epochs without improvement",
                            config.patience)
                break

    return history


def predict(model: nn.Module, loader: DataLoader, device: torch.device
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Score one split with the current weights, returning labels and probabilities."""
    result = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    return result.labels, result.probabilities
