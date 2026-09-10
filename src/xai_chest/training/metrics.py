"""The metric set fixed in the experimental configuration of the project.

Accuracy, precision, recall, F1 and AUC, the protocol of the PLOS ONE 2024 paper of the
state of the art table. Recall is read first in this project since a false negative is
a missed pneumonia, and AUC is what model selection monitors because it does not depend
on the decision threshold.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

DECISION_THRESHOLD = 0.5


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray,
                           threshold: float = DECISION_THRESHOLD) -> Dict[str, float]:
    """Score one set of predictions against its ground truth."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
    }


def confusion(labels: np.ndarray, probabilities: np.ndarray,
              threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    """The confusion matrix, whose error types matter more here than the totals."""
    return confusion_matrix(labels, (probabilities >= threshold).astype(int))


def confusion_category(target: int, prediction: int) -> str:
    """Name the confusion cell of one image.

    The saliency study samples by category, because the heatmap of an error is what
    reveals the failure mode, a use of explanations promoted by the PLOS ONE 2024 study.
    """
    if target == 1 and prediction == 1:
        return "true_positive"
    if target == 0 and prediction == 0:
        return "true_negative"
    if target == 0 and prediction == 1:
        return "false_positive"
    return "false_negative"
