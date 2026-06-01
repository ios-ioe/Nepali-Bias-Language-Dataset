import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report,
)
from typing import Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABEL_COLUMNS, THRESHOLD


def binarize(probs: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Convert sigmoid probabilities → binary predictions."""
    return (probs >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = THRESHOLD,
) -> Dict[str, float]:
    """
    Compute micro/macro F1, precision, recall, and Hamming loss.

    Parameters
    ----------
    y_true : shape (N, num_labels)  — ground-truth binary labels
    y_prob : shape (N, num_labels)  — predicted probabilities
    """
    y_pred = binarize(y_prob, threshold)

    return {
        "hamming_loss":    hamming_loss(y_true, y_pred),
        "f1_micro":        f1_score(y_true, y_pred, average="micro",  zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro",  zero_division=0),
        "f1_samples":      f1_score(y_true, y_pred, average="samples",zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro":    recall_score(y_true, y_pred, average="micro", zero_division=0),
    }


def classification_report_str(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = THRESHOLD,
) -> str:
    """Per-label classification report."""
    y_pred = binarize(y_prob, threshold)
    return classification_report(
        y_true, y_pred, target_names=LABEL_COLUMNS, zero_division=0
    )
