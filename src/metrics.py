"""
metrics.py - shared evaluation utilities (used by the classical and RoBERTa scripts).

Main metric: true-positive rate (share of AI texts detected) at a threshold chosen
so that only 5% of VALIDATION human texts are flagged as AI (Dugan et al., 2024).
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def threshold_at_fpr(human_scores, target_fpr=0.05):
    """Score threshold above which at most `target_fpr` of human texts fall.

    Must be computed on VALIDATION human texts only, never on test data.
    """
    return float(np.quantile(np.asarray(human_scores), 1.0 - target_fpr))


def evaluate(scores, labels, threshold):
    """Metrics for one test condition. Label 1 = AI-generated, 0 = human."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = (scores > threshold).astype(int)
    ai, human = labels == 1, labels == 0
    both_classes = ai.any() and human.any()
    return {
        "tpr_at_5fpr": float(preds[ai].mean()) if ai.any() else np.nan,
        "fpr_observed": float(preds[human].mean()) if human.any() else np.nan,
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else np.nan,
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "n": int(len(labels)),
    }