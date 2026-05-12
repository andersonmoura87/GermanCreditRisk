"""Credit-style metrics: KS, Gini, AUC-ROC, calibration (Brier/ECE) e fairness slices."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métricas de ranking (crédito)
# ---------------------------------------------------------------------------


def credit_ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute the Kolmogorov–Smirnov statistic between score distributions by class.

    Args:
        y_true: Binary labels (0 = mau, 1 = bom).
        y_score: Predicted probability of the positive class (bom).

    Returns:
        KS statistic in ``[0, 1]``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    bad = y_score[y_true == 0]
    good = y_score[y_true == 1]
    if bad.size == 0 or good.size == 0:
        return 0.0
    thresholds = np.sort(np.unique(np.concatenate([bad, good])))
    ks_max = 0.0
    for t in thresholds:
        cdf_bad = (bad <= t).mean()
        cdf_good = (good <= t).mean()
        ks_max = max(ks_max, abs(cdf_good - cdf_bad))
    return float(ks_max)


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Gini = 2 * AUC - 1 for a binary classifier (ranking quality).

    Args:
        y_true: Binary labels.
        y_score: Predicted probability of the positive class.

    Returns:
        Gini coefficient.
    """
    auc = roc_auc_score(y_true, y_score)
    return float(2 * auc - 1)


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Wrapper for ``roc_auc_score`` with consistent typing."""
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Métricas de calibração
# ---------------------------------------------------------------------------


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Brier score — mean squared error between predicted probabilities and outcomes.

    Lower is better; a naïve classifier predicting the base rate yields ~0.21 for 70/30 splits.

    Args:
        y_true: Binary labels.
        y_proba: Predicted probability of the positive class.

    Returns:
        Brier score in ``[0, 1]``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    return float(np.mean((y_proba - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures average absolute difference between predicted confidence and observed
    accuracy, weighted by bin size.

    Args:
        y_true: Binary labels.
        y_proba: Predicted probability of the positive class.
        n_bins: Number of equally-spaced bins in ``[0, 1]``.

    Returns:
        ECE in ``[0, 1]`` (lower = better calibrated).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=False):
        # inclui borda superior no último bin
        mask = (y_proba >= low) & (y_proba <= high if high == 1.0 else y_proba < high)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_proba[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Análise de fairness
# ---------------------------------------------------------------------------


def fairness_score_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sensitive: pd.Series,
    *,
    positive_label: int = 1,
) -> dict[str, Any]:
    """
    Group-wise mean predicted score by sensitive attribute (exploratory fairness view).

    Args:
        y_true: Binary labels.
        y_score: Predicted scores.
        sensitive: Series aligned with y (e.g., gender proxy / age bin).
        positive_label: Label treated as positive for AUC within each slice.

    Returns:
        Dictionary of group -> summary metrics.
    """
    df = pd.DataFrame({"y": y_true, "s": y_score, "a": sensitive})
    out: dict[str, Any] = {}
    for key, grp in df.groupby("a", dropna=False):
        if grp["y"].nunique() < 2:
            continue
        label = str(key)
        out[label] = {
            "auc": roc_auc_score(grp["y"], grp["s"]),
            "mean_score": float(grp["s"].mean()),
            "n": int(len(grp)),
            "rate_positive": float((grp["y"] == positive_label).mean()),
        }
    return out


def fairness_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sensitive: pd.Series,
    *,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> pd.DataFrame:
    """
    Per-group fairness DataFrame with demographic parity and equalized odds metrics.

    Metrics produced per group:
    - ``n``: sample count
    - ``rate_positive``: proportion of positive labels (ground truth)
    - ``selection_rate``: proportion predicted positive at ``threshold``
    - ``auc``: within-group AUC-ROC
    - ``f1``: within-group F1 score (positive class)
    - ``demographic_parity_diff``: difference in selection_rate vs. overall rate
    - ``equal_opportunity_diff``: difference in TPR vs. overall TPR

    Args:
        y_true: Binary labels.
        y_score: Predicted probability of the positive class.
        sensitive: Series aligned with y (e.g., gender proxy, age bin).
        threshold: Decision threshold for converting probabilities to labels.
        positive_label: Positive class label.

    Returns:
        DataFrame indexed by group value, one row per group.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred_all = (y_score_arr >= threshold).astype(int)

    overall_selection = float(y_pred_all.mean())
    # TPR global (taxa de verdadeiros positivos)
    pos_mask_all = y_true_arr == positive_label
    overall_tpr = float(y_pred_all[pos_mask_all].mean()) if pos_mask_all.any() else 0.0

    rows: list[dict[str, Any]] = []
    sens_arr = sensitive.to_numpy()
    for group in sorted(set(sens_arr)):
        mask = sens_arr == group
        yt = y_true_arr[mask]
        ys = y_score_arr[mask]
        yp = y_pred_all[mask]

        if len(np.unique(yt)) < 2:
            # grupos sem os dois rótulos não permitem AUC
            auc_val = float("nan")
        else:
            auc_val = float(roc_auc_score(yt, ys))

        pos_mask = yt == positive_label
        tpr = float(yp[pos_mask].mean()) if pos_mask.any() else float("nan")
        sel_rate = float(yp.mean())

        try:
            f1 = float(f1_score(yt, yp, pos_label=positive_label, zero_division=0))
        except Exception:
            f1 = float("nan")

        rows.append(
            {
                "group": str(group),
                "n": int(mask.sum()),
                "rate_positive": float(yt.mean()),
                "selection_rate": sel_rate,
                "auc": auc_val,
                "f1": f1,
                "demographic_parity_diff": sel_rate - overall_selection,
                "equal_opportunity_diff": tpr - overall_tpr,
            }
        )

    return pd.DataFrame(rows).set_index("group")
