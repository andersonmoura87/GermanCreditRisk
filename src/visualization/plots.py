"""Reusable plotting helpers with the project color palette."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# Paleta exigida pelo AGENTS.md
COLOR_BAD = "#E24B4A"
COLOR_GOOD = "#639922"


# ---------------------------------------------------------------------------
# IV ranking
# ---------------------------------------------------------------------------
IV_STRENGTH = [
    (0.0, "< 0.02 — inútil"),
    (0.02, "0.02–0.1 — fraco"),
    (0.1, "0.1–0.3 — médio"),
    (0.3, "0.3–0.5 — forte"),
    (0.5, "> 0.5 — suspeito (overfit)"),
]


def _iv_color(iv: float) -> str:
    if iv < 0.02:
        return "#aaaaaa"
    if iv < 0.1:
        return "#f5c842"
    if iv < 0.3:
        return "#639922"
    if iv < 0.5:
        return "#2e86ab"
    return "#E24B4A"


def plot_iv_ranking(
    iv_summary: pd.DataFrame,
    *,
    top_n: int = 20,
    ax: Any | None = None,
) -> Any:
    """
    Horizontal bar chart of Information Value per feature.

    Args:
        iv_summary: DataFrame with columns ``feature`` and ``iv``.
        top_n: Number of top features to display.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib ``Axes`` used.
    """
    df = iv_summary.nlargest(top_n, "iv").reset_index(drop=True)
    colors = [_iv_color(v) for v in df["iv"]]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    ax.barh(df["feature"][::-1], df["iv"][::-1], color=colors[::-1])
    ax.axvline(0.02, color="#aaaaaa", linestyle="--", linewidth=0.8, label="inútil")
    ax.axvline(0.1, color="#f5c842", linestyle="--", linewidth=0.8, label="fraco")
    ax.axvline(0.3, color="#639922", linestyle="--", linewidth=0.8, label="médio")
    ax.set_xlabel("Information Value")
    ax.set_title("IV por feature — poder preditivo")
    ax.legend(fontsize=7)
    return ax


def plot_woe_bars(
    bin_stats: pd.DataFrame,
    feature_name: str,
    *,
    ax: Any | None = None,
) -> Any:
    """
    Bar chart of WoE per bin for a single feature.

    Args:
        bin_stats: DataFrame with ``bin`` index and ``good`` / ``bad`` counts
                   (output of ``iv_woe_single_feature``).
        feature_name: Used as axis label.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib ``Axes`` used.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    total_g = float(bin_stats["good"].sum())
    total_b = float(bin_stats["bad"].sum())
    EPS = 1e-9
    pct_g = bin_stats["good"] / (total_g + EPS)
    pct_b = bin_stats["bad"] / (total_b + EPS)
    woe = np.log((pct_g + EPS) / (pct_b + EPS))
    colors = [COLOR_BAD if w < 0 else COLOR_GOOD for w in woe]
    ax.bar(bin_stats["bin"].astype(str), woe, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Bin — {feature_name}")
    ax.set_ylabel("WoE")
    ax.set_title(f"WoE por bin: {feature_name}")
    return ax


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    label: str = "Modelo",
    color: str = COLOR_BAD,
    ax: Any | None = None,
) -> Any:
    """
    Plot a reliability (calibration) diagram for predicted probabilities.

    Args:
        y_true: Binary ground truth.
        y_proba: Predicted probability of the positive class.
        n_bins: Number of bins for ``calibration_curve``.
        label: Legend label for this curve.
        color: Line color.
        ax: Optional matplotlib axes.

    Returns:
        The matplotlib ``Axes`` used.
    """
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_proba,
        n_bins=n_bins,
        strategy="uniform",
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfeito")
    ax.plot(prob_pred, prob_true, marker="o", color=color, label=label)
    ax.set_xlabel("Probabilidade média predita")
    ax.set_ylabel("Fração de positivos")
    ax.set_title("Reliability Diagram — Curva de Calibração")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


def plot_fairness_auc(
    fairness_df: pd.DataFrame,
    *,
    metric: str = "auc",
    title: str = "AUC por grupo — análise de fairness",
    ax: Any | None = None,
) -> Any:
    """
    Horizontal bar chart of a fairness metric across demographic groups.

    Args:
        fairness_df: Output of ``evaluate.fairness_report`` (indexed by group).
        metric: Column to plot (``'auc'``, ``'selection_rate'``, ``'demographic_parity_diff'``…).
        title: Plot title.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib ``Axes`` used.
    """
    df = fairness_df[[metric]].dropna().sort_values(metric, ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.5)))

    colors = [COLOR_GOOD if v >= 0 else COLOR_BAD for v in df[metric]]
    ax.barh(df.index.astype(str), df[metric], color=colors)

    # linha de referência: zero (para diffs) ou 0.5 (para AUC)
    ref = 0.0 if "diff" in metric else 0.5
    ax.axvline(ref, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel(metric)
    ax.set_title(title)
    return ax


def plot_score_distribution_by_group(
    y_score: np.ndarray,
    sensitive: pd.Series,
    *,
    bins: int = 20,
    ax: Any | None = None,
) -> Any:
    """
    Overlapping score distribution histograms per demographic group.

    Args:
        y_score: Predicted probability of the positive class.
        sensitive: Series aligned with y_score (group labels).
        bins: Number of histogram bins.
        ax: Optional matplotlib Axes.

    Returns:
        The matplotlib ``Axes`` used.
    """
    palette = [COLOR_GOOD, COLOR_BAD, "#2e86ab", "#f5c842", "#9b59b6"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for i, (group, sub) in enumerate(pd.Series(sensitive).groupby(pd.Series(sensitive))):
        scores = y_score[sub.index]
        ax.hist(
            scores,
            bins=bins,
            alpha=0.55,
            label=str(group),
            color=palette[i % len(palette)],
            density=True,
        )
    ax.set_xlabel("Score (prob. adimplente)")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de scores por grupo")
    ax.legend()
    return ax
