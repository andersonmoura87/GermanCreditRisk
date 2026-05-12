"""Persistence of reference statistics used as baseline for PSI/CSI monitoring.

The reference snapshot is saved once at the end of the evaluate pipeline stage
and loaded each time the monitor stage or API endpoint needs to compare.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "monitoring_reference.json"


def save_reference(
    scores: np.ndarray,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    output_path: Path | None = None,
    model_version: str = "unknown",
    n_score_bins: int = 10,
) -> Path:
    """
    Serialize reference score distribution and feature summary statistics to JSON.

    Called once after training/evaluation. The resulting file is the baseline
    for all future PSI/CSI computations.

    Args:
        scores: Model probabilities from the reference (test) split.
        feature_df: Feature DataFrame aligned with ``scores``.
        feature_cols: Feature columns to persist.
        output_path: Destination path (defaults to ``data/processed/monitoring_reference.json``).
        model_version: Model version tag stored in metadata.
        n_score_bins: Number of equal-width bins in [0, 1] for the score histogram.

    Returns:
        Path to the written JSON file.
    """
    path = output_path or DEFAULT_REFERENCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    scores_arr = np.asarray(scores, dtype=float)
    bins = np.linspace(0.0, 1.0, n_score_bins + 1)
    score_counts, _ = np.histogram(scores_arr, bins=bins)

    feature_stats: dict[str, Any] = {}
    for col in feature_cols:
        if col not in feature_df.columns:
            continue
        vals = feature_df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        # Bins de quantil para cada feature (resistentes a outliers)
        quantiles = np.linspace(0, 100, n_score_bins + 1)
        edges = np.unique(np.percentile(vals, quantiles))
        if len(edges) < 2:
            edges = np.array([vals.min() - 1e-9, vals.max() + 1e-9])
        edges[0] = vals.min() - 1e-9
        edges[-1] = vals.max() + 1e-9
        counts, _ = np.histogram(vals, bins=edges)
        feature_stats[col] = {
            "bins": edges.tolist(),
            "counts": counts.tolist(),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(len(vals)),
        }

    payload: dict[str, Any] = {
        "model_version": model_version,
        "n_samples": int(len(scores_arr)),
        "score_bins": bins.tolist(),
        "score_counts": score_counts.tolist(),
        "score_mean": float(scores_arr.mean()),
        "score_std": float(scores_arr.std()),
        "feature_stats": feature_stats,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Referência de monitoramento salva em %s (%d features).", path, len(feature_stats))
    return path


def load_reference(path: Path | None = None) -> dict[str, Any]:
    """
    Load a previously saved reference snapshot.

    Args:
        path: JSON path (defaults to ``data/processed/monitoring_reference.json``).

    Returns:
        Reference dict with keys ``score_bins``, ``score_counts``, ``feature_stats``, etc.

    Raises:
        FileNotFoundError: If the reference file does not exist.
    """
    p = path or DEFAULT_REFERENCE_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Referência de monitoramento não encontrada: {p}. "
            "Execute `make evaluate` para gerá-la."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def reference_exists(path: Path | None = None) -> bool:
    """Return True if the reference snapshot file exists."""
    p = path or DEFAULT_REFERENCE_PATH
    return p.is_file()
