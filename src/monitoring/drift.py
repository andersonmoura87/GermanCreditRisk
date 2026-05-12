"""PSI (Population Stability Index) e CSI (Characteristic Stability Index).

Métricas padrão do mercado financeiro brasileiro para detecção de degradação
de modelos de crédito — alinhadas a BACEN Resolução 4.557/2017.

Referências de limiar (padrão de mercado):
    PSI / CSI < 0.10  → estável (nenhuma ação necessária)
    PSI / CSI 0.10–0.25 → atenção (investigar causa raiz)
    PSI / CSI ≥ 0.25  → deriva significativa (retreinar / escalar)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Limiares de alerta (configuráveis via parâmetro)
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25

_EPS = 1e-9  # evita log(0)


# ---------------------------------------------------------------------------
# Núcleo — bins e cálculo PSI/CSI
# ---------------------------------------------------------------------------


def _safe_proportions(counts: np.ndarray) -> np.ndarray:
    """Converte contagens em proporções, adicionando epsilon para evitar divisão por zero."""
    total = counts.sum()
    return (counts + _EPS) / (total + len(counts) * _EPS)


def psi_from_bins(
    ref_counts: np.ndarray,
    cur_counts: np.ndarray,
) -> float:
    """
    Compute PSI given pre-computed bin counts for reference and current distributions.

    PSI = Σ (cur_i - ref_i) * ln(cur_i / ref_i)

    Args:
        ref_counts: Integer counts per bin for the reference (training) period.
        cur_counts: Integer counts per bin for the current (production) period.

    Returns:
        PSI value ≥ 0. Higher = more drift.
    """
    ref_p = _safe_proportions(np.asarray(ref_counts, dtype=float))
    cur_p = _safe_proportions(np.asarray(cur_counts, dtype=float))
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _histogram(
    values: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Bin values into a fixed set of bin edges, returning counts per bin."""
    counts, _ = np.histogram(values, bins=bins)
    return counts.astype(float)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
    bins: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute Population Stability Index for model score distributions.

    Compares the distribution of predicted probabilities between a reference
    (training/validation) period and a current (production) period.

    Args:
        reference: Reference score array (probabilities in [0, 1]).
        current: Current score array (probabilities in [0, 1]).
        n_bins: Number of equal-width bins in [0, 1]. Ignored if ``bins`` is provided.
        bins: Optional pre-defined bin edges (overrides ``n_bins``).

    Returns:
        Dictionary with keys ``psi``, ``status``, ``bins``, ``ref_proportions``,
        ``cur_proportions``, ``bin_psi`` (per-bin contribution).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if bins is None:
        bins = np.linspace(0.0, 1.0, n_bins + 1)

    ref_counts = _histogram(reference, bins)
    cur_counts = _histogram(current, bins)

    ref_p = _safe_proportions(ref_counts)
    cur_p = _safe_proportions(cur_counts)
    bin_psi = (cur_p - ref_p) * np.log(cur_p / ref_p)
    total_psi = float(bin_psi.sum())

    return {
        "psi": round(total_psi, 6),
        "status": drift_status(total_psi),
        "bins": bins.tolist(),
        "ref_proportions": ref_p.tolist(),
        "cur_proportions": cur_p.tolist(),
        "bin_psi": bin_psi.tolist(),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
    }


def characteristic_stability_index(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_bins: int = 10,
    feature_bins: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Compute CSI for each feature column, using bins derived from the reference distribution.

    CSI is the per-feature analogue of PSI — detects which input variables are drifting,
    helping root-cause analysis of score degradation.

    Args:
        reference_df: Reference DataFrame (training / validation).
        current_df: Current DataFrame (production window).
        feature_cols: Columns to evaluate.
        n_bins: Equal-frequency bins derived from the reference distribution.
        feature_bins: Pre-defined per-feature bin edges (overrides derived bins).

    Returns:
        DataFrame indexed by feature with columns ``csi``, ``status``, ``n_reference``,
        ``n_current``.
    """
    feature_bins = feature_bins or {}
    rows = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            logger.warning("Coluna '%s' ausente em um dos DataFrames — CSI ignorado.", col)
            continue

        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        if col in feature_bins:
            bins = feature_bins[col]
        else:
            # Bins derivados da distribuição de referência (quantis de n_bins intervalos)
            quantiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.unique(np.percentile(ref_vals, quantiles))
            # Garante mínimo 2 bordas
            if len(bin_edges) < 2:
                bin_edges = np.array([ref_vals.min() - _EPS, ref_vals.max() + _EPS])
            # Estende bordas para cobrir current_df
            bin_edges[0] = min(bin_edges[0], cur_vals.min()) - _EPS
            bin_edges[-1] = max(bin_edges[-1], cur_vals.max()) + _EPS
            bins = bin_edges

        ref_counts = _histogram(ref_vals, bins)
        cur_counts = _histogram(cur_vals, bins)
        csi_val = psi_from_bins(ref_counts, cur_counts)

        rows.append(
            {
                "feature": col,
                "csi": round(csi_val, 6),
                "status": drift_status(csi_val),
                "n_reference": int(len(ref_vals)),
                "n_current": int(len(cur_vals)),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("csi", ascending=False).reset_index(drop=True)
    return df.set_index("feature") if not df.empty else df


# ---------------------------------------------------------------------------
# Status semafórico
# ---------------------------------------------------------------------------


def drift_status(
    value: float,
    *,
    warning_threshold: float = PSI_WARNING,
    critical_threshold: float = PSI_CRITICAL,
) -> str:
    """
    Classify a PSI/CSI value into a traffic-light status.

    Args:
        value: PSI or CSI value.
        warning_threshold: Below = ``'stable'``, above = ``'warning'``.
        critical_threshold: Above = ``'drift'``.

    Returns:
        One of ``'stable'``, ``'warning'``, ``'drift'``.
    """
    if value < warning_threshold:
        return "stable"
    if value < critical_threshold:
        return "warning"
    return "drift"


# ---------------------------------------------------------------------------
# Relatório consolidado
# ---------------------------------------------------------------------------


def stability_report(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Generate a full stability report: PSI on scores + CSI per feature.

    Args:
        reference_scores: Model scores from reference (training) period.
        current_scores: Model scores from current (production) period.
        reference_df: Feature DataFrame for reference period.
        current_df: Feature DataFrame for current period.
        feature_cols: Feature columns to evaluate.
        n_bins: Number of bins for PSI/CSI.

    Returns:
        Dict with keys ``score_psi`` (dict) and ``feature_csi`` (DataFrame).
    """
    score_psi = population_stability_index(reference_scores, current_scores, n_bins=n_bins)
    feature_csi = characteristic_stability_index(
        reference_df, current_df, feature_cols, n_bins=n_bins
    )

    n_warning = int((feature_csi["status"] == "warning").sum()) if not feature_csi.empty else 0
    n_drift = int((feature_csi["status"] == "drift").sum()) if not feature_csi.empty else 0

    logger.info(
        "PSI score=%.4f (%s) | CSI features: %d warning, %d drift",
        score_psi["psi"],
        score_psi["status"],
        n_warning,
        n_drift,
    )

    return {
        "score_psi": score_psi,
        "feature_csi": feature_csi,
        "summary": {
            "score_psi": score_psi["psi"],
            "score_status": score_psi["status"],
            "n_features_warning": n_warning,
            "n_features_drift": n_drift,
        },
    }
