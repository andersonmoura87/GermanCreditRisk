"""Information Value (IV) and Weight of Evidence (WoE) encoding (pandas-only, no scorecardpy)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-9


def scalar_key(value: Any) -> str:
    """
    Serialize a scalar feature value to a dict key for ``valor -> WoE`` maps.

    Args:
        value: Cell value from a pandas Series.

    Returns:
        Stable string key (``__nan__`` for missing values).
    """
    if pd.isna(value):
        return "__nan__"
    fv = float(value)
    if abs(fv - round(fv)) < 1e-9:
        return str(int(round(fv)))
    return f"{fv:.12g}"


def _woe_iv_vectors(count_good: np.ndarray, count_bad: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute per-bin WoE and total IV from bin counts.

    Args:
        count_good: Count of ``target == 1`` (bom) per bin.
        count_bad: Count of ``target == 0`` (mau) per bin.

    Returns:
        Tuple ``(woe_per_bin, iv_total)`` aligned to bin order.
    """
    cg = np.asarray(count_good, dtype=float)
    cb = np.asarray(count_bad, dtype=float)
    total_g = float(cg.sum())
    total_b = float(cb.sum())
    if total_g <= 0.0 or total_b <= 0.0:
        return np.zeros_like(cg, dtype=float), 0.0
    pct_g = cg / total_g
    pct_b = cb / total_b
    woe = np.log((pct_g + EPS) / (pct_b + EPS))
    iv_total = float(((pct_g - pct_b) * woe).sum())
    return woe, iv_total


def _discretize_feature(series: pd.Series, max_bins: int) -> pd.Series:
    """
    Map a numeric feature to discrete bin ids for WoE stability.

    Low-cardinality columns use ``factorize``; high-cardinality use quantile bins.
    """
    s = series.astype(float)
    valid = s.dropna()
    nunique = int(valid.nunique())
    if nunique <= 0:
        return pd.Series(0, index=series.index, dtype=int)
    if nunique <= max_bins:
        codes, _ = pd.factorize(s, sort=True)
        return pd.Series(codes, index=series.index, dtype=int).fillna(-1).astype(int)
    q = min(max_bins, max(2, nunique))
    try:
        binned = pd.qcut(s, q=q, duplicates="drop", labels=False)
        return pd.Series(binned, index=series.index).fillna(-1).astype(int)
    except ValueError:
        codes, _ = pd.factorize(s, sort=True)
        return pd.Series(codes, index=series.index, dtype=int).fillna(-1).astype(int)


def iv_woe_single_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    *,
    max_bins: int = 10,
) -> tuple[float, dict[int, float], pd.DataFrame]:
    """
    Fit WoE mapping and IV for one feature using discretized bins.

    Args:
        df: Modeling frame including ``target``.
        feature: Feature column name.
        target: Binary target (1 = bom, 0 = mau).
        max_bins: Maximum quantile bins for continuous-like columns.

    Returns:
        ``(iv_total, woe_by_bin_id, bin_stats_table)``.
    """
    bins = _discretize_feature(df[feature], max_bins)
    tmp = pd.DataFrame({"bin": bins, "y": df[target].astype(int)})
    grp = tmp.groupby("bin", dropna=False)["y"].agg(["count", "sum"])
    grp = grp.rename(columns={"sum": "good"})
    grp["bad"] = grp["count"] - grp["good"]
    woe_vec, iv_total = _woe_iv_vectors(grp["good"].to_numpy(), grp["bad"].to_numpy())
    woe_by_bin = {
        int(idx): float(w) for idx, w in zip(grp.index.astype(int), woe_vec, strict=False)
    }
    return iv_total, woe_by_bin, grp.reset_index()


def encode_feature_woe(
    df: pd.DataFrame, feature: str, target: str, *, max_bins: int
) -> tuple[float, pd.Series]:
    """
    Return IV and a WoE-encoded series aligned to ``df.index``.

    Args:
        df: Source frame.
        feature: Column to encode.
        target: Binary target column.
        max_bins: Binning budget.

    Returns:
        ``(iv_total, woe_series)``.
    """
    bins = _discretize_feature(df[feature], max_bins)
    iv_total, woe_by_bin, _ = iv_woe_single_feature(df, feature, target, max_bins=max_bins)
    mapped = bins.map(lambda b: woe_by_bin.get(int(b), 0.0)).astype(float)
    return iv_total, mapped


def build_woe_features(
    df: pd.DataFrame,
    target_col: str,
    *,
    max_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Replace all non-target numeric columns with WoE encodings; keep ``target_col``.

    Args:
        df: Interim / cleaned modeling table.
        target_col: Name of binary target.
        max_bins: Max bins per feature for discretization.

    Returns:
        ``(encoded_df, iv_summary, value_to_woe_maps)`` where ``iv_summary`` has columns
        ``feature``, ``iv``, and maps store ``scalar_key(valor) -> WoE`` for scoring online.
    """
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' ausente.")
    # WoE deve ser ajustado sempre sobre a distribuição original da feature vs alvo
    base = df.copy()
    out = df.copy()
    rows: list[dict[str, Any]] = []
    maps: dict[str, dict[str, float]] = {}
    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(base[col]):
            logger.warning("Ignorando coluna não numérica em WoE: %s", col)
            continue
        iv_val, woe_series = encode_feature_woe(base, col, target_col, max_bins=max_bins)
        out[col] = woe_series
        rows.append({"feature": col, "iv": iv_val})
        col_map: dict[str, float] = {}
        for raw_v, woe_v in zip(base[col], woe_series, strict=False):
            col_map[scalar_key(raw_v)] = float(woe_v)
        maps[col] = col_map
    summary = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return out, summary, maps


def apply_value_to_woe(df: pd.DataFrame, maps: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Apply ``valor -> WoE`` maps learned during ``build_woe_features``.

    Args:
        df: Raw feature frame (same column names as fit).
        maps: Mapping ``feature_name -> {scalar_key: woe}``.

    Returns:
        DataFrame with WoE-encoded columns for keys present in ``maps``.
    """
    out = df.copy()
    for col, col_map in maps.items():
        if col not in out.columns:
            continue

        def _woe_mapper(v: object, m: dict = col_map) -> float:  # noqa: B023
            return float(m.get(scalar_key(v), 0.0))

        out[col] = out[col].map(_woe_mapper)  # type: ignore[call-overload]
    return out
