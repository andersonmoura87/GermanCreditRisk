"""Feature table for modeling with IV/WoE encoding and metadata exports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.features.iv_woe import build_woe_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build X-ready feature matrix and feature name list (target excluded).

    Args:
        df: Interim dataframe including ``credit_risk``.

    Returns:
        Tuple of (full table with same columns order, feature column names).

    Raises:
        ValueError: If ``credit_risk`` is missing.
    """
    if "credit_risk" not in df.columns:
        raise ValueError("Coluna 'credit_risk' obrigatória ausente.")
    feature_cols = [c for c in df.columns if c != "credit_risk"]
    return df, feature_cols


def run() -> Path:
    """
    Read interim Parquet, apply IV/WoE encoding (configurable), write processed Parquet.

    Returns:
        Path to processed Parquet.
    """
    logging.basicConfig(level=logging.INFO)
    params = _load_params()
    interim = PROJECT_ROOT / params["paths"]["interim_parquet"]
    processed = PROJECT_ROOT / params["paths"]["processed_parquet"]
    processed.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(interim)
    iv_cfg = params.get("iv_woe", {})
    encoding = str(iv_cfg.get("encoding", "woe")).lower()
    max_bins = int(iv_cfg.get("max_bins", 10))

    maps_path = processed.parent / "woe_value_maps.joblib"
    iv_path = processed.parent / "iv_summary.csv"

    if encoding == "raw":
        out_df, feats = build_feature_matrix(df)
        _, iv_summary, maps = build_woe_features(df, "credit_risk", max_bins=max_bins)
        iv_summary.to_csv(iv_path, index=False)
        joblib.dump({"target_col": "credit_risk", "maps": {}}, maps_path)
        logger.info(
            "IV/WoE (somente relatório) — top feature: %s",
            iv_summary.head(1).to_dict("records"),
        )
    else:
        out_df, iv_summary, maps = build_woe_features(df, "credit_risk", max_bins=max_bins)
        feats = [c for c in out_df.columns if c != "credit_risk"]
        iv_summary.to_csv(iv_path, index=False)
        joblib.dump({"target_col": "credit_risk", "maps": maps}, maps_path)
        logger.info(
            "IV/WoE gravado — top feature: %s",
            iv_summary.head(1).to_dict("records"),
        )

    meta_path = processed.parent / "feature_columns.json"
    meta_path.write_text(json.dumps(feats), encoding="utf-8")
    out_df.to_parquet(processed, index=False)
    logger.info(
        "Features gravadas em %s (%s colunas, encoding=%s)", processed, len(feats), encoding
    )
    return processed


def main() -> None:
    run()


if __name__ == "__main__":
    main()
