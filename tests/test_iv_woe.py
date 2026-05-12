"""Tests for IV/WoE helpers."""

from __future__ import annotations

import pandas as pd
from src.features.iv_woe import apply_value_to_woe, build_woe_features, scalar_key


def test_scalar_key_int_float() -> None:
    assert scalar_key(1) == "1"
    assert scalar_key(1.0) == "1"
    assert scalar_key(1.5) == "1.5"


def test_build_woe_features_shape_and_maps() -> None:
    df = pd.DataFrame(
        {
            "x": [0, 0, 1, 1, 2, 2],
            "credit_risk": [0, 0, 1, 1, 1, 1],
        }
    )
    out, summary, maps = build_woe_features(df, "credit_risk", max_bins=5)
    assert "credit_risk" in out.columns
    assert "x" in out.columns
    assert not summary.empty
    assert "x" in maps


def test_apply_value_to_woe_roundtrip() -> None:
    df = pd.DataFrame({"x": [0, 1, 2], "credit_risk": [0, 1, 1]})
    out, _, maps = build_woe_features(df, "credit_risk", max_bins=5)
    enc = apply_value_to_woe(df, maps)
    assert enc["x"].tolist() == out["x"].tolist()
