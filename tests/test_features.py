"""Feature pipeline unit tests."""

from __future__ import annotations

import pandas as pd
from src.features.build_features import build_feature_matrix


def test_build_feature_matrix_columns() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "credit_risk": [0, 1],
        }
    )
    out, feats = build_feature_matrix(df)
    assert list(out.columns) == ["a", "credit_risk"]
    assert feats == ["a"]
