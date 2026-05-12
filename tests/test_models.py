"""Tests for credit metrics: ranking, calibration and fairness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.evaluate import (
    auc_roc,
    brier_score,
    credit_ks_statistic,
    expected_calibration_error,
    fairness_report,
    gini_coefficient,
)

# ---------------------------------------------------------------------------
# Métricas de ranking
# ---------------------------------------------------------------------------


def test_perfect_separation_ks_and_auc() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_roc(y, s) == 1.0
    assert gini_coefficient(y, s) == 1.0
    assert credit_ks_statistic(y, s) > 0.5


def test_ks_symmetric() -> None:
    """KS deve ser igual independente da ordenação."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0.3, 0.7, 0.4, 0.6])
    ks1 = credit_ks_statistic(y, s)
    ks2 = credit_ks_statistic(y[::-1], s[::-1])
    assert abs(ks1 - ks2) < 1e-10


def test_gini_range() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    s = rng.uniform(0, 1, size=100)
    g = gini_coefficient(y, s)
    assert -1.0 <= g <= 1.0


# ---------------------------------------------------------------------------
# Métricas de calibração
# ---------------------------------------------------------------------------


def test_brier_perfect() -> None:
    """Brier score zero quando probabilidades são 0 ou 1 e corretas."""
    y = np.array([0, 0, 1, 1], dtype=float)
    s = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    assert brier_score(y, s) == pytest.approx(0.0)


def test_brier_worst() -> None:
    """Brier score máximo quando probabilidades são invertidas."""
    y = np.array([0, 0, 1, 1], dtype=float)
    s = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
    assert brier_score(y, s) == pytest.approx(1.0)


def test_brier_range() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200).astype(float)
    s = rng.uniform(0, 1, size=200)
    bs = brier_score(y, s)
    assert 0.0 <= bs <= 1.0


def test_ece_perfect_calibration() -> None:
    """ECE perto de zero quando cada bin está perfeitamente calibrado."""
    # probabilidades constantes por faixa com acurácia correspondente
    rng = np.random.default_rng(7)
    # 100 amostras com prob 0.1: esperamos ~10% positivos
    y = rng.binomial(1, 0.1, 100)
    s = np.full(100, 0.1)
    ece = expected_calibration_error(y, s)
    assert ece < 0.15  # tolerância razoável para n pequeno


def test_ece_range() -> None:
    rng = np.random.default_rng(99)
    y = rng.integers(0, 2, size=300).astype(float)
    s = rng.uniform(0, 1, size=300)
    ece = expected_calibration_error(y, s)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


def _make_fairness_data(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.integers(0, 2, size=n)
    s = rng.uniform(0, 1, size=n)
    groups = pd.Series(rng.choice(["male", "female"], size=n))
    return y, s, groups


def test_fairness_report_returns_dataframe() -> None:
    y, s, groups = _make_fairness_data()
    df = fairness_report(y, s, groups)
    assert isinstance(df, pd.DataFrame)
    assert "auc" in df.columns
    assert "selection_rate" in df.columns
    assert "demographic_parity_diff" in df.columns
    assert "equal_opportunity_diff" in df.columns


def test_fairness_report_groups() -> None:
    y, s, groups = _make_fairness_data()
    df = fairness_report(y, s, groups)
    assert set(df.index) == {"male", "female"}


def test_fairness_report_selection_rate_range() -> None:
    y, s, groups = _make_fairness_data()
    df = fairness_report(y, s, groups)
    assert (df["selection_rate"] >= 0).all()
    assert (df["selection_rate"] <= 1).all()


def test_fairness_report_auc_range() -> None:
    y, s, groups = _make_fairness_data()
    df = fairness_report(y, s, groups)
    for val in df["auc"].dropna():
        assert 0.0 <= val <= 1.0
