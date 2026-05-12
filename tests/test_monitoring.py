"""Tests for PSI/CSI monitoring metrics and reference persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.monitoring.drift import (
    characteristic_stability_index,
    drift_status,
    population_stability_index,
    psi_from_bins,
    stability_report,
)
from src.monitoring.reference import load_reference, reference_exists, save_reference

# ---------------------------------------------------------------------------
# psi_from_bins — cálculo direto
# ---------------------------------------------------------------------------


def test_psi_identical_distributions_is_zero() -> None:
    """Distribuições idênticas → PSI ≈ 0."""
    counts = np.array([10, 20, 30, 25, 15], dtype=float)
    psi = psi_from_bins(counts, counts)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_very_different_distributions_is_high() -> None:
    """Distribuição completamente invertida → PSI alto."""
    ref = np.array([80, 10, 5, 3, 2], dtype=float)
    cur = np.array([2, 3, 5, 10, 80], dtype=float)
    psi = psi_from_bins(ref, cur)
    assert psi > 0.5


def test_psi_non_negative() -> None:
    rng = np.random.default_rng(0)
    ref = rng.integers(1, 100, size=10).astype(float)
    cur = rng.integers(1, 100, size=10).astype(float)
    assert psi_from_bins(ref, cur) >= 0.0


# ---------------------------------------------------------------------------
# drift_status — semáforo
# ---------------------------------------------------------------------------


def test_drift_status_stable() -> None:
    assert drift_status(0.05) == "stable"


def test_drift_status_warning() -> None:
    assert drift_status(0.15) == "warning"


def test_drift_status_drift() -> None:
    assert drift_status(0.30) == "drift"


def test_drift_status_boundary_warning() -> None:
    assert drift_status(0.10) == "warning"  # limiar inferior é exclusivo


def test_drift_status_boundary_drift() -> None:
    assert drift_status(0.25) == "drift"


def test_drift_status_custom_thresholds() -> None:
    assert drift_status(0.05, warning_threshold=0.03, critical_threshold=0.10) == "warning"


# ---------------------------------------------------------------------------
# population_stability_index — score PSI
# ---------------------------------------------------------------------------


def test_psi_identical_arrays() -> None:
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 500)
    result = population_stability_index(scores, scores)
    assert result["psi"] == pytest.approx(0.0, abs=0.01)
    assert result["status"] == "stable"


def test_psi_result_schema() -> None:
    rng = np.random.default_rng(2)
    ref = rng.uniform(0, 1, 300)
    cur = rng.uniform(0, 1, 200)
    result = population_stability_index(ref, cur)
    assert "psi" in result
    assert "status" in result
    assert "bins" in result
    assert "ref_proportions" in result
    assert "cur_proportions" in result
    assert "bin_psi" in result


def test_psi_shifted_distribution() -> None:
    """Score que migra de ~0.7 para ~0.3 deve gerar PSI alto."""
    rng = np.random.default_rng(3)
    ref = rng.normal(0.7, 0.05, 1000).clip(0, 1)
    cur = rng.normal(0.3, 0.05, 1000).clip(0, 1)
    result = population_stability_index(ref, cur, n_bins=10)
    assert result["psi"] >= 0.25
    assert result["status"] == "drift"


# ---------------------------------------------------------------------------
# characteristic_stability_index — por feature
# ---------------------------------------------------------------------------


def _make_feature_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.exponential(1.0, n),
            "f3": rng.integers(0, 5, n).astype(float),
        }
    )


def test_csi_stable_same_distribution() -> None:
    df = _make_feature_df(500, seed=7)
    csi = characteristic_stability_index(df, df, ["f1", "f2", "f3"])
    assert not csi.empty
    assert (csi["status"] == "stable").all()


def test_csi_drift_shifted_feature() -> None:
    ref = _make_feature_df(500, seed=8)
    # cur tem f1 shifted +5 — deriva óbvia
    cur = ref.copy()
    cur["f1"] = cur["f1"] + 5.0
    csi = characteristic_stability_index(ref, cur, ["f1", "f2"])
    assert csi.loc["f1", "status"] in ("warning", "drift")


def test_csi_returns_dataframe() -> None:
    ref = _make_feature_df(200)
    cur = _make_feature_df(150, seed=99)
    result = characteristic_stability_index(ref, cur, ["f1", "f2", "f3"])
    assert isinstance(result, pd.DataFrame)
    assert "csi" in result.columns
    assert "status" in result.columns


def test_csi_missing_column_skipped() -> None:
    ref = _make_feature_df(100)
    cur = _make_feature_df(100)
    # Passa coluna inexistente — não deve levantar erro
    csi = characteristic_stability_index(ref, cur, ["f1", "nonexistent"])
    assert "f1" in csi.index
    assert "nonexistent" not in csi.index


# ---------------------------------------------------------------------------
# stability_report — relatório consolidado
# ---------------------------------------------------------------------------


def test_stability_report_schema() -> None:
    rng = np.random.default_rng(5)
    ref_scores = rng.uniform(0, 1, 300)
    cur_scores = rng.uniform(0, 1, 200)
    ref_df = _make_feature_df(300, seed=5)
    cur_df = _make_feature_df(200, seed=6)
    report = stability_report(ref_scores, cur_scores, ref_df, cur_df, ["f1", "f2"])
    assert "score_psi" in report
    assert "feature_csi" in report
    assert "summary" in report
    assert "score_psi" in report["summary"]


# ---------------------------------------------------------------------------
# save_reference / load_reference
# ---------------------------------------------------------------------------


def test_save_and_load_reference_roundtrip() -> None:
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, 200)
    df = _make_feature_df(200, seed=42)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ref.json"
        saved = save_reference(scores, df, ["f1", "f2", "f3"], output_path=path)
        assert saved == path
        assert path.is_file()

        data = load_reference(path)
        assert "score_bins" in data
        assert "score_counts" in data
        assert "feature_stats" in data
        assert "f1" in data["feature_stats"]
        assert data["n_samples"] == 200


def test_reference_exists_false_when_missing() -> None:
    path = Path("/nonexistent/path/ref.json")
    assert reference_exists(path) is False


def test_load_reference_raises_when_missing() -> None:
    path = Path("/nonexistent/path/ref.json")
    with pytest.raises(FileNotFoundError):
        load_reference(path)
