"""Tests for registry helpers — unit tests that do not require a live MLflow server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from src.models.registry import (
    _passes_threshold,
    compare_runs,
    get_latest_run_id,
    list_versions,
)

# ---------------------------------------------------------------------------
# _passes_threshold — lógica de quality gate (sem MLflow server)
# ---------------------------------------------------------------------------


def _make_mock_run(metrics: dict[str, float]) -> MagicMock:
    """Cria um mock de mlflow.entities.Run com métricas customizadas."""
    run = MagicMock()
    run.data.metrics = metrics
    return run


def _patch_client(mock_run: MagicMock):
    """Contexto que injeta um MlflowClient falso."""
    client = MagicMock()
    client.get_run.return_value = mock_run
    return patch("src.models.registry._client", return_value=client)


def test_passes_threshold_all_pass() -> None:
    run = _make_mock_run({"ks_statistic": 0.40, "gini": 0.50, "auc_roc": 0.80, "brier_score": 0.20})
    thresholds = {"ks_statistic": 0.25, "gini": 0.30, "auc_roc": 0.70, "brier_score": 0.25}
    with _patch_client(run):
        assert _passes_threshold("fake_run_id", thresholds) is True


def test_passes_threshold_ks_fails() -> None:
    run = _make_mock_run({"ks_statistic": 0.15, "gini": 0.50, "auc_roc": 0.80, "brier_score": 0.20})
    thresholds = {"ks_statistic": 0.25, "gini": 0.30, "auc_roc": 0.70, "brier_score": 0.25}
    with _patch_client(run):
        assert _passes_threshold("fake_run_id", thresholds) is False


def test_passes_threshold_brier_fails() -> None:
    """Brier score: limiar é um teto (menor = melhor). Reprovado se acima do limiar."""
    run = _make_mock_run({"ks_statistic": 0.40, "gini": 0.50, "auc_roc": 0.80, "brier_score": 0.30})
    thresholds = {"brier_score": 0.25}  # brier > 0.25 deve reprovar
    with _patch_client(run):
        assert _passes_threshold("fake_run_id", thresholds) is False


def test_passes_threshold_missing_metric_ignored() -> None:
    """Métricas ausentes no run são ignoradas (gate não bloqueia)."""
    run = _make_mock_run({})  # sem nenhuma métrica
    thresholds = {"ks_statistic": 0.25}
    with _patch_client(run):
        # gate ignorado quando métrica não existe no run
        assert _passes_threshold("fake_run_id", thresholds) is True


def test_passes_threshold_empty_thresholds() -> None:
    run = _make_mock_run({"ks_statistic": 0.10})
    with _patch_client(run):
        assert _passes_threshold("fake_run_id", {}) is True


# ---------------------------------------------------------------------------
# list_versions — retorna DataFrame mesmo sem versões
# ---------------------------------------------------------------------------


def test_list_versions_returns_dataframe() -> None:
    client = MagicMock()
    client.search_model_versions.return_value = []
    with patch("src.models.registry._client", return_value=client):
        df = list_versions(model_name="test_model")
    assert isinstance(df, pd.DataFrame)


def test_list_versions_schema() -> None:
    """Colunas esperadas presentes mesmo quando há versões."""
    mv = MagicMock()
    mv.version = "1"
    mv.current_stage = "Staging"
    mv.run_id = "abc123" * 4
    mv.creation_timestamp = 0

    run = _make_mock_run(
        {"ks_statistic": 0.40, "gini": 0.50, "auc_roc": 0.80, "brier_score": 0.20, "ece": 0.05}
    )

    client = MagicMock()
    client.search_model_versions.return_value = [mv]
    client.get_run.return_value = run

    with patch("src.models.registry._client", return_value=client):
        df = list_versions(model_name="test_model")

    assert "version" in df.columns
    assert "stage" in df.columns
    assert "ks_statistic" in df.columns
    assert "auc_roc" in df.columns


# ---------------------------------------------------------------------------
# compare_runs — retorna DataFrame vazio se experimento não existir
# ---------------------------------------------------------------------------


def test_compare_runs_no_experiment() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    with patch("src.models.registry._client", return_value=client):
        df = compare_runs(experiment_name="inexistente")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ---------------------------------------------------------------------------
# get_latest_run_id — retorna None se não há runs
# ---------------------------------------------------------------------------


def test_get_latest_run_id_no_runs() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = []
    with patch("src.models.registry._client", return_value=client):
        result = get_latest_run_id(experiment_name="test")
    assert result is None


def test_get_latest_run_id_returns_run_id() -> None:
    run = MagicMock()
    run.info.run_id = "abc123"
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    client.search_runs.return_value = [run]
    with patch("src.models.registry._client", return_value=client):
        result = get_latest_run_id(experiment_name="test")
    assert result == "abc123"
