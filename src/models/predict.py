"""Batch or single-row scoring — via joblib artifact ou MLflow Model Registry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.iv_woe import apply_value_to_woe

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_model_path() -> Path:
    params = _load_params()
    return PROJECT_ROOT / params["paths"]["model_dir"] / "baseline.joblib"


def _maybe_apply_woe_maps(df: pd.DataFrame) -> pd.DataFrame:
    """If WoE maps exist (treino com ``iv_woe.encoding=woe``), converte valores brutos para WoE."""
    params = _load_params()
    rel = params["paths"].get("woe_value_maps", "data/processed/woe_value_maps.joblib")
    maps_path = PROJECT_ROOT / rel
    if not maps_path.is_file():
        return df
    bundle: dict[str, Any] = joblib.load(maps_path)  # nosec B301
    maps: dict[str, dict[str, float]] = bundle.get("maps") or {}
    if not maps:
        return df
    return apply_value_to_woe(df, maps)


def load_artifact(path: Path | None = None) -> dict[str, Any]:
    """
    Load the joblib bundle produced by ``train.run``.

    Args:
        path: Optional path to ``baseline.joblib``.

    Returns:
        Dict with keys ``pipeline`` and ``feature_names``.
    """
    p = path or _default_model_path()
    if not p.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {p}")
    return joblib.load(p)  # nosec B301


def load_from_registry(
    stage: str = "Production",
    *,
    model_name: str | None = None,
) -> Any:
    """
    Load a model directly from the MLflow Model Registry.

    Preferred over ``load_artifact`` in staging/production environments where the
    registry is the source of truth for promoted models.

    Args:
        stage: Registry stage to load (``'Production'``, ``'Staging'``).
        model_name: Registry model name (defaults to ``params.yaml``).

    Returns:
        MLflow ``pyfunc`` model exposing ``.predict(DataFrame)`` → probability array.

    Raises:
        mlflow.exceptions.MlflowException: If no model at the requested stage exists.
    """
    import mlflow

    params = _load_params()
    name = model_name or params["registry"]["model_name"]
    model_uri = f"models:/{name}/{stage}"
    logger.info("Carregando modelo do Registry: %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def predict_proba(
    df: pd.DataFrame,
    path: Path | None = None,
    *,
    use_registry: bool = False,
    registry_stage: str = "Production",
) -> np.ndarray:
    """
    Predict probability of class 1 (bom pagador) for a feature dataframe.

    Supports two loading strategies:
    - **Joblib** (default): loads ``models/baseline.joblib`` — suitable for local/offline use.
    - **Registry** (``use_registry=True``): loads from MLflow Registry — suitable for serving.

    Args:
        df: Rows with the same columns used in training.
        path: Optional model path (joblib strategy only).
        use_registry: If True, load from MLflow Model Registry instead of joblib.
        registry_stage: Stage to load when ``use_registry=True``.

    Returns:
        1d array of probabilities.
    """
    if use_registry:
        model = load_from_registry(stage=registry_stage)
        # MLflow pyfunc retorna um DataFrame; extraímos a coluna de probabilidade
        result = model.predict(df)
        if isinstance(result, pd.DataFrame):
            return result.iloc[:, 0].to_numpy()
        return np.asarray(result)

    bundle = load_artifact(path)
    pipe: Any = bundle["pipeline"]
    feats: list[str] = bundle["feature_names"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no input: {missing}")
    x = _maybe_apply_woe_maps(df)
    return pipe.predict_proba(x[feats])[:, 1]
