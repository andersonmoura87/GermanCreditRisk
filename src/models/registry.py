"""MLflow Model Registry — ciclo de vida do modelo (register, promote, compare, load)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _client() -> MlflowClient:
    return MlflowClient()


# ---------------------------------------------------------------------------
# Registro e promoção
# ---------------------------------------------------------------------------


def register_run(
    run_id: str,
    *,
    model_name: str | None = None,
    artifact_path: str = "model",
) -> Any:
    """
    Register a trained MLflow run in the Model Registry under ``model_name``.

    Args:
        run_id: MLflow run ID to register.
        model_name: Registry model name (defaults to ``params.yaml``).
        artifact_path: Sub-path inside the run where the model artifact lives.

    Returns:
        ``mlflow.entities.model_registry.ModelVersion`` object.
    """
    params = _load_params()
    name = model_name or params["registry"]["model_name"]
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, name)
    logger.info("Modelo registrado: %s versão %s (run_id=%s)", name, mv.version, run_id)
    return mv


def _passes_threshold(run_id: str, thresholds: dict[str, float]) -> bool:
    """Verifica se as métricas de um run passam nos limiares definidos em params.yaml."""
    client = _client()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    for metric, threshold in thresholds.items():
        val = metrics.get(metric)
        if val is None:
            logger.warning("Métrica '%s' não encontrada no run %s — gate ignorado.", metric, run_id)
            continue
        # Brier score: menor é melhor; demais: maior é melhor
        if metric == "brier_score":
            if val > threshold:
                logger.info("Gate reprovado: %s=%.4f > limiar=%.4f", metric, val, threshold)
                return False
        else:
            if val < threshold:
                logger.info("Gate reprovado: %s=%.4f < limiar=%.4f", metric, val, threshold)
                return False
    return True


def promote_to_staging(
    version: str | int,
    *,
    model_name: str | None = None,
    check_threshold: bool = True,
) -> None:
    """
    Transition a registered model version to the ``Staging`` stage.

    Args:
        version: Model version number (int or string).
        model_name: Registry model name (defaults to ``params.yaml``).
        check_threshold: If True, validate quality gates before promoting.

    Raises:
        ValueError: If quality gates are not met and ``check_threshold`` is True.
    """
    params = _load_params()
    name = model_name or params["registry"]["model_name"]
    client = _client()

    if check_threshold:
        mv = client.get_model_version(name, str(version))
        thresholds = params.get("registry", {}).get("staging_threshold", {})
        if not _passes_threshold(mv.run_id, thresholds):
            raise ValueError(
                f"Versão {version} não passa nos quality gates. Verifique as métricas."
            )

    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info("Modelo %s v%s → Staging", name, version)


def promote_to_production(
    version: str | int,
    *,
    model_name: str | None = None,
    archive_existing: bool = True,
) -> None:
    """
    Transition a model version to ``Production``, optionally archiving existing Production models.

    Args:
        version: Model version number to promote.
        model_name: Registry model name (defaults to ``params.yaml``).
        archive_existing: If True, archive all currently-Production versions.
    """
    params = _load_params()
    name = model_name or params["registry"]["model_name"]
    client = _client()

    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage="Production",
        archive_existing_versions=archive_existing,
    )
    logger.info(
        "Modelo %s v%s → Production%s",
        name,
        version,
        " (versões anteriores arquivadas)" if archive_existing else "",
    )


def archive_version(
    version: str | int,
    *,
    model_name: str | None = None,
) -> None:
    """Move a model version to the ``Archived`` stage."""
    params = _load_params()
    name = model_name or params["registry"]["model_name"]
    _client().transition_model_version_stage(
        name=name,
        version=str(version),
        stage="Archived",
    )
    logger.info("Modelo %s v%s → Archived", name, version)


# ---------------------------------------------------------------------------
# Consulta e comparação
# ---------------------------------------------------------------------------


def list_versions(
    *,
    model_name: str | None = None,
    stage: str | None = None,
) -> pd.DataFrame:
    """
    List all registered versions for a model, optionally filtered by stage.

    Args:
        model_name: Registry model name (defaults to ``params.yaml``).
        stage: Optional stage filter (``'Staging'``, ``'Production'``, ``'Archived'``).

    Returns:
        DataFrame with version metadata and key run metrics.
    """
    params = _load_params()
    name = model_name or params["registry"]["model_name"]
    client = _client()

    filter_str = f"name='{name}'"
    versions = client.search_model_versions(filter_str)
    rows = []
    for mv in versions:
        try:
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics
        except Exception:
            metrics = {}
        rows.append(
            {
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id[:8],
                "created": mv.creation_timestamp,
                "ks_statistic": metrics.get("ks_statistic"),
                "gini": metrics.get("gini"),
                "auc_roc": metrics.get("auc_roc"),
                "brier_score": metrics.get("brier_score"),
                "ece": metrics.get("ece"),
            }
        )

    df = pd.DataFrame(rows)
    if stage and not df.empty:
        df = df[df["stage"] == stage]
    if stage and not df.empty:
        df = df.sort_values("version", ascending=False)
    return df.reset_index(drop=True)


def compare_runs(
    experiment_name: str | None = None,
    *,
    top_n: int = 10,
    sort_by: str = "metrics.ks_statistic",
) -> pd.DataFrame:
    """
    Compare MLflow runs in an experiment, returning a sorted metrics table.

    Args:
        experiment_name: MLflow experiment name (defaults to ``params.yaml``).
        top_n: Number of top runs to return.
        sort_by: Run property to sort by (e.g. ``'metrics.auc_roc'``).

    Returns:
        DataFrame with run_id, params, and key metrics.
    """
    params = _load_params()
    exp_name = experiment_name or params["mlflow"]["experiment_name"]
    client = _client()

    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        logger.warning("Experimento '%s' não encontrado.", exp_name)
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"{sort_by} DESC"],
        max_results=top_n,
    )

    rows = []
    for r in runs:
        row: dict[str, Any] = {
            "run_id": r.info.run_id[:8],
            "run_name": r.info.run_name,
            "status": r.info.status,
        }
        row.update(
            {
                f"p_{k}": v
                for k, v in r.data.params.items()
                if k in ("model", "calibration_method", "n_features")
            }
        )
        row.update(
            {
                k: round(v, 4)
                for k, v in r.data.metrics.items()
                if k in ("ks_statistic", "gini", "auc_roc", "brier_score", "ece")
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Carregamento do modelo em produção
# ---------------------------------------------------------------------------


def load_production_model(
    *,
    model_name: str | None = None,
) -> Any:
    """
    Load the current ``Production`` model from the MLflow Model Registry.

    Args:
        model_name: Registry model name (defaults to ``params.yaml``).

    Returns:
        MLflow ``pyfunc`` model with a ``.predict()`` method.

    Raises:
        mlflow.exceptions.MlflowException: If no Production version exists.
    """
    params = _load_params()
    name = model_name or params["registry"]["model_name"]
    model_uri = f"models:/{name}/Production"
    logger.info("Carregando modelo Production: %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def get_latest_run_id(experiment_name: str | None = None) -> str | None:
    """
    Return the run_id of the most recent successful run in an experiment.

    Args:
        experiment_name: MLflow experiment name (defaults to ``params.yaml``).

    Returns:
        Run ID string, or None if no runs found.
    """
    params = _load_params()
    exp_name = experiment_name or params["mlflow"]["experiment_name"]
    client = _client()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


# ---------------------------------------------------------------------------
# CLI rápido: python -m src.models.registry
# ---------------------------------------------------------------------------


def _cli() -> None:
    """CLI mínima: lista versões e últimos runs."""
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="MLflow Model Registry CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="Lista versões registradas")
    p_promote = sub.add_parser("promote", help="Promove versão para Staging ou Production")
    p_promote.add_argument("version", type=int)
    p_promote.add_argument("--stage", choices=["staging", "production"], default="staging")
    p_promote.add_argument("--no-check", action="store_true")
    sub.add_parser("compare", help="Compara runs do experimento")
    p_reg = sub.add_parser("register", help="Registra o último run no Model Registry")
    p_reg.add_argument("--run-id", default=None)

    args = parser.parse_args()

    if args.cmd == "list":
        df = list_versions()
        print(df.to_string(index=False) if not df.empty else "Nenhuma versão registrada.")

    elif args.cmd == "promote":
        if args.stage == "staging":
            promote_to_staging(args.version, check_threshold=not args.no_check)
        else:
            promote_to_production(args.version)

    elif args.cmd == "compare":
        df = compare_runs()
        print(df.to_string(index=False) if not df.empty else "Nenhum run encontrado.")

    elif args.cmd == "register":
        run_id = args.run_id or get_latest_run_id()
        if not run_id:
            print("Nenhum run encontrado. Execute `make train` primeiro.")
            return
        mv = register_run(run_id)
        print(f"Registrado: versão {mv.version}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
