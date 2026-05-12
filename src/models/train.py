"""Train a baseline sklearn pipeline with Platt scaling, fairness logging e MLflow tracking."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.models
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import (
    auc_roc,
    brier_score,
    credit_ks_statistic,
    expected_calibration_error,
    fairness_report,
    gini_coefficient,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fingerprint_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _gender_proxy(series: pd.Series) -> pd.Series:
    """
    Map ``personal_status_sex`` (1–5) to a binary gender proxy for fairness analysis.

    GCR encoding: 1=male divorced, 2=female divorced/married, 3=male single,
    4=male married, 5=female single.

    Returns:
        Series with values ``"male"`` or ``"female"``.
    """
    mapping = {1: "male", 2: "female", 3: "male", 4: "male", 5: "female"}
    return series.map(mapping).fillna("unknown")


def _age_bins(series: pd.Series, bins: list[int], labels: list[str]) -> pd.Series:
    """Discretize age into labeled buckets for group fairness analysis."""
    return pd.cut(series, bins=bins, labels=labels, right=True).astype(str)


def build_base_pipeline(feature_names: list[str]) -> Pipeline:
    """
    Create a standardized logistic regression pipeline (uncalibrated).

    Args:
        feature_names: Ordered training columns (excluding target).

    Returns:
        Unfitted sklearn ``Pipeline``.
    """
    pre = ColumnTransformer(
        [("num", StandardScaler(), feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("prep", pre), ("clf", clf)])


# Mantido para compatibilidade retroativa com predict.py
build_pipeline = build_base_pipeline


def build_calibrated_pipeline(
    feature_names: list[str],
    method: str = "sigmoid",
    cv: int = 5,
) -> Pipeline:
    """
    Wrap the base logistic regression pipeline with Platt scaling (sigmoid) or isotonic calibration.

    Args:
        feature_names: Ordered training columns (excluding target).
        method: ``'sigmoid'`` (Platt scaling) or ``'isotonic'``.
        cv: Cross-validation folds for calibration.

    Returns:
        Unfitted calibrated ``Pipeline``.
    """
    base = build_base_pipeline(feature_names)
    calibrated = CalibratedClassifierCV(base, method=method, cv=cv)
    # Embrulha em pipeline para manter interface uniforme de fit/predict_proba
    return Pipeline([("calibrated_clf", calibrated)])


def _log_fairness_to_mlflow(report_df: pd.DataFrame, prefix: str) -> None:
    """Flatten fairness DataFrame e loga cada métrica como parâmetro MLflow."""
    for group, row in report_df.iterrows():
        safe_group = (
            str(group).replace(" ", "_").replace("-", "_").replace("+", "plus").replace("/", "_")
        )
        cols = ["auc", "f1", "selection_rate", "demographic_parity_diff", "equal_opportunity_diff"]
        for col in cols:
            val = row[col]
            if not (isinstance(val, float) and np.isnan(val)):
                mlflow.log_metric(f"{prefix}_{safe_group}_{col}", float(val))


def run() -> Path:
    """
    Train calibrated model, persist joblib artifact, log MLflow metrics.

    Returns:
        Path to saved joblib model.
    """
    logging.basicConfig(level=logging.INFO)
    params = _load_params()

    # Carrega features processadas
    processed = PROJECT_ROOT / params["paths"]["processed_parquet"]
    meta_path = processed.parent / "feature_columns.json"
    feature_names: list[str] = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(processed)

    # Carrega interim para atributos sensíveis (valores originais antes do WoE)
    interim = PROJECT_ROOT / params["paths"]["interim_parquet"]
    df_interim = pd.read_parquet(interim) if interim.exists() else None

    X = df[feature_names]
    y = df["credit_risk"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y,
    )

    # Índices do conjunto de teste para alinhar atributos sensíveis
    test_idx = X_test.index

    # Treina pipeline calibrado (Platt scaling por padrão)
    cal_method = params.get("calibration", {}).get("method", "sigmoid")
    cal_cv = params.get("calibration", {}).get("cv", 5)
    pipe = build_calibrated_pipeline(feature_names, method=cal_method, cv=cal_cv)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    # Métricas de crédito (ranking)
    ks = credit_ks_statistic(y_test, proba)
    gini = gini_coefficient(y_test, proba)
    auc = auc_roc(y_test, proba)

    # Métricas de calibração
    bs = brier_score(y_test, proba)
    ece = expected_calibration_error(y_test, proba)

    # Relatório de classificação (threshold 0.5)
    y_pred = (proba >= params.get("fairness", {}).get("threshold", 0.5)).astype(int)
    clf_report_str = classification_report(y_test, y_pred, target_names=["mau", "bom"])

    model_dir = PROJECT_ROOT / params["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline.joblib"
    joblib.dump({"pipeline": pipe, "feature_names": feature_names}, model_path)

    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    dataset_fp = _fingerprint_path(processed)

    with mlflow.start_run(run_name=f"logistic_calibrated_{cal_method}_v1"):
        # Parâmetros
        mlflow.log_params(
            {
                "model": params["model"]["baseline"],
                "calibration_method": cal_method,
                "calibration_cv": cal_cv,
                "split_test_size": params["split"]["test_size"],
                "random_state": params["split"]["random_state"],
                "dataset_fingerprint": dataset_fp,
                "n_features": len(feature_names),
            }
        )

        # Métricas de crédito + calibração
        mlflow.log_metrics(
            {
                "ks_statistic": ks,
                "gini": gini,
                "auc_roc": auc,
                "brier_score": bs,
                "ece": ece,
            }
        )

        # Relatório de classificação como artefato
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "classification_report.txt"
        report_path.write_text(clf_report_str, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        # Fairness — atributos sensíveis do interim (antes do WoE)
        if df_interim is not None:
            fairness_cfg = params.get("fairness", {})
            threshold = fairness_cfg.get("threshold", 0.5)

            # Proxy de gênero (personal_status_sex → male/female)
            if "personal_status_sex" in df_interim.columns:
                gender_test = _gender_proxy(df_interim.loc[test_idx, "personal_status_sex"])
                gender_fr = fairness_report(y_test, proba, gender_test, threshold=threshold)
                _log_fairness_to_mlflow(gender_fr, "fairness_gender")
                logger.info("Fairness por gênero:\n%s", gender_fr.to_string())

            # Faixa etária
            if "age" in df_interim.columns:
                age_bins = fairness_cfg.get("age_bins", [18, 30, 45, 60, 120])
                age_labels = fairness_cfg.get("age_labels", ["18-30", "31-45", "46-60", "60+"])
                age_test = _age_bins(df_interim.loc[test_idx, "age"], age_bins, age_labels)
                age_fr = fairness_report(y_test, proba, age_test, threshold=threshold)
                _log_fairness_to_mlflow(age_fr, "fairness_age")
                logger.info("Fairness por faixa etária:\n%s", age_fr.to_string())

        # Assinatura do modelo — esquema de input/output para MLflow Registry
        signature = mlflow.models.infer_signature(X_test, proba)
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:3],
        )

        # Auto-registro no Model Registry (se habilitado em params.yaml)
        reg_cfg = params.get("registry", {})
        if reg_cfg.get("auto_register", False):
            active = mlflow.active_run()
            if active is not None:
                _auto_register(run_id=active.info.run_id, params=params)

    logger.info(
        "Treino concluído — KS=%.4f Gini=%.4f AUC=%.4f Brier=%.4f ECE=%.4f",
        ks,
        gini,
        auc,
        bs,
        ece,
    )
    return model_path


def _auto_register(run_id: str, params: dict) -> None:
    """Registra o run no Model Registry se as métricas passarem no quality gate."""
    try:
        from src.models.registry import _passes_threshold, register_run

        reg_cfg = params.get("registry", {})
        thresholds = reg_cfg.get("staging_threshold", {})
        model_name = reg_cfg.get("model_name", "german_credit_risk")

        if thresholds and not _passes_threshold(run_id, thresholds):
            logger.warning(
                "Auto-registro cancelado: métricas abaixo do quality gate definido em params.yaml."
            )
            return

        mv = register_run(run_id, model_name=model_name)
        logger.info("Auto-registrado no Registry: %s versão %s", model_name, mv.version)
    except Exception as exc:
        logger.warning("Auto-registro falhou (não crítico): %s", exc)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
