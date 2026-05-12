"""DVC evaluate stage — computes metrics from the trained model and writes reports/.

Outputs consumed by DVC:
  - reports/metrics.json        DVC metrics (tracked per commit)
  - reports/calibration_curve.csv  DVC plot (fraction positive vs mean predicted prob)
  - reports/score_distribution.csv DVC plot (score histogram per class)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from src.models.evaluate import (
    auc_roc,
    brier_score,
    credit_ks_statistic,
    expected_calibration_error,
    fairness_report,
    gini_coefficient,
)
from src.models.train import _age_bins, _gender_proxy

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def run() -> None:
    """
    Load the trained model, reproduce the test split, compute all metrics, and write DVC outputs.

    Writes:
        - ``reports/metrics.json``: DVC metrics file
        - ``reports/calibration_curve.csv``: DVC plots (calibration diagram)
        - ``reports/score_distribution.csv``: DVC plots (score histogram by class)
    """
    logging.basicConfig(level=logging.INFO)
    params = _load_params()
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # Carrega features e reproduz o mesmo split do treino
    processed = PROJECT_ROOT / params["paths"]["processed_parquet"]
    meta_path = processed.parent / "feature_columns.json"
    feature_names: list[str] = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(processed)
    X = df[feature_names]
    y = df["credit_risk"].astype(int).to_numpy()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y,
    )
    test_idx = X_test.index

    # Carrega modelo treinado
    model_path = PROJECT_ROOT / params["paths"]["model_dir"] / "baseline.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Execute `make train` primeiro."
        )
    bundle = joblib.load(model_path)  # nosec B301
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(X_test)[:, 1]

    # ---- Métricas de crédito + calibração ----
    threshold = params.get("fairness", {}).get("threshold", 0.5)
    y_pred = (proba >= threshold).astype(int)

    ks = credit_ks_statistic(y_test, proba)
    gini = gini_coefficient(y_test, proba)
    auc = auc_roc(y_test, proba)
    bs = brier_score(y_test, proba)
    ece = expected_calibration_error(y_test, proba)

    # Contagens de classificação
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ---- Fairness (resumo) ----
    interim = PROJECT_ROOT / params["paths"]["interim_parquet"]
    fairness_metrics: dict[str, float] = {}
    if interim.exists():
        df_int = pd.read_parquet(interim)
        fairness_cfg = params.get("fairness", {})

        if "personal_status_sex" in df_int.columns:
            gender = _gender_proxy(df_int.loc[test_idx, "personal_status_sex"])
            fr = fairness_report(y_test, proba, gender, threshold=threshold)
            for grp, row in fr.iterrows():
                key = f"fairness_gender_{str(grp).replace(' ', '_')}"
                auc_val = float(row["auc"]) if not np.isnan(row["auc"]) else 0.0
                fairness_metrics[f"{key}_auc"] = auc_val
                fairness_metrics[f"{key}_dp_diff"] = float(row["demographic_parity_diff"])

        if "age" in df_int.columns:
            age_bins = fairness_cfg.get("age_bins", [18, 30, 45, 60, 120])
            age_labels = fairness_cfg.get("age_labels", ["18-30", "31-45", "46-60", "60+"])
            age_grp = _age_bins(df_int.loc[test_idx, "age"], age_bins, age_labels)
            fr_age = fairness_report(y_test, proba, age_grp, threshold=threshold)
            max_dp = float(fr_age["demographic_parity_diff"].abs().max())
            fairness_metrics["fairness_age_max_dp_diff"] = max_dp

    # ---- Escreve metrics.json (DVC metrics) ----
    metrics: dict[str, float] = {
        "ks_statistic": round(ks, 6),
        "gini": round(gini, 6),
        "auc_roc": round(auc, 6),
        "brier_score": round(bs, 6),
        "ece": round(ece, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        **{k: round(v, 6) for k, v in fairness_metrics.items()},
    }
    (reports / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("metrics.json salvo em reports/")

    # ---- Curva de calibração (DVC plot) ----
    prob_true, prob_pred_bins = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
    cal_df = pd.DataFrame({"mean_predicted": prob_pred_bins, "fraction_positive": prob_true})
    cal_df.to_csv(reports / "calibration_curve.csv", index=False)
    logger.info("calibration_curve.csv salvo.")

    # ---- Distribuição de scores por classe (DVC plot) ----
    score_df = pd.DataFrame({"score": proba, "label": y_test})
    n_bins_hist = 20
    hist_rows = []
    for label_val, label_name in [(0, "mau"), (1, "bom")]:
        scores_cls = score_df.loc[score_df["label"] == label_val, "score"]
        counts, edges = np.histogram(scores_cls, bins=n_bins_hist, range=(0, 1), density=True)
        for i, count in enumerate(counts):
            hist_rows.append(
                {
                    "bin_center": round(float((edges[i] + edges[i + 1]) / 2), 3),
                    "density": round(float(count), 4),
                    "class": label_name,
                }
            )
    pd.DataFrame(hist_rows).to_csv(reports / "score_distribution.csv", index=False)
    logger.info("score_distribution.csv salvo.")

    # ---- Referência de monitoramento (PSI/CSI baseline) ----
    try:
        from src.monitoring.reference import save_reference

        save_reference(
            scores=proba,
            feature_df=X_test,
            feature_cols=feature_names,
            model_version=params.get("registry", {}).get("model_name", "unknown"),
        )
        logger.info("monitoring_reference.json salvo para baseline PSI/CSI.")
    except Exception as exc:
        logger.warning("Não foi possível salvar referência de monitoramento: %s", exc)

    # Log resumido no terminal
    logger.info("─" * 50)
    logger.info("  KS=%.4f  Gini=%.4f  AUC=%.4f", ks, gini, auc)
    logger.info("  Brier=%.4f  ECE=%.4f", bs, ece)
    logger.info("  F1=%.4f  Precision=%.4f  Recall=%.4f", f1, precision, recall)
    if fairness_metrics:
        for k, v in fairness_metrics.items():
            logger.info("  %s=%.4f", k, v)
    logger.info("─" * 50)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
