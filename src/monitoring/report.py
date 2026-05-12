"""Monitoring report generator — PSI/CSI JSON + standalone HTML.

Entry point for the DVC ``monitor`` stage and ``make monitor``.

Usage:
    python -m src.monitoring.report [--current-parquet PATH]

The script:
1. Loads the trained model and reproduces the reference split (evaluate split).
2. Loads an optional "current" window parquet (simulating production data).
   If not provided, uses the last 20% of the processed dataset as the "current" window.
3. Computes PSI (score drift) and CSI (feature drift).
4. Writes ``reports/monitoring_report.json`` (DVC metrics) and
   ``reports/monitoring_report.html`` (human-readable dashboard).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.monitoring.drift import characteristic_stability_index, population_stability_index

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# HTML report — standalone, sem dependências front-end
# ---------------------------------------------------------------------------

_STATUS_EMOJI = {"stable": "🟢", "warning": "🟡", "drift": "🔴"}
_STATUS_COLOR = {"stable": "#639922", "warning": "#f5a623", "drift": "#E24B4A"}


def _html_report(metrics: dict, csi_df: pd.DataFrame | None) -> str:
    """Generate a self-contained HTML monitoring dashboard."""
    ts = metrics.get("generated_at", "—")
    psi = metrics.get("score_psi", 0.0)
    psi_status = metrics.get("score_psi_status", "unknown")
    psi_color = _STATUS_COLOR.get(psi_status, "#999")
    n_warn = metrics.get("n_features_warning", 0)
    n_drift = metrics.get("n_features_drift", 0)

    # CSI table rows
    csi_rows = ""
    if csi_df is not None and not csi_df.empty:
        for feat, row in csi_df.iterrows():
            st = row.get("status", "unknown")
            color = _STATUS_COLOR.get(st, "#999")
            emoji = _STATUS_EMOJI.get(st, "⚪")
            csi_rows += (
                f"<tr>"
                f"<td>{feat}</td>"
                f"<td style='color:{color};font-weight:bold'>{emoji} {st}</td>"
                f"<td>{row.get('csi', 0.0):.4f}</td>"
                f"<td>{row.get('n_reference', '—')}</td>"
                f"<td>{row.get('n_current', '—')}</td>"
                f"</tr>\n"
            )
    else:
        csi_rows = "<tr><td colspan='5'>Sem dados de CSI disponíveis.</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Monitoring Report — German Credit Risk</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1   {{ border-bottom: 3px solid {psi_color}; padding-bottom: 8px; }}
  .card {{ background: #f9f9f9; border-radius: 8px; padding: 16px 24px; margin: 16px 0; }}
  .big  {{ font-size: 2.2em; font-weight: bold; color: {psi_color}; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }}
  th    {{ background: #f0f0f0; }}
  .stable  {{ color: #639922; }} .warning {{ color: #f5a623; }} .drift {{ color: #E24B4A; }}
  footer   {{ margin-top: 40px; color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>📊 Monitoring Report — German Credit Risk</h1>
<p>Gerado em: <strong>{ts}</strong></p>

<div class="card">
  <h2>Score PSI (Population Stability Index)</h2>
  <p class="big">{psi:.4f} &nbsp; {_STATUS_EMOJI.get(psi_status, '⚪')} {psi_status.upper()}</p>
  <p>Referência: {metrics.get('n_reference', '—')} amostras &nbsp;|&nbsp;
     Atual: {metrics.get('n_current', '—')} amostras</p>
  <p><small>Limiares: &lt; 0.10 = estável · 0.10–0.25 = atenção · ≥ 0.25 = deriva</small></p>
</div>

<div class="card">
  <h2>Resumo CSI (Characteristic Stability Index)</h2>
  <p>Features em <span class='warning'>atenção</span>: <strong>{n_warn}</strong> &nbsp;|&nbsp;
     Features em <span class='drift'>deriva</span>: <strong>{n_drift}</strong></p>
  <table>
    <thead>
      <tr><th>Feature</th><th>Status</th><th>CSI</th><th>N referência</th><th>N atual</th></tr>
    </thead>
    <tbody>
      {csi_rows}
    </tbody>
  </table>
</div>

<footer>
  <p>German Credit Risk · PSI/CSI alinhados a BACEN Res. 4.557/2017 ·
     <a href="http://localhost:8000/v1/monitoring">API endpoint</a></p>
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------


def run(current_parquet: Path | None = None) -> dict:
    """
    Compute PSI/CSI, save JSON metrics and HTML dashboard.

    Args:
        current_parquet: Optional path to a "current window" parquet.
            If None, uses the last ``split.test_size`` fraction of the processed dataset
            (simulates production data when no real window is available).

    Returns:
        Metrics dict written to ``reports/monitoring_report.json``.
    """
    logging.basicConfig(level=logging.INFO)
    params = _load_params()
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Carrega dados processados e reproduz o split de treino/teste
    processed = PROJECT_ROOT / params["paths"]["processed_parquet"]
    meta_path = processed.parent / "feature_columns.json"
    feature_names: list[str] = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(processed)

    X = df[feature_names]
    y = df["credit_risk"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
        stratify=y,
    )

    # Carrega modelo e obtém scores de referência (test split)
    model_path = PROJECT_ROOT / params["paths"]["model_dir"] / "baseline.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Execute `make train` primeiro."
        )
    bundle = joblib.load(model_path)  # nosec B301
    pipe = bundle["pipeline"]

    ref_scores = pipe.predict_proba(X_test)[:, 1]
    ref_df = X_test.copy()

    # Janela "atual" — pode ser dados reais de produção ou simulação com X_train
    if current_parquet and Path(current_parquet).is_file():
        df_cur = pd.read_parquet(current_parquet)
        cur_df = df_cur[[c for c in feature_names if c in df_cur.columns]]
        cur_scores = pipe.predict_proba(cur_df)[:, 1]
        logger.info("Janela atual carregada de %s (%d linhas).", current_parquet, len(cur_df))
    else:
        # Fallback: usa X_train como janela "atual" (demonstração sem dados reais)
        logger.info("Nenhuma janela atual fornecida — usando X_train como proxy de produção.")
        cur_df = X_train.copy()
        cur_scores = pipe.predict_proba(cur_df)[:, 1]

    # PSI no score
    psi_result = population_stability_index(ref_scores, cur_scores)

    # CSI por feature
    csi_df = characteristic_stability_index(ref_df, cur_df, feature_names)

    n_warn = int((csi_df["status"] == "warning").sum()) if not csi_df.empty else 0
    n_drift = int((csi_df["status"] == "drift").sum()) if not csi_df.empty else 0

    ts = datetime.now(tz=UTC).isoformat()

    # Métricas para DVC (e API)
    metrics: dict = {
        "generated_at": ts,
        "score_psi": round(psi_result["psi"], 6),
        "score_psi_status": psi_result["status"],
        "n_reference": len(ref_scores),
        "n_current": len(cur_scores),
        "n_features_evaluated": len(csi_df),
        "n_features_warning": n_warn,
        "n_features_drift": n_drift,
    }

    # Top-5 features com maior CSI
    if not csi_df.empty:
        top5 = csi_df.nlargest(5, "csi")
        for feat, row in top5.iterrows():
            safe = str(feat).replace(" ", "_")
            metrics[f"csi_{safe}"] = round(float(row["csi"]), 6)

    # Escreve JSON
    json_path = reports_dir / "monitoring_report.json"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("monitoring_report.json salvo.")

    # Escreve HTML
    html_path = reports_dir / "monitoring_report.html"
    html_path.write_text(_html_report(metrics, csi_df), encoding="utf-8")
    logger.info("monitoring_report.html salvo em %s", html_path)

    # Log resumido
    logger.info("─" * 50)
    logger.info("  PSI=%.4f (%s)", psi_result["psi"], psi_result["status"])
    logger.info("  CSI: %d atenção, %d deriva", n_warn, n_drift)
    if not csi_df.empty:
        top = csi_df.head(3)
        for feat, row in top.iterrows():
            logger.info("    %-30s CSI=%.4f (%s)", feat, row["csi"], row["status"])
    logger.info("─" * 50)

    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gerar relatório de monitoramento PSI/CSI.")
    parser.add_argument("--current-parquet", default=None, help="Parquet da janela atual.")
    args = parser.parse_args()
    run(current_parquet=Path(args.current_parquet) if args.current_parquet else None)


if __name__ == "__main__":
    main()
