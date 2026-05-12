"""FastAPI application for online credit scoring — German Credit Risk."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from src.models.predict import load_artifact, predict_proba
from src.monitoring.reference import reference_exists

from api.schemas import (
    BatchScoringRequest,
    BatchScoringResponse,
    CreditRequest,
    CreditResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Limiares de decisão (assimetria de custo 5:1 — falso negativo de negócio)
_THRESHOLD_APPROVE = 0.65  # prob_good >= 0.65 → aprovado
_THRESHOLD_REVIEW = 0.40  # 0.40 <= prob_good < 0.65 → em análise
# prob_good < 0.40 → negado


def _decision(
    prob_good: float,
) -> tuple[Literal["aprovado", "em_analise", "negado"], Literal["baixo", "medio", "alto"]]:
    """Retorna (decision, risk_category) baseado na probabilidade de adimplência."""
    if prob_good >= _THRESHOLD_APPROVE:
        return "aprovado", "baixo"
    if prob_good >= _THRESHOLD_REVIEW:
        return "em_analise", "medio"
    return "negado", "alto"


def _build_response(prob_good: float) -> CreditResponse:
    decision, risk_cat = _decision(prob_good)
    return CreditResponse(
        probability_good=round(prob_good, 6),
        probability_bad=round(1.0 - prob_good, 6),
        decision=decision,
        risk_category=risk_cat,
        threshold_used=_THRESHOLD_APPROVE,
    )


# ---------------------------------------------------------------------------
# Lifespan — modelo carregado 1× no boot da aplicação
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):  # type: ignore[type-arg]
    """Carrega o artefato do modelo na inicialização; libera recursos no shutdown."""
    try:
        bundle = load_artifact()
        application.state.model_bundle = bundle
        feature_count = len(bundle.get("feature_names", []))
        logger.info("Modelo carregado — %d features.", feature_count)
    except FileNotFoundError:
        logger.warning(
            "Artefato de modelo não encontrado. " "Execute `make train` e reinicie o servidor."
        )
        application.state.model_bundle = None
    yield
    # Cleanup explícito (se necessário em versões futuras)
    application.state.model_bundle = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Factory for ASGI servers / tests."""
    _app = FastAPI(
        title="German Credit Risk API",
        version="0.2.0",
        description=(
            "API de score de crédito baseada no German Credit Risk dataset. "
            "Modelo calibrado (Platt scaling) com análise de fairness integrada. "
            "Governança alinhada a BACEN Resolução 4.557/2017 e LGPD."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    return _app


app = create_app()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"], summary="Liveness probe")
def health() -> dict[str, str]:
    """Confirma que o processo está rodando (sem verificar o modelo)."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], summary="Readiness probe")
def ready(request: Request) -> JSONResponse:
    """
    Confirma que o modelo está carregado e pronto para scoring.

    Retorna 503 se o artefato de modelo não estiver disponível.
    """
    bundle: Any = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "Modelo não carregado."},
        )
    return JSONResponse({"status": "ready"})


# ---------------------------------------------------------------------------
# Informações do modelo (governança BACEN 4.557)
# ---------------------------------------------------------------------------


@app.get("/v1/model-info", response_model=ModelInfo, tags=["model"])
def model_info(request: Request) -> ModelInfo:
    """
    Retorna metadados do modelo em produção para fins de governança e auditoria.

    Alinhado ao requisito de documentação de modelos da BACEN 4.557/2017.
    """
    bundle: Any = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        return ModelInfo(
            model_name="german_credit_risk",
            version="unavailable",
            feature_count=0,
            features=[],
            calibration_method="unknown",
            status="unavailable",
        )

    feature_names: list[str] = bundle.get("feature_names", [])

    # Lê método de calibração do params.yaml
    try:
        params_path = PROJECT_ROOT / "params.yaml"
        params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
        cal_method = params.get("calibration", {}).get("method", "sigmoid")
        model_name = params.get("registry", {}).get("model_name", "german_credit_risk")
    except Exception:
        cal_method = "sigmoid"
        model_name = "german_credit_risk"

    return ModelInfo(
        model_name=model_name,
        version="0.2.0",
        feature_count=len(feature_names),
        features=feature_names,
        calibration_method=cal_method,
        status="loaded",
    )


# ---------------------------------------------------------------------------
# Scoring individual
# ---------------------------------------------------------------------------


@app.post("/v1/score", response_model=CreditResponse, tags=["scoring"])
def score_v1(req: CreditRequest, request: Request) -> CreditResponse:
    """
    Pontua um único tomador e retorna probabilidade, decisão e categoria de risco.

    O modelo utiliza Platt scaling para garantir probabilidades calibradas.
    A decisão segue a assimetria de custo 5:1 (falso negativo de negócio).
    """
    bundle: Any = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível. Tente novamente.")

    try:
        row = pd.DataFrame([req.model_dump()])
        prob_good = float(predict_proba(row)[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Erro interno no scoring")
        raise HTTPException(status_code=500, detail="Erro interno no scoring.") from exc

    return _build_response(prob_good)


# ---------------------------------------------------------------------------
# Monitoramento — PSI/CSI (Camada 6)
# ---------------------------------------------------------------------------


@app.get("/v1/monitoring", tags=["monitoring"])
def monitoring_report() -> JSONResponse:
    """
    Return the latest PSI/CSI monitoring report.

    Reads ``reports/monitoring_report.json`` generated by ``make monitor``.
    Returns 404 if the report has not been generated yet.

    Typical use: scheduled health-check or dashboard integration.
    Alinhado a BACEN Resolução 4.557/2017 — monitoramento contínuo de modelos.
    """
    report_path = PROJECT_ROOT / "reports" / "monitoring_report.json"
    if not report_path.is_file():
        return JSONResponse(
            status_code=404,
            content={
                "detail": (
                    "Relatório de monitoramento não encontrado. "
                    "Execute `make monitor` para gerá-lo."
                )
            },
        )
    import json as _json

    data = _json.loads(report_path.read_text(encoding="utf-8"))
    return JSONResponse(data)


@app.get("/v1/monitoring/status", tags=["monitoring"])
def monitoring_status() -> JSONResponse:
    """
    Quick health summary for monitoring: reference exists, last PSI/CSI status.

    Returns 200 with ``reference_ready`` flag regardless of report availability.
    """
    report_path = PROJECT_ROOT / "reports" / "monitoring_report.json"
    ref_ready = reference_exists()

    if not report_path.is_file():
        return JSONResponse(
            {
                "reference_ready": ref_ready,
                "report_ready": False,
                "detail": "Execute `make monitor` para gerar o relatório.",
            }
        )

    import json as _json

    data = _json.loads(report_path.read_text(encoding="utf-8"))
    return JSONResponse(
        {
            "reference_ready": ref_ready,
            "report_ready": True,
            "score_psi": data.get("score_psi"),
            "score_psi_status": data.get("score_psi_status"),
            "n_features_warning": data.get("n_features_warning"),
            "n_features_drift": data.get("n_features_drift"),
            "generated_at": data.get("generated_at"),
        }
    )


# Alias legado sem versionamento (retrocompatibilidade)
@app.post("/score", response_model=CreditResponse, tags=["scoring"], include_in_schema=False)
def score_legacy(req: CreditRequest, request: Request) -> CreditResponse:
    return score_v1(req, request)


# ---------------------------------------------------------------------------
# Scoring em lote
# ---------------------------------------------------------------------------


@app.post("/v1/batch", response_model=BatchScoringResponse, tags=["scoring"])
def batch_score(body: BatchScoringRequest, request: Request) -> BatchScoringResponse:
    """
    Pontua até 100 tomadores em uma única chamada.

    Útil para pré-aprovação em batch ou simulações de carteira.
    Retorna a lista de respostas na mesma ordem dos registros enviados.
    """
    bundle: Any = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível.")

    try:
        df = pd.DataFrame([r.model_dump() for r in body.records])
        probas = predict_proba(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Erro interno no batch scoring")
        raise HTTPException(status_code=500, detail="Erro interno no scoring em lote.") from exc

    results = [_build_response(float(p)) for p in probas]
    return BatchScoringResponse(results=results, count=len(results))
