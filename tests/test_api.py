"""API tests — health, readiness, scoring, batch, model-info e validação de erros."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from api.main import app
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Payload válido mínimo
# ---------------------------------------------------------------------------
VALID_PAYLOAD: dict = {
    "account_status": 1,
    "credit_duration": 12,
    "history_of_compliance": 4,
    "credit_purpose": 2,
    "credit_amount": 1500.0,
    "savings": 1,
    "employment_duration": 3,
    "installment_rate": 2,
    "personal_status_sex": 3,
    "other_debtors": 1,
    "present_residence": 2,
    "property": 2,
    "age": 35,
    "other_installment_plans": 3,
    "type_of_housing": 1,
    "number_credits": 1,
    "job": 3,
    "people_liable": 1,
    "telephone": -1,
    "foreign_worker": 0,
    "level_of_education": 3,
    "entry_payment": 100.0,
}


def _mock_bundle() -> dict:
    """Bundle de modelo falso (injetado em app.state DENTRO do contexto TestClient)."""
    return {
        "pipeline": MagicMock(),
        "feature_names": list(VALID_PAYLOAD.keys()),
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def test_health_always_ok() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_without_model_returns_503() -> None:
    with TestClient(app) as client:
        app.state.model_bundle = None  # garante ausência do bundle
        r = client.get("/ready")
    assert r.status_code == 503
    assert "unavailable" in r.json()["status"]


def test_ready_with_model_returns_200() -> None:
    with TestClient(app) as client:
        # Injetado após o lifespan (que reseta para None quando não há artefato em disco)
        app.state.model_bundle = _mock_bundle()
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


def test_model_info_unavailable_when_no_bundle() -> None:
    with TestClient(app) as client:
        app.state.model_bundle = None
        r = client.get("/v1/model-info")
    assert r.status_code == 200
    assert r.json()["status"] == "unavailable"


def test_model_info_loaded() -> None:
    with TestClient(app) as client:
        app.state.model_bundle = _mock_bundle()
        r = client.get("/v1/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "loaded"
    assert body["feature_count"] == len(VALID_PAYLOAD)
    assert isinstance(body["features"], list)


# ---------------------------------------------------------------------------
# /v1/score — scoring individual
# ---------------------------------------------------------------------------


@patch("api.main.predict_proba", return_value=np.array([0.80]))
def test_score_aprovado(mock_pp: object) -> None:
    with TestClient(app) as client:
        app.state.model_bundle = _mock_bundle()
        r = client.post("/v1/score", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "aprovado"
    assert body["risk_category"] == "baixo"
    assert abs(body["probability_good"] - 0.80) < 1e-5
    assert abs(body["probability_bad"] - 0.20) < 1e-5


@patch("api.main.predict_proba", return_value=np.array([0.52]))
def test_score_em_analise(mock_pp: object) -> None:
    with TestClient(app) as client:
        app.state.model_bundle = _mock_bundle()
        r = client.post("/v1/score", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert r.json()["decision"] == "em_analise"
    assert r.json()["risk_category"] == "medio"


@patch("api.main.predict_proba", return_value=np.array([0.25]))
def test_score_negado(mock_pp: object) -> None:
    with TestClient(app) as client:
        app.state.model_bundle = _mock_bundle()
        r = client.post("/v1/score", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert r.json()["decision"] == "negado"
    assert r.json()["risk_category"] == "alto"


def test_score_503_when_model_unavailable() -> None:
    with TestClient(app) as client:
        app.state.model_bundle = None
        r = client.post("/v1/score", json=VALID_PAYLOAD)
    assert r.status_code == 503


def test_score_422_invalid_account_status() -> None:
    bad = {**VALID_PAYLOAD, "account_status": 99}
    with TestClient(app) as client:
        r = client.post("/v1/score", json=bad)
    assert r.status_code == 422


def test_score_422_negative_credit_amount() -> None:
    bad = {**VALID_PAYLOAD, "credit_amount": -500.0}
    with TestClient(app) as client:
        r = client.post("/v1/score", json=bad)
    assert r.status_code == 422


def test_score_422_credit_duration_exceeds_120() -> None:
    bad = {**VALID_PAYLOAD, "credit_duration": 200}
    with TestClient(app) as client:
        r = client.post("/v1/score", json=bad)
    assert r.status_code == 422


def test_score_422_extra_field_forbidden() -> None:
    """Campos extras devem ser rejeitados (ConfigDict extra='forbid')."""
    bad = {**VALID_PAYLOAD, "campo_desconhecido": 999}
    with TestClient(app) as client:
        r = client.post("/v1/score", json=bad)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /score (alias legado)
# ---------------------------------------------------------------------------


@patch("api.main.predict_proba", return_value=np.array([0.72]))
def test_score_legacy_alias(mock_pp: object) -> None:
    """O alias /score deve funcionar identicamente ao /v1/score."""
    with TestClient(app) as client:
        app.state.model_bundle = _mock_bundle()
        r = client.post("/score", json=VALID_PAYLOAD)
    assert r.status_code == 200
    assert "decision" in r.json()


# ---------------------------------------------------------------------------
# /v1/batch — scoring em lote
# ---------------------------------------------------------------------------


@patch("api.main.predict_proba", return_value=np.array([0.80, 0.30]))
def test_batch_two_records(mock_pp: object) -> None:
    batch_body = {"records": [VALID_PAYLOAD, VALID_PAYLOAD]}
    with TestClient(app) as client:
        app.state.model_bundle = _mock_bundle()
        r = client.post("/v1/batch", json=batch_body)
    assert r.status_code == 200
    resp = r.json()
    assert resp["count"] == 2
    assert resp["results"][0]["decision"] == "aprovado"
    assert resp["results"][1]["decision"] == "negado"


def test_batch_503_when_model_unavailable() -> None:
    with TestClient(app) as client:
        app.state.model_bundle = None
        r = client.post("/v1/batch", json={"records": [VALID_PAYLOAD]})
    assert r.status_code == 503


def test_batch_422_empty_list() -> None:
    """Lista vazia deve retornar 422 (min_length=1)."""
    with TestClient(app) as client:
        r = client.post("/v1/batch", json={"records": []})
    assert r.status_code == 422


def test_batch_422_over_limit() -> None:
    """Mais de 100 registros deve retornar 422 (max_length=100)."""
    with TestClient(app) as client:
        r = client.post("/v1/batch", json={"records": [VALID_PAYLOAD] * 101})
    assert r.status_code == 422
