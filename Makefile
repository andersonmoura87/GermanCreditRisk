.PHONY: install lint test test-ci test-cov-main test-slow \
        security security-audit \
        train evaluate monitor \
        mlflow mlflow-register mlflow-promote-staging mlflow-promote-prod mlflow-list mlflow-compare \
        build build-no-cache build-push \
        serve streamlit clean report dvc-repro

# Imagem local (deve coincidir com env.IMAGE_NAME no ci.yml)
IMAGE_NAME ?= german-credit-risk-api
IMAGE_TAG  ?= local

PYTHON ?= python
POETRY ?= poetry

# ---------------------------------------------------------------------------
# Dependências
# ---------------------------------------------------------------------------
install:
	$(POETRY) install

# ---------------------------------------------------------------------------
# Qualidade de código
# ---------------------------------------------------------------------------
lint:
	$(POETRY) run ruff check src api tests
	$(POETRY) run black --check src api tests
	$(POETRY) run mypy src api

# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------
test:
	$(POETRY) run pytest

# CI rápido: sem cobertura + exclui ``slow`` (GX completo).
test-ci:
	$(POETRY) run pytest -m "not slow" --no-cov -q

# CI main/master: cobertura + XML.
test-cov-main:
	$(POETRY) run pytest -m "not slow" -q \
		--cov=src --cov=api \
		--cov-report=term-missing --cov-report=xml \
		--cov-fail-under=30

# Apenas testes marcados com @pytest.mark.slow (ex.: GX batch.validate).
test-slow:
	$(POETRY) run pytest -m slow --no-cov -q

# ---------------------------------------------------------------------------
# DevSecOps
# ---------------------------------------------------------------------------
security:
	$(POETRY) run bandit -r src api -ll -q

security-audit:
	$(POETRY) run pip-audit

# ---------------------------------------------------------------------------
# Pipeline DVC (treino reproduzível)
# ---------------------------------------------------------------------------
train:
	$(POETRY) run dvc repro train

evaluate:
	$(POETRY) run dvc repro evaluate

# Camada 6 — PSI/CSI drift report (standalone, sem DVC)
# Uso: make monitor
#      make monitor CURRENT=data/processed/producao_janela.parquet
monitor:
	$(POETRY) run python -m src.monitoring.report $(if $(CURRENT),--current-parquet $(CURRENT),)

dvc-repro:
	$(POETRY) run dvc repro

# ---------------------------------------------------------------------------
# MLflow — UI e Model Registry
# ---------------------------------------------------------------------------
mlflow:
	$(POETRY) run mlflow ui --host 127.0.0.1 --port 5000

# Registra o último run no Model Registry
mlflow-register:
	$(POETRY) run python -m src.models.registry register

# Promove versão para Staging (VERSION=<n>)
mlflow-promote-staging:
	$(POETRY) run python -m src.models.registry promote $(VERSION) --stage staging

# Promove versão para Production (VERSION=<n>)
mlflow-promote-prod:
	$(POETRY) run python -m src.models.registry promote $(VERSION) --stage production

# Lista todas as versões no Registry
mlflow-list:
	$(POETRY) run python -m src.models.registry list

# Compara runs do experimento (top 10 por KS)
mlflow-compare:
	$(POETRY) run python -m src.models.registry compare

# ---------------------------------------------------------------------------
# Docker local (alinhado ao Dockerfile multi-stage em api/)
# ---------------------------------------------------------------------------
# Build padrão (usa cache do Docker)
build:
	docker build -f api/Dockerfile -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Build sem cache — útil para auditar dependências da imagem base
build-no-cache:
	docker build --no-cache -f api/Dockerfile -t $(IMAGE_NAME):$(IMAGE_TAG) .

# Build + push para GHCR (requer: docker login ghcr.io -u <user> -p <PAT>)
# Uso: make build-push IMAGE_TAG=sha-abcdef
build-push:
	docker build -f api/Dockerfile -t ghcr.io/$(GITHUB_REPOSITORY_OWNER)/$(IMAGE_NAME):$(IMAGE_TAG) .
	docker push ghcr.io/$(GITHUB_REPOSITORY_OWNER)/$(IMAGE_NAME):$(IMAGE_TAG)

# ---------------------------------------------------------------------------
# Serving & demo
# ---------------------------------------------------------------------------
serve:
	$(POETRY) run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	$(POETRY) run streamlit run streamlit_app.py --server.port 8501

# ---------------------------------------------------------------------------
# Relatório executivo
# ---------------------------------------------------------------------------
report:
	$(POETRY) run jupyter nbconvert --execute --inplace notebooks/04_report.ipynb

# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------
clean:
	$(RM) -r mlruns .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage 2>/dev/null || true
	$(RM) -r reports/metrics.json reports/calibration_curve.csv reports/score_distribution.csv 2>/dev/null || true
