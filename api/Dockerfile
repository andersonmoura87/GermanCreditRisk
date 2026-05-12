# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder: instala dependências em venv isolado
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /build

# Poetry como ferramenta de build, sem instalar no PATH global
ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

RUN pip install --no-cache-dir poetry==1.8.3

# Copia apenas lock files — camada de cache que só invalida quando deps mudam
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root && rm -rf $POETRY_CACHE_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime: imagem final mínima e segura
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

# curl necessário apenas para o HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Usuário não-root — boa prática de segurança de container (CIS Benchmark)
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Venv do builder — sem instalar pip, poetry ou compiladores no runtime
COPY --from=builder /build/.venv /app/.venv

# Código da aplicação
COPY --chown=appuser:appgroup src/     ./src/
COPY --chown=appuser:appgroup api/     ./api/
COPY --chown=appuser:appgroup params.yaml ./

# Modelo treinado (pode ser montado via volume em ambientes com Registry)
COPY --chown=appuser:appgroup models/  ./models/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# Healthcheck alinhado ao endpoint /health da FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
