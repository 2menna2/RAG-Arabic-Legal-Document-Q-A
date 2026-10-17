# ------------------------------------------------------------------------------------------------
# Egyptian Civil Code RAG — API image
#   * dependencies installed from uv.lock (reproducible), project installed non-editable
#   * the vector index is BUILT INSIDE the image from data/processed/*.jsonl (pulled with DVC), so
#     the image is self-consistent: corpus hash + chunking + embedder + library versions match
#   * runs as non-root, no secrets baked in (GEMINI_API_KEY comes from the runtime environment)
# Base image: pin by digest in production, e.g.
#   docker build --build-arg BASE_IMAGE=python:3.12-slim-bookworm@sha256:<digest> ...
# ------------------------------------------------------------------------------------------------
ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG UV_VERSION=0.11.32

FROM ${BASE_IMAGE} AS builder
ARG UV_VERSION
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# Optional build secret "ca": extra CA bundle for TLS-intercepting corporate proxies (never stored in a layer).
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca PIP_CERT=/run/secrets/ca; fi; \
    python -m venv /opt/uv && /opt/uv/bin/pip install --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
# dependencies only (layer cached until uv.lock changes)
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca; fi; \
    /opt/uv/bin/uv sync --frozen --no-dev --no-install-project --python python
COPY src ./src
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export SSL_CERT_FILE=/run/secrets/ca; fi; \
    /opt/uv/bin/uv sync --frozen --no-dev --no-editable --python python

FROM builder AS index
ENV ECC_HOME=/app DATA_DIR=/app/data INDEX_DIR=/app/data/index RAG_CONFIG_PATH=/app/configs/rag.yaml QDRANT_PATH=/app/data/index/qdrant
WORKDIR /app
COPY configs ./configs
COPY data/processed/articles.jsonl data/processed/chunks.jsonl ./data/processed/
RUN /opt/venv/bin/python -m ecc_rag.rag.index --chunks data/processed/chunks.jsonl \
        --articles data/processed/articles.jsonl --index-dir data/index

FROM ${BASE_IMAGE} AS runtime
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="ecc-rag-api" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 (code only; see docs/DATA_SOURCES.md for the corpus)"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH \
    ECC_HOME=/app DATA_DIR=/app/data INDEX_DIR=/app/data/index RAG_CONFIG_PATH=/app/configs/rag.yaml \
    QDRANT_PATH=/tmp/ecc-qdrant APP_ENV=prod LOG_LEVEL=INFO GIT_SHA=${GIT_SHA}
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app --from=index /app/data ./data
COPY --chown=app:app reports/eval_summary.json ./reports/eval_summary.json
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]
# One worker: embedded Qdrant is single-process. Scale with replicas + QDRANT_URL (see docs/DEPLOYMENT.md).
CMD ["uvicorn", "ecc_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--timeout-keep-alive", "30"]
