# One entry point per task. Every target is runnable from a fresh clone (after `make setup` + `dvc pull`).
PY ?= python
IMAGE ?= ecc-rag-api
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
BASE_IMAGE ?= python:3.12-slim-bookworm
MLFLOW_URI ?= sqlite:///mlflow.db

.PHONY: help setup data index eval eval-llm benchmark gate baseline test test-unit test-integration lint typecheck \
        serve docker-build docker-run smoke loadtest mlflow-ui clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

setup:           ## create .venv from uv.lock (dev + pipeline + mlops extras)
	uv sync --frozen --extra dev --extra pipeline --extra mlops

data:            ## extract → parse → validate → chunk (DVC stages)
	dvc repro chunk

index: data      ## build the vector index (DVC stage)
	dvc repro index

eval: index      ## offline evaluation (retrieval + decision layer) logged to MLflow
	$(PY) -m ecc_rag.evaluation.run_eval --mlflow-uri $(MLFLOW_URI)

eval-llm:        ## end-to-end evaluation with Gemini (needs GEMINI_API_KEY); add ARGS="--judge --ragas"
	$(PY) -m ecc_rag.evaluation.rag_eval --split test --min-interval 7 $(ARGS)

benchmark:       ## retrieval benchmark (corpora × chunking × methods) logged to MLflow
	dvc repro benchmark

gate:            ## quality gate against configs/quality_gates.yaml and the committed baseline
	$(PY) -m ecc_rag.evaluation.quality_gate

baseline:        ## DELIBERATELY promote the current eval summary to the new baseline (commit the result)
	cp reports/eval_summary.json reports/baseline/eval_summary.json

test-unit:       ## unit tests (synthetic fixtures, no corpus needed)
	$(PY) -m pytest tests/unit --cov --cov-report=term-missing --cov-fail-under=60

test-integration: ## integration + e2e tests on the real corpus/index
	$(PY) -m pytest tests/integration tests/e2e -m "integration or e2e"

test:            ## all tests with coverage gate
	$(PY) -m pytest tests --cov --cov-report=term --cov-report=xml --cov-fail-under=70

lint:            ## ruff lint + format check
	ruff check src tests scripts && ruff format --check src tests scripts

typecheck:       ## mypy
	mypy

serve:           ## run the API locally (embedded Qdrant)
	uvicorn ecc_rag.api.main:app --host 127.0.0.1 --port 8000

docker-build:    ## build the image tagged with the git SHA
	docker build -f docker/Dockerfile --build-arg BASE_IMAGE=$(BASE_IMAGE) --build-arg GIT_SHA=$(TAG) -t $(IMAGE):$(TAG) .

docker-run:      ## run the image read-only (reads secrets from .env)
	docker run --rm -d --name ecc-rag --read-only --tmpfs /tmp -p 127.0.0.1:8000:8000 --env-file .env $(IMAGE):$(TAG)

smoke:           ## smoke-test a running API: make smoke URL=http://127.0.0.1:8000
	bash scripts/smoke_test.sh $(or $(URL),http://127.0.0.1:8000)

loadtest:        ## closed-loop load test against URL
	$(PY) scripts/loadtest.py --base-url $(or $(URL),http://127.0.0.1:8000) --endpoint search --concurrency 1 4 8 16 --duration 20

mlflow-ui:       ## browse local experiments
	mlflow ui --backend-store-uri $(MLFLOW_URI) --port 5000

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache coverage.xml htmlcov
