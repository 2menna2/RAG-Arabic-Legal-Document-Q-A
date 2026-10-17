"""FastAPI application.

Endpoints
---------
``POST /api/v1/ask``             grounded Q&A (status-typed response, never a fake answer)
``POST /api/v1/search``          retrieval only (works without an LLM)
``GET  /api/v1/articles/{n}``    article text + provenance (to verify any citation)
``GET  /api/v1/info``            versions of corpus / index / model / prompt
``GET  /health``                 liveness (process is up)
``GET  /ready``                  readiness (index compatible, vector store reachable, LLM configured)
``GET  /metrics``                Prometheus metrics

No streaming endpoint: answers are shown only after citation validation, so
streaming raw tokens would expose unvalidated text (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from ecc_rag import __version__
from ecc_rag.api.ratelimit import TokenBucketLimiter
from ecc_rag.api.schemas import (
    ArticleOut,
    AskRequest,
    AskResponse,
    ReadyResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from ecc_rag.config import PROJECT_ROOT, Settings, get_settings, load_rag_config
from ecc_rag.monitoring import metrics as M
from ecc_rag.monitoring.logging import question_for_log, request_id_var, setup_logging
from ecc_rag.rag.pipeline import RAGPipeline, load_articles_map
from ecc_rag.rag.retriever import Retriever
from ecc_rag.rag.vector_store import VectorStoreError

logger = logging.getLogger("ecc_rag.api")

HTTP_STATUS_FOR_ERROR = {"llm_unavailable": 503, "llm_timeout": 504, "llm_bad_output": 502, "retrieval_unavailable": 503, "timeout": 504}


@dataclass
class Services:
    settings: Settings
    cfg: dict[str, Any]
    pipeline: RAGPipeline | None = None
    articles: dict[int, dict[str, Any]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    startup_error: str | None = None
    llm_error: str | None = None


def build_services(settings: Settings) -> Services:
    """Load index + LLM. Failures are recorded (→ /ready = 503), not hidden."""
    from ecc_rag.rag.index import IndexIncompatibleError, load_index
    from ecc_rag.rag.llm import LLMError, make_llm_client

    cfg = load_rag_config(settings.rag_config_path)
    svc = Services(settings=settings, cfg=cfg)
    data = settings.data_dir
    # Read-only containers: embedded Qdrant needs a writable directory (lock file). Materialise the baked
    # index into QDRANT_PATH (e.g. /tmp) once at start-up; the manifest stays in INDEX_DIR for compatibility checks.
    baked = settings.index_dir / "qdrant"
    qdrant_path = settings.qdrant_path
    if settings.qdrant_url is None and qdrant_path is None and baked.exists() and not os.access(baked, os.W_OK):
        # QDRANT_PATH unset (or set empty by an env file) on a read-only filesystem → use a temp copy
        qdrant_path = Path(tempfile.gettempdir()) / "ecc-qdrant"
    if settings.qdrant_url is None and qdrant_path and not qdrant_path.exists() and baked.exists():
        shutil.copytree(baked, qdrant_path)
    try:
        li = load_index(
            data / "processed/chunks.jsonl",
            data / "processed/articles.jsonl",
            settings.index_dir,
            cfg,
            settings=settings,
            qdrant_path=qdrant_path,
        )
    except (IndexIncompatibleError, VectorStoreError, FileNotFoundError) as exc:
        svc.startup_error = f"{type(exc).__name__}: {exc}"
        logger.error("index not loaded: %s", svc.startup_error)
        return svc
    svc.manifest = li.manifest
    M.INDEX_INFO.labels(
        li.manifest["collection"], li.manifest["embedder"]["name"], li.manifest["corpus_hash"][:16], cfg["chunking"]["strategy"]
    ).set(1)
    llm = None
    try:
        llm = make_llm_client()
    except LLMError as exc:
        svc.llm_error = str(exc)
        logger.warning("LLM not configured: %s", exc)
    reranker = None
    if cfg.get("reranker", {}).get("enabled"):
        from ecc_rag.rag.retriever import CrossEncoderReranker

        reranker = CrossEncoderReranker(cfg["reranker"]["model"])
    retriever = Retriever(li.chunks, li.embedder, li.store, li.manifest["collection"], cfg["retrieval"])
    svc.articles = load_articles_map(data / "processed/articles.jsonl")
    svc.pipeline = RAGPipeline(
        retriever, svc.articles, cfg, llm, reranker=reranker, max_question_chars=settings.max_question_chars, observer=M.observer
    )
    M.load_eval_metrics(PROJECT_ROOT / "reports" / "eval_summary.json")
    return svc


def readiness(svc: Services) -> tuple[bool, dict[str, Any]]:
    """Index loaded + vector store consistent + LLM configured (unless LLM_PROVIDER=none). Updates ecc_ready."""
    checks: dict[str, Any] = {"index": "ok" if svc.pipeline else f"failed: {svc.startup_error}"}
    if svc.pipeline:
        r = svc.pipeline.retriever
        try:
            info = r.store.info(r.collection) if (r.store and r.collection) else {}
            checks["vector_store"] = "ok" if info.get("points") == len(r.chunks) else f"mismatch: {info}"
        except VectorStoreError as exc:
            M.VECTOR_STORE_ERRORS.inc()
            checks["vector_store"] = f"failed: {exc}"
        checks["llm"] = "configured" if svc.pipeline.llm is not None else f"not configured: {svc.llm_error or 'LLM_PROVIDER=none'}"
    require_llm = svc.settings.llm_provider != "none"
    ok = checks["index"] == "ok" and checks.get("vector_store") == "ok" and (not require_llm or checks.get("llm") == "configured")
    M.READY.set(1 if ok else 0)
    return ok, checks


def create_app(services: Services | None = None) -> FastAPI:
    settings = services.settings if services else get_settings()
    setup_logging(settings.log_level)
    limiter = TokenBucketLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or await run_in_threadpool(build_services, settings)
        await run_in_threadpool(readiness, app.state.services)  # initialise the ecc_ready gauge
        yield

    app = FastAPI(
        title="Egyptian Civil Code RAG API",
        version=__version__,
        lifespan=lifespan,
        description="Grounded Q&A over the Egyptian Civil Code. Research aid — not legal advice.",
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID"],
            allow_credentials=False,
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        rid = rid[:64]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            if request.url.path.startswith("/api/") and not limiter.allow(request.client.host if request.client else "unknown"):
                M.RATE_LIMITED.inc()
                response: Response = JSONResponse(
                    status_code=429,
                    content={
                        "request_id": rid,
                        "status": "rejected",
                        "reason": "rate_limited",
                        "message": "عدد الطلبات تجاوز الحد المسموح.",
                    },
                )
            else:
                response = await call_next(request)
        except Exception:
            logger.exception("unhandled error")
            response = JSONResponse(
                status_code=500, content={"request_id": rid, "status": "error", "reason": "internal_error", "message": "حدث خطأ داخلي."}
            )
        matched = request.scope.get("route")
        route = getattr(matched, "path", "unmatched")
        elapsed = time.perf_counter() - start
        M.HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        M.HTTP_LATENCY.labels(route).observe(elapsed)
        response.headers["X-Request-ID"] = rid
        logger.info(
            "request",
            extra={
                "fields": {
                    "method": request.method,
                    "route": route,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                }
            },
        )
        request_id_var.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request_id_var.get(),
                "status": "rejected",
                "reason": "invalid_input",
                "message": "مدخلات غير صالحة.",
                "detail": [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()],
            },
        )

    def _svc(request: Request) -> Services:
        return request.app.state.services

    def _pipeline(request: Request) -> RAGPipeline:
        svc = _svc(request)
        if svc.pipeline is None:
            raise HTTPException(status_code=503, detail={"reason": "index_unavailable", "message": "الفهرس غير محمّل."})
        return svc.pipeline

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"reason": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"request_id": request_id_var.get(), "status": "error", **detail})

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready", tags=["ops"], response_model=ReadyResponse)
    async def ready(request: Request):
        ok, checks = await run_in_threadpool(readiness, _svc(request))
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(M.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/info", tags=["meta"])
    async def info(request: Request) -> dict[str, Any]:
        svc = _svc(request)
        m = svc.manifest
        p = svc.pipeline
        return {
            "version": __version__,
            "corpus_hash": m.get("corpus_hash"),
            "collection": m.get("collection"),
            "embedder": m.get("embedder"),
            "chunking": svc.cfg["chunking"],
            "retrieval_method": svc.cfg["retrieval"]["method"],
            "llm_model": getattr(p.llm, "model", None) if p else None,
            "prompt_version": svc.cfg["generation"]["prompt_version"],
            "index_git_commit": m.get("git_commit"),
        }

    @app.post(
        "/api/v1/ask",
        tags=["rag"],
        response_model=AskResponse,
        responses={503: {"description": "dependency unavailable"}, 504: {"description": "timeout"}},
    )
    async def ask(body: AskRequest, request: Request):
        pipe = _pipeline(request)
        svc = _svc(request)
        start = time.perf_counter()
        logger.info("ask", extra={"fields": {"question": question_for_log(body.question, svc.settings.log_questions)}})
        try:
            res = await asyncio.wait_for(run_in_threadpool(pipe.answer, body.question, body.top_k), timeout=svc.settings.request_timeout_s)
        except TimeoutError:
            M.RAG_OUTCOMES.labels("error", "timeout").inc()
            return JSONResponse(
                status_code=504,
                content={
                    "request_id": request_id_var.get(),
                    "status": "error",
                    "reason": "timeout",
                    "message": "انتهت المهلة قبل اكتمال الإجابة. لم يتم توليد إجابة.",
                    "latency_ms": round(1000 * (time.perf_counter() - start), 2),
                },
            )
        payload = AskResponse(
            request_id=request_id_var.get(),
            status=res.status,
            answer=res.answer,
            reason=res.reason,
            message=res.message,
            citations=[c.__dict__ for c in res.citations],
            sources=[{k: v for k, v in s.__dict__.items() if k != "scores"} for s in res.sources],
            clarification_question=res.clarification_question,
            notices=res.notices,
            latency_ms=round(1000 * (time.perf_counter() - start), 2),
            diagnostics=res.diagnostics if body.include_diagnostics else None,
        )
        code = HTTP_STATUS_FOR_ERROR.get(res.reason or "", 500) if res.status == "error" else 200
        return JSONResponse(status_code=code, content=payload.model_dump())

    @app.post("/api/v1/search", tags=["rag"], response_model=SearchResponse)
    async def search(body: SearchRequest, request: Request):
        pipe = _pipeline(request)
        start = time.perf_counter()
        try:
            _q, hits = await run_in_threadpool(pipe.search, body.question, body.top_k)
        except VectorStoreError:
            M.VECTOR_STORE_ERRORS.inc()
            raise HTTPException(
                status_code=503, detail={"reason": "retrieval_unavailable", "message": "خدمة الاسترجاع غير متاحة."}
            ) from None
        out = [
            SearchHit(
                article_number=s.article_number,
                citation=s.citation,
                source_page=s.source_page,
                legal_status=s.legal_status,
                rank=s.rank,
                text=s.text,
                scores=s.scores,
            )
            for s in pipe._sources(hits, english=False)
        ]
        return SearchResponse(request_id=request_id_var.get(), hits=out, latency_ms=round(1000 * (time.perf_counter() - start), 2))

    @app.get("/api/v1/articles/{number}", tags=["corpus"], response_model=ArticleOut)
    async def article(number: int, request: Request):
        svc = _svc(request)
        a = svc.articles.get(number)
        if a is None:
            raise HTTPException(
                status_code=404, detail={"reason": "article_not_found", "message": "المادة غير موجودة في الـ Corpus (1–1149)."}
            )
        return ArticleOut.model_validate({k: a.get(k) for k in ArticleOut.model_fields})

    return app


app = create_app()
