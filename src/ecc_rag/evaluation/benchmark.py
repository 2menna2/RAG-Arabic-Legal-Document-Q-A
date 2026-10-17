"""Retrieval benchmark: corpora × chunking × retrieval method, on a fixed golden set.

Every configuration is evaluated on the same golden items, logged to MLflow
(params, metrics per split, per-question artefact) and summarised in
``reports/retrieval_benchmark.{json,md}``.

Selection rule (declared before looking at results): the configuration with
the highest **dev** Recall@5, ties broken by dev MRR, is selected; **test**
numbers are reported for that configuration only as the unbiased estimate.

Neural embedders / cross-encoders that are unavailable in the execution
environment are recorded as ``not_run`` with the reason — never silently
replaced.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from ecc_rag.config import PROJECT_ROOT, config_hash, load_rag_config
from ecc_rag.data_pipeline.chunking import build_chunks, load_articles
from ecc_rag.evaluation.dataset import GoldenItem, dataset_hash, load_golden
from ecc_rag.evaluation.retrieval_metrics import all_metrics
from ecc_rag.rag.embeddings import LsaEmbedder
from ecc_rag.rag.index import corpus_hash_of, git_commit
from ecc_rag.rag.query import analyze_query
from ecc_rag.rag.retriever import Retriever
from ecc_rag.rag.vector_store import QdrantStore

logger = logging.getLogger(__name__)
KS = (1, 3, 5, 10)


def reference_corpus_chunks(ref_json: Path) -> list[dict[str, Any]]:
    """Rebuild the reference project's retrieval units (Arabic chunk per article, 'المادة N:' prefix)."""
    arts = json.loads(ref_json.read_text(encoding="utf-8"))
    chunks = []
    for a in arts:
        n = a["article_number"]
        txt = (a.get("text_ar") or "").strip()
        chunks.append(
            {
                "chunk_id": f"ref-{n}-ar",
                "article_number": n,
                "article_id": f"ref-{n}",
                "chunk_index": 0,
                "kind": "article",
                "language": "ar",
                "display_text": txt,
                "search_text": f"المادة {n}:\n{txt}",
                "content_hash": "",
                "legal_status": "repealed_per_source" if a.get("is_repealed") else "unverified",
                "source_page": a.get("source_page"),
            }
        )
    return chunks


def build_retriever(chunks: list[dict[str, Any]], emb_cfg: dict[str, Any], ret_cfg: dict[str, Any], workdir: Path) -> Retriever:
    lsa = emb_cfg.get("lsa", {})
    emb = LsaEmbedder(lsa.get("n_components", 256), tuple(lsa.get("ngram_range", (2, 5))), lsa.get("min_df", 2))
    texts = [c["search_text"] for c in chunks]
    emb.fit(texts)
    store = QdrantStore(path=workdir / f"q_{config_hash([len(chunks), texts[0][:50]])}")
    coll = "bench"
    store.recreate(coll, emb.dim)
    store.upsert(coll, emb.embed_documents(texts), [{"article_number": c["article_number"]} for c in chunks])
    return Retriever(chunks, emb, store, coll, ret_cfg)


def evaluate_config(retriever: Retriever, items: list[GoldenItem], method: str, top_k: int = 10) -> list[dict[str, Any]]:
    rows = []
    for it in items:
        if not it.is_retrieval_item:
            continue
        q = analyze_query(it.question)
        t0 = time.perf_counter()
        hits = retriever.retrieve(q, method=method, top_k=top_k)
        latency = time.perf_counter() - t0
        ranked = [h.article_number for h in hits]
        m = all_metrics(ranked, it.expected_articles, it.acceptable_articles, KS)
        rows.append(
            {
                "id": it.id,
                "split": it.split,
                "style": it.style,
                "type": it.type,
                "difficulty": it.difficulty,
                "expected": it.expected_articles,
                "ranked": ranked,
                "latency_s": round(latency, 4),
                **m,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], split: str | None = None, key: str | None = None, value: str | None = None) -> dict[str, float]:
    sel = [r for r in rows if (split is None or r["split"] == split) and (key is None or r[key] == value)]
    if not sel:
        return {}
    out = {
        m: round(statistics.fmean(r[m] for r in sel), 4)
        for m in ["mrr"] + [f"{p}@{k}" for k in KS for p in ("recall", "hit", "precision", "ndcg")]
    }
    lat = sorted(r["latency_s"] for r in sel)
    out["n"] = len(sel)
    out["latency_p50_ms"] = round(1000 * lat[len(lat) // 2], 2)
    out["latency_p95_ms"] = round(1000 * lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlflow

    cfg = load_rag_config()
    version, items = load_golden(args.golden)
    ds_hash = dataset_hash(args.golden)
    articles = load_articles(args.articles)
    corpus_h = corpus_hash_of(args.articles)
    commit = git_commit()
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    grid: list[dict[str, Any]] = []
    if args.reference_corpus and args.reference_corpus.exists():
        for m in ("bm25", "tfidf", "dense", "hybrid"):
            grid.append(
                {
                    "corpus": "reference",
                    "chunking": "reference_article_ar",
                    "method": m,
                    "components": ["bm25", "tfidf", "dense"],
                    "dialect": False,
                }
            )
    for ch in ("article", "article_ctx", "paragraph"):
        for m in ("bm25", "tfidf", "dense", "hybrid"):
            grid.append({"corpus": "new", "chunking": ch, "method": m, "components": ["bm25", "tfidf", "dense"], "dialect": True})
    for comps in (["bm25", "tfidf"], ["bm25", "dense"], ["tfidf", "dense"]):
        grid.append({"corpus": "new", "chunking": "article_ctx", "method": "hybrid", "components": comps, "dialect": True})
    grid.append(
        {"corpus": "new", "chunking": "article_ctx", "method": "hybrid", "components": ["bm25", "tfidf", "dense"], "dialect": False}
    )
    grid.append(
        {
            "corpus": "new",
            "chunking": "article_ctx",
            "method": "hybrid",
            "components": ["bm25", "tfidf", "dense", "definition"],
            "dialect": False,
        }
    )
    grid.append(
        {
            "corpus": "new",
            "chunking": "article_ctx",
            "method": "hybrid",
            "components": ["bm25", "tfidf", "dense"],
            "dialect": False,
            "strip": False,
        }
    )

    results: list[dict[str, Any]] = []
    cache: dict[str, Retriever] = {}
    with tempfile.TemporaryDirectory() as td:
        for g in grid:
            key = f"{g['corpus']}|{g['chunking']}"
            if key not in cache:
                if g["corpus"] == "reference":
                    chunks = reference_corpus_chunks(args.reference_corpus)
                else:
                    chunks = build_chunks(articles, g["chunking"], cfg["chunking"]["params"])
                cache[key] = build_retriever(chunks, cfg["embedding"], dict(cfg["retrieval"]), Path(td))
            r = cache[key]
            r.cfg = {
                **cfg["retrieval"],
                "hybrid_components": g["components"],
                "use_dialect_expansion": g["dialect"],
                "strip_question_frame": g.get("strip", True),
            }
            name = (
                f"{g['corpus']}-{g['chunking']}-{g['method']}"
                + (f"[{'+'.join(g['components'])}]" if g["method"] == "hybrid" else "")
                + ("" if g["dialect"] else "-nodialect")
                + ("" if g.get("strip", True) else "-nostrip")
            )
            rows = evaluate_config(r, items, g["method"])
            summary = {s: summarize(rows, s) for s in ("dev", "test")}
            summary["all"] = summarize(rows)
            by_style = {st: summarize(rows, None, "style", st) for st in ("msa", "egyptian", "english")}
            res = {"name": name, **g, "summary": summary, "by_style": by_style, "rows": rows}
            results.append(res)
            with mlflow.start_run(run_name=name):
                mlflow.set_tags(
                    {
                        "git_commit": commit,
                        "dataset_version": version,
                        "dataset_hash": ds_hash,
                        "corpus_hash": corpus_h if g["corpus"] == "new" else "reference-pipeline-output",
                        "stage": "retrieval_benchmark",
                        "embedder": "lsa(char-tfidf+svd) — classical, not neural",
                    }
                )
                mlflow.log_params(
                    {
                        "corpus": g["corpus"],
                        "chunking": g["chunking"],
                        "method": g["method"],
                        "hybrid_components": ",".join(g["components"]),
                        "dialect_expansion": g["dialect"],
                        "strip_question_frame": g.get("strip", True),
                        "rrf_k": cfg["retrieval"]["rrf_k"],
                        "candidate_k": cfg["retrieval"]["candidate_k"],
                        "lsa_dim": cfg["embedding"]["lsa"]["n_components"],
                    }
                )
                for split, s in summary.items():
                    for mk, mv in s.items():
                        mlflow.log_metric(f"{split}_{mk.replace('@', '_at_')}", float(mv))
                art = Path(td) / f"{name}.json"
                art.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
                mlflow.log_artifact(str(art), artifact_path="per_question")
            logger.info(
                "%-60s dev R@5=%.3f MRR=%.3f | test R@5=%.3f",
                name,
                summary["dev"]["recall@5"],
                summary["dev"]["mrr"],
                summary["test"]["recall@5"],
            )

    candidates = [r for r in results if r["corpus"] == "new"]
    best = max(candidates, key=lambda r: (r["summary"]["dev"]["recall@5"], r["summary"]["dev"]["mrr"]))
    not_run = [
        {
            "config": "dense (neural: gemini-embedding-001 / multilingual-e5 / bge-m3)",
            "reason": "no network access to Gemini API / HuggingFace in the execution environment",
        },
        {
            "config": "cross-encoder reranking (BAAI/bge-reranker-v2-m3)",
            "reason": "model download from HuggingFace blocked in the execution environment",
        },
    ]
    out = {
        "dataset_version": version,
        "dataset_hash": ds_hash,
        "corpus_hash": corpus_h,
        "git_commit": commit,
        "selection_rule": "max dev recall@5, tie-break dev MRR",
        "selected": best["name"],
        "not_run": not_run,
        "results": [{k: v for k, v in r.items() if k != "rows"} for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    (args.output.parent / "retrieval_benchmark_rows.json").write_text(
        json.dumps({r["name"]: r["rows"] for r in results}, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    write_markdown(out, args.output.with_suffix(".md"))
    return out


def write_markdown(out: dict[str, Any], path: Path) -> None:
    lines = [
        "# Retrieval benchmark",
        "",
        f"- Golden dataset: `{out['dataset_version']}` (sha256/16 `{out['dataset_hash']}`) — authored, **not lawyer-reviewed**",
        f"- Corpus hash: `{out['corpus_hash'][:16]}` · git commit: `{out['git_commit'][:10]}`",
        "- Dense component in this run = **LSA (char n-gram TF-IDF + SVD-256)**, a classical vector model; neural embedders were not available (see *Not run*).",
        f"- Selection rule (fixed in advance): {out['selection_rule']} → **{out['selected']}**",
        "- Numbers below are computed by `python -m ecc_rag.evaluation.benchmark`; they are not hand-written.",
        "",
        "| config | dev R@1 | dev R@5 | dev R@10 | dev MRR | dev nDCG@5 | dev P@5 | test R@5 | test MRR | test nDCG@5 | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["results"]:
        d, t = r["summary"]["dev"], r["summary"]["test"]
        lines.append(
            f"| {r['name']} | {d['recall@1']:.3f} | {d['recall@5']:.3f} | {d['recall@10']:.3f} | {d['mrr']:.3f} | {d['ndcg@5']:.3f} | {d['precision@5']:.3f} | {t['recall@5']:.3f} | {t['mrr']:.3f} | {t['ndcg@5']:.3f} | {r['summary']['all']['latency_p50_ms']} |"
        )
    lines += ["", "## Recall@5 by question style (all splits)", "", "| config | MSA | Egyptian | English |", "|---|---|---|---|"]
    for r in out["results"]:
        bs = r["by_style"]
        lines.append(
            f"| {r['name']} | {bs['msa'].get('recall@5', 0):.3f} (n={bs['msa'].get('n', 0)}) | {bs['egyptian'].get('recall@5', 0):.3f} (n={bs['egyptian'].get('n', 0)}) | {bs['english'].get('recall@5', 0):.3f} (n={bs['english'].get('n', 0)}) |"
        )
    lines += ["", "## Not run", ""] + [f"- {n['config']}: {n['reason']}" for n in out["not_run"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--articles", type=Path, default=PROJECT_ROOT / "data/processed/articles.jsonl")
    p.add_argument("--golden", type=Path, default=None)
    p.add_argument("--reference-corpus", type=Path, default=None, help="reference project's civil_code_ready.json (baseline)")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/retrieval_benchmark.json")
    p.add_argument("--mlflow-uri", default=f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    p.add_argument("--experiment", default="ecc-rag-retrieval-benchmark")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    out = run(args)
    print(json.dumps({"selected": out["selected"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
