# MLflow runs (exported from the local tracking store `mlflow.db`)

Produced by `dvc repro` (stages `benchmark`, `evaluate`). Re-create locally with `make benchmark eval` and browse with `make mlflow-ui`.

## Experiment `ecc-rag-eval` — 1 runs

| run | git_commit | dataset_hash | key params | test_recall_at_5 | test_mrr |
|---|---|---|---|---|---|
| eval-c0e6e8d8 | c0e6e8d8 | d162de2ae7c56cc8 | retrieval.method=hybrid, chunking.strategy=article_ctx | 0.8193 | 0.6802 |

## Experiment `ecc-rag-retrieval-benchmark` — 22 runs

| run | git_commit | dataset_hash | key params | test_recall_at_5 | test_mrr |
|---|---|---|---|---|---|
| new-article-bm25 | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article, method=bm25, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7181 | 0.6398 |
| new-article-dense | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article, method=dense, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7710 | 0.6358 |
| new-article-hybrid[bm25+tfidf+dense] | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article, method=hybrid, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7632 | 0.6821 |
| new-article-tfidf | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article, method=tfidf, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7866 | 0.6809 |
| new-article_ctx-bm25 | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=bm25, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7679 | 0.6658 |
| new-article_ctx-dense | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=dense, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7960 | 0.6545 |
| new-article_ctx-hybrid[bm25+dense] | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=hybrid, hybrid_components=bm25,dense, dialect_expansion=True | 0.8053 | 0.6567 |
| new-article_ctx-hybrid[bm25+tfidf+dense+definition]-nodialect | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=hybrid, hybrid_components=bm25,tfidf,dense,definition, dialect_expansion=False | 0.8240 | 0.6984 |
| new-article_ctx-hybrid[bm25+tfidf+dense] | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=hybrid, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.8287 | 0.6848 |
| new-article_ctx-hybrid[bm25+tfidf+dense]-nodialect | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=hybrid, hybrid_components=bm25,tfidf,dense, dialect_expansion=False | 0.8193 | 0.6802 |
| new-article_ctx-hybrid[bm25+tfidf+dense]-nodialect-nostrip | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=hybrid, hybrid_components=bm25,tfidf,dense, dialect_expansion=False | 0.8193 | 0.6899 |
| new-article_ctx-hybrid[bm25+tfidf] | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=hybrid, hybrid_components=bm25,tfidf, dialect_expansion=True | 0.8006 | 0.6941 |
| new-article_ctx-hybrid[tfidf+dense] | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=hybrid, hybrid_components=tfidf,dense, dialect_expansion=True | 0.8380 | 0.6742 |
| new-article_ctx-tfidf | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=article_ctx, method=tfidf, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.8240 | 0.6974 |
| new-paragraph-bm25 | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=paragraph, method=bm25, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7305 | 0.6783 |
| new-paragraph-dense | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=paragraph, method=dense, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.8333 | 0.6811 |
| new-paragraph-hybrid[bm25+tfidf+dense] | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=paragraph, method=hybrid, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.7664 | 0.6989 |
| new-paragraph-tfidf | c0e6e8d8 | d162de2ae7c56cc8 | corpus=new, chunking=paragraph, method=tfidf, hybrid_components=bm25,tfidf,dense, dialect_expansion=True | 0.8193 | 0.7012 |
| reference-reference_article_ar-bm25-nodialect | c0e6e8d8 | d162de2ae7c56cc8 | corpus=reference, chunking=reference_article_ar, method=bm25, hybrid_components=bm25,tfidf,dense, dialect_expansion=False | 0.6947 | 0.6066 |
| reference-reference_article_ar-dense-nodialect | c0e6e8d8 | d162de2ae7c56cc8 | corpus=reference, chunking=reference_article_ar, method=dense, hybrid_components=bm25,tfidf,dense, dialect_expansion=False | 0.7586 | 0.6332 |
| reference-reference_article_ar-hybrid[bm25+tfidf+dense]-nodialect | c0e6e8d8 | d162de2ae7c56cc8 | corpus=reference, chunking=reference_article_ar, method=hybrid, hybrid_components=bm25,tfidf,dense, dialect_expansion=False | 0.7492 | 0.6661 |
| reference-reference_article_ar-tfidf-nodialect | c0e6e8d8 | d162de2ae7c56cc8 | corpus=reference, chunking=reference_article_ar, method=tfidf, hybrid_components=bm25,tfidf,dense, dialect_expansion=False | 0.7710 | 0.6885 |
