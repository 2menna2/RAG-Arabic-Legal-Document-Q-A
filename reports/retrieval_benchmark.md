# Retrieval benchmark

- Golden dataset: `golden_v1` (sha256/16 `d162de2ae7c56cc8`) — authored, **not lawyer-reviewed**
- Corpus hash: `09b6949d3b3aee1b` · git commit: `c0e6e8d8bc`
- Dense component in this run = **LSA (char n-gram TF-IDF + SVD-256)**, a classical vector model; neural embedders were not available (see *Not run*).
- Selection rule (fixed in advance): max dev recall@5, tie-break dev MRR → **new-article_ctx-hybrid[bm25+tfidf+dense]-nodialect**
- Numbers below are computed by `python -m ecc_rag.evaluation.benchmark`; they are not hand-written.

| config | dev R@1 | dev R@5 | dev R@10 | dev MRR | dev nDCG@5 | dev P@5 | test R@5 | test MRR | test nDCG@5 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| reference-reference_article_ar-bm25-nodialect | 0.399 | 0.601 | 0.643 | 0.493 | 0.466 | 0.193 | 0.695 | 0.607 | 0.589 | 1.1 |
| reference-reference_article_ar-tfidf-nodialect | 0.429 | 0.643 | 0.708 | 0.534 | 0.511 | 0.207 | 0.771 | 0.689 | 0.650 | 2.6 |
| reference-reference_article_ar-dense-nodialect | 0.339 | 0.643 | 0.750 | 0.494 | 0.493 | 0.212 | 0.759 | 0.633 | 0.625 | 2.6 |
| reference-reference_article_ar-hybrid[bm25+tfidf+dense]-nodialect | 0.387 | 0.661 | 0.750 | 0.515 | 0.504 | 0.217 | 0.749 | 0.666 | 0.634 | 6.8 |
| new-article-bm25 | 0.441 | 0.684 | 0.750 | 0.562 | 0.542 | 0.226 | 0.718 | 0.640 | 0.618 | 1.1 |
| new-article-tfidf | 0.441 | 0.643 | 0.738 | 0.547 | 0.536 | 0.221 | 0.787 | 0.681 | 0.652 | 2.7 |
| new-article-dense | 0.381 | 0.643 | 0.744 | 0.509 | 0.506 | 0.212 | 0.771 | 0.636 | 0.625 | 2.6 |
| new-article-hybrid[bm25+tfidf+dense] | 0.429 | 0.679 | 0.768 | 0.551 | 0.543 | 0.233 | 0.763 | 0.682 | 0.651 | 7.6 |
| new-article_ctx-bm25 | 0.500 | 0.696 | 0.780 | 0.608 | 0.576 | 0.238 | 0.768 | 0.666 | 0.655 | 1.4 |
| new-article_ctx-tfidf | 0.464 | 0.667 | 0.750 | 0.583 | 0.552 | 0.226 | 0.824 | 0.697 | 0.684 | 4.7 |
| new-article_ctx-dense | 0.494 | 0.732 | 0.809 | 0.617 | 0.600 | 0.252 | 0.796 | 0.654 | 0.662 | 2.5 |
| new-article_ctx-hybrid[bm25+tfidf+dense] | 0.482 | 0.726 | 0.821 | 0.625 | 0.599 | 0.245 | 0.829 | 0.685 | 0.686 | 11.5 |
| new-paragraph-bm25 | 0.476 | 0.684 | 0.768 | 0.593 | 0.563 | 0.236 | 0.731 | 0.678 | 0.653 | 1.2 |
| new-paragraph-tfidf | 0.452 | 0.655 | 0.756 | 0.576 | 0.551 | 0.221 | 0.819 | 0.701 | 0.683 | 3.3 |
| new-paragraph-dense | 0.434 | 0.679 | 0.804 | 0.581 | 0.568 | 0.240 | 0.833 | 0.681 | 0.682 | 2.5 |
| new-paragraph-hybrid[bm25+tfidf+dense] | 0.446 | 0.679 | 0.792 | 0.586 | 0.567 | 0.233 | 0.766 | 0.699 | 0.667 | 8.3 |
| new-article_ctx-hybrid[bm25+tfidf] | 0.470 | 0.702 | 0.792 | 0.602 | 0.563 | 0.231 | 0.801 | 0.694 | 0.680 | 6.9 |
| new-article_ctx-hybrid[bm25+dense] | 0.512 | 0.696 | 0.816 | 0.633 | 0.601 | 0.245 | 0.805 | 0.657 | 0.664 | 4.0 |
| new-article_ctx-hybrid[tfidf+dense] | 0.470 | 0.702 | 0.804 | 0.599 | 0.569 | 0.233 | 0.838 | 0.674 | 0.680 | 7.6 |
| new-article_ctx-hybrid[bm25+tfidf+dense]-nodialect | 0.482 | 0.750 | 0.816 | 0.625 | 0.608 | 0.255 | 0.819 | 0.680 | 0.678 | 10.4 |
| new-article_ctx-hybrid[bm25+tfidf+dense+definition]-nodialect | 0.494 | 0.744 | 0.863 | 0.638 | 0.617 | 0.260 | 0.824 | 0.698 | 0.686 | 12.1 |
| new-article_ctx-hybrid[bm25+tfidf+dense]-nodialect-nostrip | 0.482 | 0.750 | 0.816 | 0.625 | 0.609 | 0.255 | 0.819 | 0.690 | 0.680 | 11.1 |

## Recall@5 by question style (all splits)

| config | MSA | Egyptian | English |
|---|---|---|---|
| reference-reference_article_ar-bm25-nodialect | 0.735 (n=161) | 0.211 (n=26) | 0.250 (n=4) |
| reference-reference_article_ar-tfidf-nodialect | 0.814 (n=161) | 0.173 (n=26) | 0.250 (n=4) |
| reference-reference_article_ar-dense-nodialect | 0.790 (n=161) | 0.269 (n=26) | 0.250 (n=4) |
| reference-reference_article_ar-hybrid[bm25+tfidf+dense]-nodialect | 0.790 (n=161) | 0.288 (n=26) | 0.250 (n=4) |
| new-article-bm25 | 0.775 (n=161) | 0.327 (n=26) | 0.250 (n=4) |
| new-article-tfidf | 0.815 (n=161) | 0.231 (n=26) | 0.250 (n=4) |
| new-article-dense | 0.801 (n=161) | 0.250 (n=26) | 0.250 (n=4) |
| new-article-hybrid[bm25+tfidf+dense] | 0.805 (n=161) | 0.308 (n=26) | 0.250 (n=4) |
| new-article_ctx-bm25 | 0.805 (n=161) | 0.308 (n=26) | 0.750 (n=4) |
| new-article_ctx-tfidf | 0.827 (n=161) | 0.269 (n=26) | 1.000 (n=4) |
| new-article_ctx-dense | 0.843 (n=161) | 0.308 (n=26) | 0.750 (n=4) |
| new-article_ctx-hybrid[bm25+tfidf+dense] | 0.849 (n=161) | 0.346 (n=26) | 1.000 (n=4) |
| new-paragraph-bm25 | 0.784 (n=161) | 0.327 (n=26) | 0.250 (n=4) |
| new-paragraph-tfidf | 0.843 (n=161) | 0.231 (n=26) | 0.250 (n=4) |
| new-paragraph-dense | 0.858 (n=161) | 0.269 (n=26) | 0.250 (n=4) |
| new-paragraph-hybrid[bm25+tfidf+dense] | 0.807 (n=161) | 0.308 (n=26) | 0.250 (n=4) |
| new-article_ctx-hybrid[bm25+tfidf] | 0.824 (n=161) | 0.308 (n=26) | 1.000 (n=4) |
| new-article_ctx-hybrid[bm25+dense] | 0.824 (n=161) | 0.308 (n=26) | 1.000 (n=4) |
| new-article_ctx-hybrid[tfidf+dense] | 0.861 (n=161) | 0.269 (n=26) | 0.750 (n=4) |
| new-article_ctx-hybrid[bm25+tfidf+dense]-nodialect | 0.849 (n=161) | 0.385 (n=26) | 1.000 (n=4) |
| new-article_ctx-hybrid[bm25+tfidf+dense+definition]-nodialect | 0.849 (n=161) | 0.385 (n=26) | 1.000 (n=4) |
| new-article_ctx-hybrid[bm25+tfidf+dense]-nodialect-nostrip | 0.855 (n=161) | 0.385 (n=26) | 0.750 (n=4) |

## Not run

- dense (neural: gemini-embedding-001 / multilingual-e5 / bge-m3): no network access to Gemini API / HuggingFace in the execution environment
- cross-encoder reranking (BAAI/bge-reranker-v2-m3): model download from HuggingFace blocked in the execution environment
