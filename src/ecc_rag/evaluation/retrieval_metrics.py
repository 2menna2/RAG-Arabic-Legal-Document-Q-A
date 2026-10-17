"""Retrieval metrics (article level).

Ground truth comes from the golden dataset:
* ``expected_articles``  — must-have articles (relevance grade 2)
* ``acceptable_articles`` — also relevant (grade 1)

Definitions (K = cut-off, ranked = article numbers in rank order, de-duplicated):

* Recall@K    = |expected ∩ top-K| / |expected|
* Hit@K       = 1 if any expected article is in top-K
* Precision@K = |(expected ∪ acceptable) ∩ top-K| / K
* MRR         = 1 / rank of the first *expected* article (0 if none in the list)
* nDCG@K      = DCG@K / IDCG@K with graded gains (2 expected, 1 acceptable) and log2 discount
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[int], expected: Sequence[int], k: int) -> float:
    if not expected:
        raise ValueError("recall undefined without expected articles")
    return len(set(ranked[:k]) & set(expected)) / len(set(expected))


def hit_at_k(ranked: Sequence[int], expected: Sequence[int], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(expected) else 0.0


def precision_at_k(ranked: Sequence[int], expected: Sequence[int], acceptable: Sequence[int], k: int) -> float:
    relevant = set(expected) | set(acceptable)
    return len(set(ranked[:k]) & relevant) / k


def mrr(ranked: Sequence[int], expected: Sequence[int]) -> float:
    exp = set(expected)
    for i, a in enumerate(ranked, start=1):
        if a in exp:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], expected: Sequence[int], acceptable: Sequence[int], k: int) -> float:
    gains = {a: 1.0 for a in acceptable}
    gains.update({a: 2.0 for a in expected})
    dcg = sum(gains.get(a, 0.0) / math.log2(i + 1) for i, a in enumerate(ranked[:k], start=1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def all_metrics(
    ranked: Sequence[int], expected: Sequence[int], acceptable: Sequence[int], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    out: dict[str, float] = {"mrr": mrr(ranked, expected)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, expected, k)
        out[f"hit@{k}"] = hit_at_k(ranked, expected, k)
        out[f"precision@{k}"] = precision_at_k(ranked, expected, acceptable, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, expected, acceptable, k)
    return out
