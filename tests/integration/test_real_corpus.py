"""Integration tests on the REAL corpus (versioned by DVC). Skipped if absent — never replaced by synthetic data."""

import json

import pytest
import yaml

from ecc_rag.config import PROJECT_ROOT
from tests.conftest import REAL_DATA, require_real_artifacts

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def articles():
    require_real_artifacts()
    return {r["article_number"]: r for r in map(json.loads, REAL_DATA.read_text(encoding="utf-8").splitlines())}


def test_corpus_passes_validation(articles):
    from ecc_rag.data_pipeline.validate import validate

    known = yaml.safe_load((PROJECT_ROOT / "configs/known_source_issues.yaml").read_text(encoding="utf-8"))
    rep = validate([articles[k] for k in sorted(articles)], known, PROJECT_ROOT / "data/raw/Law.pdf")
    assert rep["passed"], rep["errors"][:5]
    assert rep["stats"]["ligature_artefact_tokens"] == 0


def test_corpus_shape_and_known_facts(articles):
    assert sorted(articles) == list(range(1, 1150))
    rep = [n for n, a in articles.items() if a["legal_status"] == "repealed_per_source"]
    assert rep == list(range(54, 81)) + list(range(389, 418))
    assert "فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين" in articles[147]["article_text"]
    assert "إحدى وعشرون سنة ميلادية كاملة" in articles[44]["article_text"]
    assert articles[452]["article_text"].startswith("(١) تسقط بالتقادم دعوى الضمان")
    assert "ar_text_missing" in articles[1022]["quality_flags"]
    assert "٣٤٨ لسنة ١٩٥٦" in articles[60]["status_evidence"]["note_ar"]
    assert articles[1085]["article_text"].count("ولا يجوز للدائن بعد موت المدين") == 1


def test_index_is_compatible_with_corpus():
    require_real_artifacts()
    from ecc_rag.config import load_rag_config
    from ecc_rag.rag.index import load_index

    li = load_index(PROJECT_ROOT / "data/processed/chunks.jsonl", REAL_DATA, PROJECT_ROOT / "data/index", load_rag_config())
    assert li.manifest["n_points"] == len(li.chunks) == 1149


@pytest.fixture(scope="module")
def pipe():
    require_real_artifacts()
    from ecc_rag.evaluation.gates_eval import build_offline_pipeline

    return build_offline_pipeline()


@pytest.mark.parametrize(
    "q,article",
    [
        ("ماذا تنص المادة 163؟", 163),
        ("حارس الحيوان مسئول عن الضرر الذي يحدثه الحيوان", 176),
        ("الشفعة رخصة تجيز الحلول محل المشتري في بيع العقار", 935),
        ("Sale is a contract whereby the vendor undertakes to transfer ownership for a price in money", 418),
    ],
)
def test_known_passages_are_retrieved(pipe, q, article):
    _q, hits = pipe.search(q, 5)
    assert article in [h.article_number for h in hits]


def test_decision_mirror_matches_pipeline(pipe):
    """gates_eval._decide (used for fast calibration) must agree with the real pipeline."""
    from ecc_rag.evaluation.dataset import load_golden
    from ecc_rag.evaluation.gates_eval import _decide, _features, predicted_outcome

    _v, items = load_golden()
    for it in items:
        f = _features(pipe, it)
        assert _decide(it.question, f, pipe.cfg["scope"], pipe.cfg["evidence"]) == predicted_outcome(pipe, it.question)[0], it.id


def test_quality_gate_passes_on_committed_results():
    from ecc_rag.evaluation.quality_gate import run_gate

    if not (PROJECT_ROOT / "reports/retrieval_eval.json").exists():
        pytest.skip("run `make eval` first")
    result = run_gate(PROJECT_ROOT / "configs/quality_gates.yaml")
    assert result["passed"], result["failures"]
