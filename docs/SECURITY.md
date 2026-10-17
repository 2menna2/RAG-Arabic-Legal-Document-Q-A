# Security, reliability and threat model

## Threat model (short)

| # | Threat | Vector | Controls in place | Residual risk |
|---|---|---|---|---|
| T1 | Prompt injection in the question | "ignore previous instructions", fake `<system>` tags, role switching | multilingual pattern + structural detection → `rejected`; system prompt treats input as data; output citation validation | novel phrasings evade patterns (tested: 9 attack forms detected, 4 legal look-alikes not flagged) |
| T2 | Injection inside retrieved documents | poisoned corpus / future uploads | same detector on retrieved text → document dropped; `<document>` wrapping with `<>` escaped; answers citing dropped docs fail validation (tested) | corpus is trusted today; no upload endpoint exists |
| T3 | Hallucinated law | invented articles, quotes, repeal status | evidence gate; model may abstain; citation validator (article ∈ context, verbatim quotes, no foreign "المادة N", repealed noticed); `article_not_in_corpus` for N ∉ 1..1149 and "مكرر" | paraphrased *inferred* claims are only heuristically checked |
| T4 | Misleading legal reliance | user treats output as advice | disclaimer on every answer; `legal_status=unverified`; sources and page numbers returned | inherent to the domain |
| T5 | Secret leakage | keys in Git/images/logs | `.env` git- and docker-ignored; secrets only via env/Secret Manager; image check in CI (`/app/.env` absent); log scrubbing | operator error |
| T6 | Abuse / cost exhaustion | request floods, very long inputs | per-IP token bucket (per replica), 1000-char limit, request timeout, LLM timeout + bounded retries | global limits need a gateway |
| T7 | Information disclosure via errors | stack traces | uniform error bodies, global exception handler, validation errors without internals | – |
| T8 | Privacy of questions | questions may contain personal data | `LOG_QUESTIONS=hashed` default; diagnostics only on request and without question text; no tracing export | Gemini receives the question (provider terms apply) |
| T9 | Supply chain | vulnerable deps / images | `uv.lock`, pip-audit + Trivy in CI, Dependabot, non-root read-only container, no shell needed | actions pinned by tag, not SHA |
| T10 | Unauthorised admin actions | – | no admin/eval endpoints exposed; evaluation runs offline | – |
| T11 | Stale or mismatched index | wrong corpus served | manifest hash checks at load → `/ready` 503; image builds its own index | – |

## Tests that back these claims

`tests/unit/test_guardrails_scope.py`, `test_citations.py`, `test_index_and_pipeline.py`
(document-injection drop, LLM failure → error, invalid citations → abstain, vector-store failure → error),
`test_api.py` (422/429/503/504, no traceback in bodies), `tests/e2e/test_api_e2e.py` (negative cases on the real corpus).

## Not guaranteed

The system is not proven secure. Not performed: penetration testing, red-teaming with an LLM attacker, Trivy/pip-audit
runs (configured in CI, not executed locally), authentication (the API has none — protect it at the gateway or keep the
Cloud Run service private).

## Operational recommendations

Keep the service private behind an authenticated gateway; set CORS explicitly; rotate `GEMINI_API_KEY`; set budget
alerts on the Gemini project; review `ecc_guardrail_events_total` and abstention reasons weekly.
