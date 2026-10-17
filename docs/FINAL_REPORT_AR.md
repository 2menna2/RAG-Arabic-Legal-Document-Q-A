# التقرير النهائي — Egyptian Civil Code RAG (الإصدار الثاني)

> كل الأرقام هنا منقولة من ملفات مولَّدة آليًا داخل `reports/` (commit `c0e6e8d`). ما لم يُنفَّذ مذكور صراحةً في القسم 13.
> هذا النظام **ليس** استشارة قانونية، ومجموعة الأسئلة الذهبية **لم يراجعها محامٍ**.

---

## 1. ملخص العيوب المكتشفة في المشروع المرجعي

التفاصيل الكاملة والأدلة: `docs/AUDIT_REPORT.md`.

| الرمز | الخطورة | العيب |
|---|---|---|
| C1 | حرج | تلف حروف "لا" (lam-alef ligature) أثناء الاستخراج: 432 مادة بها كلمات مكسورة، و666 ظهورًا لـ "إال/اإل/األ" في المخرجات |
| C2 | حرج | الأرقام داخل النص العربي معكوسة (RTL) |
| C3 | حرج | المادة 452 مفقودة من الـ corpus |
| C4 | حرج | نصوص مواد مكتوبة يدويًا داخل الكود (1021، 1022، 279، 884، 901، 935)، ونص 1022 العربي غير موجود في المصدر |
| C5 | حرج | نص إلغاء مختلق للمواد الملغاة مع `source_page=1` |
| C7 | حرج | عند فشل الـ LLM يُرجَع جواب قانوني شكلي (fallback) بدل خطأ واضح |
| C8 | حرج | أرقام RAGAS المنشورة (0.94/0.915/0.885/0.92/0.915) لا تطابق ملف النتائج نفسه (المحسوب منه 0.80/0.64/0.43/0.45/0.58) |
| C9 | عالٍ | مقاييس heuristic مسمّاة RAGAS |
| C10 | حرج | الـ CI يولّد بيانات اصطناعية ليمرّ الـ quality gate |
| H1–H8 | عالٍ/متوسط | DVC remote على `D:/` و md5 غير قابل للتكرار؛ retrieval كثيف فقط؛ لا فحص توافق للـ index؛ guardrails بكلمات مفتاحية؛ لا `/ready`؛ قائمة latency بلا حد (تسرّب ذاكرة)؛ gauge ثابت 0.885؛ Docker ينسخ بيانات مستبعدة من git؛ اعتماديات غير مثبتة؛ coverage يستثني 12 module |
| اختبارات | – | 12 نجح، 1 فشل (`evaluate_gate`)، و`test_api`/`test_rag` فشل تجميعهما |

## 2. التغييرات المنفذة والملفات المعدلة

المشروع أُعيد بناؤه في `src/ecc_rag/` مع الاحتفاظ بالمرجع في `baselines/reference_pipeline/` للمقارنة.

| المجال | ما تغيّر | الملفات |
|---|---|---|
| استخراج PDF | استخراج الجداول بـ PyMuPDF على مستوى الحرف، إصلاح lam-alef، إعادة ترتيب الأرقام، إزالة الصفوف المكررة المتداخلة | `data_pipeline/pdf_extract.py` |
| تنظيف/تطبيع عربي | نص عرض نظيف + نص بحث مطبَّع + stemming خفيف | `data_pipeline/arabic.py` |
| تحليل المواد | 1149 مادة، عناوين هرمية (باب/فصل/...)، حالة إلغاء من المصدر فقط (`repealed_per_source`)، لا نصوص مكتوبة يدويًا | `parse_articles.py`, `schema.py` |
| عقد بيانات وتحقق | Pydantic schema صارم + فحص فجوات/تكرار/تلف/hash المصدر + عيوب المصدر الموثقة | `validate.py`, `configs/known_source_issues.yaml` |
| Chunking | 3 استراتيجيات (article / article_ctx / paragraph) قورنت؛ المختارة `article_ctx` | `chunking.py` |
| Retrieval | BM25 + char TF-IDF + dense (LSA) بدمج RRF، إزالة تكرار المادة، دعم "المادة N" و"مكرر"، فحص manifest للـ index | `rag/retriever.py`, `rag/index.py`, `rag/query.py`, `rag/vector_store.py`, `rag/embeddings.py` |
| القرار قبل الـ LLM | guardrails (prompt injection في السؤال وفي المستندات)، تصنيف النطاق (قوانين أخرى، غير قانوني، غير واضح)، بوابة كفاية الأدلة | `rag/guardrails.py`, `rag/scope.py`, `rag/evidence.py` |
| التوليد | Gemini عبر `google-genai` بـ JSON schema، أخطاء typed (unavailable/timeout/bad_output) بلا fallback، محقق اقتباسات + جولة إصلاح واحدة | `rag/llm.py`, `rag/citations.py`, `rag/pipeline.py`, `configs/prompts/v1.yaml` |
| API | FastAPI: `/api/v1/ask`, `/search`, `/articles/{n}`, `/info`, `/health`, `/ready`, `/metrics`؛ request IDs؛ rate limit؛ timeouts؛ CORS؛ أخطاء موحّدة | `api/main.py`, `api/schemas.py`, `api/ratelimit.py` |
| مراقبة | مقاييس Prometheus حقيقية، JSON logs مع تجزئة الأسئلة، 9 alerts، dashboard Grafana | `monitoring/`, `deploy/prometheus/`, `deploy/grafana/` |
| تقييم | 241 سؤالًا ذهبيًا (dev/test بـ hash ثابت)، benchmark لـ 22 إعدادًا، تقييم الـ gates، quality gate بحدود دنيا وانحدار، أداة RAGAS حقيقية (تتطلب مفتاحًا) | `evaluation/*`, `data/evaluation/golden_v1.yaml`, `configs/quality_gates.yaml` |
| MLOps | DVC pipeline من 8 مراحل قابل للتكرار، MLflow (23 run)، baseline مُرقّى | `dvc.yaml`, `dvc.lock`, `reports/` |
| نشر | Dockerfile متعدد المراحل non-root read-only، compose بـ profiles، GitHub Actions CI + release مع rollback | `docker/`, `docker-compose.yml`, `.github/workflows/` |
| اختبارات | unit/integration/e2e | `tests/` |
| توثيق | README + docs/* | `README.md`, `docs/` |

## 3. مخطط المعمارية

```
            ┌──────────── Offline (DVC pipeline) ────────────┐
Law.pdf ──► extract ──► parse ──► validate ──► chunk ──► index (Qdrant + BM25/TF-IDF + manifest)
                                                     └──► benchmark / evaluate ──► MLflow + reports/ ──► quality gate
            └────────────────────────────────────────────────┘
Online:
client ─► FastAPI (/api/v1/ask) ─► guardrails ─► retrieval (BM25+TF-IDF+LSA, RRF)
                                      │               │
                                rejected        scope ─► out_of_scope / needs_clarification
                                                      │
                                                evidence gate ─► abstained (insufficient_evidence)
                                                      │
                                                Gemini (JSON) ─► error (llm_unavailable/timeout/bad_output)
                                                      │
                                                citation validator (+1 repair) ─► abstained (citation_validation_failed)
                                                      │
                                                   answered + sources + disclaimer
        /metrics ─► Prometheus ─► Grafana / alerts          logs (JSON, hashed questions) ─► stdout
```
تفصيل أكبر: `docs/ARCHITECTURE.md`.

## 4. التشغيل المحلي

دليل تفصيلي بالعامية لكل خطوة (Windows/Linux، مفتاح Gemini، Google Drive، GitHub، Cloud Run): `docs/QUICKSTART_AR.md`.

```bash
make setup && source .venv/bin/activate         # uv sync --frozen (Python 3.12)
cp .env.example .env                            # ضع GEMINI_API_KEY
dvc remote add -d --local storage <remote> && dvc pull    # أو ضع Law.pdf في data/raw/ ثم: dvc repro index
make serve                                      # http://localhost:8000/docs
curl -s localhost:8000/ready
curl -s -X POST localhost:8000/api/v1/ask -H 'Content-Type: application/json' -d '{"question":"ما هو تعريف عقد البيع؟"}'
```

## 5. التشغيل بـ Docker

```bash
make docker-build TAG=$(git rev-parse --short HEAD)
docker run -d --read-only --tmpfs /tmp --memory 1g -p 127.0.0.1:8000:8000 --env-file .env ecc-rag-api:<sha>
make smoke URL=http://127.0.0.1:8000
docker compose up -d                              # الـ API فقط (index مدمج)
docker compose --profile qdrant up -d             # Qdrant server + indexer + api-qdrant
docker compose --profile monitoring up -d         # Prometheus + Grafana (يتطلب GRAFANA_ADMIN_PASSWORD)
docker compose --profile mlflow up -d
```

## 6. الاختبارات والتقييم

```bash
make lint typecheck        # ruff + mypy
make test                  # unit + integration + e2e
make eval                  # retrieval + gates على golden_v1 → reports/ + MLflow
make gate                  # quality gate مقابل الحدود والـ baseline
make benchmark             # 22 إعدادًا
make eval-llm              # يتطلب GEMINI_API_KEY (+ --judge --ragas)
scripts/demo_gate_catches_regression.sh   # يثبت أن الـ gate يرفض تغييرًا سيئًا
```

## 7. إعداد CI/CD

- `ci.yml` (كل PR/push): lint+mypy → unit tests (coverage ≥ 60%) → pip-audit → data-quality-gate (`dvc pull` → `dvc repro` → تحقق أن `dvc.lock` لم يتغير → integration/e2e → eval → gate) → Docker build بوسم SHA + Trivy + smoke داخل حاوية read-only.
- `release.yml` (tag `v*`): يعيد CI كاملًا → نشر الصورة على GHCR (SHA + version) → environment `production` بموافقة يدوية → Cloud Run revision بدون traffic → smoke بسؤال حقيقي → تحويل الـ traffic → rollback تلقائي عند الفشل.
- بدون أسرار DVC يُتخطى الـ data gate **بتحذير**، ولا تُولَّد بيانات اصطناعية أبدًا.
- التفاصيل والأسرار المطلوبة: `docs/CI_CD.md`.

## 8. النشر

الهدف الموصى به: Google Cloud Run (حاوية واحدة stateless، revisions للـ rollback، Secret Manager، نفس سحابة Gemini). الأوامر، الموارد المقاسة، إعداد Kubernetes البديل، وخطوات الـ rollback في `docs/DEPLOYMENT.md`. الإعداد الافتراضي `min-instances 0` (scale to zero) ليناسب الحصة المجانية، ويمكن تغييره بالمتغير `CLOUD_RUN_MIN_INSTANCES`. **لم يُنفَّذ نشر حقيقي.**

## 9. المراقبة والـ dashboard

- مقاييس: الطلبات/الأخطاء/الـ latency، نتائج الـ RAG حسب الحالة، أسباب الامتناع، أحداث الـ guardrails، latency وأخطاء وtokens الـ LLM، أخطاء الـ vector store، rate limiting، `ecc_ready`، معلومات الـ index، ونتائج التقييم offline (معنونة كـ offline).
- 9 قواعد alert (تحقق بـ `promtool`، وتم تشغيلها فعليًا على Prometheus حقيقي: `EccNotReady` و`EccHigh5xxRate` دخلتا حالة pending في الظروف المتوقعة).
- Dashboard: `deploy/grafana/dashboards/ecc_rag_ops.json` (13 لوحة). **لم يُتحقق من عرض Grafana بصريًا.**
- التفاصيل: `docs/MONITORING.md`.

## 10. نتائج الاختبارات الفعلية

| البند | النتيجة |
|---|---|
| pytest (unit + integration + e2e) | **126 passed, 1 skipped** (اختبار Gemini الحي يُتخطى لعدم وجود مفتاح) |
| coverage | 72% إجمالي، 68% لـ unit فقط (حد الـ CI 60%) |
| ruff / mypy | نظيف |
| actionlint (workflows) | نجح |
| promtool (alerts) | نجح |
| `dvc repro` على clone نظيف + `dvc pull` من remote محلي | نجح، والـ hashes متطابقة (articles.jsonl `8137e1c7…`) |
| Docker build + تشغيل read-only + healthcheck + smoke | نجح |
| load test (retrieval فقط، 2 vCPU، 1 GiB) | p50 15.5ms عند مستخدم واحد، ~80 req/s عند 4، بلا أخطاء ولا OOM حتى 32 مستخدمًا متزامنًا (قبل الإصلاح: OOM عند 16) |
| `.env.example` مع `docker --env-file` | كان بيكسر الحاوية (التعليقات بتدخل في القيم، والقيم الفاضية بتمسح `QDRANT_PATH`) → اتصلّح واتضافله اختباران، واتجرّب على الحاوية |
| demo الـ quality gate | الـ gate **رفض** إعداد BM25-فقط المتعمَّد إضعافه |

## 11. نتائج تقييم الـ RAG الفعلية

**Retrieval** (الإعداد المنتج، golden_v1، index حقيقي):

| split | n | R@1 | R@5 | R@10 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dev | 84 | 0.482 | 0.750 | 0.816 | 0.625 | 0.608 |
| test | 107 | 0.533 | 0.819 | 0.888 | 0.680 | 0.678 |

حسب الأسلوب: فصحى R@5 0.849 (n=161)، عامية مصرية **0.385** (n=26)، إنجليزي 1.000 (n=4 فقط).
أضعف نوع: أسئلة السيناريو R@5 0.375.

**طبقة القرار (بدون LLM):**

| split | accuracy | balanced acc. | false-proceed | false-block |
|---|---|---|---|---|
| dev | 0.981 | 0.995 | 0.000 | 0.024 |
| test | 0.941 | **0.763** | **0.241** | 0.000 |

الفرق بين dev و test يعني أن القواعد مضبوطة زيادة على dev؛ رقم test هو التقدير الصادق. الأسئلة السلبية التي تمر تصل إلى الـ LLM ثم محقق الاقتباسات كخطَّي دفاع إضافيين.

**التوليد (Gemini / RAGAS / جودة الاقتباس مع LLM): لم يُقَس** — لا مفتاح ولا وصول شبكي لـ Gemini في بيئة التنفيذ. لا توجد أرقام بديلة.

## 12. مقارنة الـ baseline بالنسخة المحسنة (نفس الأسئلة)

| الإعداد | dev R@5 | dev MRR | test R@5 | test MRR | test nDCG@5 |
|---|---|---|---|---|---|
| corpus المرجع + dense (LSA) | 0.643 | 0.494 | 0.759 | 0.633 | 0.625 |
| corpus المرجع + hybrid | 0.661 | 0.515 | 0.749 | 0.666 | 0.634 |
| corpus جديد، article + hybrid | 0.679 | 0.551 | 0.763 | 0.682 | 0.651 |
| **المختار: corpus جديد، article_ctx + hybrid** | **0.750** | **0.625** | **0.819** | 0.680 | **0.678** |

ملاحظات صادقة: الـ dense في المرجع كان embeddings عصبية لم يمكن تشغيلها هنا، فالمقارنة تستخدم نفس الـ LSA للطرفين (تقيس أثر الـ corpus والـ chunking والـ hybrid، لا النموذج). مكاسب جودة البيانات (0 كلمات تالفة بدل 432 مادة، المادة 452 موجودة، لا نصوص مختلقة) لا تظهر كاملةً في R@5 لكنها تؤثر مباشرة على صحة الاقتباسات. عامية مصرية: 0.288 → 0.385.

## 13. ما لم يُنفَّذ ولماذا

| البند | السبب |
|---|---|
| تقييم التوليد بـ Gemini، RAGAS، LLM-as-judge | لا `GEMINI_API_KEY` ولا وصول شبكي. الأدوات جاهزة (`make eval-llm`) و`llm_eval` floors فارغة حتى يُقاس أول baseline |
| embeddings عصبية (Gemini / sentence-transformers) و cross-encoder reranker | HuggingFace وPyTorch index وGemini محجوبة. الكود موجود ومعطَّل في الإعداد؛ الـ reranker يُفعَّل فقط بعد إثبات تحسن على dev |
| نشر حقيقي على Cloud Run وتشغيل الـ workflows على GitHub | لا حساب سحابي ولا repository ولا أسرار |
| Trivy و pip-audit محليًا | غير متاحين في البيئة؛ مُعدّان في الـ CI |
| عرض Grafana بصريًا | لم يمكن تشغيل Grafana في البيئة؛ الـ JSON صالح ومُعد للـ provisioning |
| مراجعة قانونية للأسئلة الذهبية | تتطلب محاميًا؛ كل البنود `review_status: unreviewed`، وتوجد أداة مراجعة (`evaluation/human_review.py`) |
| تحقق من سريان المواد بعد التعديلات التشريعية | المصدر نسخة واحدة؛ `legal_status=unverified` لكل مادة غير ملغاة صراحةً في المصدر |
| مصادقة (auth) على الـ API، tracing، Alertmanager routing | خارج نطاق ما يمكن التحقق منه هنا؛ موثّق كتوصيات |

## 14. المخاطر والقيود المتبقية

1. جودة التوليد والاقتباس مع Gemini غير مقاسة — الخطر الأكبر قبل الإنتاج.
2. الأسئلة العامية/السيناريو ضعيفة (R@5 ≈ 0.38): المستخدم العادي قد يحصل على امتناع أو مواد غير مناسبة.
3. طبقة القرار مضبوطة زيادة على dev (false-proceed 0.24 على test)؛ الفئات السلبية صغيرة (4–14 لكل فئة) فالفاصل الإحصائي واسع.
4. مجموعة الأسئلة لم يراجعها محامٍ؛ لا يجوز اعتبار الأرقام شهادة على صحة قانونية.
5. المصدر قد لا يعكس التعديلات اللاحقة؛ ولا يوجد تحقق من السريان.
6. محقق الاقتباسات يتحقق من الاقتباسات الحرفية وأرقام المواد، لكن الاستنتاجات المعاد صياغتها تُفحص heuristically فقط.
7. rate limit لكل replica؛ لا مصادقة — يجب وضع gateway أمام الخدمة.
8. 25 glyph خاص (private-use) غير مُعيَّن و405 token من حرف واحد محتمل أنها كلمات مقسومة في المصدر — موثقة في تقرير جودة البيانات.
9. حقوق استخدام وتوزيع ملف القانون وترجمته الإنجليزية لم تُؤكَّد (الترخيص Apache-2.0 للكود فقط).

## 15. ما يحتاج إلى إعداد يدوي

| # | البند | أين |
|---|---|---|
| 1 | `GEMINI_API_KEY` (الافتراضي `gemini-3.8-flash`؛ `gemini-2.5-flash` لم يعد متاحًا للمستخدمين الجدد) | `.env` محليًا؛ GCP Secret Manager باسم `gemini-api-key` للإنتاج |
| 2 | DVC remote حقيقي (Google Drive مجانًا، أو S3/GCS/Azure) + `dvc push` للبيانات | خطوات Google Drive في `docs/QUICKSTART_AR.md`؛ أسرار GitHub `DVC_REMOTE_URL` + `GDRIVE_CREDENTIALS_DATA` |
| 3 | GitHub repository + جعل `data-quality-gate` و`docker-build-scan-smoke` checks إجبارية على `main` | إعدادات الـ branch protection |
| 4 | GCP: project، Workload Identity Federation، service account للنشر وآخر للتشغيل | أسرار `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `GCP_PROJECT`, `GCP_REGION` |
| 5 | GitHub environment `production` بمراجعين مطلوبين | Settings → Environments |
| 6 | gateway/IAP/Cloud Armor أمام الخدمة (مصادقة + rate limit عام) | البنية التحتية |
| 7 | `GRAFANA_ADMIN_PASSWORD` وتوجيه الـ alerts (Alertmanager → Slack/email) | `.env` / Alertmanager |
| 8 | تشغيل `make eval-llm ARGS="--judge --ragas"` ثم تعيين حدود `llm_eval` في `configs/quality_gates.yaml` وترقية baseline | بعد توفر المفتاح |
| 9 | مراجعة محامٍ للأسئلة الذهبية وتحديث `review_status` | `data/evaluation/golden_v1.yaml` |
| 10 | تأكيد حقوق استخدام ملف القانون والترجمة | قانوني |
| 11 | budget alerts لاستهلاك Gemini | GCP Billing |
