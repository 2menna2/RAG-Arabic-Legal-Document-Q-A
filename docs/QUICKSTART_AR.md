# التشغيل خطوة بخطوة

الدليل ده بيمشي من الأسهل للأصعب. **المراحل 1–4 كفاية عشان المشروع يشتغل على جهازك**، والباقي اختياري.

> اتجرّب على Linux. الأوامر مكتوبة لـ Windows (PowerShell) و Linux/macOS. لو قابلتك مشكلة غريبة على Windows،
> جرّب WSL2 (Ubuntu) وكمّل بأوامر Linux.

---

## المرحلة 0 — البرامج المطلوبة (مرة واحدة)

| البرنامج | لازم؟ | التثبيت |
|---|---|---|
| Git | أيوه | https://git-scm.com/downloads |
| uv (بيثبت Python 3.12 لوحده) | أيوه | Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` — Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | لو هتشغّل بـ Docker بس | https://docs.docker.com/get-docker/ |

بعد التثبيت اقفل الـ terminal وافتحه تاني، واتأكد: `git --version` و `uv --version`.

**Windows فقط:** شغّل الأمر ده مرة واحدة عشان Python يطبع العربي من غير أخطاء encoding:
```powershell
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")   # وبعدين افتح terminal جديد
```

---

## المرحلة 1 — تجهيز المشروع

```bash
# فك الضغط وادخل الفولدر
cd egyptian-civil-code-rag

# الـ zip مفيهوش تاريخ git، و DVC محتاج repo
git init
git add -A
git commit -m "Initial import"

# تثبيت المكتبات بالنسخ المقفولة في uv.lock بالظبط
uv sync --frozen --extra dev --extra pipeline --extra mlops
```

## المرحلة 2 — ملف القانون وبناء البيانات والفهرس

1. حط ملف القانون اللي معاك بالاسم ده بالظبط: `data/raw/Law.pdf`
2. اتأكد إنه نفس الملف (لازم الـ MD5 يطلع `5086ef5f16f15d856278db943e58e948`):
   ```powershell
   Get-FileHash data\raw\Law.pdf -Algorithm MD5        # Windows
   ```
   ```bash
   md5sum data/raw/Law.pdf                              # Linux  (macOS: md5 data/raw/Law.pdf)
   ```
   لو طلع مختلف، الخطوات هتشتغل برضه بس فحص الجودة هيعترض على الـ hash، لأن المشروع مضبوط على النسخة دي.
3. ابني كل حاجة (استخراج → تحليل → فحص جودة → تقطيع → فهرس). بياخد دقيقة أو اتنين:
   ```bash
   uv run dvc repro index
   ```
   هيتعمل `data/processed/articles.jsonl` (1149 مادة) و `data/index/`.
   بعدها شغّل فحص جودة البيانات بشكل صريح. DVC بيتخطاه لو البيانات طالعة مطابقة للي اتفحص قبل كده:
   ```bash
   uv run dvc repro --force --single-item validate
   ```
   لازم يطبع `validation PASSED` ومعاها `records: 1149` و `ligature_artefact_tokens: 0`.
4. شغّل الاختبارات:
   ```bash
   uv run pytest -q
   ```
   النتيجة المتوقعة: `119 passed, 1 skipped` (اللي اتخطى ده اختبار Gemini الحي، وهيشتغل لما تحط المفتاح).

## المرحلة 3 — مفتاح Gemini (مجاني)

1. ادخل https://aistudio.google.com واعمل **Get API key → Create API key**.
2. انسخ ملف الإعدادات وحط المفتاح:
   ```powershell
   Copy-Item .env.example .env      # Windows
   ```
   ```bash
   cp .env.example .env             # Linux/macOS
   ```
   افتح `.env` واكتب المفتاح في السطر ده: `GEMINI_API_KEY=المفتاح_بتاعك`
   (ملف `.env` مستبعد من git ومن Docker، فمش هيترفع في أي حتة.)

> **تنبيه الخصوصية:** في الخطة المجانية ممكن جوجل تستخدم الطلبات لتحسين خدماتها. ماتستخدمش الخطة المجانية
> مع أسئلة مستخدمين حقيقيين فيها بيانات شخصية.

## المرحلة 4 — تشغيل الـ API وتجربته

```bash
uv run uvicorn ecc_rag.api.main:app --host 127.0.0.1 --port 8000
```

افتح في المتصفح **http://127.0.0.1:8000/docs**. هتلاقي واجهة تقدر تجرب منها كل الـ endpoints.

- `GET /ready` → لازم يرجّع `"ready": true` (لو فيه false، الرد بيقول السبب).
- `POST /api/v1/ask` بالـ body ده:
  ```json
  {"question": "ما هو تعريف عقد البيع؟"}
  ```
  أو من terminal تاني:
  ```bash
  curl -s -X POST http://127.0.0.1:8000/api/v1/ask -H "Content-Type: application/json" -d "{\"question\": \"ما هو تعريف عقد البيع؟\"}"
  ```

الرد بيبقى فيه `status`:

| status | المعنى |
|---|---|
| `answered` | إجابة مع المواد المستشهد بيها وأرقام الصفحات |
| `abstained` | الأدلة مش كفاية، أو الاقتباسات فشلت في التحقق (السبب في `reason`) |
| `out_of_scope` | السؤال عن قانون تاني (جنائي، عمل، ...) أو مش سؤال قانوني |
| `needs_clarification` | السؤال غامض |
| `rejected` | محاولة prompt injection |
| `error` | Gemini مش متاح أو اتأخر (HTTP 503/504) — **مش** إجابة |

لو `/ask` رجّع 503 بـ `llm_unavailable`: يا المفتاح غلط/مش موجود في `.env`، يا عدّيت حد الطلبات بتاع الخطة المجانية (استنى دقيقة وجرب تاني).

**مبروك — لحد هنا المشروع شغال.** المراحل الجاية اختيارية.

---

## المرحلة 5 (اختياري) — التقييم و MLflow

```bash
uv run python -m ecc_rag.evaluation.run_eval --mlflow-uri sqlite:///mlflow.db   # retrieval + طبقة القرار
uv run python -m ecc_rag.evaluation.quality_gate                                # لازم يطبع QUALITY GATE: PASSED
uv run python -m ecc_rag.evaluation.rag_eval --split test                       # تقييم إجابات Gemini فعليًا
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000            # http://127.0.0.1:5000
```

`rag_eval` هو التقييم اللي ماتعملش لعدم وجود مفتاح. لما تشغّله:
- خلي بالك من حدود الخطة المجانية (فيه 136 سؤال في الـ test split، وللتجربة الأول ضيف `--limit 20`).
- النتائج بتطلع في `reports/`.
- بعدها حط الحدود الدنيا في `configs/quality_gates.yaml` تحت `llm_eval`.

## المرحلة 6 (اختياري) — التشغيل بـ Docker

المرحلة 2 لازم تكون اتعملت الأول، لأن الـ image بتاخد `data/processed/`.

```bash
docker build -f docker/Dockerfile -t ecc-rag-api:local .
docker run -d --name ecc --read-only --tmpfs /tmp -p 127.0.0.1:8000:8000 --env-file .env ecc-rag-api:local
curl -s http://127.0.0.1:8000/ready
docker logs ecc            # لو فيه مشكلة
docker rm -f ecc           # للإيقاف
```

أو بـ compose:
- `docker compose up -d` بيشغّل الـ API.
- `docker compose --profile monitoring up -d` بيضيف Prometheus و Grafana (حط `GRAFANA_ADMIN_PASSWORD` في `.env` الأول) → Grafana على http://127.0.0.1:3000.

## المرحلة 7 (اختياري) — Google Drive كمخزن DVC مجاني

محتاجه بس لو عايز تشارك البيانات مع حد تاني، أو تخلي الـ CI على GitHub يشغّل اختبارات البيانات الحقيقية.
ملف القانون نفسه **مايترفعش على git**.

> الخطوات دي حسب توثيق DVC، ومااتجربتش على Google Drive حقيقي من بيئة البناء.

إضافة Google Drive لـ DVC بيسحب مكتبات قديمة (`cryptography` قديمة)، عشان كده بنشغّله في بيئة معزولة بـ `uvx`
ومش بنضيفه للمشروع:
```bash
uvx --from "dvc==3.67.1" --with "dvc-gdrive==3.0.1" dvc version     # اتأكد إن gdrive ظاهر في Supports
```

**أ) فولدر على Drive:** اعمل فولدر فاضي (مثلًا `ecc-rag-dvc`) وافتحه. الـ ID هو الجزء الأخير من اللينك
`https://drive.google.com/drive/folders/<FOLDER_ID>`.

**ب) OAuth client خاص بيك** (الـ client الافتراضي بتاع DVC جوجل ممكن تمنعه):
1. https://console.cloud.google.com → اعمل project جديد.
2. **APIs & Services → Library → Google Drive API → Enable**.
3. **OAuth consent screen**: نوع External، وضيف إيميلك كـ Test user.
4. **Credentials → Create credentials → OAuth client ID** ونوعه **Desktop app**، وانسخ الـ Client ID والـ Secret.

**ج) ربط DVC ورفع البيانات:**
```bash
uv run dvc remote add -d gdrive gdrive://<FOLDER_ID>
uv run dvc remote modify --local gdrive gdrive_client_id     "<CLIENT_ID>"
uv run dvc remote modify --local gdrive gdrive_client_secret "<CLIENT_SECRET>"
git add .dvc/config && git commit -m "DVC remote: Google Drive"
uvx --from "dvc==3.67.1" --with "dvc-gdrive==3.0.1" dvc push       # أول مرة هيفتح المتصفح عشان تسجّل دخول
```
`--local` معناها إن الـ ID والـ Secret بيتحفظوا في `.dvc/config.local`، وده مش بيترفع على git.

أي حد معاه صلاحية على الفولدر يقدر بعد كده يعمل `git clone` وبعدها `uvx ... dvc pull`، وبعدين `uv run dvc repro index`.

**د) للـ CI على GitHub (service account):**
1. في نفس الـ project: **IAM & Admin → Service Accounts → Create**. بعدها **Keys → Add key → JSON** (هينزل ملف).
2. اعمل Share لفولدر Drive مع إيميل الـ service account (`...@...iam.gserviceaccount.com`) بصلاحية **Viewer**.
3. في GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**:
   - `DVC_REMOTE_URL` = `gdrive://<FOLDER_ID>`
   - `GDRIVE_CREDENTIALS_DATA` = محتوى ملف الـ JSON كله
4. امسح ملف الـ JSON من جهازك بعد ما تحطه في GitHub.

## المرحلة 8 (اختياري) — GitHub و CI

```bash
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```

- أول push هيشغّل `ci.yml`. من غير أسرار DVC، المهام اللي محتاجة البيانات بتتخطى **بتحذير واضح** ومش بتتنجح بالغش.
- في الـ repo الخاص، GitHub Actions ليها دقايق مجانية محدودة كل شهر.
- **Settings → Branches → Add rule على `main`** → اختار *Require status checks* وحدد `data-quality-gate` و `docker-build-scan-smoke`.

## المرحلة 9 (اختياري) — النشر على Cloud Run

محتاج حساب Google Cloud **بفوترة مفعّلة (كارت)**، حتى لو هتفضل جوه الحصة المجانية. الخطوات كاملة في
`docs/DEPLOYMENT.md` §4. الملخص:
1. Secret Manager: اعمل secret اسمه `gemini-api-key`.
2. Workload Identity Federation لـ GitHub، وأسرار `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `GCP_PROJECT`, `GCP_REGION`.
3. GitHub environment اسمه `production` بمراجعين.
4. `git tag v1.0.0 && git push --tags` → بعد موافقتك بينشر.

الإعداد الافتراضي `min-instances 0`، يعني ماتدفعش وهو مش شغال، بس أول طلب بعد فترة سكون بياخد وقت أطول
(بدء الحاوية لحد ما تبقى جاهزة كان ~7.5 ثانية على 1 vCPU محليًا، غير وقت سحب الـ image). لو عايز نسخة شغالة
طول الوقت، اعمل repository variable اسمه `CLOUD_RUN_MIN_INSTANCES=1` (ودي بتتحاسب).

---

## مشاكل شائعة

| العَرَض | الحل |
|---|---|
| `dvc repro` بيقول `not a git repository` | اعمل خطوات `git init` اللي في المرحلة 1 |
| `data/raw/Law.pdf` not found | الملف لازم يكون في المكان ده وبالاسم ده بالظبط |
| `/ready` → `index manifest missing` | ماعملتش `uv run dvc repro index` |
| `/ready` → `corpus_hash differs` | البيانات اتغيرت بعد ما الفهرس اتبنى → `uv run dvc repro index` |
| `UnicodeEncodeError` على Windows | `PYTHONUTF8=1` (المرحلة 0) |
| `/ask` → 503 `llm_unavailable` | المفتاح ناقص أو غلط، أو عدّيت حد الخطة المجانية |
| `/ask` → 504 `llm_timeout` | Gemini اتأخر. زوّد `LLM_TIMEOUT_S` في `.env` لو الشبكة بطيئة |
| 429 من الـ API نفسه | حد `RATE_LIMIT_PER_MINUTE` (30 في الدقيقة لكل IP). غيّره في `.env` |
| Docker: `/ready` false | `docker logs ecc`. واتأكد إن `data/processed/` موجود قبل الـ build |
