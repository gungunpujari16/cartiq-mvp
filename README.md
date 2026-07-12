# CartIQ MVP

The Phase 1 product build described in `CartIQ_PRD_v1.0.docx` / `CartIQ_TRD_v1.0.docx`: a JS snippet that
scores shopper purchase-intent in real time, a discount engine that intervenes on low-intent sessions,
and a multi-tenant analytics dashboard -- built with a lightweight stack that runs on a laptop and deploys
for free/cheap, instead of the TRD's funded-startup infrastructure (Kafka/Cassandra/Snowflake/Kubernetes).

`IPBL/` (Phase 0, the market-research dashboard) is untouched by this build.

## Architecture

| TRD layer | TRD tech | This build |
|---|---|---|
| Capture | Vanilla JS snippet | `demo-store/cartiq.js` -- same job (session id, batched events every 500ms, zero PII, consent check) |
| Ingestion | FastAPI + API Gateway + Kafka | FastAPI only; events write straight to the DB (no queue -- the one deliberate simplification at this scale) |
| Intelligence | XGBoost + Spark/MLflow/SageMaker retraining | XGBoost, trained offline by `backend/ml/train_model.py` instead of a scheduled pipeline |
| Storage | Cassandra + Snowflake + Redis | One Postgres/SQLite DB, `brand_id` on every table = logical tenant isolation (same idea as a partition key, not physically separate infra) |
| Presentation | React + Plotly + Streamlit | Streamlit + Plotly |

Full design rationale: see the plan this was built from, or ask Claude Code to re-derive it from
`backend/app/*.py` docstrings -- every file explains which TRD section it maps to and what was simplified.

### Model quality (current training run)

XGBoost purchase-probability classifier, trained on `backend/ml/data/ecommerce_cleaned.csv`
(a copy of the Phase 0 dataset, duplicated in so deploys don't depend on a sibling folder outside this repo):

| Metric | Result | TRD S3.1 target |
|---|---|---|
| AUC-ROC | 0.77 | > 0.75 ✅ |
| Recall @ 0.40 | 0.72 | > 0.55 ✅ |
| Precision @ 0.40 | 0.31 | > 0.60 ❌ |

Precision is below the TRD's target -- `scale_pos_weight` is tuned for recall (catching more true converters)
at the cost of more false positives on this small, synthetic dataset. Worth knowing before treating the score
as production-grade; retraining on real demo-store/brand traffic (with the 3 features historical data lacks --
`scroll_depth_avg`, `exit_intent_count`, `payment_attempts`, already captured live by the snippet) would be the
next step to close this gap.

## Local setup

Requires Python 3.12. **If you hit `OSError: [WinError 32]` creating a venv inside a OneDrive-synced folder**,
create the venv somewhere OneDrive doesn't sync (e.g. `C:\venvs\...`) and point commands at that path instead --
OneDrive actively locks files it's scanning, which races with pip mid-install.

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements.txt        # Windows
# source venv/bin/activate && pip install -r requirements.txt   # Mac/Linux

python ml\train_model.py          # trains model.json from ml/data/ecommerce_cleaned.csv
python ml\seed_demo_data.py       # creates a demo brand + API key, replays historical sessions
                                   # -> prints the API key, also saved to backend/demo_api_key.txt

venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Sanity check: `curl http://127.0.0.1:8000/v1/health`

### 2. Demo storefront

```bash
cd demo-store
cp config.example.js config.js   # gitignored -- paste your real API key in here, never commit it
python -m http.server 5500
```

Edit `config.js` and paste the key printed by `seed_demo_data.py` (or from `backend/demo_api_key.txt`).
Open `http://localhost:5500`, accept the CartIQ consent banner, click into a product, add it to cart,
check out. Watch the score badge (top-right) and any discount toast appear live.

### 3. Dashboard

```bash
cd dashboard
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python generate_brand_survey.py   # writes data/brand_survey.csv (simulated B2B survey data)
venv\Scripts\python -m streamlit run app.py
```

It auto-fills the API key from `backend/demo_api_key.txt` if present. Open `http://localhost:8501`.
This is a two-page app (see `dashboard/pages/`):

- **🛍️ Shopper Insights** -- the actual product dashboard (PRD Feature 4): what a brand sees about
  its own shoppers. Live XGBoost purchase-intent scoring, K-Means behavioral segments, funnel/channel
  diagnostics, discount A/B lift, plus an interpretable Decision Tree comparison and Apriori
  association rules for teaching purposes. Every chart is followed by a data-driven insight box and,
  where a model produced it, a methodology box explaining which algorithm, why it was chosen over
  alternatives, and what it calculates (phrased to match BRD Section 4).
- **🧠 Business Intelligence** -- CartIQ's *own* go-to-market analytics (BRD Section 4/7): will
  another brand buy CartIQ, and how much would they pay. Random Forest (adoption), Ridge Regression
  (willingness-to-pay), K-Means (brand segments: Premium/Growth/Budget), Apriori (what drives high
  WTP), Decision Tree (company-size inference). Runs entirely on a **simulated** B2B survey dataset
  (`dashboard/generate_brand_survey.py`, N=220) -- no real survey has been run yet, and the page says
  so. This is a different dataset and audience than Shopper Insights; see `dashboard/bi_models.py`
  for the model code and `dashboard/app.py` for the plain-English explanation of the split.

### Tests

```bash
cd backend
venv\Scripts\python -m pytest tests/ -v
```

Covers the discount rule engine (threshold, cart-value minimum, one-per-session cap, 10% holdout) and
event-schema validation (enum rejection, PII query-string stripping). Not full TRD-spec coverage (80%
across all modules) -- scoped to the logic most likely to silently break.

## What "test locally" actually covered

Verified end-to-end during this build: event ingestion → live scoring → discount rule engine → checkout/purchase
→ dashboard reflecting new sessions, plus multi-tenant isolation (a second brand's API key sees zero rows from
the first brand's data, and a 403 if it tries to query the other brand's `brand_id` directly).

## Deployment

Free/cheap tiers throughout. Ask when you're ready to actually execute these -- they're written here so you
know the shape of the work in advance.

| Component | Where | Notes |
|---|---|---|
| Backend + API | [Render](https://render.com) (free Web Service) | Build: `pip install -r requirements.txt`. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Set `DATABASE_URL` env var (see below) and `CORS_ORIGINS` to your deployed demo-store origin. |
| Database | [Neon](https://neon.tech) (free Postgres) | Create a project, copy the connection string into Render's `DATABASE_URL`. Same SQLAlchemy code -- no migration needed, `init_db()` creates tables on first boot. |
| Dashboard | [Streamlit Community Cloud](https://share.streamlit.io) | Point at `dashboard/app.py`. Add an `API_BASE_URL` secret set to your Render URL (small code tweak to read it as the sidebar default instead of localhost). |
| Demo store + snippet | [Vercel](https://vercel.com) or GitHub Pages (static) | Deploy the `demo-store/` folder as-is. Update `config.js`'s `apiBaseUrl` to the Render URL, and add that Vercel/Pages origin to the backend's `CORS_ORIGINS`. |

Steps once you're ready:
1. Push this repo to GitHub (`cartiq-mvp/` at minimum; can stay in the same repo as `IPBL/`).
2. Render: New Web Service → point at the repo, root directory `cartiq-mvp/backend` → set env vars → deploy.
3. Neon: New project → copy connection string → paste into Render's `DATABASE_URL` → redeploy backend.
4. Run `python ml/seed_demo_data.py` once against the deployed backend (or locally with `DATABASE_URL` pointed at Neon) to get a real API key for the deployed brand.
5. Streamlit Cloud: New app → repo, root `cartiq-mvp/dashboard`, main file `app.py` → add secret.
6. Vercel/Pages: deploy `demo-store/` → update `config.js` → update backend `CORS_ORIGINS` → redeploy backend.
