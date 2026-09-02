# Claim-Sentinel

**Autonomous Insurance Claims Fraud Detection & Auto-Adjudication Agent System**

Live API: https://ai-agent-sentinal.onrender.com
Frontend: `index.html` / `style.css` / `script.js` (static, deployable anywhere)

---

## 1. Problem Statement

Insurance is a regulated, high-friction vertical that's underserved by generic AI portfolio projects. Most student/portfolio work in fraud detection stops at "train a classifier and report accuracy." Real insurance workflows need more:

- Claims must be **explained**, not just scored — regulators and adjusters need to know *why* a claim was flagged.
- Decisions must be **auditable** — every action needs a traceable record.
- Low-risk claims should be **auto-approved** to reduce manual workload; high-risk claims should be **escalated** with a clear, written justification.
- The system should behave like an **agentic pipeline**, not a single script: intake → risk scoring → adjudication, each a distinct responsibility.

Claim-Sentinel was built to demonstrate this full loop — data infrastructure → ML model → explainability → autonomous agent pipeline → deployment — end to end, not just the ML slice.

---

## 2. Proposed Workflow (Build Order)

The project was deliberately sequenced so each layer could be verified before building the next:

1. **Data infrastructure** — static raw data → cleaned/split processed CSVs → relational schema in MySQL.
2. **Live enrichment** — cross-check claimed vehicle info against a real external API (fraud signal).
3. **ML layer** — trained fraud classifier + SHAP explainability, tracked via MLflow.
4. **Agentic layer** — three-agent pipeline (Intake → FraudRisk → Adjudication) orchestrated with LangGraph.
5. **LLM reasoning layer** — SHAP factors turned into a natural-language adjuster note via a free LLM API.
6. **Pipeline reproducibility** — the whole thing wired into a DVC pipeline with params, tracked in DagsHub.
7. **Deployment** — API on Render, database moved to a cloud-hosted MySQL instance, static frontend on top.

This mirrors the same architecture pattern used in a companion project, SCOUT (supply chain disruption early-warning system): **data infra → ML model → agentic layer → deployment**.

---

## 3. What Was Actually Followed

The build order above was followed closely, with a few adjustments made along the way as real constraints surfaced (see Challenges section):

- Started from a Kaggle-style auto insurance claims dataset (`insurance_claims`, ~1000 rows, `fraud_reported` label) instead of a synthetic dataset, for realism.
- Split it into `policyholders` and `claims` tables reflecting an actual insurance schema (not a single flat table).
- Replaced a planned VIN-decode enrichment step with a **make/model/year existence check** against NHTSA's vPIC API once it became clear the dataset had no VIN column — a working pivot rather than forcing the original plan.
- Added **fuzzy matching** (`difflib`) to the NHTSA check after discovering the dataset itself contains typos (`Suburu`→Subaru, `Accura`→Acura, `Ultima`→Altima) that made exact matching useless as a fraud signal.
- Trained a LightGBM classifier with class-weight balancing for the 75/25 fraud imbalance, plus a SHAP TreeExplainer for per-claim explainability.
- Built three agents (Intake, FraudRisk, Adjudication) as separate, testable Python classes before wiring them together.
- Chose **LangGraph** over CrewAI for orchestration — simpler state graph model, sufficient for a linear pipeline, no added complexity for a 3-node chain.
- Added an LLM step (Groq, free tier) so adjudication notes read as real written justifications instead of templated string concatenation.
- Wired the whole thing into a **DVC pipeline** (`dvc.yaml` + `params.yaml`) with 5 stages, tracked via MLflow hosted on DagsHub, so the ML side is versioned and reproducible, not just scripted.
- Exposed the pipeline via **FastAPI**, deployed on **Render's free tier**, backed by a cloud-hosted **MySQL on Aiven** (critical — Render cannot reach a `localhost` database).
- Built a static HTML/CSS/JS frontend (deliberately no framework, no Streamlit) styled as a claims-desk "case file" — a rubber-stamp verdict, SHAP factors as signed bars, and the LLM-written adjuster note.

---

## 4. Tech Stack

| Layer | Technology |
|---|---|
| **Database** | **MySQL, cloud-hosted on Aiven** (free tier, SSL-required connection) — this is the system of record for claims, policyholders, decisions, and the audit trail. Not local/localhost in production. |
| **Data processing** | Python, pandas, SQLAlchemy, PyMySQL |
| **ML model** | LightGBM (fraud classifier, class-weight balanced) |
| **Explainability** | SHAP (`TreeExplainer`) |
| **Live enrichment** | NHTSA vPIC API (free, no key) + `difflib` fuzzy matching |
| **Experiment tracking** | MLflow, hosted via DagsHub |
| **Pipeline / versioning** | DVC (data + model versioning, 5-stage reproducible pipeline), DagsHub as DVC remote |
| **Agent orchestration** | LangGraph (3-node state graph: Intake → FraudRisk → Adjudication) |
| **LLM reasoning** | Groq API, free tier (`openai/gpt-oss-20b`) — generates the written adjuster note from SHAP factors |
| **API** | FastAPI, deployed on Render (free tier) |
| **Frontend** | Static HTML / CSS / JS (no framework), calling the FastAPI endpoint directly |
| **Version control** | Git/GitHub (code), DVC + DagsHub (data & model artifacts) |

---

## 5. Architecture

```
insurance_claims.xlsx (raw)
        │
   prep_data.py  →  policyholders.csv, claims.csv
        │
    seed_db (MySQL/Aiven)  →  policyholders, claims tables
        │
  vehicle_enrich.py  →  NHTSA vPIC fuzzy-match  →  vehicle_check table
        │
  preprocess.py  →  encoded train/test splits, encoders.pkl, feature_order.pkl
        │
  train_model.py  →  LightGBM + SHAP  →  fraud_model.pkl, shap_explainer.pkl
        │            (tracked via MLflow on DagsHub, metrics + confusion matrix)
        │
 ┌──────────────────────────────────────────────┐
 │            LangGraph pipeline                 │
 │  IntakeAgent → FraudRiskAgent → AdjudicationAgent │
 │  (parses claim)   (scores + SHAP)   (decides + LLM note → writes to     │
 │                                       adjuster_decisions + audit_log)   │
 └──────────────────────────────────────────────┘
        │
   FastAPI (/submit-claim)  →  deployed on Render
        │
   Static frontend (index.html/style.css/script.js)
```

---

## 6. Database Design

MySQL schema (hosted on Aiven, not local):

- `policyholders` — customer/policy attributes (age, state, premium, occupation, etc.)
- `claims` — incident details, vehicle info, claim amounts, `fraud_reported` label, plus `fraud_probability`/`risk_flag` columns filled in by the model at inference time
- `vehicle_check` — NHTSA-verified make/model/year existence, used as a fraud signal
- `adjuster_decisions` — every automated decision (approved/escalated) with the LLM-written reasoning
- `audit_log` — full agent-by-agent trace of every action taken on a claim, for compliance-style traceability

Foreign keys enforce referential integrity (claims → policyholders, decisions/audit → claims), which surfaced real integrity issues during development (see Challenges).

---

## 7. ML Model

- **Algorithm**: LightGBM classifier, `class_weight="balanced"` to handle the ~75/25 non-fraud/fraud imbalance
- **Explainability**: SHAP `TreeExplainer`, top-5 factors returned per prediction with signed impact values
- **Metrics** (800/200 stratified split): ROC-AUC ≈ 0.83, fraud recall ≈ 0.59, fraud precision ≈ 0.56
- **Tracking**: every training run logs params, metrics, the model artifact, and a confusion matrix plot to MLflow (hosted on DagsHub), so runs are comparable over time
- **Reproducibility**: hyperparameters live in `params.yaml`, not hardcoded, so they're tracked and can be swept without code changes

---

## 8. Agent Pipeline

Three single-responsibility agents, orchestrated as a LangGraph state graph:

1. **IntakeAgent** — takes a raw claim (matching the DB schema), imputes missing categorical fields, encodes them using the training-time encoders, and reindexes to the exact feature order the model expects.
2. **FraudRiskAgent** — loads the trained model + SHAP explainer, scores the claim, and returns the fraud probability, a risk flag (`low`/`high`), and the top 5 SHAP-driven factors.
3. **AdjudicationAgent** — takes the risk result, decides `approved` or `escalated`, calls the LLM to draft a natural-language adjuster note from the SHAP factors, and writes both the decision and an audit log entry to MySQL.

This mirrors a real claims desk: intake → risk assessment → adjudication with a written rationale, fully automated but fully explainable.

---

## 9. LLM Reasoning Layer

SHAP values are numeric and not directly readable by a human adjuster. A Groq-hosted LLM (`openai/gpt-oss-20b`, free tier) converts the top factors into a 3–4 sentence professional adjuster note — factual, tied to the given SHAP factors, no speculation. A retry-and-fallback mechanism guards against truncated or empty LLM responses (see Challenges).

---

## 10. Frontend

A static HTML/CSS/JS console (no framework, no Streamlit) styled as a claims-desk case file:

- **Left panel**: full claim intake form matching the API's schema, with a "Load sample claim" shortcut for quick testing.
- **Right panel**: the "case file" — a rotated rubber-stamp verdict (APPROVED/ESCALATED), a fraud-probability bar, SHAP factors rendered as signed bars, and the LLM-written adjuster note.
- **Wake API button**: since Render's free tier spins down on inactivity, the UI polls `/health` every 4s (up to 90s) so the person testing it gets clear feedback instead of a silent failure.
- Talks directly to the deployed FastAPI backend via `fetch`; CORS is enabled on the API for this.

---

## 11. Deployment

- **API**: FastAPI app deployed on **Render's free tier**. Root directory set to `notebooks/`; `requirements.txt` and Python version pin live at the repo root (Render reads `runtime.txt`-style pins via a `PYTHON_VERSION` env var, not a file, on this platform).
- **Database**: originally local MySQL — **moved to Aiven's free-tier cloud MySQL** because Render cannot reach a `localhost` database. This was a required, non-optional change for a working deployment, not just an upgrade.
- **Model artifacts**: `fraud_model.pkl`, `shap_explainer.pkl`, `encoders.pkl`, `feature_order.pkl` are committed directly to git (after DVC-remote pull reliability issues on Render's build step — see Challenges) so the API has them on cold start without depending on an external pull at build time.
- **Frontend**: static files, deployable to any static host (GitHub Pages, Netlify, etc.) or opened locally; calls the Render API URL directly.

---

## 12. Challenges Faced & How They Were Solved

**Password with `@` broke the DB connection string**
SQLAlchemy connection URLs parsed the `@` in the MySQL password as a delimiter, silently connecting to the wrong host. Fixed by wrapping the password with `urllib.parse.quote_plus()` before building the connection string — applied consistently across every script that connects to MySQL.

**Dataset typos broke the NHTSA fraud signal**
Exact-match checks against NHTSA's vehicle catalog returned false negatives for legitimate vehicles because the dataset itself has typos (`Suburu`, `Accura`, `Ultima`) and formatting differences (`F150` vs `F-150`). Solved with `difflib.SequenceMatcher` fuzzy matching plus a small manual fix-up table for known recurring typos, so the signal reflects real mismatches rather than data-entry noise.

**LLM returned empty/truncated adjuster notes**
The first Groq model used (`llama-3.1-8b-instant`) was deprecated mid-project; the replacement reasoning model (`openai/gpt-oss-20b`) silently burned its token budget on internal reasoning, leaving nothing for the actual answer. Solved by raising `max_tokens`, setting `reasoning_effort="low"`, and adding a length-check-and-retry so an incomplete response triggers one automatic retry before falling back to a templated note.

**DVC/git double-tracking conflicts**
Processed CSVs and model `.pkl` files ended up tracked by both git and DVC simultaneously, which DVC refuses to reproduce over. Solved by explicitly untracking those paths from git (`git rm --cached`) and letting DVC own them, with `.gitignore` updated to prevent recurrence.

**Foreign key constraints blocked table resets**
Clearing tables for a clean pipeline rerun failed on FK violations (`claims` referenced by `adjuster_decisions`/`audit_log`; `policyholders` referenced by `claims`). Solved by deleting in correct dependency order (children first) and, in the seed script, wrapping the reset in `SET FOREIGN_KEY_CHECKS=0/1` so `dvc repro` is idempotent and safe to rerun from scratch.

**DVC push/pull reliability on Render's free tier**
DVC's push to the DagsHub remote reported "everything up to date" while Render's build-time `dvc pull` failed with missing cache files — a cache/lock desync that persisted across multiple `dvc commit -f` / `dvc repro -f` attempts. Given repeated failures with diminishing returns, the pragmatic fix was to **commit the four model artifacts directly to git** instead of relying on a DVC pull inside the Render build step, removing that failure point entirely while keeping DVC for local reproducibility and DagsHub for experiment tracking.

**scikit-learn version mismatch broke unpickling in production**
The model was trained locally against one scikit-learn version; Render's `pip install` resolved a newer one, and the SHAP explainer's internal numba-compiled state couldn't unpickle across versions (`TypeError: code() argument 13 must be str, not int`). Solved by pinning exact package versions (`scikit-learn`, `shap`, `lightgbm`, `numba`) in `requirements.txt` matching the local training environment, then retraining and recommitting the artifacts.

**Local MySQL is unreachable from Render**
Render (like any external host) cannot reach a database running on `localhost`. This required migrating to a cloud-hosted MySQL provider — **Aiven** was chosen (free tier, SSL-enforced, no shady free-hosting sites) over alternatives like Railway. This meant updating every connection string across the codebase to support a non-default port and an SSL CA certificate (`connect_args={"ssl": {"ca": ...}}`), and re-running the seed/enrichment scripts against the new host.

**CORS blocked the frontend from calling the API**
The deployed API rejected cross-origin requests from the static frontend by default. Solved by adding `CORSMiddleware` to the FastAPI app with an open origin policy appropriate for a public portfolio demo.

**Render free-tier cold starts**
The free instance spins down after inactivity, so the first request after a period of idleness can take up to a minute and initially looks like a failure. Solved on the frontend with a "Wake API" button that polls `/health` on a retry loop with clear status feedback, rather than leaving the person guessing whether the demo is broken.

---

## 13. Repo Structure

```
AI_AGENT_SENTINAL/
├── data/
│   ├── raw/                  # insurance_claims.xlsx (static source)
│   └── processed/             # DVC-tracked cleaned CSVs
├── db/
│   ├── schema.sql
│   ├── prep_data.py
│   ├── seed.py
│   └── vehicle_enrich.py
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocess.py
│   ├── train_model.py
│   ├── intake_agent.py
│   ├── fraud_risk_agent.py
│   ├── adjudication_agent.py
│   ├── llm_reasoning.py
│   ├── orchestrator.py
│   └── api.py
├── outputs/                   # risk_metrics.json, confusion_matrix.png (DVC-tracked)
├── index.html / style.css / script.js   # frontend
├── dvc.yaml / dvc.lock / params.yaml
├── requirements.txt
├── ca.pem                     # Aiven SSL cert
└── .env                       # DB + API credentials (not committed)
```

---

## 14. Possible Next Steps

- Batch/bulk claim submission endpoint
- Model retraining trigger tied to drift monitoring (data already versioned via DVC/MLflow to support this)
- Role-based auth on the API before any real-world use
- Graph-based collision detection across claimants (originally scoped, not yet built)
