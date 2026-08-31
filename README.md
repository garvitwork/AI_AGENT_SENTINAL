# Claim-Sentinel

Autonomous insurance claims fraud & auto-adjudication agent.

## Current stage: data + storage (pre-ML)

## Data sources
- **Static base data**: `insurance_claims` dataset — 1000 auto claims, 39 features, `fraud_reported` label (in `data/raw/insurance_claims.xlsx`).
- **Live enrichment**: NHTSA vPIC API (free, no key) — for each claim's `(auto_make, auto_model, auto_year)`, checks whether that model actually exists in NHTSA's official catalog. A mismatch is a fraud signal (fabricated/mistyped vehicle). `db/vehicle_enrich.py` does this.

## Setup
1. `pip install -r requirements.txt`
2. `python db/prep_data.py` — splits raw xlsx into `data/processed/policyholders.csv` and `claims.csv`
3. Run `db/schema.sql` in MySQL to create tables
4. Copy `.env.example` -> `.env` and fill in DB creds
5. `python db/seed.py` to load data into MySQL
6. `python db/vehicle_enrich.py` to live-check each vehicle against NHTSA
7. Explore in `notebooks/eda.ipynb`

## Next
ML fraud classifier (XGBoost/LightGBM) + SHAP, then agent pipeline.
