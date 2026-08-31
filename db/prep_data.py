"""
Splits data/raw/insurance_claims.xlsx into:
  - data/processed/policyholders.csv
  - data/processed/claims.csv
Cleans '?' placeholders -> NULL, generates claim_id.
Run once before db/seed.py.
"""

import pandas as pd
import numpy as np

RAW = "data/raw/insurance_claims.xlsx"
OUT_DIR = "data/processed"

df = pd.read_excel(RAW)
df = df.replace("?", np.nan)

# --- policyholders ---
policyholder_cols = [
    "policy_number", "months_as_customer", "age", "policy_bind_date",
    "policy_state", "policy_csl", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "insured_zip", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship",
    "capital-gains", "capital-loss",
]
policyholders = df[policyholder_cols].drop_duplicates(subset="policy_number").copy()
policyholders = policyholders.rename(columns={
    "capital-gains": "capital_gains",
    "capital-loss": "capital_loss",
})
policyholders.to_csv(f"{OUT_DIR}/policyholders.csv", index=False)

# --- claims ---
claims = df.drop(columns=[c for c in policyholder_cols if c != "policy_number"]).copy()
claims.insert(0, "claim_id", ["CLM-" + str(i + 1).zfill(5) for i in range(len(claims))])
claims.to_csv(f"{OUT_DIR}/claims.csv", index=False)

print(f"policyholders: {policyholders.shape}")
print(f"claims: {claims.shape}")
