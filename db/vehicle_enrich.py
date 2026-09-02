"""
Cross-checks each claim's (auto_make, auto_model, auto_year) against NHTSA's
official vPIC catalog. Free, no key: https://vpic.nhtsa.dot.gov/api/
"""

import os
import difflib
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = quote_plus(os.getenv("DB_PASS"))
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")
DB_SSL_CA = os.getenv("DB_SSL_CA")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={"ssl": {"ca": DB_SSL_CA}} if DB_SSL_CA else {},
)

MAKE_FIXES = {"suburu": "subaru", "accura": "acura"}

NHTSA_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/{make}/modelyear/{year}?format=json"

def best_match_score(model: str, official_models: list) -> float:
    if not official_models:
        return 0.0
    scores = [difflib.SequenceMatcher(None, model.lower(), m.lower()).ratio() for m in official_models]
    return max(scores)

def check_model(make: str, model: str, year: int) -> tuple[bool, float]:
    clean_make = MAKE_FIXES.get(make.lower(), make)
    resp = requests.get(NHTSA_URL.format(make=clean_make, year=year), timeout=10)
    resp.raise_for_status()
    results = resp.json().get("Results", [])
    official_models = [r["Model_Name"] for r in results]
    score = best_match_score(model, official_models)
    return score >= 0.6, round(score, 2)

def enrich_vehicle_checks():
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE vehicle_check"))

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT auto_make, auto_model, auto_year FROM claims"
        )).fetchall()

    for make, model, year in rows:
        try:
            exists, score = check_model(make, model, year)
            pd.DataFrame([{
                "auto_make": make, "auto_model": model, "auto_year": year,
                "model_exists": int(exists),
            }]).to_sql("vehicle_check", engine, if_exists="append", index=False)
            print(f"{make} {model} {year}: exists={exists} (match={score})")
        except Exception as e:
            print(f"Failed for {make} {model} {year}: {e}")

if __name__ == "__main__":
    enrich_vehicle_checks()
    os.makedirs("data", exist_ok=True)
    with open("data/.enrich_marker", "w") as f:
        f.write("enriched")
