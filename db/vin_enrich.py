"""
Enriches claims with live vehicle data from NHTSA vPIC (free, no key).
Decodes each claim's VIN and stores results in vehicle_info table.
Docs: https://vpic.nhtsa.dot.gov/api/
"""

import os
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "claim_sentinel")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

NHTSA_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json"

def decode_vin(vin: str) -> dict:
    resp = requests.get(NHTSA_URL.format(vin=vin), timeout=10)
    resp.raise_for_status()
    result = resp.json()["Results"][0]
    return {
        "vin": vin,
        "make": result.get("Make"),
        "model": result.get("Model"),
        "model_year": result.get("ModelYear"),
        "body_class": result.get("BodyClass"),
        "engine_cylinders": result.get("EngineCylinders"),
        "fuel_type": result.get("FuelTypePrimary"),
        "plant_country": result.get("PlantCountry"),
    }

def enrich_claims():
    with engine.connect() as conn:
        vins = conn.execute(text("SELECT DISTINCT vin FROM claims WHERE vin IS NOT NULL")).fetchall()

    for (vin,) in vins:
        try:
            data = decode_vin(vin)
            pd.DataFrame([data]).to_sql("vehicle_info", engine, if_exists="append", index=False)
            print(f"Enriched {vin}: {data['make']} {data['model']} {data['model_year']}")
        except Exception as e:
            print(f"Failed for {vin}: {e}")

if __name__ == "__main__":
    enrich_claims()
