"""
Loads data/processed/*.csv into MySQL (claim_sentinel db).
Run schema.sql first. Adjust column mapping once actual Kaggle
dataset columns are confirmed.
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = quote_plus(os.getenv("DB_PASS"))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "claim_sentinel")

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

def load_table(csv_path, table_name):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into {table_name}")

if __name__ == "__main__":
    load_table("data/processed/policyholders.csv", "policyholders")
    load_table("data/processed/claims.csv", "claims")
    print("Done. Next: python db/vehicle_enrich.py")
