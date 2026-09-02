"""
Loads data/processed/*.csv into MySQL (Aiven). Clears tables first for
idempotent DVC reruns. Requires ca.pem in project root (see .env DB_SSL_CA).
"""

import os
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

def load_table(csv_path, table_name):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into {table_name}")

def clear_tables():
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        conn.execute(text("TRUNCATE TABLE audit_log"))
        conn.execute(text("TRUNCATE TABLE adjuster_decisions"))
        conn.execute(text("TRUNCATE TABLE vehicle_check"))
        conn.execute(text("TRUNCATE TABLE claims"))
        conn.execute(text("TRUNCATE TABLE policyholders"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

if __name__ == "__main__":
    clear_tables()
    load_table("data/processed/policyholders.csv", "policyholders")
    load_table("data/processed/claims.csv", "claims")
    os.makedirs("data", exist_ok=True)
    with open("data/.load_marker", "w") as f:
        f.write("loaded")
    print("Done. Next: python db/vehicle_enrich.py")
