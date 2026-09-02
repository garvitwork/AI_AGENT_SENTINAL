"""
Preprocess claims data before model training:
- impute missing values
- encode categoricals
- train/test split
Run from notebooks/ folder (same dir as eda.ipynb).
"""

import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

def load_data():
    claims = pd.read_sql("SELECT * FROM claims", engine)
    policyholders = pd.read_sql("SELECT * FROM policyholders", engine)
    df = claims.merge(policyholders, on="policy_number")
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # impute missing categoricals as "Unknown" instead of dropping rows
    for col in ["collision_type", "property_damage", "police_report_available", "authorities_contacted"]:
        df[col] = df[col].fillna("Unknown")

    # drop redundant/leaky columns
    drop_cols = [
        "claim_id", "policy_number", "incident_date", "policy_bind_date",
        "incident_location", "insured_zip",
        "fraud_probability", "risk_flag",   # filled later by model, not features
        "injury_claim", "property_claim", "vehicle_claim",  # sum to total_claim_amount
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    target = df["fraud_reported"].map({"Y": 1, "N": 0})
    features = df.drop(columns=["fraud_reported"])

    # encode remaining categoricals
    cat_cols = features.select_dtypes(include="object").columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        encoders[col] = le

    return features, target, encoders

if __name__ == "__main__":
    df = load_data()
    X, y, encoders = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean():.3f}, Test fraud rate: {y_test.mean():.3f}")

    X_train.to_csv("notebooks/X_train.csv", index=False)
    X_test.to_csv("notebooks/X_test.csv", index=False)
    y_train.to_csv("notebooks/y_train.csv", index=False)
    y_test.to_csv("notebooks/y_test.csv", index=False)

    joblib.dump(encoders, "notebooks/encoders.pkl")
    joblib.dump(list(X.columns), "notebooks/feature_order.pkl")
    print("Saved encoders.pkl and feature_order.pkl")
