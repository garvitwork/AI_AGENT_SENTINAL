"""
IntakeAgent: takes a raw claim (dict, matching claims+policyholders columns)
and converts it into the exact feature schema the model expects
(same columns/order/encoding as X_train.csv).
Place in notebooks/.
"""

import joblib
import pandas as pd

# columns dropped during preprocessing (see preprocess.py)
DROP_COLS = [
    "claim_id", "policy_number", "incident_date", "policy_bind_date",
    "incident_location", "insured_zip",
    "fraud_probability", "risk_flag",
    "injury_claim", "property_claim", "vehicle_claim",
    "fraud_reported",
]

class IntakeAgent:
    def __init__(self, encoders_path="encoders.pkl", feature_order_path="feature_order.pkl"):
        self.encoders = joblib.load(encoders_path)
        self.feature_order = joblib.load(feature_order_path)

    def parse(self, raw_claim: dict) -> pd.DataFrame:
        """
        raw_claim: dict with raw claim + policyholder fields (as in MySQL row,
        merged claims+policyholders). Returns single-row DataFrame ready for
        FraudRiskAgent.score().
        """
        df = pd.DataFrame([raw_claim])

        # impute missing categoricals same as training
        for col in ["collision_type", "property_damage", "police_report_available", "authorities_contacted"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # encode categoricals using saved training-time encoders
        for col, le in self.encoders.items():
            if col in df.columns:
                # unseen category -> fallback to most frequent class seen in training
                df[col] = df[col].astype(str).map(
                    lambda v: v if v in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        # ensure exact column order model expects
        df = df.reindex(columns=self.feature_order, fill_value=0)
        return df


if __name__ == "__main__":
    # quick manual test claim
    sample_claim = {
        "incident_severity": "Major Damage",
        "incident_type": "Multi-vehicle Collision",
        "collision_type": "Rear Collision",
        "authorities_contacted": "Police",
        "incident_state": "OH",
        "incident_city": "Columbus",
        "incident_hour_of_the_day": 14,
        "number_of_vehicles_involved": 2,
        "property_damage": "YES",
        "bodily_injuries": 1,
        "witnesses": 2,
        "police_report_available": "YES",
        "total_claim_amount": 55000,
        "auto_make": "Honda",
        "auto_model": "Civic",
        "auto_year": 2015,
        "months_as_customer": 100,
        "age": 40,
        "policy_state": "OH",
        "policy_csl": "250/500",
        "policy_deductable": 1000,
        "policy_annual_premium": 1400.0,
        "umbrella_limit": 0,
        "insured_sex": "MALE",
        "insured_education_level": "MD",
        "insured_occupation": "craft-repair",
        "insured_hobbies": "reading",
        "insured_relationship": "husband",
        "capital_gains": 0,
        "capital_loss": 0,
    }
    agent = IntakeAgent()
    features = agent.parse(sample_claim)
    print(features)
