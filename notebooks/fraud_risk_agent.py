"""
FraudRiskAgent: scores a single claim's fraud probability using the trained
model, and returns the top SHAP-driven reasons behind the score.
Place in notebooks/ (same dir as fraud_model.pkl, shap_explainer.pkl).
"""

import joblib
import pandas as pd

RISK_THRESHOLD = 0.5   # tune based on desired precision/recall tradeoff

class FraudRiskAgent:
    def __init__(self, model_path="fraud_model.pkl", explainer_path="shap_explainer.pkl"):
        self.model = joblib.load(model_path)
        self.explainer = joblib.load(explainer_path)

    def score(self, claim_features: pd.DataFrame) -> dict:
        """
        claim_features: single-row DataFrame, same columns/order as X_train.
        Returns fraud probability, risk flag, and top SHAP reasons.
        """
        proba = self.model.predict_proba(claim_features)[0][1]
        risk_flag = "high" if proba >= RISK_THRESHOLD else "low"

        shap_values = self.explainer.shap_values(claim_features)
        # LightGBM binary classifier -> shap_values is a list [class0, class1]
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        contributions = pd.Series(sv, index=claim_features.columns)
        top_reasons = contributions.abs().sort_values(ascending=False).head(5)
        reasons = [
            {
                "feature": feat,
                "value": claim_features.iloc[0][feat],
                "impact": round(float(contributions[feat]), 3),
            }
            for feat in top_reasons.index
        ]

        return {
            "fraud_probability": round(float(proba), 4),
            "risk_flag": risk_flag,
            "top_reasons": reasons,
        }


if __name__ == "__main__":
    # quick test using one row from X_test.csv
    agent = FraudRiskAgent()
    X_test = pd.read_csv("X_test.csv")
    sample = X_test.iloc[[0]]
    result = agent.score(sample)
    print(result)
