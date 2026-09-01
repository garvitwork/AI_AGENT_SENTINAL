"""
AdjudicationAgent: takes FraudRiskAgent's output, decides approve/escalate,
writes a SHAP-based reasoning note to adjuster_decisions + audit_log.
Place in notebooks/.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from dotenv import load_dotenv
from llm_reasoning import draft_adjuster_note

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = quote_plus(os.getenv("DB_PASS"))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "claim_sentinel")
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")


class AdjudicationAgent:
    def decide(self, claim_id: str, risk_result: dict) -> dict:
        risk_flag = risk_result["risk_flag"]
        proba = risk_result["fraud_probability"]
        reasons = risk_result["top_reasons"]

        decision = "escalated" if risk_flag == "high" else "approved"

        reasoning = draft_adjuster_note(claim_id, risk_result)

        self._log_decision(claim_id, decision, reasoning)
        self._log_audit(claim_id, "AdjudicationAgent", f"Claim {decision}", reasoning)

        return {"claim_id": claim_id, "decision": decision, "reasoning": reasoning}

    def _log_decision(self, claim_id, decision, reasoning):
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO adjuster_decisions (claim_id, decision, reasoning, decided_by)
                    VALUES (:claim_id, :decision, :reasoning, 'agent')
                """),
                {"claim_id": claim_id, "decision": decision, "reasoning": reasoning},
            )

    def _log_audit(self, claim_id, agent_name, action, details):
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO audit_log (claim_id, agent_name, action, details)
                    VALUES (:claim_id, :agent_name, :action, :details)
                """),
                {"claim_id": claim_id, "agent_name": agent_name, "action": action, "details": details},
            )


if __name__ == "__main__":
    # test with a fake risk result
    fake_result = {
        "fraud_probability": 0.72,
        "risk_flag": "high",
        "top_reasons": [
            {"feature": "incident_severity", "value": 3.0, "impact": 0.5},
            {"feature": "total_claim_amount", "value": 90000, "impact": 0.3},
        ],
    }
    agent = AdjudicationAgent()
    result = agent.decide("CLM-00664", fake_result)
    print(result)
