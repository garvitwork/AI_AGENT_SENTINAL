"""
FastAPI wrapper around the orchestrator pipeline.
Place in notebooks/. Run: uvicorn api:app --reload
Docs at http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from orchestrator import build_graph

app = FastAPI(title="Claim-Sentinel API")
pipeline = build_graph()


class ClaimRequest(BaseModel):
    claim_id: str
    incident_severity: str
    incident_type: str
    collision_type: Optional[str] = None
    authorities_contacted: Optional[str] = None
    incident_state: str
    incident_city: str
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    property_damage: Optional[str] = None
    bodily_injuries: int
    witnesses: int
    police_report_available: Optional[str] = None
    total_claim_amount: float
    auto_make: str
    auto_model: str
    auto_year: int
    months_as_customer: int
    age: int
    policy_state: str
    policy_csl: str
    policy_deductable: int
    policy_annual_premium: float
    umbrella_limit: int
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: str
    insured_relationship: str
    capital_gains: int
    capital_loss: int


@app.post("/submit-claim")
def submit_claim(claim: ClaimRequest):
    raw = claim.model_dump()
    claim_id = raw.pop("claim_id")

    try:
        result = pipeline.invoke({
            "claim_id": claim_id,
            "raw_claim": raw,
            "features": None,
            "risk_result": None,
            "decision": None,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "claim_id": claim_id,
        "risk_result": result["risk_result"],
        "decision": result["decision"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
