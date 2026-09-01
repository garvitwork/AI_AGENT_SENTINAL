"""
Orchestrates the full claim pipeline using LangGraph:
IntakeAgent -> FraudRiskAgent -> AdjudicationAgent
Place in notebooks/. Run: python orchestrator.py
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from intake_agent import IntakeAgent
from fraud_risk_agent import FraudRiskAgent
from adjudication_agent import AdjudicationAgent

intake_agent = IntakeAgent()
fraud_agent = FraudRiskAgent()
adjudication_agent = AdjudicationAgent()


class ClaimState(TypedDict):
    claim_id: str
    raw_claim: dict
    features: Optional[object]
    risk_result: Optional[dict]
    decision: Optional[dict]


def intake_node(state: ClaimState) -> ClaimState:
    state["features"] = intake_agent.parse(state["raw_claim"])
    return state


def fraud_risk_node(state: ClaimState) -> ClaimState:
    state["risk_result"] = fraud_agent.score(state["features"])
    return state


def adjudication_node(state: ClaimState) -> ClaimState:
    state["decision"] = adjudication_agent.decide(state["claim_id"], state["risk_result"])
    return state


def build_graph():
    graph = StateGraph(ClaimState)
    graph.add_node("intake", intake_node)
    graph.add_node("fraud_risk", fraud_risk_node)
    graph.add_node("adjudication", adjudication_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "fraud_risk")
    graph.add_edge("fraud_risk", "adjudication")
    graph.add_edge("adjudication", END)

    return graph.compile()


if __name__ == "__main__":
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

    app = build_graph()
    result = app.invoke({
        "claim_id": "CLM-00664",   # must exist in claims table (FK constraint)
        "raw_claim": sample_claim,
        "features": None,
        "risk_result": None,
        "decision": None,
    })
    print(result["decision"])
