"""
Uses Groq's free-tier API to turn SHAP factors into a natural-language
adjuster note. Get a free key at https://console.groq.com/keys
Add to .env: GROQ_API_KEY=your_key_here
Place in notebooks/.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-20b"  # free tier (llama-3.1-8b-instant was deprecated)

def draft_adjuster_note(claim_id: str, risk_result: dict) -> str:
    proba = risk_result["fraud_probability"]
    flag = risk_result["risk_flag"]
    reasons = risk_result["top_reasons"]

    factors_text = "\n".join(
        f"- {r['feature']} = {r['value']} (SHAP impact {r['impact']:+.3f}, "
        f"{'raises' if r['impact'] > 0 else 'lowers'} fraud risk)"
        for r in reasons
    )

    prompt = f"""You are an insurance claims adjuster writing an internal note.

Claim ID: {claim_id}
Fraud probability: {proba} ({flag} risk)
Top model-driven factors:
{factors_text}

Write a concise 3-4 sentence professional adjuster note explaining the risk
assessment and recommended action (approve if low risk, escalate for manual
review if high risk). Plain prose only, no markdown, no headers, no bullet
points, no bold text. Be factual, no speculation beyond the given factors."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
            reasoning_effort="low",
        )
        note = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason

        # retry once if response looks cut off/incomplete
        if len(note) < 50 or finish_reason not in ("stop", None):
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
                reasoning_effort="low",
            )
            note = response.choices[0].message.content.strip()

        if len(note) < 50:
            raise ValueError(f"LLM returned incomplete note: '{note}'")

        return note
    except Exception as e:
        # fallback to plain templated reasoning if API fails
        return (
            f"[LLM unavailable: {e}] Fraud probability: {proba}. Key factors: "
            + "; ".join(f"{r['feature']}={r['value']}" for r in reasons)
        )


if __name__ == "__main__":
    fake_result = {
        "fraud_probability": 0.72,
        "risk_flag": "high",
        "top_reasons": [
            {"feature": "incident_severity", "value": 3.0, "impact": 0.5},
            {"feature": "total_claim_amount", "value": 90000, "impact": 0.3},
        ],
    }
    print(draft_adjuster_note("CLM-00664", fake_result))
