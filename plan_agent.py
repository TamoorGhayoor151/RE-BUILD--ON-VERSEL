"""
Recovery Plan Generator Agent
-------------------------------
Synthesizes Agents 1-3's outputs into a prioritized, actionable Markdown
report plus a JSON summary for the UI.

IMPORTANT: this agent is told the exact numbers to use (from value_agent.py)
and instructed never to invent or recompute them. It phrases and prioritizes;
it does not do math.
"""

import json
import os
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a construction site recovery planning consultant.
You will be given, as JSON, the full analysis of a waste stream: material classification,
recovery pathway assessments, and DETERMINISTIC financial/environmental calculations.

Write a prioritized, actionable Markdown recovery plan a site manager could execute today.

CRITICAL RULES:
- Never invent, recompute, or alter any dollar figure or CO2 number. Use ONLY the numbers
  given to you in the "value" data, verbatim.
- Prioritize actions by net_value (highest first), and flag any material with
  feasibility_score below 0.5 as requiring extra attention or possible landfill fallback.
- Structure: a short summary paragraph, a numbered action list (most valuable/urgent first),
  then a totals section restating total_net_value and total_co2_avoided_kg exactly as given.
- Keep it concise and practical — this is for a busy site manager, not a report for its own sake.
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.lower().startswith("markdown") or text.lower().startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else text
    return text.strip()


def generate_plan(
    classification: Dict[str, Any],
    recoverability: Dict[str, Any],
    value: Dict[str, Any],
    source_description: str = "",
    llm: Optional[ChatGroq] = None,
) -> Dict[str, Any]:
    """
    Combine all three prior agent outputs into a final Markdown + JSON plan.
    """
    if llm is None:
        llm = ChatGroq(model=MODEL_NAME, temperature=0.2)

    payload = {
        "source_description": source_description,
        "classification": classification,
        "recoverability": recoverability,
        "value": value,
    }

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload)),
        ]
    )
    markdown_plan = _strip_code_fences(response.content)

    # JSON summary for the UI — built from the deterministic data directly,
    # not re-derived from the LLM's prose, so the UI's headline numbers can
    # never drift from the source of truth.
    summary = {
        "total_net_value": value.get("total_net_value"),
        "total_co2_avoided_kg": value.get("total_co2_avoided_kg"),
        "num_materials": len(classification.get("materials", [])),
        "low_feasibility_flags": [
            a["material"] for a in recoverability.get("assessments", [])
            if a.get("feasibility_score", 1) < 0.5
        ],
    }

    return {
        "markdown_plan": markdown_plan,
        "summary": summary,
    }


if __name__ == "__main__":
    from value_agent import calculate_batch_value

    demo_classification = {
        "materials": [
            {"type": "concrete", "sub_material": "rubble", "est_percent": 85, "contamination": "low"},
            {"type": "steel", "sub_material": "rebar", "est_percent": 15, "contamination": "none"},
        ],
        "total_quantity": 2.5,
        "unit": "ton",
        "confidence": 0.87,
    }
    demo_recoverability = {
        "assessments": [
            {"material": "concrete", "pathway": "recycle_aggregate", "feasibility_score": 0.9, "constraints": []},
            {"material": "steel", "pathway": "recycle_scrap", "feasibility_score": 0.97, "constraints": []},
        ]
    }
    demo_value = calculate_batch_value(demo_classification["materials"], 2.5, "ton")
    print(json.dumps(generate_plan(demo_classification, demo_recoverability, demo_value), indent=2))
