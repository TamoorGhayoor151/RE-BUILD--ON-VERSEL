"""
Recoverability Assessment Agent
---------------------------------
Takes Agent 1's classification output and, for each material, determines
the recovery pathway. Grounded in data/recovery_rules.json (a transparent,
auditable local knowledge base) rather than letting the LLM invent
feasibility scores from nothing.

The LLM's job here is narrow: read the retrieved rule + the specific
contamination/context for this material, and produce a short, situation-
aware note. The feasibility_score and pathway always come from the rules
file unless the LLM has a well-justified reason to adjust — and even then,
adjustments are logged, not silently substituted.
"""

import json
import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_DATA_DIR = os.path.dirname(__file__)
_RULES_PATH = os.path.join(_DATA_DIR, "recovery_rules.json")

SYSTEM_PROMPT = """You are a construction waste recovery pathway advisor.
You will be given ONE material, its contamination level, and the BASE recovery rule
from a vetted local knowledge base (pathway, feasibility_score, constraints, notes).

Your job: write a short (1-2 sentence) situational note explaining how this specific
material's contamination level affects the base recovery pathway, and whether the base
feasibility_score should be adjusted (small nudge only, e.g. +/-0.1 max, and only if
contamination is medium or high).

Respond with ONLY valid JSON, no markdown fences:
{
  "situational_note": "<1-2 sentences>",
  "adjusted_feasibility_score": <number, usually equal to base score unless contamination warrants a nudge>
}
"""


def load_recovery_rules(path: str = _RULES_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def assess_material(
    material_type: str,
    contamination: str,
    rules: Dict[str, Any],
    llm: Optional[ChatGroq] = None,
) -> Dict[str, Any]:
    """
    Assess a single material's recovery pathway. Base data always comes from
    the local rules file; the LLM only adds a situational note and, at most,
    a small bounded adjustment to feasibility.
    """
    key = material_type.lower().strip().replace(" ", "_")
    base_rule = rules.get(key, rules.get("mixed_debris"))
    used_fallback = key not in rules

    if llm is None:
        llm = ChatGroq(model=MODEL_NAME, temperature=0)

    context = {
        "material": material_type,
        "contamination": contamination,
        "base_rule": base_rule,
    }

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(context)),
        ]
    )
    raw = _strip_code_fences(response.content)

    try:
        parsed = json.loads(raw)
        adjusted_score = float(parsed.get("adjusted_feasibility_score", base_rule["feasibility_score"]))
        # Guardrail: never let the LLM swing the score by more than 0.15 from the base rule
        base_score = base_rule["feasibility_score"]
        if abs(adjusted_score - base_score) > 0.15:
            adjusted_score = base_score
        note = parsed.get("situational_note", "")
    except (json.JSONDecodeError, Exception):
        adjusted_score = base_rule["feasibility_score"]
        note = base_rule.get("notes", "")

    return {
        "material": material_type,
        "matched_key": key if not used_fallback else "mixed_debris",
        "used_fallback_rule": used_fallback,
        "pathway": base_rule["pathway"],
        "feasibility_score": round(adjusted_score, 3),
        "base_feasibility_score": base_rule["feasibility_score"],
        "constraints": base_rule.get("constraints", []),
        "situational_note": note,
    }


def assess_recoverability(classification_output: Dict[str, Any], llm: Optional[ChatGroq] = None) -> Dict[str, Any]:
    """
    Run recoverability assessment across all materials from Agent 1's output.
    """
    rules = load_recovery_rules()
    assessments = []
    for m in classification_output.get("materials", []):
        assessment = assess_material(m["type"], m.get("contamination", "none"), rules, llm=llm)
        assessments.append(assessment)
    return {"assessments": assessments}


if __name__ == "__main__":
    demo_classification = {
        "materials": [
            {"type": "concrete", "sub_material": "rubble", "est_percent": 85, "contamination": "low"},
            {"type": "steel", "sub_material": "rebar", "est_percent": 15, "contamination": "none"},
        ]
    }
    print(json.dumps(assess_recoverability(demo_classification), indent=2))
