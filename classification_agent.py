"""
Waste Classification & Analysis Agent
--------------------------------------
LLM-only agent. Takes a free-text waste description or manifest line and
returns a strict JSON breakdown of materials, composition, and contamination.

No calculation happens here — that is Agent 3's job (value_agent.py).
"""

import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a construction & demolition (C&D) waste classification expert.
Given a free-text description of a waste stream, identify the distinct materials present,
estimate their composition as percentages (must sum to ~100), sub-material detail, and
contamination level (none | low | medium | high).

Respond with ONLY valid JSON matching this exact schema, no markdown fences, no commentary:
{
  "materials": [
    {"type": "<material>", "sub_material": "<detail or null>", "est_percent": <number>, "contamination": "<none|low|medium|high>"}
  ],
  "total_quantity": <number>,
  "unit": "<ton|kg|cubic_yard|item>",
  "confidence": <number between 0 and 1>
}

Valid material "type" values (use these exact lowercase keys when they apply, otherwise
your best lowercase single-word/underscore label): concrete, steel, wood, drywall, asphalt,
brick, glass, plastic, insulation, mixed_debris.
If a quantity isn't stated, make a reasonable estimate and lower your confidence score.
"""


class MaterialItem(BaseModel):
    type: str
    sub_material: Optional[str] = None
    est_percent: float = Field(ge=0, le=100)
    contamination: str = Field(default="none")

    @field_validator("contamination")
    @classmethod
    def validate_contamination(cls, v):
        allowed = {"none", "low", "medium", "high"}
        v = (v or "none").lower()
        return v if v in allowed else "none"


class ClassificationOutput(BaseModel):
    materials: List[MaterialItem]
    total_quantity: float
    unit: str = "ton"
    confidence: float = Field(ge=0, le=1, default=0.5)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def classify_waste(description: str, llm: Optional[ChatGroq] = None) -> dict:
    """
    Run the classification agent on a single waste description.

    Args:
        description: free-text waste description / manifest line.
        llm: optional pre-built ChatGroq client (useful for testing/mocking).

    Returns:
        dict matching ClassificationOutput schema.
    """
    if llm is None:
        llm = ChatGroq(model=MODEL_NAME, temperature=0)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=description),
        ]
    )

    raw = _strip_code_fences(response.content)

    try:
        parsed = json.loads(raw)
        validated = ClassificationOutput(**parsed)
        return validated.model_dump()
    except (json.JSONDecodeError, Exception) as e:
        # Fail safe: route to mixed_debris with low confidence rather than crash
        return {
            "materials": [
                {"type": "mixed_debris", "sub_material": None, "est_percent": 100, "contamination": "medium"}
            ],
            "total_quantity": 1.0,
            "unit": "ton",
            "confidence": 0.1,
            "error": f"Failed to parse LLM output cleanly: {e}",
            "raw_output": raw,
        }


if __name__ == "__main__":
    demo = "2.5 tons mixed concrete rubble with embedded rebar, from Site B demolition"
    print(json.dumps(classify_waste(demo), indent=2))
