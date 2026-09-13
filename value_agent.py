"""
Value Calculation Agent
------------------------
This is the credibility feature of the system: every dollar and every kg of
CO2 avoided in the final report comes from THIS deterministic module, never
from an LLM. The LLM is only ever handed the *result* of these calculations
to phrase in natural language downstream (see plan_agent.py).

Judges can open data/pricing_table.json and verify the numbers themselves.
"""

import json
import os
from typing import List, Dict, Any

_DATA_DIR = os.path.dirname(__file__)
_PRICING_PATH = os.path.join(_DATA_DIR, "pricing_table.json")

_FALLBACK_MATERIAL = "mixed_debris"


def load_pricing_table(path: str = _PRICING_PATH) -> Dict[str, Any]:
    """Load the pricing/CO2 lookup table from disk."""
    with open(path, "r") as f:
        return json.load(f)


def calculate_value(
    material: str,
    weight_ton: float,
    pricing_table: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Pure deterministic calculation for a single material line item.

    Args:
        material: material key, e.g. "concrete". Falls back to
            "mixed_debris" rates if the material is not in the table
            (so the pipeline never silently drops value to zero).
        weight_ton: weight of this material in metric tons.
        pricing_table: optional pre-loaded table (avoids re-reading disk
            for every line item in a batch).

    Returns:
        A line-item dict with gross value, processing cost, net value,
        and CO2 avoided — all computed here, never by an LLM.
    """
    if pricing_table is None:
        pricing_table = load_pricing_table()

    key = material.lower().strip().replace(" ", "_")
    used_fallback = False
    rates = pricing_table.get(key)
    if rates is None:
        rates = pricing_table[_FALLBACK_MATERIAL]
        used_fallback = True

    if weight_ton < 0:
        raise ValueError("weight_ton cannot be negative")

    gross_value = round(weight_ton * rates["market_rate_per_ton"], 2)
    processing_cost = round(weight_ton * rates["processing_cost_per_ton"], 2)
    net_value = round(gross_value - processing_cost, 2)
    co2_avoided_kg = round(weight_ton * rates["co2_avoided_per_ton"], 2)

    return {
        "material": material,
        "matched_key": key if not used_fallback else _FALLBACK_MATERIAL,
        "used_fallback_rate": used_fallback,
        "weight_ton": weight_ton,
        "market_rate_per_ton": rates["market_rate_per_ton"],
        "gross_value": gross_value,
        "processing_cost": processing_cost,
        "net_value": net_value,
        "co2_avoided_kg": co2_avoided_kg,
    }


def calculate_batch_value(
    materials: List[Dict[str, Any]],
    total_quantity: float,
    unit: str = "ton",
) -> Dict[str, Any]:
    """
    Given Agent 1's classification output (materials with est_percent) and
    the total waste quantity, compute per-material weight and value.

    Args:
        materials: list of {"type": str, "est_percent": float, ...}
        total_quantity: total quantity of the waste stream.
        unit: unit of total_quantity. Only "ton" is fully supported by the
            pricing table today; other units are converted defensively
            (kg -> ton) so the pipeline degrades gracefully instead of
            producing wildly wrong numbers.

    Returns:
        {"line_items": [...], "total_net_value": float, "total_co2_avoided_kg": float}
    """
    if unit.lower() in ("kg", "kilogram", "kilograms"):
        total_tons = total_quantity / 1000.0
    else:
        total_tons = total_quantity

    pricing_table = load_pricing_table()
    line_items = []
    for m in materials:
        pct = m.get("est_percent", 0) / 100.0
        weight_ton = round(total_tons * pct, 4)
        line_item = calculate_value(m["type"], weight_ton, pricing_table)
        # carry through classification context for the report
        line_item["sub_material"] = m.get("sub_material")
        line_item["contamination"] = m.get("contamination")
        line_items.append(line_item)

    total_net_value = round(sum(li["net_value"] for li in line_items), 2)
    total_co2_avoided_kg = round(sum(li["co2_avoided_kg"] for li in line_items), 2)

    return {
        "line_items": line_items,
        "total_net_value": total_net_value,
        "total_co2_avoided_kg": total_co2_avoided_kg,
    }


if __name__ == "__main__":
    # Quick manual smoke test
    demo_materials = [
        {"type": "concrete", "sub_material": "rubble", "est_percent": 85, "contamination": "low"},
        {"type": "steel", "sub_material": "rebar", "est_percent": 15, "contamination": "none"},
    ]
    result = calculate_batch_value(demo_materials, total_quantity=2.5, unit="ton")
    print(json.dumps(result, indent=2))
