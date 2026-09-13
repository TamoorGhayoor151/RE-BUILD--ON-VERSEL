"""
Orchestrator
-------------
Wires the 4 independent agents into a LangGraph graph:

  classify -> assess_recoverability -> calculate_value -> generate_plan

Each node is a thin wrapper around the corresponding agent function, so the
graph itself stays readable and each agent remains independently testable
and swappable. This graph is what you screenshot/diagram for judges — it
makes the "independent specialized agents" framing literal and visible.
"""

from typing import TypedDict, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END

from classification_agent import classify_waste
from recoverability_agent import assess_recoverability
from value_agent import calculate_batch_value
from plan_agent import generate_plan


class WasteState(TypedDict, total=False):
    description: str
    classification: Dict[str, Any]
    recoverability: Dict[str, Any]
    value: Dict[str, Any]
    plan: Dict[str, Any]
    error: Optional[str]


def node_classify(state: WasteState) -> WasteState:
    result = classify_waste(state["description"])
    return {"classification": result}


def node_assess_recoverability(state: WasteState) -> WasteState:
    result = assess_recoverability(state["classification"])
    return {"recoverability": result}


def node_calculate_value(state: WasteState) -> WasteState:
    classification = state["classification"]
    result = calculate_batch_value(
        classification["materials"],
        classification["total_quantity"],
        classification.get("unit", "ton"),
    )
    return {"value": result}


def node_generate_plan(state: WasteState) -> WasteState:
    result = generate_plan(
        classification=state["classification"],
        recoverability=state["recoverability"],
        value=state["value"],
        source_description=state["description"],
    )
    return {"plan": result}


def build_graph():
    graph = StateGraph(WasteState)

    graph.add_node("classify", node_classify)
    graph.add_node("assess_recoverability", node_assess_recoverability)
    graph.add_node("calculate_value", node_calculate_value)
    graph.add_node("generate_plan", node_generate_plan)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "assess_recoverability")
    graph.add_edge("assess_recoverability", "calculate_value")
    graph.add_edge("calculate_value", "generate_plan")
    graph.add_edge("generate_plan", END)

    return graph.compile()


def run_pipeline(description: str) -> WasteState:
    """Run the full 4-agent pipeline on a single waste description."""
    app = build_graph()
    final_state = app.invoke({"description": description})
    return final_state


def run_pipeline_batch(descriptions: List[str]) -> Dict[str, Any]:
    """
    Run the pipeline over multiple waste items and aggregate site-level
    totals. Aggregation math is, as always, deterministic Python — not LLM.
    """
    results = [run_pipeline(desc) for desc in descriptions]

    total_net_value = round(sum(r["value"]["total_net_value"] for r in results), 2)
    total_co2_avoided_kg = round(sum(r["value"]["total_co2_avoided_kg"] for r in results), 2)

    return {
        "items": results,
        "site_summary": {
            "total_net_value": total_net_value,
            "total_co2_avoided_kg": total_co2_avoided_kg,
            "num_items": len(results),
        },
    }


if __name__ == "__main__":
    import json

    demo = "2.5 tons mixed concrete rubble with embedded rebar, from Site B demolition"
    print(json.dumps(run_pipeline(demo), indent=2, default=str))
