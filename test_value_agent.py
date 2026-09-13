import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from value_agent import calculate_value, calculate_batch_value, load_pricing_table


def test_load_pricing_table():
    table = load_pricing_table()
    assert "concrete" in table
    assert "steel" in table
    assert table["steel"]["market_rate_per_ton"] == 220


def test_calculate_value_concrete():
    result = calculate_value("concrete", 2.125)
    assert result["gross_value"] == pytest.approx(17.0, rel=1e-3)
    assert result["processing_cost"] == pytest.approx(11.6875, rel=1e-3)
    assert result["net_value"] == pytest.approx(5.3125, rel=1e-2)
    assert result["co2_avoided_kg"] == pytest.approx(148.75, rel=1e-3)
    assert result["used_fallback_rate"] is False


def test_calculate_value_unknown_material_falls_back():
    result = calculate_value("unobtainium", 1.0)
    assert result["used_fallback_rate"] is True
    assert result["matched_key"] == "mixed_debris"


def test_calculate_value_negative_weight_raises():
    with pytest.raises(ValueError):
        calculate_value("steel", -1)


def test_calculate_value_case_and_whitespace_insensitive():
    a = calculate_value("Steel", 1.0)
    b = calculate_value(" steel ", 1.0)
    assert a["net_value"] == b["net_value"]


def test_calculate_batch_value_sums_correctly():
    materials = [
        {"type": "concrete", "sub_material": "rubble", "est_percent": 85, "contamination": "low"},
        {"type": "steel", "sub_material": "rebar", "est_percent": 15, "contamination": "none"},
    ]
    result = calculate_batch_value(materials, total_quantity=2.5, unit="ton")
    assert len(result["line_items"]) == 2
    manual_total = sum(li["net_value"] for li in result["line_items"])
    assert result["total_net_value"] == pytest.approx(manual_total, rel=1e-6)
    assert result["total_co2_avoided_kg"] > 0


def test_calculate_batch_value_kg_conversion():
    materials = [{"type": "steel", "sub_material": None, "est_percent": 100, "contamination": None}]
    result_kg = calculate_batch_value(materials, total_quantity=1000, unit="kg")
    result_ton = calculate_batch_value(materials, total_quantity=1, unit="ton")
    assert result_kg["total_net_value"] == pytest.approx(result_ton["total_net_value"], rel=1e-6)
