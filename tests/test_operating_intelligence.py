import numpy as np
import pandas as pd
import pytest

from src.adapters import BusinessModelAdapter, get_adapter
from src.forecast_model import build_forecast_from_inputs
from src.operating_intelligence import (
    build_adapter_health,
    build_investment_diagnostics,
    build_scenario_decomposition,
    build_valuation_attribution,
)
from src.valuation import value_scenarios


def _forecast(growth, margin):
    return build_forecast_from_inputs(100, [growth] * 3, [margin] * 3, 0.20,
        0.03, 0.025, 0.01, start_year=2027)


def _inputs():
    forecasts = {"bear": _forecast(.05, .25), "base": _forecast(.08, .30), "bull": _forecast(.11, .35)}
    assumptions = {"terminal_growth": .025, "scenarios": {
        "bear": {"revenue_growth": [.05]*3, "operating_margin": [.25]*3, "tax_rate": .20, "delta_nwc_pct_incremental_revenue": .01},
        "base": {"revenue_growth": [.08]*3, "operating_margin": [.30]*3, "tax_rate": .20, "delta_nwc_pct_incremental_revenue": .01},
        "bull": {"revenue_growth": [.11]*3, "operating_margin": [.35]*3, "tax_rate": .20, "delta_nwc_pct_incremental_revenue": .01},
    }}
    values = value_scenarios(forecasts, .08, .025, 10, 5, 10)
    return forecasts, assumptions, values


def test_adapter_selection_and_interface():
    adapter = get_adapter("payment_network")
    assert isinstance(adapter, BusinessModelAdapter)
    assert adapter.name == "payment_network"
    with pytest.raises(ValueError, match="Unknown business-model adapter"):
        get_adapter("airline")


def test_kpi_normalization_and_defensible_growth():
    adapter = get_adapter("payment_network")
    kpis = adapter.normalize_kpis([
        {"metric": "payments_volume", "period": 2023, "value": 10, "unit": "USDtn", "source": "10-K"},
        {"metric": "payments_volume", "period": 2024, "value": 11, "unit": "USDtn", "source": "10-K"},
        {"metric": "unknown", "period": 2024, "value": None},
    ])
    derived = adapter.derive_metrics(kpis)
    assert derived.loc[0, "metric"] == "payments_volume_growth"
    assert derived.loc[0, "value"] == pytest.approx(.10)
    assert kpis.loc[kpis.metric.eq("unknown"), "status"].iloc[0] == "not_available"


def test_driver_forecast_and_top_down_fallback():
    adapter = get_adapter("payment_network")
    growth, meta = adapter.forecast_growth("base", {"revenue_growth": [.09, .08], "operating_drivers": {
        "payments_volume_growth": [.08, .07], "cross_border_volume_growth": [.10, .09],
        "processed_transactions_growth": [.09, .08], "monetization_overlay": .01}}, 2)
    assert growth == pytest.approx([.099, .089])
    assert not meta["fallback_used"]
    fallback, meta = adapter.forecast_growth("base", {"revenue_growth": [.09, .08]}, 2)
    assert fallback == [.09, .08] and meta["fallback_used"]


def test_scenario_comparison_and_attribution_reconcile():
    forecasts, assumptions, values = _inputs()
    table = build_scenario_decomposition(forecasts, values, {**assumptions, "resolved_wacc": .08}, 2.0)
    assert set(table.scenario) == {"bear", "base", "bull"}
    bridge = build_valuation_attribution(forecasts, assumptions, .08, .025, 10, 5, 10)
    for _, group in bridge.groupby("bridge"):
        assert group.contribution.sum() == pytest.approx(group.closing_value.iloc[0] - group.opening_value.iloc[0])
        assert group.methodology.eq("sequential_order_dependent").all()


def test_diagnostic_rules_and_missing_kpi_health_are_nonfatal():
    forecasts, assumptions, values = _inputs()
    table = build_scenario_decomposition(forecasts, values, {**assumptions, "resolved_wacc": .08}, 2.0)
    bridge = build_valuation_attribution(forecasts, assumptions, .08, .025, 10, 5, 10)
    reverse = pd.DataFrame({"implied_assumption": [.07], "status": ["converged"]}, index=["revenue_growth"])
    historical = pd.DataFrame({"revenue": [80, 90, 100]}, index=[2024, 2025, "LTM"])
    result = build_investment_diagnostics(table, bridge, reverse, historical, forecasts, 2.0)
    assert result.question.str.contains("priced in", case=False).any()
    assert result.question.str.contains("thesis breaker", case=False).sum() == 3
    health = build_adapter_health("payment_network", pd.DataFrame(), {}, "optional source unavailable")
    assert health.loc[0, "status"] == "WARN"
