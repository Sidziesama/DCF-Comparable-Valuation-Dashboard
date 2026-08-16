"""Scenario comparison, sequential valuation bridges, and deterministic diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np
import pandas as pd

try:
    from valuation import run_dcf
except ModuleNotFoundError:  # pragma: no cover
    from .valuation import run_dcf


def build_scenario_decomposition(forecasts, scenario_values, assumptions, current_price, forecast_metadata=None, config_lineage="config/company.yaml"):
    """Long-form Bear/Base/Bull table suitable for machines and presentation clients."""
    rows = []
    metadata = forecast_metadata or {}
    for scenario in ("bear", "base", "bull"):
        forecast = forecasts[scenario]
        config = assumptions["scenarios"][scenario]
        first, last = forecast.iloc[0], forecast.iloc[-1]
        values = {
            "Forecast method": (metadata.get(scenario, {}).get("forecast_method", "top_down_fallback"), "text"),
            "Top-down fallback used": (bool(metadata.get(scenario, {}).get("fallback_used", True)), "boolean"),
            "Revenue growth - Year 1": (first.get("revenue_growth"), "ratio"),
            "Revenue CAGR": ((last.revenue / first.revenue) ** (1 / max(len(forecast) - 1, 1)) - 1, "ratio"),
            "Operating margin - Year 1": (first.get("operating_margin"), "ratio"),
            "Operating margin - terminal year": (last.get("operating_margin"), "ratio"),
            "Tax rate": (first.get("tax_rate"), "ratio"),
            "Capex / revenue": (first.get("capex") / first.get("revenue"), "ratio"),
            "Change NWC / incremental revenue": (config.get("delta_nwc_pct_incremental_revenue"), "ratio"),
            "FCFF - Year 1": (first.get("fcff"), "USD millions"),
            "FCFF - terminal year": (last.get("fcff"), "USD millions"),
            "WACC": (config.get("wacc", assumptions.get("resolved_wacc")), "ratio"),
            "Terminal growth": (config.get("terminal_growth", assumptions.get("terminal_growth")), "ratio"),
            "Exit multiple": (assumptions.get("terminal_exit_multiple", {}).get(scenario), "multiple"),
            "Implied share price": (scenario_values.loc[scenario, "implied_share_price"], "USD/share"),
            "Upside / downside": (scenario_values.loc[scenario, "implied_share_price"] / current_price - 1, "ratio"),
        }
        drivers = metadata.get(scenario, {}).get("driver_assumptions", {})
        for name, value in drivers.items():
            if name == "revenue_growth_weights":
                continue
            display = value[0] if isinstance(value, list) and value else value
            values[f"Driver: {name.replace('_', ' ')} - Year 1"] = (display, "ratio")
        for metric, (value, unit) in values.items():
            rows.append({"scenario": scenario, "metric": metric, "value": value, "unit": unit,
                "source": "model assumptions / DCF output", "lineage": f"{config_lineage}; model forecast; scenario_valuation.csv",
                "status": "available" if pd.notna(value) else "not_available",
                "quality": "calculated" if unit != "text" else "model_method"})
    return pd.DataFrame(rows)


def _value(forecast, wacc, terminal_growth, cash, debt, shares):
    return run_dcf(forecast, wacc, terminal_growth, cash, debt, shares)["implied_share_price"]


def build_valuation_attribution(forecasts, assumptions, wacc, terminal_growth, cash, debt, shares):
    """Sequential bridge: growth, margin/reinvestment, WACC, then terminal growth.

    Order-dependent by design; reconciliation is exact and the methodology is not Shapley.
    """
    rows = []
    for start_name, end_name in (("bear", "base"), ("base", "bull")):
        start, end = forecasts[start_name], forecasts[end_name]
        start_cfg, end_cfg = assumptions["scenarios"][start_name], assumptions["scenarios"][end_name]
        start_wacc = float(start_cfg.get("wacc", wacc)); end_wacc = float(end_cfg.get("wacc", wacc))
        start_tg = float(start_cfg.get("terminal_growth", terminal_growth)); end_tg = float(end_cfg.get("terminal_growth", terminal_growth))
        current = start.copy()
        opening = _value(current, start_wacc, start_tg, cash, debt, shares)
        previous = opening

        growth_case = current.copy()
        growth_case["revenue_growth"] = end["revenue_growth"].values
        base_revenue = start.iloc[0].revenue / (1 + start.iloc[0].revenue_growth)
        revenues = []; prior = base_revenue
        for growth in growth_case.revenue_growth:
            prior *= 1 + growth; revenues.append(prior)
        scale = np.array(revenues) / current.revenue.values
        for col in ("revenue", "ebit", "cash_taxes", "nopat", "da", "ebitda", "capex", "change_nwc", "fcff"):
            if col in growth_case: growth_case[col] = current[col].values * scale
        value = _value(growth_case, start_wacc, start_tg, cash, debt, shares)
        rows.append(("Growth", previous, value)); previous = value

        operating_case = end.copy()
        value = _value(operating_case, start_wacc, start_tg, cash, debt, shares)
        rows.append(("Margin / reinvestment", previous, value)); previous = value
        value = _value(operating_case, end_wacc, start_tg, cash, debt, shares)
        rows.append(("WACC", previous, value)); previous = value
        value = _value(operating_case, end_wacc, end_tg, cash, debt, shares)
        rows.append(("Terminal assumptions", previous, value)); previous = value
        for order, (driver, before, after) in enumerate(rows[-4:], 1):
            rows[-4 + order - 1] = {"bridge": f"{start_name}_to_{end_name}", "order": order,
                "driver": driver, "value_before": before, "value_after": after, "contribution": after-before,
                "opening_value": opening, "closing_value": value, "methodology": "sequential_order_dependent",
                "status": "available", "quality": "calculated", "lineage": "forecast scenarios; DCF assumptions"}
    return pd.DataFrame(rows)


def build_investment_diagnostics(scenario_decomposition, attribution, reverse_dcf, historical, forecasts, current_price):
    """Deterministic calculations plus explicitly labeled interpretation rules."""
    rows = []
    def add(question, calculation, interpretation, value=np.nan, unit="", status="available"):
        rows.append({"question": question, "calculation": calculation, "interpretive_rule": interpretation,
            "value": value, "unit": unit, "status": status, "quality": "rule_based",
            "lineage": "reverse_dcf.csv; scenario_decomposition.csv; valuation_attribution.csv"})
    base_price = float(scenario_decomposition.query("scenario == 'base' and metric == 'Implied share price'").value.iloc[0])
    add("What is priced in?", "Reverse DCF solves the selected assumption while holding other base-case inputs fixed.",
        "Report converged market-implied assumptions; solver failures remain explicit.", current_price, "USD/share")
    for mode, item in reverse_dcf.iterrows():
        add(f"Market-implied {mode.replace('_', ' ')}", "Bounded reverse-DCF solution.",
            "Compare with historical and Base only when the solver converged.", item.get("implied_assumption"), "ratio",
            item.get("status", "failed"))
    annual = historical.loc[historical.index.astype(str) != "LTM"]
    hist_growth = pd.to_numeric(annual.get("revenue"), errors="coerce").pct_change(fill_method=None).mean()
    base_growth = pd.to_numeric(forecasts["base"]["revenue_growth"], errors="coerce").mean()
    add("Where does Base differ from history?", "Base average revenue growth minus historical average growth.",
        "Positive means Base assumes faster growth than the observed history; it is not a recommendation.", base_growth-hist_growth, "ratio")
    implied_growth = reverse_dcf.loc["revenue_growth", "implied_assumption"] if "revenue_growth" in reverse_dcf.index else np.nan
    if np.isfinite(implied_growth):
        add("How does market-implied growth compare?", "Market-implied growth minus Base average and historical average.",
            f"Gap versus Base: {implied_growth-base_growth:+.1%}; gap versus historical average: {implied_growth-hist_growth:+.1%}.",
            implied_growth-base_growth, "ratio")
    hist_margin = (pd.to_numeric(annual.get("operating_income"), errors="coerce") /
        pd.to_numeric(annual.get("revenue"), errors="coerce")).mean()
    base_margin = pd.to_numeric(forecasts["base"]["operating_margin"], errors="coerce").mean()
    implied_margin = reverse_dcf.loc["operating_margin", "implied_assumption"] if "operating_margin" in reverse_dcf.index else np.nan
    if np.isfinite(implied_margin):
        add("How does market-implied margin compare?", "Market-implied operating margin minus Base average and historical average.",
            f"Gap versus Base: {implied_margin-base_margin:+.1%}; gap versus historical average: {implied_margin-hist_margin:+.1%}.",
            implied_margin-base_margin, "ratio")
    ranked = attribution.assign(abs_contribution=attribution.contribution.abs()).sort_values("abs_contribution", ascending=False)
    if not ranked.empty:
        top = ranked.iloc[0]
        add("Which assumption has greatest scenario sensitivity?", "Largest absolute sequential bridge contribution.",
            f"{top.driver} is largest under the documented order-dependent bridge; order can affect attribution.", top.contribution, "USD/share")
    add("What operational outcome is required to justify current price?", "Compare current price with scenario values and reverse-DCF outputs.",
        "Current price is justified within this model when a scenario or converged implied assumption produces at least that value.", base_price-current_price, "USD/share")
    for scenario in ("bear", "base", "bull"):
        block = scenario_decomposition[scenario_decomposition.scenario.eq(scenario)].set_index("metric")
        threshold = block.loc["Revenue growth - Year 1", "value"]
        margin = block.loc["Operating margin - terminal year", "value"]
        add(f"Candidate thesis breaker: {scenario}", "Thresholds are taken from the scenario boundary assumptions.",
            f"Flag if first-year revenue growth falls below {threshold:.1%} or terminal operating margin falls below {margin:.1%}.", threshold, "ratio")
    return pd.DataFrame(rows)


def build_adapter_health(adapter_name, kpis, metadata, error=""):
    available = int(kpis.status.eq("available").sum()) if not kpis.empty else 0
    return pd.DataFrame([{"category": "Operating KPI adapter", "adapter": adapter_name,
        "status": "WARN" if error or available == 0 else "PASS", "available": available,
        "unavailable": int(kpis.status.ne("available").sum()) if not kpis.empty else 0,
        "fallback_scenarios": sum(bool(item.get("fallback_used")) for item in metadata.values()),
        "detail": error or "KPI coverage and forecast fallback are non-fatal and explicitly tracked."}])
