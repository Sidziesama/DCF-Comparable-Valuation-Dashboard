"""Reusable operating forecast builders.

This module is deliberately free of I/O so it can be used by both the batch
pipeline and interactive clients without triggering data downloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


SCENARIOS = ("bear", "base", "bull")


def get_ltm_base(historical_model: pd.DataFrame) -> pd.Series:
    if "LTM" not in historical_model.index:
        raise ValueError("Historical model does not contain an LTM row.")
    return historical_model.loc["LTM"]


def _year_values(value: float | Sequence[float], years: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be numeric.")
    if isinstance(value, Sequence):
        values = [float(item) for item in value]
        if len(values) != years:
            raise ValueError(f"{name} must contain {years} values.")
        return values
    return [float(value)] * years


def build_forecast_from_inputs(
    base_revenue: float,
    revenue_growth: Sequence[float],
    operating_margin: float | Sequence[float],
    tax_rate: float | Sequence[float],
    da_pct_revenue: float | Sequence[float],
    capex_pct_revenue: float | Sequence[float],
    delta_nwc_pct_incremental_revenue: float | Sequence[float],
    *,
    scenario: str = "custom",
    start_year: int = 2027,
) -> pd.DataFrame:
    """Build an FCFF forecast from explicit, testable inputs."""
    growth = [float(item) for item in revenue_growth]
    if not growth:
        raise ValueError("revenue_growth cannot be empty.")
    if base_revenue <= 0:
        raise ValueError("base_revenue must be positive.")

    count = len(growth)
    margins = _year_values(operating_margin, count, "operating_margin")
    taxes = _year_values(tax_rate, count, "tax_rate")
    da_rates = _year_values(da_pct_revenue, count, "da_pct_revenue")
    capex_rates = _year_values(capex_pct_revenue, count, "capex_pct_revenue")
    nwc_rates = _year_values(
        delta_nwc_pct_incremental_revenue,
        count,
        "delta_nwc_pct_incremental_revenue",
    )

    previous_revenue = float(base_revenue)
    rows: list[dict[str, float | int | str]] = []
    for offset, rate in enumerate(growth):
        revenue = previous_revenue * (1 + rate)
        incremental_revenue = revenue - previous_revenue
        ebit = revenue * margins[offset]
        cash_taxes = ebit * taxes[offset]
        nopat = ebit - cash_taxes
        da = revenue * da_rates[offset]
        capex = revenue * capex_rates[offset]
        change_nwc = incremental_revenue * nwc_rates[offset]
        fcff = nopat + da - capex - change_nwc
        rows.append(
            {
                "year": start_year + offset,
                "scenario": scenario,
                "revenue": revenue,
                "revenue_growth": rate,
                "operating_margin": margins[offset],
                "ebit": ebit,
                "tax_rate": taxes[offset],
                "cash_taxes": cash_taxes,
                "nopat": nopat,
                "da": da,
                "ebitda": ebit + da,
                "capex": capex,
                "change_nwc": change_nwc,
                "fcff": fcff,
                "fcff_margin": fcff / revenue,
            }
        )
        previous_revenue = revenue
    return pd.DataFrame(rows).set_index("year")


def build_forecast(
    historical_model: pd.DataFrame,
    assumptions: Mapping,
    scenario: str = "base",
    start_year: int = 2027,
) -> pd.DataFrame:
    scenarios = assumptions.get("scenarios", {})
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {tuple(scenarios)}.")
    selected = scenarios[scenario]
    return build_forecast_from_inputs(
        base_revenue=float(get_ltm_base(historical_model)["revenue"]),
        revenue_growth=selected["revenue_growth"],
        operating_margin=selected["operating_margin"],
        tax_rate=selected["tax_rate"],
        da_pct_revenue=selected["da_pct_revenue"],
        capex_pct_revenue=selected["capex_pct_revenue"],
        delta_nwc_pct_incremental_revenue=selected[
            "delta_nwc_pct_incremental_revenue"
        ],
        scenario=scenario,
        start_year=start_year,
    )


def build_all_scenarios(
    historical_model: pd.DataFrame,
    assumptions: Mapping,
    start_year: int = 2027,
) -> dict[str, pd.DataFrame]:
    return {
        scenario: build_forecast(historical_model, assumptions, scenario, start_year)
        for scenario in SCENARIOS
    }
