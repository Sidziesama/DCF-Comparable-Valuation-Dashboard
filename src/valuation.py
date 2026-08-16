"""Pure valuation functions shared by the pipeline and dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # Support both ``python src/pipeline.py`` and package-style test imports.
    from forecast_model import build_forecast_from_inputs
except ModuleNotFoundError:  # pragma: no cover - exercised by package imports
    from .forecast_model import build_forecast_from_inputs


def _validate_common(forecast: pd.DataFrame, wacc: float, shares: float) -> None:
    if forecast.empty or "fcff" not in forecast:
        raise ValueError("forecast must contain at least one FCFF row.")
    if wacc <= -1:
        raise ValueError("WACC must be greater than -100%.")
    if shares <= 0:
        raise ValueError("shares_outstanding must be positive.")


def _discount_forecast(forecast: pd.DataFrame, wacc: float) -> tuple[pd.DataFrame, float]:
    result = forecast.copy()
    result["period"] = np.arange(1, len(result) + 1)
    result["discount_factor"] = 1 / (1 + wacc) ** result["period"]
    result["pv_fcff"] = result["fcff"] * result["discount_factor"]
    return result, float(result["pv_fcff"].sum())


def _valuation_result(
    forecast: pd.DataFrame,
    pv_forecast_fcff: float,
    terminal_value: float,
    cash: float,
    debt: float,
    shares_outstanding: float,
) -> dict:
    pv_terminal_value = terminal_value * float(forecast["discount_factor"].iloc[-1])
    enterprise_value = pv_forecast_fcff + pv_terminal_value
    net_debt = debt - cash
    equity_value = enterprise_value - net_debt
    return {
        "forecast": forecast,
        "pv_forecast_fcff": pv_forecast_fcff,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "cash": cash,
        "debt": debt,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "shares_outstanding": shares_outstanding,
        "implied_share_price": equity_value / shares_outstanding,
        "terminal_value_pct_ev": pv_terminal_value / enterprise_value,
        "explicit_forecast_pct_ev": pv_forecast_fcff / enterprise_value,
    }


def run_dcf(forecast, wacc, terminal_growth, cash, debt, shares_outstanding):
    """Value FCFF using a Gordon-growth terminal value."""
    _validate_common(forecast, wacc, shares_outstanding)
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth.")
    discounted, explicit_pv = _discount_forecast(forecast, wacc)
    terminal_fcff = float(discounted["fcff"].iloc[-1]) * (1 + terminal_growth)
    terminal_value = terminal_fcff / (wacc - terminal_growth)
    result = _valuation_result(
        discounted, explicit_pv, terminal_value, cash, debt, shares_outstanding
    )
    result.update({"method": "gordon_growth", "terminal_growth": terminal_growth})
    return result


def run_exit_multiple_dcf(
    forecast,
    wacc,
    exit_multiple,
    cash,
    debt,
    shares_outstanding,
):
    """Value FCFF using terminal-year EBITDA and an EV/EBITDA multiple."""
    _validate_common(forecast, wacc, shares_outstanding)
    if exit_multiple <= 0:
        raise ValueError("exit_multiple must be positive.")
    discounted, explicit_pv = _discount_forecast(forecast, wacc)
    if "ebitda" in discounted:
        terminal_ebitda = float(discounted["ebitda"].iloc[-1])
    elif {"ebit", "da"}.issubset(discounted.columns):
        terminal_ebitda = float(discounted["ebit"].iloc[-1] + discounted["da"].iloc[-1])
    else:
        raise ValueError("forecast requires EBITDA or both EBIT and D&A.")
    result = _valuation_result(
        discounted,
        explicit_pv,
        terminal_ebitda * exit_multiple,
        cash,
        debt,
        shares_outstanding,
    )
    result.update(
        {
            "method": "exit_multiple",
            "exit_multiple": exit_multiple,
            "terminal_ebitda": terminal_ebitda,
        }
    )
    return result


def build_sensitivity_table(
    forecast, wacc_values, terminal_growth_values, cash, debt, shares_outstanding
):
    return pd.DataFrame(
        {
            wacc: {
                growth: (
                    np.nan
                    if wacc <= growth
                    else run_dcf(
                        forecast, wacc, growth, cash, debt, shares_outstanding
                    )["implied_share_price"]
                )
                for growth in terminal_growth_values
            }
            for wacc in wacc_values
        }
    ).rename_axis(index="Terminal Growth", columns="WACC")


def value_scenarios(forecasts, wacc, terminal_growth, cash, debt, shares_outstanding):
    rows = []
    for scenario, forecast in forecasts.items():
        value = run_dcf(forecast, wacc, terminal_growth, cash, debt, shares_outstanding)
        rows.append(
            {
                "scenario": scenario,
                **{
                    key: value[key]
                    for key in (
                        "enterprise_value",
                        "equity_value",
                        "implied_share_price",
                        "explicit_forecast_pct_ev",
                        "terminal_value_pct_ev",
                    )
                },
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def _solve_bounded(objective, lower, upper, *, tolerance=1e-6, max_iterations=200):
    """Solve a monotonic scalar objective with explicit bracketing diagnostics."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Solver bounds must be finite and lower must be below upper.")
    low_value, high_value = float(objective(lower)), float(objective(upper))
    if not np.isfinite(low_value) or not np.isfinite(high_value):
        raise ValueError("Reverse DCF objective is non-finite at a solver bound.")
    if abs(low_value) <= tolerance:
        return lower, 0, low_value
    if abs(high_value) <= tolerance:
        return upper, 0, high_value
    if low_value * high_value > 0:
        raise ValueError(
            "Target share price is outside the values produced by the configured "
            f"bounds (endpoint residuals {low_value:,.4f} and {high_value:,.4f})."
        )
    for iteration in range(1, max_iterations + 1):
        midpoint = (lower + upper) / 2
        mid_value = float(objective(midpoint))
        if not np.isfinite(mid_value):
            raise ValueError("Reverse DCF objective became non-finite during iteration.")
        if abs(mid_value) <= tolerance:
            return midpoint, iteration, mid_value
        if low_value * mid_value <= 0:
            upper, high_value = midpoint, mid_value
        else:
            lower, low_value = midpoint, mid_value
    raise RuntimeError(f"Reverse DCF did not converge after {max_iterations} iterations.")


def _forecast_driver(forecast: pd.DataFrame, column: str, fallback: float) -> list[float]:
    if column not in forecast:
        return [fallback] * len(forecast)
    values = pd.to_numeric(forecast[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Forecast column {column!r} contains non-numeric values.")
    return values.astype(float).tolist()


def solve_reverse_dcf(
    base_revenue, reference_forecast, target_share_price, wacc, terminal_growth,
    cash, debt, shares_outstanding, *, mode="revenue_growth", bounds=None,
    operating_margin=None, tolerance=1e-6, max_iterations=200,
):
    """Solve for a market-implied revenue growth, margin, or terminal growth."""
    if target_share_price <= 0:
        raise ValueError("target_share_price must be positive.")
    if base_revenue <= 0:
        raise ValueError("base_revenue must be positive.")
    if reference_forecast.empty:
        raise ValueError("reference_forecast cannot be empty.")
    allowed = {"revenue_growth", "operating_margin", "terminal_growth"}
    if mode not in allowed:
        raise ValueError(f"Unsupported reverse DCF mode {mode!r}; expected {sorted(allowed)}.")
    default_bounds = {
        "revenue_growth": (-0.20, 0.35),
        "operating_margin": (0.05, 0.85),
        "terminal_growth": (-0.02, min(0.06, wacc - 0.0025)),
    }
    lower, upper = bounds or default_bounds[mode]
    if mode == "terminal_growth" and upper >= wacc:
        raise ValueError("The terminal-growth upper bound must be below WACC.")

    years = len(reference_forecast)
    start_year = int(reference_forecast.index[0]) if len(reference_forecast.index) else 1
    reference_growth = _forecast_driver(reference_forecast, "revenue_growth", 0.0)
    reference_margin = _forecast_driver(reference_forecast, "operating_margin", 0.0)
    fixed_margin = operating_margin if operating_margin is not None else reference_margin
    tax = _forecast_driver(reference_forecast, "tax_rate", 0.21)
    da_rates = ((reference_forecast["da"] / reference_forecast["revenue"]).astype(float).tolist()
                if {"da", "revenue"}.issubset(reference_forecast.columns) else [0.0] * years)
    capex_rates = ((reference_forecast["capex"] / reference_forecast["revenue"]).astype(float).tolist()
                   if {"capex", "revenue"}.issubset(reference_forecast.columns) else [0.0] * years)
    incremental_revenue = pd.to_numeric(reference_forecast["revenue"], errors="coerce").diff()
    incremental_revenue.iloc[0] = float(reference_forecast["revenue"].iloc[0]) - base_revenue
    nwc_rates = (pd.to_numeric(reference_forecast.get("change_nwc", 0.0), errors="coerce") /
                 incremental_revenue.replace(0, np.nan)).fillna(0.0).astype(float).tolist()

    def value_for(assumption):
        if mode == "terminal_growth":
            forecast, growth = reference_forecast, assumption
        else:
            growth_rates = [assumption] * years if mode == "revenue_growth" else reference_growth
            margins = [assumption] * years if mode == "operating_margin" else fixed_margin
            forecast = build_forecast_from_inputs(
                float(base_revenue), growth_rates, margins, tax, da_rates, capex_rates,
                nwc_rates, scenario="market_implied", start_year=start_year,
            )
            growth = terminal_growth
        valuation = run_dcf(forecast, wacc, growth, cash, debt, shares_outstanding)
        return valuation["implied_share_price"], forecast, valuation

    solved, iterations, _ = _solve_bounded(
        lambda assumption: value_for(assumption)[0] - target_share_price,
        float(lower), float(upper), tolerance=tolerance, max_iterations=max_iterations,
    )
    implied_price, implied_forecast, valuation = value_for(solved)
    return {
        "mode": mode, "implied_assumption": solved, "target_share_price": target_share_price,
        "implied_share_price": implied_price, "price_residual": implied_price - target_share_price,
        "converged": True, "iterations": iterations, "lower_bound": lower,
        "upper_bound": upper, "wacc": wacc,
        "terminal_growth": solved if mode == "terminal_growth" else terminal_growth,
        "forecast": implied_forecast, "valuation": valuation,
    }


def build_reverse_dcf_summary(
    forecasts, base_revenue, target_share_price, wacc, terminal_growth, cash, debt,
    shares_outstanding, *, modes=("revenue_growth", "terminal_growth", "operating_margin"),
):
    """Return market-implied assumptions plus Bear/Base/Bull comparison outputs."""
    if "base" not in forecasts:
        raise ValueError("forecasts must contain a base case.")
    rows = []
    for mode in modes:
        try:
            result = solve_reverse_dcf(
                base_revenue, forecasts["base"], target_share_price, wacc,
                terminal_growth, cash, debt, shares_outstanding, mode=mode,
            )
            rows.append({key: result[key] for key in (
                "mode", "implied_assumption", "target_share_price", "implied_share_price",
                "price_residual", "converged", "iterations", "lower_bound", "upper_bound",
                "wacc", "terminal_growth",
            )} | {"status": "converged", "failure_reason": ""})
        except (ValueError, RuntimeError) as exc:
            rows.append({
                "mode": mode, "implied_assumption": np.nan,
                "target_share_price": target_share_price, "implied_share_price": np.nan,
                "price_residual": np.nan, "converged": False, "iterations": 0,
                "lower_bound": np.nan, "upper_bound": np.nan, "wacc": wacc,
                "terminal_growth": terminal_growth, "status": "failed",
                "failure_reason": str(exc),
            })
    implied = pd.DataFrame(rows).set_index("mode")
    comparison = value_scenarios(
        forecasts, wacc, terminal_growth, cash, debt, shares_outstanding
    ).reset_index()
    comparison["case_type"] = "Forecast"
    comparison["current_price"] = target_share_price
    comparison["upside_downside"] = comparison["implied_share_price"] / target_share_price - 1
    comparison = pd.concat([comparison, pd.DataFrame([{
        "scenario": "market_implied", "case_type": "Market implied",
        "implied_share_price": target_share_price, "current_price": target_share_price,
        "upside_downside": 0.0,
    }])], ignore_index=True)
    return implied, comparison


def build_expectation_matrix(
    base_revenue, reference_forecast, growth_values, margin_values, wacc,
    terminal_growth, cash, debt, shares_outstanding, *, current_price=None,
):
    """Return a long-form growth × operating-margin DCF expectation matrix."""
    years = len(reference_forecast)
    if years == 0:
        raise ValueError("reference_forecast cannot be empty.")
    start_year = int(reference_forecast.index[0])
    tax = _forecast_driver(reference_forecast, "tax_rate", 0.21)
    da = ((reference_forecast["da"] / reference_forecast["revenue"]).astype(float).tolist()
          if {"da", "revenue"}.issubset(reference_forecast.columns) else [0.0] * years)
    capex = ((reference_forecast["capex"] / reference_forecast["revenue"]).astype(float).tolist()
             if {"capex", "revenue"}.issubset(reference_forecast.columns) else [0.0] * years)
    incremental_revenue = pd.to_numeric(reference_forecast["revenue"], errors="coerce").diff()
    incremental_revenue.iloc[0] = float(reference_forecast["revenue"].iloc[0]) - base_revenue
    nwc = (pd.to_numeric(reference_forecast.get("change_nwc", 0.0), errors="coerce") /
           incremental_revenue.replace(0, np.nan)).fillna(0.0).astype(float).tolist()
    rows = []
    for growth in growth_values:
        for margin in margin_values:
            forecast = build_forecast_from_inputs(
                base_revenue, [float(growth)] * years, [float(margin)] * years,
                tax, da, capex, nwc, scenario="expectation_matrix", start_year=start_year,
            )
            price = run_dcf(forecast, wacc, terminal_growth, cash, debt,
                            shares_outstanding)["implied_share_price"]
            rows.append({
                "revenue_growth": float(growth), "operating_margin": float(margin),
                "implied_share_price": price, "current_price": current_price,
                "valuation_gap": price - current_price if current_price is not None else np.nan,
                "upside_downside": price / current_price - 1 if current_price else np.nan,
                "status": "available",
            })
    return pd.DataFrame(rows)
