"""Pure valuation functions shared by the pipeline and dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
