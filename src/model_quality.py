"""Reusable model validation and historical-versus-forecast analytics.

The functions in this module are pure: they accept model dataframes and return
dataframes suitable for tests, batch exports, or presentation clients.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


DEFAULT_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_RELATIVE_TOLERANCE = 1e-9


def _number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _safe_divide(numerator, denominator) -> float:
    numerator, denominator = _number(numerator), _number(denominator)
    return numerator / denominator if np.isfinite(numerator) and denominator else np.nan


def _cagr(first, last, periods: int) -> float:
    first, last = _number(first), _number(last)
    if periods <= 0 or first <= 0 or last < 0:
        return np.nan
    return (last / first) ** (1 / periods) - 1


def _check_row(
    scenario: str,
    period,
    category: str,
    check: str,
    actual,
    expected=0.0,
    *,
    tolerance=DEFAULT_ABSOLUTE_TOLERANCE,
    available=True,
    detail="",
) -> dict:
    actual, expected, tolerance = _number(actual), _number(expected), abs(_number(tolerance))
    variance = actual - expected if available and np.isfinite(actual) and np.isfinite(expected) else np.nan
    passed = bool(available and np.isfinite(variance) and abs(variance) <= tolerance)
    return {
        "scenario": scenario, "period": period, "category": category,
        "check": check, "actual": actual, "expected": expected,
        "variance": variance, "tolerance": tolerance,
        "status": "PASS" if passed else ("N/A" if not available else "FAIL"),
        "detail": detail,
    }


def build_model_checks(
    statements: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    wacc: float | None = None,
    terminal_growth: float | None = None,
    terminal_value_pct_ev: Mapping[str, float] | None = None,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    max_terminal_value_pct: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return check-level diagnostics and a compact category summary."""
    rows: list[dict] = []
    terminal_value_pct_ev = terminal_value_pct_ev or {}
    for scenario, model in statements.items():
        income = model["income_statement"]
        balance = model["balance_sheet"]
        cash_flow = model["cash_flow_statement"]
        fcff = model["fcff_forecast"]
        engine_checks = model.get("checks", pd.DataFrame())
        opening_debt = None
        opening_equity = None
        for period in balance.index:
            bs, cf, inc, bridge = balance.loc[period], cash_flow.loc[period], income.loc[period], fcff.loc[period]
            scale = max(abs(_number(bs.get("total_assets"))), 1.0)
            statement_tol = max(absolute_tolerance, scale * relative_tolerance)
            rows.extend([
                _check_row(scenario, period, "Statements", "Balance sheet balances",
                    bs.get("total_assets"), bs.get("total_liabilities_and_equity"), tolerance=statement_tol),
                _check_row(scenario, period, "Cash", "Cash-flow statement reconciles",
                    cf.get("net_change_in_cash"), cf.get("cfo") + cf.get("cfi") + cf.get("cff"), tolerance=statement_tol),
                _check_row(scenario, period, "Cash", "Cash roll-forward",
                    cf.get("ending_cash"), cf.get("beginning_cash") + cf.get("net_change_in_cash"), tolerance=statement_tol),
                _check_row(scenario, period, "Cash", "Cash agrees to balance sheet",
                    cf.get("ending_cash"), bs.get("cash"), tolerance=statement_tol),
                _check_row(scenario, period, "FCFF", "FCFF reconciliation",
                    bridge.get("fcff"), bridge.get("nopat") + bridge.get("da") - bridge.get("capex") - bridge.get("change_nwc"), tolerance=statement_tol),
            ])
            if period in engine_checks.index:
                control = engine_checks.loc[period]
                rows.extend([
                    _check_row(scenario, period, "Debt", "Debt engine control",
                        control.get("debt_roll_forward_check"), 0.0, tolerance=statement_tol),
                    _check_row(scenario, period, "Equity", "Equity engine control",
                        control.get("equity_roll_forward_check"), 0.0, tolerance=statement_tol,
                        detail="Total-equity roll-forward; retained earnings is not separately modeled."),
                ])
            ending_debt = _number(bs.get("short_term_debt")) + _number(bs.get("long_term_debt"))
            if opening_debt is not None:
                expected_debt = opening_debt + _number(cf.get("debt_issuance")) + _number(cf.get("debt_repayment"))
                rows.append(_check_row(scenario, period, "Debt", "Debt roll-forward", ending_debt, expected_debt, tolerance=statement_tol))
            else:
                rows.append(_check_row(scenario, period, "Debt", "Debt balance is non-negative",
                    min(ending_debt, 0.0), 0.0, tolerance=absolute_tolerance))
            if opening_equity is not None:
                expected_equity = opening_equity + _number(inc.get("net_income")) + _number(cf.get("dividends")) + _number(cf.get("buybacks"))
                rows.append(_check_row(scenario, period, "Equity", "Equity / retained earnings roll-forward",
                    bs.get("equity"), expected_equity, tolerance=statement_tol,
                    detail="Uses total equity because retained earnings is not separately forecast."))
            else:
                rows.append(_check_row(scenario, period, "Equity", "Equity balance available",
                    bs.get("equity"), bs.get("equity"), tolerance=statement_tol,
                    detail="Opening retained earnings is unavailable; first-year total-equity roll-forward is tested in the model engine."))
            opening_debt, opening_equity = ending_debt, _number(bs.get("equity"))

        tv_pct = _number(terminal_value_pct_ev.get(scenario))
        rows.append(_check_row(scenario, "Terminal", "Terminal value", "WACC exceeds terminal growth",
            _number(wacc) - _number(terminal_growth), 0.0, tolerance=0.0,
            available=wacc is not None and terminal_growth is not None,
            detail="PASS requires a positive spread."))
        if rows[-1]["status"] != "N/A":
            rows[-1]["status"] = "PASS" if rows[-1]["actual"] > 0 else "FAIL"
        rows.append(_check_row(scenario, "Terminal", "Terminal value", "Terminal value concentration",
            min(tv_pct, max_terminal_value_pct), tv_pct, tolerance=absolute_tolerance,
            available=np.isfinite(tv_pct), detail=f"Threshold: {max_terminal_value_pct:.0%} of enterprise value."))

    ordered = [name for name in ("bear", "base", "bull") if name in statements]
    if len(ordered) == 3:
        metrics = ("revenue", "ebit", "fcff")
        common_periods = statements["bear"]["fcff_forecast"].index
        for period in common_periods:
            for metric in metrics:
                values = [_number(statements[name]["fcff_forecast"].loc[period, metric]) for name in ordered]
                passed = all(np.isfinite(values)) and values[0] <= values[1] <= values[2]
                row = _check_row("all", period, "Scenarios", f"Bear ≤ Base ≤ Bull: {metric}", 0.0 if passed else 1.0, 0.0)
                rows.append(row)

    detail = pd.DataFrame(rows)
    summary = (detail.assign(is_fail=detail["status"].eq("FAIL"), is_pass=detail["status"].eq("PASS"))
        .groupby(["scenario", "category"], as_index=False)
        .agg(checks=("check", "size"), passed=("is_pass", "sum"), failed=("is_fail", "sum"), max_abs_variance=("variance", lambda s: s.abs().max())))
    summary["status"] = np.where(summary["failed"].gt(0), "FAIL", "PASS")
    return detail, summary


def build_historical_forecast_analytics(
    historical: pd.DataFrame,
    forecasts: Mapping[str, pd.DataFrame],
    latest_balance: pd.DataFrame | None = None,
    statements: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build metric trends and assumption reasonableness diagnostics."""
    hist = historical.copy()
    annual = hist.loc[hist.index.astype(str) != "LTM"].copy()
    rows: list[dict] = []

    def add(scope, scenario, period, metric, value, unit="ratio"):
        rows.append({"scope": scope, "scenario": scenario, "period": period,
                     "metric": metric, "value": _number(value), "unit": unit})

    for period, row in hist.iterrows():
        revenue = row.get("revenue")
        ebit = row.get("operating_income", row.get("ebit"))
        ebitda = row.get("ebitda")
        fcf = row.get("fcf", row.get("fcff"))
        add("Historical", "historical", period, "Operating margin", _safe_divide(ebit, revenue))
        add("Historical", "historical", period, "EBITDA margin", _safe_divide(ebitda, revenue))
        add("Historical", "historical", period, "FCF margin", _safe_divide(fcf, revenue))
        invested_capital = row.get("invested_capital")
        nopat = row.get("nopat")
        if pd.isna(_number(nopat)) and np.isfinite(_number(ebit)):
            tax = row.get("effective_tax_rate", row.get("tax_rate"))
            nopat = _number(ebit) * (1 - _number(tax)) if np.isfinite(_number(tax)) else np.nan
        add("Historical", "historical", period, "FCF conversion", _safe_divide(fcf, nopat))
        add("Historical", "historical", period, "ROIC", _safe_divide(nopat, invested_capital))
        add("Historical", "historical", period, "Capex / revenue", _safe_divide(row.get("capex"), revenue))
        add("Historical", "historical", period, "Working-capital efficiency", _safe_divide(row.get("working_capital"), revenue))

    if latest_balance is not None and not latest_balance.empty and "LTM" in hist.index:
        balance = latest_balance.set_index("metric")["value"].to_dict()
        ltm = hist.loc["LTM"]
        ltm_revenue = ltm.get("revenue")
        debt = _number(balance.get("short_term_debt")) + _number(balance.get("long_term_debt"))
        invested_capital = debt + _number(balance.get("equity")) - _number(balance.get("cash"))
        ltm_ebit = ltm.get("operating_income", ltm.get("ebit"))
        ltm_tax = ltm.get("effective_tax_rate", ltm.get("tax_rate"))
        ltm_nopat = _number(ltm_ebit) * (1 - _number(ltm_tax))
        add("Historical", "historical", "LTM", "ROIC", _safe_divide(ltm_nopat, invested_capital))
        operating_nwc = _number(balance.get("accounts_receivable")) - _number(balance.get("accounts_payable"))
        add("Historical", "historical", "LTM", "Working-capital efficiency", _safe_divide(operating_nwc, ltm_revenue))

    if len(annual) >= 2:
        add("Historical", "historical", f"{annual.index[0]}-{annual.index[-1]}", "Revenue CAGR",
            _cagr(annual.iloc[0].get("revenue"), annual.iloc[-1].get("revenue"), len(annual) - 1))
        revenues = pd.to_numeric(annual.get("revenue"), errors="coerce")
        ebit = pd.to_numeric(annual.get("operating_income", annual.get("ebit")), errors="coerce")
        for period, value in revenues.pct_change(fill_method=None).items():
            add("Historical", "historical", period, "Revenue growth", value)
        for period, value in (ebit.diff() / revenues.diff()).items():
            add("Historical", "historical", period, "Incremental operating margin", value)

    for scenario, forecast in forecasts.items():
        prior_revenue = _number(hist.loc["LTM", "revenue"]) if "LTM" in hist.index else np.nan
        prior_ebit = _number(hist.loc["LTM"].get("operating_income", hist.loc["LTM"].get("ebit"))) if "LTM" in hist.index else np.nan
        for period, row in forecast.iterrows():
            revenue, ebit = row.get("revenue"), row.get("ebit")
            nopat, fcff = row.get("nopat"), row.get("fcff")
            reinvestment = _number(nopat) - _number(fcff)
            add("Forecast", scenario, period, "Revenue growth", row.get("revenue_growth"))
            add("Forecast", scenario, period, "Operating margin", _safe_divide(ebit, revenue))
            add("Forecast", scenario, period, "EBITDA margin", _safe_divide(row.get("ebitda"), revenue))
            add("Forecast", scenario, period, "FCF margin", _safe_divide(fcff, revenue))
            add("Forecast", scenario, period, "FCF conversion", _safe_divide(fcff, nopat))
            add("Forecast", scenario, period, "Reinvestment rate", _safe_divide(reinvestment, nopat))
            add("Forecast", scenario, period, "Sales-to-capital proxy", _safe_divide(_number(revenue) - prior_revenue, reinvestment), "multiple")
            add("Forecast", scenario, period, "Incremental working-capital intensity",
                _safe_divide(row.get("change_nwc"), _number(revenue) - prior_revenue))
            if statements and scenario in statements:
                forecast_balance = statements[scenario]["balance_sheet"].loc[period]
                forecast_debt = _number(forecast_balance.get("short_term_debt")) + _number(forecast_balance.get("long_term_debt"))
                forecast_capital = forecast_debt + _number(forecast_balance.get("equity")) - _number(forecast_balance.get("cash"))
                add("Forecast", scenario, period, "ROIC", _safe_divide(nopat, forecast_capital))
                forecast_nwc = _number(forecast_balance.get("accounts_receivable")) - _number(forecast_balance.get("accounts_payable"))
                add("Forecast", scenario, period, "Working-capital efficiency", _safe_divide(forecast_nwc, revenue))
            add("Forecast", scenario, period, "Incremental operating margin",
                _safe_divide(_number(ebit) - prior_ebit, _number(revenue) - prior_revenue))
            prior_revenue, prior_ebit = _number(revenue), _number(ebit)
        add("Forecast", scenario, f"{forecast.index[0]}-{forecast.index[-1]}", "Revenue CAGR",
            _cagr(hist.loc["LTM", "revenue"], forecast.iloc[-1].get("revenue"), len(forecast)))

    trends = pd.DataFrame(rows)
    historical_stats = trends[(trends.scope == "Historical") & trends.value.notna()].groupby("metric")["value"].agg(["mean", "min", "max"])
    diagnostics = []
    for scenario, forecast in forecasts.items():
        comparisons = {
            "Revenue growth": pd.to_numeric(forecast.get("revenue_growth"), errors="coerce").mean(),
            "Operating margin": pd.to_numeric(forecast.get("operating_margin"), errors="coerce").mean(),
            "FCF margin": pd.to_numeric(forecast.get("fcff_margin"), errors="coerce").mean(),
        }
        for metric, value in comparisons.items():
            available = metric in historical_stats.index and np.isfinite(value)
            low = historical_stats.loc[metric, "min"] if available else np.nan
            high = historical_stats.loc[metric, "max"] if available else np.nan
            status = "N/A" if not available else ("PASS" if low <= value <= high else "WARN")
            diagnostics.append({"scenario": scenario, "metric": metric, "forecast_average": value,
                "historical_average": historical_stats.loc[metric, "mean"] if available else np.nan,
                "historical_min": low, "historical_max": high, "status": status,
                "detail": "Forecast average is compared with the available historical range."})
    return trends, pd.DataFrame(diagnostics)
