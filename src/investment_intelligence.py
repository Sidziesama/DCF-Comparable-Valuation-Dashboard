"""Generic quantitative investment-intelligence calculations and output schema."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


SCHEMA_COLUMNS = [
    "category", "metric", "scope", "scenario", "period", "value", "units",
    "source", "lineage", "status", "quality", "interpretation",
]


def _number(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def safe_divide(numerator, denominator) -> float:
    numerator, denominator = _number(numerator), _number(denominator)
    return numerator / denominator if np.isfinite(numerator) and denominator != 0 else np.nan


def cagr(first, last, periods: int) -> float:
    first, last = _number(first), _number(last)
    if periods <= 0 or first <= 0 or last < 0:
        return np.nan
    return (last / first) ** (1 / periods) - 1


def calculate_roic(ebit, tax_rate, debt, equity, cash) -> float:
    """NOPAT / invested capital; NOPAT=EBIT*(1-tax), IC=debt+equity-cash."""
    values = [_number(x) for x in (ebit, tax_rate, debt, equity, cash)]
    if not all(np.isfinite(values)):
        return np.nan
    ebit, tax_rate, debt, equity, cash = values
    return safe_divide(ebit * (1 - tax_rate), debt + equity - cash)


def calculate_fcf_conversion(fcf, nopat) -> float:
    return safe_divide(fcf, nopat)


def calculate_incremental_margin(current_ebit, prior_ebit, current_revenue, prior_revenue) -> float:
    return safe_divide(_number(current_ebit) - _number(prior_ebit),
                       _number(current_revenue) - _number(prior_revenue))


def _row(category, metric, scope, scenario, period, value, units, source, lineage,
         *, status=None, quality=None, interpretation="") -> dict:
    value = _number(value)
    available = np.isfinite(value)
    return {
        "category": category, "metric": metric, "scope": scope,
        "scenario": scenario, "period": str(period), "value": value,
        "units": units, "source": source, "lineage": lineage,
        "status": status or ("available" if available else "not_available"),
        "quality": quality or ("calculated" if available else "insufficient_data"),
        "interpretation": interpretation if available else "",
    }


def build_business_quality_metrics(
    historical: pd.DataFrame,
    forecasts: Mapping[str, pd.DataFrame] | None = None,
    latest_balance: pd.DataFrame | None = None,
    statements: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Return reusable quality metrics, retaining explicit NA observations."""
    rows: list[dict] = []
    hist = historical.copy()
    annual = hist.loc[hist.index.astype(str) != "LTM"]
    source = "SEC normalized financials"
    lineage = "normalized/historical_model.csv"
    latest_map = {}
    if latest_balance is not None and not latest_balance.empty and {"metric", "value"}.issubset(latest_balance.columns):
        latest_map = latest_balance.set_index("metric")["value"].to_dict()

    previous = None
    for period, item in hist.iterrows():
        revenue = item.get("revenue")
        ebit = item.get("operating_income", item.get("ebit"))
        ebitda = item.get("ebitda")
        fcf = item.get("fcf", item.get("fcff"))
        tax = item.get("effective_tax_rate", item.get("tax_rate"))
        nopat = _number(ebit) * (1 - _number(tax)) if np.isfinite(_number(tax)) else np.nan
        equity = latest_map.get("equity") if str(period) == "LTM" else item.get("equity")
        if not np.isfinite(_number(equity)):
            assets = latest_map.get("total_assets") if str(period) == "LTM" else item.get("total_assets")
            liabilities = latest_map.get("total_liabilities") if str(period) == "LTM" else item.get("total_liabilities")
            equity = _number(assets) - _number(liabilities)
        short_debt = latest_map.get("short_term_debt") if str(period) == "LTM" else item.get("short_term_debt")
        long_debt = latest_map.get("long_term_debt") if str(period) == "LTM" else item.get("long_term_debt")
        cash = latest_map.get("cash") if str(period) == "LTM" else item.get("cash")
        debt = _number(short_debt) + _number(long_debt)
        invested_capital = debt + _number(equity) - _number(cash)
        metrics = {
            "Operating margin": (safe_divide(ebit, revenue), "ratio"),
            "EBITDA margin": (safe_divide(ebitda, revenue), "ratio"),
            "FCF margin": (safe_divide(fcf, revenue), "ratio"),
            "ROIC": (safe_divide(nopat, invested_capital), "ratio"),
            "ROE": (safe_divide(item.get("net_income"), equity), "ratio"),
            "ROA": (safe_divide(item.get("net_income"), item.get("total_assets")), "ratio"),
            "FCF conversion": (calculate_fcf_conversion(fcf, nopat), "ratio"),
            "Capex / revenue": (safe_divide(item.get("capex"), revenue), "ratio"),
            "Working-capital efficiency": (safe_divide(
                _number(item.get("accounts_receivable")) - _number(item.get("accounts_payable")), revenue), "ratio"),
            "Debt / EBITDA": (safe_divide(debt, ebitda), "multiple"),
            "Net debt / EBITDA": (safe_divide(debt - _number(cash), ebitda), "multiple"),
            "Interest coverage": (safe_divide(ebit, item.get("interest_expense")), "multiple"),
            "Dividends / net income": (safe_divide(item.get("dividends"), item.get("net_income")), "ratio"),
            "Buybacks / net income": (safe_divide(item.get("buybacks"), item.get("net_income")), "ratio"),
        }
        if previous is not None:
            metrics.update({
                "Revenue growth": (safe_divide(_number(revenue) - _number(previous.get("revenue")), previous.get("revenue")), "ratio"),
                "EBITDA growth": (safe_divide(_number(ebitda) - _number(previous.get("ebitda")), previous.get("ebitda")), "ratio"),
                "EBIT growth": (safe_divide(_number(ebit) - _number(previous.get("operating_income", previous.get("ebit"))), previous.get("operating_income", previous.get("ebit"))), "ratio"),
                "FCF growth": (safe_divide(_number(fcf) - _number(previous.get("fcf", previous.get("fcff"))), previous.get("fcf", previous.get("fcff"))), "ratio"),
                "Incremental operating margin": (calculate_incremental_margin(ebit, previous.get("operating_income", previous.get("ebit")), revenue, previous.get("revenue")), "ratio"),
                "Share-count change": (safe_divide(_number(item.get("diluted_shares")) - _number(previous.get("diluted_shares")), previous.get("diluted_shares")), "ratio"),
            })
        for metric, (value, units) in metrics.items():
            detail = "NOPAT=EBIT×(1-tax); invested capital=debt+equity-cash." if metric == "ROIC" else ""
            rows.append(_row("business_quality", metric, "historical", "historical", period,
                             value, units, source, lineage, interpretation=detail))
        previous = item

    if len(annual) >= 2:
        span = f"{annual.index[0]}-{annual.index[-1]}"
        for metric, column in (("Revenue CAGR", "revenue"), ("EBITDA CAGR", "ebitda"),
                               ("EBIT CAGR", "operating_income"), ("FCF CAGR", "fcf")):
            value = cagr(annual.iloc[0].get(column), annual.iloc[-1].get(column), len(annual) - 1)
            rows.append(_row("business_quality", metric, "historical", "historical", span,
                             value, "ratio", source, lineage))

    for scenario, forecast in (forecasts or {}).items():
        prior_revenue = hist.loc["LTM"].get("revenue") if "LTM" in hist.index else np.nan
        prior_ebit = hist.loc["LTM"].get("operating_income") if "LTM" in hist.index else np.nan
        for period, item in forecast.iterrows():
            revenue, ebit, nopat, fcff = (item.get(x) for x in ("revenue", "ebit", "nopat", "fcff"))
            metrics = {
                "Revenue growth": (item.get("revenue_growth"), "ratio"),
                "Operating margin": (safe_divide(ebit, revenue), "ratio"),
                "EBITDA margin": (safe_divide(item.get("ebitda"), revenue), "ratio"),
                "FCF margin": (safe_divide(fcff, revenue), "ratio"),
                "FCF conversion": (calculate_fcf_conversion(fcff, nopat), "ratio"),
                "Reinvestment rate": (safe_divide(_number(nopat) - _number(fcff), nopat), "ratio"),
                "Capex / revenue": (safe_divide(item.get("capex"), revenue), "ratio"),
                "Incremental operating margin": (calculate_incremental_margin(ebit, prior_ebit, revenue, prior_revenue), "ratio"),
            }
            if statements and scenario in statements:
                bs = statements[scenario]["balance_sheet"].loc[period]
                debt = _number(bs.get("short_term_debt")) + _number(bs.get("long_term_debt"))
                metrics["ROIC"] = (calculate_roic(ebit, item.get("tax_rate"), debt, bs.get("equity"), bs.get("cash")), "ratio")
            else:
                metrics["ROIC"] = (np.nan, "ratio")
            for metric, (value, units) in metrics.items():
                rows.append(_row("business_quality", metric, "forecast", scenario, period, value,
                                 units, "three-statement model", f"model/fcff_forecast_{scenario}.csv"))
            prior_revenue, prior_ebit = revenue, ebit
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


def combine_investment_intelligence(*frames: pd.DataFrame) -> pd.DataFrame:
    normalized = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        item = frame.copy()
        for column in SCHEMA_COLUMNS:
            if column not in item:
                item[column] = "" if column != "value" else np.nan
        normalized.append(item[SCHEMA_COLUMNS])
    return pd.concat(normalized, ignore_index=True) if normalized else pd.DataFrame(columns=SCHEMA_COLUMNS)
