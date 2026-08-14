"""Reusable, fully linked three-statement forecast model.

All values are in $ millions. Ending cash is calculated from the cash-flow
statement; the balance-sheet check independently proves reconciliation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

try:
    from forecast_model import build_all_scenarios
except ImportError:
    from .forecast_model import build_all_scenarios


STATEMENT_TOLERANCE = 1e-6


def _value(balance: Mapping[str, float], name: str, default: float = 0.0) -> float:
    value = balance.get(name, default)
    return default if pd.isna(value) else float(value)


def _driver(value: float | Sequence[float], offset: int) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return float(value[offset])
    return float(value)


def _safe_ratio(numerator: float, denominator: float, default: float) -> float:
    return numerator / denominator if denominator else default


def build_three_statement_forecast(
    historical: pd.DataFrame,
    latest_balance: pd.DataFrame,
    operating_forecast: pd.DataFrame,
    assumptions: Mapping | None = None,
) -> dict[str, pd.DataFrame]:
    """Build linked statements and a statement-derived FCFF forecast."""
    if "LTM" not in historical.index:
        raise ValueError("historical must contain an LTM row.")
    required = {"revenue", "ebit", "tax_rate", "da", "capex"}
    missing = required.difference(operating_forecast.columns)
    if missing:
        raise ValueError(f"operating_forecast is missing: {sorted(missing)}")

    assumptions = dict(assumptions or {})
    base = historical.loc["LTM"]
    balance = latest_balance.set_index("metric")["value"].to_dict()
    base_revenue = float(base["revenue"])
    ar_pct = assumptions.get(
        "accounts_receivable_pct_revenue",
        _safe_ratio(_value(balance, "accounts_receivable"), base_revenue, 0.03),
    )
    ap_pct = assumptions.get(
        "accounts_payable_pct_revenue",
        _safe_ratio(_value(balance, "accounts_payable"), base_revenue, 0.01),
    )
    interest_rate = assumptions.get("interest_rate", 0.035)
    dividends_pct = assumptions.get("dividends_pct_net_income", 0.22)
    buybacks_pct = assumptions.get("buybacks_pct_net_income", 0.70)
    debt_issuance = assumptions.get("debt_issuance", 0.0)
    debt_repayment = assumptions.get("debt_repayment", 0.0)

    short_term_debt = _value(balance, "short_term_debt")
    long_term_debt = _value(balance, "long_term_debt")
    opening_debt = short_term_debt + long_term_debt
    other_current_assets = (
        _value(balance, "current_assets") - _value(balance, "cash")
        - _value(balance, "accounts_receivable")
    )
    other_assets = (
        _value(balance, "total_assets") - _value(balance, "current_assets")
        - _value(balance, "ppe")
    )
    other_current_liabilities = (
        _value(balance, "current_liabilities") - _value(balance, "accounts_payable")
        - short_term_debt
    )
    other_long_term_liabilities = (
        _value(balance, "total_liabilities") - _value(balance, "current_liabilities")
        - long_term_debt
    )
    previous = {
        "ar": _value(balance, "accounts_receivable"),
        "ap": _value(balance, "accounts_payable"),
        "ppe": _value(balance, "ppe"),
        "cash": _value(balance, "cash"),
        "equity": _value(balance, "equity"),
        "debt": opening_debt,
    }
    income_rows, balance_rows, cash_rows, fcff_rows, check_rows = [], [], [], [], []

    for offset, (year, operating) in enumerate(operating_forecast.iterrows()):
        revenue, ebit = float(operating["revenue"]), float(operating["ebit"])
        da, capex = float(operating["da"]), float(operating["capex"])
        issuance = _driver(debt_issuance, offset)
        repayment = min(_driver(debt_repayment, offset), previous["debt"] + issuance)
        ending_debt = previous["debt"] + issuance - repayment
        interest_expense = (previous["debt"] + ending_debt) / 2 * _driver(interest_rate, offset)
        pretax_income = ebit - interest_expense
        tax_rate = float(operating["tax_rate"])
        tax_expense = max(pretax_income, 0.0) * tax_rate
        net_income = pretax_income - tax_expense
        dividends = max(net_income, 0.0) * _driver(dividends_pct, offset)
        buybacks = max(net_income, 0.0) * _driver(buybacks_pct, offset)

        ar, ap = revenue * _driver(ar_pct, offset), revenue * _driver(ap_pct, offset)
        change_ar, change_ap = ar - previous["ar"], ap - previous["ap"]
        change_nwc = change_ar - change_ap
        ppe = previous["ppe"] + capex - da
        equity = previous["equity"] + net_income - dividends - buybacks
        cfo = net_income + da - change_nwc
        cfi = -capex
        cff = issuance - repayment - dividends - buybacks
        net_change_cash = cfo + cfi + cff
        cash = previous["cash"] + net_change_cash

        ending_short_term_debt = min(short_term_debt, ending_debt)
        ending_long_term_debt = ending_debt - ending_short_term_debt
        current_assets = cash + ar + other_current_assets
        total_assets = current_assets + ppe + other_assets
        current_liabilities = ap + ending_short_term_debt + other_current_liabilities
        total_liabilities = current_liabilities + ending_long_term_debt + other_long_term_liabilities
        total_liabilities_equity = total_liabilities + equity
        nopat = ebit * (1 - tax_rate)
        fcff = nopat + da - capex - change_nwc
        operating_fcff = float(operating.get("fcff", fcff))
        checks = {
            "balance_sheet": total_assets - total_liabilities_equity,
            "cash_roll_forward": cash - previous["cash"] - net_change_cash,
            "ppe_roll_forward": ppe - previous["ppe"] - capex + da,
            "debt_roll_forward": ending_debt - previous["debt"] - issuance + repayment,
            "equity_roll_forward": equity - previous["equity"] - net_income + dividends + buybacks,
            "fcff_formula": fcff - (nopat + da - capex - change_nwc),
        }

        income_rows.append({"year": year, "revenue": revenue,
            "operating_expenses": revenue - ebit, "ebit": ebit,
            "interest_expense": interest_expense, "pretax_income": pretax_income,
            "tax_expense": tax_expense, "net_income": net_income})
        balance_rows.append({"year": year, "cash": cash, "accounts_receivable": ar,
            "other_current_assets": other_current_assets, "current_assets": current_assets,
            "ppe": ppe, "other_assets": other_assets, "total_assets": total_assets,
            "accounts_payable": ap, "other_current_liabilities": other_current_liabilities,
            "short_term_debt": ending_short_term_debt, "current_liabilities": current_liabilities,
            "long_term_debt": ending_long_term_debt,
            "other_long_term_liabilities": other_long_term_liabilities,
            "total_liabilities": total_liabilities, "equity": equity,
            "total_liabilities_and_equity": total_liabilities_equity})
        cash_rows.append({"year": year, "net_income": net_income, "depreciation": da,
            "change_in_working_capital": -change_nwc, "cfo": cfo, "capex": -capex,
            "cfi": cfi, "debt_issuance": issuance, "debt_repayment": -repayment,
            "dividends": -dividends, "buybacks": -buybacks, "cff": cff,
            "net_change_in_cash": net_change_cash, "beginning_cash": previous["cash"],
            "ending_cash": cash})
        fcff_rows.append({**operating.to_dict(), "year": year, "nopat": nopat,
            "change_nwc": change_nwc, "fcff": fcff, "fcff_margin": fcff / revenue,
            "ebitda": ebit + da})
        check_rows.append({"year": year,
            **{f"{name}_check": value for name, value in checks.items()},
            "operating_to_linked_fcff_variance": fcff - operating_fcff,
            "max_abs_error": max(abs(value) for value in checks.values()),
            "status": "OK" if all(abs(value) <= STATEMENT_TOLERANCE for value in checks.values()) else "ERROR"})
        previous.update(ar=ar, ap=ap, ppe=ppe, cash=cash, equity=equity, debt=ending_debt)

    def frame(rows):
        return pd.DataFrame(rows).set_index("year")

    return {"income_statement": frame(income_rows), "balance_sheet": frame(balance_rows),
        "cash_flow_statement": frame(cash_rows), "fcff_forecast": frame(fcff_rows),
        "checks": frame(check_rows)}


def build_and_save_all(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Build every scenario from cached inputs and save pipeline-ready CSVs."""
    import yaml

    processed = root / "data" / "processed"
    historical = pd.read_csv(processed / "historical_model.csv", index_col=0)
    latest_balance = pd.read_csv(processed / "latest_balance_sheet.csv")
    with (root / "config" / "company.yaml").open() as handle:
        config = yaml.safe_load(handle)
    forecasts = build_all_scenarios(historical, config["assumptions"])
    model_assumptions = config["assumptions"].get("three_statement", {})
    results = {}
    for scenario, operating_forecast in forecasts.items():
        statements = build_three_statement_forecast(
            historical, latest_balance, operating_forecast, model_assumptions)
        results[scenario] = statements
        for name, statement in statements.items():
            statement.to_csv(processed / f"{name}_{scenario}.csv")
        statements["fcff_forecast"].to_csv(processed / f"forecast_{scenario}.csv")
    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = build_and_save_all(project_root)
    maximum_error = max(s["checks"]["max_abs_error"].max() for s in result.values())
    if maximum_error > STATEMENT_TOLERANCE:
        raise RuntimeError(f"Three-statement model failed checks: {maximum_error:,.8f}")
    print(f"Three-statement forecasts saved. Maximum error: {maximum_error:,.8f}")
