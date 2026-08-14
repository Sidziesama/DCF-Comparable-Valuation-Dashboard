"""Linked three-statement forecast built from saved pipeline outputs.

The schedule is intentionally explicit: operating assumptions drive the income
statement and working-capital/PPE balances; equity rolls through earnings and
capital returns; cash is the balance-sheet plug; and the resulting financing
requirement is shown separately in the cash-flow statement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

try:
    from forecast_model import build_all_scenarios
except ImportError:  # Package import used by tests and notebooks.
    from .forecast_model import build_all_scenarios


def _value(balance: Mapping[str, float], name: str, default: float = 0.0) -> float:
    return float(balance.get(name, default))


def build_three_statement_forecast(
    historical: pd.DataFrame,
    latest_balance: pd.DataFrame,
    operating_forecast: pd.DataFrame,
    assumptions: Mapping | None = None,
) -> dict[str, pd.DataFrame]:
    """Return linked statements and reconciliation checks in $ millions."""
    assumptions = dict(assumptions or {})
    base = historical.loc["LTM"]
    balance = latest_balance.set_index("metric")["value"].to_dict()
    base_revenue = float(base["revenue"])

    ar_pct = assumptions.get("accounts_receivable_pct_revenue", _value(balance, "accounts_receivable") / base_revenue)
    ap_pct = assumptions.get("accounts_payable_pct_revenue", _value(balance, "accounts_payable") / base_revenue)
    interest_rate = float(assumptions.get("interest_rate", 0.035))
    dividend_pct = float(assumptions.get("dividends_pct_net_income", 0.22))
    buyback_pct = float(assumptions.get("buybacks_pct_net_income", 0.70))

    debt = _value(balance, "short_term_debt") + _value(balance, "long_term_debt")
    other_current_assets = _value(balance, "current_assets") - _value(balance, "cash") - _value(balance, "accounts_receivable")
    other_assets = _value(balance, "total_assets") - _value(balance, "current_assets") - _value(balance, "ppe")
    other_current_liabilities = _value(balance, "current_liabilities") - _value(balance, "accounts_payable") - _value(balance, "short_term_debt")
    other_long_term_liabilities = _value(balance, "total_liabilities") - _value(balance, "current_liabilities") - _value(balance, "long_term_debt")

    previous_ar = _value(balance, "accounts_receivable")
    previous_ap = _value(balance, "accounts_payable")
    previous_ppe = _value(balance, "ppe")
    previous_cash = _value(balance, "cash")
    previous_equity = _value(balance, "equity")
    income_rows, balance_rows, cash_rows, check_rows = [], [], [], []

    for year, operating in operating_forecast.iterrows():
        revenue = float(operating["revenue"])
        ebit = float(operating["ebit"])
        da = float(operating["da"])
        capex = float(operating["capex"])
        interest_expense = debt * interest_rate
        pretax_income = ebit - interest_expense
        tax_rate = float(operating["tax_rate"])
        tax_expense = max(pretax_income, 0.0) * tax_rate
        net_income = pretax_income - tax_expense
        dividends = net_income * dividend_pct
        buybacks = net_income * buyback_pct

        accounts_receivable = revenue * ar_pct
        accounts_payable = revenue * ap_pct
        ppe = previous_ppe + capex - da
        equity = previous_equity + net_income - dividends - buybacks
        current_liabilities = accounts_payable + _value(balance, "short_term_debt") + other_current_liabilities
        total_liabilities = current_liabilities + _value(balance, "long_term_debt") + other_long_term_liabilities
        noncash_assets = accounts_receivable + other_current_assets + ppe + other_assets
        cash = total_liabilities + equity - noncash_assets
        current_assets = cash + accounts_receivable + other_current_assets
        total_assets = current_assets + ppe + other_assets

        change_ar = accounts_receivable - previous_ar
        change_ap = accounts_payable - previous_ap
        change_nwc = change_ar - change_ap
        cfo = net_income + da - change_nwc
        cfi = -capex
        pre_financing_cash_change = cfo + cfi - dividends - buybacks
        cash_change = cash - previous_cash
        other_financing = cash_change - pre_financing_cash_change
        cff = -dividends - buybacks + other_financing

        income_rows.append({"year": year, "revenue": revenue, "operating_expenses": revenue - ebit, "ebit": ebit, "interest_expense": interest_expense, "pretax_income": pretax_income, "tax_expense": tax_expense, "net_income": net_income})
        balance_rows.append({"year": year, "cash": cash, "accounts_receivable": accounts_receivable, "other_current_assets": other_current_assets, "current_assets": current_assets, "ppe": ppe, "other_assets": other_assets, "total_assets": total_assets, "accounts_payable": accounts_payable, "other_current_liabilities": other_current_liabilities, "short_term_debt": _value(balance, "short_term_debt"), "current_liabilities": current_liabilities, "long_term_debt": _value(balance, "long_term_debt"), "other_long_term_liabilities": other_long_term_liabilities, "total_liabilities": total_liabilities, "equity": equity, "total_liabilities_and_equity": total_liabilities + equity})
        cash_rows.append({"year": year, "net_income": net_income, "depreciation": da, "change_in_working_capital": -change_nwc, "cfo": cfo, "capex": -capex, "cfi": cfi, "dividends": -dividends, "buybacks": -buybacks, "other_financing": other_financing, "cff": cff, "net_change_in_cash": cfo + cfi + cff, "ending_cash": cash})
        check_rows.append({"year": year, "balance_sheet_check": total_assets - total_liabilities - equity, "cash_flow_check": cash_change - (cfo + cfi + cff)})

        previous_ar, previous_ap, previous_ppe = accounts_receivable, accounts_payable, ppe
        previous_cash, previous_equity = cash, equity

    def frame(rows):
        return pd.DataFrame(rows).set_index("year")

    return {"income_statement": frame(income_rows), "balance_sheet": frame(balance_rows), "cash_flow_statement": frame(cash_rows), "checks": frame(check_rows)}


def build_and_save_all(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Build all scenarios exclusively from cached pipeline artifacts."""
    processed = root / "data" / "processed"
    historical = pd.read_csv(processed / "historical_model.csv", index_col=0)
    latest_balance = pd.read_csv(processed / "latest_balance_sheet.csv")
    with (root / "config" / "company.yaml").open() as handle:
        config = yaml.safe_load(handle)
    forecasts = build_all_scenarios(historical, config["assumptions"])
    model_assumptions = config["assumptions"].get("three_statement", {})
    results = {}
    for scenario, forecast in forecasts.items():
        statements = build_three_statement_forecast(historical, latest_balance, forecast, model_assumptions)
        results[scenario] = statements
        for name, statement in statements.items():
            statement.to_csv(processed / f"{name}_{scenario}.csv")
    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    result = build_and_save_all(project_root)
    maximum_error = max(statement["checks"].abs().to_numpy().max() for statement in result.values())
    print(f"Three-statement forecasts saved. Maximum reconciliation error: {maximum_error:,.8f}")
