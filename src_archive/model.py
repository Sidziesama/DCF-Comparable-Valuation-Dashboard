import numpy as np
import pandas as pd

def forecast_financials(hist, assumptions):
    last_year = int(hist.index.max())
    years = list(range(last_year + 1, last_year + 6))

    last_revenue = float(hist["revenue"].dropna().iloc[-1])
    growth = assumptions["revenue_growth"]

    rows = []
    prev_rev = last_revenue
    for i, year in enumerate(years):
        rev = prev_rev * (1 + growth[i])
        op_margin = assumptions["operating_margin"]
        ebit = rev * op_margin
        tax = ebit * assumptions["tax_rate"]
        da = rev * assumptions["da_pct_revenue"]
        capex = rev * assumptions["capex_pct_revenue"]
        nwc = rev * assumptions["nwc_pct_revenue"]
        rows.append({
            "year": year,
            "revenue": rev,
            "revenue_growth": growth[i],
            "ebit": ebit,
            "tax_on_ebit": tax,
            "nopat": ebit - tax,
            "da": da,
            "capex": capex,
            "nwc": nwc,
        })
        prev_rev = rev

    fc = pd.DataFrame(rows)
    fc["change_nwc"] = fc["nwc"].diff().fillna(
        fc["nwc"].iloc[0] - hist["revenue"].iloc[-1] * assumptions["nwc_pct_revenue"]
    )
    fc["fcff"] = fc["nopat"] + fc["da"] - fc["capex"] - fc["change_nwc"]
    return fc.set_index("year")

def calculate_wacc(market, assumptions):
    equity = market["market_cap"]
    debt = market.get("total_debt") or 0
    rf = assumptions["risk_free_rate"]
    erp = assumptions["equity_risk_premium"]
    beta = assumptions["beta"]
    ke = rf + beta * erp
    kd = assumptions["pre_tax_cost_of_debt"]
    tax = assumptions["tax_rate"]
    total = equity + debt
    wacc = (equity / total) * ke + (debt / total) * kd * (1 - tax)
    return {
        "cost_of_equity": ke,
        "pre_tax_cost_of_debt": kd,
        "after_tax_cost_of_debt": kd * (1-tax),
        "equity_weight": equity / total,
        "debt_weight": debt / total,
        "wacc": wacc,
    }

def dcf_valuation(forecast, wacc, terminal_growth, cash, debt, shares):
    years = np.arange(1, len(forecast) + 1)
    fcff = forecast["fcff"].to_numpy()

    pv_fcff = fcff / (1 + wacc) ** years
    terminal_value = fcff[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** years[-1]

    enterprise_value = pv_fcff.sum() + pv_terminal
    equity_value = enterprise_value + cash - debt
    price = equity_value / shares

    return {
        "enterprise_value": enterprise_value,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "equity_value": equity_value,
        "intrinsic_price": price,
        "pv_fcff": pv_fcff,
    }

def sensitivity_table(forecast, waccs, gs, cash, debt, shares):
    table = pd.DataFrame(index=gs, columns=waccs, dtype=float)
    for g in gs:
        for w in waccs:
            if w <= g:
                table.loc[g, w] = np.nan
                continue
            result = dcf_valuation(forecast, w, g, cash, debt, shares)
            table.loc[g, w] = result["intrinsic_price"]
    return table
