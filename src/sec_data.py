import os
import time
import requests
import pandas as pd

SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Common US-GAAP tags. The loader tries aliases because companies use
# slightly different XBRL tags.
ALIASES = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "interest_expense": [
        "InterestExpenseNonOperating",
        "InterestAndDebtExpense",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "cfi": ["NetCashProvidedByUsedInInvestingActivities"],
    "cff": ["NetCashProvidedByUsedInFinancingActivities"],
    "debt_current": ["LongTermDebtCurrent", "ShortTermBorrowings"],
    "debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "buybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfCommonStockAndOther",
    ],
}

def fetch_companyfacts(cik: str) -> dict:
    ua = os.getenv("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "Set SEC_USER_AGENT in .env. SEC requires a descriptive User-Agent."
        )
    url = SEC_URL.format(cik=str(cik).zfill(10))
    r = requests.get(url, headers={"User-Agent": ua}, timeout=30)
    r.raise_for_status()
    return r.json()

def _pick_tag(facts, aliases):
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in aliases:
        if tag in usgaap:
            return tag, usgaap[tag]
    return None, None

def _annual_values(tag_obj):
    rows = []
    if not tag_obj:
        return pd.DataFrame(columns=["fy", "form", "filed", "value"])

    for unit, observations in tag_obj.get("units", {}).items():
        for x in observations:
            if x.get("form") != "10-K":
                continue
            if x.get("fp") not in (None, "FY"):
                continue
            if "fy" not in x or "val" not in x:
                continue
            rows.append({
                "fy": int(x["fy"]),
                "filed": x.get("filed"),
                "value": x["val"],
                "unit": unit,
                "start": x.get("start"),
                "end": x.get("end"),
            })

    if not rows:
        return pd.DataFrame(columns=["fy", "form", "filed", "value"])

    df = pd.DataFrame(rows)
    # Keep the latest filed observation for each fiscal year.
    df = (
        df.sort_values(["fy", "filed"])
          .drop_duplicates("fy", keep="last")
          .sort_values("fy")
          .reset_index(drop=True)
    )
    return df

def build_historical(cik: str, years: int = 5) -> pd.DataFrame:
    facts = fetch_companyfacts(cik)
    out = None
    selected_tags = {}

    for metric, aliases in ALIASES.items():
        tag, obj = _pick_tag(facts, aliases)
        selected_tags[metric] = tag
        df = _annual_values(obj)
        if df.empty:
            continue
        s = df.set_index("fy")["value"].rename(metric)
        out = s.to_frame() if out is None else out.join(s, how="outer")

    if out is None or out.empty:
        raise RuntimeError("No annual XBRL data found.")

    out = out.sort_index().tail(years)
    out.index.name = "fiscal_year"

    # Convert SEC dollar values to $ millions.
    numeric_cols = out.select_dtypes("number").columns
    out[numeric_cols] = out[numeric_cols] / 1_000_000

    out.attrs["selected_tags"] = selected_tags
    return out

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    df = build_historical("0001403161", years=5)
    print(df)
    print("\nSelected SEC tags:")
    print(df.attrs["selected_tags"])
