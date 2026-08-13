import os
import requests
import pandas as pd


SEC_COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)


ALIASES = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],

    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ],

    "gross_profit": [
        "GrossProfit",
    ],

    "operating_expenses": [
        "OperatingExpenses",
    ],

    "operating_income": [
        "OperatingIncomeLoss",
    ],

    "interest_expense": [
        "InterestExpenseNonOperating",
        "InterestAndDebtExpense",
    ],

    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],

    "tax_expense": [
        "IncomeTaxExpenseBenefit",
    ],

    "net_income": [
        "NetIncomeLoss",
    ],

    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],

    "short_term_investments": [
        "ShortTermInvestments",
        "MarketableSecuritiesCurrent",
    ],

    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
    ],

    "current_assets": [
        "AssetsCurrent",
    ],

    "ppe": [
        "PropertyPlantAndEquipmentNet",
    ],

    "total_assets": [
        "Assets",
    ],

    "accounts_payable": [
        "AccountsPayableCurrent",
        "AccountsPayable",
    ],

    "current_liabilities": [
        "LiabilitiesCurrent",
    ],

    "short_term_debt": [
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
    ],

    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],

    "total_liabilities": [
        "Liabilities",
    ],

    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],

    "depreciation": [
        "DepreciationAndAmortization",
        "DepreciationDepletionAndAmortization",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
        "DepreciationAmortizationAndAccretionNet",
    ],

    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],

    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],

    "cfi": [
        "NetCashProvidedByUsedInInvestingActivities",
    ],

    "cff": [
        "NetCashProvidedByUsedInFinancingActivities",
    ],

    "dividends": [
        "PaymentsOfDividends",
        "PaymentsOfDividendsCommonStock",
    ],

    "buybacks": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfCommonStockAndOther",
    ],
}


def fetch_companyfacts(cik: str) -> dict:

    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is missing from .env"
        )

    url = SEC_COMPANYFACTS_URL.format(
        cik=str(cik).zfill(10)
    )

    response = requests.get(
        url,
        headers={"User-Agent": user_agent},
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def find_xbrl_tag(facts: dict, aliases: list):

    us_gaap = (
        facts
        .get("facts", {})
        .get("us-gaap", {})
    )

    for tag in aliases:

        if tag in us_gaap:
            return tag, us_gaap[tag]

    return None, None

def find_custom_xbrl_tag(
    facts: dict,
    keywords: list,
):
    """
    Search all available XBRL namespaces for a concept
    whose label/name contains the requested keywords.

    Useful when a company uses a custom taxonomy concept.
    """

    all_facts = facts.get(
        "facts",
        {}
    )

    for namespace, concepts in all_facts.items():

        for tag, obj in concepts.items():

            searchable = (
                tag.lower()
            )

            if all(
                keyword.lower() in searchable
                for keyword in keywords
            ):
                return tag, obj

    return None, None

def extract_observations(
    tag_object: dict,
    metric: str
) -> pd.DataFrame:

    if not tag_object:
        return pd.DataFrame()

    observations = []

    for unit, values in tag_object.get(
        "units", {}
    ).items():

        for item in values:

            if "val" not in item:
                continue

            observations.append({

                "metric": metric,

                "value": item["val"],

                "unit": unit,

                "fy": item.get("fy"),

                "fp": item.get("fp"),

                "form": item.get("form"),

                "filed": item.get("filed"),

                "start": item.get("start"),

                "end": item.get("end"),

                "frame": item.get("frame"),

            })

    return pd.DataFrame(observations)


def build_raw_dataset(
    cik: str
) -> pd.DataFrame:

    facts = fetch_companyfacts(cik)

    datasets = []

    selected_tags = {}

    for metric, aliases in ALIASES.items():

        tag, tag_object = find_xbrl_tag(
            facts,
            aliases
        )

        selected_tags[metric] = tag

        if tag_object:

            df = extract_observations(
                tag_object,
                metric
            )

            if not df.empty:
                datasets.append(df)

    if not datasets:
        raise RuntimeError(
            "No SEC financial data was found."
        )

    data = pd.concat(
        datasets,
        ignore_index=True
    )

    data["value"] = (
        pd.to_numeric(
            data["value"],
            errors="coerce"
        )
    )

    # SEC commonly reports financial statement
    # values in USD. Convert to USD millions.
    financial_metrics = [
        "revenue",
        "cost_of_revenue",
        "gross_profit",
        "gross_profit",
        "operating_expenses",
        "operating_income",
        "interest_expense",
        "pretax_income",
        "tax_expense",
        "net_income",
        "cash",
        "short_term_investments",
        "accounts_receivable",
        "current_assets",
        "ppe",
        "total_assets",
        "accounts_payable",
        "current_liabilities",
        "short_term_debt",
        "long_term_debt",
        "total_liabilities",
        "equity",
        "depreciation",
        "cfo",
        "capex",
        "cfi",
        "cff",
        "dividends",
        "buybacks",
    ]

    mask = data["metric"].isin(
        financial_metrics
    )

    data.loc[mask, "value"] = (
        data.loc[mask, "value"] / 1_000_000
    )

    data.attrs["selected_tags"] = selected_tags

    return data


def build_annual_dataset(
    raw: pd.DataFrame,
    years: int = 5
) -> pd.DataFrame:

    data = raw.copy()

    annual = data[
        (data["form"] == "10-K")
        &
        (
            data["fp"].isna()
            |
            (data["fp"] == "FY")
        )
    ].copy()

    annual = annual.dropna(
        subset=["fy", "value"]
    )

    annual["fy"] = annual["fy"].astype(int)

    # Latest filed observation for each
    # metric/year combination.
    annual = (
        annual
        .sort_values(
            ["metric", "fy", "filed"]
        )
        .drop_duplicates(
            ["metric", "fy"],
            keep="last"
        )
    )

    annual = annual[
        annual["fy"]
        >= annual["fy"].max() - years + 1
    ]

    return annual


def build_quarterly_dataset(
    raw: pd.DataFrame
) -> pd.DataFrame:

    data = raw.copy()

    quarterly = data[
        data["form"] == "10-Q"
    ].copy()

    quarterly = quarterly.dropna(
        subset=["end", "value"]
    )

    return quarterly.sort_values(
        ["metric", "end"]
    )


if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    cik = "0001403161"

    raw = build_raw_dataset(cik)

    annual = build_annual_dataset(
        raw,
        years=5
    )

    quarterly = build_quarterly_dataset(
        raw
    )

    print("\nRAW DATA")
    print(raw.head())

    print("\nANNUAL DATA")
    print(annual.head(20))

    print("\nQUARTERLY DATA")
    print(quarterly.head(20))

    print("\nSELECTED XBRL TAGS")

    for metric, tag in raw.attrs[
        "selected_tags"
    ].items():

        print(
            f"{metric:30} {tag}"
        )