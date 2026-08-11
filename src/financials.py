import pandas as pd
import numpy as np


FLOW_METRICS = {
    "revenue",
    "operating_income",
    "pretax_income",
    "tax_expense",
    "net_income",
    "cfo",
    "capex",
    "cfi",
    "cff",
    "dividends",
    "buybacks",
    "depreciation",
}


BALANCE_METRICS = {
    "cash",
    "accounts_receivable",
    "accounts_payable",
    "current_assets",
    "current_liabilities",
    "ppe",
    "total_assets",
    "total_liabilities",
    "equity",
    "short_term_debt",
    "long_term_debt",
}


def prepare_raw_financials(df):

    data = df.copy()

    data["start"] = pd.to_datetime(
        data["start"],
        errors="coerce",
    )

    data["end"] = pd.to_datetime(
        data["end"],
        errors="coerce",
    )

    data["filed"] = pd.to_datetime(
        data["filed"],
        errors="coerce",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data["days"] = (
        data["end"] - data["start"]
    ).dt.days

    return data

def classify_fact(row):

    metric = row["metric"]

    if metric in BALANCE_METRICS:
        return "instant"

    if metric not in FLOW_METRICS:
        return "unknown"

    days = row["days"]

    if pd.isna(days):
        return "unknown"

    if 70 <= days <= 110:
        return "quarter"

    if 150 <= days <= 210:
        return "six_month_ytd"

    if 240 <= days <= 300:
        return "nine_month_ytd"

    if 330 <= days <= 380:
        return "annual"

    return "other"


def classify_financials(df):

    data = prepare_raw_financials(df)

    data["period_type"] = data.apply(
        classify_fact,
        axis=1,
    )

    return data

def deduplicate_financials(df):

    data = classify_financials(df)

    # Prefer direct filing XBRL over CompanyFacts
    # when both provide the same observation.

    if "source" not in data.columns:
        data["source"] = "companyfacts"

    source_priority = {
        "companyfacts": 1,
        "filing_xbrl": 2,
    }

    data["source_priority"] = (
        data["source"]
        .map(source_priority)
        .fillna(0)
    )

    data = data.sort_values(
        [
            "metric",
            "start",
            "end",
            "period_type",
            "source_priority",
            "filed",
        ]
    )

    data = data.drop_duplicates(
        subset=[
            "metric",
            "start",
            "end",
            "period_type",
        ],
        keep="last",
    )

    return data.reset_index(drop=True)

def build_quarterly_flows(df):

    data = deduplicate_financials(df)

    flows = data[
        data["metric"].isin(FLOW_METRICS)
    ].copy()

    quarters = flows[
        flows["period_type"] == "quarter"
    ].copy()

    quarters = (
        quarters
        .sort_values(
            [
                "metric",
                "end",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=[
                "metric",
                "end",
            ],
            keep="last",
        )
    )

    return quarters

def build_latest_balance_sheet(df):

    data = deduplicate_financials(df)

    balance = data[
        data["metric"].isin(
            BALANCE_METRICS
        )
    ].copy()

    latest_date = (
        balance["end"].max()
    )

    latest = balance[
        balance["end"] == latest_date
    ].copy()

    latest = (
        latest
        .sort_values(
            [
                "metric",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=["metric"],
            keep="last",
        )
    )

    return latest

def build_ltm(
    annual_df,
    quarterly_df,
    fiscal_year=2026,
):
    """
    Build LTM financials using:

        LTM
        =
        Prior FY
        - Prior-year 9M YTD
        + Current-year 9M YTD

    Example for Visa:
        FY2025
        - 9M FY2025
        + 9M FY2026
    """

    annual = annual_df.copy()

    annual["fy"] = pd.to_numeric(
        annual["fy"],
        errors="coerce",
    )

    # ---------------------------------------------
    # Normalize quarterly / YTD facts
    # ---------------------------------------------

    q = deduplicate_financials(
        quarterly_df
    )

    # ---------------------------------------------
    # Prior fiscal year
    # ---------------------------------------------

    prior_year = fiscal_year - 1

    annual_prior = annual[
        annual["fy"] == prior_year
    ].copy()

    annual_prior = (
        annual_prior
        .sort_values(
            ["metric", "filed"]
        )
        .drop_duplicates(
            subset=["metric"],
            keep="last",
        )
        .set_index("metric")
    )

    # ---------------------------------------------
    # June 30 comparative periods
    # ---------------------------------------------

    prior_ytd_end = pd.Timestamp(
        f"{prior_year}-06-30"
    )

    current_ytd_end = pd.Timestamp(
        f"{fiscal_year}-06-30"
    )

    # ---------------------------------------------
    # Select nine-month YTD facts
    # ---------------------------------------------

    prior_9m = q[
        (q["end"] == prior_ytd_end)
        &
        (
            q["period_type"]
            == "nine_month_ytd"
        )
    ].copy()

    current_9m = q[
        (q["end"] == current_ytd_end)
        &
        (
            q["period_type"]
            == "nine_month_ytd"
        )
    ].copy()

    # Prefer latest / direct filing observation
    prior_9m = (
        prior_9m
        .sort_values(
            [
                "metric",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=["metric"],
            keep="last",
        )
        .set_index("metric")
    )

    current_9m = (
        current_9m
        .sort_values(
            [
                "metric",
                "filed",
            ]
        )
        .drop_duplicates(
            subset=["metric"],
            keep="last",
        )
        .set_index("metric")
    )

    # ---------------------------------------------
    # Construct LTM
    # ---------------------------------------------

    results = []

    metrics = (
        FLOW_METRICS
        .intersection(
            annual_prior.index
        )
        .intersection(
            prior_9m.index
        )
        .intersection(
            current_9m.index
        )
    )

    for metric in sorted(metrics):

        fy_value = float(
            annual_prior.loc[
                metric,
                "value"
            ]
        )

        prior_value = float(
            prior_9m.loc[
                metric,
                "value"
            ]
        )

        current_value = float(
            current_9m.loc[
                metric,
                "value"
            ]
        )

        ltm_value = (
            fy_value
            - prior_value
            + current_value
        )

        results.append(
            {
                "metric": metric,
                "fy2025": fy_value,
                "prior_9m": prior_value,
                "current_9m": current_value,
                "ltm": ltm_value,
            }
        )

    return pd.DataFrame(results)

if __name__ == "__main__":

    from pathlib import Path

    ROOT = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    annual = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "financials_annual.csv"
    )

    quarterly = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "financials_quarterly.csv"
    )

    clean = deduplicate_financials(
        quarterly
    )

    q = build_quarterly_flows(
        quarterly
    )

    balance = build_latest_balance_sheet(
        quarterly
    )

    ltm = build_ltm(
        annual,
        quarterly,
        fiscal_year=2026,
    )

    print(
        "\n=============================="
    )
    print("LATEST QUARTERLY FLOWS")
    print("==============================")

    recent_q = q[
        q["end"]
        >= pd.Timestamp(
            "2025-09-30"
        )
    ]

    print(
        recent_q[
            [
                "metric",
                "start",
                "end",
                "value",
                "period_type",
                "source",
            ]
        ]
        .sort_values(
            [
                "metric",
                "end",
            ]
        )
        .to_string(index=False)
    )


    print(
        "\n=============================="
    )
    print("LATEST BALANCE SHEET")
    print("==============================")

    print(
        balance[
            [
                "metric",
                "end",
                "value",
                "source",
            ]
        ]
        .sort_values("metric")
        .to_string(index=False)
    )


    print(
        "\n=============================="
    )
    print("LTM FY2026")
    print("==============================")

    print(
        ltm.sort_values(
            "metric"
        ).to_string(
            index=False
        )
    )