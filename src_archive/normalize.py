import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Metric classification
# ---------------------------------------------------------

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


BALANCE_SHEET_METRICS = {
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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def prepare_quarterly_data(
    quarterly: pd.DataFrame,
) -> pd.DataFrame:

    df = quarterly.copy()

    df["start"] = pd.to_datetime(
        df["start"],
        errors="coerce",
    )

    df["end"] = pd.to_datetime(
        df["end"],
        errors="coerce",
    )

    df["filed"] = pd.to_datetime(
        df["filed"],
        errors="coerce",
    )

    df["days"] = (
        df["end"]
        - df["start"]
    ).dt.days

    return df


# ---------------------------------------------------------
# Observation classification
# ---------------------------------------------------------

def classify_observation(row):

    metric = row["metric"]

    # Balance sheet facts are point-in-time.
    if metric in BALANCE_SHEET_METRICS:
        return "instant"

    if metric not in FLOW_METRICS:
        return "unknown"

    days = row["days"]

    if pd.isna(days):
        return "unknown"

    # Approximate fiscal periods.
    if 70 <= days <= 110:
        return "quarter"

    if 150 <= days <= 210:
        return "six_month_ytd"

    if 240 <= days <= 300:
        return "nine_month_ytd"

    if 330 <= days <= 380:
        return "annual"

    return "other"


# ---------------------------------------------------------
# Normalized dataset
# ---------------------------------------------------------

def classify_quarterly_dataset(
    quarterly: pd.DataFrame,
) -> pd.DataFrame:

    df = prepare_quarterly_data(
        quarterly
    )

    df["observation_type"] = (
        df.apply(
            classify_observation,
            axis=1,
        )
    )

    return df

if __name__ == "__main__":

    from pathlib import Path

    ROOT = Path(
        __file__
    ).resolve().parents[1]

    path = (
        ROOT
        / "data"
        / "processed"
        / "financials_quarterly.csv"
    )

    quarterly = pd.read_csv(
        path
    )

    normalized = classify_quarterly_dataset(
        quarterly
    )

    recent = normalized[
        normalized["end"]
        >= pd.Timestamp("2024-09-30")
    ]

    columns = [
        "metric",
        "value",
        "fy",
        "fp",
        "start",
        "end",
        "days",
        "observation_type",
    ]

    print(
        recent[columns]
        .sort_values(
            ["metric", "end", "days"]
        )
        .to_string(index=False)
    )