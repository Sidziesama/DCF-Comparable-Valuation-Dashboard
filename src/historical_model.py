import pandas as pd
import numpy as np

from financials import (
    build_ltm,
    build_latest_balance_sheet,
)


FLOW_METRICS = [
    "revenue",
    "operating_income",
    "pretax_income",
    "tax_expense",
    "net_income",
    "depreciation",
    "cfo",
    "capex",
]


BALANCE_METRICS = [
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
]


def build_annual_wide(
    annual_df,
    start_year=2021,
    end_year=2025,
):
    """
    Convert annual SEC observations into a clean
    year-by-year financial model.
    """

    data = annual_df.copy()

    data["fy"] = pd.to_numeric(
        data["fy"],
        errors="coerce",
    )

    data = data[
        (data["fy"] >= start_year)
        &
        (data["fy"] <= end_year)
    ]

    data = (
        data
        .sort_values(
            ["metric", "fy", "filed"]
        )
        .drop_duplicates(
            ["metric", "fy"],
            keep="last",
        )
    )

    wide = data.pivot(
        index="fy",
        columns="metric",
        values="value",
    )

    return wide.sort_index()


def append_ltm(
    annual_wide,
    annual_df,
    quarterly_df,
):
    """
    Add LTM June-2026 as the latest period.
    """

    ltm = build_ltm(
        annual_df,
        quarterly_df,
        fiscal_year=2026,
    )

    ltm_series = (
        ltm
        .set_index("metric")["ltm"]
    )

    model = annual_wide.copy()

    model.loc["LTM"] = ltm_series

    return model


def calculate_ratios(model):
    """
    Calculate historical operating and cash-flow ratios.
    """

    df = model.copy()

    # ---------------------------------------------
    # Growth
    # ---------------------------------------------

    df["revenue_growth"] = (
        df["revenue"]
        .pct_change()
    )

    # ---------------------------------------------
    # Profitability
    # ---------------------------------------------

    df["operating_margin"] = (
        df["operating_income"]
        / df["revenue"]
    )

    df["pretax_margin"] = (
        df["pretax_income"]
        / df["revenue"]
    )

    df["net_margin"] = (
        df["net_income"]
        / df["revenue"]
    )

    # ---------------------------------------------
    # Taxes
    # ---------------------------------------------

    df["effective_tax_rate"] = (
        df["tax_expense"]
        / df["pretax_income"]
    )

    # ---------------------------------------------
    # Cash flow
    # ---------------------------------------------

    df["capex_pct_revenue"] = (
        df["capex"]
        / df["revenue"]
    )

    df["cfo_margin"] = (
        df["cfo"]
        / df["revenue"]
    )

    df["fcf"] = (
        df["cfo"]
        - df["capex"]
    )
    if "depreciation" in df.columns:

        df["ebitda"] = (
            df["operating_income"]
            + df["depreciation"]
        )

        df["da_pct_revenue"] = (
            df["depreciation"]
            / df["revenue"]
        )

    df["fcf_margin"] = (
        df["fcf"]
        / df["revenue"]
    )

    # ---------------------------------------------
    # Returns
    # ---------------------------------------------

    if "total_assets" in df.columns:

        df["roa"] = (
            df["net_income"]
            / df["total_assets"]
        )

    if "equity" in df.columns:

        df["roe"] = (
            df["net_income"]
            / df["equity"]
        )

    return df

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

    historical = build_annual_wide(
        annual,
        2021,
        2025,
    )

    model = append_ltm(
        historical,
        annual,
        quarterly,
    )

    model = calculate_ratios(
        model
    )

    output = (
        ROOT
        / "data"
        / "processed"
        / "historical_model.csv"
    )

    model.to_csv(output)

    columns = [
        "revenue",
        "revenue_growth",
        "operating_income",
        "operating_margin",
        "net_income",
        "net_margin",
        "effective_tax_rate",
        "cfo",
        "capex",
        "fcf",
        "fcf_margin",
        "depreciation",
        "ebitda",
        "da_pct_revenue",
    ]

    print(
        "\n=============================="
    )

    print("HISTORICAL FINANCIAL MODEL")

    print(
        "=============================="
    )

    print(
        model[
            [
                c
                for c in columns
                if c in model.columns
            ]
        ]
        .round(4)
        .to_string()
    )

    print(
        f"\nSaved to:\n{output}"
    )