from pathlib import Path

import yaml
from dotenv import load_dotenv
from filings import get_latest_10q
import pandas as pd
import os

from filing_xbrl import (
    fetch_latest_filing_facts,
)

from sec_data import (
    build_raw_dataset,
    build_annual_dataset,
    build_quarterly_dataset,
)

from validation import (
    validate_financial_data,
    print_validation_report,
    validate_data_freshness,
    print_freshness_report,
)

from market_data import get_market_data

from model import (
    forecast_financials,
    calculate_wacc,
    dcf_valuation,
    sensitivity_table,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config():

    with open(
        ROOT / "config/company.yaml",
        "r"
    ) as f:

        return yaml.safe_load(f)


def run():

    # ---------------------------------------------------------
    # 1. Load environment + configuration
    # ---------------------------------------------------------

    load_dotenv(
        ROOT / ".env"
    )

    cfg = load_config()

    processed = (
        ROOT / "data" / "processed"
    )

    processed.mkdir(
        exist_ok=True
    )


    # ---------------------------------------------------------
    # 2. SEC DATA
    # ---------------------------------------------------------

    print("\nFetching SEC data...")

    raw = build_raw_dataset(
        cfg["cik"]
    )

    annual = build_annual_dataset(
        raw,
        cfg["historical_years"]
    )

    quarterly = build_quarterly_dataset(
        raw
    )
    latest_10q = get_latest_10q(
    cfg["cik"]
    )

    freshness = validate_data_freshness(
    quarterly,
    latest_10q,
    )

    print_freshness_report(
    freshness
    )  
    if not freshness["is_current"]:
        print(
        "\nCompanyFacts is stale."
    )

        print(
        "Falling back to direct "
        "filing XBRL..."
    )

        filing_facts = (
            fetch_latest_filing_facts(
                cik=cfg["cik"],
                filing=latest_10q,
                user_agent=os.getenv(
                    "SEC_USER_AGENT"
                ),
            )
        )

        # -----------------------------------------------------
        # Align schemas before concatenating
        # -----------------------------------------------------

        if "source" not in raw.columns:
            raw["source"] = (
                "companyfacts"
            )

        if "xbrl_tag" not in raw.columns:
            raw["xbrl_tag"] = None

        common_columns = [
            "metric",
            "value",
            "unit",
            "fy",
            "fp",
            "form",
            "filed",
            "start",
            "end",
            "frame",
            "xbrl_tag",
            "source",
        ]

        for column in common_columns:

            if column not in raw.columns:
                raw[column] = None

            if column not in filing_facts.columns:
                filing_facts[column] = None

        raw = pd.concat(
            [
                raw[common_columns],
                filing_facts[
                    common_columns
                ],
            ],
            ignore_index=True,
        )

        # -----------------------------------------------------
        # Rebuild quarterly dataset
        # -----------------------------------------------------

        quarterly = (
            build_quarterly_dataset(
                raw
            )
        )

        # -----------------------------------------------------
        # Check freshness again
        # -----------------------------------------------------

        freshness = (
            validate_data_freshness(
                quarterly,
                latest_10q,
            )
        )

        print_freshness_report(
            freshness
        )

        if not freshness[
            "is_current"
        ]:

            raise RuntimeError(
                "\nDirect filing fallback "
                "failed to update the dataset."
            )

    # ---------------------------------------------------------
    # 3. DATA VALIDATION
    # ---------------------------------------------------------

    validation = validate_financial_data(
        raw
    )

    print_validation_report(
        validation
    )


    # ---------------------------------------------------------
    # 4. SAVE PROCESSED DATA
    # ---------------------------------------------------------

    raw.to_csv(
        processed / "sec_raw.csv",
        index=False
    )

    annual.to_csv(
        processed / "financials_annual.csv",
        index=False
    )

    quarterly.to_csv(
        processed / "financials_quarterly.csv",
        index=False
    )

    print(
        "\nFinancial data saved to:"
    )

    print(
        processed
    )


    # ---------------------------------------------------------
    # 5. MARKET DATA
    # ---------------------------------------------------------

    print("\nFetching market data...")

    market = get_market_data(
        cfg["ticker"]
    )


    # ---------------------------------------------------------
    # 6. TEMPORARY HISTORICAL DATA
    # ---------------------------------------------------------
    #
    # The current model.py still expects the old
    # wide-format historical dataframe.
    #
    # For this step we create that structure from
    # the new annual dataset.
    #
    # We will replace this temporary bridge in
    # Update 2 when we build the real three-statement model.
    # ---------------------------------------------------------

    hist = (
        annual
        .pivot_table(
            index="fy",
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .sort_index()
    )

    hist.index.name = "fiscal_year"


    # ---------------------------------------------------------
    # 7. FORECAST
    # ---------------------------------------------------------

    assumptions = cfg["assumptions"]

    wacc = calculate_wacc(
        market,
        assumptions
    )

    forecast = forecast_financials(
        hist,
        assumptions
    )


    # ---------------------------------------------------------
    # 8. DCF INPUTS
    # ---------------------------------------------------------

    shares = (
        market["shares_outstanding"]
        / 1_000_000
    )

    cash = (
        market.get("cash") or 0
    ) / 1_000_000

    debt = (
        market.get("total_debt") or 0
    ) / 1_000_000


    # ---------------------------------------------------------
    # 9. DCF VALUATION
    # ---------------------------------------------------------

    valuation = dcf_valuation(
        forecast,
        wacc["wacc"],
        assumptions["terminal_growth"],
        cash,
        debt,
        shares,
    )


    # ---------------------------------------------------------
    # 10. SENSITIVITY ANALYSIS
    # ---------------------------------------------------------

    sensitivity = sensitivity_table(
        forecast,

        [
            0.07,
            0.075,
            0.08,
            0.085,
            0.09,
            0.095,
            0.10,
        ],

        [
            0.015,
            0.02,
            0.025,
            0.03,
            0.035,
        ],

        cash,
        debt,
        shares,
    )


    # ---------------------------------------------------------
    # 11. SAVE MODEL OUTPUTS
    # ---------------------------------------------------------

    forecast.to_csv(
        processed / "forecast.csv"
    )

    sensitivity.to_csv(
        processed / "dcf_sensitivity.csv"
    )


    # ---------------------------------------------------------
    # 12. PRINT SUMMARY
    # ---------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "VISA VALUATION"
    )

    print(
        "=============================="
    )

    print(
        f"Share Price: "
        f"${market['price']:.2f}"
    )

    print(
        f"Market Cap: "
        f"${market['market_cap'] / 1e9:.1f}B"
    )

    print(
        f"WACC: "
        f"{wacc['wacc']:.2%}"
    )

    print(
        f"Terminal Growth: "
        f"{assumptions['terminal_growth']:.2%}"
    )

    print(
        f"DCF Intrinsic Price: "
        f"${valuation['intrinsic_price']:.2f}"
    )

    print(
        f"Enterprise Value: "
        f"${valuation['enterprise_value']:.1f}M"
    )

    print(
        "\nDCF Sensitivity:"
    )

    print(
        sensitivity.round(2)
    )


if __name__ == "__main__":

    run()

