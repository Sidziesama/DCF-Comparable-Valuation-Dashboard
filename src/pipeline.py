from pathlib import Path
import os

import pandas as pd
import yaml
from dotenv import load_dotenv


# =========================================================
# DATA INGESTION
# =========================================================

from filings import get_latest_10q

from filing_xbrl import (
    fetch_latest_filing_facts,
)

from sec_data import (
    build_raw_dataset,
    build_annual_dataset,
    build_quarterly_dataset,
)


# =========================================================
# VALIDATION
# =========================================================

from validation import (
    validate_financial_data,
    print_validation_report,
    validate_data_freshness,
    print_freshness_report,
)


# =========================================================
# FINANCIAL MODEL
# =========================================================

from financials import (
    build_latest_balance_sheet,
)

from historical_model import (
    build_annual_wide,
    append_ltm,
    calculate_ratios,
)

from forecast_model import (
    build_all_scenarios,
)


# =========================================================
# MARKET + WACC
# =========================================================

from market_data import (
    get_market_data,
    calculate_beta,
)

from wacc_model import (
    calculate_wacc,
    build_wacc_report,
)


# =========================================================
# VALUATION
# =========================================================

from valuation import (
    value_scenarios,
    build_sensitivity_table,
)

from valuation_summary import (
    build_valuation_summary,
)

# =========================================================
# PATHS
# =========================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =========================================================
# CONFIG
# =========================================================

def load_config():

    with open(
        ROOT / "config" / "company.yaml",
        "r",
    ) as f:

        return yaml.safe_load(f)


# =========================================================
# HELPER
# =========================================================

def section(title):

    print(
        "\n=============================="
    )

    print(title)

    print(
        "=============================="
    )


# =========================================================
# PIPELINE
# =========================================================

def run():

    # -----------------------------------------------------
    # 1. Environment + config
    # -----------------------------------------------------

    load_dotenv(
        ROOT / ".env"
    )

    cfg = load_config()

    processed = (
        ROOT
        / "data"
        / "processed"
    )

    processed.mkdir(
        parents=True,
        exist_ok=True,
    )

    ticker = cfg["ticker"]

    assumptions = cfg[
        "assumptions"
    ]


    # =====================================================
    # 2. SEC DATA
    # =====================================================

    section(
        "1. SEC DATA INGESTION"
    )

    print(
        f"Company: {cfg['company_name']}"
    )

    print(
        f"Ticker:  {ticker}"
    )

    raw = build_raw_dataset(
        cfg["cik"]
    )

    annual = build_annual_dataset(
        raw,
        cfg["historical_years"],
    )

    quarterly = build_quarterly_dataset(
        raw
    )


    # =====================================================
    # 3. FILING FRESHNESS
    # =====================================================

    section(
        "2. DATA FRESHNESS"
    )

    latest_10q = get_latest_10q(
        cfg["cik"]
    )

    freshness = (
        validate_data_freshness(
            quarterly,
            latest_10q,
        )
    )

    print_freshness_report(
        freshness
    )


    # -----------------------------------------------------
    # Direct filing fallback
    # -----------------------------------------------------

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

        # ---------------------------------------------
        # Align schemas
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Rebuild quarterly dataset
        # ---------------------------------------------

        quarterly = (
            build_quarterly_dataset(
                raw
            )
        )

        # ---------------------------------------------
        # Recheck freshness
        # ---------------------------------------------

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
                "Direct filing XBRL fallback "
                "failed."
            )


    # =====================================================
    # 4. DATA VALIDATION
    # =====================================================

    section(
        "3. DATA VALIDATION"
    )

    validation = (
        validate_financial_data(
            raw
        )
    )

    print_validation_report(
        validation
    )


    # =====================================================
    # 5. SAVE RAW / NORMALIZED SEC DATA
    # =====================================================

    raw.to_csv(
        processed / "sec_raw.csv",
        index=False,
    )

    annual.to_csv(
        processed
        / "financials_annual.csv",
        index=False,
    )

    quarterly.to_csv(
        processed
        / "financials_quarterly.csv",
        index=False,
    )


    # =====================================================
    # 6. HISTORICAL MODEL
    # =====================================================

    section(
        "4. HISTORICAL MODEL"
    )

    annual_wide = (
        build_annual_wide(
            annual,
            start_year=2021,
            end_year=2025,
        )
    )

    historical = append_ltm(
        annual_wide,
        annual,
        quarterly,
    )

    historical = calculate_ratios(
        historical
    )

    historical.to_csv(
        processed
        / "historical_model.csv"
    )

    historical_columns = [
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
    ]

    available_columns = [
        column
        for column in historical_columns
        if column in historical.columns
    ]

    print(
        historical[
            available_columns
        ]
        .round(4)
        .to_string()
    )


    # =====================================================
    # 7. LATEST BALANCE SHEET
    # =====================================================

    section(
        "5. LATEST BALANCE SHEET"
    )

    latest_balance = (
        build_latest_balance_sheet(
            quarterly
        )
    )

    latest_balance.to_csv(
        processed
        / "latest_balance_sheet.csv",
        index=False,
    )

    balance_map = (
        latest_balance
        .set_index("metric")[
            "value"
        ]
        .to_dict()
    )

    cash = float(
        balance_map.get(
            "cash",
            0.0,
        )
    )

    short_term_debt = float(
        balance_map.get(
            "short_term_debt",
            0.0,
        )
    )

    long_term_debt = float(
        balance_map.get(
            "long_term_debt",
            0.0,
        )
    )

    debt = (
        short_term_debt
        + long_term_debt
    )

    net_debt = (
        debt
        - cash
    )

    print(
        f"Cash:              "
        f"${cash:,.1f}M"
    )

    print(
        f"Short-Term Debt:   "
        f"${short_term_debt:,.1f}M"
    )

    print(
        f"Long-Term Debt:    "
        f"${long_term_debt:,.1f}M"
    )

    print(
        f"Total Debt:        "
        f"${debt:,.1f}M"
    )

    print(
        f"Net Debt:          "
        f"${net_debt:,.1f}M"
    )


    # =====================================================
    # 8. FORECAST MODEL
    # =====================================================

    section(
        "6. FORECAST SCENARIOS"
    )

    forecasts = (
        build_all_scenarios(
            historical,
            assumptions,
        )
    )

    for scenario, forecast in (
        forecasts.items()
    ):

        forecast.to_csv(
            processed
            / f"forecast_{scenario}.csv"
        )

        print(
            f"\n{scenario.upper()}"
        )

        print(
            forecast[
                [
                    "revenue",
                    "revenue_growth",
                    "operating_margin",
                    "ebit",
                    "nopat",
                    "fcff",
                    "fcff_margin",
                ]
            ]
            .round(4)
            .to_string()
        )


    # =====================================================
    # 9. MARKET DATA
    # =====================================================

    section(
        "7. MARKET DATA"
    )

    market = get_market_data(
        ticker
    )

    if (
        market.get("market_cap")
        is None
    ):

        raise RuntimeError(
            "Market cap unavailable."
        )

    if (
        market.get(
            "shares_outstanding"
        )
        is None
    ):

        raise RuntimeError(
            "Shares outstanding unavailable."
        )

    market_cap = (
        market["market_cap"]
        / 1_000_000
    )

    shares_outstanding = (
        market[
            "shares_outstanding"
        ]
        / 1_000_000
    )

    current_price = float(
        market["price"]
    )

    print(
        f"Share Price:       "
        f"${current_price:,.2f}"
    )

    print(
        f"Market Cap:        "
        f"${market_cap:,.1f}M"
    )

    print(
        f"Shares Outstanding:"
        f" {shares_outstanding:,.1f}M"
    )


    # =====================================================
    # 10. EMPIRICAL BETA
    # =====================================================

    section(
        "8. BETA ESTIMATION"
    )

    beta_result = calculate_beta(
        ticker=ticker,
        benchmark="SPY",
        period="5y",
    )

    empirical_beta = float(
        beta_result["beta"]
    )

    print(
        f"Benchmark:          "
        f"{beta_result['benchmark']}"
    )

    print(
        f"Observations:       "
        f"{beta_result['observations']}"
    )

    print(
        f"Beta:               "
        f"{empirical_beta:.3f}"
    )

    print(
        f"Correlation:        "
        f"{beta_result['correlation']:.3f}"
    )

    print(
        f"Stock Volatility:   "
        f"{beta_result['stock_volatility']:.2%}"
    )

    print(
        f"Market Volatility:  "
        f"{beta_result['market_volatility']:.2%}"
    )


    # =====================================================
    # 11. WACC
    # =====================================================

    section(
        "9. WACC"
    )

    wacc_assumptions = (
        assumptions["wacc"]
    )

    base_tax_rate = (
        assumptions[
            "scenarios"
        ]["base"]["tax_rate"]
    )

    wacc_result = calculate_wacc(
        market_cap=market_cap,
        debt=debt,

        risk_free_rate=(
            wacc_assumptions[
                "risk_free_rate"
            ]
        ),

        beta=empirical_beta,

        equity_risk_premium=(
            wacc_assumptions[
                "equity_risk_premium"
            ]
        ),

        pre_tax_cost_of_debt=(
            wacc_assumptions[
                "pre_tax_cost_of_debt"
            ]
        ),

        tax_rate=base_tax_rate,
    )

    wacc = float(
        wacc_result["wacc"]
    )

    wacc_report = (
        build_wacc_report(
            wacc_result
        )
    )

    wacc_report.to_csv(
        processed
        / "wacc_report.csv",
        index=False,
    )

    for key in [
        "risk_free_rate",
        "beta",
        "equity_risk_premium",
        "cost_of_equity",
        "pre_tax_cost_of_debt",
        "after_tax_cost_of_debt",
        "equity_weight",
        "debt_weight",
        "wacc",
    ]:

        value = wacc_result[key]

        if key == "beta":

            print(
                f"{key:28s}: "
                f"{value:.3f}"
            )

        else:

            print(
                f"{key:28s}: "
                f"{value:.2%}"
            )


    # =====================================================
    # 12. DCF SCENARIO VALUATION
    # =====================================================

    section(
        "10. DCF SCENARIO VALUATION"
    )

    terminal_growth = float(
        assumptions[
            "terminal_growth"
        ]
    )

    scenario_values = (
        value_scenarios(
            forecasts=forecasts,
            wacc=wacc,
            terminal_growth=terminal_growth,
            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
        )
    )

    scenario_values[
        "current_price"
    ] = current_price

    scenario_values[
        "upside_downside"
    ] = (
        scenario_values[
            "implied_share_price"
        ]
        / current_price
        - 1
    )

    scenario_values.to_csv(
        processed
        / "scenario_valuation.csv"
    )

    print(
        scenario_values
        .round(4)
        .to_string()
    )


    # =====================================================
    # 13. DCF SENSITIVITY
    # =====================================================

    section(
        "11. BASE DCF SENSITIVITY"
    )

    sensitivity = (
        build_sensitivity_table(
            forecast=forecasts[
                "base"
            ],

            wacc_values=[
                0.065,
                0.070,
                0.075,
                0.080,
                0.085,
                0.090,
            ],

            terminal_growth_values=[
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
            ],

            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
        )
    )

    sensitivity.to_csv(
        processed
        / "dcf_sensitivity_v2.csv"
    )

    print(
        sensitivity
        .round(2)
        .to_string()
    )


    # =====================================================
    # 14. FINAL SUMMARY
    # =====================================================

    section(
        "12. DCF SUMMARY"
    )

    print(
        f"Current Price:       "
        f"${current_price:,.2f}"
    )

    print(
        f"WACC:                "
        f"{wacc:.2%}"
    )

    print(
        f"Terminal Growth:     "
        f"{terminal_growth:.2%}"
    )

    print()

    for scenario in [
        "bear",
        "base",
        "bull",
    ]:

        price = float(
            scenario_values.loc[
                scenario,
                "implied_share_price",
            ]
        )

        upside = float(
            scenario_values.loc[
                scenario,
                "upside_downside",
            ]
        )

        print(
            f"{scenario.upper():5s} DCF: "
            f"${price:,.2f}  "
            f"({upside:+.1%})"
        )

    print(
        "\nPipeline completed successfully."
    )

    print(
        f"Outputs saved to:\n"
        f"{processed}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run()
# =========================================================
# CONSOLIDATED VALUATION
# =========================================================

section(
   "13. CONSOLIDATED VALUATION"
)

valuation_results = (
    build_valuation_summary()
)

consolidated = valuation_results[
    "consolidated"
]

football_field = valuation_results[
    "football_field"
]

central_range = valuation_results[
    "central_range"
]


print(
    "\nCONSOLIDATED VALUATION"
)

print(
    consolidated
    .round(4)
    .to_string(
        index=False
    )
)


print(
    "\nVALUATION FOOTBALL FIELD"
)

print(
    football_field
    .round(2)
    .to_string(
        index=False
    )
)


print(
    "\nCENTRAL VALUATION RANGE"
)

for key, value in (
    central_range.items()
):

    print(
        f"{key:20s}: "
        f"${value:,.2f}"
    )