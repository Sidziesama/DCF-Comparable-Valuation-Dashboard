from pathlib import Path

import pandas as pd
import yaml

from market_data import (
    get_market_data,
    calculate_beta,
)

from financials import (
    build_latest_balance_sheet,
)

from forecast_model import (
    build_all_scenarios,
)

from wacc_model import (
    calculate_wacc,
)

from valuation import (
    value_scenarios,
)

from comparables import (
    build_comparable_table,
    build_mastercard_implied_valuation,
)

from terminal_value import (
    value_exit_multiple_scenarios,
    build_exit_multiple_sensitivity,
)

from football_field import (
    build_football_field,
    calculate_central_range,
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
        ROOT
        / "config"
        / "company.yaml",
        "r",
    ) as f:

        return yaml.safe_load(f)


def get_peer_list(
    config,
):

    peer_config = (
        config["peers"]
    )

    return (
        peer_config.get(
            "core_network",
            [],
        )
        +
        peer_config.get(
            "payments_tech",
            [],
        )
        +
        peer_config.get(
            "financial",
            [],
        )
    )


# =========================================================
# BUILD VALUATION SUMMARY
# =========================================================

def build_valuation_summary():

    config = load_config()

    assumptions = (
        config["assumptions"]
    )

    ticker = (
        config["ticker"]
    )

    processed = (
        ROOT
        / "data"
        / "processed"
    )


    # =====================================================
    # HISTORICAL MODEL
    # =====================================================

    historical = pd.read_csv(
        processed
        / "historical_model.csv",
        index_col=0,
    )

    if "LTM" not in historical.index:

        raise RuntimeError(
            "Historical model missing LTM."
        )

    ltm = historical.loc[
        "LTM"
    ]


    # =====================================================
    # BALANCE SHEET
    # =====================================================

    quarterly = pd.read_csv(
        processed
        / "financials_quarterly.csv"
    )

    balance = (
        build_latest_balance_sheet(
            quarterly
        )
    )

    balance_map = (
        balance
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

    debt = (
        float(
            balance_map.get(
                "short_term_debt",
                0.0,
            )
        )
        +
        float(
            balance_map.get(
                "long_term_debt",
                0.0,
            )
        )
    )


    # =====================================================
    # MARKET DATA
    # =====================================================

    market = get_market_data(
        ticker
    )

    current_price = float(
        market["price"]
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


    # =====================================================
    # BETA
    # =====================================================

    beta_result = calculate_beta(
        ticker=ticker,
        benchmark="SPY",
        period="5y",
    )

    empirical_beta = float(
        beta_result["beta"]
    )


    # =====================================================
    # WACC
    # =====================================================

    wacc_config = (
        assumptions["wacc"]
    )

    base_tax_rate = (
        assumptions[
            "scenarios"
        ]["base"]["tax_rate"]
    )

    wacc_result = (
        calculate_wacc(
            market_cap=market_cap,
            debt=debt,

            risk_free_rate=(
                wacc_config[
                    "risk_free_rate"
                ]
            ),

            beta=empirical_beta,

            equity_risk_premium=(
                wacc_config[
                    "equity_risk_premium"
                ]
            ),

            pre_tax_cost_of_debt=(
                wacc_config[
                    "pre_tax_cost_of_debt"
                ]
            ),

            tax_rate=(
                base_tax_rate
            ),
        )
    )

    wacc = float(
        wacc_result["wacc"]
    )


    # =====================================================
    # FORECASTS
    # =====================================================

    forecasts = (
        build_all_scenarios(
            historical,
            assumptions,
        )
    )


    # =====================================================
    # GORDON GROWTH DCF
    # =====================================================

    terminal_growth = float(
        assumptions[
            "terminal_growth"
        ]
    )

    gordon_dcf = (
        value_scenarios(
            forecasts=forecasts,
            wacc=wacc,
            terminal_growth=(
                terminal_growth
            ),
            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
        )
    )

    gordon_dcf[
        "current_price"
    ] = current_price

    gordon_dcf[
        "upside_downside"
    ] = (
        gordon_dcf[
            "implied_share_price"
        ]
        / current_price
        - 1
    )


    # =====================================================
    # EXIT MULTIPLE DCF
    # =====================================================

    exit_multiple_assumptions = (
        assumptions[
            "terminal_exit_multiple"
        ]
    )

    exit_dcf = (
        value_exit_multiple_scenarios(
            forecasts=forecasts,
            wacc=wacc,
            exit_multiples=(
                exit_multiple_assumptions
            ),
            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
        )
    )

    exit_dcf[
        "current_price"
    ] = current_price

    exit_dcf[
        "upside_downside"
    ] = (
        exit_dcf[
            "implied_share_price"
        ]
        / current_price
        - 1
    )


    # =====================================================
    # TRADING COMPARABLES
    # =====================================================

    peers = get_peer_list(
        config
    )

    comps = (
        build_comparable_table(
            peers
        )
    )

    visa_metrics = {

        "revenue":
            float(
                ltm["revenue"]
            ),

        "ebitda":
            float(
                ltm["ebitda"]
            ),

        "ebit":
            float(
                ltm[
                    "operating_income"
                ]
            ),

        "net_income":
            float(
                ltm[
                    "net_income"
                ]
            ),
    }

    mastercard = (
        build_mastercard_implied_valuation(
            comps=comps,
            visa_metrics=(
                visa_metrics
            ),
            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
        )
    )

    mastercard[
        "current_price"
    ] = current_price

    mastercard[
        "upside_downside"
    ] = (
        mastercard[
            "implied_share_price"
        ]
        / current_price
        - 1
    )


    # =====================================================
    # EXIT SENSITIVITY
    # =====================================================

    exit_sensitivity = (
        build_exit_multiple_sensitivity(
            forecast=(
                forecasts[
                    "base"
                ]
            ),

            wacc_values=[
                0.065,
                0.070,
                0.075,
                0.080,
                0.085,
                0.090,
            ],

            exit_multiples=[
                12.0,
                14.0,
                16.0,
                18.0,
                20.0,
            ],

            cash=cash,
            debt=debt,

            shares_outstanding=(
                shares_outstanding
            ),
        )
    )


    # =====================================================
    # FOOTBALL FIELD
    # =====================================================

    football_field = (
        build_football_field(
            gordon_dcf=(
                gordon_dcf
            ),
            exit_dcf=(
                exit_dcf
            ),
            mastercard_comps=(
                mastercard
            ),
            current_price=(
                current_price
            ),
        )
    )

    central_range = (
        calculate_central_range(
            football_field
        )
    )


    # =====================================================
    # CONSOLIDATED METHOD TABLE
    # =====================================================

    rows = []


    for scenario in [
        "bear",
        "base",
        "bull",
    ]:

        price = float(
            gordon_dcf.loc[
                scenario,
                "implied_share_price",
            ]
        )

        rows.append(
            {
                "method":
                    "DCF - Gordon Growth",

                "case":
                    scenario.title(),

                "implied_share_price":
                    price,

                "current_price":
                    current_price,

                "upside_downside":
                    (
                        price
                        / current_price
                        - 1
                    ),
            }
        )


    for scenario in [
        "bear",
        "base",
        "bull",
    ]:

        price = float(
            exit_dcf.loc[
                scenario,
                "implied_share_price",
            ]
        )

        multiple = float(
            exit_dcf.loc[
                scenario,
                "exit_multiple",
            ]
        )

        rows.append(
            {
                "method":
                    "DCF - Exit Multiple",

                "case":
                    (
                        f"{scenario.title()} "
                        f"({multiple:.1f}x)"
                    ),

                "implied_share_price":
                    price,

                "current_price":
                    current_price,

                "upside_downside":
                    (
                        price
                        / current_price
                        - 1
                    ),
            }
        )


    for _, row in (
        mastercard.iterrows()
    ):

        price = row[
            "implied_share_price"
        ]

        if pd.isna(price):
            continue

        rows.append(
            {
                "method":
                    "Mastercard Trading Comp",

                "case":
                    row[
                        "multiple_type"
                    ],

                "implied_share_price":
                    float(price),

                "current_price":
                    current_price,

                "upside_downside":
                    (
                        float(price)
                        / current_price
                        - 1
                    ),
            }
        )


    consolidated = pd.DataFrame(
        rows
    )


    # =====================================================
    # SAVE
    # =====================================================

    gordon_dcf.to_csv(
        processed
        / "valuation_gordon_dcf.csv"
    )

    exit_dcf.to_csv(
        processed
        / "valuation_exit_dcf.csv"
    )

    mastercard.to_csv(
        processed
        / "valuation_mastercard_comps.csv",
        index=False,
    )

    exit_sensitivity.to_csv(
        processed
        / "exit_multiple_sensitivity.csv"
    )

    football_field.to_csv(
        processed
        / "football_field.csv",
        index=False,
    )

    consolidated.to_csv(
        processed
        / "valuation_summary.csv",
        index=False,
    )

    pd.DataFrame(
        [
            central_range
        ]
    ).to_csv(
        processed
        / "central_valuation_range.csv",
        index=False,
    )


    return {

        "gordon_dcf":
            gordon_dcf,

        "exit_dcf":
            exit_dcf,

        "mastercard":
            mastercard,

        "football_field":
            football_field,

        "central_range":
            central_range,

        "consolidated":
            consolidated,

        "exit_sensitivity":
            exit_sensitivity,

        "wacc":
            wacc_result,

        "beta":
            beta_result,

        "market":
            market,
    }


# =========================================================
# RUNNER
# =========================================================

if __name__ == "__main__":

    results = (
        build_valuation_summary()
    )


    print(
        "\n=============================="
    )

    print(
        "CONSOLIDATED VALUATION"
    )

    print(
        "=============================="
    )

    print(
        results[
            "consolidated"
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


    print(
        "\n=============================="
    )

    print(
        "VALUATION FOOTBALL FIELD"
    )

    print(
        "=============================="
    )

    print(
        results[
            "football_field"
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


    print(
        "\n=============================="
    )

    print(
        "CENTRAL VALUATION RANGE"
    )

    print(
        "=============================="
    )

    central = results[
        "central_range"
    ]

    for key, value in (
        central.items()
    ):

        print(
            f"{key:20s}: "
            f"${value:,.2f}"
        )


    print(
        "\n=============================="
    )

    print(
        "EXIT MULTIPLE SENSITIVITY"
    )

    print(
        "=============================="
    )

    print(
        results[
            "exit_sensitivity"
        ]
        .round(2)
        .to_string()
    )