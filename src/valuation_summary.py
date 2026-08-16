from pathlib import Path
import copy

import pandas as pd
from company_config import DEFAULT_CONFIG_PATH, CompanyWorkspace, load_company_config

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
from adapters import get_adapter

from wacc_model import (
    calculate_wacc,
)

from valuation import (
    value_scenarios,
)

from comparables import (
    apply_multiple_eligibility,
    build_comparable_table,
    build_direct_peer_implied_valuation,
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

def load_config(path=DEFAULT_CONFIG_PATH):
    return load_company_config(path)


def get_peer_list(
    config,
):

    peer_config = config.get("peers", {})
    # Peer groups are company-defined; the core engine does not prescribe an
    # industry taxonomy (software, payments, financials, etc.).
    return [ticker for group in peer_config.values() for ticker in group]


# =========================================================
# BUILD VALUATION SUMMARY
# =========================================================

def build_valuation_summary(config_path=DEFAULT_CONFIG_PATH):

    config = load_config(config_path)

    assumptions = (
        config["assumptions"]
    )

    ticker = (
        config["ticker"]
    )

    processed = CompanyWorkspace.from_config(config, ROOT).ensure()


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
        benchmark=(config.get("market", {}) or {}).get("benchmark", "SPY"),
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

    summary_assumptions = copy.deepcopy(assumptions)
    adapter = get_adapter(config.get("adapter", "generic"))
    years = int(assumptions.get("forecast_years", config.get("forecast_years", 5)))
    for scenario, scenario_cfg in summary_assumptions["scenarios"].items():
        growth, _ = adapter.forecast_growth(scenario, scenario_cfg, years)
        scenario_cfg["revenue_growth"] = growth
    forecasts = (
        build_all_scenarios(
            historical,
            summary_assumptions,
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

    # The linked three-statement pipeline owns the canonical Gordon-growth DCF.
    # This consolidation layer consumes that output instead of recalculating a
    # second value from the standalone operating forecast.
    canonical_path = processed / "scenario_valuation.csv"
    if not canonical_path.exists():
        raise RuntimeError(
            "Canonical scenario_valuation.csv is missing. Run src/pipeline.py "
            "for this company before building the consolidated valuation."
        )
    gordon_dcf = pd.read_csv(canonical_path, index_col=0)
    if not {"bear", "base", "bull"}.issubset(set(gordon_dcf.index)):
        raise RuntimeError("Canonical scenario valuation is missing bear/base/bull cases.")

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
    comps = apply_multiple_eligibility(
        comps,
        (config.get("peer_methodology", {}) or {}).get("multiple_eligibility", {}),
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

    direct_peer = config.get("valuation", {}).get("direct_peer", "MA")
    mastercard = (
        build_direct_peer_implied_valuation(
            comps=comps,
            visa_metrics=(
                visa_metrics
            ),
            cash=cash,
            debt=debt,
            shares_outstanding=(
                shares_outstanding
            ),
            peer_ticker=direct_peer,
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
            direct_peer_comps=(
                mastercard
            ),
            current_price=(
                current_price
            ),
            direct_peer_label=direct_peer,
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
                    f"{direct_peer} Trading Comp",

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

    comps.to_csv(processed / "trading_comparables.csv")

    mastercard.to_csv(
        processed
        / "valuation_direct_peer_comps.csv",
        index=False,
    )
    if direct_peer == "MA":
        mastercard.to_csv(processed / "valuation_mastercard_comps.csv", index=False)

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
