import numpy as np
import pandas as pd

from wacc_model import calculate_wacc

from market_data import (
    get_market_data,
    calculate_beta,
)

from financials import (
    build_latest_balance_sheet,
)

def run_dcf(
    forecast,
    wacc,
    terminal_growth,
    cash,
    debt,
    shares_outstanding,
):
    """
    Unlevered DCF using FCFF and Gordon Growth terminal value.

    All monetary values should use the same units.
    Our model uses $ millions.
    """

    if wacc <= terminal_growth:
        raise ValueError(
            "WACC must exceed terminal growth."
        )

    df = forecast.copy()

    df["period"] = np.arange(
        1,
        len(df) + 1,
    )

    df["discount_factor"] = (
        1
        / (1 + wacc) ** df["period"]
    )

    df["pv_fcff"] = (
        df["fcff"]
        * df["discount_factor"]
    )

    terminal_fcff = (
        df["fcff"].iloc[-1]
        * (1 + terminal_growth)
    )

    terminal_value = (
        terminal_fcff
        / (wacc - terminal_growth)
    )

    terminal_discount_factor = (
        df["discount_factor"].iloc[-1]
    )

    pv_terminal_value = (
        terminal_value
        * terminal_discount_factor
    )

    pv_forecast_fcff = (
        df["pv_fcff"].sum()
    )

    enterprise_value = (
        pv_forecast_fcff
        + pv_terminal_value
    )

    net_debt = debt - cash

    equity_value = (
        enterprise_value
        - net_debt
    )

    implied_share_price = (
        equity_value
        / shares_outstanding
    )
    terminal_value_pct_ev = (
    pv_terminal_value
    / enterprise_value
)

    explicit_forecast_pct_ev = (
    pv_forecast_fcff
    / enterprise_value
)
    

    return {
        "forecast": df,
        "pv_forecast_fcff": pv_forecast_fcff,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "cash": cash,
        "debt": debt,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "shares_outstanding": shares_outstanding,
        "implied_share_price": implied_share_price,
        "terminal_value_pct_ev": terminal_value_pct_ev,
        "explicit_forecast_pct_ev": explicit_forecast_pct_ev,
    }

def build_sensitivity_table(
    forecast,
    wacc_values,
    terminal_growth_values,
    cash,
    debt,
    shares_outstanding,
):
    """
    Implied share-price sensitivity:
    rows = terminal growth
    columns = WACC
    """

    table = pd.DataFrame(
        index=terminal_growth_values,
        columns=wacc_values,
        dtype=float,
    )

    for growth in terminal_growth_values:

        for wacc in wacc_values:

            if wacc <= growth:
                table.loc[growth, wacc] = np.nan
                continue

            result = run_dcf(
                forecast=forecast,
                wacc=wacc,
                terminal_growth=growth,
                cash=cash,
                debt=debt,
                shares_outstanding=shares_outstanding,
            )

            table.loc[
                growth,
                wacc,
            ] = result[
                "implied_share_price"
            ]

    table.index.name = "Terminal Growth"
    table.columns.name = "WACC"

    return table

def value_scenarios(
    forecasts,
    wacc,
    terminal_growth,
    cash,
    debt,
    shares_outstanding,
):
    rows = []

    for scenario, forecast in forecasts.items():

        result = run_dcf(
            forecast=forecast,
            wacc=wacc,
            terminal_growth=terminal_growth,
            cash=cash,
            debt=debt,
            shares_outstanding=shares_outstanding,
        )

        rows.append(
            {
                "scenario": scenario,

                "enterprise_value": result[
                    "enterprise_value"
                ],

                "equity_value": result[
                    "equity_value"
                ],

                "implied_share_price": result[
                    "implied_share_price"
                ],

                "explicit_forecast_pct_ev": result[
                    "explicit_forecast_pct_ev"
                ],

                "terminal_value_pct_ev": result[
                    "terminal_value_pct_ev"
                ],
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("scenario")
    )



if __name__ == "__main__":

    from pathlib import Path
    import yaml

    from forecast_model import (
        build_all_scenarios,
    )

    ROOT = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    historical = pd.read_csv(
        ROOT
        / "data"
        / "processed"
        / "historical_model.csv",
        index_col=0,
    )

    quarterly = pd.read_csv(
    ROOT
    / "data"
    / "processed"
    / "financials_quarterly.csv"
)

    with open(
        ROOT
        / "config"
        / "company.yaml",
        "r",
    ) as f:

        config = yaml.safe_load(f)

    assumptions = config[
        "assumptions"
    ]

    forecasts = build_all_scenarios(
        historical,
        assumptions,
    )

    market = get_market_data(
    config["ticker"]
)

    market_cap = (
        market["market_cap"]
        / 1_000_000
    )

    shares_outstanding = (
        market["shares_outstanding"]
        / 1_000_000
    )

    current_price = market[
        "price"
    ]

    beta_result = calculate_beta(
    ticker=config["ticker"],
    benchmark="SPY",
    period="5y",
)

    empirical_beta = beta_result[
    "beta"
]

    latest_balance = (
    build_latest_balance_sheet(
        quarterly
    )
)

    balance_map = (
        latest_balance
        .set_index("metric")["value"]
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

    if market_cap <= 0:
        raise RuntimeError(
        "Invalid market cap."
    )

    if shares_outstanding <= 0:
        raise RuntimeError(
            "Invalid shares outstanding."
        )

    if debt < 0:
        raise RuntimeError(
            "Debt cannot be negative."
        )

    if cash < 0:
        raise RuntimeError(
            "Cash cannot be negative."
        )
    # --------------------------------------------------
    # WACC
    # --------------------------------------------------

    wacc_assumptions = assumptions[
        "wacc"
    ]

    base_tax_rate = assumptions[
        "scenarios"
    ]["base"]["tax_rate"]

    wacc_result = calculate_wacc(
        risk_free_rate=wacc_assumptions[
            "risk_free_rate"
        ],
        equity_risk_premium=wacc_assumptions[
            "equity_risk_premium"
        ],
        beta= empirical_beta ,

        pre_tax_cost_of_debt=wacc_assumptions[
            "pre_tax_cost_of_debt"
        ],
        tax_rate=base_tax_rate,
        market_cap=market_cap,
        debt=debt,
    )

    wacc = wacc_result["wacc"]

    terminal_growth = assumptions[
        "terminal_growth"
    ]

    # --------------------------------------------------
    # Scenario valuation
    # --------------------------------------------------

    scenario_values = value_scenarios(
        forecasts=forecasts,
        wacc=wacc,
        terminal_growth=terminal_growth,
        cash=cash,
        debt=debt,
        shares_outstanding=shares_outstanding,
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

    # --------------------------------------------------
    # Base-case sensitivity
    # --------------------------------------------------

    sensitivity = build_sensitivity_table(
        forecast=forecasts["base"],
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
        shares_outstanding=shares_outstanding,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    scenario_values.to_csv(
        ROOT
        / "data"
        / "processed"
        / "scenario_valuation.csv"
    )

    sensitivity.to_csv(
        ROOT
        / "data"
        / "processed"
        / "dcf_sensitivity_v2.csv"
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print(
    "\n=============================="
)
    print("BETA ESTIMATION")
    print("==============================")

    print(
        f"Ticker:              "
        f"{beta_result['ticker']}"
    )

    print(
        f"Benchmark:           "
        f"{beta_result['benchmark']}"
    )

    print(
        f"Observations:        "
        f"{beta_result['observations']}"
    )

    print(
        f"Beta:                "
        f"{beta_result['beta']:.3f}"
    )

    print(
        f"Correlation:         "
        f"{beta_result['correlation']:.3f}"
    )

    print(
        f"Visa Volatility:     "
        f"{beta_result['stock_volatility']:.2%}"
    )

    print(
        f"Market Volatility:   "
        f"{beta_result['market_volatility']:.2%}"
    )
        
    print(
        "\n=============================="
    )
    print("WACC")
    print("==============================")

    for key, value in wacc_result.items():

        if "weight" in key or "cost" in key or key == "wacc":

            print(
                f"{key:25s}: "
                f"{value:.2%}"
            )

    print(
        "\n=============================="
    )
    print("SCENARIO VALUATION")
    print("==============================")

    print(
        scenario_values
        .round(2)
        .to_string()
    )

    print(
        "\n=============================="
    )
    print("BASE DCF SENSITIVITY")
    print("==============================")

    print(
        sensitivity
        .round(2)
        .to_string()
    )

    print(
    "\n=============================="
)
    print("LATEST BALANCE SHEET INPUTS")
    print("==============================")

    print(
        f"Cash:                "
        f"${cash:,.1f}M"
    )

    print(
        f"Short-Term Debt:     "
        f"${short_term_debt:,.1f}M"
    )

    print(
        f"Long-Term Debt:      "
        f"${long_term_debt:,.1f}M"
    )

    print(
        f"Total Debt:          "
        f"${debt:,.1f}M"
    )

    print(
        f"Net Debt:            "
        f"${debt - cash:,.1f}M"
    )