import numpy as np
import pandas as pd


# =========================================================
# TERMINAL EBITDA
# =========================================================

def calculate_terminal_ebitda(
    forecast: pd.DataFrame,
) -> float:
    """
    Terminal EBITDA = Terminal EBIT + Terminal D&A.
    """

    required = {
        "ebit",
        "da",
    }

    missing = (
        required
        - set(forecast.columns)
    )

    if missing:

        raise ValueError(
            f"Forecast missing columns: "
            f"{sorted(missing)}"
        )

    terminal_ebit = float(
        forecast["ebit"].iloc[-1]
    )

    terminal_da = float(
        forecast["da"].iloc[-1]
    )

    return (
        terminal_ebit
        + terminal_da
    )


# =========================================================
# EXIT MULTIPLE DCF
# =========================================================

def run_exit_multiple_dcf(
    forecast,
    wacc,
    exit_multiple,
    cash,
    debt,
    shares_outstanding,
):
    """
    FCFF DCF where terminal value is determined using:

        Terminal EBITDA × Exit EV/EBITDA Multiple
    """

    if wacc <= 0:

        raise ValueError(
            "WACC must be positive."
        )

    if exit_multiple <= 0:

        raise ValueError(
            "Exit multiple must be positive."
        )

    if shares_outstanding <= 0:

        raise ValueError(
            "Shares outstanding must be positive."
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

    pv_forecast_fcff = float(
        df["pv_fcff"].sum()
    )

    terminal_ebitda = (
        calculate_terminal_ebitda(
            df
        )
    )

    terminal_value = (
        terminal_ebitda
        * exit_multiple
    )

    terminal_discount_factor = float(
        df["discount_factor"].iloc[-1]
    )

    pv_terminal_value = (
        terminal_value
        * terminal_discount_factor
    )

    enterprise_value = (
        pv_forecast_fcff
        + pv_terminal_value
    )

    net_debt = (
        debt
        - cash
    )

    equity_value = (
        enterprise_value
        - net_debt
    )

    implied_share_price = (
        equity_value
        / shares_outstanding
    )

    explicit_forecast_pct_ev = (
        pv_forecast_fcff
        / enterprise_value
    )

    terminal_value_pct_ev = (
        pv_terminal_value
        / enterprise_value
    )

    return {

        "wacc":
            wacc,

        "exit_multiple":
            exit_multiple,

        "terminal_ebitda":
            terminal_ebitda,

        "terminal_value":
            terminal_value,

        "pv_terminal_value":
            pv_terminal_value,

        "pv_forecast_fcff":
            pv_forecast_fcff,

        "enterprise_value":
            enterprise_value,

        "cash":
            cash,

        "debt":
            debt,

        "net_debt":
            net_debt,

        "equity_value":
            equity_value,

        "shares_outstanding":
            shares_outstanding,

        "implied_share_price":
            implied_share_price,

        "explicit_forecast_pct_ev":
            explicit_forecast_pct_ev,

        "terminal_value_pct_ev":
            terminal_value_pct_ev,
    }


# =========================================================
# EXIT MULTIPLE SCENARIOS
# =========================================================

def value_exit_multiple_scenarios(
    forecasts,
    wacc,
    exit_multiples,
    cash,
    debt,
    shares_outstanding,
):
    """
    Apply separate terminal multiples to
    Bear / Base / Bull operating scenarios.
    """

    rows = []

    for scenario in [
        "bear",
        "base",
        "bull",
    ]:

        if scenario not in forecasts:

            continue

        if scenario not in exit_multiples:

            raise ValueError(
                f"Missing exit multiple "
                f"for {scenario}."
            )

        result = (
            run_exit_multiple_dcf(
                forecast=(
                    forecasts[
                        scenario
                    ]
                ),
                wacc=wacc,
                exit_multiple=(
                    exit_multiples[
                        scenario
                    ]
                ),
                cash=cash,
                debt=debt,
                shares_outstanding=(
                    shares_outstanding
                ),
            )
        )

        rows.append(
            {
                "scenario":
                    scenario,

                "exit_multiple":
                    result[
                        "exit_multiple"
                    ],

                "terminal_ebitda":
                    result[
                        "terminal_ebitda"
                    ],

                "enterprise_value":
                    result[
                        "enterprise_value"
                    ],

                "equity_value":
                    result[
                        "equity_value"
                    ],

                "implied_share_price":
                    result[
                        "implied_share_price"
                    ],

                "explicit_forecast_pct_ev":
                    result[
                        "explicit_forecast_pct_ev"
                    ],

                "terminal_value_pct_ev":
                    result[
                        "terminal_value_pct_ev"
                    ],
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .set_index(
            "scenario"
        )
    )


# =========================================================
# EXIT MULTIPLE SENSITIVITY
# =========================================================

def build_exit_multiple_sensitivity(
    forecast,
    wacc_values,
    exit_multiples,
    cash,
    debt,
    shares_outstanding,
):
    """
    Rows:
        Exit EV/EBITDA

    Columns:
        WACC

    Values:
        Implied share price.
    """

    table = pd.DataFrame(
        index=exit_multiples,
        columns=wacc_values,
        dtype=float,
    )

    for multiple in exit_multiples:

        for wacc in wacc_values:

            result = (
                run_exit_multiple_dcf(
                    forecast=forecast,
                    wacc=wacc,
                    exit_multiple=multiple,
                    cash=cash,
                    debt=debt,
                    shares_outstanding=(
                        shares_outstanding
                    ),
                )
            )

            table.loc[
                multiple,
                wacc,
            ] = result[
                "implied_share_price"
            ]

    table.index.name = (
        "Exit EV/EBITDA"
    )

    table.columns.name = "WACC"

    return table