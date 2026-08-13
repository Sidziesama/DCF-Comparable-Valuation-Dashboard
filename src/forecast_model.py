import pandas as pd
import numpy as np


def get_ltm_base(
    historical_model: pd.DataFrame,
):
    """
    Extract the latest LTM financial base.
    """

    if "LTM" not in historical_model.index:

        raise RuntimeError(
            "Historical model does not contain LTM."
        )

    return historical_model.loc["LTM"]


def build_forecast(
    historical_model,
    assumptions,
    scenario="base",
    start_year=2027,
):
    """
    Build a five-year operating forecast using
    LTM financials as the starting point.

    FCFF:
        EBIT
        - Cash taxes
        + D&A
        - CapEx
        - Change in NWC
    """

    scenario_data = (
        assumptions["scenarios"][
            scenario
        ]
    )

    revenue_growth = (
        scenario_data[
            "revenue_growth"
        ]
    )

    operating_margin = (
        scenario_data[
            "operating_margin"
        ]
    )

    tax_rate = (
        scenario_data[
            "tax_rate"
        ]
    )

    da_pct = (
        scenario_data[
            "da_pct_revenue"
        ]
    )

    capex_pct = (
        scenario_data[
            "capex_pct_revenue"
        ]
    )

    nwc_pct = (
        scenario_data[
            "delta_nwc_pct_incremental_revenue"
        ]
    )

    ltm = get_ltm_base(
        historical_model
    )

    previous_revenue = float(
        ltm["revenue"]
    )

    rows = []

    for i, growth in enumerate(
        revenue_growth
    ):

        year = start_year + i

        revenue = (
            previous_revenue
            * (1 + growth)
        )

        incremental_revenue = (
            revenue
            - previous_revenue
        )

        margin = (
            operating_margin[i]
        )

        ebit = (
            revenue
            * margin
        )

        taxes = (
            ebit
            * tax_rate
        )

        nopat = (
            ebit
            - taxes
        )

        da = (
            revenue
            * da_pct
        )

        capex = (
            revenue
            * capex_pct
        )

        change_nwc = (
            incremental_revenue
            * nwc_pct
        )

        fcff = (
            nopat
            + da
            - capex
            - change_nwc
        )

        rows.append(
            {
                "year": year,

                "scenario": scenario,

                "revenue": revenue,

                "revenue_growth": growth,

                "operating_margin": margin,

                "ebit": ebit,

                "tax_rate": tax_rate,

                "cash_taxes": taxes,

                "nopat": nopat,

                "da": da,

                "capex": capex,

                "change_nwc": change_nwc,

                "fcff": fcff,

                "fcff_margin": (
                    fcff / revenue
                ),
            }
        )

        previous_revenue = revenue

    forecast = pd.DataFrame(
        rows
    )

    return forecast.set_index(
        "year"
    )
def build_all_scenarios(
    historical_model,
    assumptions,
    start_year=2027,
):

    forecasts = {}

    for scenario in [
        "bear",
        "base",
        "bull",
    ]:

        forecasts[scenario] = (
            build_forecast(
                historical_model,
                assumptions,
                scenario=scenario,
                start_year=start_year,
            )
        )

    return forecasts

if __name__ == "__main__":

    from pathlib import Path
    import yaml

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

    with open(
        ROOT
        / "config"
        / "company.yaml",
        "r",
    ) as f:

        config = yaml.safe_load(f)

    forecasts = (
        build_all_scenarios(
            historical,
            config["assumptions"],
        )
    )

    for scenario, forecast in (
        forecasts.items()
    ):

        print(
            "\n=============================="
        )

        print(
            scenario.upper(),
            "FORECAST"
        )

        print(
            "=============================="
        )

        columns = [
            "revenue",
            "revenue_growth",
            "operating_margin",
            "ebit",
            "nopat",
            "da",
            "capex",
            "change_nwc",
            "fcff",
            "fcff_margin",
        ]

        print(
            forecast[
                columns
            ]
            .round(4)
            .to_string()
        )

        output = (
            ROOT
            / "data"
            / "processed"
            / f"forecast_{scenario}.csv"
        )

        forecast.to_csv(
            output
        )