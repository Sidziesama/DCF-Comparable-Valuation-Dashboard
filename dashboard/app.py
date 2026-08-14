from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from valuation import run_dcf, run_exit_multiple_dcf


# =========================================================
# PATHS
# =========================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA = (
    ROOT
    / "data"
    / "processed"
)

CONFIG_PATH = (
    ROOT
    / "config"
    / "company.yaml"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Visa Valuation Dashboard",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# LOAD CONFIG
# =========================================================

@st.cache_data
def load_config():

    with open(
        CONFIG_PATH,
        "r",
    ) as f:

        return yaml.safe_load(f)


config = load_config()


# =========================================================
# DATA HELPERS
# =========================================================

def load_csv(
    filename,
    index_col=None,
):

    path = (
        DATA
        / filename
    )

    if not path.exists():

        st.error(
            f"Missing file: {filename}. "
            "Run `python src/pipeline.py` first."
        )

        st.stop()

    return pd.read_csv(
        path,
        index_col=index_col,
    )


# =========================================================
# LOAD PIPELINE OUTPUTS
# =========================================================

@st.cache_data
def load_data():

    historical = load_csv(
        "historical_model.csv",
        index_col=0,
    )

    forecast_bear = load_csv(
        "forecast_bear.csv",
        index_col=0,
    )

    forecast_base = load_csv(
        "forecast_base.csv",
        index_col=0,
    )

    forecast_bull = load_csv(
        "forecast_bull.csv",
        index_col=0,
    )

    gordon_dcf = load_csv(
        "valuation_gordon_dcf.csv",
        index_col=0,
    )

    exit_dcf = load_csv(
        "valuation_exit_dcf.csv",
        index_col=0,
    )

    mastercard = load_csv(
        "valuation_mastercard_comps.csv",
    )

    football_field = load_csv(
        "football_field.csv",
    )

    valuation_summary = load_csv(
        "valuation_summary.csv",
    )

    central_range = load_csv(
        "central_valuation_range.csv",
    )

    dcf_sensitivity = load_csv(
        "dcf_sensitivity_v2.csv",
        index_col=0,
    )

    exit_sensitivity = load_csv(
        "exit_multiple_sensitivity.csv",
        index_col=0,
    )

    wacc_report = load_csv(
        "wacc_report.csv",
    )

    latest_balance = load_csv(
        "latest_balance_sheet.csv",
    )

    trading_comps = load_csv(
        "trading_comparables.csv",
        index_col=0,
    )

    return {
        "historical": historical,
        "forecast_bear": forecast_bear,
        "forecast_base": forecast_base,
        "forecast_bull": forecast_bull,
        "gordon_dcf": gordon_dcf,
        "exit_dcf": exit_dcf,
        "mastercard": mastercard,
        "football_field": football_field,
        "valuation_summary": valuation_summary,
        "central_range": central_range,
        "dcf_sensitivity": dcf_sensitivity,
        "exit_sensitivity": exit_sensitivity,
        "wacc_report": wacc_report,
        "latest_balance": latest_balance,
        "trading_comps": trading_comps,
    }


data = load_data()


# =========================================================
# FORMATTERS
# =========================================================

def money(
    value,
    decimals=1,
):

    if pd.isna(value):
        return "—"

    return (
        f"${value:,.{decimals}f}"
    )


def billions(
    value,
):

    if pd.isna(value):
        return "—"

    return (
        f"${value / 1000:,.1f}B"
    )


def percentage(
    value,
):

    if pd.isna(value):
        return "—"

    return (
        f"{value:.1%}"
    )


# =========================================================
# CORE VALUES
# =========================================================

historical = data[
    "historical"
]

ltm = historical.loc[
    "LTM"
]

gordon_dcf = data[
    "gordon_dcf"
]

exit_dcf = data[
    "exit_dcf"
]

football_field = data[
    "football_field"
]

central_range = data[
    "central_range"
].iloc[0]

valuation_summary = data[
    "valuation_summary"
]

current_price = float(
    valuation_summary[
        "current_price"
    ].iloc[0]
)

base_dcf = float(
    gordon_dcf.loc[
        "base",
        "implied_share_price",
    ]
)

central_value = float(
    central_range[
        "mean_base"
    ]
)

central_upside = (
    central_value
    / current_price
    - 1
)


# =========================================================
# HEADER
# =========================================================

st.title(
    f"{config['company_name']} "
    f"({config['ticker']})"
)

st.caption(
    "DCF + Trading Comparables Valuation Dashboard"
)


# =========================================================
# TOP KPIs
# =========================================================

c1, c2, c3, c4, c5 = (
    st.columns(5)
)

c1.metric(
    "Current Price",
    money(
        current_price,
        2,
    ),
)

c2.metric(
    "Central Valuation",
    money(
        central_value,
        2,
    ),
    percentage(
        central_upside
    ),
)

c3.metric(
    "Base DCF",
    money(
        base_dcf,
        2,
    ),
    percentage(
        base_dcf
        / current_price
        - 1
    ),
)

c4.metric(
    "LTM Revenue",
    billions(
        float(
            ltm["revenue"]
        )
    ),
)

c5.metric(
    "LTM EBITDA",
    billions(
        float(
            ltm["ebitda"]
        )
    ),
)


# =========================================================
# TABS
# =========================================================

tabs = st.tabs(
    [
        "Overview",
        "Financials",
        "DCF",
        "Trading Comps",
        "Valuation",
        "Interactive DCF",
        "Three Statements",
    ]
)

overview_tab = tabs[0]
financials_tab = tabs[1]
dcf_tab = tabs[2]
comps_tab = tabs[3]
valuation_tab = tabs[4]
interactive_tab = tabs[5]
statements_tab = tabs[6]


# =========================================================
# INTERACTIVE DCF TAB
# =========================================================

with interactive_tab:
    st.subheader("Interactive Scenario Engine")
    st.caption("Recalculates from cached pipeline outputs; no SEC or market-data requests.")

    control_1, control_2, control_3, control_4 = st.columns(4)
    with control_1:
        selected_scenario = st.radio(
            "Operating scenario", ["Bear", "Base", "Bull"], index=1, horizontal=True
        ).lower()
    wacc_rows = data["wacc_report"].set_index(data["wacc_report"].columns[0])
    base_wacc = float(wacc_rows.loc["WACC"].iloc[0])
    with control_2:
        selected_wacc_pct = st.slider("WACC", 5.0, 12.0, base_wacc * 100, 0.05, format="%.2f%%")
    with control_3:
        selected_growth_pct = st.slider(
            "Terminal growth", 0.0, min(6.0, selected_wacc_pct - 0.25),
            float(config["assumptions"]["terminal_growth"]) * 100, 0.05, format="%.2f%%"
        )
    with control_4:
        selected_multiple = st.slider("Terminal EV / EBITDA", 8.0, 30.0, 16.0, 0.5, format="%.1fx")

    selected_forecast = data[f"forecast_{selected_scenario}"].copy()
    if "ebitda" not in selected_forecast and {"ebit", "da"}.issubset(selected_forecast.columns):
        selected_forecast["ebitda"] = selected_forecast["ebit"] + selected_forecast["da"]
    balance_map = data["latest_balance"].set_index("metric")["value"].to_dict()
    selected_cash = float(balance_map.get("cash", 0.0))
    selected_debt = float(balance_map.get("short_term_debt", 0.0)) + float(balance_map.get("long_term_debt", 0.0))
    selected_shares = float((gordon_dcf["equity_value"] / gordon_dcf["implied_share_price"]).median())
    live_gordon = run_dcf(selected_forecast, selected_wacc_pct / 100, selected_growth_pct / 100, selected_cash, selected_debt, selected_shares)
    live_exit = run_exit_multiple_dcf(selected_forecast, selected_wacc_pct / 100, selected_multiple, selected_cash, selected_debt, selected_shares)

    result_columns = st.columns(4)
    result_columns[0].metric("Market price", money(current_price, 2))
    result_columns[1].metric("Gordon DCF", money(live_gordon["implied_share_price"], 2), percentage(live_gordon["implied_share_price"] / current_price - 1))
    result_columns[2].metric("Exit-multiple DCF", money(live_exit["implied_share_price"], 2), percentage(live_exit["implied_share_price"] / current_price - 1))
    result_columns[3].metric("Terminal value / EV", percentage(live_gordon["terminal_value_pct_ev"]))

    bridge = pd.DataFrame(
        {
            "Gordon growth": [live_gordon[key] for key in ("enterprise_value", "cash", "debt", "equity_value")],
            "Exit multiple": [live_exit[key] for key in ("enterprise_value", "cash", "debt", "equity_value")],
        },
        index=["Enterprise value", "+ Cash", "− Debt", "Equity value"],
    )
    st.dataframe(bridge.style.format("${:,.0f}M"), use_container_width=True)


# =========================================================
# THREE-STATEMENT TAB
# =========================================================

with statements_tab:
    st.subheader("Linked Three-Statement Forecast")
    statement_scenario = st.radio("Statement scenario", ["Bear", "Base", "Bull"], index=1, horizontal=True).lower()
    statement_files = {
        "Income statement": DATA / f"income_statement_{statement_scenario}.csv",
        "Balance sheet": DATA / f"balance_sheet_{statement_scenario}.csv",
        "Cash flow": DATA / f"cash_flow_statement_{statement_scenario}.csv",
        "Checks": DATA / f"checks_{statement_scenario}.csv",
    }
    if not all(path.exists() for path in statement_files.values()):
        st.info("Run `python src/three_statement_model.py` to generate linked schedules.")
    else:
        selected_statement = st.selectbox("Statement", list(statement_files))
        statement = pd.read_csv(statement_files[selected_statement], index_col=0)
        if selected_statement == "Checks":
            st.dataframe(statement.style.format("{:,.8f}"), use_container_width=True)
            st.success(f"Statements reconcile; maximum error is {statement.abs().to_numpy().max():,.8f}M.")
        else:
            st.dataframe(statement.style.format("${:,.1f}M"), use_container_width=True)


# =========================================================
# OVERVIEW TAB
# =========================================================

with overview_tab:

    st.subheader(
        "Valuation Snapshot"
    )

    col1, col2 = (
        st.columns(
            [1, 1]
        )
    )


    # -----------------------------------------------------
    # VALUATION METHODS
    # -----------------------------------------------------

    with col1:

        summary_chart = (
            valuation_summary.copy()
        )

        fig = px.bar(
            summary_chart,
            x="implied_share_price",
            y="case",
            color="method",
            orientation="h",
            title=(
                "Implied Share Price "
                "by Valuation Method"
            ),
        )

        fig.add_vline(
            x=current_price,
            line_dash="dash",
            annotation_text=(
                "Current Price"
            ),
        )

        fig.update_layout(
            xaxis_title=(
                "Implied Share Price ($)"
            ),
            yaxis_title="",
            legend_title="Method",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # -----------------------------------------------------
    # HISTORICAL REVENUE
    # -----------------------------------------------------

    with col2:

        hist = (
            historical
            .reset_index()
            .rename(
                columns={
                    historical.index.name
                    or "index":
                    "period"
                }
            )
        )

        hist[
            "period"
        ] = (
            hist[
                "period"
            ]
            .astype(str)
        )

        fig = px.line(
            hist,
            x="period",
            y="revenue",
            markers=True,
            title=(
                "Historical Revenue ($M)"
            ),
        )

        fig.update_layout(
            xaxis_title="Period",
            yaxis_title="Revenue ($M)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # -----------------------------------------------------
    # LTM METRICS
    # -----------------------------------------------------

    st.subheader(
        "LTM Financial Profile"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Operating Margin",
        percentage(
            float(
                ltm[
                    "operating_margin"
                ]
            )
        ),
    )

    c2.metric(
        "Net Margin",
        percentage(
            float(
                ltm[
                    "net_margin"
                ]
            )
        ),
    )

    c3.metric(
        "FCF Margin",
        percentage(
            float(
                ltm[
                    "fcf_margin"
                ]
            )
        ),
    )

    c4.metric(
        "D&A / Revenue",
        percentage(
            float(
                ltm[
                    "da_pct_revenue"
                ]
            )
        ),
    )


# =========================================================
# FINANCIALS TAB
# =========================================================

with financials_tab:

    st.subheader(
        "Historical Financial Performance"
    )


    # -----------------------------------------------------
    # Historical chart
    # -----------------------------------------------------

    historical_chart = (
        historical[
            [
                "revenue",
                "operating_income",
                "net_income",
                "ebitda",
                "fcf",
            ]
        ]
        .copy()
        .reset_index()
    )

    historical_chart.columns.values[
        0
    ] = "period"

    historical_long = (
        historical_chart
        .melt(
            id_vars="period",
            var_name="metric",
            value_name="value",
        )
    )

    fig = px.line(
        historical_long,
        x="period",
        y="value",
        color="metric",
        markers=True,
        title=(
            "Historical Financials ($M)"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Forecast scenarios
    # -----------------------------------------------------

    st.subheader(
        "Revenue Forecast Scenarios"
    )

    scenario_frames = []

    for scenario in [
        "bear",
        "base",
        "bull",
    ]:

        df = (
            data[
                f"forecast_{scenario}"
            ]
            .copy()
            .reset_index()
        )

        df[
            "scenario"
        ] = scenario.title()

        scenario_frames.append(
            df
        )

    scenarios = pd.concat(
        scenario_frames,
        ignore_index=True,
    )

    fig = px.line(
        scenarios,
        x="year",
        y="revenue",
        color="scenario",
        markers=True,
        title=(
            "Bear / Base / Bull Revenue"
        ),
    )

    fig.update_layout(
        yaxis_title="Revenue ($M)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Base-case FCFF
    # -----------------------------------------------------

    st.subheader(
        "Base-Case Free Cash Flow"
    )

    base_forecast = (
        data[
            "forecast_base"
        ]
        .copy()
        .reset_index()
    )

    fig = px.bar(
        base_forecast,
        x="year",
        y="fcff",
        title=(
            "Forecast FCFF ($M)"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Table
    # -----------------------------------------------------

    st.subheader(
        "Historical Model"
    )

    st.dataframe(
        historical.round(4),
        use_container_width=True,
    )


# =========================================================
# DCF TAB
# =========================================================

with dcf_tab:

    st.subheader(
        "Discounted Cash Flow"
    )


    # -----------------------------------------------------
    # Gordon scenarios
    # -----------------------------------------------------

    c1, c2, c3 = (
        st.columns(3)
    )

    for column, scenario in zip(
        [
            c1,
            c2,
            c3,
        ],
        [
            "bear",
            "base",
            "bull",
        ],
    ):

        value = float(
            gordon_dcf.loc[
                scenario,
                "implied_share_price",
            ]
        )

        column.metric(
            (
                f"{scenario.title()} "
                "DCF"
            ),
            money(
                value,
                2,
            ),
            percentage(
                value
                / current_price
                - 1
            ),
        )


    # -----------------------------------------------------
    # WACC
    # -----------------------------------------------------

    st.subheader(
        "WACC Build"
    )

    wacc_report = (
        data[
            "wacc_report"
        ]
        .copy()
    )

    st.dataframe(
        wacc_report,
        use_container_width=True,
        hide_index=True,
    )


    # -----------------------------------------------------
    # Gordon heatmap
    # -----------------------------------------------------

    st.subheader(
        "WACC × Terminal Growth"
    )

    dcf_sensitivity = (
        data[
            "dcf_sensitivity"
        ]
        .copy()
    )

    fig = px.imshow(
        dcf_sensitivity.astype(
            float
        ),
        text_auto=".2f",
        aspect="auto",
        labels={
            "x": "WACC",
            "y": "Terminal Growth",
            "color": "Share Price",
        },
        title=(
            "DCF Sensitivity ($/share)"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Exit sensitivity
    # -----------------------------------------------------

    st.subheader(
        "WACC × Exit EV/EBITDA"
    )

    exit_sensitivity = (
        data[
            "exit_sensitivity"
        ]
        .copy()
    )

    fig = px.imshow(
        exit_sensitivity.astype(
            float
        ),
        text_auto=".2f",
        aspect="auto",
        labels={
            "x": "WACC",
            "y": "Exit EV/EBITDA",
            "color": "Share Price",
        },
        title=(
            "Exit Multiple Sensitivity ($/share)"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# COMPS TAB
# =========================================================

with comps_tab:

    st.subheader(
        "Trading Comparables"
    )

    comps = (
        data[
            "trading_comps"
        ]
        .copy()
    )

    display_columns = [
        "company_name",
        "market_cap",
        "enterprise_value",
        "revenue",
        "ebitda",
        "ebit",
        "net_income",
        "ev_revenue",
        "ev_ebitda",
        "ev_ebit",
        "pe",
    ]

    available = [
        column
        for column in display_columns
        if column in comps.columns
    ]

    st.dataframe(
        comps[
            available
        ].round(2),
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Multiple chart
    # -----------------------------------------------------

    st.subheader(
        "Peer EV / EBITDA"
    )

    ev_ebitda = (
        comps[
            "ev_ebitda"
        ]
        .dropna()
        .sort_values()
        .reset_index()
    )

    fig = px.bar(
        ev_ebitda,
        x="ev_ebitda",
        y="ticker",
        orientation="h",
        title=(
            "Peer EV / EBITDA"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # MA implied valuation
    # -----------------------------------------------------

    st.subheader(
        "Mastercard Direct Comp"
    )

    mastercard = (
        data[
            "mastercard"
        ]
    )

    st.dataframe(
        mastercard.round(4),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# VALUATION TAB
# =========================================================

with valuation_tab:

    st.subheader(
        "Valuation Football Field"
    )

    ff = (
        football_field.copy()
    )


    # -----------------------------------------------------
    # Football field chart
    # -----------------------------------------------------

    fig = go.Figure()

    for _, row in (
        ff.iterrows()
    ):

        fig.add_trace(
            go.Scatter(
                x=[
                    row["low"],
                    row["high"],
                ],
                y=[
                    row["method"],
                    row["method"],
                ],
                mode="lines",
                line=dict(
                    width=12,
                ),
                name=row[
                    "method"
                ],
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    row["base"]
                ],
                y=[
                    row["method"]
                ],
                mode="markers",
                marker=dict(
                    size=14,
                ),
                name=(
                    f"{row['method']} Base"
                ),
                showlegend=False,
            )
        )

    fig.add_vline(
        x=current_price,
        line_dash="dash",
        annotation_text=(
            f"Current Price "
            f"${current_price:.2f}"
        ),
    )

    fig.update_layout(
        title=(
            "Visa Implied Valuation Range"
        ),
        xaxis_title=(
            "Implied Share Price ($)"
        ),
        yaxis_title="",
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Central valuation
    # -----------------------------------------------------

    st.subheader(
        "Central Valuation"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Minimum Base",
        money(
            central_range[
                "minimum_base"
            ],
            2,
        ),
    )

    c2.metric(
        "Median Base",
        money(
            central_range[
                "median_base"
            ],
            2,
        ),
    )

    c3.metric(
        "Maximum Base",
        money(
            central_range[
                "maximum_base"
            ],
            2,
        ),
    )

    c4.metric(
        "Mean Base",
        money(
            central_range[
                "mean_base"
            ],
            2,
        ),
        percentage(
            central_upside
        ),
    )


    # -----------------------------------------------------
    # Complete valuation table
    # -----------------------------------------------------

    st.subheader(
        "All Valuation Methods"
    )

    valuation_display = (
        valuation_summary.copy()
    )

    valuation_display[
        "upside_downside"
    ] = (
        valuation_display[
            "upside_downside"
        ]
        .map(
            lambda x:
            f"{x:.1%}"
        )
    )

    valuation_display[
        "implied_share_price"
    ] = (
        valuation_display[
            "implied_share_price"
        ]
        .map(
            lambda x:
            f"${x:,.2f}"
        )
    )

    valuation_display[
        "current_price"
    ] = (
        valuation_display[
            "current_price"
        ]
        .map(
            lambda x:
            f"${x:,.2f}"
        )
    )

    st.dataframe(
        valuation_display,
        use_container_width=True,
        hide_index=True,
    )
