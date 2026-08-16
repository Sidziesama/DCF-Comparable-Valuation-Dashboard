from pathlib import Path
import sys
import os
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from valuation import run_dcf, run_exit_multiple_dcf
from company_config import DEFAULT_CONFIG_PATH, CompanyWorkspace, load_company_config


# =========================================================
# PATHS
# =========================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

def _config_argument():
    """Support COMPANY_CONFIG or `streamlit run ... -- --config path`."""
    if os.getenv("COMPANY_CONFIG"):
        return Path(os.environ["COMPANY_CONFIG"])
    if "--config" in sys.argv:
        position = sys.argv.index("--config")
        if position + 1 < len(sys.argv):
            return Path(sys.argv[position + 1])
    return DEFAULT_CONFIG_PATH


CONFIG_PATH = _config_argument()


@st.cache_data
def load_config(path):
    return load_company_config(path)


config = load_config(str(CONFIG_PATH))
WORKSPACE = CompanyWorkspace.from_config(config, ROOT).ensure()
DATA = WORKSPACE


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=f"{config['company_name']} Valuation Dashboard",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# LOAD CONFIG
# =========================================================

# =========================================================
# DATA HELPERS
# =========================================================

def load_csv(
    filename,
    index_col=None,
):

    path = (
        DATA.path(filename, for_read=True)
    )

    if not path.exists():

        st.error(
            f"Missing file: {filename}. "
            "Run `python src/pipeline.py --config <path>` first."
        )

        st.stop()

    return pd.read_csv(
        path,
        index_col=index_col,
    )


def load_json(filename, default=None):
    path = DATA.path(filename, for_read=True)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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

    mastercard = load_csv("valuation_direct_peer_comps.csv")
    if mastercard.empty:
        mastercard = load_csv("valuation_mastercard_comps.csv")

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

    model_checks = load_csv("model_checks_detail.csv")
    model_health = load_csv("model_health_summary.csv")
    analytics_trends = load_csv("analytics_trends.csv")
    forecast_reasonableness = load_csv("forecast_reasonableness.csv")
    reverse_dcf = load_csv("reverse_dcf.csv", index_col=0)
    reverse_comparison = load_csv("reverse_dcf_comparison.csv")
    memo = load_json("investment_memo.json", {})
    recommendation = load_json("recommendation.json", {})
    claims = load_json("research_claims.json", [])
    evidence = load_json("evidence_store.json", [])
    monitoring = load_json("thesis_monitoring.json", [])
    report_health_path = DATA.path("report_health.csv", for_read=True)
    report_health = pd.read_csv(report_health_path) if report_health_path.exists() else pd.DataFrame()

    three_statements = {}
    for scenario in ("bear", "base", "bull"):
        three_statements[scenario] = {
            "Income statement": load_csv(f"income_statement_{scenario}.csv", index_col=0),
            "Balance sheet": load_csv(f"balance_sheet_{scenario}.csv", index_col=0),
            "Cash flow": load_csv(f"cash_flow_statement_{scenario}.csv", index_col=0),
            "FCFF bridge": load_csv(f"fcff_forecast_{scenario}.csv", index_col=0),
            "Checks": load_csv(f"checks_{scenario}.csv", index_col=0),
        }

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
        "three_statements": three_statements,
        "model_checks": model_checks,
        "model_health": model_health,
        "analytics_trends": analytics_trends,
        "forecast_reasonableness": forecast_reasonableness,
        "reverse_dcf": reverse_dcf,
        "reverse_comparison": reverse_comparison,
        "memo": memo, "recommendation": recommendation, "claims": claims,
        "evidence": evidence, "monitoring": monitoring, "report_health": report_health,
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
    "Primary-Method Central Value",
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
        "Model Health & Analytics",
        "Reverse DCF",
        "Research V3",
    ]
)

overview_tab = tabs[0]
financials_tab = tabs[1]
dcf_tab = tabs[2]
comps_tab = tabs[3]
valuation_tab = tabs[4]
interactive_tab = tabs[5]
statements_tab = tabs[6]
health_tab = tabs[7]
reverse_tab = tabs[8]
research_tab = tabs[9]


with research_tab:
    st.subheader("Research Dashboard V3")
    st.caption("Structured research views; analytical and recommendation logic remain in the pipeline.")
    rec = data["recommendation"]
    memo = data["memo"]
    components = rec.get("components", {})
    rcols = st.columns(5)
    rcols[0].metric("Recommendation", rec.get("rating", "N/A"))
    rcols[1].metric("Base fair value", money(memo.get("fair_value_base_case"), 2))
    rcols[2].metric("Expected return", percentage(memo.get("expected_return")))
    rcols[3].metric("Evidence coverage", percentage(components.get("evidence_coverage")))
    rcols[4].metric("Thesis confidence", percentage(components.get("thesis_confidence")))
    st.markdown("#### Recommendation rationale")
    for reason in rec.get("rationale", []):
        st.write(f"- {reason}")

    claims = pd.DataFrame(data["claims"])
    if not claims.empty:
        for label, kind in (("Investment thesis", "thesis"), ("Risks", "risk"), ("Thesis breakers", "thesis_breaker")):
            st.markdown(f"#### {label}")
            subset = claims[claims["claim_type"].eq(kind)]
            if subset.empty:
                st.info(f"No structured {label.lower()} available.")
            for _, claim in subset.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{claim['title']}**")
                    st.write(claim["statement"])
                    st.caption(f"{str(claim.get('basis','')).replace('_',' ').title()} | {str(claim.get('confidence','')).title()} confidence")

    st.markdown("#### Thesis monitoring")
    monitoring = pd.DataFrame(data["monitoring"])
    if monitoring.empty:
        st.info("No monitored conditions are currently configured.")
    else:
        status_counts = monitoring["status"].value_counts()
        mcols = st.columns(4)
        for col, status in zip(mcols, ("SAFE", "WATCH", "BREACHED", "UNKNOWN")):
            col.metric(status.title(), int(status_counts.get(status, 0)))
        st.dataframe(monitoring, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Evidence and report health")
        if data["report_health"].empty:
            st.warning("Report health has not been generated.")
        else:
            st.dataframe(data["report_health"], use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Market expectations vs base case")
        reverse = data["reverse_comparison"].copy()
        if not reverse.empty:
            fig = px.bar(reverse, x="scenario", y="implied_share_price", color="case_type")
            fig.add_hline(y=current_price, line_dash="dash", annotation_text="Current price")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### KPI coverage and sources")
    evidence = pd.DataFrame(data["evidence"])
    if not evidence.empty:
        available = evidence[evidence["status"].astype(str).str.lower().isin(["available", "converged", "pass"])]
        st.caption(f"{len(available)} of {len(evidence)} evidence records are available. Source URLs are shown when supplied.")
        columns = [x for x in ("evidence_type", "metric", "value", "unit", "period", "source", "source_url", "quality") if x in evidence]
        st.dataframe(evidence[columns], use_container_width=True, hide_index=True, column_config={"source_url": st.column_config.LinkColumn("Source")})

    st.markdown("#### Report downloads")
    downloads = st.columns(3)
    for col, filename, label, mime in (
        (downloads[0], f"{config['ticker'].lower()}_investment_report.html", "Download HTML report", "text/html"),
        (downloads[1], f"{config['ticker'].lower()}_investment_report.pdf", "Download PDF report", "application/pdf"),
        (downloads[2], f"{config['ticker'].lower()}_valuation_model.xlsx", "Download Excel model", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ):
        path = (WORKSPACE.root / "research" / filename) if filename.endswith((".html", ".pdf")) else (DATA / filename)
        if path.exists():
            col.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime)


# =========================================================
# REVERSE DCF TAB
# =========================================================

with reverse_tab:
    st.subheader("Reverse DCF")
    st.caption("Assumptions implied by the current share price using bounded, convergent DCF solves.")
    implied = data["reverse_dcf"].copy()
    mode_labels = {
        "revenue_growth": "Near-term revenue growth",
        "operating_margin": "Operating margin",
        "terminal_growth": "Terminal growth",
    }
    columns = st.columns(max(1, len(implied)))
    for column, (mode, row) in zip(columns, implied.iterrows()):
        column.metric(mode_labels.get(mode, mode.replace("_", " ").title()), percentage(row["implied_assumption"]))
        column.caption(f"Converged in {int(row['iterations'])} iterations; residual {money(row['price_residual'], 4)}")
    comparison = data["reverse_comparison"].copy()
    fig = px.bar(comparison, x="scenario", y="implied_share_price", color="case_type",
                 title="Bear / Base / Bull vs Market-Implied Price")
    fig.add_hline(y=current_price, line_dash="dash", annotation_text="Current price")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(implied.style.format({
        "implied_assumption": "{:.2%}", "target_share_price": "${:,.2f}",
        "implied_share_price": "${:,.2f}", "price_residual": "${:,.6f}",
        "lower_bound": "{:.1%}", "upper_bound": "{:.1%}", "wacc": "{:.2%}",
        "terminal_growth": "{:.2%}",
    }), use_container_width=True)
    workbook_path = DATA / f"{config['ticker'].lower()}_valuation_model.xlsx"
    if workbook_path.exists():
        st.download_button("Download professional Excel model", data=workbook_path.read_bytes(),
                           file_name=workbook_path.name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =========================================================
# MODEL HEALTH + ANALYTICS TAB
# =========================================================

with health_tab:
    st.subheader("Model Health & Analytics")
    st.caption("Check-level diagnostics, explicit tolerances, and historical-versus-forecast reasonableness.")
    checks = data["model_checks"]
    failures = int(checks["status"].eq("FAIL").sum())
    warnings = int(data["forecast_reasonableness"]["status"].eq("WARN").sum())
    unavailable = int(checks["status"].eq("N/A").sum())
    health_columns = st.columns(4)
    health_columns[0].metric("Overall model status", "PASS" if failures == 0 else "FAIL")
    health_columns[1].metric("Failed checks", failures)
    health_columns[2].metric("Reasonableness warnings", warnings)
    health_columns[3].metric("Unavailable checks", unavailable)
    if failures:
        st.error("One or more institutional model checks failed.")
    else:
        st.success("All available model integrity checks pass within their stated tolerances.")

    st.subheader("Validation Summary")
    st.dataframe(data["model_health"], use_container_width=True, hide_index=True)
    with st.expander("Check-level detail"):
        st.dataframe(checks, use_container_width=True, hide_index=True)

    st.subheader("Historical vs Forecast Trends")
    trends = data["analytics_trends"].dropna(subset=["value"]).copy()
    metric_options = sorted(trends["metric"].unique())
    selected_metric = st.selectbox("Analytical metric", metric_options)
    selected_trends = trends[trends["metric"].eq(selected_metric)].copy()
    selected_trends["series"] = np.where(
        selected_trends["scope"].eq("Historical"), "Historical", selected_trends["scenario"].str.title()
    )
    selected_trends["period"] = selected_trends["period"].astype(str)
    fig = px.line(selected_trends, x="period", y="value", color="series", markers=True, title=selected_metric)
    if selected_trends["unit"].eq("ratio").all():
        fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Reasonableness")
    reasonableness = data["forecast_reasonableness"].copy()
    st.dataframe(
        reasonableness.style.format({
            "forecast_average": "{:.1%}", "historical_average": "{:.1%}",
            "historical_min": "{:.1%}", "historical_max": "{:.1%}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )


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
    st.caption("Income statement, balance sheet, cash flow and DCF are driven by the same scenario model.")
    statement_scenario = st.radio("Statement scenario", ["Bear", "Base", "Bull"], index=1, horizontal=True).lower()
    statement_data = data["three_statements"][statement_scenario]
    checks = statement_data["Checks"]
    maximum_error = float(checks["max_abs_error"].max())
    status = "OK" if checks["status"].eq("OK").all() else "ERROR"
    status_columns = st.columns(3)
    status_columns[0].metric("Model status", status)
    status_columns[1].metric("Maximum reconciliation error", f"{maximum_error:,.8f}M")
    status_columns[2].metric(
        "Terminal FCFF", money(statement_data["FCFF bridge"]["fcff"].iloc[-1], 1)
    )
    if status == "OK":
        st.success("All statement roll-forwards and the FCFF bridge reconcile.")
    else:
        st.error("One or more model integrity checks failed.")
    selected_statement = st.selectbox("Statement", list(statement_data))
    statement = statement_data[selected_statement]
    if selected_statement == "Checks":
        numeric_columns = statement.select_dtypes(include=[np.number]).columns
        st.dataframe(statement.style.format({column: "{:,.8f}" for column in numeric_columns}), use_container_width=True)
    else:
        numeric_columns = statement.select_dtypes(include=[np.number]).columns
        st.dataframe(statement.style.format({column: "${:,.1f}M" for column in numeric_columns}), use_container_width=True)


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
            f"{config['company_name']} Implied Valuation Range"
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
        "Primary-Method Central Valuation"
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
        "Primary Mean Base",
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
