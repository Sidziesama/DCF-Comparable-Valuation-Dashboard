from pathlib import Path
import os
import argparse
import copy
from dataclasses import asdict

import pandas as pd
from dotenv import load_dotenv

from company_config import (
    DEFAULT_CONFIG_PATH,
    CompanyWorkspace,
    load_company_config,
    utc_timestamp,
)


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

from three_statement_model import (
    STATEMENT_TOLERANCE,
    build_three_statement_forecast,
)

from model_quality import (
    build_historical_forecast_analytics,
    build_model_checks,
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
    build_reverse_dcf_summary,
    build_expectation_matrix,
)

from investment_intelligence import (
    build_business_quality_metrics,
    combine_investment_intelligence,
)
from historical_valuation import build_historical_valuation_intelligence
from adapters import get_adapter
from operating_intelligence import (
    build_adapter_health,
    build_investment_diagnostics,
    build_scenario_decomposition,
    build_valuation_attribution,
)
from research_product import build_research_product
from reporting import generate_report_artifacts

from excel_export import export_valuation_workbook

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

def load_config(path=DEFAULT_CONFIG_PATH):
    """Compatibility wrapper around the validated configuration loader."""
    return load_company_config(path)


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

def run(config_path=DEFAULT_CONFIG_PATH):

    # -----------------------------------------------------
    # 1. Environment + config
    # -----------------------------------------------------

    load_dotenv(
        ROOT / ".env"
    )

    cfg = load_config(config_path)
    workspace = CompanyWorkspace.from_config(cfg, ROOT).ensure()
    # Keep the historical variable name locally; CompanyWorkspace routes each
    # artifact into raw/normalized/derived/model/research and falls back to the
    # former data/processed location when reading pre-migration outputs.
    processed = workspace
    retrieved_at = utc_timestamp()

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
        cfg["cik"],
        taxonomy_aliases=(cfg.get("taxonomy", {}) or {}).get("aliases", {}),
    )

    raw["source_system"] = raw.get("source", "companyfacts")
    raw["source_url"] = (
        "https://data.sec.gov/api/xbrl/companyfacts/"
        f"CIK{cfg['cik']}.json"
    )
    raw["retrieved_at"] = retrieved_at
    raw["company_ticker"] = ticker

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
                taxonomy_aliases=(cfg.get("taxonomy", {}) or {}).get("aliases", {}),
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
            "source_system",
            "source_url",
            "retrieved_at",
            "company_ticker",
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

        raw["source_system"] = raw["source"]
        raw["source_url"] = raw["source_url"].fillna(
            "https://www.sec.gov/Archives/edgar/data/"
        )
        raw["retrieved_at"] = raw["retrieved_at"].fillna(retrieved_at)
        raw["company_ticker"] = raw["company_ticker"].fillna(ticker)

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

    for frame in (annual, quarterly):
        frame["retrieved_at"] = retrieved_at
        frame["company_ticker"] = ticker
        if "source_system" not in frame.columns:
            frame["source_system"] = frame.get("source", "companyfacts")

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

    annual_years = pd.to_numeric(annual["fy"], errors="coerce").dropna()
    if annual_years.empty:
        raise RuntimeError("No annual fiscal years available after SEC normalization.")
    end_year = int(annual_years.max())
    start_year = end_year - int(cfg["historical_years"]) + 1
    annual_wide = build_annual_wide(annual, start_year=start_year, end_year=end_year)

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

    # The adapter is optional: disclosed operating KPIs and sector economics
    # enrich the model, but can never make the generic valuation pipeline fail.
    adapter_name = cfg.get("adapter", "generic")
    adapter_error = ""
    forecast_metadata = {}
    operating_kpis = pd.DataFrame()
    model_assumptions = copy.deepcopy(assumptions)
    try:
        adapter = get_adapter(adapter_name)
        disclosed = adapter.normalize_kpis(cfg.get("operating_kpis", []))
        if not disclosed.empty:
            disclosed["company_ticker"] = ticker
            disclosed["retrieved_at"] = disclosed["retrieved_at"].replace("", retrieved_at).fillna(retrieved_at)
        derived = adapter.derive_metrics(disclosed)
        operating_kpis = pd.concat([disclosed, derived], ignore_index=True)
        years = int(assumptions.get("forecast_years", cfg.get("forecast_years", 5)))
        for scenario, scenario_cfg in model_assumptions["scenarios"].items():
            growth, metadata = adapter.forecast_growth(scenario, scenario_cfg, years)
            scenario_cfg["revenue_growth"] = growth
            forecast_metadata[scenario] = metadata
    except Exception as exc:
        adapter_error = f"{type(exc).__name__}: {exc}"
        adapter = get_adapter("generic")
        operating_kpis = adapter.normalize_kpis(None)
        for scenario, scenario_cfg in model_assumptions["scenarios"].items():
            growth, metadata = adapter.forecast_growth(scenario, scenario_cfg, len(scenario_cfg["revenue_growth"]))
            scenario_cfg["revenue_growth"] = growth
            metadata["fallback_reason"] = f"Adapter failed; generic fallback used. {adapter_error}"
            forecast_metadata[scenario] = metadata
    operating_kpis.to_csv(processed / "operating_kpis.csv", index=False)

    operating_forecasts = (
        build_all_scenarios(
            historical,
            model_assumptions,
            start_year=int(assumptions.get("forecast_start_year", end_year + 2)),
        )
    )

    forecasts = {}
    all_statements = {}
    three_statement_assumptions = assumptions.get("three_statement", {})

    for scenario, operating_forecast in (
        operating_forecasts.items()
    ):

        # Scenario-specific financing/capital-allocation drivers may override
        # generic defaults without leaking industry logic into the engine.
        linked_assumptions = dict(three_statement_assumptions)
        linked_assumptions.update(
            model_assumptions.get("scenarios", {}).get(scenario, {}).get("three_statement", {})
        )
        statements = build_three_statement_forecast(
            historical=historical,
            latest_balance=latest_balance,
            operating_forecast=operating_forecast,
            assumptions=linked_assumptions,
        )
        all_statements[scenario] = statements

        forecast = statements["fcff_forecast"]
        forecasts[scenario] = forecast

        forecast.to_csv(
            processed
            / f"forecast_{scenario}.csv"
        )

        for statement_name in [
            "income_statement",
            "balance_sheet",
            "cash_flow_statement",
            "fcff_forecast",
            "working_capital_schedule",
            "ppe_schedule",
            "debt_schedule",
            "equity_schedule",
            "capital_returns_schedule",
            "checks",
        ]:
            statements[statement_name].to_csv(
                processed / f"{statement_name}_{scenario}.csv"
            )

        maximum_error = float(statements["checks"]["max_abs_error"].max())
        if maximum_error > STATEMENT_TOLERANCE:
            raise RuntimeError(
                f"{scenario.title()} three-statement model failed checks: "
                f"{maximum_error:,.8f}"
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

    analytics_trends, reasonableness = build_historical_forecast_analytics(
        historical=historical,
        forecasts=forecasts,
        latest_balance=latest_balance,
        statements=all_statements,
    )
    analytics_trends.to_csv(processed / "analytics_trends.csv", index=False)
    reasonableness.to_csv(processed / "forecast_reasonableness.csv", index=False)


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
        benchmark=(cfg.get("market", {}) or {}).get("benchmark", "SPY"),
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

    decomposition_assumptions = copy.deepcopy(model_assumptions)
    decomposition_assumptions["resolved_wacc"] = wacc
    scenario_decomposition = build_scenario_decomposition(
        forecasts, scenario_values, decomposition_assumptions, current_price, forecast_metadata,
        config_lineage=str(Path(cfg["_config_path"]).relative_to(ROOT)),
    )
    scenario_decomposition.to_csv(processed / "scenario_decomposition.csv", index=False)
    valuation_attribution = build_valuation_attribution(
        forecasts, model_assumptions, wacc, terminal_growth, cash, debt, shares_outstanding,
    )
    valuation_attribution.to_csv(processed / "valuation_attribution.csv", index=False)

    check_detail, check_summary = build_model_checks(
        all_statements,
        wacc=wacc,
        terminal_growth=terminal_growth,
        terminal_value_pct_ev=scenario_values["terminal_value_pct_ev"].to_dict(),
    )
    check_detail.to_csv(processed / "model_checks_detail.csv", index=False)
    check_summary.to_csv(processed / "model_health_summary.csv", index=False)
    failed_checks = check_detail[check_detail["status"].eq("FAIL")]
    if not failed_checks.empty:
        failures = failed_checks[["scenario", "period", "check"]].to_dict("records")
        raise RuntimeError(f"Institutional model checks failed: {failures}")

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
    # 14. REVERSE DCF + EXCEL EXPORT
    # =====================================================

    section("12. REVERSE DCF + EXCEL EXPORT")
    reverse_dcf, reverse_comparison = build_reverse_dcf_summary(
        forecasts=forecasts,
        base_revenue=float(historical.loc["LTM", "revenue"]),
        target_share_price=current_price,
        wacc=wacc,
        terminal_growth=terminal_growth,
        cash=cash,
        debt=debt,
        shares_outstanding=shares_outstanding,
    )
    reverse_dcf.to_csv(processed / "reverse_dcf.csv")
    reverse_comparison.to_csv(processed / "reverse_dcf_comparison.csv", index=False)

    investment_diagnostics = build_investment_diagnostics(
        scenario_decomposition, valuation_attribution, reverse_dcf,
        historical, forecasts, current_price,
    )
    investment_diagnostics.to_csv(processed / "investment_diagnostics.csv", index=False)

    expectation_matrix = build_expectation_matrix(
        base_revenue=float(historical.loc["LTM", "revenue"]),
        reference_forecast=forecasts["base"],
        growth_values=[0.06, 0.08, 0.10, 0.12, 0.14],
        margin_values=[0.58, 0.60, 0.62, 0.64, 0.66],
        wacc=wacc, terminal_growth=terminal_growth, cash=cash, debt=debt,
        shares_outstanding=shares_outstanding, current_price=current_price,
    )
    expectation_matrix.to_csv(processed / "expectation_matrix.csv", index=False)

    business_quality = build_business_quality_metrics(
        historical, forecasts, latest_balance, all_statements,
    )
    business_quality.to_csv(processed / "business_quality_metrics.csv", index=False)

    peer_path = processed / "trading_comparables.csv"
    peer_multiples = pd.read_csv(peer_path) if peer_path.exists() else None
    enterprise_value = market.get("enterprise_value")
    if enterprise_value is None:
        enterprise_value = market_cap + debt - cash
    else:
        enterprise_value = float(enterprise_value) / 1_000_000
    valuation_intelligence, historical_valuation_summary = (
        build_historical_valuation_intelligence(
            historical, current_market_cap=market_cap,
            current_enterprise_value=enterprise_value,
            market_history=None, peer_multiples=peer_multiples,
        )
    )
    historical_valuation_summary.to_csv(
        processed / "historical_valuation_summary.csv", index=False,
    )

    reverse_rows = []
    for mode, result in reverse_dcf.iterrows():
        reverse_rows.append({
            "category": "expectations", "metric": f"Market-implied {mode.replace('_', ' ')}",
            "scope": "market", "scenario": "market_implied", "period": "Forecast",
            "value": result.get("implied_assumption"), "units": "ratio",
            "source": "reverse DCF", "lineage": "derived/reverse_dcf.csv",
            "status": result.get("status", "converged" if result.get("converged") else "failed"),
            "quality": "solver_output" if result.get("converged") else "solver_failure",
            "interpretation": result.get("failure_reason", ""),
        })
    investment_intelligence = combine_investment_intelligence(
        business_quality, valuation_intelligence, pd.DataFrame(reverse_rows),
    )
    investment_intelligence.to_csv(processed / "investment_intelligence.csv", index=False)

    intelligence_health = pd.DataFrame([
        {"category": "Business quality", "status": "PASS" if not business_quality.empty else "N/A",
         "available": int(business_quality["value"].notna().sum()), "unavailable": int(business_quality["value"].isna().sum()),
         "detail": "Unavailable metrics are retained with lineage and quality flags."},
        {"category": "Reverse DCF", "status": "PASS" if reverse_dcf["converged"].any() else "WARN",
         "available": int(reverse_dcf["converged"].sum()), "unavailable": int((~reverse_dcf["converged"]).sum()),
         "detail": "Solver failures are non-fatal and carry explicit reasons."},
        {"category": "Historical valuation", "status": "PASS" if historical_valuation_summary["history_status"].eq("available").any() else "N/A",
         "available": int(historical_valuation_summary["history_status"].eq("available").sum()),
         "unavailable": int(historical_valuation_summary["history_status"].ne("available").sum()),
         "detail": "Historical market values are unavailable; current/LTM and peer comparisons remain available."},
    ])
    intelligence_health = pd.concat([
        intelligence_health,
        build_adapter_health(adapter_name, operating_kpis, forecast_metadata, adapter_error),
    ], ignore_index=True)
    intelligence_health.to_csv(processed / "investment_intelligence_health.csv", index=False)

    # Structured research product. Core valuation has already completed; any
    # failure is explicit and does not silently alter the model or its rating.
    try:
        research_product = build_research_product(
            cfg, workspace, current_price=current_price, scenario_values=scenario_values,
            operating_kpis=operating_kpis, scenario_decomposition=scenario_decomposition,
            valuation_attribution=valuation_attribution, reverse_dcf=reverse_dcf,
            diagnostics=investment_diagnostics, business_quality=business_quality,
            model_health=check_summary,
        )
        recommendation = research_product["recommendation"]
        print(f"Deterministic recommendation: {recommendation['rating']}")
    except Exception as exc:
        raise RuntimeError(f"Research product generation failed explicitly: {type(exc).__name__}: {exc}") from exc

    # Refresh consolidated valuation before Excel/report generation so every
    # presentation artifact belongs to this run, never a stale prior workspace.
    build_valuation_summary(config_path=config_path)
    optional_outputs = {}
    for key, filename in {
        "trading_comps": "trading_comparables.csv",
        "football_field": "football_field.csv",
    }.items():
        path = processed / filename
        optional_outputs[key] = pd.read_csv(path, index_col=0) if path.exists() else None
    workbook_path = processed / f"{ticker.lower()}_valuation_model.xlsx"
    export_valuation_workbook(
        workbook_path,
        company_name=cfg["company_name"], ticker=ticker,
        historical=historical, forecasts=forecasts, assumptions=assumptions,
        wacc=wacc, terminal_growth=terminal_growth, cash=cash, debt=debt,
        shares_outstanding=shares_outstanding, current_price=current_price,
        reverse_dcf=reverse_dcf, statements=all_statements,
        trading_comps=optional_outputs["trading_comps"],
        football_field=optional_outputs["football_field"],
        dcf_sensitivity=sensitivity, model_checks=check_detail,
        analytics=analytics_trends, wacc_report=wacc_report,
    )
    print(reverse_dcf.round(4).to_string())
    print(f"Excel model: {workbook_path}")

    reporting_cfg = cfg.get("reporting", {}) or {}
    if reporting_cfg.get("enabled", True):
        try:
            report_result = generate_report_artifacts(
                workspace, memo=asdict(research_product["memo"]),
                claims=[asdict(x) for x in research_product["claims"]],
                evidence=[asdict(x) for x in research_product["evidence"]],
                recommendation=recommendation, config=cfg,
                artifacts={"generated_at": utc_timestamp(), "valuation_tables": {
                    "DCF scenarios": scenario_values.reset_index().to_dict("records"),
                    "Reverse DCF": reverse_dcf.reset_index().to_dict("records"),
                    "Football field": optional_outputs["football_field"].reset_index().to_dict("records") if optional_outputs["football_field"] is not None else [],
                }},
            )
            print(f"Investment report: {report_result['html']}")
            if report_result["pdf"]:
                print(f"PDF report: {report_result['pdf']}")
        except Exception as exc:
            message = f"Report generation warning: {type(exc).__name__}: {exc}"
            print(message)
            pd.DataFrame([{"check": "report_generation", "status": "FAIL", "count": 1,
                           "detail": message}]).to_csv(workspace.root / "research" / "report_health.csv", index=False)
            if reporting_cfg.get("required", False):
                raise RuntimeError(message) from exc


    # =====================================================
    # 15. FINAL SUMMARY
    # =====================================================

    section("13. DCF SUMMARY")

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

    artifacts = []
    for path in sorted(workspace.root.rglob("*")):
        if path.is_file() and path.name != "lineage_manifest.json":
            relative = str(path.relative_to(workspace.root))
            derived_from = [] if relative.startswith("raw/") else ["raw/sec_raw.csv"]
            artifacts.append({"path": relative, "derived_from": derived_from})
    manifest_path = workspace.write_manifest(
        artifacts=artifacts,
        sources=[
            {
                "name": "SEC CompanyFacts / filing XBRL",
                "url": "https://data.sec.gov/",
                "retrieved_at": retrieved_at,
            },
            {
                "name": cfg.get("sources", {}).get("market", {}).get("provider", "yfinance"),
                "retrieved_at": retrieved_at,
            },
        ],
        retrieved_at=retrieved_at,
    )

    print(
        f"Outputs saved to:\n"
        f"{workspace.root}"
    )
    print(f"Artifact manifest:\n{manifest_path}")
    return {
        "ticker": ticker,
        "company_name": cfg["company_name"],
        "workspace": workspace.root,
        "manifest": manifest_path,
        "workbook": workbook_path,
        "report_html": workspace.path(f"{ticker.lower()}_investment_report.html"),
        "report_pdf": workspace.path(f"{ticker.lower()}_investment_report.pdf"),
    }


def run_consolidated_valuation(config_path=DEFAULT_CONFIG_PATH):
    """Print the consolidated artifacts already refreshed by the core run."""
    section("14. CONSOLIDATED VALUATION")
    config = load_config(config_path)
    workspace = CompanyWorkspace.from_config(config, ROOT).ensure()
    consolidated = pd.read_csv(workspace / "valuation_summary.csv")
    football_field = pd.read_csv(workspace / "football_field.csv")
    central_range = pd.read_csv(workspace / "central_valuation_range.csv").iloc[0].to_dict()

    print("\nCONSOLIDATED VALUATION")
    print(consolidated.round(4).to_string(index=False))
    print("\nVALUATION FOOTBALL FIELD")
    print(football_field.round(2).to_string(index=False))
    print("\nCENTRAL VALUATION RANGE")
    for key, value in central_range.items():
        display = f"{int(value)}" if key == "method_count" else f"${value:,.2f}"
        print(f"{key:20s}: {display}")
    return workspace.refresh_manifest_artifacts()


def main(argv=None):
    """CLI entry point kept separate so config selection is easy to test."""
    parser = argparse.ArgumentParser(description="Run a company valuation research pipeline.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a validated company YAML config (default: config/company.yaml).",
    )
    args = parser.parse_args(argv)
    summary = run(args.config)
    manifest = run_consolidated_valuation(args.config) or summary["manifest"]
    section("RUN COMPLETE")
    print(f"Company:           {summary['company_name']} ({summary['ticker']})")
    print(f"Workspace:         {summary['workspace']}")
    print(f"Artifact manifest: {manifest}")
    print(f"Excel model:       {summary['workbook']}")
    print(f"HTML report:       {summary['report_html']}")
    if summary["report_pdf"].exists():
        print(f"PDF report:        {summary['report_pdf']}")
    return summary


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
