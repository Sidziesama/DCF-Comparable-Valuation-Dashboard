import pandas as pd
import pytest

from src.forecast_model import build_forecast_from_inputs
from src.valuation import (
    build_reverse_dcf_summary,
    run_dcf,
    run_exit_multiple_dcf,
    solve_reverse_dcf,
)
from src.excel_export import export_valuation_workbook, validate_exported_workbook
from src.three_statement_model import build_three_statement_forecast
from src.model_quality import build_historical_forecast_analytics, build_model_checks


@pytest.fixture
def forecast():
    return build_forecast_from_inputs(
        100.0,
        [0.10, 0.08],
        [0.30, 0.31],
        0.20,
        0.03,
        0.02,
        0.05,
        start_year=2027,
    )


def test_forecast_is_formula_driven(forecast):
    assert forecast.loc[2027, "revenue"] == pytest.approx(110.0)
    assert forecast.loc[2027, "ebitda"] == pytest.approx(36.3)
    assert forecast.loc[2027, "fcff"] == pytest.approx(27.0)


def test_both_terminal_methods_reconcile_bridge(forecast):
    for result in (
        run_dcf(forecast, 0.08, 0.025, 20, 30, 10),
        run_exit_multiple_dcf(forecast, 0.08, 15, 20, 30, 10),
    ):
        assert result["equity_value"] == pytest.approx(
            result["enterprise_value"] + result["cash"] - result["debt"]
        )
        assert result["implied_share_price"] == pytest.approx(result["equity_value"] / 10)


def test_gordon_rejects_invalid_spread(forecast):
    with pytest.raises(ValueError, match="WACC must exceed"):
        run_dcf(forecast, 0.02, 0.025, 20, 30, 10)


def test_reverse_dcf_recovers_known_revenue_growth(forecast):
    target = run_dcf(forecast, 0.08, 0.025, 20, 30, 10)["implied_share_price"]
    result = solve_reverse_dcf(
        100, forecast, target, 0.08, 0.025, 20, 30, 10,
        mode="revenue_growth", operating_margin=[0.30, 0.31],
    )
    assert result["converged"]
    assert result["implied_share_price"] == pytest.approx(target, abs=1e-5)
    assert 0.08 < result["implied_assumption"] < 0.10


def test_reverse_dcf_reports_unreachable_target(forecast):
    with pytest.raises(ValueError, match="outside the values"):
        solve_reverse_dcf(
            100, forecast, 1_000_000, 0.08, 0.025, 20, 30, 10,
            mode="revenue_growth", bounds=(0.0, 0.10),
        )


def test_reverse_dcf_summary_compares_cases(forecast):
    forecasts = {name: forecast.copy() for name in ("bear", "base", "bull")}
    forecasts["bear"]["fcff"] *= 0.9
    forecasts["bull"]["fcff"] *= 1.1
    target = run_dcf(forecast, 0.08, 0.025, 20, 30, 10)["implied_share_price"]
    implied, comparison = build_reverse_dcf_summary(
        forecasts, 100, target, 0.08, 0.025, 20, 30, 10,
        modes=("revenue_growth", "terminal_growth"),
    )
    assert set(implied.index) == {"revenue_growth", "terminal_growth"}
    assert set(comparison["scenario"]) == {"bear", "base", "bull", "market_implied"}


def test_excel_export_opens_and_contains_formula_driven_sheets(tmp_path, forecast):
    forecasts = {name: forecast.copy() for name in ("bear", "base", "bull")}
    historical = pd.DataFrame(
        {"revenue": [90.0, 100.0], "ebitda": [30.0, 35.0]},
        index=[2025, "LTM"],
    )
    reverse = pd.DataFrame(
        {"implied_assumption": [0.09], "target_share_price": [100.0],
         "implied_share_price": [100.0], "price_residual": [0.0],
         "converged": [True], "iterations": [20], "lower_bound": [-0.2],
         "upper_bound": [0.35], "wacc": [0.08], "terminal_growth": [0.025]},
        index=pd.Index(["revenue_growth"], name="mode"),
    )
    output = export_valuation_workbook(
        tmp_path / "model.xlsx", company_name="Example Corp", ticker="EX",
        historical=historical, forecasts=forecasts, assumptions={}, wacc=0.08,
        terminal_growth=0.025, cash=20, debt=30, shares_outstanding=10,
        current_price=100, reverse_dcf=reverse,
    )
    validation = validate_exported_workbook(output, ["Summary", "DCF", "Reverse DCF"])
    assert validation["opens_cleanly"]
    assert validation["missing_sheets"] == []
    assert validation["formula_reference_errors"] == []
    assert validation["formula_count"] >= 10


def test_exit_method_accepts_ebit_plus_da():
    forecast = pd.DataFrame({"fcff": [10], "ebit": [12], "da": [2]})
    result = run_exit_multiple_dcf(forecast, 0.08, 10, 5, 4, 2)
    assert result["terminal_ebitda"] == 14


def test_three_statement_model_reconciles(forecast):
    historical = pd.DataFrame({"revenue": [100.0]}, index=["LTM"])
    balance = pd.DataFrame(
        {
            "metric": ["cash", "accounts_receivable", "current_assets", "ppe", "total_assets", "accounts_payable", "current_liabilities", "short_term_debt", "long_term_debt", "total_liabilities", "equity"],
            "value": [20, 10, 40, 15, 100, 5, 20, 2, 18, 55, 45],
        }
    )
    statements = build_three_statement_forecast(historical, balance, forecast)
    checks = statements["checks"]
    assert checks["status"].eq("OK").all()
    assert checks["max_abs_error"].max() < 1e-8
    assert statements["cash_flow_statement"]["ending_cash"].equals(
        statements["balance_sheet"]["cash"]
    )
    expected_fcff = (
        forecast["nopat"] + forecast["da"] - forecast["capex"]
        - statements["fcff_forecast"]["change_nwc"]
    )
    assert statements["fcff_forecast"]["fcff"].tolist() == pytest.approx(
        expected_fcff.tolist()
    )


def test_three_statement_debt_and_equity_roll_forwards(forecast):
    historical = pd.DataFrame({"revenue": [100.0]}, index=["LTM"])
    balance = pd.DataFrame({
        "metric": ["cash", "accounts_receivable", "current_assets", "ppe",
            "total_assets", "accounts_payable", "current_liabilities",
            "short_term_debt", "long_term_debt", "total_liabilities", "equity"],
        "value": [20, 10, 40, 15, 100, 5, 20, 2, 18, 55, 45],
    })
    statements = build_three_statement_forecast(
        historical, balance, forecast,
        {"debt_repayment": [2.0, 3.0], "dividends_pct_net_income": 0.25,
         "buybacks_pct_net_income": 0.50},
    )
    assert statements["balance_sheet"].loc[2028, "long_term_debt"] == pytest.approx(13.0)
    assert statements["checks"]["status"].eq("OK").all()


def test_three_statement_rejects_incomplete_operating_forecast():
    historical = pd.DataFrame({"revenue": [100.0]}, index=["LTM"])
    balance = pd.DataFrame({"metric": ["cash"], "value": [20.0]})
    with pytest.raises(ValueError, match="operating_forecast is missing"):
        build_three_statement_forecast(
            historical, balance, pd.DataFrame({"revenue": [110.0]}, index=[2027])
        )


def _quality_inputs():
    historical = pd.DataFrame({
        "revenue": [80.0, 90.0, 100.0], "operating_income": [24.0, 28.0, 32.0],
        "ebitda": [27.0, 31.0, 35.0], "net_income": [18.0, 21.0, 24.0],
        "fcf": [20.0, 23.0, 26.0], "effective_tax_rate": [0.20] * 3,
        "capex": [2.0, 2.2, 2.5],
    }, index=[2024, 2025, "LTM"])
    balance = pd.DataFrame({
        "metric": ["cash", "accounts_receivable", "current_assets", "ppe",
            "total_assets", "accounts_payable", "current_liabilities",
            "short_term_debt", "long_term_debt", "total_liabilities", "equity"],
        "value": [20, 10, 40, 15, 100, 5, 20, 2, 18, 55, 45],
    })
    forecasts = {
        name: build_forecast_from_inputs(100, growth, margin, 0.2, 0.03, 0.02, 0.05,
            scenario=name, start_year=2027)
        for name, growth, margin in (
            ("bear", [0.04, 0.04], 0.30), ("base", [0.06, 0.06], 0.32),
            ("bull", [0.08, 0.08], 0.34))
    }
    statements = {name: build_three_statement_forecast(historical, balance, frame)
                  for name, frame in forecasts.items()}
    return historical, forecasts, statements


def test_institutional_model_checks_pass_and_are_explicit():
    _, forecasts, statements = _quality_inputs()
    detail, summary = build_model_checks(
        statements, wacc=0.08, terminal_growth=0.025,
        terminal_value_pct_ev={name: 0.75 for name in forecasts},
    )
    assert {"actual", "expected", "variance", "tolerance", "status"}.issubset(detail.columns)
    assert not detail["status"].eq("FAIL").any()
    assert summary["status"].eq("PASS").all()
    assert detail["check"].str.contains("Bear ≤ Base ≤ Bull").any()


def test_terminal_concentration_and_scenario_ordering_can_fail():
    _, _, statements = _quality_inputs()
    statements["bear"]["fcff_forecast"].loc[2027, "revenue"] = 999
    detail, _ = build_model_checks(
        statements, wacc=0.08, terminal_growth=0.025,
        terminal_value_pct_ev={"bear": 0.95, "base": 0.75, "bull": 0.75},
    )
    failed = detail[detail["status"].eq("FAIL")]
    assert failed["check"].str.contains("Terminal value concentration").any()
    assert failed["check"].str.contains("Bear ≤ Base ≤ Bull: revenue").any()


def test_analytics_include_trends_and_graceful_unavailable_metrics():
    historical, forecasts, _ = _quality_inputs()
    trends, diagnostics = build_historical_forecast_analytics(historical, forecasts)
    assert {"Revenue CAGR", "Operating margin", "EBITDA margin", "FCF conversion",
            "Reinvestment rate", "Incremental operating margin"}.issubset(set(trends["metric"]))
    roic = trends[trends["metric"].eq("ROIC")]
    assert roic["value"].isna().all()
    assert set(diagnostics["status"]).issubset({"PASS", "WARN", "N/A"})
