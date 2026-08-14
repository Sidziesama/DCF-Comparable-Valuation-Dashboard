import pandas as pd
import pytest

from src.forecast_model import build_forecast_from_inputs
from src.valuation import run_dcf, run_exit_multiple_dcf
from src.three_statement_model import build_three_statement_forecast


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
    assert statements["checks"].abs().to_numpy().max() < 1e-8
