import numpy as np
import pandas as pd
import pytest

from src.forecast_model import build_forecast_from_inputs
from src.historical_valuation import build_historical_valuation_intelligence, calculate_historical_multiples
from src.investment_intelligence import build_business_quality_metrics, calculate_fcf_conversion, calculate_incremental_margin, calculate_roic
from src.valuation import build_expectation_matrix, build_reverse_dcf_summary


def _forecast():
    return build_forecast_from_inputs(100, [0.10, 0.10], [0.30, 0.30], 0.20,
                                      0.03, 0.02, 0.05, start_year=2027)


def test_roic_fcf_conversion_and_incremental_margin_math():
    assert calculate_roic(30, 0.20, 20, 50, 10) == pytest.approx(0.40)
    assert calculate_fcf_conversion(18, 24) == pytest.approx(0.75)
    assert calculate_incremental_margin(33, 30, 110, 100) == pytest.approx(0.30)


def test_business_quality_retains_unavailable_metrics_with_flags():
    historical = pd.DataFrame({"revenue": [90, 100], "operating_income": [27, 30],
        "net_income": [20, 22], "fcf": [21, 24], "effective_tax_rate": [0.2, 0.2]},
        index=[2025, "LTM"])
    result = build_business_quality_metrics(historical)
    interest = result[result.metric.eq("Interest coverage")]
    assert interest.value.isna().all()
    assert interest.status.eq("not_available").all()


def test_expectation_matrix_has_price_and_gap_for_every_cell():
    result = build_expectation_matrix(100, _forecast(), [0.08, 0.10], [0.28, 0.32],
        0.08, 0.025, 20, 30, 10, current_price=30)
    assert len(result) == 4
    assert np.allclose(result.valuation_gap, result.implied_share_price - 30)
    assert result.iloc[-1].implied_share_price > result.iloc[0].implied_share_price


def test_reverse_summary_records_solver_failure_without_breaking_pipeline():
    forecast = _forecast()
    implied, _ = build_reverse_dcf_summary({name: forecast for name in ("bear", "base", "bull")},
        100, 1_000_000, 0.08, 0.025, 20, 30, 10, modes=("revenue_growth",))
    assert not bool(implied.loc["revenue_growth", "converged"])
    assert implied.loc["revenue_growth", "status"] == "failed"
    assert "outside the values" in implied.loc["revenue_growth", "failure_reason"]


def test_historical_multiples_use_only_aligned_periods():
    financials = pd.DataFrame({"period": [2024, 2025], "net_income": [10, 12],
        "ebitda": [15, 18], "ebit": [13, 16], "fcf": [9, 11],
        "debt": [5, 6], "cash": [2, 3]})
    market = pd.DataFrame({"period": [2025], "market_cap": [120]})
    result = calculate_historical_multiples(financials, market)
    assert set(result.period) == {2025}
    assert result.loc[result.metric.eq("P/E"), "value"].iloc[0] == pytest.approx(10)
    assert result.loc[result.metric.eq("EV / EBIT"), "value"].iloc[0] == pytest.approx(123 / 16)


def test_missing_history_is_explicit_but_current_and_peers_work():
    historical = pd.DataFrame({"net_income": [20], "ebitda": [30],
        "operating_income": [28], "fcf": [24]}, index=["LTM"])
    peers = pd.DataFrame({"pe": [20, 30], "ev_ebitda": [15, 17], "ev_ebit": [18, 20]})
    intelligence, summary = build_historical_valuation_intelligence(historical,
        current_market_cap=500, current_enterprise_value=520, peer_multiples=peers)
    assert summary.history_status.eq("not_available").all()
    assert summary.current.notna().all()
    assert intelligence.status.eq("not_available").any()
