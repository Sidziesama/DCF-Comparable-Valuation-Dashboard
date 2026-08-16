from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adapters import get_adapter
from src.company_config import CompanyWorkspace, ConfigError, load_company_config, validate_company_config
from src.comparables import MULTIPLE_COLUMNS, apply_multiple_eligibility
from src.financials import build_ltm
from src.sec_data import build_raw_dataset
from src.historical_model import calculate_ratios
from src.football_field import build_football_field, calculate_central_range


ROOT = Path(__file__).resolve().parents[1]


def test_microsoft_config_selects_software_adapter_and_june_fiscal_year():
    cfg = load_company_config(ROOT / "config" / "microsoft.yaml")
    assert cfg["ticker"] == "MSFT"
    assert cfg["cik"] == "0000789019"
    assert cfg["fiscal_calendar"]["year_end_month"] == 6
    assert get_adapter(cfg["adapter"]).name == "software"


def test_invalid_peer_multiple_is_rejected():
    cfg = load_company_config(ROOT / "config" / "microsoft.yaml")
    cfg["peer_methodology"]["multiple_eligibility"]["ORCL"] = ["price_to_story"]
    with pytest.raises(ConfigError):
        validate_company_config(cfg)


def test_software_kpi_normalization_growth_and_top_down_fallback():
    adapter = get_adapter("software")
    reported = adapter.normalize_kpis([
        {"metric": "microsoft_cloud_revenue", "period": 2024, "value": 100, "unit": "USD millions", "source_url": "https://example.test/1"},
        {"metric": "microsoft_cloud_revenue", "period": 2025, "value": 125, "unit": "USD millions", "source_url": "https://example.test/2"},
        {"metric": "invented_arr", "period": 2025, "value": 50, "unit": "USD millions"},
    ])
    assert reported.loc[reported.metric.eq("invented_arr"), "status"].iat[0] == "not_available"
    derived = adapter.derive_metrics(reported)
    growth = derived.loc[derived.metric.eq("microsoft_cloud_revenue_growth")].iloc[0]
    assert growth["value"] == pytest.approx(0.25)
    assert growth["quality"] == "calculated"
    values, metadata = adapter.forecast_growth("base", {"revenue_growth": [0.1, 0.09]}, 2)
    assert values == [0.1, 0.09]
    assert metadata["fallback_used"] is True


def test_software_segment_bridge_and_weights():
    adapter = get_adapter("software")
    drivers = {
        "productivity_business_processes_growth": [0.10, 0.09],
        "intelligent_cloud_growth": [0.20, 0.18],
        "more_personal_computing_growth": [0.02, 0.02],
        "revenue_growth_weights": {
            "productivity_business_processes_growth": 0.4,
            "intelligent_cloud_growth": 0.4,
            "more_personal_computing_growth": 0.2,
        },
        "mix_pricing_overlay": [0.01, 0.0],
    }
    growth, metadata = adapter.forecast_growth("base", {"operating_drivers": drivers}, 2)
    assert growth == pytest.approx([0.134, 0.112])
    assert metadata["forecast_method"] == "segment_driver_bridge"


def test_configured_xbrl_alias_can_add_a_generic_metric(monkeypatch):
    facts = {"facts": {"us-gaap": {"ShareBasedCompensation": {"units": {"USD": [{
        "val": 1000000, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-07-30",
        "start": "2024-07-01", "end": "2025-06-30",
    }]}}}}}
    monkeypatch.setattr("src.sec_data.fetch_companyfacts", lambda cik: facts)
    raw = build_raw_dataset("789019", {"stock_based_compensation": ["ShareBasedCompensation"]})
    row = raw.loc[raw.metric.eq("stock_based_compensation")].iloc[0]
    assert row["value"] == pytest.approx(1.0)
    assert raw.attrs["selected_tags"]["stock_based_compensation"] == "ShareBasedCompensation"


def test_june_fiscal_calendar_ltm_uses_comparable_nine_month_periods():
    annual = pd.DataFrame([{"metric": "revenue", "fy": 2025, "value": 100, "filed": "2025-07-30"}])
    quarterly = pd.DataFrame([
        {"metric": "revenue", "value": 70, "start": "2023-07-01", "end": "2024-03-31", "filed": "2024-04-25", "source": "companyfacts"},
        {"metric": "revenue", "value": 90, "start": "2024-07-01", "end": "2025-03-31", "filed": "2025-04-25", "source": "companyfacts"},
    ])
    ltm = build_ltm(annual, quarterly)
    assert ltm.set_index("metric").loc["revenue", "ltm"] == pytest.approx(120)


def test_unknown_peers_keep_standard_multiples_and_rules_are_explicit():
    comps = pd.DataFrame({metric: [1.0, 2.0] for metric in MULTIPLE_COLUMNS}, index=["ORCL", "BANK"])
    result = apply_multiple_eligibility(comps, {"BANK": ["pe"]})
    assert result.loc["ORCL", MULTIPLE_COLUMNS].notna().all()
    assert np.isnan(result.loc["BANK", "ev_revenue"])
    assert result.loc["BANK", "pe"] == 2.0


def test_microsoft_workspace_never_seeds_legacy_visa_artifacts(tmp_path):
    legacy = tmp_path / "data" / "processed"
    legacy.mkdir(parents=True)
    (legacy / "historical_model.csv").write_text("visa-only", encoding="utf-8")
    cfg = {"ticker": "MSFT", "storage": {"root": "data/companies"}}
    workspace = CompanyWorkspace.from_config(cfg, tmp_path).ensure()
    assert workspace.root == tmp_path / "data" / "companies" / "msft"
    assert not workspace.path("historical_model.csv").exists()


def test_ebitda_combines_separately_reported_depreciation_and_amortization():
    model = pd.DataFrame({
        "revenue": [100.0], "operating_income": [30.0], "pretax_income": [29.0],
        "tax_expense": [5.0], "net_income": [24.0], "cfo": [35.0], "capex": [10.0],
        "depreciation": [8.0], "amortization": [2.0],
    }, index=[2025])
    result = calculate_ratios(model)
    assert result.loc[2025, "depreciation_and_amortization"] == pytest.approx(10.0)
    assert result.loc[2025, "ebitda"] == pytest.approx(40.0)


def test_football_field_uses_configured_direct_peer_label():
    scenarios = pd.DataFrame({"implied_share_price": [80.0, 100.0, 120.0]}, index=["bear", "base", "bull"])
    direct = pd.DataFrame({"implied_share_price": [90.0, 110.0]})
    result = build_football_field(scenarios, scenarios, direct, 100.0, direct_peer_label="ORCL")
    assert "ORCL Trading Comps" in result["method"].tolist()


def test_central_range_excludes_exit_multiple_by_default():
    field = pd.DataFrame({
        "method": ["DCF - Gordon Growth", "DCF - Exit Multiple", "ORCL Trading Comps"],
        "base": [100.0, 800.0, 120.0],
    })
    central = calculate_central_range(field)
    assert central["mean_base"] == pytest.approx(110.0)
    assert central["maximum_base"] == pytest.approx(120.0)
    assert central["method_count"] == 2


def test_workbook_inputs_are_company_scoped():
    cfg = load_company_config(ROOT / "config" / "microsoft.yaml")
    workspace = CompanyWorkspace.from_config(cfg, ROOT)
    assert workspace.path("trading_comparables.csv").is_relative_to(workspace.root)
    assert workspace.path("football_field.csv").is_relative_to(workspace.root)
