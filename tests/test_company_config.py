import json

import pytest

from src.company_config import ConfigError, CompanyWorkspace, load_company_config


def _config(tmp_path, body):
    path = tmp_path / "company.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_minimal_config_loads_with_defaults(tmp_path):
    path = _config(tmp_path, "ticker: ex\ncompany_name: Example Corp\ncik: '1234'\n")
    config = load_company_config(path, pipeline=False)
    assert config["ticker"] == "EX"
    assert config["cik"] == "0000001234"
    assert config["historical_years"] == 5
    assert config["adapter"] == "generic"


def test_config_errors_name_the_missing_field(tmp_path):
    with pytest.raises(ConfigError, match="company.cik"):
        load_company_config(_config(tmp_path, "ticker: EX\ncompany_name: Example\n"), pipeline=False)


def test_company_workspace_routes_and_falls_back(tmp_path):
    config = {"ticker": "EX", "storage": {"root": str(tmp_path / "companies")}}
    workspace = CompanyWorkspace.from_config(config, root=tmp_path).ensure()
    assert workspace.path("sec_raw.csv").parent.name == "raw"
    assert workspace.path("historical_model.csv").parent.name == "normalized"
    assert workspace.path("forecast_base.csv").parent.name == "model"
    assert workspace.path("football_field.csv").parent.name == "research"
    workspace.legacy_processed.mkdir(parents=True)
    legacy = workspace.legacy_processed / "old.csv"
    legacy.write_text("value\n1\n", encoding="utf-8")
    assert workspace.path("old.csv", for_read=True) == legacy


def test_lineage_manifest_records_sources_and_artifacts(tmp_path):
    workspace = CompanyWorkspace(tmp_path / "companies" / "ex", "EX", tmp_path / "processed").ensure()
    path = workspace.write_manifest(
        [{"path": "normalized/financials_annual.csv", "derived_from": ["raw/sec_raw.csv"]}],
        [{"name": "SEC CompanyFacts", "url": "https://data.sec.gov"}],
        "2026-01-01T00:00:00+00:00",
    )
    manifest = json.loads(path.read_text())
    assert manifest["ticker"] == "EX"
    assert manifest["artifacts"][0]["derived_from"] == ["raw/sec_raw.csv"]


def test_refresh_manifest_stays_inside_company_workspace(tmp_path):
    first = CompanyWorkspace(tmp_path / "companies" / "one", "ONE", tmp_path / "processed").ensure()
    second = CompanyWorkspace(tmp_path / "companies" / "two", "TWO", tmp_path / "processed").ensure()
    first.write_manifest([], [], "2026-01-01T00:00:00+00:00")
    second.write_manifest([], [], "2026-01-01T00:00:00+00:00")
    (first.root / "research" / "only_one.csv").write_text("value\n1\n")
    first.refresh_manifest_artifacts()
    manifest = json.loads((first.root / "research" / "lineage_manifest.json").read_text())
    paths = {item["path"] for item in manifest["artifacts"]}
    assert "research/only_one.csv" in paths
    assert all("two" not in path for path in paths)
