"""Validated company configuration and company-scoped artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "company.yaml"


class ConfigError(ValueError):
    """Raised when a company configuration cannot safely drive the pipeline."""


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ConfigError(f"Missing required config field: {location}.{key}")
    return value


def validate_company_config(config: Mapping[str, Any], *, pipeline: bool = True) -> None:
    if not isinstance(config, Mapping):
        raise ConfigError("Company config must be a YAML mapping.")
    for key in ("ticker", "company_name", "cik"):
        _required(config, key, "company")
    ticker = config["ticker"]
    if not isinstance(ticker, str) or not ticker.strip():
        raise ConfigError("company.ticker must be a non-empty string.")
    cik = str(config["cik"]).removeprefix("CIK")
    if not cik.isdigit() or len(cik) > 10:
        raise ConfigError("company.cik must contain at most 10 digits (with optional CIK prefix).")
    for key in ("historical_years", "forecast_years"):
        value = config.get(key, 5)
        if not isinstance(value, int) or value < 1:
            raise ConfigError(f"company.{key} must be a positive integer.")
    fiscal = config.get("fiscal_calendar", {}) or {}
    if fiscal:
        month = fiscal.get("year_end_month")
        day = fiscal.get("year_end_day")
        if not isinstance(month, int) or not 1 <= month <= 12:
            raise ConfigError("company.fiscal_calendar.year_end_month must be an integer from 1 to 12.")
        if not isinstance(day, int) or not 1 <= day <= 31:
            raise ConfigError("company.fiscal_calendar.year_end_day must be an integer from 1 to 31.")
    methodology = config.get("peer_methodology", {}) or {}
    valid_multiples = {"ev_revenue", "ev_ebitda", "ev_ebit", "pe"}
    for peer, multiples in (methodology.get("multiple_eligibility", {}) or {}).items():
        if not isinstance(multiples, list) or not set(multiples).issubset(valid_multiples):
            raise ConfigError(f"company.peer_methodology.multiple_eligibility.{peer} contains an invalid multiple.")
    if not pipeline:
        return
    assumptions = config.get("assumptions")
    if not isinstance(assumptions, Mapping):
        raise ConfigError("Missing required config mapping: company.assumptions")
    scenarios = assumptions.get("scenarios")
    if not isinstance(scenarios, Mapping):
        raise ConfigError("Missing required config mapping: company.assumptions.scenarios")
    for name in ("bear", "base", "bull"):
        scenario = scenarios.get(name)
        if not isinstance(scenario, Mapping):
            raise ConfigError(f"Missing required scenario: company.assumptions.scenarios.{name}")
        for driver in ("revenue_growth", "operating_margin", "tax_rate"):
            _required(scenario, driver, f"company.assumptions.scenarios.{name}")
    wacc = assumptions.get("wacc")
    if not isinstance(wacc, Mapping):
        raise ConfigError("Missing required config mapping: company.assumptions.wacc")
    for field in ("risk_free_rate", "equity_risk_premium", "pre_tax_cost_of_debt"):
        _required(wacc, field, "company.assumptions.wacc")
    _required(assumptions, "terminal_growth", "company.assumptions")


def load_company_config(path: str | Path = DEFAULT_CONFIG_PATH, *, pipeline: bool = True) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Company config not found: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    validate_company_config(config, pipeline=pipeline)
    result = dict(config)
    result["ticker"] = str(result["ticker"]).strip().upper()
    result["cik"] = str(result["cik"]).removeprefix("CIK").zfill(10)
    result.setdefault("historical_years", 5)
    result.setdefault("forecast_years", 5)
    result.setdefault("peers", {})
    result.setdefault("adapter", "generic")
    result.setdefault("storage", {})
    result["_config_path"] = str(config_path)
    return result


_ARTIFACT_GROUPS = {
    "sec_raw.csv": "raw",
    "financials_annual.csv": "normalized",
    "financials_quarterly.csv": "normalized",
    "historical_model.csv": "normalized",
    "latest_balance_sheet.csv": "normalized",
    "wacc_report.csv": "derived",
    "analytics_trends.csv": "derived",
    "forecast_reasonableness.csv": "derived",
    "scenario_valuation.csv": "derived",
    "dcf_sensitivity_v2.csv": "derived",
    "reverse_dcf.csv": "derived",
    "reverse_dcf_comparison.csv": "derived",
    "expectation_matrix.csv": "research",
    "business_quality_metrics.csv": "research",
    "historical_valuation_summary.csv": "research",
    "investment_intelligence.csv": "research",
    "operating_kpis.csv": "research",
    "scenario_decomposition.csv": "research",
    "valuation_attribution.csv": "research",
    "investment_diagnostics.csv": "research",
    "model_checks_detail.csv": "model",
    "model_health_summary.csv": "model",
    "investment_intelligence_health.csv": "model",
    "research_health.csv": "model",
    "evidence_store.json": "research",
    "evidence_store.csv": "research",
    "research_claims.json": "research",
    "research_claims.csv": "research",
    "investment_memo.json": "research",
    "recommendation.json": "research",
    "thesis_monitoring.json": "research",
    "thesis_monitoring.csv": "research",
    "report_health.csv": "research",
    "report_manifest.json": "research",
    "lineage_manifest.json": "research",
    "valuation_direct_peer_comps.csv": "research",
}


@dataclass(frozen=True)
class CompanyWorkspace:
    root: Path
    ticker: str
    legacy_processed: Path

    @classmethod
    def from_config(cls, config: Mapping[str, Any], root: Path = ROOT) -> "CompanyWorkspace":
        storage = config.get("storage", {}) or {}
        base = Path(storage.get("root", root / "data" / "companies"))
        if not base.is_absolute():
            base = root / base
        return cls(base / str(config["ticker"]).lower(), str(config["ticker"]).upper(), root / "data" / "processed")

    def ensure(self) -> "CompanyWorkspace":
        for group in ("raw", "normalized", "derived", "model", "research"):
            (self.root / group).mkdir(parents=True, exist_ok=True)
        # Non-destructive compatibility migration: seed missing company-scoped
        # artifacts from the previous flat output directory. Legacy files are
        # deliberately retained so older notebooks continue to work.
        if self.ticker == "V" and self.legacy_processed.exists():
            for source in self.legacy_processed.iterdir():
                if source.is_file() and not source.name.startswith("."):
                    target = self.path(source.name)
                    if not target.exists():
                        shutil.copy2(source, target)
        return self

    def group_for(self, filename: str) -> str:
        if filename in _ARTIFACT_GROUPS:
            return _ARTIFACT_GROUPS[filename]
        if filename.endswith(("_investment_report.html", "_investment_report.pdf")):
            return "research"
        if filename.endswith(".xlsx") or filename.startswith(("forecast_", "income_statement_", "balance_sheet_", "cash_flow_statement_", "fcff_forecast_", "working_capital_schedule_", "ppe_schedule_", "debt_schedule_", "equity_schedule_", "capital_returns_schedule_", "checks_")):
            return "model"
        if any(token in filename for token in ("valuation", "comparable", "football_field", "central_", "statistics")):
            return "research"
        return "derived"

    def path(self, filename: str, *, for_read: bool = False) -> Path:
        target = self.root / self.group_for(filename) / filename
        if for_read and not target.exists():
            legacy = self.legacy_processed / filename
            if legacy.exists():
                return legacy
        return target

    def __truediv__(self, filename: str) -> Path:
        return self.path(str(filename))

    def write_manifest(self, artifacts: list[dict[str, Any]], sources: list[dict[str, Any]], retrieved_at: str) -> Path:
        manifest = {
            "schema_version": 1,
            "ticker": self.ticker,
            "retrieved_at": retrieved_at,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "artifacts": artifacts,
        }
        target = self.root / "research" / "lineage_manifest.json"
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return target

    def refresh_manifest_artifacts(self) -> Path | None:
        """Include artifacts produced by post-pipeline research stages."""
        target = self.root / "research" / "lineage_manifest.json"
        if not target.exists():
            return None
        manifest = json.loads(target.read_text(encoding="utf-8"))
        artifacts = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path != target:
                relative = str(path.relative_to(self.root))
                artifacts.append({
                    "path": relative,
                    "derived_from": [] if relative.startswith("raw/") else ["raw/sec_raw.csv"],
                })
        manifest["artifacts"] = artifacts
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return target


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
