"""Deterministic evidence, research-claim, memo, and recommendation layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _clean(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _id(prefix: str, *parts: Any) -> str:
    payload = json.dumps([_clean(x) for x in parts], sort_keys=True, default=str, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    company: str
    ticker: str
    claim_ids: list[str]
    evidence_type: str
    metric: str
    value: Any = None
    unit: str = ""
    period: str = ""
    source: str = ""
    source_url: str = ""
    lineage: str = ""
    retrieved_date: str = ""
    filed_date: str = ""
    quality: str = "calculated"
    status: str = "available"
    confidence: str = "medium"
    notes: str = ""


@dataclass(frozen=True)
class ResearchClaim:
    claim_id: str
    claim_type: str
    title: str
    statement: str
    evidence_ids: list[str] = field(default_factory=list)
    basis: str = "rule_based_interpretation"
    confidence: str = "medium"
    status: str = "supported"
    rule_id: str = ""
    notes: str = ""


@dataclass
class InvestmentMemo:
    schema_version: int
    company: str
    ticker: str
    recommendation: dict
    current_price: float
    fair_value_base_case: float
    expected_return: float
    horizon: str
    executive_summary_facts: list[dict]
    investment_thesis: list[dict]
    business_overview: list[dict]
    operating_drivers: list[dict]
    historical_performance: list[dict]
    market_implied_expectations: list[dict]
    forecasts_scenarios: list[dict]
    valuation: list[dict]
    catalysts: list[dict]
    risks: list[dict]
    thesis_breakers: list[dict]
    data_quality_limitations: list[str]
    sources_appendix_references: list[dict]

    def validate(self) -> None:
        if not self.ticker or not self.company:
            raise ValueError("Memo requires company and ticker.")
        if self.current_price <= 0 or self.fair_value_base_case <= 0:
            raise ValueError("Memo prices must be positive.")
        if self.recommendation.get("rating") not in {"BUY", "HOLD", "SELL", "WATCH", "NO-RATING"}:
            raise ValueError("Memo recommendation rating is invalid.")


def _frame_evidence(company, ticker, evidence_type, frame, *, source_default, lineage_default):
    items = []
    if frame is None or frame.empty:
        return items
    for index, row in frame.iterrows():
        metric = str(row.get("metric", row.get("question", row.get("driver", row.get("mode", row.get("index", index))))))
        scenario = str(row.get("scenario", row.get("bridge", "")))
        period = str(row.get("period", scenario)) if row.get("period", scenario) is not None else ""
        value = _clean(row.get("value", row.get("implied_assumption", row.get("contribution"))))
        status = str(row.get("status", "available"))
        item_id = _id("ev", ticker, evidence_type, metric, scenario, period, value, row.get("lineage", lineage_default))
        items.append(EvidenceItem(
            item_id, company, ticker, [], evidence_type, metric, value,
            str(row.get("unit", row.get("units", ""))), period,
            str(row.get("source", source_default)), str(row.get("source_url", "")), str(row.get("lineage", lineage_default)),
            str(row.get("retrieved_at", "")), str(row.get("filing_date", "")),
            str(row.get("quality", "calculated")), status,
            "high" if status.lower() in {"available", "converged", "pass"} else "low",
            str(row.get("interpretation", row.get("interpretive_rule", ""))),
        ))
    return items


def build_evidence_store(company, ticker, *, operating_kpis, scenario_decomposition,
                         valuation_attribution, reverse_dcf, diagnostics,
                         business_quality, model_health, assumptions, config_lineage="config/company.yaml"):
    specs = [
        ("operating_kpi", operating_kpis, "company filing", "research/operating_kpis.csv"),
        ("scenario_output", scenario_decomposition, "model output", "research/scenario_decomposition.csv"),
        ("valuation_attribution", valuation_attribution, "model output", "research/valuation_attribution.csv"),
        ("reverse_dcf", reverse_dcf.reset_index(), "reverse DCF", "derived/reverse_dcf.csv"),
        ("diagnostic", diagnostics, "deterministic rule", "research/investment_diagnostics.csv"),
        ("business_quality", business_quality, "standardized artifact", "research/business_quality_metrics.csv"),
        ("model_health", model_health, "model control", "model/model_health_summary.csv"),
    ]
    items = []
    for kind, frame, source, lineage in specs:
        items.extend(_frame_evidence(company, ticker, kind, frame, source_default=source, lineage_default=lineage))
    for scenario, config in sorted(assumptions.get("scenarios", {}).items()):
        for field_name in ("revenue_growth", "operating_margin", "tax_rate"):
            value = config.get(field_name)
            eid = _id("ev", ticker, "analyst_assumption", scenario, field_name, value)
            items.append(EvidenceItem(eid, company, ticker, [], "analyst_assumption", field_name,
                value, "ratio", scenario, "analyst configuration", "", config_lineage,
                quality="assumption", status="available", confidence="medium",
                notes="Explicit configured analyst assumption; not a sourced fact."))
    unique = {item.evidence_id: item for item in items}
    return [unique[key] for key in sorted(unique)]


def _find(evidence, evidence_type, metric_contains, *, scenario=""):
    matches = [e for e in evidence if e.evidence_type == evidence_type and metric_contains.lower() in e.metric.lower()
               and (not scenario or scenario.lower() in e.period.lower()) and e.status.lower() not in {"failed", "not_available", "n/a"}]
    return matches


def build_research_claims(evidence):
    claims = []
    def add(kind, title, statement, links, rule, confidence="medium"):
        cid = _id("cl", kind, title, statement, rule)
        links = sorted({x.evidence_id for x in links})
        claims.append(ResearchClaim(cid, kind, title, statement, links,
            "rule_based_interpretation", confidence, "supported" if links else "unsupported", rule))

    base = _find(evidence, "scenario_output", "Implied share price", scenario="base")
    bear = _find(evidence, "scenario_output", "Implied share price", scenario="bear")
    growth = _find(evidence, "reverse_dcf", "revenue_growth")
    kpi_growth = _find(evidence, "operating_kpi", "growth")
    sensitivity = sorted([e for e in evidence if e.evidence_type == "valuation_attribution" and e.value is not None], key=lambda x: abs(float(x.value)), reverse=True)
    breakers = [e for e in evidence if e.evidence_type == "diagnostic" and "thesis breaker" in e.metric.lower()]
    if base:
        add("valuation_conclusion", "Base-case valuation", "Base-case fair value is determined by the configured DCF assumptions.", base, "VAL_BASE_001", "high")
    if growth:
        add("market_expectation", "Market-implied growth", "The current price embeds a model-implied revenue-growth expectation that should be compared with scenario assumptions.", growth + base, "MKT_REV_001")
    if kpi_growth:
        add("thesis", "Operating network growth", "Reported and calculated network KPIs provide evidence of operating activity trends; coverage remains disclosure-dependent.", kpi_growth, "KPI_TREND_001")
    if sensitivity:
        add("risk", "Valuation sensitivity", f"{sensitivity[0].metric} is the largest absolute sequential scenario-bridge contribution.", sensitivity[:1], "VAL_SENS_001", "high")
        add("catalyst", "Scenario upside realization", "Improvement in the most valuation-sensitive modeled driver could move value toward the higher scenario, but this is a model condition rather than a dated external catalyst.", sensitivity[:1], "CAT_MODEL_001", "low")
    if bear:
        add("risk", "Bear-case downside", "The bear-case valuation defines modeled downside under the configured adverse operating assumptions.", bear, "VAL_BEAR_001", "high")
    for item in breakers:
        add("thesis_breaker", item.metric, item.notes or "Configured scenario boundary is a candidate monitoring threshold.", [item], "BREAK_THRESHOLD_001")
    quality = _find(evidence, "business_quality", "Operating margin")
    if quality:
        add("business_quality", "Operating profitability", "Historical and forecast operating-margin observations support assessment of business quality.", quality, "BQ_MARGIN_001")
    return sorted(claims, key=lambda x: x.claim_id)


def validate_claim_links(claims, evidence):
    ids = {e.evidence_id for e in evidence}
    errors = []
    for claim in claims:
        missing = sorted(set(claim.evidence_ids) - ids)
        if missing:
            errors.append(f"{claim.claim_id}: missing evidence {missing}")
        if not claim.evidence_ids and claim.basis not in {"analyst_judgment", "analyst_assumption"}:
            errors.append(f"{claim.claim_id}: unsupported material claim")
    return errors


DEFAULT_POLICY = {
    "buy_min_expected_return": 0.15, "sell_max_expected_return": -0.10,
    "hold_min_expected_return": -0.10, "max_bear_downside_for_buy": -0.25,
    "min_evidence_coverage": 0.75, "min_high_quality_share": 0.50,
    "min_thesis_confidence": 0.50, "watch_evidence_coverage": 0.50,
}


def apply_recommendation_policy(current_price, base_value, bear_value, *, model_health_ok,
                                evidence_coverage, high_quality_share, thesis_confidence,
                                diagnostic_ok=True, policy=None):
    p = {**DEFAULT_POLICY, **(policy or {})}
    expected = base_value / current_price - 1
    bear_return = bear_value / current_price - 1
    components = {
        "base_case_expected_return": expected, "bear_case_return": bear_return,
        "model_health_ok": bool(model_health_ok), "evidence_coverage": evidence_coverage,
        "high_quality_share": high_quality_share, "thesis_confidence": thesis_confidence,
        "diagnostic_ok": bool(diagnostic_ok),
    }
    if not model_health_ok or evidence_coverage < p["watch_evidence_coverage"] or high_quality_share < p["min_high_quality_share"]:
        rating = "NO-RATING"
    elif evidence_coverage < p["min_evidence_coverage"] or thesis_confidence < p["min_thesis_confidence"] or not diagnostic_ok:
        rating = "WATCH"
    elif expected >= p["buy_min_expected_return"] and bear_return >= p["max_bear_downside_for_buy"]:
        rating = "BUY"
    elif expected <= p["sell_max_expected_return"]:
        rating = "SELL"
    else:
        rating = "HOLD"
    return {"rating": rating, "policy_version": 1, "components": components, "thresholds": p,
            "rationale": [f"Base-case expected return: {expected:.1%}", f"Bear-case return: {bear_return:.1%}",
                          f"Evidence coverage: {evidence_coverage:.1%}", f"High-quality evidence: {high_quality_share:.1%}"]}


def build_research_product(cfg, workspace, *, current_price, scenario_values, operating_kpis,
                           scenario_decomposition, valuation_attribution, reverse_dcf,
                           diagnostics, business_quality, model_health):
    evidence = build_evidence_store(cfg["company_name"], cfg["ticker"], operating_kpis=operating_kpis,
        scenario_decomposition=scenario_decomposition, valuation_attribution=valuation_attribution,
        reverse_dcf=reverse_dcf, diagnostics=diagnostics, business_quality=business_quality,
        model_health=model_health, assumptions=cfg["assumptions"],
        config_lineage=f"config/{Path(cfg.get('_config_path', 'config/company.yaml')).name}")
    claims = build_research_claims(evidence)
    linked_claims = {}
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            linked_claims.setdefault(evidence_id, []).append(claim.claim_id)
    evidence = [replace(item, claim_ids=sorted(linked_claims.get(item.evidence_id, []))) for item in evidence]
    link_errors = validate_claim_links(claims, evidence)
    supported = [c for c in claims if c.status == "supported"]
    coverage = len(supported) / len(claims) if claims else 0.0
    available = [e for e in evidence if e.status.lower() not in {"failed", "not_available", "n/a"}]
    high_quality = [e for e in available if e.confidence == "high" or e.quality in {"reported", "calculated", "solver_output"}]
    high_share = len(high_quality) / len(available) if available else 0.0
    thesis = [c for c in claims if c.claim_type == "thesis"]
    thesis_confidence = sum({"low": .25, "medium": .5, "high": 1}[c.confidence] for c in thesis) / len(thesis) if thesis else 0.0
    health_ok = not model_health["status"].astype(str).str.upper().eq("FAIL").any()
    policy = cfg.get("recommendation_policy", {})
    recommendation = apply_recommendation_policy(float(current_price), float(scenario_values.loc["base", "implied_share_price"]),
        float(scenario_values.loc["bear", "implied_share_price"]), model_health_ok=health_ok,
        evidence_coverage=coverage, high_quality_share=high_share, thesis_confidence=thesis_confidence,
        diagnostic_ok=not link_errors, policy=policy)
    claim_dicts = [asdict(c) for c in claims]
    by_type = lambda kind: [x for x in claim_dicts if x["claim_type"] == kind]
    memo = InvestmentMemo(1, cfg["company_name"], cfg["ticker"], recommendation, float(current_price),
        float(scenario_values.loc["base", "implied_share_price"]), recommendation["components"]["base_case_expected_return"],
        str(policy.get("horizon", "12-24 months")), by_type("valuation_conclusion") + by_type("market_expectation"),
        by_type("thesis"), [], by_type("business_quality"), by_type("business_quality"), by_type("market_expectation"),
        [e for e in [asdict(x) for x in evidence] if e["evidence_type"] == "scenario_output"], by_type("valuation_conclusion"),
        by_type("catalyst"), by_type("risk"), by_type("thesis_breaker"),
        ["Claims are deterministic interpretations, not management commentary.", "Historical market-value coverage is limited where point-in-time data is unavailable.",
         "A model-conditioned catalyst is not a dated or sourced external event."],
        [{"evidence_id": e.evidence_id, "source": e.source, "lineage": e.lineage} for e in evidence])
    memo.validate()
    research = workspace.root / "research"
    evidence_dicts = [asdict(e) for e in evidence]
    outputs = {"evidence_store.json": evidence_dicts, "research_claims.json": claim_dicts,
               "investment_memo.json": asdict(memo), "recommendation.json": recommendation}
    for name, payload in outputs.items():
        (research / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(evidence_dicts).to_csv(research / "evidence_store.csv", index=False)
    pd.DataFrame(claim_dicts).to_csv(research / "research_claims.csv", index=False)
    health = pd.DataFrame([{"category": "Evidence coverage", "status": "PASS" if coverage >= policy.get("min_evidence_coverage", .75) else "WARN",
        "available": len(supported), "unavailable": len(claims)-len(supported), "coverage": coverage,
        "unsupported_claims": len(link_errors), "high_quality_share": high_share,
        "detail": "; ".join(link_errors) or "All material claims have valid evidence links."}])
    health.to_csv(workspace.root / "model" / "research_health.csv", index=False)
    return {"evidence": evidence, "claims": claims, "memo": memo, "recommendation": recommendation, "health": health}
