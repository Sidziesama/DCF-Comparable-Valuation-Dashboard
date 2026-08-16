import json

import pandas as pd
import pytest

from src.research_product import (
    InvestmentMemo, ResearchClaim, apply_recommendation_policy,
    build_evidence_store, build_research_claims, validate_claim_links,
)


def _evidence():
    empty = pd.DataFrame()
    scenario = pd.DataFrame([
        {"scenario": "bear", "metric": "Implied share price", "value": 80, "unit": "USD/share"},
        {"scenario": "base", "metric": "Implied share price", "value": 120, "unit": "USD/share"},
    ])
    attribution = pd.DataFrame([{"bridge": "bear_to_base", "driver": "Growth", "contribution": 40, "status": "available"}])
    reverse = pd.DataFrame([{"implied_assumption": .09, "status": "converged", "converged": True}], index=["revenue_growth"])
    diagnostics = pd.DataFrame([{"question": "Candidate thesis breaker: base", "value": .08, "status": "available",
                                 "interpretive_rule": "Flag below 8%."}])
    assumptions = {"scenarios": {x: {"revenue_growth": [.08], "operating_margin": [.60], "tax_rate": .2}
                                  for x in ("bear", "base", "bull")}}
    return build_evidence_store("Example", "EX", operating_kpis=empty,
        scenario_decomposition=scenario, valuation_attribution=attribution,
        reverse_dcf=reverse, diagnostics=diagnostics, business_quality=empty,
        model_health=pd.DataFrame([{"metric": "Checks", "status": "PASS"}]), assumptions=assumptions)


def test_evidence_and_claim_ids_are_stable_and_links_valid():
    first, second = _evidence(), _evidence()
    assert [x.evidence_id for x in first] == [x.evidence_id for x in second]
    claims = build_research_claims(first)
    assert not validate_claim_links(claims, first)
    linked_ids = {eid for claim in claims for eid in claim.evidence_ids}
    assert linked_ids.issubset({x.evidence_id for x in first})
    json.dumps([x.__dict__ for x in first], sort_keys=True)


def test_unsupported_claim_is_rejected():
    claim = ResearchClaim("cl_bad", "thesis", "Unsupported", "No support", [])
    assert "unsupported material claim" in validate_claim_links([claim], [])[0]


@pytest.mark.parametrize("base,bear,expected", [(120, 90, "BUY"), (105, 80, "HOLD"), (85, 70, "SELL")])
def test_recommendation_thresholds(base, bear, expected):
    result = apply_recommendation_policy(100, base, bear, model_health_ok=True,
        evidence_coverage=1, high_quality_share=1, thesis_confidence=1)
    assert result["rating"] == expected


def test_no_rating_and_watch_gates_override_upside():
    no_rating = apply_recommendation_policy(100, 150, 100, model_health_ok=False,
        evidence_coverage=1, high_quality_share=1, thesis_confidence=1)
    watch = apply_recommendation_policy(100, 150, 100, model_health_ok=True,
        evidence_coverage=.6, high_quality_share=1, thesis_confidence=1)
    assert no_rating["rating"] == "NO-RATING"
    assert watch["rating"] == "WATCH"


def test_memo_schema_validation():
    memo = InvestmentMemo(1, "Example", "EX", {"rating": "HOLD"}, 100, 105, .05, "12-24 months",
        [], [], [], [], [], [], [], [], [], [], [], [], [])
    memo.validate()
    memo.recommendation["rating"] = "STRONG BUY"
    with pytest.raises(ValueError):
        memo.validate()
