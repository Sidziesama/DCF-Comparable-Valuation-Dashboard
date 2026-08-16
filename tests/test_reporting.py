import json

import pandas as pd

from src.reporting import (
    REQUIRED_MEMO_SECTIONS, build_report_health, build_thesis_monitoring,
    evaluate_monitor_condition, render_report_html, resolve_footnotes,
)


def _memo():
    memo = {name: [] for name in REQUIRED_MEMO_SECTIONS}
    memo.update({"company": "Example", "ticker": "EX", "recommendation": {"rating": "HOLD", "rationale": []},
                 "current_price": 100, "fair_value_base_case": 105, "expected_return": .05, "horizon": "12 months"})
    return memo


def _evidence(value=.12):
    return [{"evidence_id": "ev_1", "metric": "volume_growth", "value": value, "unit": "ratio",
             "period": "2025", "source": "Filing", "source_url": "https://example.com/filing", "filed_date": "2025-01-01"}]


def _claim(evidence_ids=None):
    return [{"claim_id": "cl_1", "claim_type": "thesis", "title": "Growth", "statement": "Volume grew.",
             "evidence_ids": ["ev_1"] if evidence_ids is None else evidence_ids,
             "basis": "rule_based_interpretation", "confidence": "high"}]


def test_footnote_resolution_and_html_source_appendix():
    mapping, appendix, missing = resolve_footnotes(_claim(), _evidence())
    assert mapping == {"cl_1": [1]}
    assert appendix[0]["source_url"].startswith("https://")
    assert not missing
    memo = _memo()
    memo["investment_thesis"] = _claim()
    html = render_report_html(memo, _claim(), _evidence(), [], build_report_health(memo, _claim(), _evidence()))
    assert 'href="#source-1"' in html and "Source link" in html


def test_broken_link_and_unsupported_claim_detection_are_nonfatal():
    broken = _claim(["missing"])
    unsupported = _claim([])
    health = build_report_health(_memo(), broken + unsupported, _evidence())
    statuses = health.set_index("check")["status"].to_dict()
    assert statuses["broken_evidence_ids"] == "FAIL"
    assert statuses["unsupported_material_claims"] == "FAIL"
    json.dumps(health.to_dict("records"))


def test_required_memo_schema_coverage():
    assert set(REQUIRED_MEMO_SECTIONS).issubset(_memo())
    health = build_report_health(_memo(), _claim(), _evidence())
    assert health.set_index("check").loc["memo_section_coverage", "status"] == "WARN"


def test_monitor_status_cases():
    assert evaluate_monitor_condition(observed_value=.12, threshold=.10, operator="gte")[0] == "SAFE"
    assert evaluate_monitor_condition(observed_value=.105, threshold=.10, operator="gte")[0] == "WATCH"
    assert evaluate_monitor_condition(observed_value=.09, threshold=.10, operator="gte")[0] == "BREACHED"
    assert evaluate_monitor_condition(observed_value=None, threshold=.10, operator="gte")[0] == "UNKNOWN"


def test_monitoring_is_deterministic_and_missing_data_is_unknown():
    conditions = [{"condition_id": "one", "metric": "volume_growth", "operator": "gte", "threshold": .10},
                  {"condition_id": "two", "metric": "missing", "operator": "gte", "threshold": .10}]
    first = build_thesis_monitoring(_claim(), _evidence(), configured_conditions=conditions, evaluated_at="2026-01-01T00:00:00Z")
    second = build_thesis_monitoring(_claim(), _evidence(), configured_conditions=conditions, evaluated_at="2026-01-01T00:00:00Z")
    assert first == second
    assert [x["status"] for x in first] == ["SAFE", "UNKNOWN"]


def test_duplicate_claim_detection():
    claims = _claim() + [{**_claim()[0], "claim_id": "cl_2"}]
    health = build_report_health(_memo(), claims, _evidence())
    assert health.set_index("check").loc["duplicate_claims", "status"] == "WARN"
