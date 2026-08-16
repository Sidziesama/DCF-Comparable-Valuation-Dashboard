"""Deterministic institutional report, health checks, and thesis monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REQUIRED_MEMO_SECTIONS = (
    "executive_summary_facts", "investment_thesis", "business_overview",
    "operating_drivers", "historical_performance", "market_implied_expectations",
    "forecasts_scenarios", "valuation", "catalysts", "risks", "thesis_breakers",
    "data_quality_limitations", "sources_appendix_references",
)
MATERIAL_CLAIM_TYPES = {"thesis", "risk", "catalyst", "thesis_breaker", "valuation_conclusion", "market_expectation"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _records(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict("records")
    return [dict(x) for x in value]


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_report_health(memo: Mapping[str, Any], claims: Iterable[Mapping[str, Any]],
                        evidence: Iterable[Mapping[str, Any]], *, min_evidence_coverage: float = .75) -> pd.DataFrame:
    """Return non-fatal report checks with one row per check/category."""
    claims, evidence = _records(claims), _records(evidence)
    evidence_ids = {x.get("evidence_id") for x in evidence if x.get("evidence_id")}
    rows: list[dict] = []

    missing_sections = [name for name in REQUIRED_MEMO_SECTIONS if name not in memo]
    empty_sections = [name for name in REQUIRED_MEMO_SECTIONS if name in memo and not memo.get(name)]
    rows.append({"check": "memo_section_coverage", "status": "FAIL" if missing_sections else ("WARN" if empty_sections else "PASS"),
                 "count": len(missing_sections) + len(empty_sections),
                 "detail": f"Missing: {missing_sections or 'none'}; empty: {empty_sections or 'none'}."})

    broken = []
    unsupported = []
    seen: dict[tuple, str] = {}
    duplicates = []
    for claim in claims:
        cid = claim.get("claim_id", "unknown")
        links = set(claim.get("evidence_ids") or [])
        absent = sorted(links - evidence_ids)
        if absent:
            broken.append(f"{cid}: {', '.join(absent)}")
        if claim.get("claim_type") in MATERIAL_CLAIM_TYPES and not links and claim.get("basis") not in {"analyst_judgment", "analyst_assumption"}:
            unsupported.append(cid)
        key = (str(claim.get("claim_type", "")).strip().lower(), str(claim.get("statement", "")).strip().lower())
        if key in seen:
            duplicates.append(f"{cid} duplicates {seen[key]}")
        else:
            seen[key] = cid
    rows.extend([
        {"check": "broken_evidence_ids", "status": "FAIL" if broken else "PASS", "count": len(broken), "detail": "; ".join(broken) or "All claim evidence IDs resolve."},
        {"check": "unsupported_material_claims", "status": "FAIL" if unsupported else "PASS", "count": len(unsupported), "detail": ", ".join(unsupported) or "All material claims have evidence or an explicit judgment/assumption basis."},
        {"check": "duplicate_claims", "status": "WARN" if duplicates else "PASS", "count": len(duplicates), "detail": "; ".join(duplicates) or "No duplicate claim statements."},
    ])
    material = [x for x in claims if x.get("claim_type") in MATERIAL_CLAIM_TYPES]
    supported = [x for x in material if x.get("evidence_ids") and not (set(x.get("evidence_ids") or []) - evidence_ids)]
    coverage = len(supported) / len(material) if material else 0.0
    rows.append({"check": "material_evidence_coverage", "status": "PASS" if coverage >= min_evidence_coverage else "WARN",
                 "count": len(material) - len(supported), "coverage": coverage,
                 "detail": f"{len(supported)} of {len(material)} material claims have resolved evidence."})
    unresolved_sources = [x.get("evidence_id", "unknown") for x in evidence
                          if not (x.get("source") or x.get("lineage"))]
    rows.append({"check": "source_resolution", "status": "WARN" if unresolved_sources else "PASS",
                 "count": len(unresolved_sources), "detail": ", ".join(unresolved_sources) or "Every evidence item has a source or lineage reference."})
    return pd.DataFrame(rows)


def _infer_operator(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("below", "less than", "minimum", "floor")):
        return "gte"
    if any(token in lowered for token in ("above", "greater than", "maximum", "ceiling")):
        return "lte"
    return "unknown"


def evaluate_monitor_condition(*, observed_value: Any, threshold: Any, operator: str,
                               watch_band: float = .10) -> tuple[str, float | None]:
    """Evaluate one threshold. Distance is signed safety margin / abs(threshold)."""
    try:
        observed, limit = float(observed_value), float(threshold)
    except (TypeError, ValueError):
        return "UNKNOWN", None
    if not np.isfinite(observed) or not np.isfinite(limit) or operator not in {"gte", "lte"}:
        return "UNKNOWN", None
    denominator = abs(limit) if limit else 1.0
    distance = (observed - limit) / denominator if operator == "gte" else (limit - observed) / denominator
    if distance < 0:
        return "BREACHED", distance
    if distance <= watch_band:
        return "WATCH", distance
    return "SAFE", distance


def build_thesis_monitoring(claims: Iterable[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]],
                            *, configured_conditions: Iterable[Mapping[str, Any]] | None = None,
                            evaluated_at: str | None = None) -> list[dict]:
    """Build deterministic monitoring from explicit config and thesis-breaker evidence."""
    claims, evidence = _records(claims), _records(evidence)
    evidence_by_id = {x.get("evidence_id"): x for x in evidence}
    conditions = [dict(x) for x in (configured_conditions or [])]
    if not conditions:
        for claim in claims:
            if claim.get("claim_type") != "thesis_breaker":
                continue
            linked = [evidence_by_id[x] for x in claim.get("evidence_ids", []) if x in evidence_by_id]
            source = linked[0] if linked else {}
            conditions.append({
                "condition_id": claim.get("claim_id"), "title": claim.get("title"),
                "metric": source.get("metric"), "threshold": source.get("value"),
                "operator": _infer_operator(str(source.get("notes", ""))),
                "evidence_ids": claim.get("evidence_ids", []),
            })
    generated_at = evaluated_at or utc_now()
    output = []
    for index, condition in enumerate(conditions):
        metric = str(condition.get("metric", ""))
        candidates = [x for x in evidence if str(x.get("metric", "")).lower() == metric.lower()
                      and x.get("value") is not None]
        if condition.get("observed_evidence_ids"):
            allowed = set(condition["observed_evidence_ids"])
            candidates = [x for x in candidates if x.get("evidence_id") in allowed]
        candidates.sort(key=lambda x: str(x.get("period", "")))
        latest = candidates[-1] if candidates else {}
        status, distance = evaluate_monitor_condition(
            observed_value=condition.get("observed_value", latest.get("value")),
            threshold=condition.get("threshold"), operator=str(condition.get("operator", "unknown")),
            watch_band=float(condition.get("watch_band", .10)),
        )
        output.append({
            "condition_id": condition.get("condition_id", f"monitor_{index + 1:03d}"),
            "title": condition.get("title", metric or "Unspecified thesis condition"),
            "metric": metric, "operator": condition.get("operator", "unknown"),
            "threshold": _clean(condition.get("threshold")),
            "latest_observed_value": _clean(condition.get("observed_value", latest.get("value"))),
            "status": status, "distance_to_threshold": _clean(distance),
            "latest_period": str(condition.get("latest_period", latest.get("period", ""))),
            "unit": condition.get("unit", latest.get("unit", "")),
            "evidence_ids": sorted(set(condition.get("evidence_ids", [])) | ({latest.get("evidence_id")} if latest.get("evidence_id") else set())),
            "evaluated_at": generated_at,
            "notes": condition.get("notes", "No threshold is inferred when the configured rule is ambiguous."),
        })
    return output


def resolve_footnotes(claims: Iterable[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]]) -> tuple[dict[str, list[int]], list[dict], list[str]]:
    evidence_by_id = {x.get("evidence_id"): x for x in _records(evidence)}
    referenced = sorted({eid for claim in _records(claims) for eid in (claim.get("evidence_ids") or [])})
    numbers, appendix, missing = {}, [], []
    for eid in referenced:
        item = evidence_by_id.get(eid)
        if not item:
            missing.append(eid)
            continue
        number = len(appendix) + 1
        numbers[eid] = number
        appendix.append({"number": number, **item})
    mapping = {claim.get("claim_id", ""): [numbers[x] for x in claim.get("evidence_ids", []) if x in numbers]
               for claim in _records(claims)}
    return mapping, appendix, missing


def _fmt_value(value: Any, unit: str = "") -> str:
    if value is None:
        return "Not available"
    if unit == "ratio":
        return f"{float(value):.1%}"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}" + (f" {unit}" if unit else "")
    return escape(str(value))


def _claim_html(item: Mapping[str, Any], footnotes: Mapping[str, list[int]]) -> str:
    refs = "".join(f'<sup><a href="#source-{n}">{n}</a></sup>' for n in footnotes.get(item.get("claim_id", ""), []))
    basis = escape(str(item.get("basis", "rule_based_interpretation")).replace("_", " ").title())
    confidence = escape(str(item.get("confidence", "unknown")).title())
    return f'<article class="claim"><div class="claim-meta">{basis} | {confidence} confidence</div><h3>{escape(str(item.get("title", "Untitled")))}</h3><p>{escape(str(item.get("statement", "")))} {refs}</p></article>'


def render_report_html(memo: Mapping[str, Any], claims: Iterable[Mapping[str, Any]], evidence: Iterable[Mapping[str, Any]],
                       monitoring: Iterable[Mapping[str, Any]], health: pd.DataFrame,
                       artifacts: Mapping[str, Any] | None = None) -> str:
    claims = _records(claims)
    footnotes, appendix, missing = resolve_footnotes(claims, evidence)
    by_type = lambda kind: [x for x in claims if x.get("claim_type") == kind]
    artifacts = artifacts or {}
    recommendation = memo.get("recommendation", {})
    rating = recommendation.get("rating", "NO-RATING")
    rationale = recommendation.get("rationale", [])

    def section(title: str, items: Iterable[Mapping[str, Any]], empty: str) -> str:
        content = "".join(_claim_html(x, footnotes) for x in items)
        return f"<section><h2>{escape(title)}</h2>{content or f'<p class=limitation>{escape(empty)}</p>'}</section>"

    scenarios = _records(memo.get("forecasts_scenarios"))
    scenario_focus = ("implied share price", "upside / downside", "revenue cagr", "operating margin - terminal year",
                      "fcff - terminal year", "wacc", "terminal growth")
    focused_scenarios = [x for x in scenarios if str(x.get("metric", "")).lower() in scenario_focus]
    scenarios = focused_scenarios or scenarios[:21]
    scenario_rows = "".join(f"<tr><td>{escape(str(x.get('period', ''))).title()}</td><td>{escape(str(x.get('metric', '')))}</td><td>{_fmt_value(x.get('value'), str(x.get('unit', '')))}</td></tr>" for x in scenarios)
    monitoring_rows = "".join(f"<tr><td>{escape(str(x.get('title')))}</td><td><span class='status {str(x.get('status')).lower()}'>{escape(str(x.get('status')))}</span></td><td>{_fmt_value(x.get('latest_observed_value'), str(x.get('unit', '')))}</td><td>{_fmt_value(x.get('threshold'), str(x.get('unit', '')))}</td><td>{escape(str(x.get('latest_period') or 'N/A'))}</td></tr>" for x in monitoring)
    health_rows = "".join(f"<tr><td>{escape(str(x['check']))}</td><td><span class='status {str(x['status']).lower()}'>{x['status']}</span></td><td>{escape(str(x['detail']))}</td></tr>" for x in health.to_dict("records"))
    sources = "".join(
        f"<li id='source-{x['number']}'><strong>[{x['number']}] {escape(str(x.get('source') or x.get('lineage') or 'Unresolved source'))}</strong>"
        f" - {escape(str(x.get('metric', '')))}; period {escape(str(x.get('period') or 'N/A'))}; value {_fmt_value(x.get('value'), str(x.get('unit', '')))}"
        f"; filing date {escape(str(x.get('filed_date') or 'N/A'))}. "
        + (f"<a href='{escape(str(x.get('source_url')))}'>Source link</a>." if x.get("source_url") else f"Lineage: {escape(str(x.get('lineage') or 'N/A'))}.") + "</li>"
        for x in appendix
    )
    limitations = "".join(f"<li>{escape(str(x))}</li>" for x in memo.get("data_quality_limitations", []))
    valuation_note = escape(str(artifacts.get("valuation_note", "Valuation detail is rendered from the existing DCF, exit-multiple, comps, reverse-DCF, and football-field artifacts when available.")))
    valuation_blocks = []
    for title, records in artifacts.get("valuation_tables", {}).items():
        rows = _records(records)
        if not rows:
            continue
        columns = list(rows[0])[:7]
        head = "".join(f"<th>{escape(str(x).replace('_',' ').title())}</th>" for x in columns)
        body = "".join("<tr>" + "".join(f"<td>{_fmt_value(row.get(column))}</td>" for column in columns) + "</tr>" for row in rows)
        valuation_blocks.append(f"<h3>{escape(str(title))}</h3><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
    generated = escape(str(artifacts.get("generated_at", utc_now())))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{escape(str(memo.get('company')))} Investment Report</title>
<style>
@page {{ size: Letter; margin: .65in; @bottom-right {{ content: counter(page); }} }}
:root{{--navy:#102a43;--blue:#2563a6;--light:#eef4f8;--ink:#243b53;--muted:#627d98;--green:#13795b;--amber:#946200;--red:#b42318}}
*{{box-sizing:border-box}} body{{font:10.5pt Arial,sans-serif;color:var(--ink);margin:0;line-height:1.45}} a{{color:var(--blue)}}
.cover{{background:linear-gradient(135deg,var(--navy),#1f4d78);color:white;padding:40px;border-radius:8px;margin-bottom:24px}} .kicker{{letter-spacing:.12em;text-transform:uppercase;font-size:9pt;opacity:.8}} h1{{font-size:27pt;margin:10px 0 4px}} .subtitle{{font-size:13pt;opacity:.85}}
.metric-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:26px}} .metric{{background:#ffffff18;padding:10px;border:1px solid #ffffff30;border-radius:5px}} .metric b{{display:block;font-size:15pt}} .metric span{{font-size:8pt;text-transform:uppercase}}
h2{{color:var(--navy);font-size:16pt;border-bottom:1px solid #bcccdc;padding-bottom:5px;margin-top:26px}} h3{{margin:2px 0 4px;font-size:11.5pt;color:var(--navy)}} .lead{{font-size:12pt}} .claim{{border-left:3px solid var(--blue);padding:8px 12px;margin:10px 0;background:#f8fafc;break-inside:avoid}} .claim p{{margin:0}} .claim-meta{{font-size:8pt;text-transform:uppercase;color:var(--muted);letter-spacing:.05em}}
.rationale{{background:var(--light);padding:12px 18px;border-radius:5px}} table{{border-collapse:collapse;width:100%;margin:10px 0 16px;font-size:9pt;break-inside:auto}} th{{background:var(--navy);color:white;text-align:left}} th,td{{padding:7px;border:1px solid #d9e2ec;vertical-align:top}} tr{{break-inside:avoid}} .status{{font-weight:bold}} .safe,.pass{{color:var(--green)}} .watch,.warn{{color:var(--amber)}} .breached,.fail{{color:var(--red)}} .unknown{{color:var(--muted)}} .limitation{{color:var(--muted);font-style:italic}} .sources li{{margin-bottom:9px;break-inside:avoid}} footer{{margin-top:30px;border-top:1px solid #bcccdc;padding-top:8px;color:var(--muted);font-size:8pt}}
@media(max-width:800px){{.metric-grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<div class='cover'><div class='kicker'>Institutional research | governed output</div><h1>{escape(str(memo.get('company')))} ({escape(str(memo.get('ticker')))} )</h1><div class='subtitle'>Manager-ready investment report | Generated {generated}</div>
<div class='metric-grid'><div class='metric'><span>Recommendation</span><b>{escape(str(rating))}</b></div><div class='metric'><span>Current Price</span><b>${float(memo.get('current_price',0)):,.2f}</b></div><div class='metric'><span>Base Fair Value</span><b>${float(memo.get('fair_value_base_case',0)):,.2f}</b></div><div class='metric'><span>Expected Return</span><b>{float(memo.get('expected_return',0)):.1%}</b></div><div class='metric'><span>Horizon</span><b>{escape(str(memo.get('horizon','N/A')))}</b></div></div></div>
<section><h2>Executive Summary</h2><p class='lead'>The recommendation is produced by a deterministic policy. The report renders structured claims and existing model outputs; it does not generate independent analysis.</p><div class='rationale'><strong>Policy rationale</strong><ul>{''.join(f'<li>{escape(str(x))}</li>' for x in rationale)}</ul></div>{''.join(_claim_html(x,footnotes) for x in memo.get('executive_summary_facts',[]))}</section>
{section('Investment Thesis', memo.get('investment_thesis',[]), 'No supported thesis claim is currently available.')}
{section('Business Overview', memo.get('business_overview',[]), 'No structured business-overview claim is currently available; the report does not invent one.')}
{section('Operating Drivers and KPIs', memo.get('operating_drivers',[]), 'No structured operating-driver claim is currently available.')}
{section('Historical Financial Performance', memo.get('historical_performance',[]), 'No structured historical-performance claim is currently available.')}
{section('Market-Implied Expectations', memo.get('market_implied_expectations',[]), 'Market-implied expectations are unavailable.')}
<section><h2>Bear / Base / Bull Scenarios</h2><table><thead><tr><th>Scenario</th><th>Metric</th><th>Value</th></tr></thead><tbody>{scenario_rows or '<tr><td colspan=3>Scenario outputs unavailable.</td></tr>'}</tbody></table></section>
<section><h2>Valuation</h2>{''.join(_claim_html(x,footnotes) for x in memo.get('valuation',[]))}{''.join(valuation_blocks)}<p class='limitation'>{valuation_note}</p></section>
{section('Catalysts', by_type('catalyst'), 'No sourced event calendar or dated catalyst is available. Model-conditioned upside is not an external catalyst.')}
{section('Risks', by_type('risk'), 'No structured risk claim is currently available.')}
{section('Thesis Breakers', by_type('thesis_breaker'), 'No thesis breaker is currently configured.')}
<section><h2>Thesis Monitoring</h2><table><thead><tr><th>Condition</th><th>Status</th><th>Latest</th><th>Threshold</th><th>Period</th></tr></thead><tbody>{monitoring_rows or '<tr><td colspan=5>No monitored condition is configured.</td></tr>'}</tbody></table></section>
<section><h2>Data Quality and Limitations</h2><ul>{limitations or '<li>No limitations were supplied.</li>'}</ul><h3>Report health</h3><table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>{health_rows}</tbody></table>{f'<p class=limitation>Unresolved evidence IDs: {escape(str(missing))}</p>' if missing else ''}</section>
<section><h2>Evidence and Source Appendix</h2><ol class='sources'>{sources or '<li>No cited evidence items.</li>'}</ol></section>
<footer>Classification labels distinguish sourced/calculated evidence, rule-based interpretations, analyst assumptions, and limitations. This artifact is not investment advice.</footer></body></html>"""


def _html_to_pdf(html_path: Path, pdf_path: Path) -> str | None:
    """Use an installed clean HTML-to-PDF backend; HTML remains canonical."""
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(pdf_path))
        return None
    except Exception as primary_exc:
        try:
            from bs4 import BeautifulSoup
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

            soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                                      fontSize=24, leading=28, textColor=colors.HexColor("#102a43"), alignment=TA_CENTER, spaceAfter=16))
            styles.add(ParagraphStyle(name="ReportH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                                      fontSize=15, leading=18, textColor=colors.HexColor("#102a43"), spaceBefore=14, spaceAfter=7))
            styles.add(ParagraphStyle(name="ReportH3", parent=styles["Heading3"], fontName="Helvetica-Bold",
                                      fontSize=11, leading=14, textColor=colors.HexColor("#1f4d78"), spaceBefore=8, spaceAfter=3))
            styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontName="Helvetica",
                                      fontSize=9.2, leading=13, textColor=colors.HexColor("#243b53"), spaceAfter=6))
            styles.add(ParagraphStyle(name="ReportMeta", parent=styles["BodyText"], fontName="Helvetica",
                                      fontSize=7.5, leading=10, textColor=colors.HexColor("#627d98"), spaceAfter=2))
            story = [Paragraph(soup.title.get_text(" ", strip=True), styles["ReportTitle"]), Spacer(1, .1 * inch)]
            metric_cells = []
            for metric in soup.select(".metric"):
                label = metric.find("span")
                value = metric.find("b")
                metric_cells.append(Paragraph(
                    f"<font size='7' color='#627d98'>{escape(label.get_text(strip=True) if label else '')}</font><br/>"
                    f"<b>{escape(value.get_text(strip=True) if value else '')}</b>", styles["ReportBody"]))
            if metric_cells:
                metrics = Table([metric_cells], colWidths=[(letter[0] - 1.3 * inch) / len(metric_cells)] * len(metric_cells))
                metrics.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4f8")),
                                             ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#bcccdc")),
                                             ("INNERGRID", (0, 0), (-1, -1), .35, colors.HexColor("#bcccdc")),
                                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                             ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
                story.extend([metrics, Spacer(1, .12 * inch)])
            for node in soup.body.find_all(["h2", "h3", "p", "li", "table"], recursive=True):
                if node.find_parent("table") and node.name != "table":
                    continue
                if node.name == "h2":
                    story.append(Paragraph(escape(node.get_text(" ", strip=True)), styles["ReportH2"]))
                elif node.name == "h3":
                    story.append(Paragraph(escape(node.get_text(" ", strip=True)), styles["ReportH3"]))
                elif node.name in {"p", "li"}:
                    prefix = "- " if node.name == "li" else ""
                    pdf_text = node.get_text(" ", strip=True).replace("three-statement", "three statement")
                    paragraph = Paragraph(prefix + escape(pdf_text), styles["ReportBody"])
                    if node.name == "li":
                        item = Table([[paragraph]], colWidths=[letter[0] - 1.3 * inch], hAlign="LEFT")
                        item.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                                                  ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                                  ("TOPPADDING", (0, 0), (-1, -1), 0),
                                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
                        story.append(item)
                    else:
                        story.append(paragraph)
                elif node.name == "table":
                    rows = [[Paragraph(escape(cell.get_text(" ", strip=True)), styles["ReportMeta"])
                             for cell in row.find_all(["th", "td"], recursive=False)]
                            for row in node.find_all("tr")]
                    if rows and all(len(row) == len(rows[0]) for row in rows):
                        table = Table(rows, repeatRows=1, hAlign="LEFT")
                        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102a43")),
                                                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                                   ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#bcccdc")),
                                                   ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                   ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                                   ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                        story.extend([table, Spacer(1, .08 * inch)])
            def footer(canvas, doc):
                canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#627d98"))
                canvas.drawRightString(letter[0] - .65 * inch, .38 * inch, f"Page {doc.page}"); canvas.restoreState()
            SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=.65 * inch, rightMargin=.65 * inch,
                              topMargin=.55 * inch, bottomMargin=.55 * inch, title=soup.title.get_text()).build(story, onFirstPage=footer, onLaterPages=footer)
            return None
        except Exception as fallback_exc:
            return (f"PDF export unavailable: WeasyPrint {type(primary_exc).__name__}: {primary_exc}; "
                    f"ReportLab {type(fallback_exc).__name__}: {fallback_exc}")


def generate_report_artifacts(workspace, *, memo: Mapping[str, Any], claims: Iterable[Mapping[str, Any]],
                              evidence: Iterable[Mapping[str, Any]], recommendation: Mapping[str, Any],
                              config: Mapping[str, Any] | None = None, artifacts: Mapping[str, Any] | None = None) -> dict:
    research = workspace.root / "research"
    research.mkdir(parents=True, exist_ok=True)
    conditions = (config or {}).get("thesis_monitoring", {}).get("conditions", [])
    monitoring = build_thesis_monitoring(claims, evidence, configured_conditions=conditions)
    health = build_report_health(memo, claims, evidence,
        min_evidence_coverage=float((config or {}).get("recommendation_policy", {}).get("min_evidence_coverage", .75)))
    html = render_report_html(memo, claims, evidence, monitoring, health, artifacts=artifacts)
    stem = f"{str(memo.get('ticker','company')).lower()}_investment_report"
    html_path, pdf_path = research / f"{stem}.html", research / f"{stem}.pdf"
    html_path.write_text(html, encoding="utf-8")
    pdf_error = _html_to_pdf(html_path, pdf_path)
    (research / "thesis_monitoring.json").write_text(json.dumps(monitoring, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(monitoring).to_csv(research / "thesis_monitoring.csv", index=False)
    if pdf_error:
        health = pd.concat([health, pd.DataFrame([{"check": "pdf_export", "status": "WARN", "count": 1, "detail": pdf_error}])], ignore_index=True)
    else:
        health = pd.concat([health, pd.DataFrame([{"check": "pdf_export", "status": "PASS", "count": 0, "detail": "PDF generated successfully."}])], ignore_index=True)
    health.to_csv(research / "report_health.csv", index=False)
    manifest = {"schema_version": 1, "generated_at": utc_now(), "html": str(html_path),
                "pdf": str(pdf_path) if pdf_path.exists() else None, "pdf_error": pdf_error,
                "recommendation": recommendation.get("rating"), "monitor_count": len(monitoring)}
    (research / "report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"html": html_path, "pdf": pdf_path if pdf_path.exists() else None, "health": health,
            "monitoring": monitoring, "manifest": manifest}
