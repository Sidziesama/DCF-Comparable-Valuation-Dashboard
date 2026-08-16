"""Generate a factual, artifact-backed Visa/Microsoft portfolio summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from company_config import CompanyWorkspace, load_company_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (ROOT / "config" / "company.yaml", ROOT / "config" / "microsoft.yaml")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _company_snapshot(config_path: str | Path, root: Path = ROOT) -> dict[str, Any]:
    cfg = load_company_config(config_path)
    workspace = CompanyWorkspace.from_config(cfg, root)
    required = {
        "valuation": workspace.path("scenario_valuation.csv"),
        "recommendation": workspace.path("recommendation.json"),
        "model_health": workspace.path("model_health_summary.csv"),
        "research_health": workspace.path("research_health.csv"),
        "adapter_health": workspace.path("investment_intelligence_health.csv"),
        "kpis": workspace.path("operating_kpis.csv"),
        "monitoring": workspace.path("thesis_monitoring.csv"),
        "intelligence": workspace.path("investment_intelligence.csv"),
        "report_manifest": workspace.path("report_manifest.json"),
        "lineage_manifest": workspace.path("lineage_manifest.json"),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{cfg['ticker']} is missing generated artifacts: {', '.join(missing)}")

    valuation = pd.read_csv(required["valuation"]).set_index("scenario")
    recommendation = json.loads(required["recommendation"].read_text(encoding="utf-8"))
    report = json.loads(required["report_manifest"].read_text(encoding="utf-8"))
    model_health = pd.read_csv(required["model_health"])
    research_health = pd.read_csv(required["research_health"])
    adapter_health = pd.read_csv(required["adapter_health"])
    kpis = pd.read_csv(required["kpis"])
    monitoring = pd.read_csv(required["monitoring"])
    intelligence = pd.read_csv(required["intelligence"])
    adapter_row = adapter_health.loc[adapter_health["category"].eq("Operating KPI adapter")]

    scenarios = {}
    for name in ("bear", "base", "bull"):
        row = valuation.loc[name]
        scenarios[name] = {
            "implied_share_price": float(row["implied_share_price"]),
            "upside_downside": float(row["upside_downside"]),
        }
    report_paths = {
        kind: _relative(Path(path), root) if path else None
        for kind, path in {"html": report.get("html"), "pdf": report.get("pdf")}.items()
    }
    report_paths["excel"] = _relative(workspace.path(f"{cfg['ticker'].lower()}_valuation_model.xlsx"), root)

    return {
        "ticker": cfg["ticker"],
        "company_name": cfg["company_name"],
        "adapter": cfg.get("adapter", "generic"),
        "recommendation": recommendation["rating"],
        "recommendation_horizon": recommendation.get("thresholds", {}).get("horizon"),
        "current_price": float(valuation["current_price"].dropna().iloc[0]),
        "scenarios": scenarios,
        "model_health": "PASS" if model_health["status"].eq("PASS").all() else "REVIEW",
        "research_health": "PASS" if research_health["status"].isin(["PASS", "N/A"]).all() else "REVIEW",
        "adapter_health": adapter_row["status"].iloc[0] if not adapter_row.empty else "N/A",
        "kpi_coverage": {
            "available": int(kpis.get("status", pd.Series(dtype=str)).eq("available").sum()),
            "total": int(len(kpis)),
            "metrics": sorted(kpis.loc[kpis.get("status", "").eq("available"), "metric"].dropna().unique().tolist()),
        },
        "thesis_monitoring": monitoring["status"].value_counts().sort_index().to_dict(),
        "investment_intelligence": {
            "available_rows": int(intelligence.get("value", pd.Series(dtype=float)).notna().sum()),
            "total_rows": int(len(intelligence)),
            "categories": sorted(intelligence["category"].dropna().unique().tolist()),
        },
        "reports": report_paths,
        "artifact_manifest": _relative(required["lineage_manifest"], root),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Visa + Microsoft Portfolio Demo",
        "",
        "> Generated only from existing company-scoped pipeline artifacts. Values are point-in-time outputs, not investment advice.",
        "",
        "The same SEC normalization, linked three-statement, scenario, valuation, controls, research, reporting, and export core runs for both companies. Sector adapters supply disclosed KPI normalization and operating-driver bridges without changing the shared valuation engine.",
        "",
        "## Side-by-side case study",
        "",
        "| Capability | Visa | Microsoft |",
        "|---|---:|---:|",
    ]
    by_ticker = {company["ticker"]: company for company in payload["companies"]}
    visa, msft = by_ticker["V"], by_ticker["MSFT"]
    rows = [
        ("Business-model adapter", visa["adapter"], msft["adapter"]),
        ("Recommendation", visa["recommendation"], msft["recommendation"]),
        ("Base implied value", f"${visa['scenarios']['base']['implied_share_price']:,.2f}", f"${msft['scenarios']['base']['implied_share_price']:,.2f}"),
        ("Base upside / (downside)", f"{visa['scenarios']['base']['upside_downside']:+.1%}", f"{msft['scenarios']['base']['upside_downside']:+.1%}"),
        ("Model health", visa["model_health"], msft["model_health"]),
        ("Research health", visa["research_health"], msft["research_health"]),
        ("Available KPI rows", f"{visa['kpi_coverage']['available']} / {visa['kpi_coverage']['total']}", f"{msft['kpi_coverage']['available']} / {msft['kpi_coverage']['total']}"),
        ("Thesis monitor states", ", ".join(f"{k}: {v}" for k, v in visa["thesis_monitoring"].items()), ", ".join(f"{k}: {v}" for k, v in msft["thesis_monitoring"].items())),
    ]
    lines.extend(f"| {label} | {left} | {right} |" for label, left, right in rows)
    lines += ["", "## Generated deliverables", ""]
    for company in (visa, msft):
        lines += [
            f"### {company['company_name']} ({company['ticker']})",
            "",
            f"- Report: [{company['reports']['html']}](../../{company['reports']['html']})",
            f"- PDF: [{company['reports']['pdf']}](../../{company['reports']['pdf']})" if company["reports"]["pdf"] else "- PDF: unavailable",
            f"- Excel model: [{company['reports']['excel']}](../../{company['reports']['excel']})",
            f"- Artifact lineage: [{company['artifact_manifest']}](../../{company['artifact_manifest']})",
            f"- Intelligence categories: {', '.join(company['investment_intelligence']['categories'])}",
            "",
        ]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "python src/pipeline.py --config config/company.yaml",
        "python src/pipeline.py --config config/microsoft.yaml",
        "python src/portfolio_demo.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def generate_portfolio_demo(config_paths=DEFAULT_CONFIGS, output_dir: str | Path | None = None, root: Path = ROOT) -> dict[str, Path]:
    output = Path(output_dir) if output_dir else root / "docs" / "demo"
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "existing generated artifacts only",
        "companies": [_company_snapshot(path, root) for path in config_paths],
    }
    json_path = output / "portfolio_summary.json"
    markdown_path = output / "portfolio_summary.md"
    html_path = output / "portfolio_summary.html"
    markdown = _markdown(payload)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Portfolio Demo</title>"
        "<style>body{max-width:1000px;margin:40px auto;font:16px system-ui;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}"
        "th{background:#f4f4f4;text-align:left}pre{white-space:pre-wrap}</style></head>"
        f"<body><pre>{escape(markdown)}</pre></body></html>\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path, "html": html_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Visa/Microsoft artifact-backed portfolio demo.")
    parser.add_argument("--output-dir", default="docs/demo", help="Output directory (default: docs/demo).")
    args = parser.parse_args()
    for kind, path in generate_portfolio_demo(output_dir=args.output_dir).items():
        print(f"{kind.title()}: {path}")
