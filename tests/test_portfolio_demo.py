import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src.portfolio_demo import generate_portfolio_demo


def _seed(root: Path, ticker: str, company: str, adapter: str):
    config = root / "config" / f"{ticker.lower()}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"ticker: {ticker}\ncompany_name: {company}\ncik: '1234'\nadapter: {adapter}\n"
        "storage: {root: data/companies}\nassumptions:\n  scenarios:\n"
        "    bear: {revenue_growth: [0.01], operating_margin: [0.1], tax_rate: 0.2}\n"
        "    base: {revenue_growth: [0.02], operating_margin: [0.2], tax_rate: 0.2}\n"
        "    bull: {revenue_growth: [0.03], operating_margin: [0.3], tax_rate: 0.2}\n"
        "  wacc: {risk_free_rate: 0.04, equity_risk_premium: 0.05, pre_tax_cost_of_debt: 0.03}\n"
        "  terminal_growth: 0.02\n",
        encoding="utf-8",
    )
    base = root / "data" / "companies" / ticker.lower()
    for folder in ("derived", "model", "research"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"scenario": name, "implied_share_price": value, "upside_downside": change, "current_price": 100}
        for name, value, change in (("bear", 80, -0.2), ("base", 110, 0.1), ("bull", 130, 0.3))
    ]).to_csv(base / "derived" / "scenario_valuation.csv", index=False)
    pd.DataFrame([{"status": "PASS"}]).to_csv(base / "model" / "model_health_summary.csv", index=False)
    pd.DataFrame([{"status": "PASS"}]).to_csv(base / "model" / "research_health.csv", index=False)
    pd.DataFrame([{"category": "Operating KPI adapter", "status": "PASS"}]).to_csv(base / "model" / "investment_intelligence_health.csv", index=False)
    pd.DataFrame([{"metric": "growth", "status": "available"}]).to_csv(base / "research" / "operating_kpis.csv", index=False)
    pd.DataFrame([{"status": "SAFE"}]).to_csv(base / "research" / "thesis_monitoring.csv", index=False)
    pd.DataFrame([{"category": "expectations", "value": 0.1}]).to_csv(base / "research" / "investment_intelligence.csv", index=False)
    (base / "research" / "recommendation.json").write_text(json.dumps({"rating": "HOLD", "thresholds": {"horizon": "12-24 months"}}))
    (base / "research" / "report_manifest.json").write_text(json.dumps({"html": str(base / "research" / "report.html"), "pdf": None}))
    (base / "research" / "lineage_manifest.json").write_text("{}")
    return config


def test_demo_summary_is_company_scoped_and_artifact_backed(tmp_path):
    visa = _seed(tmp_path, "V", "Visa", "payment_network")
    microsoft = _seed(tmp_path, "MSFT", "Microsoft", "software")
    outputs = generate_portfolio_demo((visa, microsoft), tmp_path / "demo", tmp_path)
    payload = json.loads(outputs["json"].read_text())
    assert [item["ticker"] for item in payload["companies"]] == ["V", "MSFT"]
    assert payload["companies"][1]["adapter"] == "software"
    assert payload["companies"][0]["scenarios"]["base"]["implied_share_price"] == 110
    assert all(path.exists() for path in outputs.values())
