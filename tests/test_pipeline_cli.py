from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src import pipeline


def test_cli_passes_selected_config_and_prints_manifest(monkeypatch, tmp_path, capsys):
    config = tmp_path / "selected.yaml"
    config.write_text("placeholder")
    workspace = tmp_path / "company"
    manifest = workspace / "research" / "lineage_manifest.json"
    seen = []

    def fake_run(path):
        seen.append(Path(path))
        return {"ticker": "EX", "company_name": "Example", "workspace": workspace,
                "manifest": manifest, "workbook": workspace / "model.xlsx",
                "report_html": workspace / "report.html", "report_pdf": workspace / "report.pdf"}

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(pipeline, "run_consolidated_valuation", lambda path: manifest)
    pipeline.main(["--config", str(config)])
    assert seen == [config]
    assert str(manifest) in capsys.readouterr().out
