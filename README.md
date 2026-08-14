# Visa DCF + Comparable Valuation Dashboard

Reusable Python valuation architecture for public companies.

## Stack
- SEC XBRL Company Facts API — historical financial statements
- yfinance — market data
- pandas / numpy — modeling
- Plotly — visualization
- Streamlit — dashboard
- YAML — company-specific configuration

## Current company
Visa Inc. (NYSE: V)

## Architecture
1. `src/sec_data.py` — SEC/XBRL ingestion
2. `src/market_data.py` — market data ingestion
3. `src/forecast_model.py` — reusable Bear/Base/Bull FCFF forecasts
4. `src/valuation.py` — Gordon-growth and exit-multiple DCF engines
5. `src/three_statement_model.py` — linked forecast statements and checks
6. `src/model_quality.py` — institutional controls and historical/forecast analytics
7. `src/pipeline.py` — end-to-end data and valuation orchestration
8. `dashboard/app.py` — cached-data interactive scenario dashboard
9. `config/company.yaml` — company assumptions and peers
10. `src/excel_export.py` — reusable formula-driven professional workbook export

## Setup on macOS / VS Code

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```text
SEC_USER_AGENT=Your Name your.email@example.com
```

Run the pipeline (this now builds the linked statements before valuation):

```bash
python src/pipeline.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard reads only `data/processed/` outputs. Changing scenarios, WACC,
terminal growth, or the terminal multiple does not refetch SEC or market data.

## Model checks

```bash
pytest -q
```

Every three-statement scenario writes explicit controls to `checks_<scenario>.csv`:

- assets minus liabilities and equity
- cash, PP&E, debt, and equity roll-forwards
- FCFF formula and operating-to-linked FCFF variance

All reconciliation controls must be zero before the pipeline proceeds to valuation.
The operating-to-linked FCFF variance is an informational bridge: the linked
model replaces the standalone working-capital estimate with forecast A/R and A/P.

The institutional quality layer also writes `model_checks_detail.csv` and
`model_health_summary.csv`. Each available control reports actual, expected,
variance, tolerance, and an explicit PASS/FAIL status. Checks cover statement
reconciliation, cash, debt, total equity/retained-earnings consistency, FCFF,
scenario ordering, the WACC/terminal-growth spread, and terminal-value
concentration. The pipeline stops before completing when an available check fails.

`analytics_trends.csv` and `forecast_reasonableness.csv` contain reusable
historical/forecast analytics and range comparisons. ROIC and working-capital
efficiency are reported only when the required historical or balance-sheet inputs
are available; unavailable observations remain blank rather than being estimated.

## Important modeling note

The first version intentionally separates:
- raw historical data
- assumptions
- forecast logic
- valuation logic
- presentation

This makes the same engine reusable for MA, AXP, JPM, MSFT, BLK, or another public company by changing configuration and, where necessary, adding company-specific XBRL aliases.

## Completed capabilities

- SEC/XBRL ingestion with filing fallback and freshness validation
- annual and LTM historical model
- Bear/Base/Bull operating forecasts
- beta, WACC, Gordon-growth DCF, exit-multiple DCF, and sensitivities
- comparable-company valuation and football field
- interactive cached-data dashboard
- linked Bear/Base/Bull income statement, balance sheet, and cash-flow forecasts
- statement-derived FCFF feeding the existing DCF and sensitivity engines
- visible model-integrity checks in both pipeline outputs and dashboard
- Model Health & Analytics dashboard with check-level diagnostics and trend views
- bounded Reverse DCF solvers for market-implied revenue growth, operating margin,
  and terminal growth, including Bear/Base/Bull comparison outputs
- professional Excel export with linked valuation formulas, formatting, checks,
  sensitivities, and dashboard download control

## Next build stages

1. Expand automated tests for SEC taxonomy edge cases.
2. Add an investment-thesis and risk/catalyst research layer.
3. Package deployment configuration and CI.
