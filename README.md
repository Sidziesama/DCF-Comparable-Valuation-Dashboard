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
6. `src/pipeline.py` — end-to-end data and valuation orchestration
7. `dashboard/app.py` — cached-data interactive scenario dashboard
8. `config/company.yaml` — company assumptions and peers

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

Run the pipeline:

```bash
python src/pipeline.py
```

Run the dashboard:

```bash
python src/three_statement_model.py
streamlit run dashboard/app.py
```

The dashboard reads only `data/processed/` outputs. Changing scenarios, WACC,
terminal growth, or the terminal multiple does not refetch SEC or market data.

## Model checks

```bash
pytest -q
```

Every three-statement scenario writes two controls to `checks_<scenario>.csv`:

- assets minus liabilities and equity
- balance-sheet cash change minus cash-flow cash change

Both controls must be zero before the model is considered reconciled.

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
- linked three-statement forecast with reconciliation checks

## Next build stages

1. Add a formula-driven Excel export.
2. Expand automated tests for SEC taxonomy edge cases.
3. Add an investment-thesis and risk/catalyst research layer.
4. Package deployment configuration and CI.
