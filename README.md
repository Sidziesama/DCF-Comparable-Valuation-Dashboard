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
3. `src/model.py` — forecast, WACC, DCF, sensitivity
4. `src/pipeline.py` — end-to-end orchestration
5. `dashboard/app.py` — interactive dashboard
6. `config/company.yaml` — company assumptions and peers

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
streamlit run dashboard/app.py
```

## Important modeling note

The first version intentionally separates:
- raw historical data
- assumptions
- forecast logic
- valuation logic
- presentation

This makes the same engine reusable for MA, AXP, JPM, MSFT, BLK, or another public company by changing configuration and, where necessary, adding company-specific XBRL aliases.

## Next build stages

1. Improve SEC extraction for all three statements and quarterly LTM data.
2. Add historical ratios and margin analysis.
3. Add trading comparables with standardized LTM metrics.
4. Add a full Excel export.
5. Add scenario controls: Bear / Base / Bull.
6. Add a professional equity-research style dashboard.
7. Add automated data validation tests.
