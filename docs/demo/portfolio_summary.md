# Visa + Microsoft Portfolio Demo

> Generated only from existing company-scoped pipeline artifacts. Values are point-in-time outputs, not investment advice.

The same SEC normalization, linked three-statement, scenario, valuation, controls, research, reporting, and export core runs for both companies. Sector adapters supply disclosed KPI normalization and operating-driver bridges without changing the shared valuation engine.

## Side-by-side case study

| Capability | Visa | Microsoft |
|---|---:|---:|
| Business-model adapter | payment_network | software |
| Recommendation | HOLD | HOLD |
| Base implied value | $370.63 | $470.89 |
| Base upside / (downside) | +1.8% | -4.9% |
| Model health | PASS | PASS |
| Research health | PASS | PASS |
| Available KPI rows | 14 / 14 | 24 / 24 |
| Thesis monitor states | SAFE: 1, WATCH: 1 | SAFE: 3 |

## Generated deliverables

### Visa Inc. (V)

- Report: [data/companies/v/research/v_investment_report.html](../../data/companies/v/research/v_investment_report.html)
- PDF: [data/companies/v/research/v_investment_report.pdf](../../data/companies/v/research/v_investment_report.pdf)
- Excel model: [data/companies/v/model/v_valuation_model.xlsx](../../data/companies/v/model/v_valuation_model.xlsx)
- Artifact lineage: [data/companies/v/research/lineage_manifest.json](../../data/companies/v/research/lineage_manifest.json)
- Intelligence categories: business_quality, expectations, historical_valuation

### Microsoft Corporation (MSFT)

- Report: [data/companies/msft/research/msft_investment_report.html](../../data/companies/msft/research/msft_investment_report.html)
- PDF: [data/companies/msft/research/msft_investment_report.pdf](../../data/companies/msft/research/msft_investment_report.pdf)
- Excel model: [data/companies/msft/model/msft_valuation_model.xlsx](../../data/companies/msft/model/msft_valuation_model.xlsx)
- Artifact lineage: [data/companies/msft/research/lineage_manifest.json](../../data/companies/msft/research/lineage_manifest.json)
- Intelligence categories: business_quality, expectations, historical_valuation

## Reproduce

```bash
python src/pipeline.py --config config/company.yaml
python src/pipeline.py --config config/microsoft.yaml
python src/portfolio_demo.py
```
