# Known limitations

- Market prices, peer multiples, beta, and filing freshness require live third-party access and can change between runs.
- SEC taxonomy differences still require aliases for some issuers; the current proof covers Visa and Microsoft, not every public company.
- Configured forecasts are analyst assumptions. Sector adapters bridge disclosed KPIs to revenue growth but do not forecast every business segment independently.
- Historical valuation percentiles remain unavailable without point-in-time historical market values.
- Comparable companies are imperfect and eligibility rules reduce, but do not eliminate, business-mix differences.
- PDF generation is optional for Visa and required by the Microsoft sample config; HTML remains the canonical report artifact.
- Dashboard assets are not committed as fabricated screenshots. Run the dashboard locally using the documented command to inspect current artifacts.
- Outputs are point-in-time research artifacts and are not investment advice.
