# Sprint 5 - Microsoft second-company proof

## Conclusion

The platform now credibly supports mature, non-financial public companies across two materially different business models: payment networks and diversified software/cloud. Microsoft completed the same SEC-to-research pipeline as Visa, including normalized historicals, linked three statements, scenarios, DCF and exit-multiple valuation, reverse DCF, historical/business-quality analytics, trading comps, evidence, deterministic recommendation, monitoring, Excel, dashboard-ready files, and institutional HTML/PDF reporting.

This is evidence of portability, not proof of universality. A third company in another sector is still required before claiming broad cross-sector coverage.

## What worked unchanged

- CompanyFacts and direct-filing ingestion architecture, annual/quarterly normalization, validation, and freshness controls.
- Fiscal-calendar-agnostic LTM construction using comparable YTD periods; Microsoft's June year-end required configuration and tests, not a Microsoft-only calculation branch.
- Forecast, linked three-statement, supporting-schedule, reconciliation, DCF, exit multiple, reverse DCF, model-health, research-product, recommendation, monitoring, report, Excel, and dashboard pipelines.
- Company-scoped workspace layout and artifact manifest after the isolation fix described below.

## Generic refactors required

- Configured XBRL aliases are now consumed by both CompanyFacts and direct-filing parsers, including concepts not present in the default metric map.
- Depreciation and amortization can be reported as separate concepts and are combined transparently for historical EBITDA.
- Benchmark selection is company-configured and consistent between core and consolidated valuation.
- Comparable-multiple eligibility is config-driven. Unknown peers retain standard multiples unless an explicit rule removes one.
- Direct-peer output names and football-field labels are generic; Visa retains its legacy Mastercard-named compatibility file.
- New-company workspaces no longer inherit flat legacy Visa artifacts. Legacy seeding is restricted to Visa.
- Evidence lineage resolves to the active company config rather than a hard-coded Visa config path.

## Sector-adapter-specific logic

- `payment_network` remains responsible for payment volume, cross-border volume, processed transactions, and its revenue bridge.
- `software` owns Microsoft Cloud, disclosed segment revenue, Azure/cloud growth, subscriber/margin KPI normalization, and a three-segment revenue bridge.
- Microsoft peer selection, peer limitations, scenario drivers, capex/SBC/margin assumptions, monitoring thresholds, fiscal metadata, and company-specific aliases live in `config/microsoft.yaml`.

## Microsoft data limitations

- Microsoft does not disclose standalone Azure revenue; the adapter stores only the reported Azure and other cloud services growth rate.
- Microsoft 365 Commercial seat history and commercial RPO were not populated without a consistent, audited multi-period series. Missing data is not fabricated.
- Segment definitions were recast in FY2025; the configured 2023-2025 series uses the recast presentation.
- AI infrastructure makes recent capex unusually high. Forecast capex normalization is an analyst assumption and materially affects FCFF.
- No single public company is a clean Microsoft comparable. ORCL/CRM/ADBE/SAP anchor enterprise software; GOOGL/AMZN are cloud-platform references with conglomerate-mix limitations.
- The generated report health check warns that the structured business-overview section is empty. All material generated claims remain evidence-resolved.

## Remaining generalization work

- Prove the system on a third mature non-financial sector with different working-capital and asset-intensity economics.
- Add a richer, sourced business-overview claim type so report section coverage can pass without templated prose.
- Expand component-aware XBRL aggregation beyond D&A where issuers split other economically identical line items.
- Add point-in-time market-data snapshots or a licensed data source for fully reproducible comps and valuation runs.
- Consider sector-specific terminal-value policy ranges; Microsoft's exit-multiple outputs demonstrate how sensitive mature software valuations are to terminal assumptions.
