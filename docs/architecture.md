# Architecture

The platform separates company identity and assumptions from reusable financial logic. Each run selects one YAML config and writes only to `data/companies/<ticker>/`.

```mermaid
flowchart LR
  C[Company YAML] --> P[Pipeline orchestration]
  S[SEC XBRL] --> N[Normalization]
  M[Market data] --> V
  P --> S
  N --> H[Historical model]
  C --> A[Sector adapter]
  A --> F[Bear / Base / Bull forecasts]
  H --> F
  F --> T[Linked three statements + controls]
  T --> V[DCF / comps / reverse DCF]
  V --> I[Investment intelligence]
  I --> R[Recommendation + evidence + monitoring]
  R --> O[HTML / PDF / Excel / dashboard artifacts]
  O --> D[Visa + Microsoft portfolio demo]
```

The shared core owns SEC ingestion, statement linkage, valuation, controls, research schemas, reporting, and workspace routing. Adapters own sector-specific KPI normalization and the operating-driver bridge. A failed optional adapter falls back explicitly to configured top-down growth; it does not silently alter valuation logic.
