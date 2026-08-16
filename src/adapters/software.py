"""Software/cloud operating economics isolated from generic model engines."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .base import BusinessModelAdapter, KPI_COLUMNS


class SoftwareAdapter(BusinessModelAdapter):
    """Normalize reported software KPIs and bridge segment drivers to revenue.

    The adapter deliberately does not infer undisclosed Azure revenue or fabricate
    subscriber histories. Absolute disclosed series may be converted to growth;
    reported growth observations remain reported observations.
    """

    name = "software"
    supported_metrics = {
        "microsoft_cloud_revenue",
        "microsoft_cloud_gross_margin",
        "azure_growth",
        "microsoft_365_commercial_cloud_growth",
        "microsoft_365_consumer_subscribers",
        "productivity_business_processes_revenue",
        "intelligent_cloud_revenue",
        "more_personal_computing_revenue",
        "commercial_remaining_performance_obligation",
        "capex",
        "stock_based_compensation",
    }

    def normalize_kpis(self, records: list[Mapping] | None) -> pd.DataFrame:
        rows = []
        for record in records or []:
            metric = str(record.get("metric", "")).strip().lower()
            value = pd.to_numeric(record.get("value"), errors="coerce")
            available = metric in self.supported_metrics and np.isfinite(value)
            rows.append({
                "adapter": self.name,
                "company_ticker": record.get("company_ticker", ""),
                "metric": metric,
                "period": str(record.get("period", "")),
                "period_type": record.get("period_type", "fiscal_year"),
                "value": float(value) if np.isfinite(value) else np.nan,
                "unit": record.get("unit", ""),
                "source": record.get("source", ""),
                "source_url": record.get("source_url", ""),
                "filing_date": record.get("filing_date", ""),
                "retrieved_at": record.get("retrieved_at", ""),
                "mapping": record.get("mapping", metric),
                "status": "available" if available else "not_available",
                "quality": record.get("quality", "reported" if available else "insufficient_data"),
                "definition": record.get("definition", ""),
            })
        return pd.DataFrame(rows, columns=KPI_COLUMNS)

    def derive_metrics(self, kpis: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if kpis.empty:
            return pd.DataFrame(columns=KPI_COLUMNS)
        available = kpis[kpis.status.eq("available")].copy()
        growth_eligible = available[~available.metric.str.endswith("_growth")]
        for metric, group in growth_eligible.groupby("metric"):
            group = group.sort_values("period")
            if len(group) < 2 or group["unit"].nunique() != 1:
                continue
            prior = None
            for _, item in group.iterrows():
                if prior is not None and prior["value"] > 0:
                    rows.append({
                        **{column: item.get(column, "") for column in KPI_COLUMNS},
                        "metric": f"{metric}_growth",
                        "value": item["value"] / prior["value"] - 1,
                        "unit": "ratio",
                        "source": "derived from disclosed KPI observations",
                        "mapping": f"pct_change({metric})",
                        "status": "available",
                        "quality": "calculated",
                        "definition": "Fiscal-period growth calculated only from consistent reported units.",
                    })
                prior = item
        return pd.DataFrame(rows, columns=KPI_COLUMNS)

    @staticmethod
    def _series(value, years, name):
        values = value if isinstance(value, (list, tuple)) else [value] * years
        if len(values) != years:
            raise ValueError(f"{name} must contain {years} values")
        return [float(x) for x in values]

    def forecast_growth(self, scenario, scenario_config, years):
        drivers = scenario_config.get("operating_drivers", {}) or {}
        required = (
            "productivity_business_processes_growth",
            "intelligent_cloud_growth",
            "more_personal_computing_growth",
        )
        if not all(name in drivers for name in required):
            growth = self._series(scenario_config["revenue_growth"], years, f"{scenario}.revenue_growth")
            return growth, {
                "forecast_method": "top_down_fallback",
                "fallback_used": True,
                "fallback_reason": "Incomplete software segment-driver assumptions.",
                "driver_assumptions": dict(drivers),
            }
        weights = drivers.get("revenue_growth_weights", {})
        if set(weights) != set(required) or abs(sum(float(weights[k]) for k in required) - 1.0) > 1e-9:
            raise ValueError("Software revenue growth weights must cover the three segments and sum to 1.0")
        series = {key: self._series(drivers[key], years, key) for key in required}
        overlay = self._series(drivers.get("mix_pricing_overlay", 0.0), years, "mix_pricing_overlay")
        growth = [sum(float(weights[key]) * series[key][i] for key in required) + overlay[i] for i in range(years)]
        return growth, {
            "forecast_method": "segment_driver_bridge",
            "fallback_used": False,
            "fallback_reason": "",
            "driver_assumptions": {**series, "mix_pricing_overlay": overlay, "revenue_growth_weights": dict(weights)},
        }
