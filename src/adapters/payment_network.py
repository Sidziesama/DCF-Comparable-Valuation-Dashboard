"""Payment-network operating economics, isolated from generic model engines."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .base import BusinessModelAdapter, KPI_COLUMNS

MULTIPLE_ELIGIBILITY = {
    "MA": {"ev_revenue", "ev_ebitda", "ev_ebit", "pe"},
    "PYPL": {"ev_revenue", "ev_ebitda", "ev_ebit", "pe"},
    "FISV": {"ev_revenue", "ev_ebitda", "ev_ebit", "pe"},
    "GPN": {"ev_revenue", "ev_ebitda", "ev_ebit", "pe"},
    # Lending economics make EV multiples less comparable for AXP.
    "AXP": {"pe"},
}

DEFAULT_DIRECT_PEER = "MA"


class PaymentNetworkAdapter(BusinessModelAdapter):
    name = "payment_network"
    supported_metrics = {
        "payments_volume", "cross_border_volume", "cross_border_volume_growth", "processed_transactions",
        "value_added_services_revenue", "client_incentives", "net_revenue",
    }

    def normalize_kpis(self, records: list[Mapping] | None) -> pd.DataFrame:
        rows = []
        for record in records or []:
            metric = str(record.get("metric", "")).strip().lower()
            value = pd.to_numeric(record.get("value"), errors="coerce")
            status = "available" if metric in self.supported_metrics and np.isfinite(value) else "not_available"
            rows.append({
                "adapter": self.name, "company_ticker": record.get("company_ticker", ""),
                "metric": metric, "period": str(record.get("period", "")),
                "period_type": record.get("period_type", "fiscal_year"),
                "value": float(value) if np.isfinite(value) else np.nan,
                "unit": record.get("unit", ""), "source": record.get("source", ""),
                "source_url": record.get("source_url", ""), "filing_date": record.get("filing_date", ""),
                "retrieved_at": record.get("retrieved_at", ""), "mapping": record.get("mapping", metric),
                "status": status,
                "quality": record.get("quality", "reported" if status == "available" else "insufficient_data"),
                "definition": record.get("definition", ""),
            })
        return pd.DataFrame(rows, columns=KPI_COLUMNS)

    def derive_metrics(self, kpis: pd.DataFrame) -> pd.DataFrame:
        rows = []
        if kpis.empty:
            return pd.DataFrame(columns=KPI_COLUMNS)
        available = kpis[kpis.status.eq("available")].copy()
        for metric, group in available.groupby("metric"):
            group = group.sort_values("period")
            if group["unit"].nunique() != 1:
                continue
            prior = None
            for _, item in group.iterrows():
                if prior is not None and prior["value"] > 0:
                    rows.append({
                        **{column: item.get(column, "") for column in KPI_COLUMNS},
                        "metric": f"{metric}_growth", "value": item["value"] / prior["value"] - 1,
                        "unit": "ratio", "source": "derived from disclosed KPI observations",
                        "mapping": f"pct_change({metric})", "status": "available", "quality": "calculated",
                        "definition": "Period-over-period growth; calculated only for consistent units.",
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
        required = ("payments_volume_growth", "cross_border_volume_growth", "processed_transactions_growth")
        if not all(name in drivers for name in required):
            values = self._series(scenario_config["revenue_growth"], years, f"{scenario}.revenue_growth")
            return values, {"forecast_method": "top_down_fallback", "fallback_used": True,
                "fallback_reason": "Incomplete payment-network driver assumptions.",
                "driver_assumptions": dict(drivers)}
        weights = drivers.get("revenue_growth_weights", {
            "payments_volume_growth": 0.45, "cross_border_volume_growth": 0.35,
            "processed_transactions_growth": 0.20,
        })
        if abs(sum(float(weights.get(k, 0)) for k in required) - 1) > 1e-9:
            raise ValueError("Payment-network revenue growth weights must sum to 1.0")
        series = {key: self._series(drivers[key], years, key) for key in required}
        overlay = self._series(drivers.get("monetization_overlay", 0.0), years, "monetization_overlay")
        growth = [sum(float(weights[key]) * series[key][i] for key in required) + overlay[i] for i in range(years)]
        return growth, {"forecast_method": "operating_driver_bridge", "fallback_used": False,
            "fallback_reason": "", "driver_assumptions": {**series, "monetization_overlay": overlay,
            "revenue_growth_weights": dict(weights)}}
