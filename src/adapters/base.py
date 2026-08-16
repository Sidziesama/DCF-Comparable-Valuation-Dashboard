"""Business-model adapter contract; generic engines depend only on this module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import pandas as pd


KPI_COLUMNS = [
    "adapter", "company_ticker", "metric", "period", "period_type", "value", "unit",
    "source", "source_url", "filing_date", "retrieved_at", "mapping",
    "status", "quality", "definition",
]


class BusinessModelAdapter(ABC):
    """Optional boundary between sector economics and generic forecasting."""

    name = "generic"

    @abstractmethod
    def normalize_kpis(self, records: list[Mapping] | None) -> pd.DataFrame:
        """Normalize disclosed KPI records without inventing missing observations."""

    @abstractmethod
    def derive_metrics(self, kpis: pd.DataFrame) -> pd.DataFrame:
        """Return only defensible metrics derived from normalized disclosures."""

    @abstractmethod
    def forecast_growth(
        self, scenario: str, scenario_config: Mapping, years: int
    ) -> tuple[list[float], dict]:
        """Return revenue growth and transparent driver/fallback metadata."""


class GenericAdapter(BusinessModelAdapter):
    name = "generic"

    def normalize_kpis(self, records=None):
        return pd.DataFrame(columns=KPI_COLUMNS)

    def derive_metrics(self, kpis):
        return pd.DataFrame(columns=KPI_COLUMNS)

    def forecast_growth(self, scenario, scenario_config, years):
        values = scenario_config.get("revenue_growth", [])
        if not isinstance(values, (list, tuple)):
            values = [values] * years
        if len(values) != years:
            raise ValueError(f"{scenario}.revenue_growth must contain {years} values")
        return [float(x) for x in values], {
            "forecast_method": "top_down_fallback",
            "fallback_used": True,
            "fallback_reason": "No business-model driver forecast configured.",
            "driver_assumptions": {},
        }
