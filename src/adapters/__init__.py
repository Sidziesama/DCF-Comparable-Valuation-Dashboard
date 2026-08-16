"""Optional company/business-model adapters; core modeling stays generic."""

from .base import BusinessModelAdapter, GenericAdapter, KPI_COLUMNS
from .payment_network import PaymentNetworkAdapter
from .software import SoftwareAdapter


_ADAPTERS = {
    "generic": GenericAdapter,
    "payment_network": PaymentNetworkAdapter,
    "software": SoftwareAdapter,
}


def get_adapter(name: str | None) -> BusinessModelAdapter:
    key = (name or "generic").strip().lower()
    if key not in _ADAPTERS:
        raise ValueError(f"Unknown business-model adapter {name!r}; available: {tuple(_ADAPTERS)}")
    return _ADAPTERS[key]()


__all__ = ["BusinessModelAdapter", "GenericAdapter", "KPI_COLUMNS", "SoftwareAdapter", "get_adapter"]
