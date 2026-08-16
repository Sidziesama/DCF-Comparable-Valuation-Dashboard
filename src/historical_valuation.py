"""Point-in-time historical valuation framework."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from investment_intelligence import SCHEMA_COLUMNS
except ModuleNotFoundError:  # pragma: no cover
    from .investment_intelligence import SCHEMA_COLUMNS


def calculate_historical_multiples(financials: pd.DataFrame, market_history: pd.DataFrame) -> pd.DataFrame:
    """Calculate multiples only for exactly aligned fiscal periods.

    market_history must contain period and contemporaneous market_cap. EV can be
    supplied directly or derived from same-period debt/cash in financials.
    """
    if "period" not in market_history or "period" not in financials:
        raise ValueError("financials and market_history require a period column.")
    merged = financials.merge(market_history, on="period", how="inner", validate="one_to_one")
    rows = []
    for _, item in merged.iterrows():
        market_cap = pd.to_numeric(pd.Series([item.get("market_cap")]), errors="coerce").iloc[0]
        ev = item.get("enterprise_value")
        if pd.isna(ev):
            ev = market_cap + item.get("debt", 0) - item.get("cash", 0)
        for metric, value in (
            ("P/E", market_cap / item.get("net_income") if item.get("net_income") else np.nan),
            ("EV / EBITDA", ev / item.get("ebitda") if item.get("ebitda") else np.nan),
            ("EV / EBIT", ev / item.get("ebit") if item.get("ebit") else np.nan),
            ("FCF yield", item.get("fcf") / market_cap if market_cap else np.nan),
        ):
            rows.append({"period": item["period"], "metric": metric, "value": value})
    return pd.DataFrame(rows)


def build_historical_valuation_intelligence(
    historical: pd.DataFrame, *, current_market_cap: float, current_enterprise_value: float,
    market_history: pd.DataFrame | None = None, peer_multiples: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ltm = historical.loc["LTM"] if "LTM" in historical.index else historical.iloc[-1]
    current = {
        "P/E": current_market_cap / ltm.get("net_income") if ltm.get("net_income") else np.nan,
        "EV / EBITDA": current_enterprise_value / ltm.get("ebitda") if ltm.get("ebitda") else np.nan,
        "EV / EBIT": current_enterprise_value / ltm.get("operating_income") if ltm.get("operating_income") else np.nan,
        "FCF yield": ltm.get("fcf") / current_market_cap if current_market_cap else np.nan,
    }
    aligned = pd.DataFrame(columns=["period", "metric", "value"])
    if market_history is not None and not market_history.empty:
        financials = historical.loc[historical.index.astype(str) != "LTM"].reset_index().rename(columns={historical.index.name or "index": "period", "operating_income": "ebit"})
        financials["debt"] = financials.get("short_term_debt", 0) + financials.get("long_term_debt", 0)
        aligned = calculate_historical_multiples(financials, market_history)
    rows = []
    summary = []
    peer_map = {"P/E": "pe", "EV / EBITDA": "ev_ebitda", "EV / EBIT": "ev_ebit"}
    for metric, value in current.items():
        history = aligned.loc[aligned.metric.eq(metric), "value"].dropna()
        peer_values = pd.Series(dtype=float)
        if peer_multiples is not None and peer_map.get(metric) in peer_multiples:
            peer_values = pd.to_numeric(peer_multiples[peer_map[metric]], errors="coerce").dropna()
        peer_median = peer_values.median() if not peer_values.empty else np.nan
        stats = {
            "current": value, "historical_median": history.median() if len(history) else np.nan,
            "historical_min": history.min() if len(history) else np.nan,
            "historical_max": history.max() if len(history) else np.nan,
            "historical_percentile": (history.le(value).mean() if len(history) else np.nan),
            "peer_median": peer_median,
            "premium_discount_to_peers": value / peer_median - 1 if peer_median else np.nan,
            "history_status": "available" if len(history) else "not_available",
        }
        summary.append({"metric": metric, **stats})
        for label, stat_value in stats.items():
            if label == "history_status":
                continue
            status = "available" if np.isfinite(stat_value) else "not_available"
            quality = "point_in_time_aligned" if label.startswith("historical") and status == "available" else ("current_ltm" if label == "current" else ("current_peer_set" if "peer" in label and status == "available" else "insufficient_data"))
            rows.append({
                "category": "historical_valuation", "metric": f"{metric} {label.replace('_', ' ')}",
                "scope": "market", "scenario": "current", "period": "Current/LTM",
                "value": stat_value, "units": "ratio" if metric == "FCF yield" or "percentile" in label or "premium" in label else "multiple",
                "source": "market data + normalized financials", "lineage": "market/current and normalized/historical_model.csv",
                "status": status, "quality": quality, "interpretation": "",
            })
    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS), pd.DataFrame(summary)
