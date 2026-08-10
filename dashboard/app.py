from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sec_data import build_historical
from market_data import get_market_data
from model import forecast_financials, calculate_wacc, dcf_valuation, sensitivity_table
import yaml
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

with open(ROOT / "config/company.yaml") as f:
    cfg = yaml.safe_load(f)

st.set_page_config(page_title="Visa Valuation Dashboard", layout="wide")
st.title(f"{cfg['company_name']} ({cfg['ticker']}) — Valuation Dashboard")

@st.cache_data
def load_hist():
    return build_historical(cfg["cik"], cfg["historical_years"])

@st.cache_data
def load_market():
    return get_market_data(cfg["ticker"])

hist = load_hist()
market = load_market()

st.sidebar.header("DCF Assumptions")
growth = st.sidebar.slider("Terminal Growth", 0.0, 0.05, cfg["assumptions"]["terminal_growth"], 0.0025)
wacc_override = st.sidebar.slider("WACC", 0.05, 0.12, 0.08, 0.0025)
margin = st.sidebar.slider("Operating Margin", 0.50, 0.75, cfg["assumptions"]["operating_margin"], 0.01)

a = cfg["assumptions"].copy()
a["terminal_growth"] = growth
a["operating_margin"] = margin

forecast = forecast_financials(hist, a)
shares = market["shares_outstanding"] / 1_000_000
cash = (market.get("cash") or 0) / 1_000_000
debt = (market.get("total_debt") or 0) / 1_000_000

valuation = dcf_valuation(forecast, wacc_override, growth, cash, debt, shares)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Share Price", f"${market['price']:.2f}")
c2.metric("DCF Value", f"${valuation['intrinsic_price']:.2f}")
c3.metric("Upside / (Downside)", f"{valuation['intrinsic_price']/market['price']-1:.1%}")
c4.metric("WACC", f"{wacc_override:.2%}")

st.subheader("Historical Revenue")
h = hist.reset_index()
if "revenue" in h.columns:
    st.plotly_chart(px.line(h, x="fiscal_year", y="revenue", markers=True, title="Revenue ($mm)"), use_container_width=True)

st.subheader("Forecast")
f = forecast.reset_index()
st.plotly_chart(
    px.bar(f, x="year", y=["revenue", "fcff"], barmode="group", title="Forecast Revenue & FCFF ($mm)"),
    use_container_width=True
)

st.subheader("DCF Sensitivity — Intrinsic Value / Share")
waccs = [0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10]
gs = [0.015, 0.02, 0.025, 0.03, 0.035]
sens = sensitivity_table(forecast, waccs, gs, cash, debt, shares)
st.dataframe(sens.style.format("${:.2f}"), use_container_width=True)
