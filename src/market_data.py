import pandas as pd
import yfinance as yf

def get_market_data(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="1y", auto_adjust=False)

    return {
        "ticker": ticker,
        "price": float(hist["Close"].dropna().iloc[-1]),
        "market_cap": info.get("marketCap"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "beta": info.get("beta"),
        "enterprise_value": info.get("enterpriseValue"),
        "total_debt": info.get("totalDebt"),
        "cash": info.get("totalCash"),
    }

def get_peer_market_data(tickers):
    rows = []
    for ticker in tickers:
        try:
            d = get_market_data(ticker)
            d["ticker"] = ticker
            rows.append(d)
        except Exception as e:
            rows.append({"ticker": ticker, "error": str(e)})
    return pd.DataFrame(rows)
