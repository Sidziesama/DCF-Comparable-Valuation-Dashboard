import pandas as pd
import yfinance as yf
import numpy as np

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


def calculate_beta(
    ticker,
    benchmark="SPY",
    period="5y",
):
    """
    Estimate beta using weekly returns.

    Beta =
    Cov(stock, market)
    /
    Var(market)
    """

    prices = yf.download(
        [ticker, benchmark],
        period=period,
        interval="1wk",
        auto_adjust=True,
        progress=False,
    )

    close = prices["Close"]

    returns = (
        close
        .pct_change()
        .dropna()
    )

    stock_returns = returns[
        ticker
    ]

    market_returns = returns[
        benchmark
    ]

    covariance = np.cov(
        stock_returns,
        market_returns,
        ddof=1,
    )[0, 1]

    market_variance = np.var(
        market_returns,
        ddof=1,
    )

    beta = (
        covariance
        / market_variance
    )

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "observations": len(returns),
        "beta": beta,
        "stock_volatility": (
            stock_returns.std()
            * np.sqrt(52)
        ),
        "market_volatility": (
            market_returns.std()
            * np.sqrt(52)
        ),
        "correlation": (
            stock_returns.corr(
                market_returns
            )
        ),
    }
