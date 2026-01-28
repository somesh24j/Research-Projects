import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["SPY", "IEF", "GLD", "DBC", "SHY"]

def load_prices(
    tickers=DEFAULT_TICKERS,
    start="2006-01-01",
    end=None,
) -> pd.DataFrame:
    """
    Downloads adjusted (total-return) prices from Yahoo Finance via yfinance.
    auto_adjust=True adjusts for splits/dividends.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance sometimes returns multi-index columns (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            # fallback: pick first level if needed
            prices = data.xs(data.columns.levels[0][0], level=0, axis=1)
    else:
        prices = data.copy()

    prices = prices.dropna(how="all").ffill().dropna()
    prices.columns = [c.upper() for c in prices.columns]
    return prices

def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna()
