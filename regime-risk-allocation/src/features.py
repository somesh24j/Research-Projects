import numpy as np
import pandas as pd

def compute_drawdown_from_returns(r: pd.Series) -> pd.Series:
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd

def make_regime_features(
    returns: pd.DataFrame,
    market_col: str = "SPY",
    vol_lookback: int = 20,
    dd_lookback: int = 252,
) -> pd.DataFrame:
    """
    Build a small but powerful feature set for regime detection.

    Features (all based on market_col):
      - r: daily return
      - rv: realized vol (rolling std)
      - dd: current drawdown (from peak)
      - dd_min: rolling min drawdown over dd_lookback (severity proxy)
    """
    if market_col not in returns.columns:
        raise ValueError(f"{market_col} not found in returns columns: {list(returns.columns)}")

    r = returns[market_col].copy()
    rv = r.rolling(vol_lookback).std()

    dd = compute_drawdown_from_returns(r)
    dd_min = dd.rolling(dd_lookback).min()

    X = pd.DataFrame(
        {
            "r": r,
            "rv": rv,
            "dd": dd,
            "dd_min": dd_min,
        },
        index=returns.index,
    ).dropna()

    return X

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, np.nan)
    return (df - mu) / sd
