import numpy as np
import pandas as pd

TRADING_DAYS = 252

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()

def cagr(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float(eq.iloc[-1] ** (1 / years) - 1)

def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))

def sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    vol = excess.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float((excess.mean() / vol) * np.sqrt(TRADING_DAYS))

def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())  # negative number

def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    """
    Turnover per date: sum(|w_t - w_{t-1}|)/2
    """
    return weights.diff().abs().sum(axis=1) / 2.0

