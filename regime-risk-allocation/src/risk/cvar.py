import numpy as np
import pandas as pd

def cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """
    CVaR (Expected Shortfall) of returns at confidence alpha.
    Returns a NEGATIVE number (expected return in worst (1-alpha)% tail).
    """
    r = returns.dropna().values
    if len(r) == 0:
        return float("nan")

    # VaR threshold at (1-alpha) quantile (e.g., 5% left tail)
    q = np.quantile(r, 1 - alpha)
    tail = r[r <= q]
    if len(tail) == 0:
        return float("nan")
    return float(tail.mean())

def rolling_cvar(returns: pd.Series, window: int = 252, alpha: float = 0.95) -> pd.Series:
    """
    Rolling CVaR series (negative values).
    """
    return returns.rolling(window).apply(lambda x: cvar(pd.Series(x), alpha=alpha), raw=False)
