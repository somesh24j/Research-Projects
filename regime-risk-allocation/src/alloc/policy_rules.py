import pandas as pd
import numpy as np


def regime_weight_policy(
    probs: pd.DataFrame,
    labels: dict,
    assets=("SPY", "IEF", "GLD", "SHY"),
    risk_off_high: float = 0.7,
    risk_on_low: float = 0.3,
) -> pd.DataFrame:
    """
    Map regime probabilities to portfolio weights.

    - Uses smoothed risk-off probability
    - Linearly blends weights in the transition region
    """
    risk_off_col = labels["risk_off"]
    p_off = probs[risk_off_col]

    W = pd.DataFrame(index=probs.index, columns=assets, dtype=float)

    # Define anchor portfolios
    w_risk_on = pd.Series(
        {"SPY": 0.60, "IEF": 0.20, "GLD": 0.10, "SHY": 0.10}
    )
    w_risk_off = pd.Series(
        {"SPY": 0.30, "IEF": 0.45, "GLD": 0.15, "SHY": 0.10}
    )

    for t in W.index:
        p = p_off.loc[t]

        if p >= risk_off_high:
            w = w_risk_off
        elif p <= risk_on_low:
            w = w_risk_on
        else:
            # Linear interpolation
            alpha = (p - risk_on_low) / (risk_off_high - risk_on_low)
            w = (1 - alpha) * w_risk_on + alpha * w_risk_off

        W.loc[t] = w

    # Normalize (numerical safety)
    W = W.div(W.sum(axis=1), axis=0)
    return W
