import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM


def fit_gaussian_hmm(X: pd.DataFrame, n_states: int = 2, seed: int = 42) -> GaussianHMM:
    """
    Fits a Gaussian HMM on feature matrix X (rows=time).
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=seed,
    )
    model.fit(X.values)
    return model


def regime_probabilities(model: GaussianHMM, X: pd.DataFrame) -> pd.DataFrame:
    """
    Smoothed posterior probabilities for each regime.
    """
    post = model.predict_proba(X.values)  # (T, K)
    return pd.DataFrame(
        post,
        index=X.index,
        columns=[f"regime_{k}" for k in range(post.shape[1])],
    )


def smooth_probabilities(probs: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """
    Exponentially smooth regime probabilities to reduce regime whipsaw.
    span ~ 10–30 (daily data) is a reasonable range.
    """
    return probs.ewm(span=span, adjust=False).mean()


def label_risk_off_by_return(X_raw: pd.DataFrame, probs: pd.DataFrame) -> dict:
    """
    Label regimes using average market return when each regime is most likely.
    The regime with lower avg return is 'risk_off' (simple, explainable).
    """
    hard = probs.idxmax(axis=1)
    avg_ret = {}
    for col in probs.columns:
        mask = hard == col
        avg_ret[col] = float(X_raw.loc[mask, "r"].mean())

    risk_off = min(avg_ret, key=avg_ret.get)
    risk_on = max(avg_ret, key=avg_ret.get)
    return {"risk_off": risk_off, "risk_on": risk_on, "avg_ret_by_regime": avg_ret}


def plot_regimes(
    portfolio_returns: pd.Series,
    probs: pd.DataFrame,
    labels: dict,
    title: str = "Regime Detection (HMM)",
    shade_threshold: float = 0.6,
):
    """
    1) Equity curve with risk-off shading (risk_off prob > shade_threshold)
    2) Regime probability chart
    """
    # Align indices
    idx = portfolio_returns.index.intersection(probs.index)
    pr = portfolio_returns.loc[idx]
    probs = probs.loc[idx]

    eq = (1.0 + pr).cumprod()
    risk_off_col = labels["risk_off"]
    risk_off_prob = probs[risk_off_col]
    risk_off_mask = risk_off_prob > shade_threshold

    # --- Plot 1: Equity curve + shading ---
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title(f"{title} — Equity curve (shaded = risk-off)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")

    # Shade contiguous risk-off regions
    in_region = False
    start = None
    for t, is_off in zip(eq.index, risk_off_mask):
        if is_off and not in_region:
            in_region = True
            start = t
        if (not is_off) and in_region:
            plt.axvspan(start, t, alpha=0.2)
            in_region = False
    if in_region:
        plt.axvspan(start, eq.index[-1], alpha=0.2)

    plt.show()

    # --- Plot 2: Regime probabilities ---
    plt.figure()
    for col in probs.columns:
        plt.plot(probs.index, probs[col].values, label=col)
    plt.title(f"{title} — Posterior regime probabilities")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()
