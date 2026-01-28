from src.data_loader import load_prices, prices_to_returns
from src.backtest_baselines import run_baselines
from src.features import make_regime_features, zscore
from src.regimes.hmm import (
    fit_gaussian_hmm,
    regime_probabilities,
    smooth_probabilities,
    label_risk_off_by_return,
    plot_regimes,
)

if __name__ == "__main__":
    # Keep universe stable/robust on Yahoo
    tickers = ["SPY", "IEF", "GLD", "SHY"]
    prices = load_prices(tickers, start="2006-01-01")
    rets = prices_to_returns(prices)

    # Baselines (we'll visualize regimes on the EqualWeight portfolio)
    summary, bt_eq, bt_6040 = run_baselines(rets, cost_bps=10.0, rebalance_freq="M")
    print("\n===== BASELINE PERFORMANCE =====\n")
    print(summary.to_string())

    # Build regime features from SPY
    X_raw = make_regime_features(
        rets,
        market_col="SPY",
        vol_lookback=20,
        dd_lookback=252,
    )
    X = zscore(X_raw)

    # Fit HMM + probabilities (then smooth probabilities for stability)
    model = fit_gaussian_hmm(X, n_states=2, seed=42)
    raw_probs = regime_probabilities(model, X)
    probs = smooth_probabilities(raw_probs, span=20)

    # Label risk-off/risk-on using mean market return by regime
    labels = label_risk_off_by_return(X_raw, probs)
    print("\nRegime mean returns:", labels["avg_ret_by_regime"])
    print("Risk-off regime:", labels["risk_off"], "| Risk-on regime:", labels["risk_on"])

    # Plot using smoothed probabilities; adjust threshold if you want fewer/more shaded periods
    plot_regimes(
        portfolio_returns=bt_eq["returns"],
        probs=probs,
        labels=labels,
        title="HMM Regimes on SPY features (smoothed)",
        shade_threshold=0.6,
    )

