from src.risk.cvar import cvar, rolling_cvar
from src.data_loader import load_prices, prices_to_returns
from src.backtest_baselines import run_baselines, backtest
from src.features import make_regime_features, zscore
from src.regimes.hmm import (
    fit_gaussian_hmm,
    regime_probabilities,
    smooth_probabilities,
    label_risk_off_by_return,
)
from src.alloc.policy_rules import regime_weight_policy
from src.metrics import equity_curve, max_drawdown

import matplotlib.pyplot as plt

if __name__ == "__main__":
    tickers = ["SPY", "IEF", "GLD", "SHY"]
    prices = load_prices(tickers, start="2006-01-01")
    rets = prices_to_returns(prices)

    # Baselines
    summary, bt_eq, bt_6040 = run_baselines(rets, cost_bps=10.0)

    # Regime detection
    X_raw = make_regime_features(rets, market_col="SPY")
    X = zscore(X_raw)

    model = fit_gaussian_hmm(X, n_states=2)
    raw_probs = regime_probabilities(model, X)
    probs = smooth_probabilities(raw_probs, span=20)

    labels = label_risk_off_by_return(X_raw, probs)

    # Regime based weights
    W_regime = regime_weight_policy(probs, labels)

    # Align weights to returns index
    W_regime = W_regime.reindex(rets.index).ffill().dropna()


    # Backtest regime strategy
    bt_regime = backtest(
        returns=rets,
        target_weights=W_regime,
        rebalance_freq="M",
        cost_bps=10.0,
    )

    from src.metrics import turnover_from_weights

    turn = turnover_from_weights(bt_regime["weights"]).dropna()
    print("\nTurnover (RegimeAware)")
    print("Average per rebalance day:", turn.mean())
    print("Annualized (approx):", turn.mean() * 12)  # monthly rebalance


    # CVaR comparisons (tail risk) 
    alpha = 0.95
    window = 252  # ~1 trading year

    r_eq = bt_eq["returns"]
    r_6040 = bt_6040["returns"]
    r_reg = bt_regime["returns"]

    # Align to common dates so comparisons are fair
    common_idx = r_eq.index.intersection(r_6040.index).intersection(r_reg.index)
    r_eq = r_eq.loc[common_idx]
    r_6040 = r_6040.loc[common_idx]
    r_reg = r_reg.loc[common_idx]

    print("\nCVaR (Expected Shortfall) at 95% (more negative = worse tail risk)")
    print("EqualWeight :", cvar(r_eq, alpha=alpha))
    print("60/40       :", cvar(r_6040, alpha=alpha))
    print("RegimeAware :", cvar(r_reg, alpha=alpha))

    # Rolling CVaR
    rc_eq = rolling_cvar(r_eq, window=window, alpha=alpha)
    rc_6040 = rolling_cvar(r_6040, window=window, alpha=alpha)
    rc_reg = rolling_cvar(r_reg, window=window, alpha=alpha)

    import matplotlib.pyplot as plt

    plt.figure()
    rc_eq.plot(label="EqualWeight")
    rc_6040.plot(label="60/40")
    rc_reg.plot(label="Regime-Aware")
    plt.title(f"Rolling CVaR {int(alpha*100)}% (window={window}d)")
    plt.xlabel("Date")
    plt.ylabel("CVaR (daily return, negative)")
    plt.legend()
    plt.show()


    # Plot equity curves 
    plt.figure()
    equity_curve(bt_eq["returns"]).plot(label="EqualWeight")
    equity_curve(bt_6040["returns"]).plot(label="60/40")
    equity_curve(bt_regime["returns"]).plot(label="Regime-Aware")
    plt.title("Equity Curves: Baseline vs Regime-Aware")
    plt.legend()
    plt.show()

    # Print drawdowns 
    print("\nMax Drawdowns")
    print("EqualWeight :", max_drawdown(bt_eq["returns"]))
    print("60/40       :", max_drawdown(bt_6040["returns"]))
    print("RegimeAware :", max_drawdown(bt_regime["returns"]))
