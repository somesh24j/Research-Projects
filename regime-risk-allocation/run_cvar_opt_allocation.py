import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_prices, prices_to_returns
from src.backtest_baselines import run_baselines, backtest
from src.features import make_regime_features, zscore
from src.regimes.hmm import (
    fit_gaussian_hmm,
    regime_probabilities,
    smooth_probabilities,
    label_risk_off_by_return,
)
from src.alloc.optimizer_cvar import (
    cvar_optimize_weights,
    build_regime_mixture_scenarios,
)
from src.metrics import equity_curve, max_drawdown, turnover_from_weights
from src.risk.cvar import cvar, rolling_cvar


def make_cvar_opt_weights_over_time(
    returns: pd.DataFrame,
    probs: pd.DataFrame,
    lookback: int = 756,          # ~3 years of daily data
    rebalance_freq: str = "M",
    alpha: float = 0.95,
    mu_target_mode: str = "equal_weight",  # 'none' | 'equal_weight'
    turnover_limit: float = 0.10,          # 10% turnover per rebalance (0.5*L1)
) -> pd.DataFrame:
    """
    Create target weights at each rebalance date using regime-mixture scenarios
    and CVaR minimization.

    mu_target_mode:
      - 'none': no expected return constraint
      - 'equal_weight': require expected return >= equal-weight expected return in the scenario set
    """
    tickers = list(returns.columns)

    # Hard regimes from probs
    hard = probs.idxmax(axis=1)
    regime_cols = list(probs.columns)

    # Determine rebalance dates (month-end trading day)
    rb_dates = pd.Series(returns.index, index=returns.index).resample(rebalance_freq).last().dropna().values

    # Build weights dataframe (only filled on rb dates; later backtest aligns/ffills)
    W = pd.DataFrame(index=returns.index, columns=tickers, dtype=float)

    w_prev = np.array([1.0 / len(tickers)] * len(tickers), dtype=float)

    for dt in rb_dates:
        if dt not in returns.index:
            continue
        # Need regime probs at dt; if missing, skip
        if dt not in probs.index:
            continue

        end_loc = returns.index.get_loc(dt)
        start_loc = max(0, end_loc - lookback)
        window_idx = returns.index[start_loc:end_loc + 1]

        Rw = returns.loc[window_idx].dropna()
        if len(Rw) < 200:
            continue

        # Build scenario mixture from regime probs at dt
        current_probs = probs.loc[dt]
        scen_R, scen_p = build_regime_mixture_scenarios(
            returns_window=Rw,
            hard_regimes_window=hard,
            current_regime_probs=current_probs,
            regime_cols=regime_cols,
            min_per_regime=30,
        )

        # Optional expected return target
        mu_target = None
        if mu_target_mode == "equal_weight":
            w_eq = np.array([1.0 / len(tickers)] * len(tickers))
            mu_eq = float((scen_p[:, None] * scen_R).sum(axis=0).dot(w_eq))
            mu_target = mu_eq

        try:
            # Attempt 1: full constraints
            w_star = cvar_optimize_weights(
                scenario_returns=scen_R,
                scenario_probs=scen_p,
                alpha=alpha,
                mu_target=mu_target,
                w_prev=w_prev,
                turnover_limit=turnover_limit,
                w_bounds=(0.0, 1.0),
            )
        except RuntimeError:
            try:
                # Attempt 2: drop expected return constraint
                w_star = cvar_optimize_weights(
                    scenario_returns=scen_R,
                    scenario_probs=scen_p,
                    alpha=alpha,
                    mu_target=None,
                    w_prev=w_prev,
                    turnover_limit=turnover_limit,
                    w_bounds=(0.0, 1.0),
                )
            except RuntimeError:
                # Attempt 3: drop turnover constraint too (always feasible long-only)
                w_star = cvar_optimize_weights(
                    scenario_returns=scen_R,
                    scenario_probs=scen_p,
                    alpha=alpha,
                    mu_target=None,
                    w_prev=None,
                    turnover_limit=None,
                    w_bounds=(0.0, 1.0),
                )


        W.loc[dt] = w_star
        w_prev = w_star

    # Forward fill weights for daily backtest usage; drop pre-first allocation
    W = W.ffill().dropna()
    return W


if __name__ == "__main__":
    tickers = ["SPY", "IEF", "GLD", "SHY"]
    prices = load_prices(tickers, start="2006-01-01")
    rets = prices_to_returns(prices)

    # Baselines
    summary, bt_eq, bt_6040 = run_baselines(rets, cost_bps=10.0, rebalance_freq="M")
    print("\n===== BASELINE PERFORMANCE =====\n")
    print(summary.to_string())

    # Regime detection
    X_raw = make_regime_features(rets, market_col="SPY", vol_lookback=20, dd_lookback=252)
    X = zscore(X_raw)

    model = fit_gaussian_hmm(X, n_states=2, seed=42)
    raw_probs = regime_probabilities(model, X)
    probs = smooth_probabilities(raw_probs, span=20)

    labels = label_risk_off_by_return(X_raw, probs)
    print("\nRegime mean returns:", labels["avg_ret_by_regime"])
    print("Risk-off regime:", labels["risk_off"], "| Risk-on regime:", labels["risk_on"])

    # CVaR-optimized weights 
    W_cvar = make_cvar_opt_weights_over_time(
        returns=rets,
        probs=probs,
        lookback=756,         # 3 years
        rebalance_freq="M",
        alpha=0.95,
        mu_target_mode="equal_weight",
        turnover_limit=0.10,  # 10% turnover per rebalance
    )

    bt_cvar = backtest(
        returns=rets,
        target_weights=W_cvar,
        rebalance_freq="M",
        cost_bps=10.0,
    )

    # Compare curves 
    plt.figure()
    equity_curve(bt_eq["returns"]).plot(label="EqualWeight")
    equity_curve(bt_6040["returns"]).plot(label="60/40")
    equity_curve(bt_cvar["returns"]).plot(label="CVaR-Optim (Regime Mixture)")
    plt.title("Equity Curves: Baseline vs CVaR-Optim (Regime-Aware)")
    plt.legend()
    plt.show()

    # Tail risk 
    alpha = 0.95
    common_idx = bt_eq["returns"].index.intersection(bt_6040["returns"].index).intersection(bt_cvar["returns"].index)

    r_eq = bt_eq["returns"].loc[common_idx]
    r_6040 = bt_6040["returns"].loc[common_idx]
    r_cvar = bt_cvar["returns"].loc[common_idx]

    from src.metrics import cagr, annualized_vol, sharpe

    print("\n===== CVaR-Optim Performance =====\n")
    print("CAGR   :", cagr(r_cvar))
    print("AnnVol :", annualized_vol(r_cvar))
    print("Sharpe :", sharpe(r_cvar))


    print("\nCVaR (Expected Shortfall) at 95% (more negative = worse tail risk)")
    print("EqualWeight :", cvar(r_eq, alpha=alpha))
    print("60/40       :", cvar(r_6040, alpha=alpha))
    print("CVaR-Optim  :", cvar(r_cvar, alpha=alpha))

    # Rolling CVaR
    window = 252
    plt.figure()
    rolling_cvar(r_eq, window=window, alpha=alpha).plot(label="EqualWeight")
    rolling_cvar(r_6040, window=window, alpha=alpha).plot(label="60/40")
    rolling_cvar(r_cvar, window=window, alpha=alpha).plot(label="CVaR-Optim")
    plt.title(f"Rolling CVaR {int(alpha*100)}% (window={window}d)")
    plt.xlabel("Date")
    plt.ylabel("CVaR (daily return, negative)")
    plt.legend()
    plt.show()

    # Drawdowns 
    print("\nMax Drawdowns")
    print("EqualWeight :", max_drawdown(r_eq))
    print("60/40       :", max_drawdown(r_6040))
    print("CVaR-Optim  :", max_drawdown(r_cvar))

    # Turnover 
    turn = turnover_from_weights(bt_cvar["weights"]).dropna()
    print("\nTurnover (CVaR-Optim)")
    print("Average per rebalance day:", float(turn.mean()))
    print("Annualized (approx):", float(turn.mean() * 12))
