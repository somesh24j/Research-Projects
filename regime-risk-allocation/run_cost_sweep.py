import os
import numpy as np
import pandas as pd

from src.data_loader import load_prices, prices_to_returns
from src.features import make_regime_features, zscore
from src.regimes.hmm import fit_gaussian_hmm, regime_probabilities, smooth_probabilities
from src.alloc.optimizer_cvar import cvar_optimize_weights, build_regime_mixture_scenarios
from src.backtest_baselines import backtest, make_equal_weight, make_6040
from src.metrics import cagr, annualized_vol, sharpe, max_drawdown, turnover_from_weights
from src.risk.cvar import cvar


def make_cvar_opt_weights_over_time(
    returns: pd.DataFrame,
    probs: pd.DataFrame,
    lookback: int = 756,          # ~3 years
    rebalance_freq: str = "M",
    alpha: float = 0.95,
    mu_target_mode: str = "none",  # "none" is most robust
    turnover_limit: float = 0.10,  # 10% turnover per rebalance (0.5*L1)
) -> pd.DataFrame:
    """
    Create target weights at each rebalance date using regime-mixture scenarios
    and CVaR minimization.
    """
    tickers = list(returns.columns)

    hard = probs.idxmax(axis=1)
    regime_cols = list(probs.columns)

    rb_dates = pd.Series(returns.index, index=returns.index).resample(rebalance_freq).last().dropna().values

    W = pd.DataFrame(index=returns.index, columns=tickers, dtype=float)
    w_prev = np.array([1.0 / len(tickers)] * len(tickers), dtype=float)

    for dt in rb_dates:
        if dt not in returns.index:
            continue
        if dt not in probs.index:
            continue

        end_loc = returns.index.get_loc(dt)
        start_loc = max(0, end_loc - lookback)
        window_idx = returns.index[start_loc:end_loc + 1]

        Rw = returns.loc[window_idx].dropna()
        if len(Rw) < 200:
            continue

        current_probs = probs.loc[dt]
        scen_R, scen_p = build_regime_mixture_scenarios(
            returns_window=Rw,
            hard_regimes_window=hard,
            current_regime_probs=current_probs,
            regime_cols=regime_cols,
            min_per_regime=30,
        )

        mu_target = None
        if mu_target_mode == "equal_weight":
            w_eq = np.array([1.0 / len(tickers)] * len(tickers))
            mu_eq = float((scen_p[:, None] * scen_R).sum(axis=0).dot(w_eq))
            mu_target = mu_eq

        # Fallback ladder (robust)
        try:
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

    W = W.ffill().dropna()
    return W


def summarize_strategy(name: str, r: pd.Series, w: pd.DataFrame, alpha: float = 0.95) -> dict:
    turn = turnover_from_weights(w).dropna()
    ann_turn = float(turn.mean() * 12) if len(turn) else float("nan")

    return {
        "Strategy": name,
        "CAGR": cagr(r),
        "AnnVol": annualized_vol(r),
        "Sharpe": sharpe(r),
        "MaxDD": max_drawdown(r),
        "CVaR95": cvar(r, alpha=alpha),
        "AnnTurnover": ann_turn,
    }


if __name__ == "__main__":
    # Setup output folder 
    os.makedirs("results/tables", exist_ok=True)

    # Data 
    tickers = ["SPY", "IEF", "GLD", "SHY"]
    prices = load_prices(tickers, start="2006-01-01")
    rets = prices_to_returns(prices)

    # Regime probabilities (HMM) 
    X_raw = make_regime_features(rets, market_col="SPY", vol_lookback=20, dd_lookback=252)
    X = zscore(X_raw)

    model = fit_gaussian_hmm(X, n_states=2, seed=42)
    raw_probs = regime_probabilities(model, X)
    probs = smooth_probabilities(raw_probs, span=20)

    # Weights (independent of trading costs) 
    W_eq = make_equal_weight(rets.index, rets.columns)
    W_6040 = make_6040(rets.index, rets.columns)

    W_cvar = make_cvar_opt_weights_over_time(
        returns=rets,
        probs=probs,
        lookback=756,
        rebalance_freq="M",
        alpha=0.95,
        mu_target_mode="none",   # robust for sweeps
        turnover_limit=0.10,
    )

    # Align CVaR weights to returns index
    W_cvar = W_cvar.reindex(rets.index).ffill().dropna()
    rets2 = rets.loc[W_cvar.index[0]:]  # start when CVaR weights exist

    # Cost sweep
    cost_levels = [0, 5, 10, 25]  # bps
    rows = []
    alpha = 0.95

    for cost_bps in cost_levels:
        bt_eq = backtest(rets2, W_eq.loc[rets2.index], rebalance_freq="M", cost_bps=cost_bps)
        bt_6040 = backtest(rets2, W_6040.loc[rets2.index], rebalance_freq="M", cost_bps=cost_bps)
        bt_cvar = backtest(rets2, W_cvar.loc[rets2.index], rebalance_freq="M", cost_bps=cost_bps)

        # Align returns for fair comparison
        common = bt_eq["returns"].index.intersection(bt_6040["returns"].index).intersection(bt_cvar["returns"].index)

        r_eq = bt_eq["returns"].loc[common]
        r_6040 = bt_6040["returns"].loc[common]
        r_cvar = bt_cvar["returns"].loc[common]

        rows.append({"Cost_bps": cost_bps, **summarize_strategy("EqualWeight", r_eq, bt_eq["weights"].loc[common], alpha=alpha)})
        rows.append({"Cost_bps": cost_bps, **summarize_strategy("60/40", r_6040, bt_6040["weights"].loc[common], alpha=alpha)})
        rows.append({"Cost_bps": cost_bps, **summarize_strategy("CVaR-Optim", r_cvar, bt_cvar["weights"].loc[common], alpha=alpha)})

    df = pd.DataFrame(rows)

    # Pretty printing
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print("\n=== COST SWEEP RESULTS ===\n")
        print(df.sort_values(["Cost_bps", "Strategy"]).to_string(index=False))

    # Save
    outpath = "results/tables/cost_sweep.csv"
    df.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath}")
