import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import equity_curve, cagr, annualized_vol, sharpe, max_drawdown

def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> set:
    # last trading day in each month/week/quarter
    rb = pd.Series(index, index=index).resample(freq).last().dropna().values
    return set(rb)

def backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    rebalance_freq: str = "M",
    cost_bps: float = 10.0,
) -> dict:
    """
    Periodic-rebalance backtest with simple transaction costs.
    cost_bps applies per $ traded on rebalance days using turnover.
    """
    idx = returns.index
    tickers = list(returns.columns)

    Wt = target_weights.reindex(idx).ffill().fillna(0.0)[tickers]
    # normalize weights
    Wt = Wt.div(Wt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # start on first date where weights are valid (sum > 0) 
    valid = Wt.sum(axis=1) > 0
    if not valid.any():
        raise ValueError("All target weights are zero after alignment. Check your weight generation.")
    start_date = valid.idxmax()

    returns = returns.loc[start_date:]
    Wt = Wt.loc[start_date:]
    idx = returns.index


    rb_dates = _rebalance_dates(idx, rebalance_freq)

    weights = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    port_ret = pd.Series(index=idx, dtype=float)

    w = Wt.iloc[0].copy()
    weights.iloc[0] = w.values
    port_ret.iloc[0] = float((w * returns.iloc[0]).sum())

    for i in range(1, len(idx)):
        date = idx[i]
        r = returns.iloc[i]

        # drift weights
        w = w * (1.0 + r)
        s = w.sum()
        w = (w / s) if s != 0 else w

        traded_cost = 0.0
        if date in rb_dates:
            w_target = Wt.loc[date]
            turnover = float((w_target - w).abs().sum() / 2.0)
            traded_cost = turnover * (cost_bps / 1e4)
            w = w_target.copy()

        weights.iloc[i] = w.values
        port_ret.iloc[i] = float((w * r).sum() - traded_cost)

    return {"returns": port_ret.dropna(), "weights": weights.dropna()}

def make_equal_weight(index: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    return pd.DataFrame(1.0 / len(tickers), index=index, columns=tickers)

def make_6040(index: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=index, columns=tickers)
    if "SPY" in tickers and "IEF" in tickers:
        W["SPY"] = 0.60
        W["IEF"] = 0.40
    else:
        W.loc[:] = 1.0 / len(tickers)
    return W

def summarize(name: str, r: pd.Series) -> dict:
    return {
        "Strategy": name,
        "CAGR": cagr(r),
        "AnnVol": annualized_vol(r),
        "Sharpe": sharpe(r),
        "MaxDD": max_drawdown(r),
    }

def run_baselines(returns: pd.DataFrame, cost_bps: float = 10.0, rebalance_freq: str = "M"):
    tickers = list(returns.columns)

    W_eq = make_equal_weight(returns.index, tickers)
    W_6040 = make_6040(returns.index, tickers)

    bt_eq = backtest(returns, W_eq, rebalance_freq=rebalance_freq, cost_bps=cost_bps)
    bt_6040 = backtest(returns, W_6040, rebalance_freq=rebalance_freq, cost_bps=cost_bps)

    summary = pd.DataFrame([
        summarize("EqualWeight", bt_eq["returns"]),
        summarize("60/40", bt_6040["returns"]),
    ]).set_index("Strategy")

    plt.figure()
    equity_curve(bt_eq["returns"]).plot(label="EqualWeight")
    equity_curve(bt_6040["returns"]).plot(label="60/40")
    plt.title("Baseline Equity Curves")
    plt.legend()
    plt.show()

    return summary, bt_eq, bt_6040
