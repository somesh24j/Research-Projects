from src.data_loader import load_prices, prices_to_returns
from src.backtest_baselines import run_baselines

if __name__ == "__main__":
    tickers = ["SPY", "IEF", "GLD", "SHY"]
    prices = load_prices(tickers, start="2006-01-01")
    rets = prices_to_returns(prices)

    summary, _, _ = run_baselines(rets, cost_bps=10.0, rebalance_freq="M")

    print("\n===== BASELINE PERFORMANCE =====\n")
    print(summary.to_string())
