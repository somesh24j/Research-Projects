# Regime-Aware Portfolio Allocation with CVaR Optimization

## Overview
Traditional asset allocation models—such as equal-weighting or fixed 60/40 splits—often fail during periods of high market stress because they assume stable return distributions. In reality, markets shift between distinct regimes where downside risk becomes highly asymmetric.

This project implements a system that explicitly manages these shifts. By combining **probabilistic regime detection** with **Conditional Value at Risk (CVaR) optimization**, the model prioritizes capital preservation during market turbulence without relying on static, historical averages.



---

## Methodology

### 1. Probabilistic Regime Detection
Instead of making "hard" binary guesses about market states, the system utilizes a **Gaussian Hidden Markov Model (HMM)**. 
* **Latent States:** The model infers unobserved market regimes (e.g., low-volatility growth vs. high-volatility contraction) based on log returns, realized volatility, and rolling drawdowns.
* **Continuous Beliefs:** The system maintains a posterior probability distribution across regimes (e.g., a 70% belief in a "risk-on" state), which allows for smoother transitions and reduced portfolio turnover.

### 2. Regime-Mixture Scenario Construction
Rather than assuming the future will mirror the unconditional past, we construct a forward-looking scenario set:
* Historical returns are partitioned by their inferred regime.
* We generate a mixture of these returns weighted by the model's current regime probabilities.
* This ensures the optimizer is solving for the risks most relevant to the current environment.

### 3. CVaR Optimization
The portfolio weights are determined by solving a linear program designed to minimize **Conditional Value at Risk (Expected Shortfall)** at the 95% confidence level.



**Constraints & Parameters:**
* **Long-Only:** Weights are constrained to $[0, 1]$.
* **Turnover Control:** An $L_1$ penalty is applied to minimize unnecessary trading and slippage.
* **Objective:** Unlike variance-based (Mean-Variance) optimization, CVaR focuses specifically on the expected loss in the worst 5% of outcomes, making it far more robust against fat-tailed distributions.

---

## Performance & Results

The strategy was backtested against standard benchmarks (60/40 and Equal-Weight) with the following key findings:

| Metric | CVaR Optimized | 60/40 Benchmark |
| :--- | :--- | :--- |
| **Expected Tail Loss (95%)** | ~40% Improvement | Baseline |
| **Annualized Volatility** | ~6.9% | Higher |
| **Sharpe Ratio** | ~1.23 | ~0.85 - 0.95 |
| **Annualized Turnover** | ~3.0% | N/A |

### Key Insights
* **Risk Mitigation:** The strategy achieves significantly lower tail loss by proactively de-risking when regime probabilities shift toward high-volatility states.
* **Execution Realism:** By enforcing turnover constraints, the model remains stable under transaction costs up to 25 bps, proving its viability for real-world deployment.

---

## Repository Structure

regime-risk-allocation/
├── src/
│   ├── regimes/       # HMM inference and state detection
│   ├── alloc/         # Linear programming for CVaR weights
│   └── risk/          # Mathematical definitions for tail risk metrics
├── run_cvar_opt.py    # Main backtest and visualization
└── run_cost_sweep.py  # Transaction cost sensitivity analysis

## Future Work

* **Multi-Regime Expansion:** Exploring 3-state models (Bull, Bear, and Stagnation).

* **Monte Carlo Integration:** Using regime-specific parameters to simulate synthetic stress scenarios.

* **Alternative Risk Measures:** Comparing CVaR outcomes against Entropic Value at Risk (EVaR).

```text