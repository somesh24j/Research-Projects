import numpy as np
import pandas as pd
from scipy.optimize import linprog


def cvar_optimize_weights(
    scenario_returns: np.ndarray,
    scenario_probs: np.ndarray,
    alpha: float = 0.95,
    mu_target: float | None = None,
    w_prev: np.ndarray | None = None,
    turnover_limit: float | None = None,
    w_bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Minimize CVaR_alpha of portfolio returns using a linear program.

    scenario_returns: (M, N) array of scenario asset returns
    scenario_probs:   (M,) probabilities, must sum to 1
    alpha: CVaR confidence (0.95 => 5% tail)
    mu_target: optional minimum expected return constraint under scenario_probs
    w_prev: optional previous weights for turnover constraint
    turnover_limit: optional limit on 0.5*L1 turnover per rebalance (e.g., 0.10 => 10% turnover)
    w_bounds: (low, high) bounds for each asset weight (long-only default)

    Returns:
      w*: (N,) optimal weights summing to 1
    """
    R = np.asarray(scenario_returns, dtype=float)
    p = np.asarray(scenario_probs, dtype=float)

    if R.ndim != 2:
        raise ValueError("scenario_returns must be (M, N)")
    M, N = R.shape

    if p.shape != (M,):
        raise ValueError("scenario_probs must be shape (M,)")

    psum = p.sum()
    if psum <= 0:
        raise ValueError("scenario_probs sum must be > 0")
    p = p / psum

    eps = 1e-12
    p = np.maximum(p, eps)
    p = p / p.sum()


    # Decision variables:
    # x = [w_0..w_{N-1}, t, u_0..u_{M-1}, (optional) d_0..d_{N-1}]
    # CVaR LP:
    # minimize  t + (1/(1-alpha)) * sum_i p_i * u_i
    # s.t.      u_i >= -(w' r_i) - t
    #           u_i >= 0
    #           sum_j w_j = 1
    #           (optional) w' E[r] >= mu_target
    #           (optional) turnover: 0.5*sum_j |w_j - w_prev_j| <= turnover_limit
    #                    linearize with d_j >= w_j - w_prev_j, d_j >= -(w_j - w_prev_j)
    #
    use_turnover = (w_prev is not None) and (turnover_limit is not None)

    if use_turnover:
        w_prev = np.asarray(w_prev, dtype=float)
        if w_prev.shape != (N,):
            raise ValueError("w_prev must be shape (N,)")

    n_w = N
    idx_t = n_w
    idx_u0 = n_w + 1
    idx_d0 = idx_u0 + M if use_turnover else None

    n_vars = n_w + 1 + M + (N if use_turnover else 0)

    # Objective c
    c = np.zeros(n_vars)
    c[idx_t] = 1.0
    c[idx_u0:idx_u0 + M] = (1.0 / (1.0 - alpha)) * p

    # Inequality constraints A_ub x <= b_ub
    A_ub = []
    b_ub = []

    # For each scenario i: -u_i - w'r_i - t <= 0
    # Row: [-r_i (for w vars), -1 (t), -1 at u_i] <= 0
    for i in range(M):
        row = np.zeros(n_vars)
        row[:n_w] = -R[i, :]
        row[idx_t] = -1.0
        row[idx_u0 + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Expected return constraint: w' mu >= mu_target  =>  -w'mu <= -mu_target
    if mu_target is not None:
        mu = (p[:, None] * R).sum(axis=0)  # (N,)
        row = np.zeros(n_vars)
        row[:n_w] = -mu
        A_ub.append(row)
        b_ub.append(-float(mu_target))

    # Turnover constraints
    # d_j >= w_j - w_prev_j  =>  w_j - d_j <= w_prev_j
    # d_j >= -(w_j - w_prev_j) => -w_j - d_j <= -w_prev_j
    # and sum d_j <= 2*turnover_limit
    if use_turnover:
        for j in range(N):
            # w_j - d_j <= w_prev_j
            row1 = np.zeros(n_vars)
            row1[j] = 1.0
            row1[idx_d0 + j] = -1.0
            A_ub.append(row1)
            b_ub.append(float(w_prev[j]))

            # -w_j - d_j <= -w_prev_j
            row2 = np.zeros(n_vars)
            row2[j] = -1.0
            row2[idx_d0 + j] = -1.0
            A_ub.append(row2)
            b_ub.append(-float(w_prev[j]))

        # sum d_j <= 2*turnover_limit
        row3 = np.zeros(n_vars)
        row3[idx_d0:idx_d0 + N] = 1.0
        A_ub.append(row3)
        b_ub.append(2.0 * float(turnover_limit))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Equality constraint: sum w = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_w] = 1.0
    b_eq = np.array([1.0])

    # Bounds
    bounds = []
    lo, hi = w_bounds
    for _ in range(n_w):
        bounds.append((lo, hi))
    bounds.append((None, None))          # t unbounded
    for _ in range(M):
        bounds.append((0.0, None))       # u_i >= 0
    if use_turnover:
        for _ in range(N):
            bounds.append((0.0, None))   # d_j >= 0

    # Improve numerical conditioning (helps HiGHS)
    scale = 100.0
    R_scaled = R * scale

    # Rebuild A_ub using scaled returns
    A_ub = []
    b_ub = []

    for i in range(M):
        row = np.zeros(n_vars)
        row[:n_w] = -R_scaled[i, :]
        row[idx_t] = -1.0
        row[idx_u0 + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    if mu_target is not None:
        mu = (p[:, None] * R_scaled).sum(axis=0)
        row = np.zeros(n_vars)
        row[:n_w] = -mu
        A_ub.append(row)
        b_ub.append(-float(mu_target * scale))

    if use_turnover:
        for j in range(N):
            row1 = np.zeros(n_vars)
            row1[j] = 1.0
            row1[idx_d0 + j] = -1.0
            A_ub.append(row1)
            b_ub.append(float(w_prev[j]))

            row2 = np.zeros(n_vars)
            row2[j] = -1.0
            row2[idx_d0 + j] = -1.0
            A_ub.append(row2)
            b_ub.append(-float(w_prev[j]))

        row3 = np.zeros(n_vars)
        row3[idx_d0:idx_d0 + N] = 1.0
        A_ub.append(row3)
        b_ub.append(2.0 * float(turnover_limit))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Try multiple HiGHS solvers and accept numerically valid solutions
    for method in ["highs-ds", "highs-ipm", "highs"]:
        res = linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method=method,
            options={"presolve": True},
        )

        if res.x is not None and np.all(np.isfinite(res.x)):
            msg = (res.message or "").lower()
            if res.success or ("scaled model" in msg and "optimal" in msg):
                w = res.x[:n_w]
                w[w < 0] = 0
                s = w.sum()
                return w / s if s != 0 else w

    raise RuntimeError(f"CVaR optimization failed: {res.message}")



def build_regime_mixture_scenarios(
    returns_window: pd.DataFrame,
    hard_regimes_window: pd.Series,
    current_regime_probs: pd.Series,
    regime_cols: list[str],
    min_per_regime: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build scenario set by taking historical returns in a window, and weighting them by
    the *current* regime probabilities (mixture of regimes).

    returns_window: (T, N) asset returns
    hard_regimes_window: (T,) regime label (e.g., 'regime_0') per date
    current_regime_probs: (K,) probabilities for each regime column at current date
    regime_cols: list of regime column names ['regime_0', 'regime_1', ...]

    Returns:
      scenario_returns: (M, N)
      scenario_probs: (M,) sum to 1
    """
    # Keep only rows where we have a regime label
    df = returns_window.copy()
    df["regime"] = hard_regimes_window.reindex(df.index)

    df = df.dropna(subset=["regime"])
    if df.empty:
        raise ValueError("No labeled scenarios in the window.")

    # For each regime, assign equal probability mass within regime,
    # scaled by current regime probability.
    scenario_probs = np.zeros(len(df), dtype=float)

    for reg in regime_cols:
        mask = (df["regime"] == reg).values
        count = int(mask.sum())
        if count == 0:
            continue

        # Require some minimum sample per regime (soft fail by just skipping)
        if count < min_per_regime:
            continue

        mass = float(current_regime_probs.get(reg, 0.0))
        scenario_probs[mask] = mass / count

    # If all got skipped due to min_per_regime, fall back: ignore min_per_regime
    if scenario_probs.sum() == 0:
        for reg in regime_cols:
            mask = (df["regime"] == reg).values
            count = int(mask.sum())
            if count == 0:
                continue
            mass = float(current_regime_probs.get(reg, 0.0))
            scenario_probs[mask] = mass / count

    if scenario_probs.sum() == 0:
        raise ValueError("Scenario probabilities sum to zero. Check regime probs/labels.")

    scenario_probs = scenario_probs / scenario_probs.sum()

    scenario_returns = df.drop(columns=["regime"]).values
    return scenario_returns, scenario_probs
