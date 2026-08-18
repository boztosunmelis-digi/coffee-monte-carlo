"""
Monte Carlo Simulation — Coffee Price Paths
============================================

Simulates future coffee price paths under Geometric Brownian Motion, then uses
them to (a) build a distribution of outcomes, (b) price the same option as the
Black-Scholes project, and (c) stress-test a supply shock scenario.

THE MODEL
---------
Under GBM, the price evolves as:

    dS = mu * S * dt + sigma * S * dW

The exact solution, which is what we simulate (no discretisation error in S):

    S(t + dt) = S(t) * exp( (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z )

    where Z ~ N(0, 1) drawn independently at each step.

TWO THINGS THAT TRIP PEOPLE UP
------------------------------
1. The  -sigma^2 / 2  term is the Ito correction. It is there because the
   EXPECTATION of a lognormal is not the exponential of the expectation. Drop
   it and your simulated mean drifts too high, and your Monte Carlo option
   price will not match Black-Scholes.

2. For PRICING you must set mu = r (the risk-neutral drift), not the asset's
   real expected return. For FORECASTING a real-world distribution you would
   use the real drift. Same engine, different mu, completely different question.

HOW TO USE THIS FILE
--------------------
Fill in every block marked "TODO". Then run:

    python monte_carlo.py

and check your output against the target in README.md.
"""

import numpy as np

# ---------------------------------------------------------------------------
# INPUTS  (hardcoded — see README for where each number comes from)
# ---------------------------------------------------------------------------

SPOT_PRICE = 1.20          # S0    : USD per pound
RISK_FREE_RATE = 0.02      # r     : 2%, used as the risk-neutral drift
VOLATILITY = 0.25          # sigma : 25% annualised
TIME_HORIZON = 0.5         # T     : six months
NUM_SIMULATIONS = 10_000   # number of independent paths
NUM_STEPS = 126            # trading days in six months (252 / 2)
STRIKE_PRICE = 1.25        # X     : for the option pricing cross-check
SEED = 42                  # fixed seed so results are reproducible


# ---------------------------------------------------------------------------
# THE SIMULATION ENGINE
# ---------------------------------------------------------------------------

def simulate_paths(spot, drift, sigma, horizon, num_steps, num_sims, seed=None):
    """Simulate GBM price paths.

    Returns
    -------
    np.ndarray of shape (num_steps + 1, num_sims)
        Row 0 is the starting price, row `num_steps` is the price at `horizon`.

    NOTE ON SHAPE: the array has num_steps + 1 ROWS, not num_steps. You need one
    row for the initial price plus one per time step. Allocating only num_steps
    rows is a common off-by-one that silently simulates a slightly shorter
    horizon than you intended — and it will show up as a small but stubborn
    mismatch against the Black-Scholes price.
    """
    rng = np.random.default_rng(seed)
    dt = horizon / num_steps

    paths = np.zeros((num_steps + 1, num_sims))
    paths[0] = spot

    drift_term = (drift - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    for t in range(1, num_steps + 1):
        z = rng.standard_normal(num_sims)
        paths[t] = paths[t - 1] * np.exp(drift_term + diffusion * z)

    return paths


# ---------------------------------------------------------------------------
# ANALYSIS OF THE TERMINAL DISTRIBUTION
# ---------------------------------------------------------------------------

def summarise_terminal(paths):
    """Return a dict describing the distribution of prices at maturity.

    Keys: mean, median, std, p05, p25, p75, p95, min, max
    """
    terminal = paths[-1]

    return {
        "mean": np.mean(terminal),
        "median": np.median(terminal),
        "std": np.std(terminal),
        "p05": np.percentile(terminal, 5),
        "p25": np.percentile(terminal, 25),
        "p75": np.percentile(terminal, 75),
        "p95": np.percentile(terminal, 95),
        "min": terminal.min(),
        "max": terminal.max(),
    }


def value_at_risk(paths, spot, confidence=0.95):
    """Return the VaR of a LONG position, as a positive loss number.

    At 95% confidence this is the loss you exceed 5% of the time:

        VaR = spot - percentile(terminal, 5)

    Reported as a positive number, so a VaR of 0.18 means "5% of the time you
    lose more than 18 cents per pound".
    """
    terminal = paths[-1]
    threshold = np.percentile(terminal, (1 - confidence) * 100)
    return spot - threshold


def expected_shortfall(paths, spot, confidence=0.95):
    """Return the average loss in the tail BEYOND the VaR threshold.

    Also called Conditional VaR. It answers "when things go badly, how badly?"
    — which is the question VaR itself refuses to answer, and the reason
    regulators moved toward ES after 2008.
    """
    terminal = paths[-1]
    threshold = np.percentile(terminal, (1 - confidence) * 100)

    tail_mean = terminal[terminal <= threshold].mean()
    return spot - tail_mean


# ---------------------------------------------------------------------------
# OPTION PRICING BY SIMULATION  (the cross-check against Black-Scholes)
# ---------------------------------------------------------------------------

def mc_call_price(paths, strike, rate, horizon):
    """Price a European call by Monte Carlo.

    The risk-neutral pricing formula is:

        C = exp(-r*T) * E[ max(S_T - X, 0) ]

    So: take the payoff on every path, average them, discount back.

    This is the single most valuable function in the project, because you can
    check it against the closed-form Black-Scholes answer. If they agree to
    within Monte Carlo error, your simulation engine is correct.
    """
    terminal = paths[-1]

    payoffs = np.maximum(terminal - strike, 0)
    return np.exp(-rate * horizon) * payoffs.mean()


def mc_standard_error(paths, strike, rate, horizon):
    """Return the standard error of the Monte Carlo call price estimate.

        SE = exp(-r*T) * std(payoffs) / sqrt(n)

    Quoting a Monte Carlo number without its standard error is like quoting a
    poll without a margin of error. The 95% confidence interval is
    roughly price +/- 1.96 * SE.

    Note the sqrt(n): to halve the error you need FOUR times the paths. This is
    why variance reduction techniques (antithetic variates, control variates)
    matter in practice.
    """
    terminal = paths[-1]

    payoffs = np.maximum(terminal - strike, 0)
    return np.exp(-rate * horizon) * payoffs.std(ddof=1) / np.sqrt(len(payoffs))


# ---------------------------------------------------------------------------
# SCENARIO ANALYSIS  (the part the brief actually asks for)
# ---------------------------------------------------------------------------

# Each scenario adjusts the drift and volatility to represent a market regime.
# A frost in Minas Gerais is a positive price shock with much higher vol; a
# bumper harvest is the mirror image.
SCENARIOS = {
    "Base case":            {"drift_adj": 0.00, "vol_mult": 1.0},
    "Frost / supply shock": {"drift_adj": 0.30, "vol_mult": 1.8},
    "Bumper harvest":       {"drift_adj": -0.20, "vol_mult": 1.2},
    "Demand slump":         {"drift_adj": -0.10, "vol_mult": 1.1},
    "Calm market":          {"drift_adj": 0.00, "vol_mult": 0.6},
}


def run_scenarios(spot, rate, sigma, horizon, num_steps, num_sims, seed=None):
    """Run the simulation once per scenario.

    Returns a dict: scenario name -> {"mean", "p05", "p95", "prob_above_spot"}
    """
    results = {}
    for name, cfg in SCENARIOS.items():
        drift = rate + cfg["drift_adj"]
        vol = sigma * cfg["vol_mult"]
        paths = simulate_paths(spot, drift, vol, horizon, num_steps, num_sims, seed)
        terminal = paths[-1]
        results[name] = {
            "mean": terminal.mean(),
            "p05": np.percentile(terminal, 5),
            "p95": np.percentile(terminal, 95),
            "prob_above_spot": (terminal > spot).mean(),
        }
    return results


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def main():
    paths = simulate_paths(
        spot=SPOT_PRICE,
        drift=RISK_FREE_RATE,
        sigma=VOLATILITY,
        horizon=TIME_HORIZON,
        num_steps=NUM_STEPS,
        num_sims=NUM_SIMULATIONS,
        seed=SEED,
    )

    print("=" * 64)
    print("MONTE CARLO — 6M COFFEE PRICE SIMULATION")
    print("=" * 64)
    print(f"Spot        S0    = ${SPOT_PRICE:.4f} / lb")
    print(f"Drift       mu    = {RISK_FREE_RATE:.2%}  (risk-neutral)")
    print(f"Volatility  sigma = {VOLATILITY:.2%}")
    print(f"Horizon     T     = {TIME_HORIZON} years, {NUM_STEPS} steps")
    print(f"Paths             = {NUM_SIMULATIONS:,}")
    print(f"Array shape       = {paths.shape}")
    print()

    s = summarise_terminal(paths)
    print("Terminal price distribution")
    print("-" * 64)
    print(f"  mean            ${s['mean']:.4f}")
    print(f"  median          ${s['median']:.4f}")
    print(f"  std dev         ${s['std']:.4f}")
    print(f"  5th pctile      ${s['p05']:.4f}")
    print(f"  25th pctile     ${s['p25']:.4f}")
    print(f"  75th pctile     ${s['p75']:.4f}")
    print(f"  95th pctile     ${s['p95']:.4f}")
    print(f"  min / max       ${s['min']:.4f} / ${s['max']:.4f}")
    print()
    print(f"  Theory says the mean should be S0*exp(r*T) = "
          f"${SPOT_PRICE * np.exp(RISK_FREE_RATE * TIME_HORIZON):.4f}")
    print(f"  The median sits BELOW the mean — that is the lognormal skew.")
    print()

    print("Risk measures (long 1 lb)")
    print("-" * 64)
    print(f"  95% VaR                 ${value_at_risk(paths, SPOT_PRICE):.4f} / lb")
    print(f"  95% Expected Shortfall  ${expected_shortfall(paths, SPOT_PRICE):.4f} / lb")
    print()

    mc = mc_call_price(paths, STRIKE_PRICE, RISK_FREE_RATE, TIME_HORIZON)
    se = mc_standard_error(paths, STRIKE_PRICE, RISK_FREE_RATE, TIME_HORIZON)
    print(f"Option cross-check (call, X = ${STRIKE_PRICE:.2f})")
    print("-" * 64)
    print(f"  Monte Carlo price   ${mc:.6f}")
    print(f"  Standard error      ${se:.6f}")
    print(f"  95% CI              [${mc - 1.96 * se:.6f}, ${mc + 1.96 * se:.6f}]")
    print(f"  Black-Scholes says  $0.068358   <-- must fall inside the CI")
    print()

    print("Scenario analysis")
    print("-" * 64)
    print(f"  {'Scenario':<22} {'Mean':>9} {'5%':>9} {'95%':>9} {'P(up)':>8}")
    for name, r in run_scenarios(
        SPOT_PRICE, RISK_FREE_RATE, VOLATILITY, TIME_HORIZON,
        NUM_STEPS, NUM_SIMULATIONS, SEED,
    ).items():
        print(f"  {name:<22} {r['mean']:>9.4f} {r['p05']:>9.4f} "
              f"{r['p95']:>9.4f} {r['prob_above_spot']:>7.1%}")
    print("=" * 64)


if __name__ == "__main__":
    main()
