"""
Cross-Model Consistency Check
==============================

Ties the three coffee projects together and proves they are the same model
viewed from three angles.

THE CHAIN
---------
    1. COST OF CARRY  turns a spot price into a futures price:

           F = S * exp(b * T),      b = r + d - y

    2. BLACK-76  prices an option on that futures contract:

           C = exp(-r*T) * [ F * N(d1) - X * N(d2) ]
           d1 = [ ln(F / X) + (sigma^2 / 2) * T ] / (sigma * sqrt(T))
           d2 = d1 - sigma * sqrt(T)

    3. MONTE CARLO  simulates the same contract and prices it by expectation.

THE CLAIM BEING TESTED
----------------------
These are not three approximations that happen to land nearby. Two of the three
relationships are EXACT ALGEBRAIC IDENTITIES, and the third is exact in the
limit of infinite paths:

    (a) Black-76 on F  ==  generalised Black-Scholes on S with cost of carry b
        Exact. Substitute F = S*exp(b*T) into Black-76 and the two collapse to
        the same expression. Agreement should be ~1e-16, i.e. floating-point
        dust, not "close enough".

    (b) generalised Black-Scholes with b = r  ==  plain Black-Scholes
        Exact. When storage and convenience yield are zero the carry IS the
        risk-free rate, and the generalised form reduces to the textbook one.
        This is the bridge back to the coffee-black-scholes project, which
        prices this same option at $0.068358.

    (c) Monte Carlo  ==  Black-76, within Monte Carlo standard error
        Statistical. The simulated price should sit inside its own 95%
        confidence interval around the analytic answer.

WHY (a) MATTERS
---------------
A futures price already contains the carry. Feeding a futures price into SPOT
Black-Scholes therefore charges for the carry twice. Black-76 exists precisely
to avoid that, and the identity above is the proof that it does: the same
number arrives whether you carry the spot forward first and price off the
futures, or price off the spot and let the model handle the carry.

Getting this wrong is a real and common error, and it is not a small one.

A NOTE ON THE TWO MONTE CARLO ROUTES
------------------------------------
Under the risk-neutral measure a futures price is a MARTINGALE — it has zero
drift, because entering a futures contract costs nothing. So:

    simulate the FUTURES with drift 0        starting from F
    simulate the SPOT with drift b           starting from S

produce the *same terminal distribution*. Not similar: identical, path by path,
given the same random draws. The code below demonstrates that directly, which
is a cleaner way to understand why futures are driftless than any amount of
prose about measures.

This module reuses `simulate_paths` from monte_carlo.py rather than
reimplementing it, which also shows that engine was general enough to take a
drift it was never specifically written for.

NO NEW DEPENDENCIES
-------------------
The normal CDF is implemented with math.erf rather than scipy.stats.norm, so
this module adds nothing to requirements.txt. SciPy is a ~30 MB dependency and
this needs exactly one function from it.
"""

from math import erf, exp, log, sqrt

import numpy as np

from monte_carlo import simulate_paths

# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------

SPOT_PRICE = 1.20          # S     : USD per pound
STRIKE_PRICE = 1.25        # X     : strike
RISK_FREE_RATE = 0.02      # r     : 2%
STORAGE_COST = 0.01        # d     : 1% per year
CONVENIENCE_YIELD = 0.00   # y     : 0% base case
TIME_TO_MATURITY = 0.5     # T     : six months
VOLATILITY = 0.25          # sigma : 25% annualised

NUM_SIMULATIONS = 200_000  # more paths than the base project: we are trying to
NUM_STEPS = 126            # resolve a difference of a few basis points
SEED = 42


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    """Standard normal CDF, via the error function.

    N(x) = 0.5 * (1 + erf(x / sqrt(2)))

    Equivalent to scipy.stats.norm.cdf to full double precision, without the
    dependency.
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def cost_of_carry(rate, storage, convenience_yield):
    """Return the net cost of carry b = r + d - y."""
    return rate + storage - convenience_yield


def futures_price(spot, carry, maturity):
    """Cost of carry model: F = S * exp(b * T).

    This is the coffee-cost-of-carry project's output, restated in terms of
    the net carry b rather than its three components.
    """
    return spot * exp(carry * maturity)


# ---------------------------------------------------------------------------
# 1. BLACK-76  —  option on a FUTURES contract
# ---------------------------------------------------------------------------

def black76_call(futures, strike, rate, maturity, sigma):
    """Black-76 price of a European call on a futures contract.

    Note there is no `r` inside d1. That is the whole point: the futures price
    already embeds the carry, so the interest rate appears only in the final
    discount factor.
    """
    d1 = (log(futures / strike) + 0.5 * sigma ** 2 * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return exp(-rate * maturity) * (futures * _norm_cdf(d1) - strike * _norm_cdf(d2))


def black76_put(futures, strike, rate, maturity, sigma):
    """Black-76 price of a European put on a futures contract."""
    d1 = (log(futures / strike) + 0.5 * sigma ** 2 * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return exp(-rate * maturity) * (strike * _norm_cdf(-d2) - futures * _norm_cdf(-d1))


# ---------------------------------------------------------------------------
# 2. GENERALISED BLACK-SCHOLES  —  option on SPOT, with a cost of carry
# ---------------------------------------------------------------------------

def generalised_bs_call(spot, strike, rate, carry, maturity, sigma):
    """Merton's generalised Black-Scholes, parameterised by cost of carry b.

        C = S * exp((b - r)*T) * N(d1) - X * exp(-r*T) * N(d2)
        d1 = [ ln(S/X) + (b + sigma^2/2) * T ] / (sigma * sqrt(T))

    Setting b = r recovers plain Black-Scholes.
    Setting b = 0 recovers Black-76 (with S read as the futures price).
    One formula, three models.
    """
    d1 = (log(spot / strike) + (carry + 0.5 * sigma ** 2) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return (spot * exp((carry - rate) * maturity) * _norm_cdf(d1)
            - strike * exp(-rate * maturity) * _norm_cdf(d2))


def black_scholes_call(spot, strike, rate, maturity, sigma):
    """Plain Black-Scholes — the special case b = r.

    This is what the coffee-black-scholes project computes.
    """
    return generalised_bs_call(spot, strike, rate, rate, maturity, sigma)


# ---------------------------------------------------------------------------
# 3. MONTE CARLO  —  two routes to the same distribution
# ---------------------------------------------------------------------------

def _discounted_call_payoff(terminal, strike, rate, maturity):
    """Return (price, standard_error) from an array of terminal prices."""
    payoffs = np.maximum(terminal - strike, 0.0)
    discount = exp(-rate * maturity)
    price = discount * payoffs.mean()
    stderr = discount * payoffs.std(ddof=1) / sqrt(len(payoffs))
    return price, stderr


def mc_call_on_futures(futures, strike, rate, maturity, sigma,
                       num_steps=NUM_STEPS, num_sims=NUM_SIMULATIONS, seed=SEED):
    """Price the call by simulating the FUTURES price with ZERO drift.

    A futures price is a martingale under the risk-neutral measure, so the
    drift is 0 — not r. Using r here is a common and expensive mistake.
    """
    paths = simulate_paths(futures, 0.0, sigma, maturity, num_steps, num_sims, seed)
    return _discounted_call_payoff(paths[-1], strike, rate, maturity)


def mc_call_on_spot(spot, strike, rate, carry, maturity, sigma,
                    num_steps=NUM_STEPS, num_sims=NUM_SIMULATIONS, seed=SEED):
    """Price the call by simulating the SPOT price with drift b.

    Given the same seed, the terminal prices from this function and from
    mc_call_on_futures are identical to the last bit — see
    `terminal_distributions_match` below.
    """
    paths = simulate_paths(spot, carry, sigma, maturity, num_steps, num_sims, seed)
    return _discounted_call_payoff(paths[-1], strike, rate, maturity)


def terminal_distributions_match(spot, carry, maturity, sigma,
                                 num_steps=NUM_STEPS, num_sims=10_000, seed=SEED):
    """Return the max absolute difference between the two simulation routes.

    Simulating the spot at drift b from S, and the futures at drift 0 from
    S*exp(b*T), give the same terminal array. This returns how far apart they
    actually are, which should be at the level of floating-point rounding.
    """
    fut = futures_price(spot, carry, maturity)
    spot_paths = simulate_paths(spot, carry, sigma, maturity, num_steps, num_sims, seed)
    fut_paths = simulate_paths(fut, 0.0, sigma, maturity, num_steps, num_sims, seed)
    return float(np.max(np.abs(spot_paths[-1] - fut_paths[-1])))


# ---------------------------------------------------------------------------
# RECONCILIATION
# ---------------------------------------------------------------------------

def reconcile(spot=SPOT_PRICE, strike=STRIKE_PRICE, rate=RISK_FREE_RATE,
              storage=STORAGE_COST, convenience_yield=CONVENIENCE_YIELD,
              maturity=TIME_TO_MATURITY, sigma=VOLATILITY,
              num_sims=NUM_SIMULATIONS, seed=SEED):
    """Run every route and return a dict of results."""
    b = cost_of_carry(rate, storage, convenience_yield)
    f = futures_price(spot, b, maturity)

    b76 = black76_call(f, strike, rate, maturity, sigma)
    gbs = generalised_bs_call(spot, strike, rate, b, maturity, sigma)

    mc_f, se_f = mc_call_on_futures(f, strike, rate, maturity, sigma,
                                    num_sims=num_sims, seed=seed)
    mc_s, se_s = mc_call_on_spot(spot, strike, rate, b, maturity, sigma,
                                 num_sims=num_sims, seed=seed)

    return {
        "carry": b,
        "futures": f,
        "black76": b76,
        "generalised_bs": gbs,
        "mc_futures": mc_f,
        "mc_futures_se": se_f,
        "mc_spot": mc_s,
        "mc_spot_se": se_s,
        "identity_gap": gbs - b76,
        "path_gap": terminal_distributions_match(spot, b, maturity, sigma, seed=seed),
    }


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def main():
    r = reconcile()
    b, f = r["carry"], r["futures"]
    b76 = r["black76"]

    print("=" * 68)
    print("CROSS-MODEL CONSISTENCY — COST OF CARRY -> BLACK-76 -> MONTE CARLO")
    print("=" * 68)
    print(f"Spot              S     = ${SPOT_PRICE:.4f} / lb")
    print(f"Strike            X     = ${STRIKE_PRICE:.4f} / lb")
    print(f"Risk-free rate    r     = {RISK_FREE_RATE:.2%}")
    print(f"Storage cost      d     = {STORAGE_COST:.2%}")
    print(f"Convenience yield y     = {CONVENIENCE_YIELD:.2%}")
    print(f"Maturity          T     = {TIME_TO_MATURITY} years")
    print(f"Volatility        sigma = {VOLATILITY:.2%}")
    print("-" * 68)
    print(f"Step 1  cost of carry   b = r + d - y = {b:.4f}")
    print(f"Step 2  futures price   F = S*exp(b*T) = ${f:.6f}")
    print()

    print("Step 3  the same option, priced four ways")
    print("-" * 68)
    print(f"  {'Route':<34} {'Price':>12} {'vs Black-76':>16}")
    print(f"  {'Black-76 on F':<34} {b76:>12.8f} {'(reference)':>16}")
    print(f"  {'Generalised BS on S, carry b':<34} {r['generalised_bs']:>12.8f} "
          f"{r['generalised_bs'] - b76:>16.2e}")
    print(f"  {'Monte Carlo on F, drift 0':<34} {r['mc_futures']:>12.8f} "
          f"{r['mc_futures'] - b76:>16.2e}")
    print(f"  {'Monte Carlo on S, drift b':<34} {r['mc_spot']:>12.8f} "
          f"{r['mc_spot'] - b76:>16.2e}")
    print()

    lo = r["mc_futures"] - 1.96 * r["mc_futures_se"]
    hi = r["mc_futures"] + 1.96 * r["mc_futures_se"]
    inside = lo < b76 < hi
    print(f"  Monte Carlo standard error   ${r['mc_futures_se']:.8f}")
    print(f"  Monte Carlo 95% CI           [${lo:.6f}, ${hi:.6f}]")
    print(f"  Black-76 inside the CI?      {'YES' if inside else 'NO'}")
    print()

    print("Exactness checks")
    print("-" * 68)
    print(f"  Black-76 vs generalised BS   {abs(r['identity_gap']):.2e}   "
          f"(algebraic identity, expect ~1e-16)")
    print(f"  Two simulation routes        {r['path_gap']:.2e}   "
          f"(same paths, expect ~1e-16)")
    print()

    # The bridge back to the coffee-black-scholes project.
    print("Bridge to the plain Black-Scholes project (d = y = 0)")
    print("-" * 68)
    b_zero = cost_of_carry(RISK_FREE_RATE, 0.0, 0.0)
    f_zero = futures_price(SPOT_PRICE, b_zero, TIME_TO_MATURITY)
    b76_zero = black76_call(f_zero, STRIKE_PRICE, RISK_FREE_RATE,
                            TIME_TO_MATURITY, VOLATILITY)
    bs_plain = black_scholes_call(SPOT_PRICE, STRIKE_PRICE, RISK_FREE_RATE,
                                  TIME_TO_MATURITY, VOLATILITY)
    print(f"  With no storage and no convenience yield, b = r = {b_zero:.2%}")
    print(f"  F = ${f_zero:.6f}")
    print(f"  Black-76 on that F           ${b76_zero:.6f}")
    print(f"  Plain Black-Scholes on S     ${bs_plain:.6f}")
    print(f"  coffee-black-scholes says    $0.068358")
    print(f"  Gap                          {abs(b76_zero - bs_plain):.2e}")
    print()

    # The error this whole module exists to warn about.
    print("The mistake this guards against")
    print("-" * 68)
    wrong = black_scholes_call(f, STRIKE_PRICE, RISK_FREE_RATE,
                               TIME_TO_MATURITY, VOLATILITY)
    print(f"  Correct   (Black-76 on F)              ${b76:.6f}")
    print(f"  Wrong     (spot Black-Scholes on F)    ${wrong:.6f}")
    print(f"  Overpriced by                          ${wrong - b76:.6f}  "
          f"({(wrong / b76 - 1):.2%})")
    print(f"  Feeding a futures price into spot Black-Scholes charges for the")
    print(f"  carry twice. On one ICE contract (37,500 lb) that is "
          f"${(wrong - b76) * 37_500:,.2f}.")
    print("=" * 68)


if __name__ == "__main__":
    main()
