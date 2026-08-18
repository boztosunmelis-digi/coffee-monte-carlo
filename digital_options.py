"""
Digital (Binary) Options on Coffee Futures
===========================================

Prices cash-or-nothing and asset-or-nothing digital options under Black-76,
computes their Greeks, and quantifies the hedging problem that makes them
genuinely difficult to sell.

THE PAYOFFS
-----------
    Cash-or-nothing call    pays Q if F_T >  X,  else 0
    Cash-or-nothing put     pays Q if F_T <  X,  else 0
    Asset-or-nothing call   pays F_T if F_T > X, else 0

THE PRICES (Black-76, on a futures underlying)
----------------------------------------------
    cash call  = Q * exp(-r*T) * N(d2)
    cash put   = Q * exp(-r*T) * N(-d2)
    asset call = exp(-r*T) * F * N(d1)

    d1 = [ ln(F/X) + (sigma^2/2)*T ] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

THE INTERPRETATION THAT MAKES DIGITALS USEFUL
---------------------------------------------
A cash-or-nothing call paying $1 is worth exp(-r*T) * N(d2). Strip out the
discount factor and the price IS the risk-neutral probability of finishing in
the money. Nothing else in derivatives reads back a probability that directly.

That makes digitals the natural instrument for a client with a VIEW ON AN
EVENT rather than a view on magnitude: "will coffee be above $1.25 in six
months" pays the same whether it settles at $1.26 or $2.60.

TWO STRUCTURAL FACTS THAT DOMINATE THE RISK
-------------------------------------------
1. VANILLA = ASSET-OR-NOTHING - X * CASH-OR-NOTHING
   A standard call decomposes exactly into the two digitals. This is not a
   curiosity: it means digitals are the building blocks, and it gives an exact
   test of the implementation.

2. A DIGITAL IS THE DERIVATIVE OF A VANILLA WITH RESPECT TO STRIKE
       cash call (Q=1) = -dC_vanilla / dX
   Which is why a digital can be REPLICATED by a tight call spread, and why
   its gamma explodes near the strike close to expiry. Section
   `call_spread_replication` below prices the hedge an issuer would actually
   run, and `pin_risk_profile` measures how bad the unhedged position gets.

NO NEW DEPENDENCIES
-------------------
Normal CDF and PDF are built from math.erf and math.exp, so requirements.txt
is unchanged.
"""

from math import erf, exp, log, pi, sqrt

import numpy as np

from monte_carlo import simulate_paths

# ---------------------------------------------------------------------------
# INPUTS  (identical to the valuation report, so the numbers tie out)
# ---------------------------------------------------------------------------

SPOT_PRICE = 1.20          # S     : USD per pound
STRIKE_PRICE = 1.25        # X     : strike / barrier
RISK_FREE_RATE = 0.02      # r     : 2%
STORAGE_COST = 0.01        # d     : 1% per year
CONVENIENCE_YIELD = 0.00   # y     : 0% base case
TIME_TO_MATURITY = 0.5     # T     : six months
VOLATILITY = 0.25          # sigma : 25% annualised

PAYOUT = 10_000.0          # Q     : cash settlement amount, USD
NUM_SIMULATIONS = 200_000
NUM_STEPS = 126
SEED = 42

# Scenario definitions, matching monte_carlo.SCENARIOS
SCENARIOS = {
    "Base case":            {"drift_adj": 0.00, "vol_mult": 1.0},
    "Frost / supply shock": {"drift_adj": 0.30, "vol_mult": 1.8},
    "Bumper harvest":       {"drift_adj": -0.20, "vol_mult": 1.2},
    "Demand slump":         {"drift_adj": -0.10, "vol_mult": 1.1},
    "Calm market":          {"drift_adj": 0.00, "vol_mult": 0.6},
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(futures, strike, maturity, sigma):
    d1 = (log(futures / strike) + 0.5 * sigma ** 2 * maturity) / (sigma * sqrt(maturity))
    return d1, d1 - sigma * sqrt(maturity)


def cost_of_carry(rate, storage, convenience_yield):
    return rate + storage - convenience_yield


def futures_price(spot, carry, maturity):
    return spot * exp(carry * maturity)


# ---------------------------------------------------------------------------
# PRICING
# ---------------------------------------------------------------------------

def digital_call(futures, strike, rate, maturity, sigma, payout=1.0):
    """Cash-or-nothing digital call: pays `payout` if F_T > X."""
    _, d2 = _d1_d2(futures, strike, maturity, sigma)
    return payout * exp(-rate * maturity) * _norm_cdf(d2)


def digital_put(futures, strike, rate, maturity, sigma, payout=1.0):
    """Cash-or-nothing digital put: pays `payout` if F_T < X."""
    _, d2 = _d1_d2(futures, strike, maturity, sigma)
    return payout * exp(-rate * maturity) * _norm_cdf(-d2)


def asset_or_nothing_call(futures, strike, rate, maturity, sigma):
    """Pays the futures price itself if F_T > X, else nothing."""
    d1, _ = _d1_d2(futures, strike, maturity, sigma)
    return exp(-rate * maturity) * futures * _norm_cdf(d1)


def vanilla_call(futures, strike, rate, maturity, sigma):
    """Standard Black-76 call, for the decomposition check."""
    d1, d2 = _d1_d2(futures, strike, maturity, sigma)
    return exp(-rate * maturity) * (futures * _norm_cdf(d1) - strike * _norm_cdf(d2))


def risk_neutral_probability_itm(futures, strike, maturity, sigma):
    """N(d2): the risk-neutral probability the call finishes in the money.

    This is the digital call price with the discount factor stripped out.
    """
    _, d2 = _d1_d2(futures, strike, maturity, sigma)
    return _norm_cdf(d2)


# ---------------------------------------------------------------------------
# GREEKS  (cash-or-nothing call)
# ---------------------------------------------------------------------------

def digital_call_greeks(futures, strike, rate, maturity, sigma, payout=1.0):
    """Return the Greeks of a cash-or-nothing digital call.

        delta = e^(-rT) * n(d2) / (F * sigma * sqrt(T))
        gamma = -e^(-rT) * n(d2) * d1 / (F^2 * sigma^2 * T)
        vega  = -e^(-rT) * n(d2) * d1 / sigma          (per 1 vol point / 100)
        theta = -dV/dT                                  (per calendar day)
        rho   = -T * V                                  (per 1 rate point / 100)

    TWO SIGN BEHAVIOURS WORTH NOTING, both driven by d1:

      GAMMA CHANGES SIGN at d1 = 0. Below the strike gamma is positive, above
      it is negative. A vanilla call has strictly positive gamma; a digital
      does not. That alone makes it a different hedging problem.

      VEGA CHANGES SIGN at d1 = 0 too. An out-of-the-money digital call GAINS
      from higher volatility; an in-the-money one LOSES. Volatility helps only
      while it can still carry you across the barrier.
    """
    d1, d2 = _d1_d2(futures, strike, maturity, sigma)
    df = exp(-rate * maturity)
    n2 = _norm_pdf(d2)
    root_t = sqrt(maturity)

    delta = payout * df * n2 / (futures * sigma * root_t)
    gamma = -payout * df * n2 * d1 / (futures ** 2 * sigma ** 2 * maturity)
    vega = -payout * df * n2 * d1 / sigma

    # dV/dT, then negate and convert to per-day decay
    dd2_dt = -log(futures / strike) / (2 * sigma * maturity ** 1.5) - sigma / (4 * root_t)
    dv_dt = payout * df * (-rate * _norm_cdf(d2) + n2 * dd2_dt)
    theta = -dv_dt

    value = payout * df * _norm_cdf(d2)
    rho = -maturity * value

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,      # per 1 volatility point
        "theta": theta / 365,    # per calendar day
        "rho": rho / 100,        # per 1 rate point
    }


# ---------------------------------------------------------------------------
# STATIC REPLICATION  —  how an issuer actually hedges this
# ---------------------------------------------------------------------------

def call_spread_replication(futures, strike, rate, maturity, sigma,
                            width, payout=1.0):
    """Replicate the digital with a bull call spread of half-width `width`.

    Buy  1/(2w) calls struck at X - w
    Sell 1/(2w) calls struck at X + w

    The spread pays 0 below X-w, `payout` above X+w, and ramps linearly
    between. As w -> 0 it converges to the digital, because a digital IS the
    negative strike-derivative of a vanilla.

    A desk does NOT run w -> 0. A finite width is what makes the position
    hedgeable at all: it converts an infinite gamma at the barrier into a
    finite one. The price of that safety is the OVER-HEDGE COST returned here,
    which is the difference between the spread and the true digital value.

    Returns a dict with the spread cost, the digital value, and the overhedge.
    """
    n = payout / (2.0 * width)
    long_leg = vanilla_call(futures, strike - width, rate, maturity, sigma)
    short_leg = vanilla_call(futures, strike + width, rate, maturity, sigma)
    spread = n * (long_leg - short_leg)
    exact = digital_call(futures, strike, rate, maturity, sigma, payout)
    return {
        "width": width,
        "spread_cost": spread,
        "digital_value": exact,
        "overhedge": spread - exact,
        "overhedge_pct": spread / exact - 1.0,
        "max_delta": n,  # worst-case futures position the spread ever requires
    }


def pin_risk_profile(futures, strike, rate, sigma, maturities, payout=1.0):
    """Show how digital delta and gamma behave as expiry approaches at the strike.

    This is the number that explains why digitals are hard. Delta is the size
    of the futures hedge required per unit of payout; as T -> 0 with F at the
    barrier it diverges. No finite hedge exists at expiry.
    """
    rows = []
    for t in maturities:
        g = digital_call_greeks(futures, strike, rate, t, sigma, payout)
        rows.append((t, g["delta"], g["gamma"]))
    return rows


# ---------------------------------------------------------------------------
# MONTE CARLO  —  validation and scenario probabilities
# ---------------------------------------------------------------------------

def mc_digital_call(futures, strike, rate, maturity, sigma, payout=1.0,
                    num_steps=NUM_STEPS, num_sims=NUM_SIMULATIONS, seed=SEED):
    """Price the digital by simulation. Returns (price, standard_error).

    The futures price is simulated driftless, because a futures price is a
    martingale under the risk-neutral measure.
    """
    paths = simulate_paths(futures, 0.0, sigma, maturity, num_steps, num_sims, seed)
    payoffs = np.where(paths[-1] > strike, payout, 0.0)
    df = exp(-rate * maturity)
    price = df * payoffs.mean()
    stderr = df * payoffs.std(ddof=1) / sqrt(num_sims)
    return price, stderr


def scenario_probabilities(spot, strike, rate, storage, convenience_yield,
                           maturity, sigma, payout=PAYOUT,
                           num_sims=50_000, seed=SEED):
    """Probability of finishing above the barrier under each market scenario.

    Note these are REAL-WORLD probabilities under each scenario's assumed
    drift, not risk-neutral probabilities. They answer "what might happen",
    not "what is it worth". Mixing the two up is a classic error.
    """
    b = cost_of_carry(rate, storage, convenience_yield)
    out = {}
    for name, cfg in SCENARIOS.items():
        drift = b + cfg["drift_adj"]
        vol = sigma * cfg["vol_mult"]
        f = futures_price(spot, drift, maturity)
        paths = simulate_paths(spot, drift, vol, maturity, NUM_STEPS, num_sims, seed)
        terminal = paths[-1]
        prob = float((terminal > strike).mean())
        out[name] = {
            "futures": f,
            "vol": vol,
            "prob_above": prob,
            "expected_payout": prob * payout,
            "digital_value": digital_call(f, strike, rate, maturity, vol, payout),
        }
    return out


def barrier_ladder(futures, rate, maturity, sigma, strikes, payout=PAYOUT):
    """Digital call value and implied probability across a range of barriers."""
    return [
        (
            k,
            risk_neutral_probability_itm(futures, k, maturity, sigma),
            digital_call(futures, k, rate, maturity, sigma, payout),
        )
        for k in strikes
    ]


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def main():
    b = cost_of_carry(RISK_FREE_RATE, STORAGE_COST, CONVENIENCE_YIELD)
    f = futures_price(SPOT_PRICE, b, TIME_TO_MATURITY)
    args = (f, STRIKE_PRICE, RISK_FREE_RATE, TIME_TO_MATURITY, VOLATILITY)
    d1, d2 = _d1_d2(f, STRIKE_PRICE, TIME_TO_MATURITY, VOLATILITY)

    print("=" * 70)
    print("DIGITAL OPTIONS ON COFFEE FUTURES — SCENARIO 3, HIGH-RISK INVESTOR")
    print("=" * 70)
    print(f"Spot            S = ${SPOT_PRICE:.4f}    Carry b = {b:.4f}")
    print(f"Futures         F = ${f:.6f}")
    print(f"Barrier         X = ${STRIKE_PRICE:.4f}")
    print(f"Maturity        T = {TIME_TO_MATURITY} yr   Volatility = {VOLATILITY:.2%}")
    print(f"Cash payout     Q = ${PAYOUT:,.2f}")
    print(f"d1 = {d1:+.6f}   d2 = {d2:+.6f}")
    print()

    dc_unit = digital_call(*args)
    dp_unit = digital_put(*args)
    dc = digital_call(*args, payout=PAYOUT)
    dp = digital_put(*args, payout=PAYOUT)
    prob = risk_neutral_probability_itm(f, STRIKE_PRICE, TIME_TO_MATURITY, VOLATILITY)

    print("Pricing")
    print("-" * 70)
    print(f"  Risk-neutral P(F_T > X) = N(d2)     {prob:.6f}   ({prob:.2%})")
    print(f"  Digital call, $1 payout             ${dc_unit:.6f}")
    print(f"  Digital put,  $1 payout             ${dp_unit:.6f}")
    print(f"  Sum (must equal e^-rT)              ${dc_unit + dp_unit:.6f}"
          f"   vs {exp(-RISK_FREE_RATE * TIME_TO_MATURITY):.6f}")
    print()
    print(f"  Digital CALL premium, ${PAYOUT:,.0f} payout   ${dc:,.2f}")
    print(f"  Digital PUT  premium, ${PAYOUT:,.0f} payout   ${dp:,.2f}")
    print(f"  Call: max gain ${PAYOUT - dc:,.2f}  /  max loss ${dc:,.2f}")
    print(f"  Breakeven probability needed        {dc / PAYOUT:.2%}")
    print()

    print("Decomposition check: vanilla = asset-or-nothing - X * cash-or-nothing")
    print("-" * 70)
    aon = asset_or_nothing_call(*args)
    con = digital_call(*args)
    van = vanilla_call(*args)
    print(f"  Asset-or-nothing call               ${aon:.8f}")
    print(f"  X * cash-or-nothing call            ${STRIKE_PRICE * con:.8f}")
    print(f"  Difference                          ${aon - STRIKE_PRICE * con:.8f}")
    print(f"  Black-76 vanilla call               ${van:.8f}")
    print(f"  Gap                                 {abs(aon - STRIKE_PRICE * con - van):.2e}")
    print()

    print("Greeks (digital call, per $1 of payout)")
    print("-" * 70)
    g = digital_call_greeks(*args)
    for k in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"  {k:6s} {g[k]:+.6f}")
    print(f"  Scaled to ${PAYOUT:,.0f} payout:")
    gq = digital_call_greeks(*args, payout=PAYOUT)
    print(f"    delta ${gq['delta']:,.2f} of futures per $1 move")
    print(f"    vega  ${gq['vega']:,.2f} per +1 vol point")
    print(f"    theta ${gq['theta']:,.2f} per day")
    print()

    print("Monte Carlo validation")
    print("-" * 70)
    mc, se = mc_digital_call(*args, payout=PAYOUT)
    print(f"  Monte Carlo price     ${mc:,.2f}")
    print(f"  Standard error        ${se:,.2f}")
    print(f"  95% CI                [${mc - 1.96 * se:,.2f}, ${mc + 1.96 * se:,.2f}]")
    print(f"  Analytic price        ${dc:,.2f}")
    print(f"  Inside CI?            "
          f"{'YES' if mc - 1.96 * se < dc < mc + 1.96 * se else 'NO'}")
    print()

    print("Static replication by call spread (issuer's actual hedge)")
    print("-" * 70)
    print(f"  {'Half-width':>12} {'Spread cost':>14} {'Overhedge':>12} "
          f"{'Overhedge %':>13} {'Max delta':>12}")
    for w in (0.10, 0.05, 0.025, 0.01, 0.005, 0.001):
        rep = call_spread_replication(*args, width=w, payout=PAYOUT)
        print(f"  {w:>12.4f} ${rep['spread_cost']:>13,.2f} "
              f"${rep['overhedge']:>11,.2f} {rep['overhedge_pct']:>12.2%} "
              f"{rep['max_delta']:>12,.0f}")
    print("  Narrower spread tracks the digital better but requires a larger")
    print("  futures position to hedge. That is the trade-off, priced.")
    print()

    print("Pin risk: digital delta at the barrier as expiry approaches")
    print("-" * 70)
    print(f"  {'T (years)':>12} {'Days':>7} {'Delta':>14} {'Gamma':>16}")
    for t, delta, gamma in pin_risk_profile(
        STRIKE_PRICE, STRIKE_PRICE, RISK_FREE_RATE, VOLATILITY,
        [0.5, 0.25, 0.08, 0.02, 0.004, 0.0004], payout=PAYOUT,
    ):
        print(f"  {t:>12.4f} {t * 365:>7.1f} {delta:>14,.2f} {gamma:>16,.0f}")
    print("  Delta diverges as T -> 0. At expiry no finite hedge exists.")
    print()

    print("Barrier ladder (risk-neutral)")
    print("-" * 70)
    print(f"  {'Barrier':>9} {'P(above)':>11} {'Premium':>12} {'Payout ratio':>14}")
    for k, p, v in barrier_ladder(f, RISK_FREE_RATE, TIME_TO_MATURITY, VOLATILITY,
                                  [1.10, 1.20, 1.25, 1.35, 1.50, 1.75]):
        tag = "  <-- proposed" if abs(k - STRIKE_PRICE) < 1e-9 else ""
        print(f"  ${k:>8.2f} {p:>10.2%} ${v:>11,.2f} {PAYOUT / v:>13.2f}x{tag}")
    print()

    print("Scenario analysis: real-world probability of finishing above barrier")
    print("-" * 70)
    print(f"  {'Scenario':<22} {'F':>9} {'Vol':>7} {'P(above)':>10} "
          f"{'E[payout]':>12} {'Value':>11}")
    for name, s in scenario_probabilities(
        SPOT_PRICE, STRIKE_PRICE, RISK_FREE_RATE, STORAGE_COST,
        CONVENIENCE_YIELD, TIME_TO_MATURITY, VOLATILITY,
    ).items():
        print(f"  {name:<22} {s['futures']:>9.4f} {s['vol']:>6.0%} "
              f"{s['prob_above']:>9.2%} ${s['expected_payout']:>11,.0f} "
              f"${s['digital_value']:>10,.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
