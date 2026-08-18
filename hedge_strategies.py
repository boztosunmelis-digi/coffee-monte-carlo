"""
Hedging Strategies for a Coffee Buyer Facing Rising Prices
==========================================================

Scenario 1 of the risk management exercise: the desk expects coffee prices to
rise on poor weather, and holds a forward purchase obligation. Rising prices
are the loss direction.

THE EXPOSURE
------------
A commitment to buy REQUIRED_VOLUME pounds of coffee in six months at whatever
the market price then is. Unhedged, every cent of price rise is a cent of cost.

    unhedged cost = F_T * volume

THE FIVE STRATEGIES COMPARED
----------------------------
    1. UNHEDGED             full exposure, no cost
    2. LONG FUTURES         locks the price exactly; gives up all benefit if
                            prices fall; no premium but margin is required
    3. LONG CALLS           caps the purchase price; keeps the benefit of a
                            fall; costs premium upfront
    4. ZERO-COST COLLAR     buy a call, sell a put struck so the premiums net
                            to zero; caps the cost AND the benefit, free
    5. CALL SPREAD          buy a call, sell a higher call; cheaper than a
                            straight call but protection runs out at the upper
                            strike

THE POINT OF THE COMPARISON
---------------------------
There is no strategy that is best on every measure. Futures give certainty and
forfeit all upside. Calls keep the upside and cost real money. The collar is
free and gives away the fall. A call spread is cheap and fails in exactly the
scenario you bought it for.

This module prices all five, computes the effective cost per pound at any
settlement price, and runs each through the five market scenarios so the
trade-offs are numbers rather than adjectives.

CONVENTION
----------
Everything is expressed as EFFECTIVE COST PER POUND to the buyer. Lower is
better. Hedge payoffs reduce cost; premiums increase it.
"""

from math import erf, exp, log, sqrt

import numpy as np

from monte_carlo import simulate_paths

# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------

SPOT_PRICE = 1.20
RISK_FREE_RATE = 0.02
STORAGE_COST = 0.01
CONVENIENCE_YIELD = 0.00
TIME_TO_MATURITY = 0.5
VOLATILITY = 0.25

CONTRACT_SIZE = 37_500          # lb, one ICE "C" contract
NUM_CONTRACTS = 10
REQUIRED_VOLUME = CONTRACT_SIZE * NUM_CONTRACTS   # 375,000 lb

CALL_STRIKE = 1.25              # protection strike
SPREAD_UPPER_STRIKE = 1.40      # where call-spread protection stops

NUM_SIMULATIONS = 100_000
NUM_STEPS = 126
SEED = 42

SCENARIOS = {
    "Base case":            {"drift_adj": 0.00, "vol_mult": 1.0},
    "Frost / supply shock": {"drift_adj": 0.30, "vol_mult": 1.8},
    "Bumper harvest":       {"drift_adj": -0.20, "vol_mult": 1.2},
    "Demand slump":         {"drift_adj": -0.10, "vol_mult": 1.1},
    "Calm market":          {"drift_adj": 0.00, "vol_mult": 0.6},
}


# ---------------------------------------------------------------------------
# PRICING PRIMITIVES  (Black-76, no new dependencies)
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _d1_d2(futures, strike, maturity, sigma):
    d1 = (log(futures / strike) + 0.5 * sigma ** 2 * maturity) / (sigma * sqrt(maturity))
    return d1, d1 - sigma * sqrt(maturity)


def cost_of_carry(rate, storage, convenience_yield):
    return rate + storage - convenience_yield


def futures_price(spot, carry, maturity):
    return spot * exp(carry * maturity)


def call_price(futures, strike, rate, maturity, sigma):
    d1, d2 = _d1_d2(futures, strike, maturity, sigma)
    return exp(-rate * maturity) * (futures * _norm_cdf(d1) - strike * _norm_cdf(d2))


def put_price(futures, strike, rate, maturity, sigma):
    d1, d2 = _d1_d2(futures, strike, maturity, sigma)
    return exp(-rate * maturity) * (strike * _norm_cdf(-d2) - futures * _norm_cdf(-d1))


# ---------------------------------------------------------------------------
# THE ZERO-COST COLLAR
# ---------------------------------------------------------------------------

def zero_cost_collar_put_strike(futures, call_strike, rate, maturity, sigma,
                                tol=1e-12, max_iter=200):
    """Find the put strike whose premium exactly funds the protective call.

    The buyer wants protection above `call_strike`. To pay for it without cash
    outlay, they sell a put — accepting that if prices FALL below that strike
    they are obliged to buy at it anyway, forfeiting the benefit.

    Solved by bisection on put_price(K) - call_price(call_strike) = 0. The put
    premium is strictly increasing in K, so the root is unique and bisection is
    both safe and sufficient; no derivative is needed.

    Returns the put strike.
    """
    target = call_price(futures, call_strike, rate, maturity, sigma)
    lo, hi = 1e-6, futures * 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = put_price(futures, mid, rate, maturity, sigma) - target
        if abs(diff) < tol:
            return mid
        if diff < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# EFFECTIVE COST PER POUND UNDER EACH STRATEGY
# ---------------------------------------------------------------------------

def cost_unhedged(terminal):
    """No hedge. Cost is whatever the market charges."""
    return terminal


def cost_futures(terminal, entry_futures):
    """Long futures. The hedge gain exactly offsets the price move.

    cost = F_T - (F_T - F_0) = F_0

    Works for arrays as well as scalars, hence the multiplication by
    np.ones_like rather than a bare constant.
    """
    return np.full_like(np.asarray(terminal, dtype=float), entry_futures)


def cost_long_call(terminal, strike, premium, rate, maturity):
    """Buy a call. Cost is capped at the strike, plus the future value of the
    premium paid today.

    The premium is compounded forward to expiry, because it was paid six months
    earlier. Ignoring that understates the cost of the hedge.
    """
    terminal = np.asarray(terminal, dtype=float)
    payoff = np.maximum(terminal - strike, 0.0)
    return terminal - payoff + premium * exp(rate * maturity)


def cost_collar(terminal, call_strike, put_strike, net_premium, rate, maturity):
    """Buy a call, sell a put. Cost is trapped between the two strikes.

    Above call_strike the long call caps the cost. Below put_strike the short
    put obliges the buyer to pay put_strike anyway. Between them the buyer pays
    the market.
    """
    terminal = np.asarray(terminal, dtype=float)
    long_call = np.maximum(terminal - call_strike, 0.0)
    short_put = np.maximum(put_strike - terminal, 0.0)
    return terminal - long_call + short_put + net_premium * exp(rate * maturity)


def cost_call_spread(terminal, lower_strike, upper_strike, net_premium,
                     rate, maturity):
    """Buy a call at the lower strike, sell one at the upper strike.

    Protection applies only between the two strikes. Above the upper strike the
    buyer is exposed again, one for one. Cheaper than a straight call and it
    fails precisely in the scenario it was bought for.
    """
    terminal = np.asarray(terminal, dtype=float)
    payoff = (np.maximum(terminal - lower_strike, 0.0)
              - np.maximum(terminal - upper_strike, 0.0))
    return terminal - payoff + net_premium * exp(rate * maturity)


# ---------------------------------------------------------------------------
# STRATEGY SET
# ---------------------------------------------------------------------------

def build_strategies(futures, rate, maturity, sigma,
                     call_strike=CALL_STRIKE,
                     spread_upper=SPREAD_UPPER_STRIKE):
    """Return a dict describing all five strategies with their upfront costs."""
    c_lower = call_price(futures, call_strike, rate, maturity, sigma)
    c_upper = call_price(futures, spread_upper, rate, maturity, sigma)
    put_strike = zero_cost_collar_put_strike(futures, call_strike, rate, maturity, sigma)
    p_collar = put_price(futures, put_strike, rate, maturity, sigma)

    return {
        "Unhedged": {
            "premium": 0.0,
            "cost_fn": lambda t: cost_unhedged(np.asarray(t, dtype=float)),
            "detail": "No position",
        },
        "Long futures": {
            "premium": 0.0,
            "cost_fn": lambda t: cost_futures(t, futures),
            "detail": f"Buy {NUM_CONTRACTS} contracts at ${futures:.6f}",
        },
        "Long calls": {
            "premium": c_lower,
            "cost_fn": lambda t: cost_long_call(t, call_strike, c_lower, rate, maturity),
            "detail": f"Buy ${call_strike:.2f} calls at ${c_lower:.6f}",
        },
        "Zero-cost collar": {
            "premium": c_lower - p_collar,
            "cost_fn": lambda t: cost_collar(t, call_strike, put_strike,
                                             c_lower - p_collar, rate, maturity),
            "detail": f"Buy ${call_strike:.2f} call, sell ${put_strike:.4f} put",
        },
        "Call spread": {
            "premium": c_lower - c_upper,
            "cost_fn": lambda t: cost_call_spread(t, call_strike, spread_upper,
                                                  c_lower - c_upper, rate, maturity),
            "detail": f"Buy ${call_strike:.2f} call, sell ${spread_upper:.2f} call",
        },
    }


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------

def cost_ladder(strategies, terminal_prices):
    """Effective cost per pound for each strategy across a grid of outcomes."""
    return {
        name: [float(np.asarray(s["cost_fn"](t)).item()) for t in terminal_prices]
        for name, s in strategies.items()
    }


def scenario_outcomes(spot, rate, storage, convenience_yield, maturity, sigma,
                      call_strike=CALL_STRIKE, spread_upper=SPREAD_UPPER_STRIKE,
                      volume=REQUIRED_VOLUME, num_sims=NUM_SIMULATIONS, seed=SEED):
    """Expected and worst-case cost per strategy under each market scenario.

    Worst case is the 95th percentile of cost, which for a BUYER is the tail
    that matters. Value at Risk for a buyer lives in the upper tail, not the
    lower one — getting that backwards is a classic sign error.
    """
    b = cost_of_carry(rate, storage, convenience_yield)
    f0 = futures_price(spot, b, maturity)
    strategies = build_strategies(f0, rate, maturity, sigma, call_strike, spread_upper)

    results = {}
    for sc_name, cfg in SCENARIOS.items():
        drift = b + cfg["drift_adj"]
        vol = sigma * cfg["vol_mult"]
        paths = simulate_paths(spot, drift, vol, maturity, NUM_STEPS, num_sims, seed)
        terminal = paths[-1]

        per_strategy = {}
        for name, s in strategies.items():
            costs = np.asarray(s["cost_fn"](terminal), dtype=float)
            per_strategy[name] = {
                "mean_cost": float(costs.mean()),
                "p95_cost": float(np.percentile(costs, 95)),
                "total_mean": float(costs.mean() * volume),
                "total_p95": float(np.percentile(costs, 95) * volume),
                "cost_std": float(costs.std()),
            }
        results[sc_name] = {
            "mean_terminal": float(terminal.mean()),
            "vol": vol,
            "strategies": per_strategy,
        }
    return results


def hedge_effectiveness(strategies, terminal, volume=REQUIRED_VOLUME):
    """Variance reduction of each hedge against the unhedged position.

        effectiveness = 1 - Var(hedged) / Var(unhedged)

    1.0 is a perfect hedge, 0.0 is no hedge at all. This is the standard
    measure, and it deliberately ignores cost — a perfect hedge that is
    expensive still scores 1.0, which is why it must be read alongside the
    cost table rather than instead of it.
    """
    base_var = float(np.var(np.asarray(strategies["Unhedged"]["cost_fn"](terminal))))
    out = {}
    for name, s in strategies.items():
        v = float(np.var(np.asarray(s["cost_fn"](terminal), dtype=float)))
        out[name] = 1.0 - v / base_var if base_var > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def main():
    b = cost_of_carry(RISK_FREE_RATE, STORAGE_COST, CONVENIENCE_YIELD)
    f0 = futures_price(SPOT_PRICE, b, TIME_TO_MATURITY)
    strategies = build_strategies(f0, RISK_FREE_RATE, TIME_TO_MATURITY, VOLATILITY)
    put_k = zero_cost_collar_put_strike(f0, CALL_STRIKE, RISK_FREE_RATE,
                                        TIME_TO_MATURITY, VOLATILITY)

    print("=" * 78)
    print("SCENARIO 1 — HEDGING MARKET RISK: STRATEGY COMPARISON")
    print("=" * 78)
    print(f"Exposure          buy {REQUIRED_VOLUME:,} lb "
          f"({NUM_CONTRACTS} ICE contracts) in 6 months")
    print(f"Spot              ${SPOT_PRICE:.4f}   Carry b = {b:.4f}")
    print(f"Futures  F        ${f0:.6f}")
    print(f"Unhedged notional ${f0 * REQUIRED_VOLUME:,.2f}")
    print(f"Volatility        {VOLATILITY:.2%}")
    print()

    print("Strategy costs, upfront")
    print("-" * 78)
    print(f"  {'Strategy':<20} {'Premium/lb':>12} {'Total premium':>16}  Detail")
    for name, s in strategies.items():
        print(f"  {name:<20} ${s['premium']:>11.6f} ${s['premium'] * REQUIRED_VOLUME:>15,.2f}"
              f"  {s['detail']}")
    print()
    print(f"  Zero-cost collar solve: put strike ${put_k:.6f}")
    print(f"    call premium ${call_price(f0, CALL_STRIKE, RISK_FREE_RATE, TIME_TO_MATURITY, VOLATILITY):.8f}")
    print(f"    put  premium ${put_price(f0, put_k, RISK_FREE_RATE, TIME_TO_MATURITY, VOLATILITY):.8f}")
    print()

    print("Effective cost per pound by settlement price")
    print("-" * 78)
    grid = [0.90, 1.00, 1.10, 1.1465, 1.20, 1.25, 1.30, 1.40, 1.50, 1.80]
    ladder = cost_ladder(strategies, grid)
    header = f"  {'F_T':>8}" + "".join(f"{n:>13}" for n in strategies)
    print(header)
    for i, t in enumerate(grid):
        row = f"  ${t:>7.4f}"
        for name in strategies:
            row += f"{ladder[name][i]:>13.4f}"
        print(row)
    print("  Lower is better. Every number is the all-in cost of one pound.")
    print()

    print("Total cost on 375,000 lb at selected outcomes")
    print("-" * 78)
    print(f"  {'F_T':>8}" + "".join(f"{n:>16}" for n in strategies))
    for t in (1.00, 1.20, 1.40, 1.80):
        row = f"  ${t:>7.4f}"
        for name, s in strategies.items():
            row += f"${float(np.asarray(s['cost_fn'](t)).item()) * REQUIRED_VOLUME:>15,.0f}"
        print(row)
    print()

    print("Scenario analysis: mean and 95th-percentile cost per pound")
    print("-" * 78)
    out = scenario_outcomes(SPOT_PRICE, RISK_FREE_RATE, STORAGE_COST,
                            CONVENIENCE_YIELD, TIME_TO_MATURITY, VOLATILITY)
    for sc_name, sc in out.items():
        print(f"  {sc_name}  (E[F_T] = ${sc['mean_terminal']:.4f}, vol {sc['vol']:.0%})")
        print(f"    {'Strategy':<20} {'Mean $/lb':>11} {'P95 $/lb':>11} "
              f"{'Mean total':>14} {'P95 total':>14}")
        for name, r in sc["strategies"].items():
            print(f"    {name:<20} {r['mean_cost']:>11.4f} {r['p95_cost']:>11.4f} "
                  f"${r['total_mean']:>13,.0f} ${r['total_p95']:>13,.0f}")
        print()

    print("Hedge effectiveness (variance reduction, base case)")
    print("-" * 78)
    paths = simulate_paths(SPOT_PRICE, b, VOLATILITY, TIME_TO_MATURITY,
                           NUM_STEPS, NUM_SIMULATIONS, SEED)
    eff = hedge_effectiveness(strategies, paths[-1])
    for name, e in eff.items():
        print(f"  {name:<20} {e:>8.2%}")
    print("  Ignores cost by construction. Read alongside the tables above.")
    print("=" * 78)


if __name__ == "__main__":
    main()
