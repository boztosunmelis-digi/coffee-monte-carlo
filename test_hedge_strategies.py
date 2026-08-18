"""Tests for the hedging strategy comparison.

Run with:  pytest -q

The tests encode the structural properties each hedge must have. A hedge that
fails one of these is not a hedge with a bug in it — it is a different
instrument from the one described in the documentation.
"""

from math import exp

import numpy as np
import pytest

from hedge_strategies import (
    build_strategies,
    call_price,
    cost_call_spread,
    cost_collar,
    cost_futures,
    cost_long_call,
    cost_of_carry,
    cost_unhedged,
    futures_price,
    hedge_effectiveness,
    put_price,
    scenario_outcomes,
    zero_cost_collar_put_strike,
)
from monte_carlo import simulate_paths

S, R, D, Y, T, SIG = 1.20, 0.02, 0.01, 0.00, 0.5, 0.25
B = cost_of_carry(R, D, Y)
F = futures_price(S, B, T)
K = 1.25
UPPER = 1.40


# --- pricing primitives ---------------------------------------------------

def test_put_call_parity_holds():
    c = call_price(F, K, R, T, SIG)
    p = put_price(F, K, R, T, SIG)
    assert c - p == pytest.approx(exp(-R * T) * (F - K), abs=1e-14)


def test_put_premium_is_increasing_in_strike():
    """Required for the collar bisection to have a unique root."""
    strikes = [0.9, 1.0, 1.1, 1.25, 1.4, 1.6]
    prices = [put_price(F, k, R, T, SIG) for k in strikes]
    assert prices == sorted(prices)


# --- the zero-cost collar -------------------------------------------------

def test_collar_solve_produces_matching_premiums():
    """The defining property: the put sold exactly funds the call bought."""
    k_put = zero_cost_collar_put_strike(F, K, R, T, SIG)
    assert put_price(F, k_put, R, T, SIG) == pytest.approx(
        call_price(F, K, R, T, SIG), abs=1e-10
    )


def test_collar_put_strike_sits_below_the_futures_price():
    """A zero-cost collar with an OTM call must sell an OTM put."""
    k_put = zero_cost_collar_put_strike(F, K, R, T, SIG)
    assert k_put < F < K


def test_collar_solve_works_across_call_strikes():
    for k_call in (1.22, 1.25, 1.30, 1.45):
        k_put = zero_cost_collar_put_strike(F, k_call, R, T, SIG)
        assert put_price(F, k_put, R, T, SIG) == pytest.approx(
            call_price(F, k_call, R, T, SIG), abs=1e-10
        )


def test_higher_call_strike_allows_lower_put_strike():
    """Cheaper protection needs less funding, so the put can sit further away."""
    near = zero_cost_collar_put_strike(F, 1.22, R, T, SIG)
    far = zero_cost_collar_put_strike(F, 1.45, R, T, SIG)
    assert far < near


# --- cost functions -------------------------------------------------------

def test_unhedged_cost_is_the_market_price():
    grid = np.array([0.9, 1.2, 1.8])
    assert np.allclose(cost_unhedged(grid), grid)


def test_futures_locks_cost_at_entry_regardless_of_outcome():
    grid = np.array([0.5, 1.0, 1.2181, 2.0, 5.0])
    costs = cost_futures(grid, F)
    assert np.allclose(costs, F)
    assert costs.std() == pytest.approx(0.0, abs=1e-15)


def test_long_call_cost_is_capped_at_strike_plus_future_value_of_premium():
    prem = call_price(F, K, R, T, SIG)
    cap = K + prem * exp(R * T)
    for t in (1.30, 1.50, 2.00, 10.0):
        assert cost_long_call(t, K, prem, R, T) == pytest.approx(cap)


def test_long_call_leaves_downside_benefit_intact():
    """Below the strike the buyer still enjoys a falling market, less premium."""
    prem = call_price(F, K, R, T, SIG)
    low = float(cost_long_call(0.90, K, prem, R, T))
    assert low == pytest.approx(0.90 + prem * exp(R * T))
    assert low < K


def test_collar_cost_is_trapped_between_the_two_strikes():
    k_put = zero_cost_collar_put_strike(F, K, R, T, SIG)
    for t in (0.30, 0.90, 1.10, 1.20, 1.40, 3.00):
        c = float(cost_collar(t, K, k_put, 0.0, R, T))
        assert k_put - 1e-12 <= c <= K + 1e-12


def test_collar_gives_away_the_downside():
    """Below the put strike the buyer pays the put strike, not the market."""
    k_put = zero_cost_collar_put_strike(F, K, R, T, SIG)
    assert float(cost_collar(0.80, K, k_put, 0.0, R, T)) == pytest.approx(k_put)
    assert float(cost_collar(0.80, K, k_put, 0.0, R, T)) > 0.80


def test_call_spread_protection_stops_at_the_upper_strike():
    """Above the upper strike the buyer is exposed one for one again.

    This is the defining weakness and it must be visible in the slope.
    """
    net = call_price(F, K, R, T, SIG) - call_price(F, UPPER, R, T, SIG)
    a = float(cost_call_spread(1.60, K, UPPER, net, R, T))
    b = float(cost_call_spread(1.70, K, UPPER, net, R, T))
    assert b - a == pytest.approx(0.10, abs=1e-12)   # slope of exactly 1


def test_call_spread_is_cheaper_than_the_outright_call():
    c_lower = call_price(F, K, R, T, SIG)
    c_upper = call_price(F, UPPER, R, T, SIG)
    assert 0 < c_lower - c_upper < c_lower


def test_call_spread_max_protection_equals_strike_width():
    """Protection is capped at (upper - lower) per pound."""
    net = call_price(F, K, R, T, SIG) - call_price(F, UPPER, R, T, SIG)
    unhedged = 3.00
    hedged = float(cost_call_spread(3.00, K, UPPER, net, R, T))
    assert unhedged - hedged == pytest.approx(UPPER - K - net * exp(R * T))


# --- strategy set ---------------------------------------------------------

def test_all_five_strategies_are_built():
    strats = build_strategies(F, R, T, SIG)
    assert set(strats) == {"Unhedged", "Long futures", "Long calls",
                           "Zero-cost collar", "Call spread"}


def test_collar_premium_is_zero_and_others_are_not_negative():
    strats = build_strategies(F, R, T, SIG)
    assert strats["Zero-cost collar"]["premium"] == pytest.approx(0.0, abs=1e-10)
    assert strats["Unhedged"]["premium"] == 0.0
    assert strats["Long futures"]["premium"] == 0.0
    assert strats["Long calls"]["premium"] > 0
    assert strats["Call spread"]["premium"] > 0


def test_call_spread_costs_less_than_long_calls():
    strats = build_strategies(F, R, T, SIG)
    assert strats["Call spread"]["premium"] < strats["Long calls"]["premium"]


# --- hedge effectiveness --------------------------------------------------

def test_futures_hedge_is_perfectly_effective():
    strats = build_strategies(F, R, T, SIG)
    paths = simulate_paths(S, B, SIG, T, 126, 20_000, seed=1)
    eff = hedge_effectiveness(strats, paths[-1])
    assert eff["Long futures"] == pytest.approx(1.0, abs=1e-12)
    assert eff["Unhedged"] == pytest.approx(0.0, abs=1e-12)


def test_effectiveness_ordering_matches_protection_breadth():
    """Collar beats a straight call, which beats a call spread.

    The collar caps both tails, so it removes the most variance. The call
    spread only protects a 15-cent window, so it removes the least.
    """
    strats = build_strategies(F, R, T, SIG)
    paths = simulate_paths(S, B, SIG, T, 126, 50_000, seed=2)
    eff = hedge_effectiveness(strats, paths[-1])
    assert eff["Long futures"] > eff["Zero-cost collar"] > eff["Long calls"] > eff["Call spread"] > 0


# --- scenario analysis ----------------------------------------------------

def test_scenario_outcomes_cover_every_scenario_and_strategy():
    out = scenario_outcomes(S, R, D, Y, T, SIG, num_sims=5_000, seed=3)
    assert len(out) == 5
    for sc in out.values():
        assert len(sc["strategies"]) == 5
        for r in sc["strategies"].values():
            assert r["p95_cost"] >= r["mean_cost"]


def test_futures_cost_is_identical_across_every_scenario():
    """A locked price does not care what the market does. That is the point."""
    out = scenario_outcomes(S, R, D, Y, T, SIG, num_sims=5_000, seed=4)
    costs = {sc["strategies"]["Long futures"]["mean_cost"] for sc in out.values()}
    assert max(costs) - min(costs) < 1e-12


def test_frost_scenario_is_where_hedging_pays():
    out = scenario_outcomes(S, R, D, Y, T, SIG, num_sims=20_000, seed=5)
    frost = out["Frost / supply shock"]["strategies"]
    assert frost["Long futures"]["mean_cost"] < frost["Unhedged"]["mean_cost"]
    assert frost["Long calls"]["mean_cost"] < frost["Unhedged"]["mean_cost"]
    assert frost["Zero-cost collar"]["mean_cost"] < frost["Unhedged"]["mean_cost"]


def test_call_spread_fails_in_the_frost_scenario():
    """The finding that drives the recommendation.

    A call spread's protection runs out at the upper strike, so in the very
    scenario it was bought for it leaves most of the exposure in place. Its
    95th-percentile cost should stay close to unhedged rather than close to
    the other hedges.
    """
    out = scenario_outcomes(S, R, D, Y, T, SIG, num_sims=20_000, seed=6)
    frost = out["Frost / supply shock"]["strategies"]
    unhedged_p95 = frost["Unhedged"]["p95_cost"]
    spread_p95 = frost["Call spread"]["p95_cost"]
    collar_p95 = frost["Zero-cost collar"]["p95_cost"]
    assert spread_p95 > 0.9 * unhedged_p95      # barely better than no hedge
    assert spread_p95 > 1.5 * collar_p95        # far worse than a real hedge


def test_collar_costs_more_than_unhedged_when_prices_fall():
    """The price of free protection, made explicit."""
    out = scenario_outcomes(S, R, D, Y, T, SIG, num_sims=20_000, seed=7)
    bumper = out["Bumper harvest"]["strategies"]
    assert bumper["Zero-cost collar"]["mean_cost"] > bumper["Unhedged"]["mean_cost"]
