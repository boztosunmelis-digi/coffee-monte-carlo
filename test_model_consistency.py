"""Tests for the cross-model consistency check.

Run with:  pytest -q

Two kinds of test here, and the distinction matters:

  EXACT tests assert algebraic identities. They use abs=1e-14 or tighter,
  because a genuine identity does not "approximately" hold. If one of these
  drifts to 1e-6, something is structurally wrong, not numerically noisy.

  STATISTICAL tests assert Monte Carlo convergence. They allow three standard
  errors, because sampling error is real and pretending otherwise produces a
  flaky suite.
"""

from math import exp, sqrt

import numpy as np
import pytest

from model_consistency import (
    _norm_cdf,
    black76_call,
    black76_put,
    black_scholes_call,
    cost_of_carry,
    futures_price,
    generalised_bs_call,
    mc_call_on_futures,
    mc_call_on_spot,
    reconcile,
    terminal_distributions_match,
)

S, X, R, D, Y, T, SIG = 1.20, 1.25, 0.02, 0.01, 0.00, 0.5, 0.25
B = R + D - Y
F = S * exp(B * T)

# A grid to check identities hold generally, not just at one lucky point.
GRID = [
    (s, k, r, d, y, t, sig)
    for s in (0.8, 1.2, 2.5)
    for k in (0.9, 1.25, 2.0)
    for r in (0.0, 0.02, 0.08)
    for d in (0.0, 0.01, 0.04)
    for y in (0.0, 0.03)
    for t in (0.25, 1.0)
    for sig in (0.1, 0.25, 0.6)
]


# --- the normal CDF stand-in --------------------------------------------

def test_norm_cdf_at_zero():
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-15)


def test_norm_cdf_is_symmetric():
    for x in (0.3, 1.0, 2.576):
        assert _norm_cdf(x) + _norm_cdf(-x) == pytest.approx(1.0, abs=1e-15)


def test_norm_cdf_known_quantiles():
    assert _norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-9)
    assert _norm_cdf(-1.281551566) == pytest.approx(0.10, abs=1e-9)


# --- step 1: cost of carry ----------------------------------------------

def test_carry_combines_the_three_components():
    assert cost_of_carry(0.02, 0.01, 0.0) == pytest.approx(0.03)
    assert cost_of_carry(0.02, 0.01, 0.03) == pytest.approx(0.0)


def test_futures_price_matches_the_cost_of_carry_project():
    """$1.2181 is what coffee-cost-of-carry prints."""
    assert futures_price(1.20, 0.03, 0.5) == pytest.approx(1.2181, abs=1e-4)


def test_large_convenience_yield_gives_backwardation():
    b = cost_of_carry(0.02, 0.01, 0.20)
    assert futures_price(1.20, b, 0.5) < 1.20


# --- identity (a): Black-76 == generalised Black-Scholes ------------------

def test_black76_equals_generalised_bs_at_base_case():
    """The central claim. Exact, so the tolerance is floating-point dust."""
    b76 = black76_call(F, X, R, T, SIG)
    gbs = generalised_bs_call(S, X, R, B, T, SIG)
    assert b76 == pytest.approx(gbs, abs=1e-15)


def test_black76_equals_generalised_bs_across_the_grid():
    """486 parameter combinations. An identity holds everywhere or it isn't one."""
    for s, k, r, d, y, t, sig in GRID:
        b = cost_of_carry(r, d, y)
        f = futures_price(s, b, t)
        b76 = black76_call(f, k, r, t, sig)
        gbs = generalised_bs_call(s, k, r, b, t, sig)
        assert b76 == pytest.approx(gbs, rel=1e-12, abs=1e-15), (s, k, r, d, y, t, sig)


# --- identity (b): generalised BS with b = r == plain Black-Scholes -------

def test_generalised_bs_with_carry_equal_to_rate_is_plain_bs():
    gbs = generalised_bs_call(S, X, R, R, T, SIG)
    plain = black_scholes_call(S, X, R, T, SIG)
    assert gbs == pytest.approx(plain, abs=1e-15)


def test_bridge_to_the_black_scholes_project():
    """With d = y = 0, Black-76 off F must reproduce $0.068358 exactly.

    That number is the coffee-black-scholes project's answer. This is the
    test that stitches the two repositories together.
    """
    b = cost_of_carry(R, 0.0, 0.0)
    f = futures_price(S, b, T)
    assert black76_call(f, X, R, T, SIG) == pytest.approx(0.068358, abs=1e-6)
    assert black_scholes_call(S, X, R, T, SIG) == pytest.approx(0.068358, abs=1e-6)


# --- Black-76 structural properties --------------------------------------

def test_black76_put_call_parity():
    """For futures options: C - P = exp(-rT) * (F - X). Exact identity."""
    c = black76_call(F, X, R, T, SIG)
    p = black76_put(F, X, R, T, SIG)
    assert c - p == pytest.approx(exp(-R * T) * (F - X), abs=1e-15)


def test_rate_enters_black76_only_as_a_discount_factor():
    """d1 and d2 contain no r, so changing r rescales the price by exp(-rT).

    This is the structural difference from spot Black-Scholes, and it is why
    Black-76 does not double-count the carry.
    """
    p1 = black76_call(F, X, 0.02, T, SIG)
    p2 = black76_call(F, X, 0.09, T, SIG)
    assert p2 / p1 == pytest.approx(exp(-(0.09 - 0.02) * T), rel=1e-12)


def test_black76_call_falls_as_strike_rises():
    assert black76_call(F, 1.00, R, T, SIG) > black76_call(F, 1.50, R, T, SIG)


def test_black76_respects_no_arbitrage_bounds():
    c = black76_call(F, X, R, T, SIG)
    lower = max(exp(-R * T) * (F - X), 0.0)
    assert lower <= c <= exp(-R * T) * F


# --- the mistake being guarded against ------------------------------------

def test_spot_bs_on_a_futures_price_overprices():
    """Feeding F into spot Black-Scholes charges for the carry twice.

    The error is one-directional whenever the carry is positive, which is the
    normal state for a storable commodity.
    """
    correct = black76_call(F, X, R, T, SIG)
    wrong = black_scholes_call(F, X, R, T, SIG)
    assert wrong > correct
    assert (wrong / correct - 1) > 0.05  # materially wrong, not a rounding issue


# --- the two Monte Carlo routes ------------------------------------------

def test_both_simulation_routes_give_identical_paths():
    """Simulating S at drift b and F at drift 0 is the same process.

    Same seed, same draws, same terminal values to the last bit. This is why
    a futures price is driftless under the risk-neutral measure.
    """
    gap = terminal_distributions_match(S, B, T, SIG, num_sims=5_000, seed=1)
    assert gap < 1e-12


def test_both_routes_price_identically():
    mc_f, _ = mc_call_on_futures(F, X, R, T, SIG, num_sims=20_000, seed=7)
    mc_s, _ = mc_call_on_spot(S, X, R, B, T, SIG, num_sims=20_000, seed=7)
    assert mc_f == pytest.approx(mc_s, abs=1e-12)


# --- identity (c): Monte Carlo converges to Black-76 ----------------------

def test_mc_on_futures_converges_to_black76():
    """Statistical, not exact: allow three standard errors."""
    analytic = black76_call(F, X, R, T, SIG)
    mc, se = mc_call_on_futures(F, X, R, T, SIG, num_sims=100_000, seed=11)
    assert abs(mc - analytic) < 3 * se


def test_mc_on_spot_converges_to_black76():
    analytic = black76_call(F, X, R, T, SIG)
    mc, se = mc_call_on_spot(S, X, R, B, T, SIG, num_sims=100_000, seed=12)
    assert abs(mc - analytic) < 3 * se


def test_mc_standard_error_shrinks_as_one_over_root_n():
    _, se_small = mc_call_on_futures(F, X, R, T, SIG, num_sims=1_000, seed=13)
    _, se_large = mc_call_on_futures(F, X, R, T, SIG, num_sims=100_000, seed=13)
    assert se_large < se_small / 5


def test_simulated_futures_price_is_a_martingale():
    """E[F_T] = F. A futures price does not drift."""
    from monte_carlo import simulate_paths
    paths = simulate_paths(F, 0.0, SIG, T, 126, 200_000, seed=14)
    assert paths[-1].mean() == pytest.approx(F, rel=0.01)


# --- the reconciliation as a whole ---------------------------------------

def test_reconcile_returns_every_key():
    r = reconcile(num_sims=5_000, seed=3)
    for key in ("carry", "futures", "black76", "generalised_bs", "mc_futures",
                "mc_futures_se", "mc_spot", "mc_spot_se", "identity_gap", "path_gap"):
        assert key in r, f"reconcile() is missing '{key}'"


def test_reconcile_identity_gap_is_floating_point_dust():
    r = reconcile(num_sims=5_000, seed=3)
    assert abs(r["identity_gap"]) < 1e-14
    assert r["path_gap"] < 1e-12


def test_reconcile_monte_carlo_lands_within_confidence_interval():
    r = reconcile(num_sims=100_000, seed=21)
    lo = r["mc_futures"] - 1.96 * r["mc_futures_se"]
    hi = r["mc_futures"] + 1.96 * r["mc_futures_se"]
    assert lo < r["black76"] < hi
