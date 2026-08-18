"""Tests for digital option pricing.

Run with:  pytest -q

As with the consistency module, two kinds of test:

  EXACT tests assert algebraic identities (digital parity, the vanilla
  decomposition, the strike-derivative relationship) at 1e-14 or tighter.

  STATISTICAL tests assert Monte Carlo convergence and allow three standard
  errors.

The finite-difference Greek tests are the ones that matter most. A digital's
delta and gamma are easy to get subtly wrong, and the consequences of an
undetected sign error on a barrier product are large.
"""

from math import exp, log, sqrt

import numpy as np
import pytest

from digital_options import (
    _d1_d2,
    _norm_cdf,
    _norm_pdf,
    asset_or_nothing_call,
    barrier_ladder,
    call_spread_replication,
    cost_of_carry,
    digital_call,
    digital_call_greeks,
    digital_put,
    futures_price,
    mc_digital_call,
    pin_risk_profile,
    risk_neutral_probability_itm,
    scenario_probabilities,
    vanilla_call,
)

S, X, R, D, Y, T, SIG = 1.20, 1.25, 0.02, 0.01, 0.00, 0.5, 0.25
B = cost_of_carry(R, D, Y)
F = futures_price(S, B, T)
Q = 10_000.0
BASE = (F, X, R, T, SIG)

GRID = [
    (f, k, r, t, sig)
    for f in (0.8, 1.2181, 2.0)
    for k in (0.9, 1.25, 1.8)
    for r in (0.0, 0.02, 0.07)
    for t in (0.25, 0.5, 1.5)
    for sig in (0.12, 0.25, 0.55)
]


# --- normal distribution helpers ----------------------------------------

def test_norm_pdf_integrates_consistently_with_cdf():
    """n(x) must be the derivative of N(x)."""
    for x in (-1.5, -0.2, 0.0, 0.8, 2.1):
        h = 1e-6
        numeric = (_norm_cdf(x + h) - _norm_cdf(x - h)) / (2 * h)
        assert _norm_pdf(x) == pytest.approx(numeric, rel=1e-6)


# --- exact identity: digital parity --------------------------------------

def test_digital_call_plus_put_equals_discount_factor():
    """One of the two must pay. Their combined value is a riskless $1."""
    total = digital_call(*BASE) + digital_put(*BASE)
    assert total == pytest.approx(exp(-R * T), abs=1e-15)


def test_digital_parity_across_the_grid():
    for f, k, r, t, sig in GRID:
        total = digital_call(f, k, r, t, sig) + digital_put(f, k, r, t, sig)
        assert total == pytest.approx(exp(-r * t), abs=1e-14), (f, k, r, t, sig)


# --- exact identity: vanilla decomposition -------------------------------

def test_vanilla_decomposes_into_two_digitals():
    """C_vanilla = asset-or-nothing - X * cash-or-nothing. Exact."""
    aon = asset_or_nothing_call(*BASE)
    con = digital_call(*BASE)
    assert aon - X * con == pytest.approx(vanilla_call(*BASE), abs=1e-15)


def test_decomposition_holds_across_the_grid():
    for f, k, r, t, sig in GRID:
        aon = asset_or_nothing_call(f, k, r, t, sig)
        con = digital_call(f, k, r, t, sig)
        assert aon - k * con == pytest.approx(
            vanilla_call(f, k, r, t, sig), rel=1e-12, abs=1e-15
        ), (f, k, r, t, sig)


# --- exact identity: digital is the strike-derivative of a vanilla --------

def test_digital_equals_negative_strike_derivative_of_vanilla():
    """cash-or-nothing = -dC_vanilla/dX. This is why a call spread replicates it."""
    h = 1e-6
    numeric = -(vanilla_call(F, X + h, R, T, SIG) - vanilla_call(F, X - h, R, T, SIG)) / (2 * h)
    assert digital_call(*BASE) == pytest.approx(numeric, rel=1e-7)


# --- pricing sanity ------------------------------------------------------

def test_digital_value_never_exceeds_discounted_payout():
    """You cannot pay more than the present value of the maximum payout."""
    for f, k, r, t, sig in GRID:
        assert 0.0 <= digital_call(f, k, r, t, sig, payout=Q) <= Q * exp(-r * t) + 1e-12


def test_digital_call_falls_as_barrier_rises():
    ladder = barrier_ladder(F, R, T, SIG, [1.10, 1.20, 1.25, 1.35, 1.50, 1.75])
    values = [v for _, _, v in ladder]
    probs = [p for _, p, _ in ladder]
    assert values == sorted(values, reverse=True)
    assert probs == sorted(probs, reverse=True)


def test_price_is_the_discounted_risk_neutral_probability():
    """Strip the discount factor and the digital price IS a probability."""
    prob = risk_neutral_probability_itm(F, X, T, SIG)
    assert digital_call(*BASE) == pytest.approx(exp(-R * T) * prob, abs=1e-15)
    assert 0.0 < prob < 1.0


def test_deep_in_the_money_digital_approaches_full_discounted_payout():
    v = digital_call(5.0, 1.25, R, T, SIG, payout=Q)
    assert v == pytest.approx(Q * exp(-R * T), rel=1e-6)


def test_deep_out_of_the_money_digital_is_worthless():
    assert digital_call(0.20, 1.25, R, T, SIG, payout=Q) == pytest.approx(0.0, abs=1e-6)


# --- Greeks against finite differences ------------------------------------

def test_delta_matches_numerical_derivative():
    h = 1e-6
    numeric = (digital_call(F + h, X, R, T, SIG) - digital_call(F - h, X, R, T, SIG)) / (2 * h)
    assert digital_call_greeks(*BASE)["delta"] == pytest.approx(numeric, rel=1e-5)


def test_gamma_matches_numerical_second_derivative():
    h = 1e-4
    up = digital_call(F + h, X, R, T, SIG)
    mid = digital_call(F, X, R, T, SIG)
    dn = digital_call(F - h, X, R, T, SIG)
    numeric = (up - 2 * mid + dn) / h ** 2
    assert digital_call_greeks(*BASE)["gamma"] == pytest.approx(numeric, rel=1e-3)


def test_vega_matches_numerical_derivative():
    h = 1e-6
    numeric = (digital_call(F, X, R, T, SIG + h) - digital_call(F, X, R, T, SIG - h)) / (2 * h)
    assert digital_call_greeks(*BASE)["vega"] * 100 == pytest.approx(numeric, rel=1e-5)


def test_theta_matches_numerical_derivative():
    h = 1e-6
    dv_dt = (digital_call(F, X, R, T + h, SIG) - digital_call(F, X, R, T - h, SIG)) / (2 * h)
    assert digital_call_greeks(*BASE)["theta"] * 365 == pytest.approx(-dv_dt, rel=1e-4)


def test_rho_matches_numerical_derivative():
    h = 1e-8
    numeric = (digital_call(F, X, R + h, T, SIG) - digital_call(F, X, R - h, T, SIG)) / (2 * h)
    assert digital_call_greeks(*BASE)["rho"] * 100 == pytest.approx(numeric, rel=1e-4)


def test_delta_is_always_positive():
    """Higher futures price always means a better chance of paying out."""
    for f, k, r, t, sig in GRID:
        assert digital_call_greeks(f, k, r, t, sig)["delta"] > 0


# --- the sign flips that make digitals different from vanillas ------------

def test_gamma_changes_sign_at_the_money():
    """Vanilla gamma is strictly positive. Digital gamma is not."""
    below = digital_call_greeks(1.10, X, R, T, SIG)["gamma"]
    above = digital_call_greeks(1.45, X, R, T, SIG)["gamma"]
    assert below > 0
    assert above < 0


def test_vega_changes_sign_at_the_money():
    """An OTM digital gains from volatility; an ITM one loses.

    Volatility only helps while it can still carry you across the barrier.
    Once you are safely past it, volatility is pure threat.
    """
    otm = digital_call_greeks(1.10, X, R, T, SIG)["vega"]
    itm = digital_call_greeks(1.45, X, R, T, SIG)["vega"]
    assert otm > 0
    assert itm < 0


def test_itm_digital_loses_value_when_volatility_rises():
    """The sign flip, demonstrated on prices rather than Greeks."""
    low = digital_call(1.45, X, R, T, 0.15, payout=Q)
    high = digital_call(1.45, X, R, T, 0.45, payout=Q)
    assert high < low


# --- static replication ---------------------------------------------------

def test_call_spread_converges_to_the_digital_as_width_shrinks():
    prev = None
    for w in (0.10, 0.05, 0.01, 0.001):
        rep = call_spread_replication(*BASE, width=w, payout=Q)
        err = abs(rep["overhedge"])
        if prev is not None:
            assert err < prev
        prev = err
    assert prev < 1.0  # within a dollar on a $10,000 payout


def test_call_spread_always_overhedges():
    """The spread must cost at least the digital, or it would be an arbitrage."""
    for w in (0.001, 0.01, 0.05, 0.10, 0.20):
        rep = call_spread_replication(*BASE, width=w, payout=Q)
        assert rep["overhedge"] > 0


def test_narrower_spread_requires_larger_hedge_position():
    """The core trade-off: tracking error against hedge size."""
    wide = call_spread_replication(*BASE, width=0.10, payout=Q)
    tight = call_spread_replication(*BASE, width=0.01, payout=Q)
    assert tight["overhedge"] < wide["overhedge"]
    assert tight["max_delta"] > wide["max_delta"]


# --- pin risk -------------------------------------------------------------

def test_delta_at_the_barrier_diverges_as_expiry_approaches():
    """The reason digitals cannot be perfectly hedged."""
    rows = pin_risk_profile(X, X, R, SIG, [0.5, 0.25, 0.02, 0.0004], payout=Q)
    deltas = [d for _, d, _ in rows]
    assert deltas == sorted(deltas)          # rising as T falls
    assert deltas[-1] > 30 * deltas[0]       # by more than an order of magnitude


def test_delta_scales_linearly_with_payout():
    small = digital_call_greeks(*BASE, payout=1.0)["delta"]
    large = digital_call_greeks(*BASE, payout=Q)["delta"]
    assert large == pytest.approx(small * Q, rel=1e-12)


# --- Monte Carlo ----------------------------------------------------------

def test_mc_digital_converges_to_analytic_price():
    analytic = digital_call(*BASE, payout=Q)
    mc, se = mc_digital_call(*BASE, payout=Q, num_sims=200_000, seed=11)
    assert abs(mc - analytic) < 3 * se


def test_mc_standard_error_shrinks_with_more_paths():
    _, se_small = mc_digital_call(*BASE, payout=Q, num_sims=2_000, seed=13)
    _, se_large = mc_digital_call(*BASE, payout=Q, num_sims=200_000, seed=13)
    assert se_large < se_small / 5


# --- scenario analysis ----------------------------------------------------

def test_scenario_probabilities_are_valid_and_ordered():
    out = scenario_probabilities(S, X, R, D, Y, T, SIG, num_sims=20_000, seed=5)
    for name, s in out.items():
        assert 0.0 <= s["prob_above"] <= 1.0, name
        assert s["expected_payout"] == pytest.approx(s["prob_above"] * 10_000)
    assert out["Frost / supply shock"]["prob_above"] > out["Base case"]["prob_above"]
    assert out["Bumper harvest"]["prob_above"] < out["Base case"]["prob_above"]


def test_frost_scenario_raises_the_digital_value():
    out = scenario_probabilities(S, X, R, D, Y, T, SIG, num_sims=20_000, seed=6)
    assert out["Frost / supply shock"]["digital_value"] > out["Base case"]["digital_value"]
