"""Sanity checks for the Monte Carlo engine.

Run with:  pytest -q

The headline test is `test_mc_call_price_matches_black_scholes` — it proves the
simulation reproduces a known closed-form answer, which is the only real
evidence that a Monte Carlo engine is correct.
"""

import numpy as np
import pytest

from monte_carlo import (
    simulate_paths,
    summarise_terminal,
    value_at_risk,
    expected_shortfall,
    mc_call_price,
    mc_standard_error,
    run_scenarios,
    SCENARIOS,
)

S0, R, SIG, T = 1.20, 0.02, 0.25, 0.5


def bs_call(spot, strike, rate, maturity, sigma):
    """Closed-form Black-Scholes call, used as ground truth."""
    from math import log, sqrt, exp, erf

    def ncdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    d1 = (log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return spot * ncdf(d1) - strike * exp(-rate * maturity) * ncdf(d2)


# --- shape and structure ---------------------------------------------------

def test_shape_has_num_steps_plus_one_rows():
    """The classic off-by-one. 126 steps means 127 rows."""
    paths = simulate_paths(S0, R, SIG, T, 126, 500, seed=1)
    assert paths.shape == (127, 500)


def test_first_row_is_the_spot_price():
    paths = simulate_paths(S0, R, SIG, T, 50, 100, seed=1)
    assert np.all(paths[0] == S0)


def test_all_prices_stay_positive():
    """GBM is multiplicative, so a price can never go to zero or negative.
    A negative value means you added the shock instead of exponentiating it."""
    paths = simulate_paths(S0, R, SIG, T, 126, 1000, seed=1)
    assert np.all(paths > 0)


def test_seed_makes_results_reproducible():
    a = simulate_paths(S0, R, SIG, T, 50, 200, seed=7)
    b = simulate_paths(S0, R, SIG, T, 50, 200, seed=7)
    assert np.allclose(a, b)


def test_different_seeds_give_different_paths():
    a = simulate_paths(S0, R, SIG, T, 50, 200, seed=7)
    b = simulate_paths(S0, R, SIG, T, 50, 200, seed=8)
    assert not np.allclose(a, b)


# --- statistical properties ------------------------------------------------

def test_zero_volatility_gives_deterministic_growth():
    """With sigma = 0 every path is identical: S0 * exp(r*T)."""
    paths = simulate_paths(S0, R, 0.0, T, 100, 50, seed=1)
    assert paths[-1] == pytest.approx(S0 * np.exp(R * T), rel=1e-10)


def test_mean_terminal_price_matches_theory():
    """E[S_T] = S0 * exp(mu*T). This is what the Ito correction buys you.
    If you dropped the -sigma^2/2 term, this test fails."""
    paths = simulate_paths(S0, R, SIG, T, 126, 200_000, seed=3)
    expected = S0 * np.exp(R * T)
    assert paths[-1].mean() == pytest.approx(expected, rel=0.01)


def test_median_is_below_the_mean():
    """The lognormal distribution is right-skewed."""
    paths = simulate_paths(S0, R, SIG, T, 126, 50_000, seed=4)
    s = summarise_terminal(paths)
    assert s["median"] < s["mean"]


def test_log_returns_have_the_right_variance():
    """Var[ln(S_T/S0)] = sigma^2 * T."""
    paths = simulate_paths(S0, R, SIG, T, 126, 100_000, seed=5)
    log_ret = np.log(paths[-1] / S0)
    assert log_ret.var() == pytest.approx(SIG ** 2 * T, rel=0.02)


def test_higher_vol_widens_the_distribution():
    calm = simulate_paths(S0, R, 0.10, T, 126, 20_000, seed=6)
    wild = simulate_paths(S0, R, 0.60, T, 126, 20_000, seed=6)
    assert wild[-1].std() > calm[-1].std()


# --- the headline test -----------------------------------------------------

def test_mc_call_price_matches_black_scholes():
    """The Monte Carlo price must land within ~3 standard errors of the
    closed-form answer. This is the test that validates the whole engine."""
    paths = simulate_paths(S0, R, SIG, T, 126, 200_000, seed=11)
    mc = mc_call_price(paths, 1.25, R, T)
    se = mc_standard_error(paths, 1.25, R, T)
    truth = bs_call(S0, 1.25, R, T, SIG)
    assert abs(mc - truth) < 3 * se


def test_standard_error_shrinks_with_more_paths():
    """SE scales as 1/sqrt(n): 100x the paths, ~10x tighter."""
    few = simulate_paths(S0, R, SIG, T, 63, 1_000, seed=12)
    many = simulate_paths(S0, R, SIG, T, 63, 100_000, seed=12)
    se_few = mc_standard_error(few, 1.25, R, T)
    se_many = mc_standard_error(many, 1.25, R, T)
    assert se_many < se_few / 5


def test_call_price_is_positive_and_below_spot():
    paths = simulate_paths(S0, R, SIG, T, 126, 20_000, seed=13)
    mc = mc_call_price(paths, 1.25, R, T)
    assert 0 < mc < S0


# --- risk measures ---------------------------------------------------------

def test_expected_shortfall_exceeds_var():
    """ES averages the tail beyond VaR, so it is always the larger loss."""
    paths = simulate_paths(S0, R, SIG, T, 126, 50_000, seed=14)
    assert expected_shortfall(paths, S0) > value_at_risk(paths, S0)


def test_var_is_a_positive_loss_number():
    paths = simulate_paths(S0, R, SIG, T, 126, 50_000, seed=15)
    assert value_at_risk(paths, S0) > 0


def test_summary_keys_are_all_present():
    paths = simulate_paths(S0, R, SIG, T, 63, 5_000, seed=16)
    s = summarise_terminal(paths)
    for key in ("mean", "median", "std", "p05", "p25", "p75", "p95", "min", "max"):
        assert key in s, f"summarise_terminal is missing '{key}'"
    assert s["p05"] < s["p25"] < s["median"] < s["p75"] < s["p95"]


# --- scenarios -------------------------------------------------------------

def test_all_scenarios_run():
    out = run_scenarios(S0, R, SIG, T, 63, 5_000, seed=17)
    assert set(out) == set(SCENARIOS)


def test_frost_scenario_is_more_bullish_than_base():
    out = run_scenarios(S0, R, SIG, T, 126, 20_000, seed=18)
    assert out["Frost / supply shock"]["mean"] > out["Base case"]["mean"]
    assert out["Frost / supply shock"]["prob_above_spot"] > out["Base case"]["prob_above_spot"]


def test_bumper_harvest_is_more_bearish_than_base():
    out = run_scenarios(S0, R, SIG, T, 126, 20_000, seed=19)
    assert out["Bumper harvest"]["mean"] < out["Base case"]["mean"]


def test_calm_market_has_the_narrowest_range():
    out = run_scenarios(S0, R, SIG, T, 126, 20_000, seed=20)
    calm_width = out["Calm market"]["p95"] - out["Calm market"]["p05"]
    base_width = out["Base case"]["p95"] - out["Base case"]["p05"]
    assert calm_width < base_width
