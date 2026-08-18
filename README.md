# Monte Carlo Simulation — Coffee Price Paths

Simulates coffee price paths under Geometric Brownian Motion, then uses them to build a distribution of outcomes, compute risk measures, price an option, and stress-test supply/demand scenarios.

## The model

```
S(t+dt) = S(t) · exp( (μ − σ²/2)·dt + σ·√dt · Z ),   Z ~ N(0,1)
```

| Symbol | Meaning | Value used |
|---|---|---|
| `S0` | Spot price, USD/lb | 1.20 |
| `μ` | Drift | 2% — set equal to `r`, the risk-neutral drift |
| `σ` | Volatility | 25% p.a. |
| `T` | Horizon | 0.5 years |
| steps | Time steps | 126 (trading days in six months) |
| paths | Simulations | 10,000 |
| seed | RNG seed | 42, for reproducibility |

**Two things that trip people up, and both are tested:**

1. **The `−σ²/2` Ito correction.** It exists because the expectation of a lognormal is not the exponential of the expectation. Drop it and your simulated mean drifts too high and the option price stops matching Black-Scholes.
2. **`μ = r` for pricing, real drift for forecasting.** Same engine, different question. Use the risk-neutral drift when you are pricing a derivative; use a real-world drift when you are forecasting an actual distribution of outcomes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python monte_carlo.py
pytest -q
```

## Expected output

With `seed = 42`, 10,000 paths, 126 steps:

```
Array shape       = (127, 10000)

  mean            $1.2143      (theory: S0·e^(rT) = $1.2121)
  median          $1.1962      (below the mean — lognormal skew)
  std dev         $0.2176
  5th pctile      $0.8903
  95th pctile     $1.6040

  95% VaR                 $0.3097 / lb
  95% Expected Shortfall  $0.3703 / lb

  Monte Carlo price   $0.069682
  Standard error      $0.001272
  95% CI              [$0.067188, $0.072176]
  Black-Scholes says  $0.068358   <-- falls inside the CI
```

and `20 passed` from pytest.

Your exact figures will match only if you use `rng.standard_normal` with the same seed and draw one `num_sims`-length vector per step, in that order. If they differ slightly but the Black-Scholes price still lands inside your confidence interval, your engine is correct — you just consumed random numbers in a different order.

**If something is wrong, debug in this order:**

1. `shape` is `(126, 10000)` not `(127, 10000)` — off-by-one, you allocated `num_steps` rows instead of `num_steps + 1`. This is the bug in most textbook versions of this code, including the one in the original brief.
2. Mean lands near `$1.2273` instead of `$1.2121` — you dropped the `−σ²/2` correction.
3. Negative prices appear — you added the shock instead of exponentiating it.
4. Monte Carlo price sits outside the confidence interval — check you discounted by `e^(−rT)` and that the drift is `r`, not something else.

## What the tests check

20 tests. The important ones:

- **`test_mc_call_price_matches_black_scholes`** — the Monte Carlo price lands within 3 standard errors of the closed-form answer, using 200,000 paths. This is the only real evidence a simulation engine is correct: reproduce a number you already know.
- **`test_mean_terminal_price_matches_theory`** — `E[S_T] = S0·e^(μT)` to within 1%. Fails immediately if the Ito correction is missing.
- **`test_log_returns_have_the_right_variance`** — `Var[ln(S_T/S0)] = σ²T` to within 2%.
- **`test_shape_has_num_steps_plus_one_rows`** — catches the off-by-one directly.
- **`test_standard_error_shrinks_with_more_paths`** — 100× the paths gives ≥5× tighter error, confirming the `1/√n` convergence rate.
- **`test_zero_volatility_gives_deterministic_growth`** — with `σ = 0`, every path collapses to `S0·e^(rT)`.

Plus positivity, reproducibility under a fixed seed, ES > VaR, ordered percentiles, and directional checks on each scenario.

## Scenario analysis

Five regimes, each adjusting drift and volatility:

| Scenario | Drift adj. | Vol multiplier | Story |
|---|---|---|---|
| Base case | 0 | 1.0× | Risk-neutral benchmark |
| Frost / supply shock | +30% | 1.8× | Frost in Minas Gerais; prices spike, uncertainty explodes |
| Bumper harvest | −20% | 1.2× | Oversupply pushes prices down |
| Demand slump | −10% | 1.1× | Consumption falls |
| Calm market | 0 | 0.6× | Quiet, range-bound conditions |

The frost scenario is the one worth talking through: it is not just a higher mean, it is a **fatter right tail**. That asymmetry is precisely why coffee options trade with a skew, and it connects this project back to the Black-Scholes one — where a flat 25% vol across all strikes is exactly the assumption this simulation shows to be wrong.

## Notes and limitations

- **GBM has no jumps.** A frost is a discontinuity; GBM produces continuous paths. A jump-diffusion (Merton) model would be the honest upgrade, and is the natural next commit.
- **Constant volatility.** Commodity vol clusters — calm periods and violent ones. GARCH or a stochastic-vol model would capture that.
- **No mean reversion.** Agricultural prices tend to revert toward the cost of production; GBM lets them wander forever. An Ornstein-Uhlenbeck or Schwartz one-factor model is the standard commodity alternative.
- **Scenario probabilities are judgemental.** The drift and vol adjustments are assumptions, not estimates. Say so out loud rather than presenting them as forecasts.
- **Variance reduction.** At 10,000 paths the standard error is ~$0.0013 on a $0.068 option — roughly 2%. Antithetic variates or a control variate (using the known Black-Scholes price) would cut that substantially for the same compute.
