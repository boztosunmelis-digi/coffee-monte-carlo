# Security Policy

## Scope

This repository contains an educational Monte Carlo simulation of commodity price paths under Geometric Brownian Motion. It has no network access, no authentication, no user input handling, and no persistent storage. The conventional software attack surface is close to nil.

The risks worth documenting here are **model risks**, and in a simulation they are unusually easy to overlook — because a simulation produces confident-looking numbers regardless of whether its assumptions hold.

## The most dangerous outputs in this repository

The VaR and Expected Shortfall figures. They are the ones that look most like risk management and are least entitled to be trusted.

A 95% VaR of `$0.3097` is not an empirical fact about coffee. It is the 5th percentile of a lognormal distribution that this code assumed into existence. If the real return distribution has fatter tails than lognormal — and for a weather-exposed agricultural commodity it certainly does — then this number **understates** the loss you should be planning for, and it does so with three decimal places of false precision.

A wrong risk number is worse than no risk number, because it gets acted on.

## Model risk disclosure

This code is **not production risk software** and must not be used to size positions, set limits, or report risk. Specifically:

- **Geometric Brownian Motion has no jumps.** A frost in Minas Gerais is a discontinuity. GBM generates continuous paths and structurally cannot produce one. Every tail measure here is therefore optimistic.
- **Volatility is constant.** Commodity volatility clusters — long calm stretches punctuated by violent ones. A single 25% figure describes neither regime.
- **There is no mean reversion.** Agricultural prices tend to revert toward the cost of production. GBM lets them wander indefinitely, which overstates dispersion at long horizons and understates it at short ones.
- **The scenario adjustments are judgements, not estimates.** The `+30% drift, 1.8× vol` frost case is an assumption someone made up. It is not calibrated to any historical frost, and it carries no probability weight. Presenting scenario output as a forecast would be misleading.
- **All inputs are hardcoded illustrative values**, not market data.
- **Monte Carlo error is reported but easy to ignore.** The standard error tells you the precision of the *estimate*, not the accuracy of the *model*. Ten million paths would tighten the confidence interval around an answer that is still wrong in the same direction.

## What the tests do and do not prove

The suite verifies the engine reproduces known analytic results: the mean terminal price matches `S0·e^(μT)`, log-return variance matches `σ²T`, and the Monte Carlo option price lands within three standard errors of the closed-form Black-Scholes value. That is strong evidence the simulation correctly implements Geometric Brownian Motion.

It is **no evidence** that Geometric Brownian Motion describes coffee. The tests validate the machinery against its own specification. Whether that specification resembles the world is a separate question, tested nowhere in this repository.

## Reproducibility

Results are seeded (`SEED = 42`) and reproducible. Note that this cuts both ways: a fixed seed makes results verifiable, but reporting a single seeded run as though it were *the* answer hides the sampling variability that the standard error exists to communicate.

## Reporting an issue

Please open a GitHub issue describing:

1. What you expected the simulation to produce
2. What it actually produced
3. The inputs and seed that reproduce it

Mathematical and statistical errors are the most valuable reports. If a test asserts something that is not actually a property of GBM, that is a bug in the test and worth reporting too.

## Supported versions

Only the `main` branch is maintained. There are no released versions and no backported fixes.
