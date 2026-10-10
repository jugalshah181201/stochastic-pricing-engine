"""Stochastic processes: Geometric Brownian Motion and Ornstein-Uhlenbeck.

Both classes expose the same small interface so the Monte Carlo pricer does not
need to know which model it is simulating:

    step(x, z, dt, scheme)   advance the state by dt using standard normals z
    mean_terminal(x0, T)     exact E[X_T] (used as a control-variate mean)

Two time-stepping schemes are supported:

    "exact"  samples the exact transition distribution (no discretisation bias)
    "euler"  Euler-Maruyama (first-order weak convergence, biased for finite dt)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GBM:
    """dS = mu S dt + sigma S dW.

    For pricing, set ``mu`` to the risk-free rate r (risk-neutral measure).
    """

    mu: float
    sigma: float

    def step(self, x: np.ndarray, z: np.ndarray, dt: float, scheme: str = "exact") -> np.ndarray:
        if scheme == "exact":
            return x * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
        if scheme == "euler":
            return x * (1.0 + self.mu * dt + self.sigma * np.sqrt(dt) * z)
        raise ValueError(f"unknown scheme: {scheme!r}")

    def mean_terminal(self, x0: float, T: float) -> float:
        return x0 * np.exp(self.mu * T)


@dataclass(frozen=True)
class OU:
    """dX = kappa (theta - X) dt + sigma dW  (mean-reverting, Gaussian, can go negative).

    As an underlying for options this suits quantities such as commodity
    spreads, rates or volatility, not equity prices.
    """

    kappa: float
    theta: float
    sigma: float

    def step(self, x: np.ndarray, z: np.ndarray, dt: float, scheme: str = "exact") -> np.ndarray:
        if scheme == "exact":
            decay = np.exp(-self.kappa * dt)
            sd = self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.kappa * dt)) / (2.0 * self.kappa))
            return self.theta + (x - self.theta) * decay + sd * z
        if scheme == "euler":
            return x + self.kappa * (self.theta - x) * dt + self.sigma * np.sqrt(dt) * z
        raise ValueError(f"unknown scheme: {scheme!r}")

    def mean_terminal(self, x0: float, T: float) -> float:
        return self.theta + (x0 - self.theta) * np.exp(-self.kappa * T)

    def var_terminal(self, T: float) -> float:
        return self.sigma**2 * (1.0 - np.exp(-2.0 * self.kappa * T)) / (2.0 * self.kappa)


def simulate_paths(
    process,
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    scheme: str = "exact",
) -> np.ndarray:
    """Full sample paths, shape (n_paths, n_steps + 1). Used for plots and path-dependent payoffs."""
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for i in range(n_steps):
        paths[:, i + 1] = process.step(paths[:, i], rng.standard_normal(n_paths), dt, scheme)
    return paths


def simulate_terminal(
    process,
    x0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    scheme: str = "exact",
    antithetic: bool = False,
):
    """Terminal values only (memory is O(n_paths), not O(n_paths * n_steps)).

    Returns (x, x_anti). With ``antithetic=True`` the second array is driven by the
    negated shocks, so n_paths // 2 independent shock streams yield n_paths paths.
    Without it, ``x_anti`` is None.
    """
    dt = T / n_steps
    n = n_paths // 2 if antithetic else n_paths
    if n < 2:
        raise ValueError("need at least 2 independent draws")
    x = np.full(n, float(x0))
    xa = np.full(n, float(x0)) if antithetic else None
    for _ in range(n_steps):
        z = rng.standard_normal(n)
        x = process.step(x, z, dt, scheme)
        if antithetic:
            xa = process.step(xa, -z, dt, scheme)
    return x, xa
