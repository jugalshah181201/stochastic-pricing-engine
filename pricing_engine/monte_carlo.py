"""Monte Carlo pricing of European options with optional variance reduction.

Variance reduction:
    antithetic       pair each shock stream z with -z and average the two payoffs
    control_variate  regress the discounted payoff on the discounted terminal value,
                     whose expectation is known exactly for both GBM and OU

The two can be combined. Standard errors are computed on the *independent* unit
(a pair mean when antithetic, a single path otherwise), so they are honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .processes import simulate_terminal


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)

    def covers(self, value: float) -> bool:
        lo, hi = self.ci95
        return lo <= value <= hi


def _payoff(x: np.ndarray, k: float, option: str) -> np.ndarray:
    if option == "call":
        return np.maximum(x - k, 0.0)
    if option == "put":
        return np.maximum(k - x, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def mc_european(
    process,
    x0: float,
    k: float,
    r: float,
    T: float,
    n_paths: int,
    rng: np.random.Generator,
    option: str = "call",
    n_steps: int = 1,
    scheme: str = "exact",
    antithetic: bool = False,
    control_variate: bool = False,
) -> MCResult:
    """Price a European option by simulation.

    ``n_steps=1`` with ``scheme='exact'`` is enough for terminal-value payoffs;
    use more steps (and ``scheme='euler'``) to study discretisation bias.
    """
    disc = np.exp(-r * T)
    x, xa = simulate_terminal(process, x0, T, n_steps, n_paths, rng, scheme, antithetic)

    y = disc * _payoff(x, k, option)
    c = disc * x
    if antithetic:
        y = 0.5 * (y + disc * _payoff(xa, k, option))
        c = 0.5 * (c + disc * xa)

    if control_variate:
        beta = np.cov(y, c, ddof=1)[0, 1] / np.var(c, ddof=1)
        y = y - beta * (c - disc * process.mean_terminal(x0, T))

    n_units = y.size
    n_effective = 2 * n_units if antithetic else n_units
    return MCResult(price=float(y.mean()), stderr=float(y.std(ddof=1) / np.sqrt(n_units)), n_paths=n_effective)
