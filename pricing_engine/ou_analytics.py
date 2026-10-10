"""Closed-form price of a European option on an Ornstein-Uhlenbeck underlying.

Black-Scholes does NOT apply to an OU process: it assumes lognormal (GBM) dynamics.
Under OU the terminal value is Gaussian,

    X_T ~ N(m, v),   m = theta + (x0 - theta) e^{-kappa T},
                     v = sigma^2 (1 - e^{-2 kappa T}) / (2 kappa),

so a European option is a Bachelier-type expectation of a truncated normal:

    call = e^{-rT} [ (m - K) Phi(d) + sqrt(v) phi(d) ],   d = (m - K) / sqrt(v)

The OU dynamics are taken to be the pricing-measure dynamics (a modelling choice
for spreads, rates or commodities) and payoffs are discounted at the constant rate r.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ou_option_price(
    x0: float, k: float, r: float, kappa: float, theta: float, sigma: float, T: float, option: str = "call"
) -> float:
    m = theta + (x0 - theta) * np.exp(-kappa * T)
    sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * T)) / (2.0 * kappa))
    d = (m - k) / sd
    disc = np.exp(-r * T)
    if option == "call":
        return float(disc * ((m - k) * norm.cdf(d) + sd * norm.pdf(d)))
    if option == "put":
        return float(disc * ((k - m) * norm.cdf(-d) + sd * norm.pdf(d)))
    raise ValueError("option must be 'call' or 'put'")
