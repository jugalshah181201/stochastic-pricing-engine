"""Closed-form Black-Scholes prices and Greeks for European options (no dividends)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(s0: float, k: float, r: float, sigma: float, T: float):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(s0: float, k: float, r: float, sigma: float, T: float, option: str = "call") -> float:
    d1, d2 = _d1_d2(s0, k, r, sigma, T)
    if option == "call":
        return float(s0 * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2))
    if option == "put":
        return float(k * np.exp(-r * T) * norm.cdf(-d2) - s0 * norm.cdf(-d1))
    raise ValueError("option must be 'call' or 'put'")


def bs_greeks(s0: float, k: float, r: float, sigma: float, T: float, option: str = "call") -> dict:
    d1, d2 = _d1_d2(s0, k, r, sigma, T)
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1.0
    gamma = pdf / (s0 * sigma * np.sqrt(T))
    vega = s0 * pdf * np.sqrt(T)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}
