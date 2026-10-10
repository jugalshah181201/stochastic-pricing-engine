import numpy as np
import pytest

from pricing_engine import GBM, OU, bs_greeks, bs_price, mc_european, ou_option_price

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
OU_ARGS = dict(x0=90.0, k=100.0, r=0.03, kappa=2.0, theta=100.0, sigma=15.0, T=1.0)


def rng(seed=7):
    return np.random.default_rng(seed)


def test_black_scholes_reference_value():
    assert bs_price(S0, K, R, SIGMA, T) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity_black_scholes():
    c = bs_price(S0, 95, R, SIGMA, T, "call")
    p = bs_price(S0, 95, R, SIGMA, T, "put")
    assert c - p == pytest.approx(S0 - 95 * np.exp(-R * T), abs=1e-10)


def test_delta_matches_finite_difference():
    h = 1e-4
    fd = (bs_price(S0 + h, K, R, SIGMA, T) - bs_price(S0 - h, K, R, SIGMA, T)) / (2 * h)
    assert bs_greeks(S0, K, R, SIGMA, T)["delta"] == pytest.approx(fd, abs=1e-6)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("kw", [{}, {"antithetic": True}, {"control_variate": True}, {"antithetic": True, "control_variate": True}])
def test_gbm_mc_matches_black_scholes(option, kw):
    res = mc_european(GBM(R, SIGMA), S0, K, R, T, 400_000, rng(), option=option, **kw)
    assert abs(res.price - bs_price(S0, K, R, SIGMA, T, option)) < 4 * res.stderr


def test_variance_reduction_lowers_stderr():
    g = GBM(R, SIGMA)
    se = {
        name: mc_european(g, S0, K, R, T, 200_000, rng(), **kw).stderr
        for name, kw in {
            "plain": {},
            "anti": {"antithetic": True},
            "cv": {"control_variate": True},
            "both": {"antithetic": True, "control_variate": True},
        }.items()
    }
    assert se["anti"] < se["plain"]
    assert se["cv"] < se["plain"]
    assert se["both"] < se["anti"] and se["both"] < se["cv"]


def test_euler_bias_shrinks_with_more_steps():
    g = GBM(R, SIGMA)
    truth = bs_price(S0, K, R, SIGMA, T)
    coarse = mc_european(g, S0, K, R, T, 400_000, rng(), n_steps=1, scheme="euler", antithetic=True)
    fine = mc_european(g, S0, K, R, T, 400_000, rng(), n_steps=64, scheme="euler", antithetic=True)
    assert abs(fine.price - truth) < abs(coarse.price - truth)


def test_exact_scheme_is_step_count_invariant():
    g = GBM(R, SIGMA)
    truth = bs_price(S0, K, R, SIGMA, T)
    res = mc_european(g, S0, K, R, T, 400_000, rng(), n_steps=12, scheme="exact", antithetic=True)
    assert abs(res.price - truth) < 4 * res.stderr


@pytest.mark.parametrize("option", ["call", "put"])
def test_ou_mc_matches_closed_form(option):
    a = OU_ARGS
    ou = OU(a["kappa"], a["theta"], a["sigma"])
    cf = ou_option_price(a["x0"], a["k"], a["r"], a["kappa"], a["theta"], a["sigma"], a["T"], option)
    res = mc_european(ou, a["x0"], a["k"], a["r"], a["T"], 400_000, rng(), option=option, antithetic=True, control_variate=True)
    assert abs(res.price - cf) < 4 * res.stderr


def test_ou_put_call_parity():
    a = OU_ARGS
    c = ou_option_price(**a, option="call")
    p = ou_option_price(**a, option="put")
    m = a["theta"] + (a["x0"] - a["theta"]) * np.exp(-a["kappa"] * a["T"])
    assert c - p == pytest.approx(np.exp(-a["r"] * a["T"]) * (m - a["k"]), abs=1e-10)


def test_ou_terminal_moments():
    a = OU_ARGS
    ou = OU(a["kappa"], a["theta"], a["sigma"])
    z = rng().standard_normal(500_000)
    x = ou.step(np.full(z.size, a["x0"]), z, a["T"])
    assert x.mean() == pytest.approx(ou.mean_terminal(a["x0"], a["T"]), abs=0.05)
    assert x.var() == pytest.approx(ou.var_terminal(a["T"]), rel=0.02)


def test_black_scholes_is_wrong_for_ou():
    """Misapplying BS to a mean-reverting underlying gives a materially different price."""
    a = OU_ARGS
    ou_price = ou_option_price(**a)
    bs_misapplied = bs_price(a["x0"], a["k"], a["r"], a["sigma"] / a["x0"], a["T"])
    assert abs(bs_misapplied - ou_price) > 1.0
