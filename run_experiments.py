"""Run every experiment and write figures + a results summary to ./results.

    python run_experiments.py

Experiments
    1. Sample paths of GBM and OU
    2. Convergence of Monte Carlo to Black-Scholes, with and without variance reduction
    3. Strike sweep: MC vs Black-Scholes (GBM)
    4. Euler-Maruyama discretisation bias vs number of time steps (GBM)
    5. OU options: MC vs closed form, and what happens if Black-Scholes is misapplied to OU
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pricing_engine import GBM, OU, bs_price, mc_european, ou_option_price, simulate_paths

OUT = Path(__file__).parent / "results"
SEED = 2026

# GBM / Black-Scholes setup
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
# OU setup (a mean-reverting spread-like quantity, NOT an equity price)
X0, THETA, KAPPA, OU_SIGMA, OU_R, OU_T = 90.0, 100.0, 2.0, 15.0, 0.03, 1.0

METHODS = {
    "Plain": dict(antithetic=False, control_variate=False),
    "Antithetic": dict(antithetic=True, control_variate=False),
    "Control variate": dict(antithetic=False, control_variate=True),
    "Antithetic + control variate": dict(antithetic=True, control_variate=True),
}

# Chart styling: validated default categorical order (blue, orange, aqua, yellow), light surface
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- experiments


def exp_paths(rng) -> None:
    n_steps, n_paths = 252, 40
    t = np.linspace(0, T, n_steps + 1)
    g = simulate_paths(GBM(R, SIGMA), S0, T, n_steps, n_paths, rng)
    o = simulate_paths(OU(KAPPA, THETA, OU_SIGMA), X0, OU_T, n_steps, n_paths, rng)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        (axes[0], g, S0 * np.exp(R * t), "GBM paths (risk-neutral, r = 5%, sigma = 20%)", "Asset price"),
        (axes[1], o, THETA + (X0 - THETA) * np.exp(-KAPPA * t), "OU paths (kappa = 2, theta = 100, sigma = 15)", "Level"),
    ]
    for ax, p, mean, title, ylab in panels:
        ax.plot(t, p.T, color=BLUE, alpha=0.28, linewidth=1)
        ax.plot(t, mean, color=INK, linewidth=2, label="Expected path")
        ax.set_title(title)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left")
    axes[1].axhline(THETA, color=MUTED, linestyle="--", linewidth=1)
    axes[1].text(0.99, THETA + 1.5, "long-run mean", color=INK2, ha="right", fontsize=9)
    save(fig, "01_sample_paths.png")


def exp_convergence(rng) -> dict:
    ns = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    reps = 20
    truth = bs_price(S0, K, R, SIGMA, T)
    gbm = GBM(R, SIGMA)
    res = {m: {"rmse": [], "stderr": []} for m in METHODS}
    for n in ns:
        for name, kw in METHODS.items():
            errs, ses = [], []
            for _ in range(reps):
                r = mc_european(gbm, S0, K, R, T, n, rng, **kw)
                errs.append(r.price - truth)
                ses.append(r.stderr)
            res[name]["rmse"].append(float(np.sqrt(np.mean(np.square(errs)))))
            res[name]["stderr"].append(float(np.mean(ses)))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (name, d), color in zip(res.items(), [BLUE, ORANGE, AQUA, YELLOW]):
        ax.loglog(ns, d["rmse"], marker="o", markersize=6, color=color, label=name)
    ref = res["Plain"]["rmse"][0] * np.sqrt(ns[0] / np.array(ns))
    ax.loglog(ns, ref, color=MUTED, linestyle="--", linewidth=1, label=r"$N^{-1/2}$ reference")
    ax.set_xlabel("Number of simulated paths N")
    ax.set_ylabel(f"RMSE vs Black-Scholes ({reps} repeats)")
    ax.set_title("Monte Carlo error vs Black-Scholes (ATM call)")
    ax.legend()
    save(fig, "02_convergence.png")
    return {"ns": ns, "truth": truth, "reps": reps, "methods": res}


def exp_strikes(rng) -> list[dict]:
    gbm = GBM(R, SIGMA)
    rows = []
    for k in np.arange(70.0, 135.0, 5.0):
        r = mc_european(gbm, S0, k, R, T, 200_000, rng, antithetic=True, control_variate=True)
        bs = bs_price(S0, k, R, SIGMA, T)
        rows.append(
            dict(strike=float(k), bs=bs, mc=r.price, stderr=r.stderr, error=r.price - bs, covered=bool(r.covers(bs)))
        )
    ks = [x["strike"] for x in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    kk = np.linspace(70, 130, 200)
    axes[0].plot(kk, [bs_price(S0, k, R, SIGMA, T) for k in kk], color=BLUE, label="Black-Scholes")
    axes[0].plot(ks, [x["mc"] for x in rows], linestyle="none", marker="o", markersize=6, color=ORANGE, label="Monte Carlo")
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("Call price")
    axes[0].set_title("GBM call price across strikes")
    axes[0].legend()
    axes[1].axhline(0, color=AXIS, linewidth=1)
    axes[1].errorbar(
        ks, [x["error"] for x in rows], yerr=[1.96 * x["stderr"] for x in rows],
        fmt="o", markersize=6, color=ORANGE, ecolor=ORANGE, capsize=3, elinewidth=1.5,
    )
    axes[1].set_xlabel("Strike")
    axes[1].set_ylabel("MC minus Black-Scholes")
    axes[1].set_title("Pricing error with 95% interval")
    save(fig, "03_strike_sweep.png")
    return rows


def exp_euler() -> dict:
    gbm = GBM(R, SIGMA)
    truth = bs_price(S0, K, R, SIGMA, T)
    steps = [1, 2, 4, 8, 16, 32, 64, 128]
    rows = []
    for n in steps:
        rng = np.random.default_rng(SEED + n)
        r = mc_european(gbm, S0, K, R, T, 2_000_000, rng, n_steps=n, scheme="euler", antithetic=True)
        rows.append(dict(steps=n, dt=T / n, price=r.price, stderr=r.stderr, error=r.price - truth))
    exact = mc_european(gbm, S0, K, R, T, 2_000_000, np.random.default_rng(SEED), antithetic=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.errorbar(
        [x["dt"] for x in rows], [x["error"] for x in rows], yerr=[1.96 * x["stderr"] for x in rows],
        fmt="o-", markersize=6, color=BLUE, ecolor=BLUE, capsize=3, elinewidth=1.5, label="Euler-Maruyama",
    )
    ax.axhline(exact.price - truth, color=ORANGE, linewidth=1.5, linestyle="--", label="Exact sampling (1 step)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Time step dt (years), smaller to the right")
    ax.set_ylabel("MC minus Black-Scholes")
    ax.set_title("Discretisation bias of the Euler scheme (GBM call)")
    ax.legend()
    save(fig, "04_euler_bias.png")
    return {"truth": truth, "rows": rows, "exact_error": exact.price - truth, "exact_stderr": exact.stderr}


def exp_ou(rng) -> dict:
    ou = OU(KAPPA, THETA, OU_SIGMA)
    rows = []
    for k in np.arange(80.0, 125.0, 5.0):
        cf = ou_option_price(X0, k, OU_R, KAPPA, THETA, OU_SIGMA, OU_T)
        r = mc_european(ou, X0, k, OU_R, OU_T, 400_000, rng, antithetic=True, control_variate=True)
        mis = bs_price(X0, k, OU_R, OU_SIGMA / X0, OU_T)
        rows.append(
            dict(strike=float(k), closed_form=cf, mc=r.price, stderr=r.stderr, error=r.price - cf,
                 covered=bool(r.covers(cf)), bs_misapplied=mis)
        )
    ks = [x["strike"] for x in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    kk = np.linspace(80, 120, 200)
    axes[0].plot(kk, [ou_option_price(X0, k, OU_R, KAPPA, THETA, OU_SIGMA, OU_T) for k in kk], color=BLUE,
                 label="OU closed form")
    axes[0].plot(ks, [x["mc"] for x in rows], linestyle="none", marker="o", markersize=6, color=ORANGE,
                 label="OU Monte Carlo")
    axes[0].plot(kk, [bs_price(X0, k, OU_R, OU_SIGMA / X0, OU_T) for k in kk], color=AQUA, linestyle="--",
                 label="Black-Scholes misapplied")
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("Call price")
    axes[0].set_title("Option on an OU underlying")
    axes[0].legend()
    axes[1].axhline(0, color=AXIS, linewidth=1)
    axes[1].errorbar(
        ks, [x["error"] for x in rows], yerr=[1.96 * x["stderr"] for x in rows],
        fmt="o", markersize=6, color=ORANGE, ecolor=ORANGE, capsize=3, elinewidth=1.5,
    )
    axes[1].set_xlabel("Strike")
    axes[1].set_ylabel("MC minus closed form")
    axes[1].set_title("OU pricing error with 95% interval")
    save(fig, "05_ou_validation.png")
    return {"rows": rows}


# --------------------------------------------------------------------------- reporting


def write_summary(conv: dict, strikes: list, euler: dict, ou: dict) -> None:
    i = conv["ns"].index(100_000)
    plain = conv["methods"]["Plain"]["stderr"][i]
    lines = [
        "# Results summary",
        "",
        f"GBM setup: S0={S0}, K={K}, r={R}, sigma={SIGMA}, T={T}. Black-Scholes call = {conv['truth']:.4f}",
        "",
        "## Variance reduction at N = 100,000 paths",
        "",
        "| Method | Std error | Variance reduction vs plain |",
        "|---|---|---|",
    ]
    for name, d in conv["methods"].items():
        se = d["stderr"][i]
        lines.append(f"| {name} | {se:.5f} | {(plain / se) ** 2:.1f}x |")

    covered = sum(r["covered"] for r in strikes)
    lines += [
        "",
        "## GBM strike sweep (200,000 paths, antithetic + control variate)",
        "",
        "| Strike | Black-Scholes | Monte Carlo | Std error | 95% CI covers BS |",
        "|---|---|---|---|---|",
    ]
    for r in strikes:
        lines.append(
            f"| {r['strike']:.0f} | {r['bs']:.4f} | {r['mc']:.4f} | {r['stderr']:.4f} | {'yes' if r['covered'] else 'no'} |"
        )
    lines += ["", f"{covered} of {len(strikes)} intervals cover the Black-Scholes value (about 95% expected).", ""]

    lines += [
        "## Euler-Maruyama bias (2,000,000 antithetic paths)",
        "",
        "| Steps | dt | Price | MC minus BS | Std error |",
        "|---|---|---|---|---|",
    ]
    for r in euler["rows"]:
        lines.append(f"| {r['steps']} | {r['dt']:.4f} | {r['price']:.4f} | {r['error']:+.4f} | {r['stderr']:.4f} |")
    lines += [
        "",
        f"Exact sampling (1 step): error {euler['exact_error']:+.4f} (std error {euler['exact_stderr']:.4f}).",
        "",
    ]

    covered = sum(r["covered"] for r in ou["rows"])
    lines += [
        "## OU options (x0=90, theta=100, kappa=2, sigma=15, r=3%, T=1; 400,000 paths)",
        "",
        "| Strike | OU closed form | OU Monte Carlo | Std error | CI covers | BS misapplied |",
        "|---|---|---|---|---|---|",
    ]
    for r in ou["rows"]:
        lines.append(
            f"| {r['strike']:.0f} | {r['closed_form']:.4f} | {r['mc']:.4f} | {r['stderr']:.4f} | "
            f"{'yes' if r['covered'] else 'no'} | {r['bs_misapplied']:.4f} |"
        )
    lines += ["", f"{covered} of {len(ou['rows'])} intervals cover the closed-form value.", ""]
    (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    exp_paths(rng)
    print(f"[1/5] sample paths        {time.time() - t0:5.1f}s")
    conv = exp_convergence(rng)
    print(f"[2/5] convergence         {time.time() - t0:5.1f}s")
    strikes = exp_strikes(rng)
    print(f"[3/5] strike sweep        {time.time() - t0:5.1f}s")
    euler = exp_euler()
    print(f"[4/5] euler bias          {time.time() - t0:5.1f}s")
    ou = exp_ou(rng)
    print(f"[5/5] OU validation       {time.time() - t0:5.1f}s")

    write_summary(conv, strikes, euler, ou)
    (OUT / "results.json").write_text(
        json.dumps({"convergence": conv, "strikes": strikes, "euler": euler, "ou": ou}, indent=2)
    )
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
