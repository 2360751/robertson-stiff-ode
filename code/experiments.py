"""
experiments.py — All numerical experiments for the Robertson project.

Each experiment returns a dict of numbers (also dumped to results.json by
run_all.py) and saves its figure(s) into ../figures/.  Figures are
publication-quality: log-log where appropriate, labelled, colorblind-safe
palette, one message per figure.
"""
from __future__ import annotations

import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problems import (K2, T0, T_END, Y0, rhs, jac, reference_solution,
                      reference_check, T_EVAL)
from solvers import (explicit_euler, rk4, implicit_euler, adaptive_rk4,
                     stab_poly_expl, stab_poly_rk4, stab_poly_impl_euler)
from stability import stability_grid, jacobian_eigenvalues_along_path

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "results.json")

# colorblind-safe palette (Okabe–Ito)
C_EULER = "#0072B2"     # blue
C_RK4 = "#D55E00"       # vermillion
C_IMPL = "#009E73"      # green
C_REF = "#555555"
C_ADAPT = "#CC79A7"     # purple-pink

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 10.5,
    "legend.frameon": False,
})


def _savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Experiment 1 — reference solution and its own validation
# ---------------------------------------------------------------------------

def exp1_reference():
    chk = reference_check()
    ref = reference_solution()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    labels = ["$y_1$ (A)", "$y_2$ (B)", "$y_3$ (C)"]
    for i in range(3):
        ax1.semilogx(ref.t[1:], ref.y[i, 1:], color=[C_EULER, C_RK4, C_IMPL][i], label=labels[i])
    ax1.set_xlabel("t")
    ax1.set_ylabel("concentration")
    ax1.set_title("(a) Radau reference, rtol $=10^{-12}$")
    ax1.legend()
    defect = np.abs(ref.y.sum(axis=0) - 1.0)
    ax2.semilogx(ref.t[1:], defect[1:], color=C_REF)
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$|y_1+y_2+y_3-1|$")
    ax2.set_title("(b) invariant-plane defect")
    _savefig(fig, "fig1_reference.png")
    chk["figure"] = "fig1_reference.png"
    return chk


# ---------------------------------------------------------------------------
# Experiment 2 — observed convergence orders
# ---------------------------------------------------------------------------

def _err_at(y_num, y_ref_val):
    return float(np.max(np.abs(y_num - y_ref_val)))


def _run_state(s):
    """Classify a fixed-step run: stable / non-negative flags."""
    y = s["y"]
    finite = np.all(np.isfinite(y))
    stable = finite and np.max(np.abs(y)) < 1e6
    nonneg = bool(np.min(y) >= 0.0) and finite
    return stable, nonneg


def exp2_convergence():
    out = {"scalar": {}, "robertson": {}}

    # ---- (a) scalar Dahlquist check: y' = -y, y(0) = 1, T = 2 -------------
    f_s = lambda t, y: -np.asarray(y, dtype=float)
    jac_s = lambda t, y: np.array([[-1.0]])
    y_exact = np.exp(-2.0)
    ladders = {"euler": 2 ** np.arange(8, 14), "rk4": 2 ** np.arange(8, 14),
               "implicit": 2 ** np.arange(8, 14)}
    for name, ns in ladders.items():
        hs, errs = [], []
        for n in ns:
            if name == "euler":
                s = explicit_euler(f_s, 0.0, [1.0], 2.0, int(n))
            elif name == "rk4":
                s = rk4(f_s, 0.0, [1.0], 2.0, int(n))
            else:
                s = implicit_euler(f_s, jac_s, 0.0, [1.0], 2.0, int(n))
            hs.append(s["h"]); errs.append(_err_at(s["y"][-1], y_exact))
        p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        out["scalar"][name] = {"h": hs, "err": errs, "observed_order": round(float(p), 2)}

    # ---- (b) Robertson baseline on [0, 0.5], errors at t = 0.5 -----------
    T_HALF = 0.5
    ref_half = reference_solution(t_eval=np.array([T_HALF]))
    y_ref_half = ref_half.y[:, -1]
    ns = [1024, 2048, 4096, 8192]     # h from 4.88e-4 down to 6.1e-5
    for name, runner in [("euler", lambda n: explicit_euler(rhs, 0.0, Y0, T_HALF, n)),
                         ("rk4", lambda n: rk4(rhs, 0.0, Y0, T_HALF, n)),
                         ("implicit", lambda n: implicit_euler(rhs, jac, 0.0, Y0, T_HALF, n))]:
        rows = []
        for n in ns:
            s = runner(n)
            stable, nonneg = _run_state(s)
            rows.append({"n": n, "h": s["h"], "err": _err_at(s["y"][-1], y_ref_half),
                         "stable": stable, "nonneg": nonneg})
        good = [r for r in rows if r["stable"] and r["err"] > 1e-13]
        p = float(np.polyfit(np.log([r["h"] for r in good]),
                             np.log([r["err"] for r in good]), 1)[0]) if len(good) >= 2 else None
        out["robertson"][name] = {"runs": rows, "observed_order": None if p is None else round(p, 2)}

    # ---- figure -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for name, col, lab in [("euler", C_EULER, "Explicit Euler (1)"),
                           ("rk4", C_RK4, "RK4 (4)"),
                           ("implicit", C_IMPL, "Implicit Euler (1)")]:
        d = out["scalar"][name]
        ax1.loglog(d["h"], d["err"], "o-", color=col, label=lab)
    for slope, col in [(1, "0.6"), (4, "0.6")]:
        h0 = np.array([2 / 256, 2 / 4096])
        ax1.loglog(h0, h0 ** slope * 3e-3, "--", color=col, lw=0.8)
    ax1.set_xlabel("step size $h$"); ax1.set_ylabel(r"$|y(2)-y_N|$")
    ax1.set_title("(a) scalar test $y'=-y$")
    ax1.legend(fontsize=8)

    for name, col, lab in [("euler", C_EULER, "Explicit Euler"),
                           ("rk4", C_RK4, "RK4"),
                           ("implicit", C_IMPL, "Implicit Euler")]:
        d = out["robertson"][name]
        hs = [r["h"] for r in d["runs"]]
        es = [max(r["err"], 1e-17) for r in d["runs"]]
        ax2.loglog(hs, es, "o-", color=col, label=lab)
    h0 = np.array([5e-4, 5e-5])
    ax2.loglog(h0, 2e-3 * (h0 / 5e-4), "--", color="0.6", lw=0.8, label="slope 1")
    ax2.loglog(h0, 1e-6 * (h0 / 5e-4) ** 4, ":", color="0.6", lw=0.8, label="slope 4")
    ax2.set_xlabel("step size $h$")
    ax2.set_ylabel(r"$\|y(0.5)-y_{ref}\|_\infty$")
    ax2.set_title("(b) Robertson, $0\\leq t\\leq 0.5$")
    ax2.legend(fontsize=8)
    _savefig(fig, "fig2_convergence.png")
    out["figure"] = "fig2_convergence.png"
    return out


# ---------------------------------------------------------------------------
# Experiment 3 — stability regions with h*lambda overlays
# ---------------------------------------------------------------------------

def exp3_stability_regions(eig_info, h_crit):
    states = [(1e-4, r"$t=10^{-4}$"), (1e-2, r"$t=10^{-2}$"), (40.0, r"$t=40$")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    methods = [("Explicit Euler", stab_poly_expl, (-4.2, 2.2), (-3.2, 3.2), C_EULER),
               ("RK4", stab_poly_rk4, (-3.4, 2.0), (-2.8, 2.8), C_RK4),
               ("Implicit Euler", stab_poly_impl_euler, (-2.4, 3.4), (-2.8, 2.8), C_IMPL)]
    for ax, (title, R, xr, yr, col) in zip(axes, methods):
        X, Y, mag = stability_grid(R, x_range=xr, y_range=yr, n=500)
        ax.contourf(X, Y, mag, levels=[0, 1], colors=["#cfe3f5"], alpha=0.9)
        ax.contour(X, Y, mag, levels=[1.0], colors=[col], linewidths=1.6)
        for (tv, tlab), mk in zip(states, ["o", "s", "^"]):
            i = int(np.argmin(np.abs(eig_info["t"] - tv)))
            lam = eig_info["eigs"][i]
            pts = h_crit * lam
            ax.plot(pts.real, pts.imag, mk, ms=6, mfc="none", mec="k", mew=1.1,
                    label=rf"$h_{{crit}}\lambda$ @{tlab}")
        ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
        ax.set_xlabel("Re$z$"); ax.set_ylabel("Im$z$")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
    _savefig(fig, "fig3_stability_regions.png")
    return {"figure": "fig3_stability_regions.png", "h_crit_used": h_crit,
            "note": ("shaded = |R(z)|<=1; markers show h_crit * lambda_j with the "
                     "frozen Jacobian at t = 1e-4, 1e-2, 40")}


# ---------------------------------------------------------------------------
# Experiment 4 — explicit Euler failure and the empirical stability threshold
# ---------------------------------------------------------------------------

def exp4_explicit_failure():
    h_list = [3e-4, 4e-4, 5e-4, 5.5e-4, 6e-4, 7e-4, 8e-4, 1e-3]
    sweep = []
    for h in h_list:
        s = explicit_euler(rhs, 0.0, Y0, T_END, int(round(T_END / h)))
        stable, nonneg = _run_state(s)
        sweep.append({"h": h, "stable": stable, "nonneg": nonneg,
                      "min_component": float(np.min(s["y"])) if np.all(np.isfinite(s["y"])) else None,
                      "max_abs": float(np.max(np.abs(s["y"]))) if np.all(np.isfinite(s["y"])) else None})
    h_stable = [r["h"] for r in sweep if r["stable"]]
    h_crit = max(h_stable) if h_stable else 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    show = [4e-4, 6e-4, 8e-4, 2e-3]
    for h in show:
        n = int(round(min(T_END / h, 400000)))
        s = explicit_euler(rhs, 0.0, Y0, T_END if n == int(round(T_END / h)) else n * h, n)
        y2 = np.clip(s["y"][:, 1], 1e-16, None)
        ax1.loglog(s["t"][1:], y2[1:], lw=1.0, label=f"$h={h:g}$")
    ref = reference_solution()
    ax1.loglog(ref.t[1:], np.clip(ref.y[1, 1:], 1e-16, None), "--", color=C_REF, lw=1.2,
               label="reference")
    ax1.set_xlabel("t"); ax1.set_ylabel("$y_2$")
    ax1.set_title("(a) explicit Euler: $y_2$ vs. reference")
    ax1.legend(fontsize=7.5)

    for h in show:
        n = int(round(T_END / h))
        if n > 400000:
            continue
        s = explicit_euler(rhs, 0.0, Y0, T_END, n)
        minc = np.minimum(np.minimum(s["y"][:, 0], s["y"][:, 1]), s["y"][:, 2])
        ax2.semilogx(s["t"][1:], minc[1:], lw=1.0, label=f"$h={h:g}$")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylim(-0.05, 0.02)     # keep positivity violations readable; a
    # diverging run leaves the box (annotated) instead of rescaling the axis
    ax2.annotate("divergence\n(leaves the box)", xy=(2.0, -0.045), fontsize=7,
                 color="0.35")
    ax2.set_xlabel("t"); ax2.set_ylabel(r"$\min_i\, y_i(t)$")
    ax2.set_title("(b) positivity violations")
    ax2.legend(fontsize=7.5)
    _savefig(fig, "fig4_explicit_failure.png")

    return {"sweep": sweep, "h_crit_empirical": h_crit,
            "eigenvalue_bound": float(2.0 / eig_max(refoverride=None)),
            "figure": "fig4_explicit_failure.png"}


def eig_max(refoverride):
    ref = reference_solution()
    info = jacobian_eigenvalues_along_path(ref)
    return float(np.max(info["lam_max_abs_re"]))


# ---------------------------------------------------------------------------
# Experiment 5 — cost vs accuracy at matched error on the full interval
# ---------------------------------------------------------------------------

def _solve_by_name(name, n):
    if name == "euler":
        return explicit_euler(rhs, 0.0, Y0, T_END, n)
    if name == "rk4":
        return rk4(rhs, 0.0, Y0, T_END, n)
    if name == "implicit":
        return implicit_euler(rhs, jac, 0.0, Y0, T_END, n)
    raise ValueError(name)


def _err40(s, y_ref40):
    y = s["y"][-1]
    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e6:
        return np.inf
    return _err_at(y, y_ref40)


def exp5_matched_cost(y_ref40, lam_max):
    """Cost vs accuracy at matched error targets on the full interval [0,40].

    Strategy per (method, target): descend from a large cheap step until a
    stable run with err <= 4*target is found, then use the method's order p
    to predict the step h* = (target / (err/h^p))^(1/p); verify with one run
    and correct the prediction at most twice.  This keeps the number of
    expensive fine-step runs to a handful.

    A run is 'stability-pinned' when its chosen h sits at the explicit
    stability limit (h*|lambda|max ~ 2 for Euler, ~2.8 for RK4) — accuracy
    was not the binding constraint there.
    """
    targets = [1e-2, 1e-3, 1e-4]
    orders = {"euler": 1, "rk4": 4, "implicit": 1}
    h_stab = {"euler": 2.0 / lam_max, "rk4": 2.8 / lam_max, "implicit": np.inf}
    out = {"targets": targets, "methods": {}, "lam_max": lam_max,
           "h_stab_bounds": {k: (None if not np.isfinite(v) else v)
                             for k, v in h_stab.items()}}
    h_lo, h_hi = 2e-5, 2.0
    for name in ["euler", "rk4", "implicit"]:
        p = orders[name]
        rows = []
        for target in targets:
            # --- descent: find any h with a stable run and err <= 4*target
            h = h_hi
            e, n = _err_for_h(name, h, y_ref40)
            tries = 0
            while (not np.isfinite(e) or e > 4 * target) and tries < 24:
                h /= 2.0
                e, n = _err_for_h(name, h, y_ref40)
                tries += 1
            if not np.isfinite(e) or e > 4 * target:
                rows.append({"target": target, "feasible": False})
                continue
            # --- order-p prediction, verified and corrected.
            # For explicit methods the prediction is additionally capped at
            # 0.9x the empirical stability limit: a larger step is useless
            # no matter what the accuracy model predicts.
            best = None
            h_cap = (0.9 * h_stab[name]) if np.isfinite(h_stab[name]) else h_hi
            for _ in range(4):
                C = e / h ** p
                h_pred = (target / C) ** (1.0 / p)
                h_pred = min(max(h_pred, h_lo), h * 1.3, h_cap)
                e2, n2 = _err_for_h(name, h_pred, y_ref40)
                if np.isfinite(e2) and e2 <= 2 * target:
                    best = (h_pred, n2, e2)
                    break
                if not np.isfinite(e2):
                    # predicted step unstable: retreat by halves until stable
                    h_t = h_pred
                    for _ in range(6):
                        h_t /= 2.0
                        e2, n2 = _err_for_h(name, h_t, y_ref40)
                        if np.isfinite(e2):
                            break
                    if not np.isfinite(e2):
                        break
                h, e = h_pred if np.isfinite(e2) else h_t, e2
            if best is None:
                best = (h, n, e)
            h_best, n_best, e_best = best
            t0 = time.perf_counter()
            s = _solve_by_name(name, n_best)
            wall = time.perf_counter() - t0
            e_final = _err40(s, y_ref40)
            stable, nonneg = _run_state(s)
            pinned = np.isfinite(h_stab[name]) and h_best > 0.9 * h_stab[name]
            rows.append({"target": target, "feasible": True, "h": h_best,
                         "n_steps": n_best, "err40": e_final, "stable": stable,
                         "nonneg": nonneg, "wall_s": round(wall, 3),
                         "nfev": s["nfev"], "njev": s["njev"],
                         "newton_iter": s["newton_iter"],
                         "stability_pinned": bool(pinned),
                         "h_over_h_stab": (None if not np.isfinite(h_stab[name])
                                           else round(h_best / h_stab[name], 3))})
        out["methods"][name] = rows

    # ---- figure: error vs cost (nfev), one marker per matched target ------
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    style = {"euler": (C_EULER, "o", "Explicit Euler (1)"),
             "rk4": (C_RK4, "s", "RK4 (4)"),
             "implicit": (C_IMPL, "^", "Implicit Euler (1)")}
    for name, rows in out["methods"].items():
        col, mk, lab = style[name]
        xs = [r["nfev"] for r in rows if r.get("feasible")]
        ys = [r["err40"] for r in rows if r.get("feasible")]
        ax.loglog(xs, ys, mk + "-", color=col, label=lab)
        for r in rows:
            if r.get("feasible") and r.get("stability_pinned"):
                ax.annotate("stab.", (r["nfev"], r["err40"]),
                            textcoords="offset points", xytext=(4, -8),
                            fontsize=7, color=col)
    ax.set_xlabel("right-hand-side evaluations (cost measure)")
    ax.set_ylabel(r"$\|y(40)-y_{ref}\|_\infty$")
    ax.set_title("cost vs. error at matched targets")
    ax.legend(fontsize=8)
    _savefig(fig, "fig5_matched_cost.png")
    out["figure"] = "fig5_matched_cost.png"
    return out


def _err_for_h(name, h, y_ref40):
    n = max(2, int(round(T_END / h)))
    s = _solve_by_name(name, n)
    return _err40(s, y_ref40), n


# ---------------------------------------------------------------------------
# Experiment 6 — adaptive step-doubling RK4
# ---------------------------------------------------------------------------

def exp6_adaptive(y_ref40, ref):
    tols = [1e-4, 1e-5, 1e-6]
    out = {"runs": []}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    pos = lambda y: bool(np.any(np.asarray(y) < 0))   # positivity veto
    for tol, col in zip(tols, [C_EULER, C_RK4, C_IMPL]):
        t0 = time.perf_counter()
        s = adaptive_rk4(rhs, 0.0, Y0, T_END, h0=1e-5, tol=tol, atol=1e-9,
                         h_max=0.05, reject=pos)
        wall = time.perf_counter() - t0
        err = _err40(s, y_ref40)
        out["runs"].append({"tol": tol, "err40": err, "wall_s": round(wall, 3),
                            "accepted": s["accepted"], "rejected": s["rejected"],
                            "nfev": s["nfev"], "h_min": float(s["h_accepted"].min()),
                            "h_max": float(s["h_accepted"].max())})
        ax1.loglog(s["t"][1:], s["h_accepted"], lw=0.9, color=col, label=f"tol={tol:g}")
    # eigenvalue-implied explicit RK4 stability limit along the trajectory
    info = jacobian_eigenvalues_along_path(ref)
    t_eig = info["t"]
    h_stab = 2.8 / np.maximum(info["lam_max_abs_re"], 1e-12)
    ax1.loglog(t_eig[1:], h_stab[1:], "--", color="0.4", lw=1.0,
               label="$2.8/|\\lambda|_{max}$ (stability)")
    ax1.set_xlabel("t"); ax1.set_ylabel("accepted half-step size")
    ax1.set_title("(a) adaptive RK4: step size profile")
    ax1.legend(fontsize=7.5)

    errs = [r["err40"] for r in out["runs"]]
    ax2.loglog(tols, errs, "o-", color=C_ADAPT)
    ax2.loglog(tols, tols, "--", color="0.6", lw=0.8, label="$err=tol$")
    ax2.set_xlabel("tolerance"); ax2.set_ylabel(r"$\|y(40)-y_{ref}\|_\infty$")
    ax2.set_title("(b) global error vs. tolerance")
    ax2.legend(fontsize=8)
    _savefig(fig, "fig6_adaptive.png")
    out["figure"] = "fig6_adaptive.png"
    out["note"] = ("error plateaus for small tol: once h is pinned at the RK4 "
                   "stability limit, a tighter tolerance cannot reduce the "
                   "global error -- adaptivity buys accuracy, not stability")
    return out


# ---------------------------------------------------------------------------
# Experiment 7 — eigenvalues and stiffness ratio along the trajectory
# ---------------------------------------------------------------------------

def exp7_stiffness(ref):
    info = jacobian_eigenvalues_along_path(ref)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    t = info["t"]
    lam1, lam2 = info["lam_two_largest_abs_re"][:, 0], info["lam_two_largest_abs_re"][:, 1]
    ax1.loglog(t[1:], lam1[1:], color=C_EULER, label=r"$\max_j|\mathrm{Re}\,\lambda_j|$ (nonzero)")
    ax1.loglog(t[1:], np.maximum(lam2[1:], 1e-8), color=C_RK4, label=r"$2^{\mathrm{nd}}$ nonzero")
    ax1.set_xlabel("t"); ax1.set_ylabel(r"$|\mathrm{Re}\,\lambda|$")
    ax1.set_title("(a) frozen-Jacobian eigenvalues")
    ax1.legend(fontsize=8)
    S = info["stiffness_ratio"]
    ax2.loglog(t[1:], S[1:], color=C_IMPL, label="$S(t)$ (two nonzero eigs)")
    for tv, sv in [(1e-4, 3.0e3), (1e-2, 5.4e3), (40.0, 1.58e5)]:
        ax2.plot(tv, sv, "kx", ms=7, mew=1.4,
                 label="Problem Pack check" if tv == 1e-4 else None)
    ax2.set_xlabel("t"); ax2.set_ylabel("$S(t)$")
    ax2.set_title("(b) stiffness ratio")
    ax2.legend(fontsize=8)
    _savefig(fig, "fig7_stiffness.png")
    return {
        "S_at_checkpoints": {str(tv): float(S[int(np.argmin(np.abs(t - tv)))])
                             for tv in [1e-4, 1e-2, 40.0]},
        "lam_max_at_40": float(info["lam_max_abs_re"][-1]),
        "figure": "fig7_stiffness.png",
    }
