"""
solvers.py — Fixed-step and adaptive IVP solvers built for the Robertson project.

Library layout
--------------
Every solver takes a right-hand-side ``f(t, y)`` (and, where needed, an analytic
Jacobian ``jac(t, y)``) so the problem can be swapped without touching the code.

Cost accounting: each solver records the quantities we quote in the report —
right-hand-side evaluations (nfev), Jacobian evaluations (njev), Newton
iterations (newton_iter), and accepted/rejected steps for the adaptive
controller.  Wall time is measured by the experiments, not inside the solvers.

Methods implemented
-------------------
* explicit_euler        — order 1
* rk4                   — order 4 (classical four-stage Runge–Kutta)
* implicit_euler        — order 1, backward Euler; the implicit relation is
                          solved with Newton's iteration using the analytic
                          Jacobian of the *modified* system
                          g(y_{n+1}) = y_{n+1} - y_n - h f(t_{n+1}, y_{n+1}).
* adaptive_rk4          — order 4 with step-doubling local-error estimation
                          (full step vs. two half steps), tolerance-driven
                          accept/reject with the standard PI-free update.

All solvers return a dict with time grid, solution values and cost counters.
"""
from __future__ import annotations

import numpy as np

NEWTON_TOL = 1e-10     # relative-ish stopping tolerance for Newton (see report)
NEWTON_MAXIT = 20


class Cost:
    """Mutable counter object shared by a solver run."""

    def __init__(self) -> None:
        self.nfev = 0
        self.njev = 0
        self.newton_iter = 0
        self.accepted = 0
        self.rejected = 0

    def as_dict(self) -> dict:
        return {
            "nfev": self.nfev,
            "njev": self.njev,
            "newton_iter": self.newton_iter,
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


def _as_array(y0):
    y = np.asarray(y0, dtype=float)
    return y.copy()


def explicit_euler(f, t0, y0, t_end, n_steps):
    """Explicit (forward) Euler with uniform step h = (t_end - t0)/n_steps."""
    cost = Cost()
    y = _as_array(y0)
    h = (t_end - t0) / n_steps
    ts = np.linspace(t0, t_end, n_steps + 1)
    ys = np.empty((n_steps + 1, y.size))
    ys[0] = y
    for i in range(n_steps):
        t = ts[i]
        cost.nfev += 1
        y = y + h * f(t, y)
        ys[i + 1] = y
    return {"t": ts, "y": ys, "h": h, **cost.as_dict()}


def rk4(f, t0, y0, t_end, n_steps):
    """Classical explicit RK4 with uniform step h."""
    cost = Cost()
    y = _as_array(y0)
    h = (t_end - t0) / n_steps
    ts = np.linspace(t0, t_end, n_steps + 1)
    ys = np.empty((n_steps + 1, y.size))
    ys[0] = y
    for i in range(n_steps):
        t = ts[i]
        cost.nfev += 4
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[i + 1] = y
    return {"t": ts, "y": ys, "h": h, **cost.as_dict()}


def implicit_euler(f, jac, t0, y0, t_end, n_steps):
    """Implicit (backward) Euler with Newton's iteration on the stage equation.

    Stage problem:  g(v) = v - y_n - h f(t_{n+1}, v) = 0
    Newton update:  (I - h J(v)) d = -g(v);   v <- v + d
    """
    cost = Cost()
    y = _as_array(y0)
    h = (t_end - t0) / n_steps
    ts = np.linspace(t0, t_end, n_steps + 1)
    ys = np.empty((n_steps + 1, y.size))
    ys[0] = y
    I = np.eye(y.size)
    for i in range(n_steps):
        t_next = ts[i + 1]
        v = y.copy()                       # initial guess: previous state
        ok = False
        for _ in range(NEWTON_MAXIT):
            cost.nfev += 1
            g = v - y - h * f(t_next, v)
            cost.njev += 1
            J = jac(t_next, v)
            A = I - h * J
            try:
                d = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                break                      # singular stage matrix: diverged
            if not np.all(np.isfinite(d)):
                break
            v = v + d
            cost.newton_iter += 1
            scale = max(1.0, np.linalg.norm(v, np.inf))
            if np.linalg.norm(d, np.inf) <= NEWTON_TOL * scale:
                ok = True
                break
        if not ok or not np.all(np.isfinite(v)) or np.linalg.norm(v, np.inf) > 1e12:
            # diverged: fill the rest with NaN so callers can detect failure
            ys[i + 1:] = np.nan
            break
        y = v
        ys[i + 1] = y
    return {"t": ts, "y": ys, "h": h, **cost.as_dict()}


def adaptive_rk4(f, t0, y0, t_end, h0, tol, atol=1e-6, h_min=1e-12, h_max=None,
                 progress=None, reject=None):
    """Step-doubling adaptive RK4 with componentwise error control.

    One full step of size h is compared with two half steps.  The difference
    ``delta = y_full - y_two_half`` estimates the error of the *half-step*
    solution scaled by (2^p - 1) with p = 4, so the per-half-step local error
    estimate is  err_i = |delta_i| / 15 (componentwise).

    Acceptance: err_i <= tol * |y_i| + atol  for every component i.

    Componentwise control matters for problems with widely separated solution
    scales (Robertson: y2 = O(1e-5) against y1 = O(1)): an inf-norm test lets
    a slightly negative overshoot of the tiny component slip through, and on
    this problem a negative y2 runs away quadratically (y2' ~ -3e7 y2^2),
    silently corrupting the whole state.

    ``reject(y_candidate)`` is an optional problem-aware veto (e.g. a
    positivity check ``lambda y: np.any(y < 0)`` — one of the diagnostics the
    Problem Pack asks to track).  Candidates it flags are rejected and the
    step is retried smaller, exactly like a tolerance failure.

    Robustness extras (all generic, none problem-specific):
    * a state whose norm explodes (> 1e8 x initial norm) is never accepted;
    * a step whose candidate values overflow (NaN/inf) is rejected and the
      size is recorded as an empirical stability limit, capping all
      subsequent growth;
    * if even h = h_min cannot satisfy the test, DivergenceError is raised
      instead of grinding forever.

    The controller returns the *half-step* result when a step is accepted.
    """
    class DivergenceError(RuntimeError):
        pass

    cost = Cost()
    y = _as_array(y0)
    y_scale0 = max(1.0, np.linalg.norm(y, np.inf))
    t = t0
    h = min(h0, t_end - t0) if h_max is None else min(h0, h_max, t_end - t0)
    ts = [t]
    ys = [y.copy()]
    hs = []            # accepted step sizes (the half-step h/2 actually used)
    h_stab_cap = np.inf   # step size known to be UNSTABLE (overflow evidence)
    growth_cap = 5.0      # max growth per accepted step (tightened after rejects)
    while t < t_end - 1e-14:
        if progress is not None:
            progress(cost.accepted, cost.rejected, t, h)
        h = min(h, t_end - t)
        cost.nfev += 11                     # 4 (full) + 2*4 (halves) + 1 f(t+h, y_full)
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y_full = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        k1a = k1
        k2a = f(t + h / 4, y + h / 4 * k1a)
        k3a = f(t + h / 4, y + h / 4 * k2a)
        k4a = f(t + h / 2, y + h / 2 * k3a)
        y_a = y + h / 12 * (k1a + 2 * k2a + 2 * k3a + k4a)

        k1b = f(t + h / 2, y_a)
        k2b = f(t + h * 3 / 4, y_a + h / 4 * k1b)
        k3b = f(t + h * 3 / 4, y_a + h / 4 * k2b)
        k4b = f(t + h, y_a + h / 2 * k3b)
        y_b = y_a + h / 12 * (k1b + 2 * k2b + 2 * k3b + k4b)

        delta_vec = np.abs(y_full - y_b)
        y_norm = np.linalg.norm(y_b, np.inf)

        if not np.all(np.isfinite(delta_vec)) or y_norm > 1e8 * y_scale0:
            # candidate step diverged (stability limit exceeded): reject,
            # shrink, and retry from the same state — never advance a NaN.
            # Record the unstable size so growth never re-enters it.
            cost.rejected += 1
            h_stab_cap = min(h_stab_cap, h)
            growth_cap = 1.3
            h = max(h * 0.25, h_min)
            if h <= h_min:
                raise DivergenceError(
                    f"adaptive RK4 diverged at t={t:.6g}; h already at h_min")
            continue

        err_vec = delta_vec / 15.0
        sc = tol * np.abs(y_b) + atol          # per-component tolerance
        ratio = float(np.max(err_vec / sc))    # >1 means reject
        vetoed = reject is not None and bool(reject(y_b))
        if (ratio <= 1.0 and not vetoed) or (h <= h_min and not vetoed):
            # accept the half-step solution
            y = y_b
            t = t + h
            if t <= ts[-1] or t_end - t <= 1e-10 * max(1.0, abs(t_end)):
                t = t_end          # snap: avoid floating-point stall at the end
            cost.accepted += 1
            ts.append(t)
            ys.append(y.copy())
            hs.append(h / 2)
            if ratio > 0:
                h = h * min(growth_cap, max(0.2, 0.9 / ratio ** 0.2))
            else:
                h = h * growth_cap
            if h_max is not None:
                h = min(h, h_max)
            # stay below the empirically observed stability limit
            h = min(h, 0.85 * h_stab_cap)
            h = max(h, h_min)
            growth_cap = min(5.0, growth_cap * 1.7)   # ease back after rejects
        else:
            cost.rejected += 1
            growth_cap = 1.3
            if vetoed:
                h = h * 0.5            # problem-aware veto: modest shrink
            else:
                h = h * max(0.1, 0.9 / ratio ** 0.2)
            if h < h_min:
                if reject is not None and reject(y):
                    raise DivergenceError(
                        f"state violates reject() at t={t:.6g} with h at h_min")
                h = h_min
    return {
        "t": np.array(ts),
        "y": np.array(ys),
        "h_accepted": np.array(hs),
        "tol": tol,
        **cost.as_dict(),
    }


# ----------------------------------------------------------------------------
# Stability polynomials / regions (for the analytical plots)
# ----------------------------------------------------------------------------

def stab_poly_expl(z):
    """R(z) for explicit Euler: R(z) = 1 + z."""
    return 1.0 + z


def stab_poly_rk4(z):
    """R(z) for classical RK4: 1 + z + z^2/2 + z^3/6 + z^4/24."""
    return 1.0 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24


def stab_poly_impl_euler(z):
    """R(z) for implicit Euler: 1/(1 - z)."""
    return 1.0 / (1.0 - z)
