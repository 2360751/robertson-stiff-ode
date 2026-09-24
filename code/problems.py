"""
problems.py — The Robertson problem (topic 3 of the Problem Pack) and the
team-built reference solution.

The ODE system (H. H. Robertson, 1966), with y = [A, B, C]:

    y1' = -0.04 y1 + 1e4 y2 y3
    y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
    y3' =  3e7 y2^2
    y(0) = [1, 0, 0],   0 <= t <= 40

Notes that shape the experiments:
* (1,1,1)^T f(y) = 0, so the exact trajectory stays on the invariant plane
  y1 + y2 + y3 = 1.  The full 3x3 Jacobian therefore always has a zero
  eigenvalue; stiffness ratios are reported over the two nonzero eigenvalues.
* Reference solution: scipy.integrate.solve_ivp with Radau at very tight
  tolerance (rtol = 1e-12, per-component atol), repeated with tolerances
  tightened by a factor of 100; only digits unchanged by the repeat are
  reported.  This is our independent oracle (Project Brief, Section 2).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

K1 = 0.04
K2 = 3.0e7
K3 = 1.0e4

Y0 = np.array([1.0, 0.0, 0.0])
T0, T_END = 0.0, 40.0

# Output grid required by the Problem Pack: t = 0 plus at least 200
# logarithmically spaced times from 1e-8 to 40, used by every method.
T_EVAL = np.concatenate(([0.0], np.logspace(-8, np.log10(40.0), 250)))


def rhs(t, y):
    """Right-hand side f(t, y) of the Robertson system."""
    y1, y2, y3 = y
    return np.array([
        -K1 * y1 + K3 * y2 * y3,
        K1 * y1 - K3 * y2 * y3 - K2 * y2 * y2,
        K2 * y2 * y2,
    ])


def jac(t, y):
    """Analytic Jacobian of the Robertson system."""
    y1, y2, y3 = y
    return np.array([
        [-K1,            K3 * y3,           K3 * y2],
        [K1,  -K3 * y3 - 2.0 * K2 * y2,   -K3 * y2],
        [0.0,            2.0 * K2 * y2,          0.0],
    ])


# ---------------------------------------------------------------------------
# Reference solution (the team's own oracle)
# ---------------------------------------------------------------------------

def reference_solution(t_eval=T_EVAL, rtol=1e-12, atol=(1e-14, 1e-20, 1e-14)):
    """Tight-tolerance Radau reference on the requested output grid."""
    sol = solve_ivp(rhs, (T0, T_END), Y0, method="Radau", jac=jac,
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"reference solve failed: {sol.message}")
    return sol


def reference_check():
    """Solve the reference twice (tight, and 100x tighter) and report the
    digits of y(40) that survive.  Also verify the invariant plane."""
    ref1 = reference_solution(rtol=1e-12, atol=(1e-14, 1e-20, 1e-14))
    ref2 = reference_solution(rtol=1e-14, atol=(1e-16, 1e-22, 1e-16))
    y40_a = ref1.y[:, -1]
    y40_b = ref2.y[:, -1]
    absdiff = np.abs(y40_a - y40_b)
    # digits that agree: largest d such that |a-b| < 0.5 * 10^-d
    digits = np.where(absdiff > 0, np.floor(-np.log10(absdiff / 0.5)), 15).astype(int)
    defect = np.max(np.abs(ref1.y.sum(axis=0) - 1.0))
    return {
        "y40": y40_a.tolist(),
        "y40_tight": y40_b.tolist(),
        "absdiff_y40": absdiff.tolist(),
        "agreed_digits": (digits + 1).tolist(),   # significant digits retained
        "max_invariant_defect": float(defect),
        "min_component": float(ref1.y.min()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(reference_check(), indent=2))
