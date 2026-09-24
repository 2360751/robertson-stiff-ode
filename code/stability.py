"""
stability.py — Absolute-stability regions of the three required methods and
frozen-Jacobian eigenvalue diagnostics along the Robertson trajectory.

The stability region of a one-step method with amplification factor R(z) is
S = { z in C : |R(z)| <= 1 }.  For

* explicit Euler   R(z) = 1 + z                -> disk of radius 1 at -1
* RK4              R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
* implicit Euler   R(z) = 1/(1 - z)            -> exterior of disk radius 1 at +1

we evaluate |R| on a complex grid and draw the |R| = 1 contour.  This
"derived and plotted" requirement is met by stating R(z), drawing S, and
overlaying h*lambda for the frozen Jacobian of the Robertson system.
"""
from __future__ import annotations

import numpy as np

from problems import rhs, jac, reference_solution


def stability_grid(R, x_range=(-5.0, 3.0), y_range=(-4.0, 4.0), n=800):
    """Evaluate |R(z)| on a complex grid (z = x + i y)."""
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mag = np.abs(R(Z))
    return X, Y, mag


def jacobian_eigenvalues_along_path(ref):
    """Eigenvalues of the Robertson Jacobian at every reference output time.

    Returns t, eigenvalues (n_t, 3 complex), and derived diagnostics:
      lam_max_abs_re  = max |Re lambda| over nonzero eigenvalues
      stiffness ratio S(t) = max|Re lam| / min|Re lam| over the two nonzero
    eigenvalues (zero classified by threshold 1e-8, cf. Problem Pack).
    """
    t = ref.t
    eigs = np.empty((t.size, 3), dtype=complex)
    for i in range(t.size):
        eigs[i] = np.linalg.eigvals(jac(t[i], ref.y[:, i]))
    re = eigs.real
    absre = np.abs(re)
    nonzero = absre > 1e-8                     # structural-zero threshold
    lam_max = np.where(nonzero, absre, 0.0).max(axis=1)
    # for the ratio use the two largest |Re| values
    sorted_abs = np.sort(absre, axis=1)
    lam_big, lam_second = sorted_abs[:, -1], sorted_abs[:, -2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lam_second > 1e-8, lam_big / lam_second, np.nan)
    return {
        "t": t,
        "eigs": eigs,
        "lam_max_abs_re": lam_max,
        "lam_two_largest_abs_re": np.stack([lam_big, lam_second], axis=1),
        "stiffness_ratio": ratio,
    }
