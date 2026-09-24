"""
run_all.py — Reproduce every figure and every number in the report.

Usage:
    python code/run_all.py

Requires: numpy, scipy, matplotlib (see ../requirements.txt).
Outputs:  ../figures/*.png and ../results/results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from problems import T_END, reference_solution
from stability import jacobian_eigenvalues_along_path
import experiments as ex


def main():
    t_start = time.perf_counter()
    results = {}

    print("[1/7] reference solution + validation ...", flush=True)
    results["reference"] = ex.exp1_reference()

    print("[2/7] convergence orders ...", flush=True)
    results["convergence"] = ex.exp2_convergence()

    print("[3/7] stiffness ratio along trajectory ...", flush=True)
    ref = reference_solution()
    results["stiffness"] = ex.exp7_stiffness(ref)

    lam_max = float(np.max(jacobian_eigenvalues_along_path(ref)["lam_max_abs_re"]))
    # preliminary explicit-Euler threshold from the eigenvalue bound, refined in [4/7]
    results["eigenvalue_bound_h"] = 2.0 / lam_max

    print("[4/7] explicit Euler failure sweep ...", flush=True)
    r4 = ex.exp4_explicit_failure()
    results["explicit_failure"] = r4
    h_crit = r4["h_crit_empirical"]

    print("[5/7] stability regions + h*lambda overlay ...", flush=True)
    eig_info = jacobian_eigenvalues_along_path(ref)
    results["stability_regions"] = ex.exp3_stability_regions(eig_info, h_crit)

    print("[6/7] cost vs accuracy at matched error ...", flush=True)
    y_ref40 = ref.y[:, -1]
    results["matched_cost"] = ex.exp5_matched_cost(y_ref40, lam_max)

    print("[7/7] adaptive step-doubling RK4 ...", flush=True)
    results["adaptive"] = ex.exp6_adaptive(y_ref40, ref)

    os.makedirs(os.path.dirname(ex.RESULTS_PATH), exist_ok=True)
    with open(ex.RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"done in {time.perf_counter() - t_start:.1f} s -> {ex.RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
