"""Generate report/numbers.tex from results/results.json (single source of truth)."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "results.json")
OUT = os.path.join(HERE, "report", "numbers.tex")


def as_bool(v):
    return v is True or v == "True"


def fmt(x, sig=4):
    """Format a float as a LaTeX MATH literal like 3.39\\times10^{3}."""
    if x is None:
        return "--"
    x = float(x)
    if x == 0:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s or "E" in s:
        mant, exp = s.split("e")
        exp = int(exp)
        return f"{mant}\\times10^{{{exp}}}"
    return s


def tol_pow(tol):
    return f"10^{{{round(-math.log10(tol))}}}"


def main():
    with open(RES, encoding="utf-8") as f:
        r = json.load(f)

    L = []
    add = L.append

    ref = r["reference"]
    add(f"\\newcommand{{\\RefYA}}{{{ref['y40'][0]:.7f}}}")
    add(f"\\newcommand{{\\RefYB}}{{{fmt(ref['y40'][1], 5)}}}")
    add(f"\\newcommand{{\\RefYC}}{{{ref['y40'][2]:.7f}}}")
    add(f"\\newcommand{{\\RefDefect}}{{{fmt(ref['max_invariant_defect'], 2)}}}")

    conv = r["convergence"]
    add(f"\\newcommand{{\\OrdEulerRob}}{{{conv['robertson']['euler']['observed_order']:.2f}}}")
    add(f"\\newcommand{{\\OrdImplRob}}{{{conv['robertson']['implicit']['observed_order']:.2f}}}")
    add(f"\\newcommand{{\\OrdRKRob}}{{{conv['robertson']['rk4']['observed_order']:.2f}}}")

    stiff = r["stiffness"]["S_at_checkpoints"]
    add(f"\\newcommand{{\\StiffOne}}{{{fmt(stiff['0.0001'], 3)}}}")
    add(f"\\newcommand{{\\StiffTwo}}{{{fmt(stiff['0.01'], 3)}}}")
    add(f"\\newcommand{{\\StiffForty}}{{{fmt(stiff['40.0'], 4)}}}")
    add(f"\\newcommand{{\\LamMaxForty}}{{{fmt(r['stiffness']['lam_max_at_40'], 3)}}}")
    add(f"\\newcommand{{\\HBoundEuler}}{{{fmt(2.0 / r['stiffness']['lam_max_at_40'], 2)}}}")
    add(f"\\newcommand{{\\HBoundRK}}{{{fmt(2.8 / r['stiffness']['lam_max_at_40'], 2)}}}")

    ef = r["explicit_failure"]
    add(f"\\newcommand{{\\HCritEmp}}{{{fmt(ef['h_crit_empirical'], 2)}}}")
    rows = []
    for srow in ef["sweep"]:
        h = srow["h"]
        if as_bool(srow["stable"]):
            state = "stable (bounded)"
            minc = "$0$" if srow["min_component"] == 0 else f"${fmt(srow['min_component'], 2)}$"
            if not as_bool(srow["nonneg"]):
                state = "bounded, \\textbf{negative}"
        else:
            state = "\\textbf{blow-up}"
            minc = "--"
        rows.append(f"${fmt(h, 2)}$ & {state} & {minc} \\\\")
    add("\\newcommand{\\SweepTableBody}{\n" + "\n".join(rows) + "\n}")

    mc = r["matched_cost"]
    name_map = {"euler": "Explicit Euler", "rk4": "RK4", "implicit": "Implicit Euler"}
    rows = []
    for mname, mrows in mc["methods"].items():
        for row in mrows:
            if not row.get("feasible"):
                rows.append(f"{name_map[mname]} & ${tol_pow(row['target'])}$ & \\multicolumn{{4}}{{c}}{{infeasible}} \\\\")
                continue
            extra = []
            if row.get("h_over_h_stab") is not None and row["h_over_h_stab"] >= 0.75:
                extra.append("stab.-pinned")
            note = ("\\,\\,(" + "; ".join(extra) + ")") if extra else ""
            rows.append(
                f"{name_map[mname]} & ${tol_pow(row['target'])}$ & ${fmt(row['h'], 2)}$ & "
                f"${fmt(row['err40'], 2)}$ & {row['nfev']:,} & {row['wall_s']:.2f}\\,s{note} \\\\")
    add("\\newcommand{\\MatchedTableBody}{\n" + "\n".join(rows) + "\n}")

    ad = r["adaptive"]
    arows = []
    for arow in ad["runs"]:
        arows.append(
            f"${tol_pow(arow['tol'])}$ & ${fmt(arow['err40'], 2)}$ & {arow['accepted']:,} & "
            f"{arow['rejected']:,} & {arow['nfev']:,} & {arow['wall_s']:.2f}\\,s \\\\")
    add("\\newcommand{\\AdaptiveTableBody}{\n" + "\n".join(arows) + "\n}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("% AUTO-GENERATED from results/results.json -- do not edit\n")
        f.write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
