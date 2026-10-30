"""make_table.py: the supplementary sensitivity table (LaTeX) from the archived sweep csvs (Dani 2026-10-30: "I dont have
the table set up"). Reads sensitivity_2026-10-30/sensitivity_rows.csv and dip_rows.csv (scripts/sensitivity/*.py), writes
sensitivity_table.tex here. Numbers are copied, not recomputed. Run: ~/miniforge3/envs/opera_disp_feb13/bin/python make_table.py"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/work/proj_TulePond_DISP/sensitivity_2026-10-30")
rows = pd.read_csv(os.path.join(SRC, "sensitivity_rows.csv"))
dips = pd.read_csv(os.path.join(SRC, "dip_rows.csv"))

def get(row, variant):
    hit = rows[(rows.row == row) & (rows.variant == variant)]
    assert len(hit) == 1, (row, variant)
    return hit.iloc[0]

def dip(geometry, family):
    hit = dips[(dips.geometry == geometry) & (dips.family == family)]
    assert len(hit) == 1, (geometry, family)
    return hit.iloc[0]

# (group, label, mesh, record)
table = [("Selected model", "3D SFSVM, vertical", "60 m", get("Baseline (fine mesh)", "Selected")),
         ("Mesh", "250~m surface cells", "250 m", get("Baseline (default mesh)", "Selected")),
         ("Fault dip", "70$^\\circ$W", "60 m", get("Fault dip, fine mesh", "70 W")),
         ("", "70$^\\circ$E", "60 m", get("Fault dip, fine mesh", "70 E")),
         ("Regularization $\\sigma$", "0.25 $\\times$ corner", "60 m", get("Regularization sigma", "x0.25")),
         ("", "0.5 $\\times$ corner", "60 m", get("Regularization sigma", "x0.5")),
         ("", "2 $\\times$ corner", "60 m", get("Regularization sigma", "x2")),
         ("", "4 $\\times$ corner", "60 m", get("Regularization sigma", "x4")),
         ("Prior lengths $\\lambda$", "0.5 $\\times$", "60 m", get("Prior lengths", "x0.5")),
         ("", "2 $\\times$", "60 m", get("Prior lengths", "x2")),
         ("Data weighting", "Full covariance", "60 m", get("Data weighting", "Full Cd")),
         ("Near-trace exclusion", "$\\pm$30~m", "60 m", get("Near-trace exclusion", "30 m")),
         ("Reference frame", "Constant offset", "60 m", get("Reference frame", "Constant")),
         ("Elastic structure", "SFSVM unshifted", "250 m", get("Elastic structure", "SFSVM as is")),
         ("SFSVM alignment", "50~m east", "250 m", get("SFSVM alignment", "+50 m (e50)")),
         ("", "50~m west", "250 m", get("SFSVM alignment", "-50 m (w50)")),
         ("Data coherence", "0.7", "250 m", get("Data coherence", "0.7")),
         ("Homogeneous half-space", "Vertical", "--", dip("vertical", "Homogeneous analytic N^0")),
         ("", "70$^\\circ$W", "--", dip("dip70w", "Homogeneous analytic N^0")),
         ("", "60$^\\circ$W", "--", dip("dip60w", "Homogeneous analytic N^0")),
         ("", "50$^\\circ$W", "--", dip("dip50w", "Homogeneous analytic N^0")),
         ("", "70$^\\circ$E", "--", dip("dip70e", "Homogeneous analytic N^0"))]

lines = [r"\begin{table}", r"\centering", r"\caption{Caption}", r"\label{sup_tab:sensitivity}", r"\small",
         r"\begin{tabular}{llcccccc}", r"\hline",
         r"Test & Variant & RMS all & RMS west & RMS east & $M_w$ & Peak slip & Depth 90\% \\",
         r" & & (mm) & (mm) & (mm) & (30~GPa) & (mm) & (m) \\", r"\hline"]
for group, label, mesh, r in table:
    if mesh == "250 m" and group != "Mesh":
        label += "$^\\dagger$"
    lines.append(f"{group} & {label} & {r.rms_mm:.2f} & {r.rms_west_mm:.2f} & {r.rms_east_mm:.2f} & {r.Mw_30gpa:.2f} & "
                 f"{r.peak_slip_mm:.1f} & {r.depth90_m:.0f} \\\\")
lines += [r"\hline", r"\multicolumn{8}{l}{\footnotesize $^\dagger$ On the mesh with 250~m surface cells; compare with the Mesh row.} \\",
          r"\multicolumn{8}{l}{\footnotesize Homogeneous rows: the 250~m mesh fault, each triangle split into 16 for the analytic solution.} \\",
          r"\end{tabular}", r"\end{table}"]
with open(os.path.join(HERE, "sensitivity_table.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
