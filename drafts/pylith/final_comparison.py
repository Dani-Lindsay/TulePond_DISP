#!/usr/bin/env python3
"""
Every PyLith inversion run in one table: gf/final_comparison.csv (+ a Markdown copy printed for STATUS.md).
Rows come from gf/results_comparison<tag>.csv; the settings (mesh, dip, Cd mode, surface, points) from the settings
json in gf/inversion_products<tag>.npz, or inferred from the tag for products made before 2026-09-30.
Provenance columns (2026-10-05; 08b refuses a table without them): sigma_lcurve_pick, sigma_corner, sigma_rule,
field_fingerprint (the 06 fp_maps.npz the run fits), made. A results table without them is from an old driver
run and stops this script.
Run in opera_disp_feb13:  python drafts/pylith/final_comparison.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def settings_for(tag):
    prod = os.path.join(HERE, "gf", f"inversion_products{tag}.npz")
    if os.path.exists(prod):
        z = np.load(prod, allow_pickle=True)
        if "settings" in z.files:
            return json.loads(str(z["settings"]))
    return dict(mesh="shallow" if "_shallow" in tag else "default", dip="90 (vertical)",
                cd_mode="diag" if "_diagcd" in tag else "full", cd_floor=None, use_surface="_nosurf" not in tag,
                points="default (2386 samples, 07 quadtree swath 4 km)")


PROVENANCE = ("sigma_lcurve_pick", "sigma_corner", "sigma_rule", "field_fingerprint", "made")
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "gf", "results_comparison*.csv"))):
    tag = re.sub(r"^results_comparison|\.csv$", "", os.path.basename(f))
    st = settings_for(tag)
    res = pd.read_csv(f).set_index("model")
    missing = [c for c in PROVENANCE if c not in res.columns]
    assert not missing, f"{f} lacks {missing}: made by the driver before 2026-10-05, re-run it"
    for model, r in res.iterrows():
        if not model.startswith("PyLith "):
            continue
        medium = model[7:]
        if medium == "sfcvmasis":
            continue                                             # dropped (Dani 2026-09-30): only the aligned SFCVM is used
        cd = st["cd_mode"] + (f"{st['cd_floor']}" if st["cd_mode"] == "floor" else "")
        rows.append(dict(run="default" + tag, medium=medium, mesh=st["mesh"], dip=st["dip"].split(" (")[0],
                         Cd=cd, surface="on" if st["use_surface"] else "off", points=st["points"].split(" (")[0],
                         sigma=r.sigma, rms_mm=r.rms_mm, rms_W300_mm=r.rms_west300_mm, rms_E300_mm=r.rms_east300_mm,
                         chi2=r.chi2, surface_pct=r.surface_pct, potency_m3=r.potency_m3, Mw_30gpa=r.Mw_30gpa,
                         Mw_sfcvm=r.Mw_sfcvm, summary_png=f"figures/summary_default{tag}_{medium}.png",
                         **{c: r[c] for c in PROVENANCE}))
table = pd.DataFrame(rows)
# bootstrap 2-sigma (pylith_sfcvm_tests.py) where available
for c in ("Mw_30gpa", "Mw_sfcvm", "surface_pct"):
    table[c + "_2sd"] = np.nan
for i, r in table.iterrows():
    f = os.path.join(HERE, "gf", f"uncertainty_{r.run}_{r.medium}.csv")
    if os.path.exists(f):
        u = pd.read_csv(f); s2 = u[u.kind == "boot_2sd"].iloc[0]
        for c in ("Mw_30gpa", "Mw_sfcvm", "surface_pct"):
            table.loc[i, c + "_2sd"] = s2[c]
# the 07 CSI baseline on the same 2386 samples, for reference
base = pd.read_csv(os.path.join(HERE, "gf", "results_comparison.csv")).set_index("model").loc["CSI homogeneous (07)"]
table = pd.concat([pd.DataFrame([dict(run="07 CSI baseline", medium="homog (Meade TDE)", mesh="07 smooth12", dip="90", Cd="full",
                                       surface="on", points="default", sigma=base.sigma, rms_mm=base.rms_mm,
                                       rms_W300_mm=base.rms_west300_mm, rms_E300_mm=base.rms_east300_mm, chi2=base.chi2,
                                       surface_pct=base.surface_pct, potency_m3=np.nan, Mw_30gpa=base.Mw_30gpa,
                                       Mw_sfcvm=base.Mw_sfcvm, summary_png="", **{c: base[c] for c in PROVENANCE})]), table],
                  ignore_index=True)
fingerprints = sorted(set(table.field_fingerprint.astype(str)))
assert len(fingerprints) == 1, f"the runs fit different 06 fields {fingerprints}: re-run the stale ones before collecting them"
out = os.path.join(HERE, "gf", "final_comparison.csv")
table.to_csv(out, index=False)
show = table.drop(columns=["summary_png", "field_fingerprint", "made"]).copy()
for c in ("rms_mm", "rms_W300_mm", "rms_E300_mm", "Mw_30gpa", "Mw_sfcvm"):
    show[c] = show[c].map(lambda v: f"{v:.2f}")
for c in ("chi2", "surface_pct", "potency_m3"):
    show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
for c in ("Mw_30gpa_2sd", "Mw_sfcvm_2sd", "surface_pct_2sd"):
    show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
show["sigma"] = show["sigma"].map(lambda v: f"{v:g}")
cols = list(show.columns)
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False)]
print("\n".join(lines))
print(f"\nwrote {out} ({len(table)} rows; every run on 06 field {fingerprints[0]}; sigma {sorted(set(table.sigma))})")
