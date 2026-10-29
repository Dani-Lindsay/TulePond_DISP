"""gf_check.py: verification of the PyLith Green's functions (registers #8, #9): per impulse, the homogeneous PyLith G against
the ANALYTIC solution for the same tent (CSI triangular dislocations, Meade 2007, on the same fault mesh:
model_utils.analytic_tent_G, n_sub 8). Ported 2026-10-30 from order2_surface_tent_check_2026-10-25/tent_check.py and
tent_check_mesh.py, which built their own sub-triangles and set each analytic column's sign to agree with PyLith
(tent_check.py l.116). Here the analytic sign is FIXED (analytic_tent_G, tested); a column whose sign disagrees is recorded
(`sign_agrees` False), not flipped.

A  element order on the default mesh: gf_default_homog (order 1) and gf_default_homog_order2, 32 impulses in km 57.5-61
   (14 at the surface, 6 each near 100, 300, 1000 m; the 2026-10-25 selection).
B  mesh refinement at order 2 in the strip km 58.85-59.25: gf_default_homog_order2, gf_fine_homog_order2_strip,
   gf_shallow_homog_order2_strip (up to 6 impulses per depth group; the 2026-10-25 selection).
Metric per impulse: relative RMS error |G_PyLith - G_analytic| / |G_analytic| over the 2386 samples; amplitude ratio.
Output (model_config.sensitivity_dir): gf_check.csv (one row per run and impulse; `panel` A or B).
Run (repo root): PYTHONPATH=. nice -n 15 python scripts/sensitivity/gf_check.py      (~10 min)
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyproj

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                         # noqa: E402
from common import cfg, model_utils                   # noqa: E402
from TulePond_filepaths import paths_uniform, model_crs, model_frame_lonlat   # noqa: E402

N_SUB = 8
DEPTH_BINS_M = [-1, 1, 100, 200, 400, 2000]
DEPTH_LABELS = ["Surface", "1-100 m", "100-200 m", "200-400 m", "400-2000 m"]
PANEL_A = dict(runs=("gf_default_homog", "gf_default_homog_order2"), km=(57.5, 61.0),
               take=((-1, 1, 14), (40, 140, 6), (250, 350, 6), (900, 1100, 6)))
PANEL_B = dict(runs=("gf_default_homog_order2", "gf_fine_homog_order2_strip", "gf_shallow_homog_order2_strip"), km=(58.85, 59.25),
               take=((-1, 1, 6), (1, 100, 6), (100, 200, 6), (200, 400, 6), (800, 1100, 4)))


def choose(tents, km_range, take):
    """The 2026-10-25 impulse selection: in each depth range (m), n impulses evenly spread along strike in km_range."""
    km, depth = tents.imp_km, tents.imp_depth_km * 1e3
    in_zone = (km >= km_range[0]) & (km <= km_range[1])
    chosen = []
    for d_lo, d_hi, n_take in take:
        idx = np.nonzero(in_zone & (depth > d_lo) & (depth <= d_hi))[0] if d_lo >= 0 else np.nonzero(in_zone & (np.abs(depth) <= d_hi))[0]
        if idx.size:
            idx = idx[np.argsort(km[idx])]
            chosen += list(idx[np.linspace(0, idx.size - 1, min(n_take, idx.size)).astype(int)])
    return np.array(sorted(set(chosen)))


if __name__ == "__main__":
    elapsed = common.timer()
    inputs = common.load_inputs()
    points_npz = inputs["points"]["default"]
    p = np.load(points_npz)
    to_mesh = pyproj.Transformer.from_crs("EPSG:4326", model_crs, always_xy=True)
    rows = []
    for panel, spec in (("A", PANEL_A), ("B", PANEL_B)):
        reference = None
        for run in spec["runs"]:
            gnpz = model_utils.load_G(run, points_npz, inputs["strike_deg"], paths_uniform["pylith_output"], paths_uniform["gf"])
            tents = model_utils.TentFault(gnpz, to_mesh, inputs["profile_lats"])
            if reference is None or panel == "B":                     # A: both orders share the mesh and the impulses
                chosen = choose(tents, spec["km"], spec["take"])
                subset = SimpleNamespace(fault_xyz=tents.fault_xyz, fault_cells=tents.fault_cells,
                                         vertex_index=tents.vertex_index[chosen], n_imp=chosen.size)
                G_analytic = model_utils.analytic_tent_G(subset, p["lon"], p["lat"], inputs["look"], model_frame_lonlat, to_mesh, n_sub=N_SUB)
                reference = (tents.fault_xyz, tents.vertex_index[chosen])
            else:
                assert np.allclose(tents.fault_xyz, reference[0]) and np.array_equal(tents.vertex_index[chosen], reference[1]), \
                    f"{run}: not the mesh / impulses of {spec['runs'][0]}"
            G_pylith = gnpz["G_rl_m_per_m"][:, chosen]
            for j, imp in enumerate(chosen):
                g_a, g_p = G_analytic[:, j], G_pylith[:, j]
                rows.append(dict(panel=panel, run=run, order=2 if "order2" in run else 1, impulse=int(imp), km=float(tents.imp_km[imp]),
                                 depth_m=float(tents.imp_depth_km[imp] * 1e3), norm_analytic=float(np.linalg.norm(g_a)),
                                 ratio=float(np.linalg.norm(g_p) / np.linalg.norm(g_a)),
                                 rel_rms=float(np.linalg.norm(g_p - g_a) / np.linalg.norm(g_a)),
                                 sign_agrees=bool(np.dot(g_p, g_a) > 0), n_sub=N_SUB))
            print(f"{panel} {run}: {chosen.size} impulses ({elapsed()})", flush=True)
    table = pd.DataFrame(rows)
    table["depth_group"] = pd.cut(table.depth_m, DEPTH_BINS_M, labels=DEPTH_LABELS)
    path = common.write(table.to_dict("records"), "gf_check.csv", __file__)
    print(table.groupby(["panel", "run", "depth_group"], observed=True).rel_rms.median().unstack().round(3).to_string())
    print(f"sign disagreements: {int((~table.sign_agrees).sum())} of {len(table)}")
    print(f"wrote {path} ({elapsed()})")
