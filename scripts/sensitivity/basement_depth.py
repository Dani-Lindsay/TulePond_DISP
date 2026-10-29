"""basement_depth.py: the SFSVM sediment / basement contact beside the Tule Pond section (the grey band of the depth-scan
figure). Ported 2026-10-30 from basement_depth_2026-10-28/basement_depth.py (paths and section from the repo, no literals).

Basement = the shallowest grid level whose geologic zone is NOT Cenozoic (zone names starting Cenozoic / QT_ / T_) and not
water or <null>, i.e. the top of Franciscan (Kjf_*), Great Valley / Valley Sequence, San Leandro or Logan gabbro, Kgr_*.
Sampled on the PyLith material grid (model_config.sfcvm_grid_npz, what the 3D model uses; nearest grid point; the grid is
rectilinear in the mesh CRS) along the trace normal at signed distances -500..+500 m (east +), every 0.1 km over the SSE
section (04). Also the Vs at 0, 100, 300, 500 m depth.
Output (model_config.sensitivity_dir): basement_depth.csv.
Run (repo root): PYTHONPATH=. python scripts/sensitivity/basement_depth.py      (seconds)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                         # noqa: E402
from common import cfg                                # noqa: E402

DISTANCES_M = (-500, -300, -200, -100, -50, 50, 100, 200, 300, 500)
VS_DEPTHS_M = (0, 100, 300, 500)

if __name__ == "__main__":
    grid = np.load(cfg.sfcvm_grid_npz, allow_pickle=True)
    names = {int(k): v for k, v in json.loads(str(grid["zone_names"])).items()}
    sediment_ids = {k for k, v in names.items() if v.startswith(("Cenozoic", "QT_", "T_")) or v in ("water", "<null>")}
    x, y, z = grid["x"], grid["y"], grid["z"]
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0), "the material grid is not rectilinear and sorted"
    surface_first = np.argsort(-z)                               # z up, m
    trace_km, trace_x, trace_y, trace_normal = grid["trace_km"], grid["trace_x"], grid["trace_y"], grid["trace_normal"]
    lo, hi = cfg.sse_section_km()
    rows = []
    for km in np.arange(round(lo, 1), hi + 0.01, 0.1):
        i = int(np.argmin(np.abs(trace_km - km)))
        for d_m in DISTANCES_M:
            px, py = trace_x[i] + d_m * trace_normal[i, 0], trace_y[i] + d_m * trace_normal[i, 1]
            ix, iy = int(np.argmin(np.abs(x - px))), int(np.argmin(np.abs(y - py)))
            column_zone, column_vs = grid["zone"][ix, iy, :], grid["vs"][ix, iy, :]
            basement_top_m = np.nan
            for iz in surface_first:
                if z[iz] > 0:
                    continue
                if int(column_zone[iz]) not in sediment_ids:
                    basement_top_m = -z[iz]
                    break
            vs_at = {f"vs_{depth}m": float(column_vs[int(np.argmin(np.abs(z + depth)))]) for depth in VS_DEPTHS_M}
            rows.append(dict(km=round(float(km), 2), d_m=d_m, side="west" if d_m < 0 else "east", basement_top_m=basement_top_m,
                             surface_zone=names[int(column_zone[int(np.argmin(np.abs(z)))])], **vs_at))
    path = common.write(rows, "basement_depth.csv", __file__)
    table = pd.DataFrame(rows)
    b_lo, b_hi = cfg.band90_km()
    for label, sel in ((f"SSE section km {lo:.1f}-{hi:.1f}", table.km.between(lo, hi)), (f"90 % band km {b_lo:.1f}-{b_hi:.1f}", table.km.between(b_lo, b_hi))):
        print(f"\n{label}: basement top (m), median [min-max] by distance from the trace")
        print(table[sel].groupby("d_m").basement_top_m.agg(["median", "min", "max"]).round(0).to_string())
    print(f"wrote {path}")
