"""sfsvm_boundary_dip.py: the dip of the Hayward fault boundary in the SFSVM grid PyLith read (model_config.sfcvm_grid_npz),
beside the Tule Pond SSE section (Dani 2026-10-30: is a 60 W dip unrealistic compared to the 3D velocity model?).
At every 8th trace point in the section, walk along the trace normal (+ = east-north-east) at depths 0, 0.2, 0.5, 1, 2,
3 km and find the change of `fault_block` nearest the trace; the median offset with depth gives the apparent dip.
Run: cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. ~/miniforge3/envs/opera_disp_feb13/bin/python <this file>"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import model_config as cfg

grid = np.load(cfg.sfcvm_grid_npz, allow_pickle=True)
assert str(grid["dip"]) == "90" and str(grid["shift_by_depth"]) == "{}", "expected the aligned grid shifted the same at every depth"
block = RegularGridInterpolator((grid["x"], grid["y"], grid["z"]), grid["fault_block"].astype(float), method="nearest", bounds_error=False)
lo, hi = cfg.sse_section_km()
trace_km = grid["trace_km"]
depths_m = [0, 200, 500, 1000, 2000, 3000]
offsets = []
for i in np.where((trace_km >= lo) & (trace_km <= hi))[0][::8]:
    x0, y0, normal = grid["trace_x"][i], grid["trace_y"][i], grid["trace_normal"][i]
    along = np.arange(-3000.0, 3001.0, 10.0)
    row = []
    for depth in depths_m:
        values = block(np.c_[x0 + along * normal[0], y0 + along * normal[1], np.full_like(along, -depth)])
        changes = along[np.where(np.diff(values) != 0)[0]]
        row.append(changes[np.argmin(np.abs(changes))] if changes.size else np.nan)
    offsets.append(row)
    print(f"km {trace_km[i]:.1f}: boundary offset (m, + = ENE) at depths {depths_m} m: {row}")
offsets = np.array(offsets)
for j, depth in enumerate(depths_m[1:], 1):
    shift = np.nanmedian(offsets[:, j] - offsets[:, 0])
    print(f"depth {depth} m: median shift {shift:+.0f} m -> apparent dip {np.degrees(np.arctan2(depth, abs(shift))):.0f} deg "
          f"{'E' if shift > 0 else 'W'}")
