#!/usr/bin/env python3
"""
DRAFT (2026-10-08): where is the DISP-S1 burst seam in each track's step maps?

The CSLC-S1-STATIC bounding-polygon edge gives the seam's orientation but not the place where the DISP-S1 mosaic
stitches: on 2026-10-08 the discontinuity was found ~700 m south of the descending polygon edge and ~550 m north
of the ascending one (later along track on both tracks; the hand-picked descending line of 2026-08-19,
lat = 37.536 - 0.14 (lon + 122), had it right to 20 m). This script refines the along-track position at 10 m
steps on the raw (box-median) step maps written by 06 (`tulepond_steps_<track>.h5`, `event<i>_raw`): the
north-minus-south median jump in +/- burst_band_deg bands about the shifted polygon line, beyond burst_far_m of
the trace, for the five-event mean and for each event, and the same in longitude strips. The chosen shift is
HARD-CODED in 06 (`burst_seam_shift_m`); rerun this when the data or the products change.
Run from the repo root in opera_disp_feb13. Output: printed table and
<fig_dir>/qc06_burst_seam_locate.png (jump against shift, both tracks).
"""
import os, re, glob, sys
import h5py, numpy as np, pandas as pd, shapely
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from TulePond_filepaths import insar_dic_list, box_lonlat, paths_out, common_paths
from insar_utils import load_trace

out06 = paths_out["06_event_field_geometry"]
band_deg, far_m, min_pixels = 0.0015, 400.0, 200        # as 06 (burst_band_deg, burst_far_m, burst_min_pixels)
step_m, scan_m = 10.0, 1500.0
strips = 4

def polygon_seam(static_dir, box):
    """(slope, intercept) of the one burst seam crossing the box: lat = intercept + slope (lon + 122), as 06 [4]."""
    polys = {}
    for f in sorted(glob.glob(os.path.join(static_dir, "OPERA_L2_CSLC-S1-STATIC_*.h5"))):
        bid = re.search(r"STATIC_(T\d{3}-\d{6}-IW[1-3])_", os.path.basename(f)).group(1)
        with h5py.File(f) as h:
            p = shapely.from_wkt(h["identification/bounding_polygon"][()].decode())
        if p.intersects(shapely.box(*box)):
            polys[bid] = p
    seams = []
    for bid in sorted(polys):
        t, n, sw = bid.split("-"); nxt = f"{t}-{int(n) + 1:06d}-{sw}"
        if nxt not in polys:
            continue
        e = shapely.line_merge(polys[bid].boundary.intersection(polys[nxt].boundary))
        if e.geom_type != "LineString" or not e.intersects(shapely.box(*box)):
            continue
        el, ea = np.array(e.coords).T; o = np.argsort(el); el, ea = el[o], ea[o]
        lw, le = np.interp([box[0], box[2]], el, ea); s = (le - lw) / (box[2] - box[0])
        seams.append((s, lw - s * (box[0] + 122.0)))
    assert len(seams) == 1, seams
    return seams[0]

trace = load_trace(common_paths["fault_trace"])
lon0, lat0 = -121.968, 37.557
kx, ky = 111.32 * np.cos(np.deg2rad(lat0)), 110.95
tx, ty = (trace[:, 0] - lon0) * kx, (trace[:, 1] - lat0) * ky

def dist_to_trace(x, y):
    best = np.full(x.size, np.inf)
    for i in range(len(tx) - 1):
        ax, ay, bx, by = tx[i], ty[i], tx[i + 1], ty[i + 1]; vx, vy = bx - ax, by - ay; L2 = vx * vx + vy * vy
        t = np.clip(((x - ax) * vx + (y - ay) * vy) / L2, 0, 1); best = np.minimum(best, np.hypot(x - (ax + t * vx), y - (ay + t * vy)))
    return best

def jump(values, above, shift_m, far):
    lat_line = shift_m / 111.2e3
    north = (above > lat_line) & (above < lat_line + band_deg) & far & np.isfinite(values)
    south = (above < lat_line) & (above > lat_line - band_deg) & far & np.isfinite(values)
    if min(north.sum(), south.sum()) < min_pixels:
        return np.nan
    return float(np.median(values[north]) - np.median(values[south]))

shifts = np.arange(-scan_m, scan_m + 1e-6, step_m)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rows = []
for ax, dic in zip(axes, insar_dic_list):
    orb = dic["ORBIT_DIRECTION"]; slope, intercept = polygon_seam(dic["static_dir"], box_lonlat)
    with h5py.File(os.path.join(out06, f"tulepond_steps_{orb}.h5")) as h:
        lon = h["lon"][:]; lat = h["lat"][:]; raw = np.stack([h[f"event{i}_raw"][:] for i in range(5)]).astype(float)
        epochs = [float(h[f"event{i}"].attrs["step_epoch"]) for i in range(5)]
    above = lat - (intercept + slope * (lon + 122.0))                    # deg north of the polygon line
    d = dist_to_trace(((lon - lon0) * kx).ravel(), ((lat - lat0) * ky).ravel()).reshape(lon.shape); far = d > far_m / 1000.0
    mean = np.nanmean(raw, 0)
    J = np.array([jump(mean, above, s, far) for s in shifts]); k = int(np.nanargmax(np.abs(J)))
    half = np.abs(J) >= 0.9 * np.abs(J[k])                              # the plateau of the jump curve (band width limits the sharpness)
    plateau = shifts[half & (np.abs(shifts - shifts[k]) < 400)]
    ax.plot(shifts, J, "k-", lw=2, label=f"mean of 5: best {shifts[k]:+.0f} m ({J[k]:+.2f} mm), 90 % plateau {plateau.min():+.0f}..{plateau.max():+.0f} m")
    rows.append(dict(track=orb, event="mean", best_shift_m=shifts[k], jump_mm=J[k], plateau_lo_m=plateau.min(), plateau_hi_m=plateau.max(), jump_at_polygon_mm=J[np.argmin(np.abs(shifts))]))
    for i, ep in enumerate(epochs):
        Ji = np.array([jump(raw[i], above, s, far) for s in shifts]); ki = int(np.nanargmax(np.abs(Ji)))
        ax.plot(shifts, Ji, lw=0.9, alpha=0.7, label=f"{ep:.2f}: best {shifts[ki]:+.0f} m ({Ji[ki]:+.2f} mm); at mean's best {Ji[k]:+.2f}")
        rows.append(dict(track=orb, event=f"{ep:.2f}", best_shift_m=shifts[ki], jump_mm=Ji[ki], plateau_lo_m=np.nan, plateau_hi_m=np.nan, jump_at_polygon_mm=Ji[np.argmin(np.abs(shifts))]))
    edges = np.linspace(box_lonlat[0], box_lonlat[2], strips + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        ms = np.where((lon >= lo) & (lon < hi), mean, np.nan); Js = np.array([jump(ms, above, s, far) for s in shifts])
        if np.isfinite(Js).sum() < 10:
            continue
        ks = int(np.nanargmax(np.abs(Js)))
        rows.append(dict(track=orb, event=f"mean, lon {lo:.2f}..{hi:.2f}", best_shift_m=shifts[ks], jump_mm=Js[ks], plateau_lo_m=np.nan, plateau_hi_m=np.nan, jump_at_polygon_mm=Js[np.argmin(np.abs(shifts))]))
    ax.axvline(0, color="0.4", ls="--", label="polygon edge"); ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("shift of the polygon line along track (m, north +)"); ax.set_ylabel("north band minus south band (mm)")
    ax.set_title(f"{orb}: band jump vs seam position (bands +/- {band_deg * 111.2e3:.0f} m, > {far_m:.0f} m from the trace)", loc="left"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(common_paths["fig_dir"], "qc06_burst_seam_locate.png"), dpi=130)
df = pd.DataFrame(rows); pd.set_option("display.width", 200); print(df.round(2).to_string(index=False))
