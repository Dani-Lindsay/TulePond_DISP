#!/usr/bin/env python3
"""
The Lake Elizabeth strip (Central Park, Fremont): pixels EAST of the trace within 200 m, km 60.2-60.8 from Point
Pinole, at map coherence 0.9 and 0.7 (coh07/make_fp_coh.py products). Per track: pixel count, median LOS mean event,
median implied fault-parallel value LOS / G_h, with G_h = sin(inc) cos(az - (-strike)) from MintPy's own
get_design_matrix4horz_vert (the matrix 06's decomposition uses), assuming no vertical motion. Also the decomposed
fault-parallel field (06, both tracks) and the quadtree samples (east side within 300 m, km 60.2-60.8) of each set.
Check: in the near field (|d| <= 500 m, km 57-62) the mean of the two tracks' implied values must track the
decomposed field (correlation reported and asserted > 0.8).
Run in opera_disp_feb13 from the repo root:  python drafts/pylith/coh07/strip_stats.py   -> coh07/strip_stats.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import pyproj

HERE = os.path.dirname(os.path.abspath(__file__))
PYL = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(PYL, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, PYL)
from TulePond_filepaths import common_paths
from mintpy.asc_desc2horz_vert import get_design_matrix4horz_vert
from plot_run_summary import trace_07, signed_distance_m

CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"
KM, STRIP_M, SAMPLE_M = (60.2, 60.8), 200.0, 300.0
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_, _, tx, ty = trace_07()
prof = pd.read_csv(common_paths["fault_profiles"])
prof_km = (len(prof) - 1 - np.arange(len(prof))) * 0.1                 # 07 [2]'s km from Point Pinole


def km_of_lat(lat):
    lat = np.asarray(lat, float).ravel()
    out = np.empty(lat.size)
    for lo in range(0, lat.size, 20000):
        c = lat[lo:lo + 20000]
        out[lo:lo + 20000] = prof_km[np.abs(prof.ft_lat.values[None, :] - c[:, None]).argmin(axis=1)]
    return out


def geometry(lon, lat):
    lon = np.where(np.asarray(lon) > 180, np.asarray(lon) - 360, lon)
    x, y = to_mesh.transform(np.asarray(lon, float).ravel(), np.asarray(lat, float).ravel())
    return signed_distance_m(tx, ty, x, y), km_of_lat(lat)


rows, check = [], []
for coh in (0.9, 0.7):
    d = os.path.join(HERE, f"coh{int(round(coh * 10)):02d}")
    los = np.load(os.path.join(d, "los_mean_event.npz"))
    fp = np.load(os.path.join(d, "fp_maps.npz"))
    dist, km = geometry(los["lon"], los["lat"])
    strike = float(los["strike"])
    strip = (dist > 0) & (dist <= STRIP_M) & (km >= KM[0]) & (km <= KM[1])
    near = (np.abs(dist) <= 500) & (km >= 57) & (km <= 62)
    implied = {}
    for o in ("ascending", "descending"):
        v = los[f"{o}_los_mm"].ravel()
        inc, azi = los[f"{o}_inc"].ravel(), los[f"{o}_azi"].ravel()
        ok = np.isfinite(v) & (strip | near)
        g_h = np.full(v.size, np.nan)
        g_h[ok] = get_design_matrix4horz_vert(inc[ok], azi[ok], horz_az_angle=-strike)[:, 0]
        implied[o] = v / g_h
        s = strip & np.isfinite(v)
        rows.append(dict(coh=coh, source=f"{o} LOS", n_pixels=int(s.sum()), median_los_mm=float(np.median(v[s])) if s.any() else np.nan,
                         median_fp_mm=float(np.median(implied[o][s])) if s.any() else np.nan,
                         p16_fp_mm=float(np.percentile(implied[o][s], 16)) if s.any() else np.nan,
                         p84_fp_mm=float(np.percentile(implied[o][s], 84)) if s.any() else np.nan))
    f = fp["fp_event"].ravel()
    s = strip & np.isfinite(f)
    rows.append(dict(coh=coh, source="decomposed fp (both tracks, 06)", n_pixels=int(s.sum()), median_los_mm=np.nan,
                     median_fp_mm=float(np.median(f[s])) if s.any() else np.nan,
                     p16_fp_mm=float(np.percentile(f[s], 16)) if s.any() else np.nan,
                     p84_fp_mm=float(np.percentile(f[s], 84)) if s.any() else np.nan))
    both = near & np.isfinite(implied["ascending"]) & np.isfinite(implied["descending"]) & np.isfinite(f)
    r = float(np.corrcoef(0.5 * (implied["ascending"][both] + implied["descending"][both]), f[both])[0, 1])
    check.append(f"coh {coh}: near-field (|d|<=500 m, km 57-62) correlation of the mean implied fp with the decomposed fp "
                 f"{r:.3f} over {both.sum()} pixels")
    assert r > 0.8, check[-1]

# the quadtree samples: east side within 300 m, km 60.2-60.8
for name, path in (("samples coh 0.9 (07, 2386)", os.path.join(REPO, "outputs", "07_slip_model_tests", "fig2_model_row.npz")),
                   ("samples coh 0.7 (3148)", os.path.join(PYL, "points", "insar_points_coh07.npz"))):
    z = np.load(path)
    lon, lat = (z["sample_lon"], z["sample_lat"]) if "sample_lon" in z.files else (z["lon"], z["lat"])
    vel = z["data_mm"] if "data_mm" in z.files else z["vel_mm"]
    dist, km = geometry(lon, lat)
    for side, sel in (("east", (dist > 0) & (dist <= SAMPLE_M)), ("west", (dist < 0) & (dist >= -SAMPLE_M))):
        s = sel & (km >= KM[0]) & (km <= KM[1])
        rows.append(dict(coh=0.9 if "0.9" in name else 0.7, source=f"{name}, {side} <= 300 m", n_pixels=int(s.sum()),
                         median_los_mm=np.nan, median_fp_mm=float(np.median(vel[s])) if s.any() else np.nan,
                         p16_fp_mm=np.nan, p84_fp_mm=np.nan))
table = pd.DataFrame(rows)
table.to_csv(os.path.join(HERE, "strip_stats.csv"), index=False)
print("\n".join(check))
print(f"Lake Elizabeth strip: east of the trace, 0-{STRIP_M:.0f} m, km {KM[0]}-{KM[1]} (samples: 0-{SAMPLE_M:.0f} m)")
print(table.round(2).to_string(index=False))
