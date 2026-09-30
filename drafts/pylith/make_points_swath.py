#!/usr/bin/env python3
"""
Quadtree sample set of 07_slip_model with a different swath (pixels within `swath_km` of the trace), written as a
PyLith PointsList + an .npz with the data, block corners, pixel counts and the noise-model parameters.

07's cells [1]-[4] are exec'd unchanged from scripts/07_slip_model.py, with the single line
`swath_km       = 4.0` replaced by the requested value (asserted to be present exactly once). Everything else
(coherence-0.9 field, 30 m exclusion, km 52-67 window, far-field plane, variogram, distance-based quadtree from 2 km
blocks to one 30 m pixel) is 07's code, so the sample set for swath 4 must reproduce 07's 2386 samples exactly
(asserted against fig2_model_row.npz when run with 4; the data equal up to 07's fitted constant).

Run in opera_disp_feb13 from the repo root:  python drafts/pylith/make_points_swath.py <swath_km> [<fp_maps.npz> <tag>]
Writes drafts/pylith/points/insar_points_swath<N>.txt / .npz (07 quadtree with swath N km).
"""
import os
import sys
import time

import numpy as np
import pyproj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)
CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"   # = generate_gmsh_vertical.py

swath = float(sys.argv[1])
fp_file = sys.argv[2] if len(sys.argv) > 2 else None           # optional: another fault-parallel field (e.g. coh07/coh07/fp_maps.npz)
tag = sys.argv[3] if len(sys.argv) > 3 else f"swath{swath:g}"
assert (fp_file is None) == (len(sys.argv) <= 3), "give both <fp_maps.npz> and <tag>"
source_07 = open(os.path.join(REPO, "scripts", "07_slip_model.py")).read()
head = source_07.split("# %% [markdown]\n# ## [5]")[0]                  # cells [1]-[4]: data, noise model, quadtree
old = "swath_km       = 4.0"
assert head.count(old) == 1, "07's swath_km line not found exactly once"
head = head.replace(old, f"swath_km       = {swath}")
if fp_file is not None:
    old_fp = 'fp = np.load(os.path.join(field_dir, "fp_maps.npz"))'
    assert head.count(old_fp) == 1, "07's fp_maps load not found exactly once"
    head = head.replace(old_fp, f"fp = np.load({os.path.abspath(fp_file)!r})")
ns = {}
exec(head, ns)
ds, cov_p = ns["samples"], ns["cov_p"]
lon, lat, vel = np.asarray(ds.lon, float), np.asarray(ds.lat, float), np.asarray(ds.vel, float)
wgt = np.asarray(ds.wgt, float)
assert wgt.size == vel.size and np.all(wgt >= 1), "quadtree pixel counts missing"
xyc = np.asarray(ds.xycorner, dtype=float)
assert xyc.shape == (vel.size, 4)
corner_x = np.c_[xyc[:, 0], xyc[:, 2], xyc[:, 2], xyc[:, 0]]
corner_y = np.c_[xyc[:, 1], xyc[:, 1], xyc[:, 3], xyc[:, 3]]
block_lon, block_lat = ds.xy2ll(corner_x.ravel(), corner_y.ravel())
block_lon, block_lat = np.reshape(block_lon, corner_x.shape), np.reshape(block_lat, corner_x.shape)
inside = ((block_lon.min(1) <= lon) & (lon <= block_lon.max(1)) & (block_lat.min(1) <= lat) & (lat <= block_lat.max(1)))
assert inside.mean() > 0.99, "corner order is off"
n_px_total = int(np.isfinite(ns["sar"].vel).sum())

if abs(swath - 4.0) < 1e-9 and fp_file is None:                      # the replication gate (07's own field only)
    row = np.load(os.path.join(REPO, "outputs", "07_slip_model_tests", "fig2_model_row.npz"))
    assert row["sample_lon"].size == lon.size == 2386, f"swath 4 gives {lon.size} samples, 07 has {row['sample_lon'].size}"
    assert np.array_equal(row["sample_lon"], lon) and np.array_equal(row["sample_lat"], lat), "swath-4 samples differ from 07's"
    assert np.ptp(row["data_mm"] - vel) < 1e-9, "swath-4 data differ from 07's (beyond the fitted constant)"
    print("PASS: swath 4 reproduces 07's 2386 samples (positions and data)")

to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = to_mesh.transform(lon, lat)
assert np.all(np.abs(x) < 45e3) and np.all(np.abs(y) < 45e3), "a sample lies outside the mesh domain"
names = np.array([f"s{k:04d}" for k in range(lon.size)])
os.makedirs(os.path.join(HERE, "points"), exist_ok=True)
txt = os.path.join(HERE, "points", f"insar_points_{tag}.txt")
with open(txt, "w") as fh:
    fh.write(f"# quadtree samples of 07_slip_model with swath_km = {swath:g} ({lon.size} points of {n_px_total} pixels), "
             f"make_points_swath.py {time.strftime('%Y-%m-%d %H:%M')}\n# name x y z (m) in {CRS}\n")
    for n, xi, yi in zip(names, x, y):
        fh.write(f"{n} {xi:.3f} {yi:.3f} 0.0\n")
np.savez(os.path.join(HERE, "points", f"insar_points_{tag}.npz"), lon=lon, lat=lat, x=x, y=y, names=names, crs=CRS,
         vel_mm=vel, wgt=wgt, block_lon=block_lon, block_lat=block_lat, cov_p=np.asarray(cov_p, float),
         swath_km=swath, n_pixels=n_px_total, strike_deg=float(ns["strike_bearing"]), fp_file=str(fp_file or "06 fp_maps.npz"),
         provenance=f"make_points_swath.py {time.strftime('%Y-%m-%d %H:%M')}: 07 cells [1]-[4] with swath_km {swath:g}"
                    f"{'' if fp_file is None else ', field ' + os.path.abspath(fp_file)}")
d_trace = ns["dist_to_trace"](ds, ns["fault"])
print(f"swath {swath:g} km: {n_px_total} pixels -> {lon.size} samples ({np.sum(d_trace <= 4.0)} within 4 km, "
      f"{np.sum(d_trace > 4.0)} beyond); noise {np.round(np.sqrt(cov_p[[0, 1, 3]]), 3)} mm, ranges {np.round(cov_p[[2, 4]], 2)} km; "
      f"x {x.min():.0f}..{x.max():.0f}, y {y.min():.0f}..{y.max():.0f} m; wrote {txt}")
