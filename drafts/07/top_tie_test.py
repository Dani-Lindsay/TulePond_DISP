#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (2026-09-28): tie slip uniform from the surface to a fixed depth
(after Jolivet et al. 2023, who tie 0-1 km), on the fault-parallel model.

1. Synthetic recovery: uniform 10 mm to 0.3 / 0.8 / 1.5 km, forward through
   the fault-parallel samples (covariance noise) and the interpolated surface
   constraint, inverted with a 0-0.2 km tie: does the sub-surface overshoot go?
2. The real fault-parallel mean event with no tie and ties of 0.1, 0.2 and
   0.5 km: top cell, peak and its depth, surface % of the crossings, rms, Mw.
Same configuration as drafts/07/fig2_model_row.py (smooth12, dip 90,
sigma 24, levelled, 30 m quadtree). Run from the repo root.
"""
import os

src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])

sigma_sel = 24

# ------------------------
# Load: the fault-parallel field and its quadtree samples
# ------------------------
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
look = np.array([np.sin(np.deg2rad(float(fp["strike"]))), np.cos(np.deg2rad(float(fp["strike"]))), 0.0])
ok = np.isfinite(fp["fp_event"])
base_fault = wrap_fault(*unrolled["smooth12"], 90, "tie_base")
sar = insar("fault-parallel", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
sar.read_from_binary(fp["fp_event"][ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()),
                     los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
trace_dist = dist_to_trace(sar, base_fault)
sar.vel[(trace_dist > swath_km) | (trace_dist < exclude_km)] = np.nan
km_pp = lat_to_km_pp(sar.lat)
sar.vel[(km_pp < km_window[0]) | (km_pp > km_window[1])] = np.nan
sar.checkNaNs()
align_track(sar)
cov_p = covariance(sar, base_fault)
ds_base = downsample(sar, base_fault, start_block_km, min_block_km)
build_cd(ds_base, cov_p)
base_surface = get_surface(surf_floor_mm)
base_km = lat_to_km_pp(base_surface.lat)
crossing_median = np.median(base_surface.vel[(base_km >= creep_lo_km) & (base_km <= creep_hi_km)])


def summarise(label, row, table):
    prof = row["depth_profile"]
    depths = sorted(prof)
    peak_z = max(prof, key=prof.get)
    in_creep = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
    ratio = 100 * table.slip_mm[in_creep].median() / crossing_median
    print(f"TIE {label}: top cell {prof[depths[0]]:.1f} mm, peak {prof[peak_z]:.1f} mm at {peak_z * 1000:.0f} m; "
          f"surface {ratio:.0f}% of crossings; rms {row['rms_mm']:.2f} mm; Mw {row['Mw']:.2f}")

# ------------------------
# 1. synthetic recovery with the 0-0.2 km tie
# ------------------------
rng = np.random.default_rng(7)
for depth_bottom in (0.3, 0.8, 1.5):
    fault = wrap_fault(*unrolled["smooth12"], 90, f"tie_rec_{depth_bottom:g}")
    table0 = patch_table(fault)
    window = np.clip(np.minimum(table0.km_pp - scan_km[0], scan_km[1] - table0.km_pp) / scan_taper + 1, 0, 1)
    slip_input = window.values * (table0.z_c.values < depth_bottom) * 10.0
    ds = copy.deepcopy(ds_base)
    fault.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
    fault.G[ds.name]["strikeslip"] *= -1.0
    chol = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
    ds.vel[:] = np.dot(fault.G[ds.name]["strikeslip"], slip_input) + np.dot(chol, rng.standard_normal(ds.vel.size))
    surface = make_surface_interp(fault, surf_floor_mm)
    fault.buildGFs(surface, slipdir="s", verbose=False)
    surface.vel[:] = (np.dot(fault.G[surface.name]["strikeslip"], slip_input)
                      + np.sqrt(np.diag(surface.Cd)) * rng.standard_normal(surface.vel.size))
    invert(fault, [ds], sigma_sel, surface=surface, tie_depth_km=0.2)
    row, table = metrics(fault, [ds], surface)
    summarise(f"synthetic 10 mm to {depth_bottom:g} km, tie 0.2 km", row, table)

# ------------------------
# 2. the real fault-parallel mean event, tie depth varied
# ------------------------
for tie_depth in (None, 0.1, 0.2, 0.5):
    fault = wrap_fault(*unrolled["smooth12"], 90, f"tie_real_{tie_depth}")
    ds = copy.deepcopy(ds_base)
    surface = make_surface_interp(fault, surf_floor_mm)
    invert(fault, [ds], sigma_sel, surface=surface, tie_depth_km=tie_depth)
    row, table = metrics(fault, [ds], surface)
    summarise(f"real, tie {tie_depth if tie_depth else 'none'}", row, table)
    if tie_depth == 0.2:
        table.to_csv(os.path.join(out_dir, "slip_fp_tie0.2.csv"), index=False)
