#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (2026-09-28): smoothing sigma re-selected on the FINAL homogeneous
configuration (fault-parallel input, 0.9 field, SSE-section strike,
levelled, smooth12, dip 90, 30 m quadtree, crossing constraint on every
surface triangle, 0-0.2 km top tie). Records misfit and model norm for an
L-curve. Output: outputs/07_slip_model_tests/sigma_final.csv
Run from the repo root.
"""
import os

src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])

sigmas = [8, 12, 16, 24, 32, 48, 64]
tie_depth_km = 0.2

# ------------------------
# Load: the fault-parallel field and its quadtree samples
# ------------------------
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
look = np.array([np.sin(np.deg2rad(float(fp["strike"]))), np.cos(np.deg2rad(float(fp["strike"]))), 0.0])
ok = np.isfinite(fp["fp_event"])
base_fault = wrap_fault(*unrolled["smooth12"], 90, "sigma_base")
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

# ------------------------
# Calculate: one inversion per sigma
# ------------------------
rows = []
for sigma in sigmas:
    fault = wrap_fault(*unrolled["smooth12"], 90, f"sigma_final_{sigma}")
    ds = copy.deepcopy(ds_base)
    surface = make_surface_interp(fault, surf_floor_mm)
    invert(fault, [ds], sigma, surface=surface, tie_depth_km=tie_depth_km)
    row, table = metrics(fault, [ds], surface)
    in_creep = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
    nodes, faces, _ = fault.mesh_sz
    tri = nodes[faces]
    area_km2 = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                            - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    rows.append(dict(sigma=sigma, chi2=row["chi2"], rms_mm=row["rms_mm"],
                     rms_near_mm=row["rms_near_" + ds.name.split()[-1][:3]],
                     surf_rms_mm=row["surf_rms_mm"],
                     surface_pct=100 * table.slip_mm[in_creep].median() / crossing_median,
                     model_norm=float(np.sqrt(np.sum(area_km2 * table.slip_mm.values ** 2))),
                     peak_slip_mm=row["peak_slip_mm"], Mw=row["Mw"]))
    print(f"SIG {sigma}: chi2 {row['chi2']:.0f}, rms {row['rms_mm']:.3f} mm, surface "
          f"{rows[-1]['surface_pct']:.0f}%, norm {rows[-1]['model_norm']:.1f}, Mw {row['Mw']:.2f}")
family = pd.DataFrame(rows)
family.to_csv(os.path.join(out_dir, "sigma_final.csv"), index=False)
plateau = family[family.rms_mm <= family.rms_mm.min() + 0.005]
print(f"SIG plateau onset: sigma {plateau.sigma.min():g}")
