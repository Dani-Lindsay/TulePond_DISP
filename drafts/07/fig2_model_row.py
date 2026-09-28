#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (2026-09-28): the best current slip model, written as the product that
Fig 2's model row (notebook 06, panels I, J, K) draws.

Configuration (the 2026-09-28 decisions): fault-parallel mean event (06
fp_maps.npz, decomposed at the SSE-section strike), far-field plane
levelling, vertical fault, smooth12 mesh (50 m cells at the surface, x1.2),
quadtree to one 30 m pixel, buildCmXY lambda (0.3, 0.05) km, sigma 24, the
crossing-stack surface constraint interpolated to every surface triangle,
NNLS. Homogeneous Green's functions: replaced by the 1D layered PyLith
model when that exists. Mw at 30 GPa and with the USGS SFCVM rigidity(z).

Output: outputs/07_slip_model_tests/fig2_model_row.npz
  sample_lon, sample_lat, data_mm, model_mm, residual_mm   (quadtree samples, fault-parallel)
  nodes_sz (s km from the north end, z km), faces, slip_mm, s0_km_pp
  mw_30gpa, mw_sfcvm, rms_mm, surf_ratio_pct, sigma, strike_deg
Run from the repo root in opera_disp_feb13 (PROJ_LIB pointed at the env).
"""
import os

src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])

sigma_sel = 24

# ------------------------
# Load: the fault-parallel mean event
# ------------------------
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
strike_bearing = float(fp["strike"])
look = np.array([np.sin(np.deg2rad(strike_bearing)), np.cos(np.deg2rad(strike_bearing)), 0.0])
ok = np.isfinite(fp["fp_event"])
fault = wrap_fault(*unrolled["smooth12"], 90, "fig2_model")
sar = insar("fault-parallel", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
sar.read_from_binary(fp["fp_event"][ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()),
                     los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
trace_dist = dist_to_trace(sar, fault)
sar.vel[(trace_dist > swath_km) | (trace_dist < exclude_km)] = np.nan
km_pp = lat_to_km_pp(sar.lat)
sar.vel[(km_pp < km_window[0]) | (km_pp > km_window[1])] = np.nan
sar.checkNaNs()
align_track(sar)
cov_p = covariance(sar, fault)
ds = downsample(sar, fault, start_block_km, min_block_km)
build_cd(ds, cov_p)
print(f"fault-parallel field at strike {strike_bearing:.2f} deg: {ds.vel.size} quadtree samples")

# ------------------------
# Calculate: the inversion
# ------------------------
surface = make_surface_interp(fault, surf_floor_mm)
invert(fault, [ds], sigma_sel, surface=surface)
row, table = metrics(fault, [ds], surface)
residual = ds.vel - ds.synth

nodes, faces, _ = fault.mesh_sz
tri = nodes[faces]
area_km2 = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
profiles_sfcvm = pd.read_csv(os.path.join(out_dir, "sfcvm_creep_section_profiles.csv"))
mu_by_depth = profiles_sfcvm.groupby("depth_m").mu_gpa.median()
mu_gpa = np.maximum(np.interp(table.z_c.values * 1000, mu_by_depth.index.values, mu_by_depth.values), 0.5)
moment_sfcvm = np.sum(mu_gpa * 1e9 * area_km2 * 1e6 * table.slip_mm.values * 1e-3)
mw_sfcvm = (2.0 / 3.0) * (np.log10(moment_sfcvm) - 9.1)

in_creep = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
base = get_surface(surf_floor_mm)
base_km = lat_to_km_pp(base.lat)
crossing_median = np.median(base.vel[(base_km >= creep_lo_km) & (base_km <= creep_hi_km)])
surf_ratio = 100 * table.slip_mm[in_creep].median() / crossing_median
# along-strike origin: km from Point Pinole at the north end of the mesh (s = 0)
s0_km_pp = float(lat_to_km_pp([ref_fault.lati[0]])[0])

print(f"Mw {row['Mw']:.2f} (30 GPa), {mw_sfcvm:.2f} (SFCVM); rms {np.sqrt(np.mean(residual ** 2)):.2f} mm; "
      f"surface slip {surf_ratio:.0f}% of the crossings; peak {row['peak_slip_mm']:.1f} mm")

# ------------------------
# Write
# ------------------------
out_file = os.path.join(out_dir, "fig2_model_row.npz")
np.savez(out_file, sample_lon=ds.lon, sample_lat=ds.lat, data_mm=ds.vel, model_mm=ds.synth,
         residual_mm=residual, nodes_sz=nodes, faces=faces, slip_mm=table.slip_mm.values,
         s0_km_pp=s0_km_pp, mw_30gpa=row["Mw"], mw_sfcvm=mw_sfcvm,
         rms_mm=float(np.sqrt(np.mean(residual ** 2))), surf_ratio_pct=float(surf_ratio),
         sigma=sigma_sel, strike_deg=strike_bearing)
print("wrote", out_file)
