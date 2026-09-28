#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (2026-09-28): L-curve for the smoothing sigma on the FINAL homogeneous
configuration (fault-parallel input, 0.9 field, SSE-section strike,
levelled, smooth12, dip 90, 30 m quadtree, crossing constraint on every
surface triangle, 0-0.2 km top tie). Denser and wider than
sigma_final_config.py (4-128) so the corner is bracketed, and the slip of
every sigma is saved so the choices can be looked at side by side.

L-curve axes (textbook, Hansen 1992): weighted misfit sqrt(chi2) of the
InSAR samples against the area-weighted slip norm, both log. Corner =
maximum curvature of the log-log curve (spline in log sigma).
Output: outputs/07_slip_model_tests/sigma_lcurve.csv, sigma_lcurve_slip.npz
Run from the repo root.
"""
import os

src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])

sigmas = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]
tie_depth_km = 0.2

# ------------------------
# Load: the fault-parallel field and its quadtree samples
# ------------------------
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
look = np.array([np.sin(np.deg2rad(float(fp["strike"]))), np.cos(np.deg2rad(float(fp["strike"]))), 0.0])
ok = np.isfinite(fp["fp_event"])
base_fault = wrap_fault(*unrolled["smooth12"], 90, "lcurve_base")
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

nodes, faces, _ = base_fault.mesh_sz
tri = nodes[faces]
area_km2 = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))

# ------------------------
# Calculate: one inversion per sigma
# ------------------------
rows, slips, residuals = [], [], []
for sigma in sigmas:
    fault = wrap_fault(*unrolled["smooth12"], 90, f"lcurve_{sigma}")
    ds = copy.deepcopy(ds_base)
    surface = make_surface_interp(fault, surf_floor_mm)
    solver = invert(fault, [ds], sigma, surface=surface, tie_depth_km=tie_depth_km)
    row, table = metrics(fault, [ds], surface)
    # the full misfit (InSAR + surface constraint, both covariance-weighted) and the
    # prior-weighted roughness m^T C_corr^-1 m (C_corr = the unit-sigma buildCmXY matrix):
    # the two terms the solver actually trades off
    surf_resid = surface.vel - surface.synth
    chi2_surf = float(np.dot(surf_resid, np.linalg.solve(surface.Cd, surf_resid)))
    corr = fault.Cm / sigma ** 2
    slip_vec = fault.slip[:, 0]
    prior_norm = float(np.sqrt(np.dot(slip_vec, np.linalg.solve(corr, slip_vec))))
    offset_values = [v for t, v in zip(solver.paramTypes, solver.mpost) if t[0] == "ramps"]
    in_creep = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
    rows.append(dict(sigma=sigma, chi2=row["chi2"], rms_mm=row["rms_mm"],
                     rms_near_mm=row["rms_near_" + ds.name.split()[-1][:3]],
                     surf_rms_mm=row["surf_rms_mm"],
                     surface_pct=100 * table.slip_mm[in_creep].median() / crossing_median,
                     model_norm=float(np.sqrt(np.sum(area_km2 * table.slip_mm.values ** 2))),
                     peak_slip_mm=row["peak_slip_mm"], Mw=row["Mw"],
                     chi2_surf=chi2_surf, chi2_total=row["chi2"] + chi2_surf, prior_norm=prior_norm,
                     offset_insar_mm=offset_values[0], offset_surface_mm=offset_values[-1]))
    slips.append(table.slip_mm.values)
    residuals.append(ds.vel - ds.synth)
    print(f"SIG {sigma}: chi2 {row['chi2']:.0f}, rms {row['rms_mm']:.3f} mm, surface "
          f"{rows[-1]['surface_pct']:.0f}%, norm {rows[-1]['model_norm']:.1f}, "
          f"peak {row['peak_slip_mm']:.1f} mm, Mw {row['Mw']:.2f}; chi2_surf {chi2_surf:.0f}, "
          f"prior norm {prior_norm:.1f}, offsets {offset_values[0]:+.2f} / {offset_values[-1]:+.2f} mm", flush=True)
family = pd.DataFrame(rows)
family.to_csv(os.path.join(out_dir, "sigma_lcurve.csv"), index=False)
np.savez(os.path.join(out_dir, "sigma_lcurve_slip.npz"), sigma=np.array(sigmas),
         slip_mm=np.array(slips), residual_mm=np.array(residuals), nodes_sz=nodes, faces=faces,
         km_pp=table.km_pp.values, z_top=table.z_top.values, z_c=table.z_c.values,
         s_km=table.s_km.values, sample_lon=ds.lon, sample_lat=ds.lat,
         crossing_km=base_km, crossing_mm=base_surface.vel, crossing_err=base_surface.err)
print("wrote sigma_lcurve.csv, sigma_lcurve_slip.npz")
