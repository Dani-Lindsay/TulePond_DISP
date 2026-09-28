"""DRAFT (2026-09-28): with the 0.2 km top-layer tie, is the crossing-stack
surface constraint still needed? Fault-parallel field at the 0.9 mask,
SSE-section strike, smooth12, sigma 24: tie + crossings vs tie only vs
neither."""
import os
src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
look = np.array([np.sin(np.deg2rad(float(fp["strike"]))), np.cos(np.deg2rad(float(fp["strike"]))), 0.0])
ok = np.isfinite(fp["fp_event"])
f0 = wrap_fault(*unrolled["smooth12"], 90, "tvc_base")
sar = insar("fault-parallel", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
sar.read_from_binary(fp["fp_event"][ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()),
                     los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
dd = dist_to_trace(sar, f0); sar.vel[(dd > swath_km) | (dd < exclude_km)] = np.nan
kk = lat_to_km_pp(sar.lat); sar.vel[(kk < km_window[0]) | (kk > km_window[1])] = np.nan
sar.checkNaNs(); align_track(sar)
cov_p = covariance(sar, f0)
ds0 = downsample(sar, f0, start_block_km, min_block_km); build_cd(ds0, cov_p)
base = get_surface(surf_floor_mm); bkm = lat_to_km_pp(base.lat)
cross_med = np.median(base.vel[(bkm >= creep_lo_km) & (bkm <= creep_hi_km)])
for label, use_surf, tie in (("tie + crossings", True, 0.2), ("tie only", False, 0.2), ("neither", False, None)):
    flt = wrap_fault(*unrolled["smooth12"], 90, f"tvc_{label}")
    ds = copy.deepcopy(ds0)
    surface = make_surface_interp(flt, surf_floor_mm) if use_surf else None
    invert(flt, [ds], 24, surface=surface, tie_depth_km=tie)
    row, table = metrics(flt, [ds], surface)
    sec = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
    prof = row["depth_profile"]; pz = max(prof, key=prof.get)
    print(f"TVC {label}: surface {100 * table.slip_mm[sec].median() / cross_med:.0f}% of crossings; "
          f"peak {prof[pz]:.1f} mm at {pz * 1000:.0f} m; rms {row['rms_mm']:.2f} mm; "
          f"near-field rms {row['rms_near_' + ds.name.split()[-1][:3]]:.2f} mm; Mw {row['Mw']:.2f}")
