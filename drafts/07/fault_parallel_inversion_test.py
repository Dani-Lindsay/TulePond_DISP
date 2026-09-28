"""Invert the decomposed FAULT-PARALLEL mean event (06 fp_maps.npz) instead
of the two LOS tracks: one dataset, look vector = horizontal unit vector
toward the strike bearing (MintPy horz_az_angle=-strike => + toward 143 deg).
Same machinery as the record model (smooth12, dip 90, sigma 24, aligned,
interpolated surface constraint)."""
src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])

fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
strike_bearing = float(fp["strike"])
look = np.array([np.sin(np.deg2rad(strike_bearing)), np.cos(np.deg2rad(strike_bearing)), 0.0])
ok = np.isfinite(fp["fp_event"])
vel = fp["fp_event"][ok]; lon = fp["lon"][ok]; lat = fp["lat"][ok]
fault = wrap_fault(*unrolled["smooth12"], 90, "fp_test")
sar = insar("fault-parallel", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
sar.read_from_binary(vel, lon, lat, err=np.ones(vel.size), los=np.tile(look, (vel.size, 1)),
                     remove_nan=True, remove_zeros=False)
d = dist_to_trace(sar, fault)
sar.vel[(d > swath_km) | (d < exclude_km)] = np.nan
km = lat_to_km_pp(sar.lat)
sar.vel[(km < km_window[0]) | (km > km_window[1])] = np.nan
sar.checkNaNs()
coef = align_track(sar)
cov_p = covariance(sar, fault)
nug2, s1, l1, s2, l2 = cov_p
print(f"FP field: {sar.vel.size} px; plane {coef[0]:+.2f} mm; noise nugget {np.sqrt(nug2):.2f} mm, "
      f"long {np.sqrt(s2):.2f} mm / {l2:.1f} km")
ds_base = downsample(sar, fault, start_block_km, min_block_km)
build_cd(ds_base, cov_p)

for use_surface in (False, True):
    flt = wrap_fault(*unrolled["smooth12"], 90, f"fp_{use_surface}")
    ds = copy.deepcopy(ds_base)
    surface = make_surface_interp(flt, surf_floor_mm) if use_surface else None
    invert(flt, [ds], 24, surface=surface)
    row, table = metrics(flt, [ds], surface)
    resid = ds.vel - ds.synth
    signed = dist_to_trace(ds, flt, signed=True)
    near_w = (signed < 0) & (signed > -0.5); near_e = (signed > 0) & (signed < 0.5)
    sec = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
    base = get_surface(surf_floor_mm)
    bkm = lat_to_km_pp(base.lat)
    ratio = table.slip_mm[sec].median() / np.median(base.vel[(bkm >= creep_lo_km) & (bkm <= creep_hi_km)])
    tag = "with surface" if use_surface else "InSAR only"
    print(f"FP {tag}: {ds.vel.size} samples; rms {np.sqrt(np.mean(resid**2)):.2f} mm; "
          f"near-field rms {row['rms_near_' + ds.name.split()[-1][:3]]:.2f} mm; "
          f"mean residual 0-0.5 km W {resid[near_w].mean():+.2f} / E {resid[near_e].mean():+.2f} mm; "
          f"surface {100*ratio:.0f}% of crossings; peak {row['peak_slip_mm']:.1f} mm; Mw {row['Mw']:.2f}")
    table.to_csv(os.path.join(out_dir, f"fp_inversion_{'surf' if use_surface else 'insar'}.csv"), index=False)
