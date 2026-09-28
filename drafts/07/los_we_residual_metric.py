src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])
base = get_surface(surf_floor_mm); bkm = lat_to_km_pp(base.lat)
cross_med = np.median(base.vel[(bkm >= creep_lo_km) & (bkm <= creep_hi_km)])
for use_surface in (False, True):
    flt = wrap_fault(*unrolled["smooth12"], 90, f"los_{use_surface}")
    datas = [copy.deepcopy(get_downsampled(t, "smooth12", 90)) for t in tracks]
    surface = make_surface_interp(flt, surf_floor_mm) if use_surface else None
    invert(flt, datas, 24, surface=surface)
    row, table = metrics(flt, datas, surface)
    sec = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
    parts = []
    for ds in datas:
        r = ds.vel - ds.synth; s = dist_to_trace(ds, flt, signed=True)
        w = (s < 0) & (s > -0.5); e = (s > 0) & (s < 0.5)
        parts.append(f"{ds.name.split()[-1][:3]} W {r[w].mean():+.2f} / E {r[e].mean():+.2f}")
    tag = "with surface" if use_surface else "InSAR only"
    print(f"LOS {tag}: mean residual 0-0.5 km {'; '.join(parts)} mm; surface "
          f"{100*table.slip_mm[sec].median()/cross_med:.0f}% of crossings; peak {row['peak_slip_mm']:.1f} mm; Mw {row['Mw']:.2f}")
