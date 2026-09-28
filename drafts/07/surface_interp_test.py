"""Test: surface constraint interpolated to EVERY surface triangle (errors
scaled so total weight = the 29-point version). Synthetic recovery at
three depths, then the real model, both at smooth12 / dip 90 / sigma 24."""
src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])

def make_surface_interp(fault, floor_mm, step_km=surf_step_km):
    """One point per surface triangle: crossing running-median profile
    interpolated to the triangle centre; error x sqrt(triangles per step)."""
    base = get_surface(floor_mm)
    base_km = lat_to_km_pp(base.lat)
    order = np.argsort(base_km)
    surf = np.where([0 in p[:, 2] for p in fault.patch])[0]
    ll = np.array([fault.patchll[i].mean(0) for i in surf])
    tri_km = lat_to_km_pp(ll[:, 1])
    keep = (tri_km >= base_km.min()) & (tri_km <= base_km.max())
    surf, ll, tri_km = surf[keep], ll[keep], tri_km[keep]
    vals = np.interp(tri_km, base_km[order], base.vel[order])
    errs = np.interp(tri_km, base_km[order], base.err[order])
    per_step = np.array([np.sum(np.abs(tri_km - k) <= step_km / 2) for k in tri_km])
    errs = errs * np.sqrt(np.maximum(per_step, 1))
    ss = surfaceslip(f"crossing interp f{floor_mm:g}", lon0=frame_lon0,
                     lat0=frame_lat0, verbose=False)
    ss.read_from_binary(vals, ll[:, 0], ll[:, 1], err=errs)
    ss.buildCd()
    return ss

def surface_profile(fault):
    t = patch_table(fault)
    return t[t.z_top <= 1e-6]

rng = np.random.default_rng(7)
for D in (0.3, 0.8, 1.5):
    flt = wrap_fault(*unrolled["smooth12"], 90, f"rec_interp_{D:g}")
    t = patch_table(flt)
    w = np.clip(np.minimum(t.km_pp - scan_km[0], scan_km[1] - t.km_pp) / scan_taper + 1, 0, 1)
    slip_in = w.values * (t.z_c.values < D) * 10.0
    datas = []
    for track in tracks:
        ds = copy.deepcopy(get_downsampled(track, "smooth12", 90))
        flt.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
        flt.G[ds.name]["strikeslip"] *= -1.0
        chol = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
        ds.vel[:] = np.dot(flt.G[ds.name]["strikeslip"], slip_in) + np.dot(chol, rng.standard_normal(ds.vel.size))
        datas.append(ds)
    ss = make_surface_interp(flt, 0.5)
    flt.buildGFs(ss, slipdir="s", verbose=False)
    ss.vel[:] = np.dot(flt.G[ss.name]["strikeslip"], slip_in) + np.sqrt(np.diag(ss.Cd)) * rng.standard_normal(ss.vel.size)
    invert(flt, datas, 24, surface=ss)
    row, _ = metrics(flt, datas, ss)
    prof = row["depth_profile"]; z = sorted(prof); pk = max(prof, key=prof.get)
    print(f"REC interp 10 mm 0-{D:g} km: top cell {prof[z[0]]:.1f} mm, max {prof[pk]:.1f} mm at {pk*1000:.0f} m, Mw {row['Mw']:.2f}")

flt = wrap_fault(*unrolled["smooth12"], 90, "real_interp")
datas = [copy.deepcopy(get_downsampled(t, "smooth12", 90)) for t in tracks]
ss = make_surface_interp(flt, 0.5)
invert(flt, datas, 24, surface=ss)
row, table = metrics(flt, datas, ss)
s = table[(table.z_top <= 1e-6) & (table.km_pp >= 58) & (table.km_pp <= 61)]
prof = row["depth_profile"]; z = sorted(prof); pk = max(prof, key=prof.get)
print(f"REAL interp: {ss.vel.size} surface points; surface km58-61 median {s.slip_mm.median():.1f} mm; "
      f"top cell {prof[z[0]]:.1f}, max {prof[pk]:.1f} mm at {pk*1000:.0f} m; "
      f"Mw {row['Mw']:.2f}; near des {row['rms_near_des']:.2f}; surf rms {row['surf_rms_mm']:.2f}")
table.to_csv(os.path.join(out_dir, "slip_surface_interp_test.csv"), index=False)
