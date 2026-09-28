"""Residual maps: LOS asc, LOS desc (joint LOS inversion) and fault-parallel
(fp inversion), all with the interpolated surface constraint, sigma 24,
smooth12, dip 90. Fault-parallel residual also shown scaled to desc LOS
(x0.49) so colour scales are comparable."""
src = open("scripts/07_slip_model_tests.py").read()
exec(src.split("# %%\n## [9]")[0])
import matplotlib.pyplot as plt
from insar_utils import mpl_paper_style

flt = wrap_fault(*unrolled["smooth12"], 90, "los_map")
datas = [copy.deepcopy(get_downsampled(t, "smooth12", 90)) for t in tracks]
invert(flt, datas, 24, surface=make_surface_interp(flt, surf_floor_mm))
panels = [(ds.lon, ds.lat, ds.vel - ds.synth, f"LOS {ds.name.split()[-1][:4]} residual") for ds in datas]

fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
look = np.array([np.sin(np.deg2rad(float(fp["strike"]))), np.cos(np.deg2rad(float(fp["strike"]))), 0.0])
ok = np.isfinite(fp["fp_event"])
f2 = wrap_fault(*unrolled["smooth12"], 90, "fp_map")
sar = insar("fault-parallel", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
sar.read_from_binary(fp["fp_event"][ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()),
                     los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
dd = dist_to_trace(sar, f2); sar.vel[(dd > swath_km) | (dd < exclude_km)] = np.nan
kk = lat_to_km_pp(sar.lat); sar.vel[(kk < km_window[0]) | (kk > km_window[1])] = np.nan
sar.checkNaNs(); align_track(sar)
cov_p = covariance(sar, f2)
dsf = downsample(sar, f2, start_block_km, min_block_km); build_cd(dsf, cov_p)
invert(f2, [dsf], 24, surface=make_surface_interp(f2, surf_floor_mm))
panels.append((dsf.lon, dsf.lat, 0.49 * (dsf.vel - dsf.synth), "fault-parallel residual x0.49"))

mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.0), constrained_layout=True)
trace = np.c_[flt.lon, flt.lat]
for ax, (lon, lat, r, title) in zip(axes, panels):
    sc = ax.scatter(lon, lat, c=r, s=4, cmap="RdBu_r", vmin=-1.5, vmax=1.5, linewidths=0)
    ax.plot(trace[:, 0], trace[:, 1], "k-", lw=0.5)
    ax.set_aspect(1 / np.cos(np.deg2rad(frame_lat0)))
    ax.set_xlim(-122.08, -121.86); ax.set_ylim(37.49, 37.62)
    ax.set_title(title, loc="left", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=axes, shrink=0.7, label="residual (mm, desc-LOS scale)")
out = os.path.join(repo_dir, "Figures_2026-09-25", "draft_residuals_los_vs_fp.png")
fig.savefig(out, dpi=250); print("wrote", out)
