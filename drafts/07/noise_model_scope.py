#!/usr/bin/env python3
"""
DRAFT (2026-09-29): is the noise model of the slip inversion representative,
and how much does the depth bracket depend on it?

The model's noise model (07 cell [4], `covariance`) is fitted to pixels more
than 1 km from the trace inside the 4 km modelling swath, both sides of the
fault together, lags up to 8 km. On the March-2026 field it finds no
long-wavelength term (amplitude 0, length at its 50 km bound). Two questions
(Dani, 2026-09-29):
  1. With lags only to 8 km, can a long-wavelength term be seen at all?
     Fit the same model over the whole 06 box instead, lags to 20 km.
  2. Pairs across the fault mix the two blocks. Fit each side separately.
The SSE section +/- 2 km along strike is left out of the one-sided fits, so
the event's own far field does not enter the noise.
Then the fixed-depth scan (07_slip_model_tests [9]) is repeated with each
noise model to see how the delta chi2 <= 4 bracket moves.

Runs the machinery cells [1]-[7] of scripts/07_slip_model_tests.py unchanged.
Run from the repo root in opera_disp_feb13 (PROJ_LIB pointed at the env).
Output: Figures_<date>/figS_noise_model_depth_scan.png/.pdf (supplement), outputs/07_slip_model_tests/noise_model_scope.csv
"""
import os
from scipy.optimize import curve_fit

source = open(os.path.join("scripts", "07_slip_model_tests.py")).read()
exec(source.split("# %% [markdown]\n# ## [8]")[0])

variogram_model = lambda hh, n2, s1, l1, s2, l2: n2 + s1 * (1 - np.exp(-hh / l1)) + s2 * (1 - np.exp(-hh / l2))


def empirical_variogram(x, y, v, hmax, npairs=400000, seed=0, same_side=None):
    """Plane-removed semivariogram, 0.1 km bins, pairs up to hmax (km).
    same_side: array of +1/-1 per pixel; pairs across the fault are dropped."""
    design = np.c_[np.ones_like(x), x, y]
    v = v - np.dot(design, np.linalg.lstsq(design, v, rcond=None)[0])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, v.size, npairs); j = rng.integers(0, v.size, npairs)
    h = np.hypot(x[i] - x[j], y[i] - y[j]); dv = v[i] - v[j]
    ok = (h > 0) & (h < hmax)
    if same_side is not None:
        ok &= same_side[i] == same_side[j]
    bins = np.arange(0, hmax + 0.1, 0.1)
    idx = np.digitize(h[ok], bins) - 1
    gam = np.array([0.5 * np.mean(dv[ok][idx == k] ** 2) if (idx == k).sum() > 50 else np.nan
                    for k in range(len(bins) - 1)])
    return 0.5 * (bins[1:] + bins[:-1]), gam


def fit_variogram(hb, gam):
    fin = np.isfinite(gam)
    sill = np.nanmean(gam[fin][-5:])
    p, _ = curve_fit(variogram_model, hb[fin], gam[fin], p0=(0.4 * sill, 0.3 * sill, 0.7, 0.5 * sill, 5.0),
                     bounds=([0, 0, 0.1, 0, 2.0], [2 * sill, 2 * sill, 2.0, 3 * sill, 50]), maxfev=20000)
    return p


# 1. the model's own noise model (07 cell [4]): swath, both sides, 8 km lags
d_model = dist_to_trace(sar, fault)
keep = (d_model > cov_mask_km) & np.isfinite(sar.vel)
hb_model, gam_model = empirical_variogram(sar.x[keep], sar.y[keep], sar.vel[keep], min(cov_distmax_km, 4 * swath_km))
fits = {"Inversion noise model: swath, both sides, lags to 8 km": (hb_model, gam_model, cov_p)}

# 2. the whole 06 box, one side at a time, SSE section +/- 2 km left out, lags to 20 km
box = insar("box", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
box.read_from_binary(fp["fp_event"][ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()),
                     los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
signed = dist_to_trace(box, fault, signed=True)
box_km = lat_to_km_pp(box.lat)
away = (np.abs(signed) > cov_mask_km) & ((box_km < creep_lo_km - 2) | (box_km > creep_hi_km + 2))
for side_name, side in (("west", signed < 0), ("east", signed > 0)):
    sel = away & side & np.isfinite(box.vel)
    hb, gam = empirical_variogram(box.x[sel], box.y[sel], box.vel[sel], 20.0)
    fits[f"{side_name.capitalize()} side, whole box, lags to 20 km"] = (hb, gam, fit_variogram(hb, gam))
    print(f"{side_name}: {sel.sum()} pixels")

# 3. the fixed-depth scan (07_slip_model_tests [9]) with each noise model
scan_fault = wrap_fault(*unrolled["smooth12"], dip_deg, "scan")
scan_ds = get_samples()
scan_fault.buildGFs(scan_ds, vertical=True, slipdir="s", verbose=False)
gmat = -scan_fault.G[scan_ds.name]["strikeslip"]
table = slip_table(scan_fault)
in_window = np.clip(np.minimum(table.km_pp - scan_km[0], scan_km[1] - table.km_pp) / scan_taper + 1, 0, 1).values
rows, scans_by_model = [], {}
for name, (_, _, p) in fits.items():
    ds = copy.deepcopy(scan_ds)
    build_cd(ds, p)
    chol = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
    data_w = np.linalg.solve(chol, ds.vel)
    chi2 = []
    for depth_bottom in scan_depths:
        if depth_bottom > table.z_bot.max() + 1e-6:
            continue
        design = np.linalg.solve(chol, np.c_[np.dot(gmat, in_window * (table.z_c.values < depth_bottom)), np.ones(ds.vel.size)])
        coef, *_ = np.linalg.lstsq(design, data_w, rcond=None)
        chi2.append((depth_bottom, float(np.sum((data_w - np.dot(design, coef)) ** 2))))
    scan = pd.DataFrame(chi2, columns=["D_km", "chi2"])
    scan["dchi2"] = scan.chi2 - scan.chi2.min()
    scans_by_model[name] = scan
    bracket = scan[scan.dchi2 <= 4]
    nug2, s1, l1, s2, l2 = p
    rows.append(dict(noise_model=name, nugget_mm=np.sqrt(nug2), short_mm=np.sqrt(s1), short_km=l1,
                     long_mm=np.sqrt(s2), long_km=l2, best_D_km=scan.loc[scan.chi2.idxmin(), "D_km"],
                     bracket_lo_km=bracket.D_km.min(), bracket_hi_km=bracket.D_km.max()))
result = pd.DataFrame(rows)
result.to_csv(os.path.join(out_dir, "noise_model_scope.csv"), index=False)
print(result.round(3).to_string(index=False))

# figure: A the semivariograms and fits, B the depth scan under each
mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
colours = ["#000000", "#0072B2", "#E69F00"]
h_fine = np.linspace(0.01, 20, 400)
for (name, (hb, gam, p)), colour in zip(fits.items(), colours):
    axes[0].plot(hb, gam, "o", ms=1.5, color=colour, alpha=0.5)
    axes[0].plot(h_fine, variogram_model(h_fine, *p), "-", color=colour, lw=1.2, label=name)
axes[0].axvline(8.0, color="0.5", ls=":", lw=0.8)
axes[0].text(8.2, 0.03, "Inversion lag limit", transform=axes[0].get_xaxis_transform(), fontsize=6, color="0.4", va="bottom")
axes[0].set_xlabel("Separation (km)"); axes[0].set_ylabel(r"Semivariance (mm$^2$)")
axes[0].legend(fontsize=6, loc="upper right")
utils.panel_title(axes[0], "A", "Semivariogram of the fault-parallel mean event")
for (name, scan), colour in zip(scans_by_model.items(), colours):
    axes[1].plot(scan.D_km, scan.dchi2, "o-", ms=2, color=colour, label=name)
axes[1].axhline(4, color="0.5", ls=":", lw=0.8)
axes[1].text(0.21, 4.5, r"$\Delta\chi^2$ = 4", fontsize=6, color="0.4")
axes[1].set_xscale("log"); axes[1].set_ylim(-1, 40); axes[1].set_xlim(0.2, 3.2)
axes[1].set_xticks([0.2, 0.5, 1, 2, 3]); axes[1].set_xticklabels(["0.2", "0.5", "1", "2", "3"]); axes[1].minorticks_off()
axes[1].set_xlabel("Bottom depth of uniform slip, D (km)"); axes[1].set_ylabel(r"$\Delta\chi^2$")
utils.panel_title(axes[1], "B", "Fixed-depth scan under each noise model")
f_fig = os.path.join(fig_dir, "figS_noise_model_depth_scan.png")   # supplement figure (Dani 2026-09-29)
fig.savefig(f_fig, dpi=300, facecolor="white")
fig.savefig(f_fig.replace(".png", ".pdf"), facecolor="white")
print("wrote", f_fig)
