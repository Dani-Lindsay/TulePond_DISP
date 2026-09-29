#!/usr/bin/env python3
"""
DRAFT (2026-09-29): can a one-sided motion of the east block explain the
west-east asymmetry of the mean event? (Dani: the final check; the two-block
LOS figure shows east-block motion perpendicular to the fault reproducing the
east shortfall on both tracks.)

The slip model inverts the fault-parallel field, whose decomposition assumes
no fault-normal motion, so it cannot see this. Here the two LOS tracks are
inverted together (ascending + descending mean event, the 0.9-coherence
pixels of 06, levelled, quadtree, their own noise models), no top tie, with
the crossing surface constraint. A rigid motion p of the east block is tested
by a profile scan: for each trial p the block's LOS projection is removed
from every east-side sample of both tracks and the slip is re-inverted;
chi2(p) shows whether the data prefer p != 0. Two directions:
  - fault-normal, horizontal, positive = east block moving AWAY from the fault
  - vertical, positive = east block UP

Runs the machinery cells [1]-[7] of scripts/07_slip_model_tests.py unchanged.
Output: outputs/07_slip_model_tests/east_block_motion_test.csv, Figures_<date>/test_east_block_motion.png
"""
import os

source = open(os.path.join("scripts", "07_slip_model_tests.py")).read()
exec(source.split("# %% [markdown]\n# ## [8]")[0])

sigma_los = 8                         # the untied model's L-curve pick (drafts/07/lcurve_no_tie.py)
trial_mm = np.arange(-4.0, 4.01, 1.0) # east-block motion tried, mm

# the 0.9-coherence pixel set of 06 (the fault-parallel field is only defined there)
keep_09 = set(zip(np.round(fp["lon"][ok], 6), np.round(fp["lat"][ok], 6)))

tracks_los = {}
for track in ("ascending", "descending"):
    z = np.load(os.path.join(field_dir, f"csi_mean_event_{track}.npz"))
    in_09 = np.array([(a, b) in keep_09 for a, b in zip(np.round(z["lon"], 6), np.round(z["lat"], 6))])
    sar_t = insar(track, lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    sar_t.read_from_binary(z["vel_mm"][in_09], z["lon"][in_09], z["lat"][in_09], err=z["err_mm"][in_09],
                           incidence=z["incidence"][in_09], azimuth=z["azimuth"][in_09], remove_nan=True, remove_zeros=False)
    dist = dist_to_trace(sar_t, fault)
    sar_t.vel[(dist > swath_km) | (dist < exclude_km)] = np.nan
    km_t = lat_to_km_pp(sar_t.lat)
    sar_t.vel[(km_t < km_window[0]) | (km_t > km_window[1])] = np.nan
    sar_t.checkNaNs()
    align_track(sar_t)
    cov_t = covariance(sar_t, fault)
    ds_t = downsample(sar_t, fault, start_block_km, min_block_km)
    build_cd(ds_t, cov_t)
    tracks_los[track] = ds_t
    print(f"[{track}] {in_09.sum()} pixels at coherence 0.9, {ds_t.vel.size} samples")

# directions: along strike (for the sign check), fault-normal away from the fault on the east side, up
strike_rad = np.deg2rad(strike_bearing)
s_hat = np.array([np.sin(strike_rad), np.cos(strike_rad), 0.0])
n_hat = np.array([np.sin(strike_rad - np.pi / 2), np.cos(strike_rad - np.pi / 2), 0.0])   # bearing strike - 90: NE, away from the fault for the east block
u_hat = np.array([0.0, 0.0, 1.0])
for track, ds_t in tracks_los.items():
    print(f"[{track}] median LOS . strike {np.median(ds_t.los @ s_hat):+.3f}, . fault-normal {np.median(ds_t.los @ n_hat):+.3f}, . up {np.median(ds_t.los @ u_hat):+.3f}")
# right-lateral moves the east block along +strike: ascending must see it negative, descending positive (06 CHECK 2)
assert np.median(tracks_los["ascending"].los @ s_hat) < 0 < np.median(tracks_los["descending"].los @ s_hat), \
    "LOS vectors do not give the observed signs of right-lateral motion"

rows = []
for component, direction in (("fault-normal", n_hat), ("vertical", u_hat)):
    for p in trial_mm:
        flt = wrap_fault(*unrolled["smooth12"], dip_deg, f"{component} {p:+.1f}")
        datas = []
        for track, ds_t in tracks_los.items():
            ds = copy.deepcopy(ds_t)
            east = dist_to_trace(ds, flt, signed=True) > 0
            ds.vel[east] -= p * (ds.los[east] @ direction)       # remove the trial east-block motion
            datas.append(ds)
        surface = make_surface_interp(flt, surf_floor_mm)
        invert(flt, datas, sigma_los, surface=surface, tie_depth_km=None)
        row = dict(component=component, p_mm=p)
        for track, ds in zip(tracks_los, datas):
            resid = ds.vel - ds.synth
            row[f"chi2_{track[:3]}"] = float(np.dot(resid, np.linalg.solve(ds.Cd, resid)))
            row[f"rms_{track[:3]}"] = float(np.sqrt(np.mean(resid ** 2)))
        row["chi2"] = row["chi2_asc"] + row["chi2_des"]
        area_km2 = np.array([0.5 * np.linalg.norm(np.cross(q[1] - q[0], q[2] - q[0])) for q in flt.patch])
        potency = float(np.sum(area_km2 * 1e6 * flt.slip[:, 0] * 1e-3))
        row["Mw_30gpa"] = (2.0 / 3.0) * (np.log10(shear_modulus * potency) - 9.1)
        row["peak_slip_mm"] = float(flt.slip[:, 0].max())
        rows.append(row)
        print(f"{component} {p:+.1f} mm: chi2 {row['chi2']:.1f} (asc {row['chi2_asc']:.1f}, desc {row['chi2_des']:.1f}); "
              f"rms asc {row['rms_asc']:.3f} desc {row['rms_des']:.3f}; Mw {row['Mw_30gpa']:.3f}")
scan = pd.DataFrame(rows)
scan["dchi2"] = scan.groupby("component").chi2.transform(lambda c: c - c.min())
scan.to_csv(os.path.join(out_dir, "east_block_motion_test.csv"), index=False)
for component, grp in scan.groupby("component"):
    best = grp.loc[grp.chi2.idxmin()]
    at0 = grp[grp.p_mm == 0].iloc[0]
    print(f"{component}: best {best.p_mm:+.1f} mm; chi2 at 0 minus best = {at0.chi2 - best.chi2:.1f}")

mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.8), constrained_layout=True)
for component, colour in (("fault-normal", "#0072B2"), ("vertical", "#E69F00")):
    grp = scan[scan.component == component]
    axes[0].plot(grp.p_mm, grp.dchi2, "o-", ms=3, color=colour, label=component)
    axes[1].plot(grp.p_mm, grp.rms_asc, "o-", ms=3, color=colour, label=f"{component}, ascending")
    axes[1].plot(grp.p_mm, grp.rms_des, "s--", ms=3, color=colour, label=f"{component}, descending")
axes[0].axhline(4, color="0.5", ls=":", lw=0.8); axes[0].axvline(0, color="0.7", lw=0.6)
axes[0].set_xlabel("East-block motion (mm; + away from the fault / + up)"); axes[0].set_ylabel(r"$\Delta\chi^2$")
axes[0].legend(fontsize=6); axes[0].set_title("A  Profile of the east-block motion", loc="left", fontsize=7)
axes[1].axvline(0, color="0.7", lw=0.6)
axes[1].set_xlabel("East-block motion (mm)"); axes[1].set_ylabel("rms (mm)")
axes[1].legend(fontsize=5.5); axes[1].set_title("B  Fit per track", loc="left", fontsize=7)
f_fig = os.path.join(fig_dir, "test_east_block_motion.png")
fig.savefig(f_fig, dpi=300)
print("wrote", f_fig)
