"""los_inversion.py (2026-10-30, Dani): TEST -- invert the ascending and descending LOS mean-event fields directly, strike-slip
only, instead of the fault-parallel decomposition: do the dip preference and the slip result hold without the
no-fault-normal-motion assumption of the decomposition?

Data: 06's per-track LOS mean event at coherence 0.9 (model_utils.los_from_steps; the 0.9 fault-parallel gate is
re-checked here), on the pixels the decomposition used (finite fault-parallel field: valid on both tracks), cut like
07a (4 km swath, 30 m straddling pixel dropped, km 52-67), averaged in the SAME 2386 quadtree blocks as 07a (CSI
imagedownsampling.downsampleFromSampler with 07a's sampler, positions asserted equal to 07a's samples); each block's LOS
unit vector = CSI's normalised mean of the per-pixel vectors (insar_utils.los_unit_vector of 06's incidence / azimuth).
Green's functions: the 3D displacements of the existing PyLith runs at the samples (points.h5) projected on each block's
LOS vector (model_utils.build_G_los; its fault-parallel projection is asserted equal to the run's G product). No new
PyLith runs. Noise: a 5 km-mask semivariogram per track (model_utils.covariance, the recipe of 07a: 15 km swath, plane
removed); diagonal weights = each track's nugget^2. A plane (1, x, y) per track solved jointly; CSI exponential prior
(model_config.lam_xz_km), slip >= 0, strike-slip only (model_utils.invert_tents_multi: both tracks in one CSI solve);
each model at its own L-curve corner (model_utils.own_corner, the locked rule; misfit summed over both tracks).
Scores: RMS per track (fitted samples; within 500 m west / east in the SSE section), Mw (30 GPa and harmonic SFSVM),
peak slip and depth, depth50/90; the M7 near-fault deflection on the FAULT-PARALLEL prediction (fault-parallel G x slip)
against the fault-parallel data with a plane fitted to their difference removed; the fault-parallel RMS of that
prediction. Comparison: the fault-parallel-only inversion of the same run (sensitivity_rows_before_70w_2026-10-30.csv,
re-solved here at its recorded sigma for the M7 metric; RMS asserted to reproduce the csv).
Runs: gf_shallow_sfcvm_order2 (vertical), gf_shallow_dip70w_sfcvm_order2, gf_shallow_dip70e_sfcvm_order2, plus
gf_shallow_dip60w/50w only if their done markers exist.
Outputs (this folder): los_inversion.csv, los_lcurve_<run>.csv, los_nuggets.csv, los_inversion_draft.png, los_inversion.log.
Run: cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 \\
     nice -n 15 ~/miniforge3/envs/opera_disp_feb13/bin/python -u <this file> > <this folder>/los_inversion.log 2>&1
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.expanduser("~/tools/internal/TulePond_DISP")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "sensitivity"))
import common                                                       # noqa: E402
from common import cfg, model_utils                                 # noqa: E402
from csi import insar, imagedownsampling as imdown                  # noqa: E402
import insar_utils as utils                                          # noqa: E402
from TulePond_filepaths import paths_out, insar_dic_list, model_frame_lonlat, data_dir   # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
t_start = time.time()
RUNS = {"Vertical": "gf_shallow_sfcvm_order2", "70 W": "gf_shallow_dip70w_sfcvm_order2", "70 E": "gf_shallow_dip70e_sfcvm_order2",
        "60 W": "gf_shallow_dip60w_sfcvm_order2", "50 W": "gf_shallow_dip50w_sfcvm_order2"}
FP_ROWS = {"Vertical": ("Baseline (fine mesh)", "Selected"), "70 W": ("Fault dip, fine mesh", "70 W"),
           "70 E": ("Fault dip, fine mesh", "70 E"), "60 W": ("Fault dip, fine mesh", "60 W"), "50 W": ("Fault dip, fine mesh", "50 W")}
TRACKS = ("ascending", "descending")
runs = {dip: run for dip, run in RUNS.items() if os.path.exists(os.path.join(common.MARK, run))}
print("runs with a done marker:", runs, flush=True)

# ---- 1. data: 06's LOS mean event at coherence 0.9 on the decomposition's pixels; gate: the fault-parallel field reproduces
inputs = common.load_inputs()
profile_lats = inputs["profile_lats"]
trace_fault = inputs["trace"]["ref_fault"]
field_dir = paths_out["06_event_field_geometry"]
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
strike_deg = float(fp["strike"])
assert abs(strike_deg - inputs["strike_deg"]) < 1e-9
steps = {d["ORBIT_DIRECTION"]: os.path.join(field_dir, f"tulepond_steps_{d['ORBIT_DIRECTION']}.h5") for d in insar_dic_list}
masks = {d["ORBIT_DIRECTION"]: d["mask_file"][0.9] for d in insar_dic_list}
fp_check = model_utils.fault_parallel_from_steps(steps, strike_deg, cfg.decomposition_window_px, masks)
assert np.array_equal(np.isfinite(fp_check), np.isfinite(fp["fp_event"])) and np.nanmax(np.abs(fp_check - fp["fp_event"])) == 0.0
los_maps = model_utils.los_from_steps(steps, masks)
valid = np.isfinite(fp["fp_event"])
for track in TRACKS:
    assert np.isfinite(los_maps[track]["mean"][valid]).all(), f"{track}: a decomposition pixel has no LOS value"
print(f"gate PASS (fault-parallel field reproduced); {valid.sum():,} pixels valid on both tracks", flush=True)


def los_dataset(track, swath_km):
    """One track's LOS field on the decomposition's pixels as a csi.insar data set with per-pixel LOS vectors, cut exactly as
    model_utils.fault_parallel_dataset cuts the fault-parallel field (swath, straddling pixel, along-strike window)."""
    values = np.where(valid, los_maps[track]["mean"], np.nan)
    los = np.moveaxis(utils.los_unit_vector(los_maps[track]["inc"], los_maps[track]["azi"]), 0, -1)   # (nr, nc, 3) E, N, U
    ok = np.isfinite(values)
    sar = insar(f"{track} LOS", lon0=model_frame_lonlat[0], lat0=model_frame_lonlat[1], verbose=False)
    sar.read_from_binary(values[ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()), los=los[ok], remove_nan=True, remove_zeros=False)
    trace_dist = model_utils.dist_to_trace(sar, trace_fault)
    sar.vel[(trace_dist > swath_km) | (trace_dist < cfg.exclude_km)] = np.nan
    km_pp = model_utils.km_from_pinole(sar.lat, profile_lats)
    sar.vel[(km_pp < cfg.km_window[0]) | (km_pp > cfg.km_window[1])] = np.nan
    sar.checkNaNs()
    return sar


# ---- 2. 07a's quadtree on the fault-parallel field, then the same blocks on each LOS track
sar_fp, _ = model_utils.fault_parallel_dataset(fp["fp_event"], fp["lon"], fp["lat"], strike_deg, trace_fault, model_frame_lonlat,
                                               cfg.swath_km, cfg.exclude_km, cfg.km_window, profile_lats)
sampler_fp = imdown(sar_fp.name, sar_fp, trace_fault, verbose=False)
sampler_fp.initialstate(cfg.start_block_km, cfg.min_block_km, tolerance=0.1, plot=False)
sampler_fp.distanceBased(chardist=0.0, expodist=1, plot=False)
md = inputs["model_data"]
assert np.array_equal(np.asarray(sampler_fp.newimage.lon, float), md["sample_lon"]), "the quadtree is not 07a's"
samples_los, block_los = {}, {}
nuggets = []
for track in TRACKS:
    sar = los_dataset(track, cfg.swath_km)
    assert sar.vel.size == sar_fp.vel.size and np.allclose(sar.x, sar_fp.x) and np.allclose(sar.y, sar_fp.y), f"{track}: other pixels"
    sampler = imdown(sar.name, sar, trace_fault, verbose=False)
    sampler.downsampleFromSampler(sampler_fp)
    ds = sampler.newimage
    for key, ref in (("lon", md["sample_lon"]), ("lat", md["sample_lat"]), ("x", md["sample_x_km"]), ("y", md["sample_y_km"]),
                     ("wgt", md["wgt"])):
        assert np.allclose(np.asarray(getattr(ds, key), float), ref, atol=1e-9), f"{track}: block {key} differ from 07a's samples"
    block_los[track] = np.asarray(ds.los, float)
    # noise: the 5 km-mask semivariogram of this track (07a's recipe: 15 km swath, plane removed)
    sar_wide = los_dataset(track, cfg.noise_swath_km)
    cov_p = np.asarray(model_utils.covariance(sar_wide, trace_fault, cfg.noise_swath_km, cfg.noise_mask_km, cfg.noise_distmax_km,
                                              cfg.noise_npairs, cfg.noise_seed), float)
    nuggets.append(dict(track=track, nugget_mm=np.sqrt(cov_p[0]), short_sd_mm=np.sqrt(cov_p[1]), short_range_km=cov_p[2],
                        long_sd_mm=np.sqrt(cov_p[3]), long_range_km=cov_p[4]))
    smp = model_utils.samples_from_arrays(f"{track} LOS samples", md["sample_lon"], md["sample_lat"], md["sample_x_km"],
                                          md["sample_y_km"], np.asarray(ds.vel, float), md["wgt"], md["look"], model_frame_lonlat,
                                          cov_p, cfg.nugget_power)
    samples_los[track] = (smp, cov_p)
    print(f"{track}: {ds.vel.size} blocks; LOS east/north/up median {np.round(np.median(block_los[track], 0), 3)}; nugget "
          f"{np.sqrt(cov_p[0]):.3f} mm, exponential {np.sqrt(cov_p[1]):.3f} mm / {cov_p[2]:.2f} km", flush=True)
pd.DataFrame(nuggets).to_csv(os.path.join(OUT, "los_nuggets.csv"), index=False)

# ---- 3. per run: LOS Green's functions, the joint inversion at its own corner, the scores
frozen = pd.read_csv(cfg.archived["sensitivity_rows_20261030"])
samples_fp = common.samples_for(inputs, "default")
fp_data = md["data_mm"].astype(float)
rows = []
for dip, run in runs.items():
    setup = common.make_setup(inputs, run)
    gnpz = setup["gnpz"]["m"]
    keep, west, east, in_section = setup["keep"], setup["west_near"], setup["east_near"], setup["in_section"]
    G_fp = setup["G"]["m"]
    G_los = {}
    for track in TRACKS:
        G_los[track], G_fp_check = model_utils.build_G_los(gnpz, inputs["points"]["default"], block_los[track], cfg.gf_pylith_output_dir)
        assert np.allclose(G_fp_check, G_fp, atol=1e-12), f"{run} {track}: the fault-parallel projection does not reproduce G"
    Cd = {track: model_utils.data_cd("nugget", samples_los[track][0], samples_los[track][1], 0.0, keep=keep) for track in TRACKS}

    def solve(sigma):
        fault_model, datas = model_utils.invert_tents_multi(
            setup["template"], [(samples_los[t][0], G_los[t], Cd[t]) for t in TRACKS], sigma, cfg.lam_xz_km, model_frame_lonlat,
            ramp_spec=cfg.ramp_spec, name=f"LOS {dip} sigma {sigma:.3f}")
        return fault_model, datas, None
    sigma, sigma_base, lcurve = model_utils.own_corner(solve, cfg.sigma_grid_base, cfg.sigma_refine_factors, setup["tents"],
                                                       cfg.mu_ref_pa, f"LOS {dip}", [keep, keep])
    lcurve.to_csv(os.path.join(OUT, f"los_lcurve_{run}.csv"), index=False)
    fault_model, datas, _ = solve(sigma)
    slip = fault_model.slip[:, 0].copy()
    mu_harmonic = common.mu_harmonic_of(run, setup)
    per_track = {}
    for track, data in zip(TRACKS, datas):
        per_track[track] = model_utils.tent_numbers(slip, data, setup["tents"], keep, west, east, cfg.mu_ref_pa, mu_harmonic)
    numbers = per_track["ascending"]
    # the fault-parallel prediction of the LOS-fitted model vs the fault-parallel data (a plane fitted to their difference)
    fp_pred = G_fp @ slip
    design = np.c_[np.ones(fp_data.size), md["sample_x_km"], md["sample_y_km"]]
    plane_coef = np.linalg.lstsq(design[keep], (fp_data - fp_pred)[keep], rcond=None)[0]
    fp_data_pr = fp_data - design @ plane_coef
    fp_res = fp_data_pr - fp_pred
    deflection = {}
    for name, values in (("data", fp_data_pr), ("model", fp_pred)):
        for side, label in ((-1, "west"), (1, "east")):
            deflection[f"{name}_{label}"] = model_utils.side_deflection_mm(values, setup["signed_d_km"], in_section, keep, side,
                                                                           cfg.near_deflection_km, cfg.far_deflection_km)[0]
    row = dict(data="LOS asc + desc", dip=dip, run=run, sigma=sigma, sigma_base_corner=sigma_base,
               rms_asc_mm=per_track["ascending"]["rms_mm"], rms_asc_west500_mm=per_track["ascending"]["rms_west_mm"],
               rms_asc_east500_mm=per_track["ascending"]["rms_east_mm"], rms_desc_mm=per_track["descending"]["rms_mm"],
               rms_desc_west500_mm=per_track["descending"]["rms_west_mm"], rms_desc_east500_mm=per_track["descending"]["rms_east_mm"],
               rms_fp_mm=float(np.sqrt(np.mean(fp_res[keep] ** 2))), rms_fp_west500_mm=float(np.sqrt(np.mean(fp_res[west] ** 2))),
               rms_fp_east500_mm=float(np.sqrt(np.mean(fp_res[east] ** 2))),
               Mw_30gpa=numbers["Mw_30gpa"], Mw_sfcvm=numbers["Mw_sfcvm"], potency_m3=numbers["potency_m3"],
               peak_slip_mm=numbers["peak_slip_mm"], peak_depth_m=numbers["peak_depth_m"], depth50_m=numbers["depth50_m"],
               depth90_m=numbers["depth90_m"], **{f"deflection_{k}_mm": v for k, v in deflection.items()})
    rows.append(row)
    # the fault-parallel-only inversion of the same run, re-solved at its recorded sigma (frozen 2026-10-30 csv)
    rec = frozen[(frozen.row == FP_ROWS[dip][0]) & (frozen.variant == FP_ROWS[dip][1]) & (frozen.run == run)]
    if len(rec) == 1:
        rec = rec.iloc[0]
        fm, d_fp, _ = common.solve(inputs, setup, G_fp, samples_fp, float(rec.sigma), label=f"fault-parallel {dip}")
        n_fp, slip_fp = common.score(inputs, setup, fm, d_fp, mu_per_triangle=mu_harmonic)
        assert abs(n_fp["rms_mm"] - rec.rms_mm) < 1e-3, f"{dip}: fault-parallel RMS {n_fp['rms_mm']:.4f} vs csv {rec.rms_mm:.4f}"
        defl_fp = {}
        for name, values in (("data", d_fp.vel), ("model", d_fp.synth)):
            for side, label in ((-1, "west"), (1, "east")):
                defl_fp[f"{name}_{label}"] = model_utils.side_deflection_mm(values, setup["signed_d_km"], in_section, keep, side,
                                                                            cfg.near_deflection_km, cfg.far_deflection_km)[0]
        rows.append(dict(data="Fault-parallel", dip=dip, run=run, sigma=float(rec.sigma), sigma_base_corner=float(rec.sigma_base_corner),
                         rms_fp_mm=n_fp["rms_mm"], rms_fp_west500_mm=n_fp["rms_west_mm"], rms_fp_east500_mm=n_fp["rms_east_mm"],
                         Mw_30gpa=n_fp["Mw_30gpa"], Mw_sfcvm=n_fp["Mw_sfcvm"], potency_m3=n_fp["potency_m3"],
                         peak_slip_mm=n_fp["peak_slip_mm"], peak_depth_m=n_fp["peak_depth_m"], depth50_m=n_fp["depth50_m"],
                         depth90_m=n_fp["depth90_m"], **{f"deflection_{k}_mm": v for k, v in defl_fp.items()}))
    table = pd.DataFrame(rows)
    table["provenance"] = common.stamp(__file__)
    table.to_csv(os.path.join(OUT, "los_inversion.csv"), index=False)
    print(f"done {dip}: LOS sigma {sigma:.2f}, Mw {numbers['Mw_30gpa']:.3f}, RMS asc {row['rms_asc_mm']:.3f} desc {row['rms_desc_mm']:.3f}, "
          f"fault-parallel RMS {row['rms_fp_mm']:.3f} (W500 {row['rms_fp_west500_mm']:.3f}, E500 {row['rms_fp_east500_mm']:.3f}) "
          f"({(time.time() - t_start) / 60:.0f} min)", flush=True)

# ---- 4. draft figure: the dip ranking, LOS fit vs fault-parallel fit
table = pd.DataFrame(rows)
x_of = {"50 W": -40, "60 W": -30, "70 W": -20, "Vertical": 0, "70 E": 20}
table["x"] = table.dip.map(x_of)
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10})
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
los = table[table.data == "LOS asc + desc"].sort_values("x")
fpo = table[table.data == "Fault-parallel"].sort_values("x")
axes[0].plot(los.x, los.rms_asc_mm, "o-", color="#D55E00", label="LOS fit: ascending")
axes[0].plot(los.x, los.rms_desc_mm, "s--", color="#0072B2", mfc="white", label="LOS fit: descending")
axes[0].set_ylabel("RMS of the LOS fit (mm)")
axes[1].plot(los.x, los.rms_fp_mm, "o-", color="#D55E00", label="LOS-fitted model")
axes[1].plot(fpo.x, fpo.rms_fp_mm, "s--", color="k", label="Fault-parallel fit")
axes[1].set_ylabel("Fault-parallel RMS (mm)")
axes[2].plot(los.x, los.Mw_30gpa, "o-", color="#D55E00", label="LOS fit")
axes[2].plot(fpo.x, fpo.Mw_30gpa, "s--", color="k", label="Fault-parallel fit")
axes[2].set_ylabel("Mw (30 GPa)")
for frame, colour, label in ((los, "#D55E00", "LOS-fitted model"), (fpo, "k", "Fault-parallel fit")):
    axes[3].plot(frame.x, frame.deflection_model_west_mm, "o-", color=colour, label=f"{label}: west")
    axes[3].plot(frame.x, frame.deflection_model_east_mm, "o--", color=colour, mfc="white", label=f"{label}: east")
axes[3].plot(fpo.x, fpo.deflection_data_west_mm, "x", color="0.4", ms=9, label="Data: west / east")
axes[3].plot(fpo.x, fpo.deflection_data_east_mm, "x", color="0.4", ms=9)
axes[3].set_ylabel("Near-fault deflection (mm)")
for ax in axes:
    ax.set_xticks(sorted(set(table.x)))
    ax.set_xticklabels([k for k, v in sorted(x_of.items(), key=lambda kv: kv[1]) if v in set(table.x)])
    ax.set_xlabel("Fault dip")
    ax.legend(loc="best")
fig.suptitle("DRAFT  LOS (ascending + descending, strike-slip only) vs fault-parallel inversions, 3D SFSVM, fine mesh", fontsize=13)
fig.savefig(os.path.join(OUT, "los_inversion_draft.png"), dpi=150, facecolor="white")
pd.set_option("display.width", 250)
print(table.drop(columns=["x"]).round(3).T.to_string())   # (fixed 2026-10-31: the table rebuilt from rows has no provenance column)
print(f"wrote {OUT} in {(time.time() - t_start) / 60:.0f} min")
