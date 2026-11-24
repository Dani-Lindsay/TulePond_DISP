"""Build Data S1 (csv files of the data behind the figures and tables) of the Tule Pond paper (Science, submitted
24 Nov 2026), from the products of a full run.

Reads the data root of a run (TULEPOND_DATA_DIR, through TulePond_filepaths; read only) and writes only into OUT_DIR:
OUT_DIR/Data_S1/ (one csv per figure panel or table, plus README.csv) and OUT_DIR/_sandbox/ (scratch of the recomputed
values). Then make_workbook.py turns Data_S1/ into the one Excel file Science asks for.
Figure numbers follow the submission, with the detector figure as fig. S6. Values the notebooks compute but never save
(fig. S6 statistic and Oakland stack, fig. S5 per-creepmeter stacks, fig. S3 interpolation flags, Fig. 3B 7-day rain)
are recomputed here with notebook 04's own cells (found by their content, exec'd with the output folders redirected to
_sandbox/ and no registry writes).
Fig 1B's earthquakes are those notebook 14 drew (outputs/14_paper_figures/fig1_earthquakes_ncedc_dd.csv, the public
NCEDC DD catalogue) and Fig 1D's creepmeter bars the larger side of notebook 05's bootstrap 95 % CI, as in Fig 1.

Run (repo root, environment.yml env):
    TULEPOND_DATA_DIR=<data root> python scripts/data_S1/build_data_S1.py OUT_DIR [--fig1-earthquakes CSV]
--fig1-earthquakes replaces notebook 14's csv (only to rebuild from a data root whose notebook 14 predates the DD
catalogue, e.g. the frozen v1.0.0 release).
Built from the 23-24 Nov 2026 submission builder (Data_S1.xlsx as uploaded); see README.md here.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import insar_utils as utils                                          # noqa: E402
from TulePond_filepaths import common_paths, data_dir, paths_out     # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("out_dir", help="where Data_S1/ and _sandbox/ are written (must not exist inside a frozen or submitted folder)")
parser.add_argument("--fig1-earthquakes", default=os.path.join(paths_out["14_paper_figures"], "fig1_earthquakes_ncedc_dd.csv"),
                    help="the Fig 1B earthquakes (notebook 14's csv by default)")
args = parser.parse_args()
OUT_ROOT = os.path.realpath(args.out_dir)
for forbidden in ("FROZEN_", "SUBMITTED_"):
    assert forbidden not in OUT_ROOT, f"refusing to write into {OUT_ROOT}"
assert not OUT_ROOT.startswith(os.path.realpath(data_dir) + os.sep), f"{OUT_ROOT} is inside the data root {data_dir} (read only here)"
REL = data_dir                                   # the data root read here
OUT = common_paths["out_dir"]                    # its outputs/
SANDBOX = os.path.join(OUT_ROOT, "_sandbox")
print(f"reading {REL}\nwriting {OUT_ROOT}")

sheets = {}     # name -> DataFrame
readme = []     # rows of the README sheet


def add(name, df, panel, contents, source, notebook):
    """Register one sheet; fail loud on an empty table or a name Excel would truncate."""
    assert len(name) <= 31, name
    assert name not in sheets, name
    assert len(df) > 0, f"{name}: empty"
    sheets[name] = df
    readme.append(dict(sheet=name, figure_panel=panel, contents=contents, source_file=source, notebook=notebook))


def npz(path):
    return dict(np.load(path, allow_pickle=False))


def csv(rel):
    return pd.read_csv(os.path.join(OUT, rel))


# ---------------------------------------------------------------- Fig 1 (notebook 04's inputs, as drawn by notebook 14)
z1 = npz(os.path.join(paths_out["04_event_detection"], "fig1_inputs.npz"))
eq = pd.read_csv(args.fig1_earthquakes)
assert list(eq.columns[:6]) == ["evID", "YYYY", "Lat", "Lon", "Depth", "Mag"], f"{args.fig1_earthquakes}: {list(eq.columns)}"
assert eq.YYYY.between(float(z1["t0"]), float(z1["t1"])).all() and eq.km.between(float(z1["w_lo"]), float(z1["w_hi"])).all()
cm_rates = {str(c): (float(k), float(r)) for c, k, r in zip(z1["cm_codes"], z1["cm_km"], z1["cm_rate"])}
bars = utils.creepmeter_bar_half_widths(os.path.join(paths_out["05_validation"], "station_agreement.csv"), cm_rates)
hov = pd.DataFrame(z1["hovmoller"], index=pd.Index(np.round(z1["hov_km"], 3), name="km_from_Point_Pinole"),
                   columns=[f"{t:.4f}" for t in z1["t_regular"]])
add("Fig1B_slip_grid", hov.reset_index(), "Fig. 1B",
    "Slip above the inter-event rate and annual cycle (mm); rows = crossing (km from Point Pinole), "
    "columns = date (decimal year); empty = data gap",
    "04/fig1_inputs.npz: hovmoller", "04_event_detection, 14_paper_figures")
add("Fig1B_earthquakes", pd.DataFrame({"decimal_year": eq.YYYY.values, "km_from_Point_Pinole": eq.km.values,
                                        "magnitude": eq.Mag.values, "depth_km": eq.Depth.values}),
    "Fig. 1B", "NCAeqDD earthquakes within 2 km of the trace, InSAR period",
    "14/fig1_earthquakes_ncedc_dd.csv (NCEDC DD catalogue)", "14_paper_figures")
add("Fig1B_repeaters", pd.DataFrame({"decimal_year": z1["cre_yr"], "km_from_Point_Pinole": z1["cre_km"]}),
    "Fig. 1B", "Repeating earthquakes (Taira GEN3) within 2 km of the trace", "04/fig1_inputs.npz: cre_*",
    "04_event_detection")
c = pd.DataFrame({"km_from_Point_Pinole": z1["km_window"]})
for i, ep in enumerate(z1["step_epochs"]):
    c[f"event_{ep:.3f}_cumulative_slip_smoothed_mm"] = z1["cum_prof"][i]
    c[f"event_{ep:.3f}_cumulative_slip_per_crossing_mm"] = z1["cum_prof_raw"][i]
add("Fig1C_cumulative_slip", c, "Fig. 1C",
    "Cumulative slip of the five events per crossing and smoothed (0.4 km Gaussian), mm",
    "04/fig1_inputs.npz: cum_prof, cum_prof_raw, km_window", "04_event_detection")
add("Fig1D_rates", pd.DataFrame({
        "km_from_Point_Pinole": z1["km_window"],
        "inter_event_rate_per_crossing_mm_yr": z1["rate_point_secular"],
        "inter_event_rate_smoothed_mm_yr": z1["rate_line_secular"],
        "event_driven_rate_per_crossing_mm_yr": z1["rate_point_event_rate"],
        "event_driven_rate_smoothed_mm_yr": z1["rate_line_event_rate"],
        "total_rate_per_crossing_mm_yr": z1["rate_point_total"],
        "total_rate_smoothed_mm_yr": z1["rate_line_total"]}),
    "Fig. 1D", "Inter-event, event-driven (event slip / record length) and total rates, mm/yr",
    "04/fig1_inputs.npz: rate_*", "04_event_detection")
add("Fig1D_arrays", pd.DataFrame({"km_from_Point_Pinole": z1["aa_km"],
                                   "alignment_array_long_term_rate_mm_yr": z1["aa_rate_longterm"]}),
    "Fig. 1D", "Whole-record alignment-array rates", "04/fig1_inputs.npz: aa_*", "04_event_detection")
add("Fig1A_1D_creepmeters", pd.DataFrame({"creepmeter": z1["cm_codes"].astype(str), "km_from_Point_Pinole": z1["cm_km"],
                                           "rate_InSAR_period_mm_yr": z1["cm_rate"],
                                           "rate_sigma_mm_yr": [bars[str(c)][0] for c in z1["cm_codes"]]}),   # Fig 1D bar half-width
    "Figs. 1A and 1D", "Creepmeter positions and rates over the InSAR window (Fig. 1D shows the 95% bootstrap CI "
    "of CFW1; see fig1 v4 script)", "04/fig1_inputs.npz: cm_*; 05/station_agreement.csv", "04_event_detection, 05_validation")
add("Fig1E_peak_zone_stack", pd.DataFrame({"decimal_year": z1["dates"], "mean_slip_mm": z1["stack_mean"],
                                            "one_sd_across_crossings_mm": z1["stack_std"],
                                            "joint_model_mm": z1["model_full"],
                                            "in_onset_window": z1["is_event_epoch"].astype(bool)}),
    "Fig. 1E", f"Mean fault-parallel slip of the {int(z1['stack_n'])} crossings in the peak-slip zone, "
    "1 SD across crossings, and the joint model", "04/fig1_inputs.npz: dates, stack_*, model_full",
    "04_event_detection")

# ---------------------------------------------------------------- Fig 2
z2 = npz(os.path.join(OUT, "06_event_field_geometry", "fig2_inputs.npz"))
for comp, label in (("fp", "fault_parallel"), ("up", "vertical")):
    for prof in ("X", "Y"):
        df = pd.DataFrame({"distance_from_trace_m_east_positive": z2[f"prof_{comp}_{prof}_d"],
                           f"inter_event_{label}_velocity_mm_yr": z2[f"prof_{comp}_{prof}_v"]})
        de = pd.DataFrame({"distance_from_trace_m_east_positive": z2[f"prof_{comp}_{prof}_de"],
                           f"mean_event_{label}_displacement_mm": z2[f"prof_{comp}_{prof}_ve"]})
        panels = {"fp": ("C", "G"), "up": ("D", "H")}[comp]
        add(f"Fig2{panels[0]}_{comp}_{prof}_rate", df, f"Fig. 2{panels[0]}",
            f"Profile {prof}-{prof}' pixels within 150 m along strike, inter-event {label} velocity",
            "06/fig2_inputs.npz", "06_event_field_geometry")
        add(f"Fig2{panels[1]}_{comp}_{prof}_event", de, f"Fig. 2{panels[1]}",
            f"Profile {prof}-{prof}' pixels within 150 m along strike, mean event {label} displacement",
            "06/fig2_inputs.npz", "06_event_field_geometry")
z3d = npz(os.path.join(OUT, "09_model_3d", "selected_model.npz"))
zh = npz(os.path.join(OUT, "08_model_homogeneous", "homogeneous_model.npz"))
assert np.allclose(z3d["sample_lon"], zh["sample_lon"]) and np.allclose(z3d["sample_lat"], zh["sample_lat"])
add("Fig2GIJ_quadtree_samples", pd.DataFrame({
        "lon": z3d["sample_lon"], "lat": z3d["sample_lat"],
        "signed_distance_from_trace_km": z3d["signed_d_km"], "km_from_Point_Pinole": z3d["sample_km_pp"],
        "fitted": z3d["keep"].astype(bool),
        "data_fault_parallel_mm": z3d["data_mm"], "data_minus_3D_plane_mm": z3d["data_plane_removed_mm"],
        "model_3D_SFSVM_mm": z3d["model_mm"], "plane_3D_mm": z3d["plane_mm"], "residual_3D_mm": z3d["residual_mm"],
        "model_homogeneous_mm": zh["model_mm"], "plane_homogeneous_mm": zh["plane_mm"]}),
    "Figs. 2G, 2H, 2I and 2J", "Mean-event fault-parallel displacement at the quadtree samples, the 3D (SFSVM) and "
    "homogeneous models (both 70 deg W) and the 3D residual; the model lines in G and H are these values along the "
    "profiles", "09/selected_model.npz, 08/homogeneous_model.npz", "08_model_homogeneous, 09_model_3d")
add("Fig2K_cross_fault_step", pd.DataFrame({"km_from_Point_Pinole": z3d["crossing_km"],
                                             "data_step_60_150m_mm": z3d["data_step_mm"],
                                             "model_3D_step_60_150m_mm": z3d["model_step_mm"]}),
    "Fig. 2K", "Cross-fault step (60-150 m either side) of the mean event and of the 3D model",
    "09/selected_model.npz", "09_model_3d")
add("Fig2L_slip_on_fault", pd.DataFrame({"km_from_Point_Pinole": z3d["imp_km"], "depth_km": z3d["imp_depth_km"],
                                          "x_m": z3d["imp_xyz"][:, 0], "y_m": z3d["imp_xyz"][:, 1],
                                          "z_m": z3d["imp_xyz"][:, 2], "right_lateral_slip_mm": z3d["slip_mm"]}),
    "Fig. 2L", "Right-lateral slip of the preferred 3D model at the fault vertices",
    "09/selected_model.npz", "09_model_3d")

# ---------------------------------------------------------------- Fig 3
hrkt = pd.read_csv(os.path.join(REL, "external_inputs", "AA_BayArea_v3.0", "datafiles", "HRKT.csv"))
add("Fig3A_HRKT_surveys", hrkt, "Fig. 3A", "HRKT alignment-array surveys as published (USGS, McFarland et al.)",
    "external_inputs/AA_BayArea_v3.0/datafiles/HRKT.csv", "11_alignment_arrays")
aa = csv("11_alignment_arrays/aa_insar_stacks.csv")
add("Fig3A_S11_InSAR_at_arrays", aa, "Fig. 3A, fig. S11B-G",
    "InSAR fault-parallel slip stacked over the crossings within 0.5 km along strike of each array (mm)",
    "11/aa_insar_stacks.csv", "11_alignment_arrays")
add("Fig3AB_S12A_onset_windows", csv("12_rain_seasonality/fig3_onset_windows.csv"),
    "Figs. 3A and 3B, fig. S12A", "Onset windows of the 9 array and 5 InSAR SSEs",
    "12/fig3_onset_windows.csv", "12_rain_seasonality")
rain = csv("12_rain_seasonality/tulepond_daily_precip.csv")
rain["rain_7day_sum_mm"] = rain["prcp_mm"].rolling(7, min_periods=7).sum()
add("Fig3B_S12A_daily_rain", rain, "Fig. 3B, fig. S12A",
    "Daily rainfall, Fremont gaps filled from Newark (prcp_mm); 7-day sum ending on the date",
    "12/tulepond_daily_precip.csv", "12_rain_seasonality")
z3 = npz(os.path.join(OUT, "13_forecast", "fig3_inputs.npz"))
rows = []
for rec, key in (("InSAR", "A_insar"), ("alignment arrays", "B_aa_dense_noLP"), ("combined", "C_merged")):
    for lo, hi in zip(z3[f"interval_shortest_{key}"], z3[f"interval_longest_{key}"]):
        rows.append(dict(record=rec, shortest_possible_interval_yr=lo, longest_possible_interval_yr=hi))
add("Fig3C_intervals", pd.DataFrame(rows), "Fig. 3C", "Possible range of each repeat interval per record",
    "13/fig3_inputs.npz: interval_*", "13_forecast")
cdf = pd.DataFrame({"decimal_year": z3["grid"]})
for i, rec in enumerate(z3["records"].astype(str)):
    cdf[f"cumulative_probability_{rec}"] = z3["bpt_cdf"][i]
add("Fig3D_forecast_cdf", cdf, "Fig. 3D",
    "Cumulative probability of the next onset (BPT posterior predictive, conditioned on no SSE through "
    "24 Oct 2026)", "13/fig3_inputs.npz: grid, bpt_cdf", "13_forecast")

# ---------------------------------------------------------------- Supplementary figures
add("figS2_GNSS_InSAR", pd.read_csv(os.path.join(REL, "GPS", "gps_insar_validation.csv")), "fig. S2A, B, D, E",
    "GNSS and InSAR line-of-sight velocities at the stations, before and after the ramp",
    "GPS/gps_insar_validation.csv", "data_prep/run05, 01_prep_figures")
acq = csv("03_creep_timeseries/acquisition_times.csv")
fp = pd.read_csv(os.path.join(OUT, "03_creep_timeseries", "fp_ts.csv"), index_col=0)
grid = np.array(fp.columns, dtype=float)
flags = pd.DataFrame({"decimal_year": grid})
for track, sub in acq.groupby("track"):
    obs = np.round(sub["decimal_year"].to_numpy(), 4)
    flags[f"{track}_acquired"] = np.isin(np.round(grid, 4), obs)
add("figS3_acquired_vs_interp", flags, "fig. S3",
    "Common 439-date grid: True where the track acquired that date, False where the value is interpolated",
    "03/acquisition_times.csv, 03/fp_ts.csv", "03_creep_timeseries (recomputed here)")
add("figS3_acquisition_times", acq, "fig. S3", "UTC time of every acquisition per track and frame",
    "03/acquisition_times.csv", "03_creep_timeseries")
for name, rel, what in (("figS4AC_LOS_ascending", "creep_ts_LOS_ascending.csv", "ascending LOS"),
                        ("figS4BD_LOS_descending", "creep_ts_LOS_descending.csv", "descending LOS"),
                        ("figS4F_fault_parallel", "fp_ts.csv", "fault-parallel")):
    add(name, pd.read_csv(os.path.join(OUT, "03_creep_timeseries", rel)), f"fig. S4 ({what})",
        f"Cross-fault {what} displacement, east minus west (mm); rows = crossing latitude, columns = decimal year. "
        "C, D are these with a whole-record linear trend removed per crossing",
        f"03/{rel}", "03_creep_timeseries")
for name, rel in (("figS4E_geometry_ascending", "creep_geom_ascending.csv"),
                  ("figS4E_geometry_descending", "creep_geom_descending.csv")):
    add(name, csv(f"03_creep_timeseries/{rel}"), "fig. S4E",
        "Mean incidence and azimuth per crossing; with the two LOS sheets and the strike these give the vertical",
        f"03/{rel}", "03_creep_timeseries")
add("figS5BF_velocity_profiles", csv("02_velocity_decomposition/fp_velocity_xfault_swath200m_dist1000m.csv"),
    "fig. S5B-F", "Fault-parallel velocity against distance from the trace at each creepmeter (mm/yr)",
    "02/fp_velocity_xfault_swath200m_dist1000m.csv", "02_velocity_decomposition")
add("figS5_station_agreement", csv("05_validation/station_agreement.csv"), "fig. S5G-K",
    "Creepmeter and InSAR rates over the common window, per station", "05/station_agreement.csv", "05_validation")
add("figS8_semivariogram", csv("07_model_data/semivariogram_bins_5km.csv"), "fig. S8",
    "Binned semivariance and fitted model (mm^2)", "07/semivariogram_bins_5km.csv", "07_model_data")
add("figS8_noise_model", csv("07_model_data/noise_model_5km.csv"), "fig. S8", "Fitted noise model parameters",
    "07/noise_model_5km.csv", "07_model_data")
for name, rel, panel in (("figS9_dip_sweep", "dip_rows.csv", "fig. S9 (dip)"),
                         ("figS9_sigma_lambda_sweep", "sensitivity_rows.csv", "fig. S9 (sigma, lambda)"),
                         ("figS10_depth_scan", "depth_scan.csv", "fig. S10"),
                         ("figS10_depth_scan_noise", "depth_scan_noise.csv", "fig. S10"),
                         ("figS10_basement_depth", "basement_depth.csv", "fig. S10 (gray band)")):
    add(name, pd.read_csv(os.path.join(REL, "sensitivity", rel)), panel, f"Sensitivity results: {rel}",
        f"sensitivity/{rel}", "10_model_sensitivity")
add("figS12A_water_year_totals", csv("12_rain_seasonality/wateryear_totals.csv"), "fig. S12A",
    "Water-year rainfall totals and completeness", "12/wateryear_totals.csv", "12_rain_seasonality")
add("figS12A_rain_thresholds", csv("12_rain_seasonality/precip_conditioning_thresholds.csv"), "fig. S12A",
    "Cumulative-rain threshold test: lags of the onsets after each threshold",
    "12/precip_conditioning_thresholds.csv", "12_rain_seasonality")
add("figS12B_storm_rule_p_values", csv("12_rain_seasonality/rain_total_sweep.csv"), "fig. S12B",
    "Storm rule sweep: Z (mm), X (days), hits, season-preserving null mean and P value",
    "12/rain_total_sweep.csv", "12_rain_seasonality")

# ---------------------------------------------------------------- recomputed with the v1.0.0 code (notebook 04)
os.makedirs(SANDBOX, exist_ok=True)
nb = json.load(open(os.path.join(REPO, "04_event_detection.ipynb")))
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell_with(marker):
    """The one code cell of notebook 04 that contains `marker`."""
    found = [s for s in code_cells if marker in s]
    assert len(found) == 1, f"notebook 04: {len(found)} code cells contain {marker!r}"
    return found[0]


ns = {}
os.chdir(REPO)
exec(cell_with("import insar_utils as utils"), ns)
assert os.path.realpath(ns["common_paths"]["out_dir"]) == os.path.realpath(OUT), ns["common_paths"]["out_dir"]


class _NoRegistry:
    def __init__(self, *a, **k): pass
    def __getattr__(self, _): return lambda *a, **k: None


ns["utils"].Registry = _NoRegistry
ns["common_paths"] = dict(ns["common_paths"], out_dir=SANDBOX, fig_dir=SANDBOX)
exec(cell_with('notebook = "04_event_detection"'), ns)
assert os.path.realpath(ns["out_dir"]).startswith(os.path.realpath(SANDBOX)), ns["out_dir"]
exec(cell_with("def load_stack("), ns)
exec(cell_with("def fit_joint("), ns)
events = csv("04_event_detection/creep_events.csv")
for title, key, label, *_ in ns["SEGS"]:
    d, stack, n = ns["load_stack"](ns["segments"][key])
    resid, _ = ns["detrend_rate_annual"](d, stack)
    det, sigma, sd, sv = ns["detect"](d, resid)
    ev = events[events.segment.str.lower().str.contains(label.lower())]
    assert len(ev) > 0, f"no events for {label}"
    # notebook 04: the step column is (dates > step_epoch) and the fitted step epoch is onset_lo
    coef, _ = ns["fit_joint"](d, stack, list(ev.onset_lo.values))
    G = ns["utils"].joint_design(d, list(ev.onset_lo.values))
    assert np.allclose(coef[4:], ev.amplitude_mm.values, atol=0.006), (label, coef[4:], ev.amplitude_mm.values)
    seg = "TulePond" if label == "tulepond" else "Oakland"
    if seg == "Oakland":
        cm_d, cm_s = ns["utils"].daily_creepmeter(ns["common_paths"]["creepmeter_dir"], "COZ1", ns["t0"], ns["t1"],
                                                  ns["creepmeter_glitches"])
        add("figS6B_COZ1_creepmeter_daily", pd.DataFrame({"decimal_year": cm_d, "creepmeter_slip_mm": cm_s}),
            "fig. S6B", "COZ1 creepmeter, daily means with glitch days removed (plotted offset by +5 mm)",
            "external_inputs/creepmeter_data (USGS, Langbein et al. 2024), read with utils.daily_creepmeter",
            "04_event_detection")
    add(f"figS6_{seg}_stack", pd.DataFrame({"decimal_year": d, "mean_fault_parallel_mm": stack,
                                             "joint_model_mm": G @ coef}),
        "fig. S6A" if seg == "TulePond" else "fig. S6B",
        f"{title}: mean of {n} crossings within 2 km of the section centre and the joint model "
        "(linear, annual, one step per detected event)", "03/fp_ts.csv (recomputed)", "04_event_detection cells 2-8")
    add(f"figS6_{seg}_statistic", pd.DataFrame({"gap_midpoint_decimal_year": sd, "step_statistic_mm": sv,
                                                 "threshold_mm": 3.0 * sigma}),
        "fig. S6C" if seg == "TulePond" else "fig. S6D",
        f"{title}: mean of the next 3 minus the previous 3 residuals; threshold = 3 robust sigma "
        f"({3.0 * sigma:.2f} mm)", "03/fp_ts.csv (recomputed)", "04_event_detection cells 2-8")
add("figS6_detected_events", events, "fig. S6", "Detected events, onset windows and amplitudes per stack",
    "04/creep_events.csv", "04_event_detection")
add("figS6B_COZ1_creep_events", csv("04_event_detection/creepmeter_events_COZ1.csv"), "fig. S6B",
    "Creep events detected in the COZ1 creepmeter record", "04/creepmeter_events_COZ1.csv", "04_event_detection")
fp_km = ns["km"]
cm = pd.DataFrame({"code": z1["cm_codes"].astype(str), "km": z1["cm_km"]})
per = pd.DataFrame({"decimal_year": grid})
for code, kmc in zip(cm.code, cm.km):
    rows_near = np.abs(fp_km - kmc) <= 0.2 + 1e-6
    assert rows_near.sum() > 0, code
    block = fp.values[rows_near]
    per[f"{code}_InSAR_mean_mm"] = np.nanmean(block, axis=0)
    per[f"{code}_InSAR_sd_mm"] = np.nanstd(block, axis=0)
    per[f"{code}_n_crossings"] = np.isfinite(block).sum(axis=0)
add("figS5GP_InSAR_at_creepmeters", per, "fig. S5G-P",
    "InSAR fault-parallel series: mean and SD of the crossings within 200 m of each creepmeter "
    "(creepmeter records: USGS, Langbein et al. 2024, in the Zenodo deposit inputs_external.zip)",
    "03/fp_ts.csv (recomputed)", "05_validation")

# ---------------------------------------------------------------- write
CSV_DIR = os.path.join(OUT_ROOT, "Data_S1")
os.makedirs(CSV_DIR, exist_ok=False)          # fail loud rather than mix with an older build
rd = pd.DataFrame(readme)
rd.insert(0, "file", rd.pop("sheet") + ".csv")
rd.to_csv(os.path.join(CSV_DIR, "README.csv"), index=False)
total = 0
for name, df in sheets.items():
    path = os.path.join(CSV_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    total += os.path.getsize(path)
print(f"wrote {len(sheets)} csv files + README.csv to {CSV_DIR}: {total / 1e6:.1f} MB")
