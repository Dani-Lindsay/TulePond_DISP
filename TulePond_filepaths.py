#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 2026

@author: daniellelindsay

Central file paths and shared constants for the TulePond_DISP publication
repository ("Predictable Surface Slow Slip Events on the Hayward Fault").

Every notebook imports from here, so the whole repo runs after editing ONE
line: point `data_dir` at the folder holding the Zenodo download (the MintPy
products written by the OPERA DISP-S1 preparation chain, see notebook 01).

Layout follows NC_ALOS-2: this file owns WHERE things are, insar_utils.py
owns HOW the shared maths is done, and the numbered notebooks own WHAT is
done in what order.
"""
import os

# The repo itself: outputs (products, stats sidecars, the number registry) and
# figures are written NEXT TO THE NOTEBOOKS, not into data_dir, so that a
# clean checkout plus the Zenodo data reproduces everything in one place.
repo_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------
# Data root -- one repo, two laptops: TULEPOND_DATA_DIR if set, else the
# proj_TulePond_DISP folder where it exists, else the repo itself (the second
# laptop keeps the inputs inside the repo). No per-machine edits.
#
# proj_TulePond_DISP (2026-09-30) holds ONLY what this repo reads, in the
# layout Zenodo will carry:
#   09156/ 09157/ 11115/   the DISP-S1 frames (data, stack, tropo, mintpy) from data_prep/run01-04
#   mosaic_velfield/       the tied same-orbit velocity + geometry mosaics (run04 step 6, run05)
#   external_inputs/       fault trace, arrays, creepmeters, GHCN, catalogues, qfaults, USGS SFCVM v26
#   GPS/                   the screened UNR series and the run05 tie table
#   prep_validation/       run05's GNSS-tie record and run20's rain tables
#   aperture_sweep/        the frozen aperture sweep notebook 04 loads (Hayward_DISP-S1/analysis/run11)
#   modeling_uniform/      products of 07a/07b that are data-sized (Green's functions, sweeps)
#   modeling_SFCVM/        PyLith: cfg, mesh, spatialdb, G matrices, inversion products (08a/08b)
# Everything else of the older proj_HaywardCreep working tree stays there, untouched.
# ------------------------
data_dir = os.environ.get("TULEPOND_DATA_DIR", "")
if not data_dir:
    data_dir = "/Users/dlindsay/work/proj_TulePond_DISP"
    if not os.path.isdir(data_dir):
        data_dir = repo_dir

# Figures go into a DATED directory -- bump this by hand at the start of a new
# round of figure work. Figure filenames carry their parameters, so the date is
# the review snapshot, not the identity of a figure (NC_ALOS-2 convention).
fig_date   = "2026-10-06"   # full update on the 2026-10-05 fixes (audit, along-trace km, doy, burst seams, sigma-12 sensitivity); 2026-10-03_final = clean sweep of every notebook in order (Dani 2026-10-03); 2026-10-03 = onset-window fix + publication style pass; 2026-10-02 = manuscript-pass figure notes; 2026-09-30 = PyLith + SFCVM model set
# Figures and notebook products live with the data, not in the repo (Dani 2026-10-01): the repo is the code
# base archived on GitHub plus the parameter files needed to regenerate everything.
fig_dir    = os.path.join(data_dir, f"Figures_{fig_date}")
out_dir    = os.path.join(data_dir, "outputs")
inputs_dir = os.path.join(data_dir, "external_inputs")
gps_dir    = os.path.join(data_dir, "GPS")
model_uniform_dir = os.path.join(data_dir, "modeling_uniform")
model_sfcvm_dir   = os.path.join(data_dir, "modeling_SFCVM")
# The PyLith side of the 3D model (notebooks 08a/08b): run ONCE (33 min per Green's-function run,
# PyLith 5.0.2), its products are data-sized and live in modeling_SFCVM; the parameter files the
# runs were made with are carried by the repo in scripts/pylith_cfg/ (helpers in scripts/pylith_helpers/).
paths_sfcvm = {
    "root":          model_sfcvm_dir,
    "cfg":           os.path.join(repo_dir, "scripts", "pylith_cfg"),      # PyLith .cfg files (mesh, materials, GF, forward runs)
    "mesh":          os.path.join(model_sfcvm_dir, "mesh"),               # gmsh meshes (.msh, *_fault.npz, *_report.txt), trace_tmerc.txt
    "spatialdb":     os.path.join(model_sfcvm_dir, "spatialdb"),          # SFCVM grids (.npz) and spatialdata databases
    "points":        os.path.join(model_sfcvm_dir, "points"),             # PyLith output points = 07a's quadtree samples
    "gf":            os.path.join(model_sfcvm_dir, "gf"),                 # G matrices; the draft run's inversion products (08b's sensitivity table)
    "pylith_output": os.path.join(model_sfcvm_dir, "pylith_output"),      # <run>-points.h5, -fault.h5, -elastic_info.h5, -parameters.json
}
# QC products of the data_prep chain (station residuals, the prep-side rain
# tables). Deliberately NOT under a second `outputs/`: the only `outputs/` in
# this project is the repo's, written by the notebooks. Nothing in notebooks
# 01-11 reads these; they are the prep chain's own record.
prep_dir   = os.path.join(data_dir, "prep_validation")

# ------------------------
# Common paths & constants
# ------------------------
common_paths = {
    "fig_dir":    fig_dir,
    "out_dir":    out_dir,
    "inputs_dir": inputs_dir,
    "gps_dir":    gps_dir,

    # The number registry: ONE append-only csv that every notebook adds its
    # manuscript numbers to, plus a rendered text view for copying into the
    # manuscript. See insar_utils.Registry.
    "registry_csv": os.path.join(out_dir, "registry.csv"),
    "registry_txt": os.path.join(out_dir, "registry.txt"),

    # InSAR reference: every displacement/velocity product is referenced at the
    # pixel co-located with GNSS station ZOA1 (Fremont, west of the fault), so
    # InSAR and GNSS share a datum before the plate-frame tie.
    "ref_station": "ZOA1",
    "ref_lat":  37.5431,
    "ref_lon": -122.0159,

    # Observation window: first and last DISP-S1 acquisition used in the paper.
    "date_start": "2016-07-29",
    "date_end":   "2026-03-27",

    # Re-mapped fault trace (single continuous strand, variable local strike;
    # from USGS Quaternary faults + steps visible in the velocity fields) and
    # the 100 m crossings derived from it (791 points, Point Pinole -> 37.41 N).
    "fault_trace":    os.path.join(inputs_dir, "hayward_fault_trace_2026-07-24.kml"),
    "fault_profiles": os.path.join(inputs_dir, "hayward_fault_profiles.csv"),
    "qfaults_dir":    os.path.join(inputs_dir, "qfaults"),

    # USGS creepmeters: instrument locations + the .10min records (the .10min.gz
    # copies are the refreshed ones; insar_utils.creepmeter_file prefers them).
    "creepmeter_locations": os.path.join(inputs_dir, "creepmeter_locations.csv"),
    "creepmeter_dir":       os.path.join(inputs_dir, "creepmeter_data"),
    # Gittins et al. 2026, Seismica 5(2), doi:10.26443/seismica.v5i2.2818: the
    # digital catalogue (v1.0.0) of the aseismic slip events the creepmeters
    # recorded, with trigger information; ends December 2023 (notebook 05).
    "gittins_catalogue": os.path.join(inputs_dir, "Gittins2026_catalogue",
                                      "Digital_catalogue_of_5000_aseismic_slip_events_recorded_by_creepmeters_with_trigger_info.csv"),

    # USGS alignment arrays, Bay Area archive v3.0 (McFarland / Lienkaemper):
    # one csv per array under datafiles/, plus the summary table.
    "aa_dir":     os.path.join(inputs_dir, "AA_BayArea_v3.0", "datafiles"),
    "aa_summary": os.path.join(inputs_dir, "AA_BayArea_v3.0", "summary_data.csv"),

    # Published creep-rate compilation (Johnson et al. 2022, SRL-2022186
    # supplement): long-term alignment-array rates for the slip budget.
    "creep_rates_Johnson22": os.path.join(
        inputs_dir, "SRL-2022186_Supplement_FINAL_2023_CREEP_Dataset.csv"),

    # NOAA GHCN-daily rain gauges near Tule Pond (downloaded once; the csv
    # files in ghcn_dir are the frozen inputs of notebook 09).
    "ghcn_dir": os.path.join(inputs_dir, "GHCN_daily"),
    # the ACWD indicator wells (levels) and their positions (public registers; see the file header)
    "acwd_dir": os.path.join(inputs_dir, "ACWD_water_well_data"),
    "well_locations": os.path.join(inputs_dir, "ACWD_water_well_data", "well_locations.csv"),

    # Seismicity / repeating-earthquake catalogues.
    # T. Taira's HFRC linkage-cluster catalogue, GEN3 (1984 to end-2025), the
    # most reliable coherence threshold (coh >= 0.97). Whitespace columns:
    # YYYY.dec Lat Lon Depth cumD(cm) EVID CSID Mag D(cm).
    "CRE_Taira": os.path.join(inputs_dir, "CRE_cat", "CRE_Taira26_GEN3",
                              "HFRC.freq8.0-24.0Hz_maxdist5.0_coh9700_linkage_cluster.txt"),
    # Waldhauser & Schaff (2021) repeating earthquakes (sequence headers + events)
    "CRE_Wald": os.path.join(inputs_dir, "CRE_cat", "CRE_Waldhauser21", "waldhauser_shaff_repeaters.txt"),
    # Area of interest (W, S, E, N) the catalogues are clipped to
    "aoi_bbox": (-122.494468, 37.269677, -121.711070, 38.10),
    # Frame 09156 covers only the northern half of the AOI; opera-utils errors
    # if asked for pixels outside a frame's footprint, so it is clipped to its
    # own (data_prep/run01, run02, run03).
    "aoi_bbox_09156": (-122.495, 37.270, -121.711, 37.719),
    # DOWNLOAD window for 09157 and 11115 (data_prep/run01 only): aoi_bbox
    # nudged 34 m west and 32 m north. opera-utils subsets by row/column, and
    # `_get_rowcol_slice` converts the bbox corners with `lonlat_to_rowcol` on
    # the stack's FIRST product; that conversion moved by one pixel between the
    # 2026-07 download and the 2026-09 refresh, so the bbox that gave a
    # 2333 x 3048 window in July now gives 2332 x 3049. Mixing windows makes
    # disp-s1-reformat fail (run01's grid check catches it). This bbox
    # reproduces the archive window exactly for BOTH frames, so new granules
    # land on the existing grid and run02's geometry stays valid. Each nudge
    # sits mid-way across its one-pixel band, ~15 m from either edge; the
    # lattice is unchanged (x, y = 15 mod 30), so these are the same ground
    # pixels. Kept separate from aoi_bbox so the science AOI (catalogue
    # clipping in notebook 04, the GNSS ROI in run00, the TROPO crop in run03,
    # which pads 0.3 deg anyway) is untouched by a download-side fix.
    "aoi_bbox_disp": (-122.494858, 37.269967, -121.711070, 38.10),
    # Bo Xie's GrowClust relocated catalogue for the Hayward area (2000-2025),
    # relocated PLUS unrelocated NCSS events, one row per event
    "growclust_cat": os.path.join(inputs_dir, "Bo_growclust", "hayward_growclust_catalog_relocated_plus_unrelocated.csv"),

    # Frozen outputs of the aperture sweep made by Hayward_DISP-S1/analysis/
    # run11b_sweep_ladder.py (July/August 2026). Notebook 05 READS these; it
    # does not re-run the sweep (Dani, 2026-09-21).
    "sweep_dir": os.path.join(data_dir, "aperture_sweep"),

    # Written by the data_prep chain, read by nobody in notebooks 01-11:
    # run05's per-station GNSS/InSAR residuals, and run20's prep-side rain
    # tables (notebook 09 builds its own composite from `ghcn_dir`).
    "validation_dir": os.path.join(prep_dir, "validation"),
    "precip_dir":     os.path.join(prep_dir, "precip"),
}

# ------------------------
# Analysis constants -- the choices a reviewer could ask about.
# Each has a one-line reason; assumptions_register.md carries the full audit.
# ------------------------
unit = 1000            # 1000 = displacement products are in m, we work in mm

# Cross-fault aperture for the creep time series (the "narrow" stream). The
# bin was chosen on creepmeter agreement in the aperture sweep (notebook 05).
dist_min = 0.03        # km, inner edge: bins touching the trace under-read the step
dist_max = 0.15        # km, outer edge: beyond ~150 m the step is diluted by off-fault strain
x_width  = 0.20        # km, half width ALONG strike of each crossing window
min_points = 5         # minimum valid pixels per side (a 60 m bin at 30 m pixels)
coh_timeseries = 0.7   # temporal-coherence mask for the creep time series (ONE stream everywhere, Dani 2026-09-21:
#                        0.7 keeps 609 crossings against 480 at 0.9, finds the same five Tule Pond events, and costs
#                        0.1 mm/yr on the creepmeter rate RMS; the frozen sweep shows the threshold is second order)
coh_velocity   = 0.7   # temporal-coherence mask for velocity maps and the GNSS tie
rate_nsigma = 3.0      # near-fault rate mask: reject pixels > 3 robust sigma from their side's median
az_smooth_n = 21       # crossings in the rolling median of the local strike (2.1 km)
max_interp_gap_days = 45   # interpolate asc/desc onto common dates only across gaps shorter than this
rate_smooth_km = 0.5   # Gaussian smoothing of the along-fault rate profile
profile_smooth_km = 0.2    # Gaussian sigma of the drawn along-strike slip profiles: Fig 1C per event, Fig 1D rates, Fig 2K mean event (Dani 2026-10-03: 0.2 km)

# Along-strike parameter string that goes into every product and figure name
# (NC_ALOS-2 convention: the filename tells you what was run).
params = f"ap{int(dist_min*1000)}-{int(dist_max*1000)}m_coh{coh_timeseries}_mp{min_points}_sw{int(x_width*1000)}m"

# The Tule Pond section (Fremont; between Alameda Creek and Lake Elizabeth) and
# the Oakland COZ1 section used ONLY as the detector's positive control.
segments = {
    "TULEPOND": {"label": "Tule Pond",      "lat": 37.5570, "lat_lo": 37.50, "lat_hi": 37.60},
    "COZ1":     {"label": "Oakland (COZ1)", "lat": 37.7500, "lat_lo": 37.68, "lat_hi": 37.82},
}
# The two LARGE Oakland (COZ1) creep events the control must find (notebooks 04 and 05): a decimal year inside
# each one's InSAR onset window (the COZ1 creepmeter's Jan 2019 event, catalogued from 2019-01-11, and its
# Feb 2022 surge, from 2022-02-14; Gittins et al. 2026). 04 asserts that each lies in exactly one COZ1 onset
# window (widened by coz1_large_tol_days each side) and 05 that they are the two COZ1 events with the largest
# creepmeter steps. The control may find more: since the along-trace stack (2026-10-05) it also finds the
# Oct 2022 COZ1 surge (Dani 2026-10-05: kept, the third control event), so COZ1 events are labelled by their
# onset day (`onset_day` in creep_events.csv), not by year.
coz1_large_events_yr = (2019.02, 2022.13)
coz1_large_tol_days = 12.0     # one long (12 d) acquisition interval

# Box for the per-pixel event maps (notebook 06) and the slip model (07):
# W, S, E, N in degrees, around the Tule Pond section.
box_lonlat = (-122.10, 37.46, -121.84, 37.65)

# Fig 2 map window (notebook 06), also drawn as the "Tule Pond" outline on the Fig 1
# map (notebook 04): centred on the trace at fig2_map_centre_km from Point Pinole,
# +/- fig2_map_half_km east-west, +/- (fig2_map_half_km + fig2_map_extra_km)
# north-south, as the draft (10.4 x 12 km).
fig2_map_centre_km, fig2_map_half_km, fig2_map_extra_km = 59.5, 5.2, 0.8

# The five USGS creepmeters, north -> south: (code, lat, lon, name).
# Instrument locations from Langbein et al. (2024); single source of truth.
creepmeters = [("CPP1", 37.9895,  -122.3559,  "Pinole Point"),
               ("CTM1", 37.8439,  -122.2273,  "Temescal"),
               ("COZ1", 37.7535,  -122.1505,  "Oakland Zoo"),
               ("CHP1", 37.66274, -122.0740,  "Palisade Street"),
               ("CFW1", 37.53237, -121.95230, "Fremont")]

# Creepmeter glitch days (insar_utils.daily_creepmeter, notebooks 04 and 05), found by inspection
# (Dani 2026-10-05: drop only the listed days, no automatic filter; QC figure qc05_creepmeter_glitches
# from notebook 05). (code, first day, last day, reason). Days are CALENDAR days: the USGS day-of-year is
# 1-based (1 January 00:00 = doy 1.000) and the readers convert it so (insar_utils.creepmeter_decimal_year,
# 2026-10-05). Both days were found and first listed in the old reader convention (doy / days, one day late:
# 2017-05-19 and 2018-02-13) and moved back one day with that fix; the raw samples are at CPP1 doy 138.50 and
# 138.54 of 2017 (18 May 12:00 and 13:00) and doy 43.33 of 2018 (12 Feb 08:00). daily_creepmeter asserts that
# each listed day is in the record. The five records over the InSAR window (2016-09 to 2026-03) were
# screened for days whose mean departs from the median of their 10-minute samples by > 0.2 mm or from a
# 15-day running median by > 0.3 mm, and for offsets across data gaps; only these two are glitches (COZ1's
# two flagged days are creep events inside the day). The whole-record 6-sigma + 10 mm cut it replaces
# dropped no day in that window and did not catch these.
creepmeter_glitches = [
    ("CPP1", "2017-05-18", "2017-05-18", "two 10-min samples at -283.7 mm (~340 mm below the record) pull the daily mean down 4.8 mm"),
    ("CPP1", "2018-02-12", "2018-02-12", "one 10-min sample at -283.7 mm (~350 mm below the record) pulls the daily mean down 2.4 mm"),
]
# Checked and NOT listed: CTM1 around 2025.3-2025.5. Its record is smooth there; the drop-and-recover in the
# cumulative panel of Fig S validation is one InSAR epoch (2025-04-01, ~5.6 mm below its neighbours), as is
# the dip at CHP1 in 2021-10 (2021-10-25, ~7.5 mm). Both are InSAR, not creepmeter (QC figure, last panels).

# Catalogued creepmeter events (Gittins et al. 2026) in notebook 05: used for TIMING only (Dani 2026-10-05:
# the catalogue-amplitude score, InSAR step / catalogued slip of every event >= 2 mm, is retired; the
# amplitude statement uses notebook 04's joint-model COZ1 steps against the creepmeter's).
# The catalogue is not drawn on Fig S validation (Dani 2026-10-05: it goes into a table in the text); 05
# lists every CFW1 catalogue event that overlaps a Tule Pond InSAR onset window widened by this many days
# on each side (outputs/05_validation/cfw1_gittins_near_windows.csv, the table's source).
gittins_window_pad_days = 7

# Shared choices read by more than one notebook (one definition each).
wet_season_months = (10, 4)    # (first month, last month) of the wet season, shared by 10 and 11 (Dani 2026-10-05: Oct-Apr)
paper_band_frac = 0.90         # the peak-slip band the paper quotes (04 band_main, read by 11)
model_sigma_locked = 12.0      # regularisation sigma of the selected slip model and all sensitivity runs (L-curve rule, Dani 2026-10-03)

# Date the forecast is issued from (notebook 09). Pinned so the forecast
# numbers are reproducible; do not use "today".
forecast_issue_date = "2026-08-25"
# The last observation with no event, on which notebook 11 conditions the forecast: the last
# descending acquisition of the CSLC + Dolphin extension check (extension_2026-09/RESULT.md,
# 2026-10-05; ascending ends 2026-09-12). The paper's InSAR record itself still ends at
# notebook 11's data_end (the end of 03's fp_ts, 2026-03-21).
forecast_last_obs_date = "2026-09-24"

# Running rain totals (notebooks 10 and 11). 11 tests a grid of totals and lags before the InSAR onsets
# (Fremont gauge); 10's event table only describes each window by the days since a 3-day total of at least
# rain_total_mm (a description, no test uses that single value).
rain_total_days = 3        # length of the running rain total (days)
rain_total_mm = 40.0       # 10's event-table description: days since a 3-day total of at least this (mm)

# ------------------------
# Per-track paths: one ascending frame (09157, track 35) and one descending
# frame (11115, track 42). Same key vocabulary for both, so notebooks loop
# `for dic in insar_dic_list:` instead of writing everything twice.
# ------------------------
def _track_paths(frame, orbit, path):
    fdir = os.path.join(data_dir, frame)
    mdir = os.path.join(fdir, "mintpy")
    return {
        "ORBIT_DIRECTION": orbit,
        "PATH":     path,            # Sentinel-1 relative orbit number
        "Frame":    frame,           # OPERA DISP-S1 frame id
        "Platform": "Sentinel-1",
        "mintpy_dir":   mdir,
        "geometry_dir": os.path.join(fdir, "geometry"),   # OPERA los_east/los_north tifs (ground truth for the LOS convention test)
        # OPERA CSLC-S1-STATIC files of the bursts covering box_lonlat (notebook 06 reads their
        # 'identification/bounding_polygon' to place the burst seams): 09157 T035-073249-IW2/-IW3 and
        # T035-073250-IW2, 11115 T042-088917-IW3 and T042-088918-IW3, copied 2026-10-05 from
        # proj_HaywardCreep_bad/<frame>/static_lyrs. data_prep/run02 deletes the static files it downloads
        # into this folder after stitching: re-copy them if run02 is rerun.
        "static_dir":   os.path.join(fdir, "static_lyrs"),
        "geo_file":    os.path.join(mdir, "geometryGeo.h5"),      # incidenceAngle, azimuthAngle, height
        "ts_file":      os.path.join(mdir, "timeseries_tropo.h5"),  # tropo-corrected, unmasked, referenced at ZOA1
        "vel_file":     os.path.join(mdir, "velocity_masked.h5"),   # coh 0.7 masked, ZOA1 datum, no GNSS ramp
        "vel_NA_file":  os.path.join(mdir, "velocity_NA.h5"),       # GNSS-tied, North America frame, zero at ZOA1
        # Temporal-coherence masks, keyed by threshold, as data_prep/run04 writes them.
        # 0.7 is coh_timeseries and coh_velocity: the creep time series, the per-pixel
        # step fits, the velocity fields and the GNSS tie. 0.9 is coh_maps: notebook
        # 06's per-pixel event maps, the cross-fault profiles and the decomposed field.
        # (0.8 was dropped 2026-09-28 -- no notebook ever selected it and run04 stopped
        # writing it, so the path resolved to a stale July product.)
        "mask_file":    {c: os.path.join(mdir, f"maskTempCoh_{int(round(c * 10)):02d}.h5") for c in (0.9, 0.7)},
    }

asc09157_dic = _track_paths("09157", "ascending",  "035")
des11115_dic = _track_paths("11115", "descending", "042")
insar_dic_list = [asc09157_dic, des11115_dic]

# ------------------------
# Frame paths for the DATA PREPARATION chain (`data_prep/run00` to `run05`).
#
# A separate view of the same data from `insar_dic_list` above, for two
# reasons: the prep chain works on THREE frames (09156 is merged with 09157
# into the ascending mosaic, and has no analysis view of its own), and it needs
# the raw and intermediate paths -- the downloaded granules, the rebased stack,
# the geometry and troposphere directories -- that the notebooks never read.
# The notebooks use `insar_dic_list`; only `data_prep/` uses this.
# ------------------------
def _prep_paths(frame):
    fdir = os.path.join(data_dir, frame)
    mdir = os.path.join(fdir, "mintpy")
    return {
        "frame":        frame,
        "dir":          fdir,
        "data_dir":     os.path.join(fdir, "data"),          # downloaded + subset granules
        "stack":        os.path.join(fdir, f"{frame}_stack.nc"),  # rebased stack (run01)
        "geometry_dir": os.path.join(fdir, "geometry"),       # LOS, incidence, DEM (run02)
        "static_dir":   os.path.join(fdir, "static_lyrs"),    # CSLC static layers (run02)
        "tropo_dir":    os.path.join(fdir, "tropo"),          # TROPO crops + corrections (run03)
        "mintpy_dir":   mdir,
        "timeseries":       os.path.join(mdir, "timeseries.h5"),
        "timeseries_tropo": os.path.join(mdir, "timeseries_tropo.h5"),
        "velocity":         os.path.join(mdir, "velocity.h5"),
        "velocity_masked":  os.path.join(mdir, "velocity_masked.h5"),
        "velocity_NA":      os.path.join(mdir, "velocity_NA.h5"),
        "geometry":         os.path.join(mdir, "geometryGeo.h5"),
    }

# 09156 ascending track 35 (northern half only), 09157 ascending track 35
# (the full creeping section), 11115 descending track 42.
frames = {f: _prep_paths(f) for f in ("09156", "09157", "11115")}

# Same-orbit velocity mosaics (both ascending frames merged) used for the
# velocity-field decomposition (notebook 02).
mosaic_dir = os.path.join(data_dir, "mosaic_velfield")
paths_mosaic = {
    "ascending":  {"vel_NA_file": os.path.join(mosaic_dir, "ascending_velocity_NA.h5"),
                   "geo_file":    os.path.join(mosaic_dir, "ascending_geometry.h5")},
    "descending": {"vel_NA_file": os.path.join(mosaic_dir, "descending_velocity_NA.h5"),
                   "geo_file":    os.path.join(mosaic_dir, "descending_geometry.h5")},
}

# ------------------------
# GPS paths (UNR / Nevada Geodetic Laboratory), used for provenance in 01 only
# ------------------------
paths_gps = {
    "DataHoldings": os.path.join(gps_dir, "DataHoldings.txt"),
    "Steps":        os.path.join(gps_dir, "steps.txt"),
    "NA_dir":       os.path.join(gps_dir, "NA"),
    "StaList":      os.path.join(gps_dir, "{frame}_StaList.txt"),   # data_prep/run00
    "Fig_Dir":      os.path.join(gps_dir, "Figures"),               # data_prep/run00 station plots
    "enu":          os.path.join(gps_dir, "{frame}_enu_NA.csv"),
    "validation":   os.path.join(gps_dir, "gps_insar_validation.csv"),
}

# ------------------------
# Outputs, keyed by notebook. Each notebook writes only into its own folder.
# ------------------------
paths_out = {nb: os.path.join(out_dir, nb) for nb in (
    "01_prep_inputs", "01_prep_figures", "02_velocity_decomposition", "03_creep_timeseries",
    "04_event_detection", "05_validation", "06_event_field_geometry",
    "07a_uniform_model", "07b_uniform_model_tests", "08a_3d_model", "08b_3d_model_tests", "09_alignment_arrays",
    "10_rain_seasonality", "11_forecast", "12_paper_figures")}
# The pre-2026-09-24 names 09_seasonality_forecast and 10_paper_figures are gone:
# 09 split into 10_rain_seasonality and 11_forecast, and the paper-figure collector
# moved to 11. The superseded notebooks are kept under drafts/originals/.
