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

# The repo itself: the code only. Every product, figure and model run is written into data_dir (the Zenodo
# results directory), never inside the repo: asserted at the end of this file (`assert_outside_repo`) and by
# tests/test_filepaths.py (2026-10-06).
repo_dir = os.path.dirname(os.path.abspath(__file__))

# ------------------------
# Data root -- one repo, two laptops: TULEPOND_DATA_DIR if set, else the
# proj_TulePond_DISP folder where it exists, else the repo itself (the second
# laptop keeps the inputs inside the repo; since 2026-10-06 assert_outside_repo
# stops there: set TULEPOND_DATA_DIR to a data folder outside the repo). No per-machine edits.
#
# proj_TulePond_DISP (2026-09-30) holds ONLY what this repo reads, in the
# layout Zenodo will carry:
#   09156/ 09157/ 11115/   the DISP-S1 frames (data, stack, tropo, mintpy) from data_prep/run01-04
#   mosaic_velfield/       the tied same-orbit velocity + geometry mosaics (run04 step 6, run05)
#   external_inputs/       fault trace, arrays, creepmeters, GHCN, catalogues, qfaults, USGS SFCVM v26
#   GPS/                   the screened UNR series and the run05 tie table
#   prep_validation/       run05's GNSS-tie record and run20's rain tables
#   aperture_sweep/        the frozen aperture sweep notebook 04 loads (Hayward_DISP-S1/analysis/run11)
#   modeling_inputs/       the slip models' shared inputs: PyLith meshes, the sample point sets (07a), impulse and test-slip dbs
#   modeling_uniform/      the HOMOGENEOUS (30 GPa) PyLith runs: G matrices, run outputs, slip dbs, the CSI validation products
#   modeling_SFCVM/        the USGS SFCVM PyLith runs: material grids and dbs (aligned, +/- 50 m, dip, no-cap), G matrices, run outputs
#   outputs/<notebook>/    every notebook's own products (the selected models, the sensitivity tables, the registry)
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
fig_date   = "2026-10-28"   # 2026-10-28 = the slip-model rewire (07a, 08a, 08b, 07c on the locked method, code_audit_2026-10-28); 2026-10-25 = 3D model without the Vs floor, dipping faults in the paper SFCVM grid, homogeneous dip scan (08c 08a 08b 07c 12); 2026-10-14 = full chain 01 -> 12 (no 07b) on the relocated burst seam and with no far-field plane; 2026-10-08 = burst-seam relocation, 06 only (seam located from the data, 06 [14]); 2026-10-06 = full update on the 2026-10-05 fixes (audit, along-trace km, doy, burst seams, sigma-12 sensitivity); 2026-10-03_final = clean sweep of every notebook in order (Dani 2026-10-03); 2026-10-03 = onset-window fix + publication style pass; 2026-10-02 = manuscript-pass figure notes; 2026-09-30 = PyLith + SFCVM model set
# Figures and notebook products live with the data, not in the repo (Dani 2026-10-01): the repo is the code
# base archived on GitHub plus the parameter files needed to regenerate everything.
fig_dir    = os.path.join(data_dir, f"Figures_{fig_date}")
out_dir    = os.path.join(data_dir, "outputs")
executed_dir = os.path.join(data_dir, "executed")   # run_all.sh: the HTML copy of every executed notebook (until 2026-10-06 in the repo)
inputs_dir = os.path.join(data_dir, "external_inputs")
gps_dir    = os.path.join(data_dir, "GPS")
# ------------------------
# The slip models (notebooks 07a, 08a, 08b, 08c) and their PyLith runs (pylith/, README there). Three folders,
# separated by what they hold (Dani 2026-10-06):
#   modeling_inputs/   shared by both media: the gmsh meshes and fault geometry, the quadtree sample sets written by
#                      07a (the PyLith output points), the impulse and test-slip spatial databases
#   modeling_uniform/  the homogeneous medium (mu 30 GPa, nu 0.25): its PyLith runs, G matrices, slip databases of
#                      its forward runs, and the CSI-vs-PyLith validation products (08a)
#   modeling_SFCVM/    the USGS SFCVM medium: its material grids and databases (aligned, +/- 50 m, dipping, no Poisson
#                      cap), its PyLith runs, G matrices and slip databases
# The PyLith runs are made from data_dir (pylith/run_gf.sh, run_fwd.sh); the cfg files in pylith/cfg give every path
# relative to it, and each material cfg names its own medium folder as the output directory.
# ------------------------
model_inputs_dir  = os.path.join(data_dir, "modeling_inputs")
model_uniform_dir = os.path.join(data_dir, "modeling_uniform")
model_sfcvm_dir   = os.path.join(data_dir, "modeling_SFCVM")
pylith_dir = os.path.join(repo_dir, "pylith")             # cfg files, run wrappers, mesh and spatial-database build scripts
paths_pylith = {
    "root":  pylith_dir,
    "cfg":   os.path.join(pylith_dir, "cfg"),             # PyLith .cfg files (common, materials, GF, forward runs, order 2)
    "build": os.path.join(pylith_dir, "build"),           # mesh, SFCVM grid and spatial-database writers
    "workdir": data_dir,                                  # every PyLith run is made from here (the cfg paths are relative to it)
}
paths_inputs = {
    "root":      model_inputs_dir,
    "mesh":      os.path.join(model_inputs_dir, "mesh"),       # gmsh meshes (.msh, *_fault.npz, *_report.txt), trace_tmerc.txt
    "points":    os.path.join(model_inputs_dir, "points"),     # the sample sets (07a): insar_points[_swath7|_coh07].{txt,npz}; fig2_points_<label>.txt
    "spatialdb": os.path.join(model_inputs_dir, "spatialdb"),  # impulse region and validation-slip databases (medium independent)
}


def _medium_paths(root):
    return {
        "root":          root,
        "gf":            os.path.join(root, "gf"),             # G_<run>.npz: the fault-parallel G matrices (model_utils.build_G)
        "pylith_output": os.path.join(root, "pylith_output"),  # <run>-points.h5, -fault.h5, -elastic_info.h5, -parameters.json
        "spatialdb":     os.path.join(root, "spatialdb"),      # material grids / databases (SFCVM) and the forward runs' slip databases
        "runs":          os.path.join(root, "runs"),           # PyLith run logs (run_gf.log: start/done times)
    }


paths_uniform = _medium_paths(model_uniform_dir)
paths_uniform["validation"] = os.path.join(model_uniform_dir, "validation")   # 08a: PyLith homogeneous vs CSI Meade, 2D screw
paths_sfcvm = _medium_paths(model_sfcvm_dir)


def medium_paths(medium):
    """The folder set of one elastic medium: `homog` -> modeling_uniform, any SFCVM variant (`sfcvm`, `sfcvm_e50`,
    `sfcvmasis`, ...) -> modeling_SFCVM."""
    if medium == "homog":
        return paths_uniform
    assert medium.startswith("sfcvm"), f"unknown medium {medium!r}"
    return paths_sfcvm


# Constants of the slip models shared by 07a, 08a, 08b and 08c (one definition each).
model_crs = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"   # PyLith mesh CRS (m)
model_frame_lonlat = (-121.968, 37.557)   # CSI local frame centre (lon0, lat0) = the mesh CRS origin (the creep-section centre)
# QC products of the data_prep chain (station residuals, the prep-side rain
# tables). Deliberately NOT under a second `outputs/`: the only `outputs/` in
# this project is the repo's, written by the notebooks. Nothing in notebooks
# 01-11 reads these; they are the prep chain's own record.
prep_dir   = os.path.join(data_dir, "prep_validation")

# ------------------------
# Notebooks retired from the paper: still in the repo (and runnable), but their registry keys are rendered in
# the RETIRED block of registry.txt (insar_utils.Registry.render), never as live numbers. registry.csv keeps
# their rows (append-only).
# 07b_uniform_model_tests: the uniform-model CSI side checks (depth scan, synthetic recovery, ...) are no
# longer part of the paper (Dani 2026-10-05: the model is PyLith only) and 07b was not re-run after
# 2026-10-03, so its 07b_* keys would otherwise render as live. 07a_uniform_model stays (the data-prep
# step of 08a and the CSI validation): its keys stay live.
# ------------------------
# 2026-10-30 (the rewire): 07b, 08b_3d_model_tests and 08c moved to drafts/retired_2026-10-28/ with the old 07a, 07c and
# 08a; their keys retire because their notebooks are no longer in the repo (Registry.render). The new 07a / 08a / 08b_sensitivity
# register fresh keys.
retired_notebooks = ("07b_uniform_model_tests", "08b_3d_model_tests", "08c_3d_model_sensitivity")

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
    "retired_notebooks": retired_notebooks,     # their keys render as RETIRED (see above)

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
# The slip models' constants (sigma rule, near-trace cut, surface constraint, weighting, ...) live in model_config.py since
# 2026-10-30 (one parameter block; model_sigma_locked, model_sigma_grid, model_near_trace_cut_km, model_use_surface removed).
# The bimonthly era of the alignment arrays (09, 10, 11): a survey interval of at most this many days resolves the
# season of an onset; 09 runs the HRKT detector only on such intervals, 10 and 11 assert that every array window is
# this short (until 2026-10-06 typed in each notebook: 09 dense_window_days, 10 and 11 resolvable_days).
aa_dense_window_days = 130
# The 1989 Loma Prieta earthquake, 17 Oct 1989 local time, as a decimal year to two places (day of year 290:
# 1989 + 289/365 = 1989.79, the convention of insar_utils.date_to_decimal_year), for the interval that spans it (09 asserts that the longest quiet HRKT
# interval spans it; 10 saves it for Fig 3A; 11 forms the array forecasts with and without that interval).
loma_prieta_yr = 1989.79

# Lake Elizabeth (Central Park, Fremont), east of the trace at the south end of the Tule Pond section: the area of
# the mean-event VERTICAL signal the text quotes (notebook 06). Centre = the centroid of the open-water no-data hole
# of 06's coherence-0.9 maps (lon -121.9614, lat 37.5499; read off fp_maps.npz 2026-10-06; TBC by the author on 06's
# QC figure qc06_lake_elizabeth_vertical_area, which draws the area on the map); the area is every valid pixel
# within lake_elizabeth_radius_m of it and east of the trace beyond 06's near-fault band.
lake_elizabeth_lonlat = (-121.9614, 37.5499)
lake_elizabeth_radius_m = 1000.0   # Dani 2026-10-14: the uplift covers ~1 km around the lake (500 m until then)
# Where the trace crosses Alameda Creek and runs along Lake Elizabeth (Dani 2026-10-16, read off
# hayward_fault_profiles.csv in Google Earth: (ft_lon, ft_lat) of the north and south ends); drawn as blue lines in
# Fig 2K. The lake is also a park, where the InSAR may be noisier.
alameda_creek_trace_lonlat = ((-121.986, 37.5751), (-121.982, 37.5687))
lake_elizabeth_trace_lonlat = ((-121.969, 37.5497), (-121.963, 37.5433))   # south end extended (Dani 2026-10-16; was -121.964, 37.5447)

# Date the forecast statement was first written (notebook 11; notebook 10 ends its rain series on it). Pinned so the
# forecast numbers are reproducible; do not use "today".
forecast_issue_date = "2026-08-25"
# The last observation with no event, on which notebook 11 conditions the forecast: the last
# descending acquisition of the CSLC + Dolphin extension check (extension_2026-09/RESULT.md,
# 2026-10-05; ascending ends 2026-09-12). The paper's InSAR record itself still ends at
# notebook 11's data_end (the end of 03's fp_ts, 2026-03-21).
forecast_last_obs_date = "2026-09-24"

# Running rain totals (notebook 10). 10 tests a grid of totals and lags before the InSAR onsets (Fremont gauge;
# the test moved from 11 on 2026-10-06); its event table only describes each window by the days since a 3-day total
# of at least rain_total_mm (a description, no test uses that single value).
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
    "07a_uniform_model", "07b_uniform_model_tests", "08a_3d_model", "08b_3d_model_tests", "08c_3d_model_sensitivity",
    "08b_sensitivity", "07c_fig2",
    "09_alignment_arrays",
    "10_rain_seasonality", "11_forecast", "12_paper_figures")}
# The pre-2026-09-24 names 09_seasonality_forecast and 10_paper_figures are gone:
# 09 split into 10_rain_seasonality and 11_forecast, and the paper-figure collector
# moved to 11. The superseded notebooks are kept under drafts/originals/.


# ------------------------
# Nothing is written inside the repo (Dani 2026-10-06): every folder a notebook, a model run or a figure writes into
# must resolve OUTSIDE repo_dir. On a machine whose data sit inside the repo (the data_dir fallback above), set
# TULEPOND_DATA_DIR to a folder outside it.
# ------------------------
def output_locations():
    """Every folder products, figures or model runs are written into, as (name, path) pairs."""
    locations = [("fig_dir", fig_dir), ("out_dir", out_dir), ("executed_dir", executed_dir), ("registry_csv", common_paths["registry_csv"]),
                 ("registry_txt", common_paths["registry_txt"])]
    locations += [(f"paths_out[{nb}]", path) for nb, path in paths_out.items()]
    for label, group in (("paths_inputs", paths_inputs), ("paths_uniform", paths_uniform), ("paths_sfcvm", paths_sfcvm)):
        locations += [(f"{label}[{key}]", path) for key, path in group.items()]
    locations.append(("paths_pylith[workdir]", paths_pylith["workdir"]))
    return locations


def assert_outside_repo():
    """Fail if any output location resolves inside the repo."""
    repo_real = os.path.realpath(repo_dir)
    for name, path in output_locations():
        real = os.path.realpath(path)
        inside = real == repo_real or real.startswith(repo_real + os.sep)
        assert not inside, (f"{name} -> {path} resolves inside the repo ({repo_dir}): products never go into the repo; "
                            f"set TULEPOND_DATA_DIR to a folder outside it")


assert_outside_repo()

