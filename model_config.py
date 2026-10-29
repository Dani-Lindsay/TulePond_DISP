#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026 (the slip-model rewire, code_audit_2026-10-28/REWIRE_BRIEF.md)

@author: daniellelindsay

model_config.py: THE parameter block of the Tule Pond slip models. Every modelling constant is declared here ONCE,
with its unit, its value and the row of the choices register (CHOICES_REGISTER_2026-10-25.md, "#n") or the reference
that justifies it. Imported by 07a_uniform_model, 08a_3d_model, 08b_sensitivity and scripts/sensitivity/*.py; no
notebook re-types any of these numbers, and no notebook reads another notebook's source.

Where things are (folders, data root) stays in TulePond_filepaths.py; how the maths is done stays in model_utils.py.
The two section limits (the SSE section and the 90 % band) are NOT typed here: they are read from notebook 04's
section_lengths.csv by sse_section_km() and band90_km().
"""
import os

import numpy as np
import pandas as pd

from TulePond_filepaths import data_dir, paths_out, paths_inputs, paths_sfcvm, paths_uniform

# =====================================================================================================================
# Data (register #1-#5)
# =====================================================================================================================
fault_trace_lat_range = (37.47, 37.65)  # deg N: the mapped trace kept for the models (the mesh region, km ~52-67)
swath_km = 4.0                # km: pixels within this distance of the trace are modelled (#3)
exclude_km = 0.03             # km: the pixel straddling the trace is dropped before the quadtree (the crossings' inner edge, 04)
km_window = (52.0, 67.0)      # km from Point Pinole: the along-strike data window (#3)
start_block_km = 2.0          # km: first quadtree block (#4; Wei 2009, Vavra & Fialko 2026)
min_block_km = 0.03           # km: smallest quadtree block = one 30 m DISP-S1 pixel at the trace (#4)
near_trace_cut_km = 0.06      # km: samples closer to the trace get no weight (#2: the DISP-S1 phase-linking half-window)
decomposition_window_px = 20  # px: MintPy asc_desc2horz_vert window of 06 (07a rebuilds the coherence-0.7 field with it)

# Noise model (#5): exponential + nugget semivariogram of the pixels beyond noise_mask_km of the trace, plane removed.
# The fit needs pixels beyond 5 km, so it is made on the whole 06 map (a 15 km swath), not on the 4 km model swath.
noise_mask_km = 5.0           # km: pixels farther than this from the trace enter the fit (#5; Vavra & Fialko 2026)
noise_swath_km = 15.0         # km: the swath of the fit (= everything 06's map holds)
noise_distmax_km = 8.0        # km: largest pixel-pair distance in the semivariogram
noise_npairs = 300000         # random pixel pairs of the semivariogram
noise_seed = 0                # their random seed

# =====================================================================================================================
# Weighting and inversion (register #6, #7, #14-#17)
# =====================================================================================================================
weighting = "nugget"          # data covariance: diagonal, nugget variance only (#6, LOCKED: "the data were equally weighted")
nugget_power = 0.0            # every sample gets nugget^2 / n_pixels**0 = nugget^2 (N^0: equal weights, #6)
ramp_spec = "1,x,y"           # CSI polynomial of the reference frame: a plane solved jointly with the slip (#7)
lam_xz_km = (0.1, 0.05)       # km: CSI exponential-prior lengths along strike / down dip (#15); CSI uses exp(-0.5 d / lambda),
#                               so the effective lengths are 0.2 / 0.1 km (Radiguet 2011)
use_surface = False           # no surface constraint: the crossings come from the same pixels (#17)
# slip >= 0 (right-lateral only, #14) is CSI's NNLS in model_utils.invert_tents

# Regularisation strength (#16): each model's own L-curve corner (Hansen 1992), not snapped to a grid value.
# Step 1: the corner on the base grid (the grid of the 2026-10-27 sweeps). Step 2: the base grid plus nine sigmas around
# that corner (corner x 2^(k/16), k = -4..4); the corner of the union is the model's sigma.
sigma_grid_base = [4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 24.0, 32.0, 40.0, 48.0,
                   64.0, 96.0, 128.0]
sigma_refine_factors = [2.0 ** (k / 16.0) for k in range(-4, 5)]

# =====================================================================================================================
# Forward models (register #8-#13)
# =====================================================================================================================
gf_run = "gf_shallow_sfcvm_order2"   # the selected PyLith 5.0.2 GreensFns run (#8 order 2, #9 shallow mesh, #10 SFSVM no cap,
#                                      #11 aligned, #12 vertical); its G is read from the archive, PyLith is never run here
gf_pylith_output_dir = paths_sfcvm["pylith_output"]
gf_gf_dir = paths_sfcvm["gf"]
sfcvm_grid_npz = os.path.join(paths_sfcvm["spatialdb"], "sfcvm_v26-0_aligned_nocap_grid.npz")   # the grid PyLith read (#10, #11)
gf_files = dict(                                     # every PyLith file of the selected run (08a's "PyLith files" cell)
    mesh=os.path.join(paths_inputs["mesh"], "mesh_vertical_tet_shallow.msh"),
    mesh_report=os.path.join(paths_inputs["mesh"], "mesh_vertical_tet_shallow_report.txt"),
    cfg=["pylith/cfg/pylithapp.cfg", "pylith/cfg/mat_sfcvm_nocap.cfg", "pylith/cfg/gf.cfg", "pylith/cfg/order2.cfg"],
    material_db=os.path.join(paths_sfcvm["spatialdb"], "mat_sfcvm_v26-0_aligned_nocap.spatialdb"),
    material_grid=sfcvm_grid_npz,
    impulse_db=os.path.join(paths_inputs["spatialdb"], "impulses_km54-65_z3000.spatialdb"),
    points=os.path.join(paths_inputs["points"], "insar_points.txt"),
    command=("bash $REPO/pylith/run_gf.sh gf_shallow_sfcvm_order2 mat_sfcvm_nocap.cfg modeling_inputs/mesh/mesh_vertical_tet_shallow.msh "
             "modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb 5 modeling_inputs/points/insar_points.txt order2.cfg"),   # run from the data root
    queue_script="run_production_2026-10-25.sh",   # the queue that ran it (data root), job line 1
    log=os.path.join(paths_sfcvm["runs"], f"{gf_run}.log"),
    done_marker=os.path.join(os.path.dirname(paths_sfcvm["root"]), "runs_done_2026-10-25", gf_run),
    outputs=[os.path.join(paths_sfcvm["pylith_output"], f"{gf_run}-{part}") for part in
             ("points.h5", "fault.h5", "elastic_info.h5", "parameters.json", "progress.txt")],
    G=os.path.join(paths_sfcvm["gf"], f"G_{gf_run}.npz"),
)
# Homogeneous comparison (#13): CSI triangular dislocations (Meade 2007) on the SAME tents (model_utils.analytic_tent_G),
# each fault triangle split into n_sub^2 constant-slip sub-triangles. n_sub is chosen in 07a: 8 if n_sub 4 and 8 differ
# by more than analytic_nsub_tolerance on analytic_nsub_check_tents tents, else 4.
analytic_nsub_options = (4, 8)
analytic_nsub_check_tents = 20
analytic_nsub_tolerance = 0.01
analytic_G_dir = paths_uniform["gf"]   # cache: G_analytic_<gf_run>_nsub<n>.npz

# =====================================================================================================================
# Outputs and metrics (register #18-#22)
# =====================================================================================================================
mu_ref_pa = 30.0e9            # Pa: reference rigidity of Mw and of the stress drop (#18)
near_field_km = 0.5           # km: RMS within 500 m west / east of the trace, in the SSE section (metric policy 2026-10-28)
depth_fractions = (0.5, 0.9)  # the depths above which 50 % and 90 % of the potency lie (#22)
fig2k_inner_km = 0.06         # km: Fig 2K step, data and model, 60-150 m from the trace (#21)
fig2k_outer_km = 0.15
profile_far_m = 1500.0        # m: start of the far band of the across-fault deflections (08a profiles; 07a/08a [14] before)

# Monte Carlo uncertainty (#20): correlated noise drawn from the full 5 km covariance (nugget / N^0.5, unscaled), added
# to the model prediction, re-inverted at the fixed corner sigma with the locked weighting.
mc_n_realisations = 100
mc_seed = 11                  # the seed of 07a's bootstrap and of depth_scan_noise.py
mc_noise_nugget_power = 0.5   # the noise draw's covariance: model_utils.covariance_matrix(..., nug_power 0.5)

# Fixed-depth scan (#19 stress drop, #22): uniform slip from the surface to D over the SSE section, tapered at each end.
depth_scan_taper_km = 0.5
depth_scan_depths_km = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25,
                        1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


# =====================================================================================================================
# Archived sweep results (made by scripts/sensitivity/*.py; read, never re-made, by 08a's checks and 08b's figures)
# =====================================================================================================================
sensitivity_dir = os.path.join(data_dir, "sensitivity_2026-10-30")   # where scripts/sensitivity/*.py write (re-run 2026-10-30)
archived = dict(
    sensitivity_nodip=os.path.join(data_dir, "sensitivity_2026-10-27", "sensitivity_rows_nodip.csv"),
    sensitivity_dip=os.path.join(data_dir, "sensitivity_2026-10-27", "sensitivity_rows_dip.csv"),
    depth_scan=os.path.join(data_dir, "depth_scan_2026-10-28", "depth_scan.csv"),
    depth_scan_noise=os.path.join(data_dir, "depth_scan_2026-10-28", "depth_scan_noise.csv"),
    depth_scan_buried_top=os.path.join(data_dir, "depth_scan_2026-10-28", "depth_scan_buried_top.csv"),
)


# =====================================================================================================================
# Section limits, read from notebook 04 (never typed)
# =====================================================================================================================
def _section_row(name):
    table = pd.read_csv(os.path.join(paths_out["04_event_detection"], "section_lengths.csv")).set_index("name")
    assert name in table.index, f"04's section_lengths.csv has no row {name!r}"
    km_lo, km_hi = float(table.loc[name, "km_lo"]), float(table.loc[name, "km_hi"])
    assert km_window[0] < km_lo < km_hi < km_window[1], f"{name}: km {km_lo}-{km_hi} outside the data window {km_window}"
    return km_lo, km_hi


def sse_section_km():
    """(km_lo, km_hi) from Point Pinole of the Tule Pond SSE section: 04's `tulepond_creep_length` row."""
    return _section_row("tulepond_creep_length")


def band90_km():
    """(km_lo, km_hi) of the 90 % peak-slip band: 04's `tulepond_band90` row."""
    return _section_row("tulepond_band90")


def summary():
    """The locked settings as printable lines (every notebook prints them in its first cell)."""
    lo, hi = sse_section_km()
    b_lo, b_hi = band90_km()
    return [f"forward model {gf_run} (PyLith 5.0.2, order 2, shallow mesh, SFSVM v26.0 aligned, no Vs floor, no cap)",
            f"data: swath {swath_km:g} km, km {km_window[0]:g}-{km_window[1]:g}, quadtree {start_block_km:g} -> {min_block_km * 1e3:.0f} m; "
            f"near-trace cut {near_trace_cut_km * 1e3:.0f} m",
            f"noise: semivariogram beyond {noise_mask_km:g} km on a {noise_swath_km:g} km swath, pairs < {noise_distmax_km:g} km",
            f"weighting: diagonal, nugget^2 / N^{nugget_power:g} (equal weights); reference frame '{ramp_spec}' solved jointly",
            f"prior: CSI exponential, lambda {lam_xz_km} km (effective {2 * lam_xz_km[0]:g} / {2 * lam_xz_km[1]:g} km); slip >= 0; "
            f"surface constraint {'ON' if use_surface else 'OFF'}",
            f"sigma: own L-curve corner on {len(sigma_grid_base)} base values + {len(sigma_refine_factors)} around the corner",
            f"SSE section km {lo:.2f}-{hi:.2f}; 90 % band km {b_lo:.2f}-{b_hi:.2f} (04 section_lengths.csv); "
            f"near field +/-{near_field_km * 1e3:.0f} m; mu_ref {mu_ref_pa / 1e9:g} GPa"]


assert np.all(np.diff(sigma_grid_base) > 0) and len(sigma_refine_factors) == 9
assert near_trace_cut_km >= exclude_km and fig2k_inner_km >= near_trace_cut_km
