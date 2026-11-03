#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

model_config.py: the settings of the Tule Pond slip models. Every modelling constant is declared here once, with its
unit, its value and its source. Imported by 07_model_data, 08_model_homogeneous, 09_model_3d, 10_model_sensitivity and
scripts/sensitivity/*.py; no notebook re-types these numbers.

Paths are in TulePond_filepaths.py; the modelling functions are in model_utils.py. The section limits (the SSE section
and the 90 % band) are read from notebook 04's section_lengths.csv by sse_section_km() and band90_km().
"""
import os

import numpy as np
import pandas as pd

from TulePond_filepaths import data_dir, paths_out, paths_inputs, paths_sfcvm, paths_uniform

# =====================================================================================================================
# Data
# =====================================================================================================================
fault_trace_lat_range = (37.47, 37.65)  # deg N: the mapped trace kept for the models (the mesh region, km ~52-67)
swath_km = 4.0                # km: pixels within this distance of the trace are modelled
exclude_km = 0.03             # km: the pixel on the trace is dropped before the quadtree (the inner edge of 04's crossings)
km_window = (52.0, 67.0)      # km from Point Pinole: the along-strike data window
start_block_km = 2.0          # km: first quadtree block (Wei 2009, Vavra & Fialko 2026)
min_block_km = 0.03           # km: smallest quadtree block = one 30 m DISP-S1 pixel at the trace
near_trace_cut_km = 0.06      # km: samples closer to the trace are not fitted (the DISP-S1 phase-linking half-window)
decomposition_window_px = 20  # px: MintPy asc_desc2horz_vert window of 06 (07 rebuilds the coherence-0.7 field with it)

# Noise model: nugget plus exponential semivariogram of the pixels beyond noise_mask_km of the trace, plane removed.
# The fit needs pixels beyond 5 km, so it is made on the whole 06 map (a 15 km swath), not on the 4 km model swath.
noise_mask_km = 5.0           # km: pixels farther than this from the trace enter the fit (Vavra & Fialko 2026)
noise_swath_km = 15.0         # km: the swath of the fit (= everything 06's map holds)
noise_distmax_km = 8.0        # km: largest pixel-pair distance in the semivariogram
noise_npairs = 300000         # random pixel pairs of the semivariogram
noise_seed = 0                # their random seed

# =====================================================================================================================
# Weighting and inversion
# =====================================================================================================================
weighting = "nugget"          # data covariance: diagonal, nugget variance only (equal weights)
nugget_power = 0.0            # every sample gets nugget^2 / n_pixels**0 = nugget^2
ramp_spec = "1,x,y"           # CSI polynomial of the reference frame: a plane solved jointly with the slip
lam_xz_km = (0.1, 0.05)       # km: CSI exponential-prior lengths along strike / down dip; CSI uses exp(-0.5 d / lambda),
#                               so the effective lengths are 0.2 / 0.1 km (Radiguet 2011)
# no surface constraint: the crossing offsets come from the same pixels as the field
# slip >= 0 (right-lateral only) is CSI's NNLS in model_utils.invert_tents

# Regularisation strength: each model's own L-curve corner (maximum curvature, Hansen 1992).
# Step 1: the corner on the base grid. Step 2: the base grid plus nine sigmas around that corner (corner x 2^(k/16),
# k = -4..4); the corner of the combined set is the model's sigma.
sigma_grid_base = [4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 24.0, 32.0, 40.0, 48.0,
                   64.0, 96.0, 128.0]
sigma_refine_factors = [2.0 ** (k / 16.0) for k in range(-4, 5)]

# =====================================================================================================================
# Forward models
# =====================================================================================================================
# The preferred model: 3D SFSVM elastic structure, fault dipping 70 deg west, fine near-surface mesh.
# Fault dip: 70.5 +/- 2 deg SW where the CFW1 creepmeter borehole crosses the fault at ~21 m depth (south end of the SSE
# section, km 62.6; Bilham & Whitehead 1997, doi 10.1029/97GL01244); ~70 deg E at 100-650 m depth on the Alameda Creek
# seismic line (north end of the section; Williams et al. 2005, doi 10.1029/2005GL023113). Vertical, 70 E, 60 W and 50 W
# faults on the same mesh are sensitivity tests (scripts/sensitivity/sensitivity_rows.py).
gf_run = "gf_shallow_dip70w_sfcvm_order2"   # PyLith 5.0.2 GreensFns run: order 2, shallow mesh, SFSVM (no cap, aligned on the
#                                             trace), 70 W fault; its G is read from the archive, PyLith is not run here
selected_dip_label = "70 W"                 # the dip of gf_run as the sensitivity rows and figures name it
selected_dip_deg = 70.0                     # deg: the dip of gf_run (mesh report: measured 70.0 deg, dipping west)
selected_dip_tag = "dip" + selected_dip_label.replace(" ", "").lower()     # "dip70w": in the file names of the model figures
gf_run_vertical = "gf_shallow_sfcvm_order2"  # the vertical fault on the same fine mesh (sensitivity test)
gf_pylith_output_dir = paths_sfcvm["pylith_output"]
gf_gf_dir = paths_sfcvm["gf"]
sfcvm_grid_npz = os.path.join(paths_sfcvm["spatialdb"], "sfcvm_v26-0_aligned_nocap_grid.npz")   # the grid PyLith read
gf_files = dict(                                     # every PyLith file of the selected run (notebook 09, section 2)
    mesh=os.path.join(paths_inputs["mesh"], "mesh_dip70w_tet_shallow.msh"),
    mesh_report=os.path.join(paths_inputs["mesh"], "mesh_dip70w_tet_shallow_report.txt"),
    mesh_log=os.path.join(paths_inputs["mesh"], "gen_dip70w_shallow_2026-10-26.log"),
    cfg=["pylith/cfg/pylithapp.cfg", "pylith/cfg/mat_sfcvm_nocap.cfg", "pylith/cfg/gf.cfg", "pylith/cfg/order2.cfg"],
    material_db=os.path.join(paths_sfcvm["spatialdb"], "mat_sfcvm_v26-0_aligned_nocap.spatialdb"),
    material_grid=sfcvm_grid_npz,
    impulse_db=os.path.join(paths_inputs["spatialdb"], "impulses_km54-65_z3000.spatialdb"),
    points=os.path.join(paths_inputs["points"], "insar_points.txt"),
    command=("bash $REPO/pylith/run_gf.sh gf_shallow_dip70w_sfcvm_order2 mat_sfcvm_nocap.cfg modeling_inputs/mesh/mesh_dip70w_tet_shallow.msh "
             "modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb 5 modeling_inputs/points/insar_points.txt order2.cfg"),   # run from the data root
    queue_script="run_shallow_dips_2026-10-26.sh",   # the queue script that ran it (data root)
    log=os.path.join(paths_sfcvm["runs"], f"{gf_run}.log"),
    done_marker=os.path.join(os.path.dirname(paths_sfcvm["root"]), "pylith_runs_done", gf_run),
    outputs=[os.path.join(paths_sfcvm["pylith_output"], f"{gf_run}-{part}") for part in
             ("points.h5", "fault.h5", "elastic_info.h5", "parameters.json", "progress.txt")],
    G=os.path.join(paths_sfcvm["gf"], f"G_{gf_run}.npz"),
)
# Homogeneous comparison: CSI triangular dislocations (Meade 2007) on the same tents (model_utils.analytic_tent_G),
# each fault triangle split into n_sub^2 constant-slip sub-triangles. n_sub is chosen in 08: 8 if n_sub 4 and 8 differ
# by more than analytic_nsub_tolerance on analytic_nsub_check_tents tents, else 4.
analytic_nsub_options = (4, 8)
analytic_nsub_check_tents = 20
analytic_nsub_tolerance = 0.01
analytic_G_dir = paths_uniform["gf"]   # cache: G_analytic_<gf_run>_nsub<n>.npz
analytic_n_workers = 3                 # parallel processes when the cache is missing (1 BLAS thread each)

# =====================================================================================================================
# Outputs and metrics
# =====================================================================================================================
mu_ref_pa = 30.0e9            # Pa: reference rigidity of Mw and of the stress drop
near_field_km = 0.5           # km: RMS within 500 m west / east of the trace, in the SSE section
depth_fractions = (0.5, 0.9)  # the depths above which 50 % and 90 % of the potency lie
fig2k_inner_km = 0.06         # km: Fig 2K step, data and model, 60-150 m from the trace
fig2k_outer_km = 0.15
# Near-fault deflection (Methods): in the SSE section, the mean over fitted samples near_deflection_km from the trace minus
# the mean over samples far_deflection_km away, on each side.
near_deflection_km = (0.06, 0.25)     # km: from the 60 m cut to 250 m
far_deflection_km = (1.5, 3.0)        # km: the far band (11-14 samples per side)
far_deflection_check_km = (1.0, 3.0)  # km: a wider far band, as a check

# Monte Carlo uncertainty: correlated noise drawn from the full covariance of the noise model (nugget / N^0.5), added to
# the modelled values and re-inverted at the fixed corner sigma with equal weights.
mc_n_realisations = 100
mc_seed = 11                  # the seed of the Monte Carlo noise draws (09, scripts/sensitivity/depth_scan.py)
mc_noise_nugget_power = 0.5   # the noise draw's covariance: model_utils.covariance_matrix(..., nug_power 0.5)

# Fixed-depth scan (stress drop, depth extent): uniform slip from the surface to D over the SSE section, tapered at each end.
depth_scan_taper_km = 0.5
depth_scan_depths_km = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25,
                        1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


# =====================================================================================================================
# Archived sweep results (made by scripts/sensitivity/*.py; read, never re-made, by 09's checks and 10's figures)
# =====================================================================================================================
sensitivity_dir = os.path.join(data_dir, "sensitivity")   # where scripts/sensitivity/*.py write
archived = dict(
    # scripts/sensitivity/sensitivity_rows.py: its baseline row is 09's independent reproduction check of the model
    sensitivity_rows=os.path.join(sensitivity_dir, "sensitivity_rows.csv"),
    depth_scan=os.path.join(sensitivity_dir, "depth_scan.csv"),          # scripts/sensitivity/depth_scan.py on gf_run
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
    """The settings as printable lines (each model notebook prints them in its first cell)."""
    lo, hi = sse_section_km()
    b_lo, b_hi = band90_km()
    return [f"forward model {gf_run} (PyLith 5.0.2, order 2, shallow mesh, fault dipping 70 deg W, SFSVM v26.0 aligned, no Vs floor, no cap)",
            f"data: swath {swath_km:g} km, km {km_window[0]:g}-{km_window[1]:g}, quadtree {start_block_km:g} -> {min_block_km * 1e3:.0f} m; "
            f"near-trace cut {near_trace_cut_km * 1e3:.0f} m",
            f"noise: semivariogram beyond {noise_mask_km:g} km on a {noise_swath_km:g} km swath, pairs < {noise_distmax_km:g} km",
            f"weighting: diagonal, nugget^2 / N^{nugget_power:g} (equal weights); reference frame '{ramp_spec}' solved jointly",
            f"prior: CSI exponential, lambda {lam_xz_km} km (effective {2 * lam_xz_km[0]:g} / {2 * lam_xz_km[1]:g} km); slip >= 0; "
            "surface constraint OFF",
            f"sigma: own L-curve corner on {len(sigma_grid_base)} base values + {len(sigma_refine_factors)} around the corner",
            f"SSE section km {lo:.2f}-{hi:.2f}; 90 % band km {b_lo:.2f}-{b_hi:.2f} (04 section_lengths.csv); "
            f"near field +/-{near_field_km * 1e3:.0f} m; mu_ref {mu_ref_pa / 1e9:g} GPa"]


assert np.all(np.diff(sigma_grid_base) > 0) and len(sigma_refine_factors) == 9
assert near_trace_cut_km >= exclude_km and fig2k_inner_km >= near_trace_cut_km
