#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 2026 (the model tidy: every modelling function of notebooks 07a, 08a, 08b and 08c, ONCE)

@author: daniellelindsay

HOW the slip models are made. Layout follows NC_ALOS-2: TulePond_filepaths.py owns WHERE things are and the shared
parameters, insar_utils.py the InSAR helpers, this file the modelling maths, and the numbered notebooks WHAT is done
in what order. Every function says what it does, its units and the SOURCE it follows; "written for this project"
where there is none. Nothing here reads a path on its own: every file comes in as an argument.

Contents, in the order the notebooks use them:
  1. Along-trace distance and the fault trace          km_from_pinole, model_trace, signed_distance_m
  2. The retired uniform model's CSI fault (07b)       unrolled_mesh, wrap_fault, triangle_areas_km2
  3. The data: fault-parallel field, noise, quadtree   look_vector, decompose_fault_parallel, fault_parallel_from_steps,
                                                       dist_to_trace, fault_parallel_dataset,
                                                       semivariogram_model, covariance, downsample, covariance_matrix,
                                                       build_cd, quadtree_block_corners, build_sample_set,
                                                       write_points_list, samples_from_arrays
  4. The retired uniform model's inversion (07b)       crossing_profile, crossing_profile_on_triangles, tie_rows,
                                                       invert_triangles
  5. Magnitudes and the L-curve                        moment_magnitude, potency_from_slip, lcurve_corner, lcurve_pick
  6. The surface constraint (sensitivity only)         surface_pct_denominator_mm, arc_length_on_polyline,
                                                       local_scatter_mm, tent_rows, crossing_constraint, surface_slip_data
  7. PyLith Green's functions -> G                     read_gf_run, trace_top, build_G, load_G
  8. The tent (vertex) parameterisation and inversion  TentFault, tent_fault_template, data_cd, invert_tents,
                                                       match_impulse_order, analytic_tent_G, positions_fingerprint,
                                                       analytic_tent_G_cached,
                                                       tent_model_setup, tent_numbers, tent_lcurve, own_corner,
                                                       uniform_slip_shape, uniform_slip_fit, profile_samples,
                                                       correlated_noise, materna_stress_drop_pa
  9. PyLith forward runs, materials, stress            read_forward_points, read_forward_slip, superposition_check,
                                                       shear_modulus_pa, mu_beside_fault, harmonic_mean_rigidity,
                                                       screw_dislocation_mm, median_edge_length_m, vertex_areas_m2,
                                                       slip_weighted_stress_drop_pa, potency_extent
 10. Provenance and running PyLith                     field_fingerprint, run_logged, ensure_product
 11. Like-for-like crossing steps                      crossing_steps_like
The locked settings of the paper's models are in model_config.py (one parameter block).
"""
import copy
import os
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay, cKDTree

import insar_utils as utils

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at commit 1277bb4)
from csi import TriangularPatches, TriangularTents, insar, surfaceslip
from csi import imagedownsampling as imdown
from csi import transformation as transform
from csi import multisourcesolve as multisrc


# =====================================================================================================================
# 1. Along-trace distance and the fault trace
# =====================================================================================================================
def km_from_pinole(lats, profile_lats):
    """km from Point Pinole of each latitude: the km of the nearest vertex of the fault-profile trace
    (hayward_fault_profiles.csv, vertices 0.1 km apart, km = (n - 1 - index) x 0.1; `profile_lats` = its ft_lat
    column). Worked in chunks of 20000 latitudes so the latitude-difference matrix stays small.
    Source: 07a [2] `lat_to_km_pp` (written for this project; the 100 m index rule is tested in tests/test_filepaths.py)."""
    profile_lats = np.asarray(profile_lats, float)
    profile_km = (len(profile_lats) - 1 - np.arange(len(profile_lats))) * 0.1
    lats = np.asarray(lats, float)
    out = np.empty(lats.size)
    for lo in range(0, lats.size, 20000):
        chunk = lats[lo:lo + 20000]
        lat_diff = np.abs(profile_lats[None, :] - chunk[:, None])
        out[lo:lo + 20000] = profile_km[lat_diff.argmin(axis=1)]
    return out


def model_trace(trace_lonlat_all, lat_lo, lat_hi, frame_lonlat, profile_lats, every_km=0.02):
    """The mapped trace of the slip models: the KML trace (lon, lat rows) between lat_lo and lat_hi, sorted north to
    south, discretised every `every_km` by CSI (`TriangularPatches.trace` + `discretize2`), with its arc length and
    km from Point Pinole. Returns a dict: lonlat (n, 2), ref_fault (the CSI object: xi, yi in km of the CSI frame,
    loni, lati), x_km, y_km, arc_s (km), arc_total (km), km_pp (km from Point Pinole of every discretised point).
    Source: 07a [2] (CSI tutorial `tutoFaultMeshing` idiom; CSI, Jolivet et al.)."""
    lonlat = trace_lonlat_all[(trace_lonlat_all[:, 1] >= lat_lo) & (trace_lonlat_all[:, 1] <= lat_hi)]
    lonlat = lonlat[np.argsort(-lonlat[:, 1])]                                  # north -> south
    ref_fault = TriangularPatches("trace frame", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    ref_fault.trace(lonlat[:, 0], lonlat[:, 1])
    ref_fault.discretize2(every=every_km)
    x_km, y_km = ref_fault.xi, ref_fault.yi
    arc_s = np.r_[0, np.cumsum(np.hypot(np.diff(x_km), np.diff(y_km)))]
    return dict(lonlat=lonlat, ref_fault=ref_fault, x_km=x_km, y_km=y_km, arc_s=arc_s, arc_total=arc_s[-1],
                km_pp=km_from_pinole(ref_fault.lati, profile_lats), every_km=every_km)


def signed_distance_m(line_x, line_y, px, py):
    """Signed perpendicular distance (m, east +) of points (px, py) from a densely sampled polyline (line_x, line_y):
    nearest polyline sample, sign from the local tangent. (07a's dist_to_trace is unsigned.)
    Source: written for this project (2026-09-30, the one-page run diagnostic); m in, m out."""
    tangent = np.c_[np.gradient(line_x), np.gradient(line_y)]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    dist, i = cKDTree(np.c_[line_x, line_y]).query(np.c_[px, py])
    normal = np.c_[tangent[i, 1], -tangent[i, 0]]
    normal[normal[:, 0] < 0] *= -1.0                                  # east +
    return np.einsum("ij,ij->i", np.c_[px - line_x[i], py - line_y[i]], normal)


# =====================================================================================================================
# 2. The uniform model's fault: CSI triangles on the mapped trace
# =====================================================================================================================
def unrolled_mesh(h_top, const_to, growth, h_bot, z_max, trace, fine_km, far_rate, smooth_iters):
    """The fault mesh in the unrolled (along-strike s, depth z) plane, km. Node rows from the surface to z_max: the
    first row h_top deep, the row height constant down to const_to, then multiplied by `growth` per row up to h_bot.
    Along strike the node spacing equals the row height inside fine_km (km from Point Pinole) and widens by far_rate
    per km outside it. Delaunay triangles (slivers dropped), then smooth_iters rounds of Laplacian smoothing of the
    interior nodes. trace = model_trace(...). Returns the nodes (s, z) and the faces.
    Source: 07a [2] `build_unrolled` (written for this project; the 07b mesh family chose its parameters)."""
    arc_s, arc_total, trace_km_pp = trace["arc_s"], trace["arc_total"], trace["km_pp"]
    row_depths = [0.0]
    row_step = h_top
    while row_depths[-1] < z_max:
        row_depths.append(min(row_depths[-1] + row_step, z_max))
        if row_depths[-1] >= const_to - 1e-9:
            row_step = min(row_step * growth, h_bot)
    row_depths = np.array(row_depths)
    row_heights = np.r_[np.diff(row_depths), np.diff(row_depths)[-1]]

    def spacing_at(arc_pos, row_height):
        km_pp = np.interp(arc_pos, arc_s, trace_km_pp)
        outside = np.maximum(0, np.maximum(fine_km[0] - km_pp, km_pp - fine_km[1]))
        return row_height + far_rate * outside

    nodes = []
    for depth, height in zip(row_depths, row_heights):
        row = [0.0]
        while row[-1] < arc_total:
            row.append(row[-1] + spacing_at(row[-1], height))
        row = np.array(row) * arc_total / row[-1]
        for pos in row:
            nodes.append((pos, depth))
    nodes = np.array(nodes)

    def triangulate(pts):
        faces = Delaunay(pts).simplices
        area2 = np.abs((pts[faces[:, 1], 0] - pts[faces[:, 0], 0])
                       * (pts[faces[:, 2], 1] - pts[faces[:, 0], 1])
                       - (pts[faces[:, 2], 0] - pts[faces[:, 0], 0])
                       * (pts[faces[:, 1], 1] - pts[faces[:, 0], 1]))
        return faces[0.5 * area2 > 1e-6]

    faces = triangulate(nodes)
    fixed = ((nodes[:, 1] < 1e-9) | (nodes[:, 1] > z_max - 1e-9)
             | (nodes[:, 0] < 1e-9) | (nodes[:, 0] > arc_total - 1e-9))
    for _ in range(smooth_iters):
        neighbours = {}
        for tri in faces:
            for node in tri:
                neighbours.setdefault(node, set()).update(tri)
        smoothed = nodes.copy()
        for node in np.nonzero(~fixed)[0]:
            ring = np.array(sorted(neighbours[node] - {node}))
            smoothed[node] = nodes[ring].mean(0)
        nodes = smoothed
        faces = triangulate(nodes)
    return nodes, faces


def wrap_fault(nodes, faces, dip_deg, name, trace, frame_lonlat):
    """The unrolled mesh wrapped onto the mapped trace as a csi.TriangularPatches fault (km, depth +), hinged east
    about the FIXED surface trace when dip_deg is not 90. Each fault owns its copy of the faces (homogeneizeStrike
    flips winding rows in place). The mesh (nodes, faces, arc length) is kept on the fault as `mesh_sz`.
    Source: 07a [2] `wrap_fault` (CSI tutorial `tutoFaultMeshing` idiom; CSI, Jolivet et al.)."""
    trace_lonlat, arc_s, trace_x, trace_y = trace["lonlat"], trace["arc_s"], trace["x_km"], trace["y_km"]
    flt = TriangularPatches(name, lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    flt.trace(trace_lonlat[:, 0], trace_lonlat[:, 1])
    flt.discretize2(every=trace["every_km"])
    node_x = np.interp(nodes[:, 0], arc_s, trace_x)
    node_y = np.interp(nodes[:, 0], arc_s, trace_y)
    vertices = np.c_[node_x, node_y, nodes[:, 1]]
    if abs(dip_deg - 90.0) > 0.1:
        strike_vec = np.array([trace_x[-1] - trace_x[0], trace_y[-1] - trace_y[0]])
        strike_vec = strike_vec / np.hypot(strike_vec[0], strike_vec[1])
        normal_vec = np.array([strike_vec[1], -strike_vec[0]])
        if normal_vec[0] < 0:
            normal_vec = -normal_vec                   # + east
        east_shift = nodes[:, 1] / np.tan(np.deg2rad(dip_deg))
        vertices[:, 0] += east_shift * normal_vec[0]
        vertices[:, 1] += east_shift * normal_vec[1]
    faces_own = faces.copy()
    flt.Vertices = vertices
    flt.Faces = faces_own
    flt.vertices2ll()
    flt.patch = []
    for face in faces_own:
        flt.patch.append(vertices[face])               # the three corners of the triangle (x, y, z)
    flt.patch2ll()
    flt.setdepth()
    flt.vertical = abs(dip_deg - 90.0) <= 0.1
    flt.homogeneizeStrike(direction=0, sign=-1.0)
    flt.initializeslip()
    flt.mesh_sz = (nodes.copy(), faces_own, trace["arc_total"])
    return flt


def triangle_areas_km2(nodes, faces):
    """Area (km^2) of every triangle of the unrolled (s, z) mesh. Source: 07a [2] (written for this project)."""
    tri_sz = nodes[faces]
    return 0.5 * np.abs((tri_sz[:, 1, 0] - tri_sz[:, 0, 0]) * (tri_sz[:, 2, 1] - tri_sz[:, 0, 1])
                        - (tri_sz[:, 2, 0] - tri_sz[:, 0, 0]) * (tri_sz[:, 1, 1] - tri_sz[:, 0, 1]))


# =====================================================================================================================
# 3. The data: the fault-parallel mean event, its noise model and the quadtree samples
# =====================================================================================================================
def look_vector(strike_deg):
    """Horizontal unit vector (east, north, up) along the strike bearing (degrees clockwise from north): the
    'look vector' of the fault-parallel field, so right-lateral motion of the east side is positive.
    Source: 07a [3] (written for this project)."""
    return np.array([np.sin(np.deg2rad(strike_deg)), np.cos(np.deg2rad(strike_deg)), 0.0])


def decompose_fault_parallel(los_asc, los_desc, inc_both, azi_both, strike_deg, window_px):
    """Fault-parallel and vertical maps (mm) from the ascending and descending LOS maps, with MintPy's
    `asc_desc2horz_vert` (horizontal direction = the strike, horz_az_angle = -strike as MintPy counts anticlockwise;
    only pixels valid on both tracks; decomposition window window_px pixels). inc_both / azi_both: (2, nr, nc),
    ascending first. Returns (fault_parallel, up).
    Source: MintPy asc_desc2horz_vert (Yunjun et al., MintPy), called exactly as notebook 06 [20] `decompose`."""
    from mintpy.asc_desc2horz_vert import asc_desc2horz_vert
    valid_both = np.isfinite(los_asc) & np.isfinite(los_desc)
    los_stack = np.stack([los_asc, los_desc]).astype(np.float32)
    los_stack[:, ~valid_both] = np.nan
    fault_parallel, up = asc_desc2horz_vert(los_stack, inc_both, azi_both, horz_az_angle=-strike_deg, step=window_px)
    return fault_parallel.astype(float), up.astype(float)


def fault_parallel_from_steps(steps_h5, strike_deg, window_px, mask_files=None):
    """The fault-parallel mean event recomputed from notebook 06's saved per-track products
    (`tulepond_steps_<track>.h5`: the per-pixel mean of the kept events at the time-series coherence, with the
    incidence and azimuth of the box), optionally masked at another temporal coherence (mask_files: track -> MintPy
    maskTempCoh file, cut to the box from the X_FIRST / Y_FIRST attributes, as 06 [20] does). steps_h5: track ->
    path, keys 'ascending' and 'descending'. Returns the fault-parallel map (mm) on 06's box grid.
    With the coherence-0.9 masks this must reproduce 06's fp_maps.npz `fp_event` exactly (07a asserts it); without
    masks it is the field at coherence 0.7, the fit mask (the coherence-0.7 sensitivity sample set).
    Source: written for this project (2026-10-06), replacing drafts/retired_2026-10-06/pylith/coh07/make_fp_coh.py, which exec'd 06."""
    means, incs, azis = {}, {}, {}
    for track in ("ascending", "descending"):
        with h5py.File(steps_h5[track], "r") as h5:
            mean = h5["mean_event"][:].astype(float)
            incs[track] = h5["incidenceAngle"][:]
            azis[track] = h5["azimuthAngle"][:]
            box = dict(x_first=float(h5.attrs["X_FIRST"]), y_first=float(h5.attrs["Y_FIRST"]), nr=int(h5.attrs["LENGTH"]),
                       nc=int(h5.attrs["WIDTH"]), dx=float(h5.attrs["X_STEP"]), dy=float(h5.attrs["Y_STEP"]))
        if mask_files is not None:
            with h5py.File(mask_files[track], "r") as mask_h5:
                col_start = int(round((box["x_first"] - float(mask_h5.attrs["X_FIRST"])) / box["dx"]))
                row_start = int(round((box["y_first"] - float(mask_h5.attrs["Y_FIRST"])) / box["dy"]))
                mask_name = [name for name in mask_h5.keys() if name != "date"][0]
                mask = mask_h5[mask_name][row_start:row_start + box["nr"], col_start:col_start + box["nc"]].astype(bool)
            mean = np.where(mask, mean, np.nan)
        means[track] = mean
    inc_both = np.stack([incs["ascending"], incs["descending"]]).astype(np.float32)
    azi_both = np.stack([azis["ascending"], azis["descending"]]).astype(np.float32)
    fault_parallel, _ = decompose_fault_parallel(means["ascending"], means["descending"], inc_both, azi_both,
                                                 strike_deg, window_px)
    return fault_parallel


def dist_to_trace(sar, fault):
    """Horizontal distance (km) from each sample of a CSI data set `sar` to the nearest point of the fault's
    discretised trace (fault.xf, fault.yf). Source: 07a [3], ported verbatim from csi_io (Hayward_DISP-S1)."""
    trace_tree = cKDTree(np.c_[fault.xf, fault.yf])
    dist, _ = trace_tree.query(np.c_[sar.x, sar.y])
    return dist


def fault_parallel_dataset(fp_event, lon, lat, strike_deg, fault, frame_lonlat, swath_km, exclude_km, km_window,
                           profile_lats):
    """The fault-parallel field (mm) as one csi.insar data set whose look vector is the horizontal unit vector along
    the strike bearing: every finite pixel of fp_event, cut to swath_km of the trace (and beyond exclude_km of it) and
    to the along-strike km_window (km from Point Pinole). Nothing is removed from the field: the inversion's constant
    is its only datum (a far-field plane was removed until 2026-10-08; it changed Mw by < 0.01 and is gone, Dani).
    Returns (sar, look).
    Source: 07a [3] (CSI insar.read_from_binary; written for this project)."""
    look = look_vector(strike_deg)
    ok = np.isfinite(fp_event)
    sar = insar("fault-parallel", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    sar.read_from_binary(fp_event[ok], lon[ok], lat[ok], err=np.ones(ok.sum()),
                         los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
    trace_dist = dist_to_trace(sar, fault)
    sar.vel[(trace_dist > swath_km) | (trace_dist < exclude_km)] = np.nan
    km_pp = km_from_pinole(sar.lat, profile_lats)
    sar.vel[(km_pp < km_window[0]) | (km_pp > km_window[1])] = np.nan
    sar.checkNaNs()
    return sar, look


def semivariogram_model(distance_km, nugget2, sill_short, range_short_km, sill_long, range_long_km):
    """Semivariance (mm^2) at a distance h (km): a nugget plus a short- and a long-range exponential,
    gamma(h) = n2 + s1 (1 - exp(-h / l1)) + s2 (1 - exp(-h / l2)).
    Source: 07a [4], ported from the m02 chain (Hayward_DISP-S1); far-field semivariogram noise model after
    Lohman and Simons (2005)."""
    short_term = sill_short * (1 - np.exp(-distance_km / range_short_km))
    long_term = sill_long * (1 - np.exp(-distance_km / range_long_km))
    return nugget2 + short_term + long_term


def covariance(sar, fault, swath_km, cov_mask_km, cov_distmax_km, npairs=300000, seed=0, return_bins=False):
    """Fit semivariogram_model to the far field: the pixels beyond cov_mask_km of the trace, a plane (1, x, y) removed;
    npairs random pixel pairs closer than cov_distmax_km, binned every 0.1 km (bins with more than 50 pairs). Returns
    (nugget2, sill_short, range_short_km, sill_long, range_long_km), in mm^2 and km; with return_bins also the binned
    semivariogram as a table (bin centre km, pairs, gamma mm^2, fitted model mm^2) and the number of pixels fitted.
    `sar` must hold pixels well beyond the mask: cov_mask_km <= 0.4 swath_km and cov_distmax_km <= 4 swath_km are
    ASSERTED (until 2026-10-30 the mask was silently capped at 0.4 swath_km, so a 5 km mask on the 4 km model swath
    became 1.6 km; the locked 5 km fit is made on a 15 km swath, model_config.noise_swath_km).
    Source: 07a [4], ported from the m02 chain (Hayward_DISP-S1; @ -> np.dot)."""
    assert cov_mask_km <= 0.4 * swath_km, (f"noise mask {cov_mask_km} km > 0.4 x the {swath_km} km swath: too few far-field pixels; "
                                           f"build the field on a wider swath")
    assert cov_distmax_km <= 4 * swath_km, f"pair distance {cov_distmax_km} km > 4 x the {swath_km} km swath"
    dist = dist_to_trace(sar, fault)
    keep = (dist > cov_mask_km) & np.isfinite(sar.vel)
    assert keep.sum() > 1000, f"only {keep.sum()} pixels beyond {cov_mask_km} km of the trace"
    x, y, values = sar.x[keep], sar.y[keep], sar.vel[keep]
    design = np.c_[np.ones_like(x), x, y]
    values = values - np.dot(design, np.linalg.lstsq(design, values, rcond=None)[0])
    rng = np.random.default_rng(seed)
    first = rng.integers(0, values.size, npairs)
    second = rng.integers(0, values.size, npairs)
    pair_dist = np.hypot(x[first] - x[second], y[first] - y[second])
    pair_diff = values[first] - values[second]
    max_dist_km = cov_distmax_km
    in_range = (pair_dist > 0) & (pair_dist < max_dist_km)
    bins = np.arange(0, max_dist_km + 0.1, 0.1)
    bin_index = np.digitize(pair_dist[in_range], bins) - 1
    diff_in_range = pair_diff[in_range]
    gamma = np.full(len(bins) - 1, np.nan)
    n_pairs = np.zeros(len(bins) - 1, int)
    for k in range(len(bins) - 1):
        in_bin = bin_index == k
        n_pairs[k] = int(in_bin.sum())
        if in_bin.sum() > 50:
            gamma[k] = 0.5 * np.mean(diff_in_range[in_bin] ** 2)
    bin_centres = 0.5 * (bins[1:] + bins[:-1])
    fitted = np.isfinite(gamma)
    sill = np.nanmean(gamma[fitted][-5:])
    params, _ = curve_fit(semivariogram_model, bin_centres[fitted], gamma[fitted],
                          p0=(0.4 * sill, 0.3 * sill, 0.7, 0.5 * sill, 5.0),
                          bounds=([0, 0, 0.1, 0, 2.0], [2 * sill, 2 * sill, 2.0, 3 * sill, 50]), maxfev=20000)
    if not return_bins:
        return params
    bins_table = pd.DataFrame(dict(distance_km=bin_centres, n_pairs=n_pairs, gamma_mm2=gamma,
                                   model_mm2=semivariogram_model(bin_centres, *params)))
    return params, bins_table, int(keep.sum())


def downsample(sar, fault, start_km, min_km):
    """Distance-based quadtree (csi.imagedownsampling): start_km blocks cut down to min_km toward the trace; a regular
    grid when start_km == min_km. Returns the downsampled csi.insar (its `wgt` = pixels per block, `xycorner` the
    block corners). Source: 07a [4], ported verbatim from the m02 chain; CSI (Jolivet et al.)."""
    sampler = imdown(sar.name, sar, fault, verbose=False)
    sampler.initialstate(start_km, min_km, tolerance=0.1, plot=False)
    if start_km > min_km + 1e-9:
        sampler.distanceBased(chardist=0.0, expodist=1, plot=False)
    return sampler.newimage


def covariance_matrix(x_km, y_km, n_pixels, cov_p, nug_power):
    """The data covariance of quadtree samples (mm^2): Cd_ij = s1 exp(-r_ij / l1) + s2 exp(-r_ij / l2), plus
    nugget2 / n_pixels**nug_power on the diagonal (n_pixels = the pixels each block averages, CSI's `wgt`).
    cov_p = (nugget2, sill_short, range_short_km, sill_long, range_long_km).
    Source: 07a [4] `build_cd`, ported from the m02 chain; estimating Cd from the far-field semivariogram and carrying
    the correlated part follows Lohman and Simons (2005) and Sudhaus and Jonsson (2009)."""
    nugget2, sill_short, range_short_km, sill_long, range_long_km = cov_p
    dx = x_km[:, None] - x_km[None, :]
    dy = y_km[:, None] - y_km[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    cd = sill_short * np.exp(-dist / range_short_km) + sill_long * np.exp(-dist / range_long_km)
    n_px = np.asarray(n_pixels, dtype=float)
    assert n_px.shape == (x_km.size,) and np.all(np.isfinite(n_px) & (n_px > 0)), \
        "the quadtree block pixel counts (wgt) are missing or not positive"
    cd[np.diag_indices_from(cd)] += nugget2 / n_px ** nug_power
    return cd


def build_cd(ds, cov_p, nug_power):
    """Set ds.Cd (mm^2) on a csi data set from its positions and block pixel counts (covariance_matrix); returns ds.
    Source: 07a [4] `build_cd` (m02 chain)."""
    ds.Cd = covariance_matrix(ds.x, ds.y, ds.wgt, cov_p, nug_power)
    return ds


def quadtree_block_corners(ds):
    """The quadtree blocks as polygons (lon, lat of the four corners, (n_samples, 4) each), so maps can fill each block
    at its true size. CSI keeps two opposite corners of every block in its local km frame (`xycorner`, one row per
    sample, in sample order); the four corners go back to lon/lat with the image's xy2ll. Asserts that > 99 % of the
    samples fall inside their own block. Source: 07a [13] (written for this project)."""
    block_corners_xy = np.asarray(ds.xycorner, dtype=float)
    assert block_corners_xy.shape == (ds.vel.size, 4), f"xycorner {block_corners_xy.shape} does not match {ds.vel.size} samples"
    corner_x = np.c_[block_corners_xy[:, 0], block_corners_xy[:, 2], block_corners_xy[:, 2], block_corners_xy[:, 0]]
    corner_y = np.c_[block_corners_xy[:, 1], block_corners_xy[:, 1], block_corners_xy[:, 3], block_corners_xy[:, 3]]
    block_lon, block_lat = ds.xy2ll(corner_x.ravel(), corner_y.ravel())
    block_lon = np.reshape(block_lon, corner_x.shape)
    block_lat = np.reshape(block_lat, corner_x.shape)
    inside = ((block_lon.min(1) <= ds.lon) & (ds.lon <= block_lon.max(1)) & (block_lat.min(1) <= ds.lat) & (ds.lat <= block_lat.max(1)))
    assert inside.mean() > 0.99, f"only {inside.mean():.1%} of samples fall inside their own block: corner order is off"
    return block_lon, block_lat


def build_sample_set(fp_event, lon, lat, strike_deg, fault, frame_lonlat, profile_lats,
                     swath_km, exclude_km, km_window, cov_mask_km, cov_distmax_km, start_block_km,
                     min_block_km, nug_power):
    """One quadtree sample set of a fault-parallel field: the data set (fault_parallel_dataset), its far-field noise
    model (covariance), the distance-based quadtree (downsample) and the full data covariance of the samples
    (build_cd). 07a calls it three times: the paper's set (coherence-0.9 field, swath 4 km) and the two sensitivity
    sets (swath 7 km; the coherence-0.7 field). Returns a dict: samples (csi.insar, Cd set), sar (the pixel data
    set), cov_p, look, block_lon, block_lat, n_pixels.
    Source: 07a cells [3]-[4] (written for this project; CSI, m02 chain); replaces drafts/retired_2026-10-06/pylith/make_points_swath.py,
    which exec'd 07a."""
    sar, look = fault_parallel_dataset(fp_event, lon, lat, strike_deg, fault, frame_lonlat, swath_km,
                                       exclude_km, km_window, profile_lats)
    cov_p = covariance(sar, fault, swath_km, cov_mask_km, cov_distmax_km)
    samples = downsample(sar, fault, start_block_km, min_block_km)
    build_cd(samples, cov_p, nug_power)
    block_lon, block_lat = quadtree_block_corners(samples)
    return dict(samples=samples, sar=sar, cov_p=cov_p, look=look, block_lon=block_lon,
                block_lat=block_lat, n_pixels=int(np.isfinite(sar.vel).sum()))


def write_points_list(stem, lon, lat, crs, header, extra):
    """Write a sample set as a PyLith PointsList (`<stem>.txt`: "name x y z", m in the mesh CRS, z = 0 on the flat free
    surface) and an npz (`<stem>.npz`: lon, lat, x, y, names, crs and the arrays of `extra`). If the set exists
    already, its sample positions (lon, lat, x, y, and the coordinates of the .txt) must be unchanged: the cached
    PyLith Green's functions are computed at them. Returns (x, y).
    Source: drafts/retired_2026-10-06/pylith/make_points.py and make_points_swath.py, merged (written for this project; the PointsList
    format of PyLith OutputSolnPoints, see pylith/README.md)."""
    import pyproj
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    to_mesh = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_mesh.transform(lon, lat)
    lon_back, lat_back = to_mesh.transform(x, y, direction="INVERSE")
    assert np.max(np.abs(lon_back - lon)) < 1e-9 and np.max(np.abs(lat_back - lat)) < 1e-9, "CRS round trip failed"
    assert np.all(np.abs(x) < 45e3) and np.all(np.abs(y) < 45e3), "a sample lies outside the mesh domain"
    names = np.array([f"s{k:04d}" for k in range(lon.size)])
    lines = [f"{n} {xi:.3f} {yi:.3f} 0.0" for n, xi, yi in zip(names, x, y)]
    if os.path.exists(stem + ".npz"):
        old = np.load(stem + ".npz")
        for key, new_values in (("lon", lon), ("lat", lat), ("x", x), ("y", y)):
            assert old[key].shape == new_values.shape and np.array_equal(old[key], new_values), \
                f"{stem}.npz: sample {key} moved -> the cached Green's functions at these points no longer match"
        old_lines = [line.strip() for line in open(stem + ".txt") if not line.startswith("#")]
        assert old_lines == lines, f"{stem}.txt: the PyLith point coordinates changed"
        print(f"{os.path.basename(stem)}: {lon.size} sample positions unchanged (the cached Green's functions match)")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".txt", "w") as fh:
        fh.write(f"# {header}\n# name x y z (m) in {crs}\n")
        fh.write("\n".join(lines) + "\n")
    np.savez(stem + ".npz", lon=lon, lat=lat, x=x, y=y, names=names, crs=crs, **extra)
    return x, y


def samples_from_arrays(name, lon, lat, x_km, y_km, vel_mm, wgt, look, frame_lonlat, cov_p, nug_power):
    """A quadtree sample set rebuilt as a csi.insar data set from saved arrays (07a's model data product or a sample-set
    npz): positions in lon/lat and in CSI's local km frame (set exactly as saved, not recomputed), data (mm), block
    pixel counts, the fault-parallel look vector, and its full covariance (build_cd). Source: written for this
    project (2026-10-06), so 08a/08b/08c read 07a's product instead of exec'ing 07a."""
    samples = insar(name, lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    samples.read_from_binary(np.asarray(vel_mm, float).copy(), np.asarray(lon, float).copy(), np.asarray(lat, float).copy(),
                             err=np.ones(len(vel_mm)), los=np.tile(look, (len(vel_mm), 1)), remove_nan=False,
                             remove_zeros=False)
    samples.lon = np.asarray(lon, float).copy()                 # CSI stores lon in 0-360: keep the saved form
    samples.lat = np.asarray(lat, float).copy()
    samples.x = np.asarray(x_km, float).copy()
    samples.y = np.asarray(y_km, float).copy()
    samples.wgt = np.asarray(wgt, float).copy()
    build_cd(samples, cov_p, nug_power)
    return samples


# =====================================================================================================================
# 4. The uniform model's inversion (CSI triangles, homogeneous half-space)
# =====================================================================================================================
def crossing_profile(crossings, floor_mm, step_km, win_km, km_window, trace_south_north, frame_lonlat):
    """csi.surfaceslip: the running median of the five-event mean step at the fault crossings (04's
    event_fraction_by_crossing.csv: columns km, lat, mean_event_mm), every step_km over windows of +/- win_km / 2
    (at least 4 crossings), error 1.4826 MAD / sqrt(n) floored at floor_mm, placed on the trace; kept within
    km_window. Source: 07a [5] `make_surface` (written for this project)."""
    rows = []
    for k_centre in np.arange(np.ceil(crossings.km.min()), crossings.km.max(), step_km):
        in_win = np.abs(crossings.km - k_centre) <= win_km / 2 + 1e-6   # 1e-6: keep the edge crossings (float roundoff dropped them, fixed 2026-10-02)
        amps = crossings.mean_event_mm[in_win].dropna()
        if len(amps) < 4:
            continue
        med = float(np.median(amps))
        err = max(1.4826 * float(np.median(np.abs(amps - med))) / np.sqrt(len(amps)), floor_mm)
        lat_c = float(np.median(crossings.lat[in_win]))
        lon_c = float(np.interp(lat_c, trace_south_north[:, 1], trace_south_north[:, 0]))
        rows.append((lon_c, lat_c, k_centre, med, err))
    rows = np.array(rows)
    rows = rows[(rows[:, 2] >= km_window[0]) & (rows[:, 2] <= km_window[1])]
    surface_data = surfaceslip(f"crossing stacks f{floor_mm:g}", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    surface_data.read_from_binary(rows[:, 3], rows[:, 0], rows[:, 1], err=rows[:, 4])
    surface_data.buildCd()
    return surface_data


def crossing_profile_on_triangles(fault, crossings, floor_mm, step_km, win_km, km_window, trace_south_north,
                                  frame_lonlat, profile_lats):
    """The uniform model's surface constraint: the running-median profile (crossing_profile) interpolated to the centre
    of EVERY surface triangle of `fault`, each error scaled by sqrt(triangles within +/- step_km / 2) so the total
    weight per profile step is unchanged (CSI's default link, `Fault.surfaceGFs`, ties each point to its single
    nearest triangle). Source: 07a [5] `make_surface_interp` (written for this project)."""
    base = crossing_profile(crossings, floor_mm, step_km, win_km, km_window, trace_south_north, frame_lonlat)
    base_km = km_from_pinole(base.lat, profile_lats)
    order = np.argsort(base_km)
    surface_tri = []
    for index, patch in enumerate(fault.patch):
        if 0 in patch[:, 2]:
            surface_tri.append(index)
    centre_ll = np.array([fault.patchll[index].mean(0) for index in surface_tri])
    tri_km = km_from_pinole(centre_ll[:, 1], profile_lats)
    keep = (tri_km >= base_km.min()) & (tri_km <= base_km.max())
    centre_ll, tri_km = centre_ll[keep], tri_km[keep]
    vals = np.interp(tri_km, base_km[order], base.vel[order])
    errs = np.interp(tri_km, base_km[order], base.err[order])
    per_step = np.zeros(tri_km.size)
    for k, km in enumerate(tri_km):
        per_step[k] = np.sum(np.abs(tri_km - km) <= step_km / 2 + 1e-6)
    errs = errs * np.sqrt(np.maximum(per_step, 1))
    surface_data = surfaceslip(f"crossing interp f{floor_mm:g}", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    surface_data.read_from_binary(vals, centre_ll[:, 0], centre_ll[:, 1], err=errs)
    surface_data.buildCd()
    return surface_data


def tie_rows(fault, n_params, tie_depth_km, tie_bin_km=0.1, tie_err_mm=0.01):
    """The optional top tie as extra data rows: in every tie_bin_km along-strike column, each cell shallower than
    tie_depth_km minus the column's shallowest cell = 0, with a tiny error (a near-hard constraint). Returns the rows
    (n_ties x n_params) and their variances (mm^2).
    Source: 07a [6], after Jolivet et al. (2023), who tie 0-1 km (as cited in 07a); written for this project."""
    nodes, faces, _ = fault.mesh_sz
    centroid = nodes[faces].mean(1)
    shallow = np.nonzero(centroid[:, 1] < tie_depth_km)[0]
    column_of = np.floor(centroid[shallow, 0] / tie_bin_km).astype(int)
    rows = []
    for column in np.unique(column_of):
        members = shallow[column_of == column]
        if members.size < 2:
            continue
        reference = members[np.argmin(centroid[members, 1])]
        for member in members[members != reference]:
            row = np.zeros(n_params)
            row[member] = 1.0
            row[reference] = -1.0
            rows.append(row)
    if rows:
        tie_matrix = np.array(rows)
    else:
        tie_matrix = np.zeros((0, n_params))
    return tie_matrix, np.full(tie_matrix.shape[0], tie_err_mm ** 2)


def invert_triangles(fault, insar_datas, sigma, lam, frame_lonlat, ramp_spec="1", surface=None, tie_depth_km=None):
    """Right-lateral slip (mm) on the triangle fault from InSAR samples (and the surface constraint when given): Meade
    (2007) triangular dislocations in a homogeneous half-space (CSI `buildGFs`, sign flipped so right-lateral is
    positive), one transformation per dataset (ramp_spec for the InSAR, a constant for the surface data), the
    exponential prior `buildCmXY(sigma, lam)`, the optional top tie, NNLS (slip >= 0, the transformation free).
    Fills fault.slip and each dataset's synth, and removes the fitted transformation from each dataset's vel IN PLACE
    (pass copies). Returns the CSI solver.
    Source: 07a [6] `invert`; CSI (Jolivet et al.) calls as in the CSI tutorials (external/csi/notebooks/)."""
    solve_datas = list(insar_datas)
    if surface is not None:
        solve_datas.append(surface)
    for ds in insar_datas:
        fault.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
        fault.G[ds.name]["strikeslip"] *= -1.0        # right-lateral positive
    if surface is not None:
        fault.buildGFs(surface, slipdir="s", verbose=False)   # NO flip
    trans = transform("ramps", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    specs = [ramp_spec] * len(insar_datas)
    if surface is not None:
        specs.append("1")
    trans.buildGFs(solve_datas, specs, verbose=False)
    fault.assembleGFs(solve_datas, slipdir="s", verbose=False)
    fault.assembled(solve_datas, verbose=False)
    fault.assembleCd(solve_datas, verbose=False)
    trans.assembleGFs(solve_datas, verbose=False)
    trans.assembled(solve_datas, verbose=False)
    trans.assembleCd(solve_datas, verbose=False)
    fault.buildCmXY(sigma, lam, verbose=False)
    trans.buildCm(1e4, verbose=False)
    solver = multisrc("mean event", [fault, trans], verbose=False)
    solver.assembleGFs()
    solver.assembleCm()
    if tie_depth_km:                                  # uniform slip over the top tie_depth_km
        tie_matrix, tie_var = tie_rows(fault, solver.G.shape[1], tie_depth_km)
        solver.G = np.vstack([solver.G, tie_matrix])
        solver.d = np.concatenate([solver.d, np.zeros(tie_matrix.shape[0])])
        n_data = solver.Cd.shape[0]
        cd_tied = np.zeros((n_data + tie_matrix.shape[0], n_data + tie_matrix.shape[0]))
        cd_tied[:n_data, :n_data] = solver.Cd
        cd_tied[n_data:, n_data:] = np.diag(tie_var)
        solver.Cd = cd_tied
    mprior = np.zeros(solver.Np)
    for index, param_type in enumerate(solver.paramTypes):
        if param_type[0] == trans.name:
            mprior[index] = -1e3                      # NNLS: the transformation may be negative
    solver.ConstrainedLeastSquareSoln(method="nnls", mprior=mprior)
    solver.distributem()
    for ds in solve_datas:
        ds.buildsynth([fault], direction="s")
        ds.removeTransformation(trans)                # residuals are ramp-removed
    return solver


# =====================================================================================================================
# 5. Magnitudes and the L-curve
# =====================================================================================================================
def moment_magnitude(moment_nm):
    """Moment magnitude from the seismic moment M0 in N m: Mw = 2/3 (log10 M0 - 9.1) (Hanks & Kanamori 1979)."""
    return (2.0 / 3.0) * (np.log10(moment_nm) - 9.1)


def potency_from_slip(area_km2, slip_mm):
    """Potency (m^3) = the sum over cells of area x slip (km^2 -> m^2, mm -> m). Source: 07a [8] (written for this project)."""
    return float(np.sum(area_km2 * 1e6 * slip_mm * 1e-3))


def lcurve_corner(sigmas, misfit, roughness):
    """The L-curve corner (Hansen 1992). x = log10 misfit (pass sqrt(chi2 total)) and y = log10 roughness as cubic
    splines of log10 sigma; curvature = |x' y'' - y' x''| / (x'^2 + y'^2)^1.5 on 400 points between the second and
    the second-last grid sigma (the spline end intervals are excluded). Returns the sigma of maximum curvature, the
    400 log10 sigmas and the curvature. Source: 07a [8] (Hansen 1992; written for this project)."""
    log_sigma = np.log10(sigmas)
    spline_misfit = CubicSpline(log_sigma, np.log10(misfit))
    spline_rough = CubicSpline(log_sigma, np.log10(roughness))
    fine_log_sigma = np.linspace(log_sigma[1], log_sigma[-2], 400)
    d_misfit = spline_misfit(fine_log_sigma, 1)
    dd_misfit = spline_misfit(fine_log_sigma, 2)
    d_rough = spline_rough(fine_log_sigma, 1)
    dd_rough = spline_rough(fine_log_sigma, 2)
    curvature = np.abs(d_misfit * dd_rough - d_rough * dd_misfit) / (d_misfit ** 2 + d_rough ** 2) ** 1.5
    corner = float(10 ** fine_log_sigma[np.argmax(curvature)])
    return corner, fine_log_sigma, curvature


def lcurve_pick(sigmas, corner):
    """The selected grid sigma: the smallest grid sigma at or above the corner (the corner leaning to
    under-smoothing; Dani 2026-09-28). Source: 07a [8] (written for this project)."""
    sigmas = np.asarray(sigmas, float)
    return float(sigmas[sigmas >= corner].min())


# =====================================================================================================================
# 6. The surface constraint of the 3D model (measured crossings tied to the tent slip on the top edge)
# =====================================================================================================================
def surface_pct_denominator_mm(km, mean_mm, km_lo, km_hi, min_n=10):
    """The ONE denominator of every surface % (modelled surface slip / measured; 07a, 08a, 08b, 08c; Dani 2026-10-05):
    the median of the measured crossings (04's five-event mean step at each crossing; finite values only) with
    km_lo <= km <= km_hi, the SSE section of 04's section_lengths.csv (the 1e-6 km tolerance keeps its end crossings
    against float roundoff). Fails if fewer than min_n crossings are inside. mm.
    Source: drafts/retired_2026-10-06/scripts_pylith_helpers/surface_crossings.py (written for this project)."""
    km = np.asarray(km, float)
    mean_mm = np.asarray(mean_mm, float)
    inside = (km >= km_lo - 1e-6) & (km <= km_hi + 1e-6) & np.isfinite(mean_mm)
    assert inside.sum() >= min_n, f"only {inside.sum()} measured crossings in km {km_lo:.2f}-{km_hi:.2f}"
    return float(np.median(mean_mm[inside]))


def arc_length_on_polyline(px, py, vx, vy):
    """Arc length along the polyline (vx, vy) of the point on it nearest to each (px, py), that distance, and the arc
    length of every polyline vertex. Units of the inputs (m in the mesh CRS).
    Source: surface_crossings.py (written for this project)."""
    seg_x, seg_y = np.diff(vx), np.diff(vy)
    seg_len = np.hypot(seg_x, seg_y)
    assert np.all(seg_len > 0), "repeated vertex on the polyline"
    s_vert = np.r_[0.0, np.cumsum(seg_len)]
    dx = px[:, None] - vx[None, :-1]
    dy = py[:, None] - vy[None, :-1]
    t = np.clip((dx * seg_x + dy * seg_y) / seg_len ** 2, 0.0, 1.0)
    dist = np.hypot(dx - t * seg_x, dy - t * seg_y)
    j = dist.argmin(axis=1)
    rows = np.arange(len(px))
    return s_vert[j] + t[rows, j] * seg_len[j], dist[rows, j], s_vert


def local_scatter_mm(km, values, half_window_km=0.2, floor_mm=0.5):
    """1.4826 x MAD (mm) of the crossings within +/- half_window_km of each crossing, floored at floor_mm (2 x floor
    with fewer than 3 neighbours). The 1e-6 km tolerance keeps the window edges (crossings sit on a 0.1 km grid).
    Source: surface_crossings.py (written for this project)."""
    err = np.empty(len(km))
    for i, k in enumerate(km):
        nb = values[np.abs(km - k) <= half_window_km + 1e-6]
        err[i] = max(1.4826 * np.median(np.abs(nb - np.median(nb))), floor_mm) if nb.size >= 3 else 2.0 * floor_mm
    return err


def tent_rows(s, s_vert, top_columns, n_param):
    """G rows of the surface constraint: each datum at arc length s is linear between the two bracketing top-edge
    vertices (the order-1 PyLith tents), d = (1 - w) slip_j + w slip_(j+1), w = (s - s_j) / (s_(j+1) - s_j).
    top_columns[i] = the parameter column of the i-th top-edge vertex (in arc-length order).
    Source: surface_crossings.py (written for this project)."""
    j = np.clip(np.searchsorted(s_vert, s, side="right") - 1, 0, len(s_vert) - 2)
    w = (s - s_vert[j]) / (s_vert[j + 1] - s_vert[j])
    assert np.all((w >= -1e-9) & (w <= 1 + 1e-9)), "a datum lies outside its bracketing vertices"
    G = np.zeros((len(s), n_param))
    rows = np.arange(len(s))
    G[rows, top_columns[j]] = 1.0 - w
    G[rows, top_columns[j + 1]] += w
    return G


def crossing_constraint(km, lat, mean_mm, trace_lon_of_lat, to_mesh_xy, top_xy, top_columns, n_param,
                        floor_mm=0.5, max_dist_m=60.0):
    """The surface constraint of the 3D model (Dani 2026-10-02): every finite measured crossing that lies within the
    mesh top edge (ends excluded) is one datum, tied to the modelled slip AT ITS OWN POSITION on the top edge
    (tent_rows); error = the local scatter of the crossings (local_scatter_mm), floored at floor_mm. No running
    median, nothing invented across data gaps. trace_lon_of_lat(lat) -> lon on the mapped trace; to_mesh_xy(lon, lat)
    -> (x, y) in the mesh CRS (m); top_xy = top-edge vertices in arc-length order. Returns a dict: vel, lon, lat, err
    (mm), G (rows x n_param), km, dist_m, n_outside. Source: surface_crossings.py (written for this project)."""
    ok = np.isfinite(mean_mm)
    km, lat, mean_mm = np.asarray(km)[ok], np.asarray(lat)[ok], np.asarray(mean_mm)[ok]
    lon = trace_lon_of_lat(lat)
    x, y = to_mesh_xy(lon, lat)
    s, dist, s_vert = arc_length_on_polyline(np.asarray(x, float), np.asarray(y, float), top_xy[:, 0], top_xy[:, 1])
    inside = (s > 1.0) & (s < s_vert[-1] - 1.0)                      # beyond the mesh ends a crossing projects onto an end vertex
    assert inside.sum() >= 10, f"only {inside.sum()} crossings fall within the mesh top edge"
    assert dist[inside].max() < max_dist_m, f"a crossing lies {dist[inside].max():.0f} m from the mesh top edge"
    err = local_scatter_mm(km, mean_mm, floor_mm=floor_mm)[inside]  # scatter from ALL crossings, kept ones only
    G = tent_rows(s[inside], s_vert, np.asarray(top_columns), n_param)
    return dict(vel=mean_mm[inside], lon=lon[inside], lat=lat[inside], err=err, G=G, km=km[inside],
                dist_m=dist[inside], n_outside=int((~inside).sum()))


def surface_slip_data(constraint, floor_mm, frame_lonlat):
    """The crossing constraint as a csi.surfaceslip data set (values, positions, diagonal Cd from the errors).
    Source: 08a [10] (CSI, Jolivet et al.)."""
    surface_data = surfaceslip(f"crossings f{floor_mm:g}", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    surface_data.read_from_binary(constraint["vel"], constraint["lon"], constraint["lat"], err=constraint["err"])
    surface_data.buildCd()
    return surface_data


# =====================================================================================================================
# 7. PyLith Green's functions -> the fault-parallel G matrix
# =====================================================================================================================
def read_gf_run(pylith_output_dir, name):
    """One PyLith GreensFns run: fault vertices (m) and cells, the impulse slip (n_impulses, n_vertices, 3: opening,
    left-lateral, reverse), the output-point coordinates (m) and their displacement (n_impulses, n_points, 3).
    Source: the reading pattern of PyLith examples/subduction-3d/invert_slip.py (readGreens)."""
    with h5py.File(os.path.join(pylith_output_dir, f"{name}-fault.h5"), "r") as h5:
        fault_xyz = h5["geometry/vertices"][:]
        fault_cells = np.asarray(h5["viz/topology/cells"][:], dtype=np.int64)
        slip = h5["vertex_fields/slip"][:]
    with h5py.File(os.path.join(pylith_output_dir, f"{name}-points.h5"), "r") as h5:
        point_xyz = h5["geometry/vertices"][:]
        disp = h5["vertex_fields/displacement"][:]
    assert slip.shape[0] == disp.shape[0], "impulse count differs between the fault and points files"
    return fault_xyz, fault_cells, slip, point_xyz, disp


def trace_top(fault_xyz):
    """(y, x) of the fault's top edge (z = 0), sorted by y: the mesh's own surface trace (m).
    Source: build_G.py (written for this project)."""
    top = fault_xyz[np.abs(fault_xyz[:, 2]) < 1e-6]
    order = np.argsort(top[:, 1])
    return top[order, 1], top[order, 0]


def build_G(names, points_npz, strike_deg, pylith_output_dir, gf_dir):
    """Assemble the fault-parallel Green's-function matrix G (n_samples x n_impulses, m of fault-parallel displacement
    per m of right-lateral slip) from PyLith GreensFns runs (several = concatenated along the impulse axis) and write
    gf_dir/G_<names[0]>.npz. The strike is EXPLICIT: G is the 3D displacement projected on the horizontal unit vector
    along strike_deg (07a's fault-parallel look vector), so a G built at another strike must be rebuilt (load_G).
    Glue (assert-checked): impulse -> fault vertex (exactly one vertex with unit left-lateral slip); output point ->
    sample (nearest neighbour in the mesh CRS, < 1 mm); sign: right-lateral = -left-lateral with PyLith's fault
    orientation, DERIVED (uniform slip over the impulse region and every surface impulse must move the east side
    along +look, as the data). Source: the reading pattern of PyLith examples/subduction-3d/invert_slip.py (readGreens:
    fault geometry/vertices, viz/topology/cells, vertex_fields/slip[:, :, 1]; responses vertex_fields/displacement)
    and its matchCoords idea; replaces drafts/retired_2026-10-06/pylith/build_G.py and drafts/retired_2026-10-06/scripts_pylith_helpers/build_G.py."""
    points = np.load(points_npz)
    look = look_vector(strike_deg)
    columns, vertex_index, fault_xyz0, cells0 = [], [], None, None
    for name in names:
        fault_xyz, fault_cells, slip, point_xyz, disp = read_gf_run(pylith_output_dir, name)
        if fault_xyz0 is None:
            fault_xyz0, cells0 = fault_xyz, fault_cells
        assert np.array_equal(fault_xyz, fault_xyz0), f"{name}: different fault mesh from {names[0]}"
        # impulse -> vertex (unit left-lateral slip on exactly one vertex)
        ll = slip[:, :, 1]
        k_vertex = np.argmax(np.abs(ll), axis=1)
        peak = ll[np.arange(ll.shape[0]), k_vertex]
        assert np.allclose(np.abs(peak), 1.0, atol=1e-6), f"{name}: impulse amplitude not 1 m: {peak.min()}..{peak.max()}"
        others = ll.copy()
        others[np.arange(ll.shape[0]), k_vertex] = 0.0
        assert np.max(np.abs(others)) < 1e-6 and np.max(np.abs(slip[:, :, [0, 2]])) < 1e-6, f"{name}: impulse not a single-vertex tent"
        # output point -> data sample
        dist, order = cKDTree(point_xyz[:, :2]).query(np.c_[points["x"], points["y"]])
        assert dist.max() < 1e-3, f"{name}: a data sample has no PyLith output point within 1 mm ({dist.max():.3g} m)"
        assert np.unique(order).size == order.size, "two samples matched the same output point"
        columns.append(np.einsum("kpc,c->pk", disp[:, order, :], look) * np.sign(peak)[None, :])
        vertex_index.append(k_vertex)
    G_ll = np.hstack(columns)                              # m of fault-parallel displacement per m of LEFT-lateral slip
    vertex_index = np.concatenate(vertex_index)
    assert np.unique(vertex_index).size == vertex_index.size, "a fault vertex received two impulses"
    # sign: right-lateral slip moves the east side along +look (07a's data convention)
    # (a) uniform left-lateral slip over the whole impulse region (sum of all tents): east-minus-west step of the
    #     samples within 500 m of the trace along the region; (b) every SURFACE-vertex impulse individually (a buried
    #     tent's fault-parallel field has lobes of both signs, so it is not a clean test)
    cross = points["x"] - np.interp(points["y"], *trace_top(fault_xyz0))
    y_imp = fault_xyz0[vertex_index, 1]
    band = (points["y"] > y_imp.min()) & (points["y"] < y_imp.max()) & (np.abs(cross) < 500.0)
    assert (band & (cross > 0)).sum() >= 5 and (band & (cross < 0)).sum() >= 5, "too few samples beside the impulse region"
    uniform = G_ll.sum(axis=1)
    step_uniform = uniform[band & (cross > 0)].mean() - uniform[band & (cross < 0)].mean()
    sign_rl = -1.0 if step_uniform < 0 else 1.0
    surface_k = np.nonzero(np.abs(fault_xyz0[vertex_index, 2]) < 1e-6)[0]
    step_k = []
    for k in surface_k:
        near = (np.abs(points["y"] - y_imp[k]) < 500.0) & (np.abs(cross) < 300.0)
        step_k.append(G_ll[near & (cross > 0), k].mean() - G_ll[near & (cross < 0), k].mean())
    step_k = sign_rl * np.array(step_k)
    print(f"right-lateral = {sign_rl:+.0f} x left-lateral (uniform-slip east-minus-west step {sign_rl * step_uniform:+.3f} m/m); "
          f"{np.sum(step_k > 0)}/{step_k.size} surface impulses east-minus-west positive")
    assert np.all(step_k > 0), "a surface impulse has the wrong east-west sign"
    G_rl = sign_rl * G_ll
    out = os.path.join(gf_dir, f"G_{names[0]}.npz")
    os.makedirs(gf_dir, exist_ok=True)
    np.savez(out, G_rl_m_per_m=G_rl, vertex_index=vertex_index, fault_xyz=fault_xyz0, fault_cells=cells0,
             sample_x=points["x"], sample_y=points["y"], sample_lon=points["lon"], sample_lat=points["lat"],
             look=look, strike_deg=strike_deg, sign_rl=sign_rl, runs=np.array(names),
             provenance=f"model_utils.build_G {time.strftime('%Y-%m-%d %H:%M')} from {names} in {pylith_output_dir}; "
                        f"points {points_npz}; strike {strike_deg:.6f}")
    print(f"wrote {out}: G {G_rl.shape}, max |G| {np.abs(G_rl).max():.3g} m/m")
    return out


def load_G(name, points_npz, strike_deg, pylith_output_dir, gf_dir):
    """The G matrix of one GreensFns run (gf_dir/G_<name>.npz), (re)built from the cached PyLith outputs when it is
    missing or was projected on another strike than strike_deg (no PyLith run either way; 2026-10-05: the along-trace
    section moved the strike 146.40 -> 146.51). Asserts the strike and that G is on the points of points_npz.
    Returns the loaded npz. Source: 08a [5] and the driver's strike fix (written for this project)."""
    path = os.path.join(gf_dir, f"G_{name}.npz")
    if not os.path.exists(path):
        build_G([name], points_npz, strike_deg, pylith_output_dir, gf_dir)
    gnpz = np.load(path)
    if abs(float(gnpz["strike_deg"]) - strike_deg) > 1e-6:
        print(f"G_{name}: built at strike {float(gnpz['strike_deg']):.4f}, field at {strike_deg:.4f} -> rebuilding")
        build_G([name], points_npz, strike_deg, pylith_output_dir, gf_dir)
        gnpz = np.load(path)
    assert abs(float(gnpz["strike_deg"]) - strike_deg) < 1e-6, f"G_{name} strike differs from the field's"
    points = np.load(points_npz)
    assert np.array_equal(gnpz["sample_lon"], points["lon"]), f"G_{name} is not on the output points of {points_npz}"
    return gnpz


# =====================================================================================================================
# 8. The tent (vertex) parameterisation of the PyLith fault, and the inversion
# =====================================================================================================================
class TentFault:
    """The PyLith fault of a G matrix as the inversion sees it: the fault vertices and triangles (m, z up), the
    impulse vertices (= G's columns, order-1 tents), each triangle's centroid, unit normal (east +) and area, the
    top-edge vertices and the km from Point Pinole of every impulse vertex. Plain container, built once per mesh.
    Source: 08a [5], [8], [10] (written for this project)."""

    def __init__(self, gnpz, to_mesh, profile_lats):
        self.fault_xyz = gnpz["fault_xyz"]
        self.fault_cells = gnpz["fault_cells"]
        self.vertex_index = gnpz["vertex_index"]
        self.n_imp = self.vertex_index.size
        self.tri_xyz = self.fault_xyz[self.fault_cells]
        self.tri_cen = self.tri_xyz.mean(1)
        tri_n = np.cross(self.tri_xyz[:, 1] - self.tri_xyz[:, 0], self.tri_xyz[:, 2] - self.tri_xyz[:, 0])
        self.tri_area = 0.5 * np.linalg.norm(tri_n, axis=1)
        tri_n /= np.linalg.norm(tri_n, axis=1)[:, None]
        tri_n[tri_n[:, 0] < 0] *= -1.0                                             # east +
        self.tri_n = tri_n
        self.imp_xyz = self.fault_xyz[self.vertex_index]
        top = self.fault_xyz[np.abs(self.fault_xyz[:, 2]) < 1e-6]                 # the mesh fault's top-edge vertices
        self.top = top[np.argsort(top[:, 1])]                                      # ... sorted south to north (mesh y)
        top_km = km_from_pinole(to_mesh.transform(self.top[:, 0], self.top[:, 1], direction="INVERSE")[1], profile_lats)
        self.imp_km = np.interp(self.imp_xyz[:, 1], self.top[:, 1], top_km)       # km from Point Pinole of every impulse vertex
        self.tri_km = np.interp(self.tri_cen[:, 1], self.top[:, 1], top_km)       # ... and of every triangle centroid
        self.tri_depth_m = -self.tri_cen[:, 2]                                     # m, depth positive
        self.imp_depth_km = -self.imp_xyz[:, 2] / 1e3
        self.imp_lon, self.imp_lat = to_mesh.transform(self.imp_xyz[:, 0], self.imp_xyz[:, 1], direction="INVERSE")
        column_of_vertex = {}                                   # PyLith vertex index -> its column in G (impulse order)
        for column, vertex in enumerate(self.vertex_index):
            column_of_vertex[vertex] = column
        sub_faces = []                                          # the fault triangles whose three vertices all carry an impulse
        for triangle in self.fault_cells:
            if all(vertex in column_of_vertex for vertex in triangle):
                sub_faces.append([column_of_vertex[vertex] for vertex in triangle])
        self.sub_faces = np.array(sub_faces)
        self.surface_imp = np.nonzero(np.abs(self.imp_xyz[:, 2]) < 1e-6)[0]
        self.top_order = self.surface_imp[np.argsort(self.imp_xyz[self.surface_imp, 1])]   # top-edge impulse vertices, south to north
        assert np.all(np.diff(self.imp_xyz[self.top_order, 1]) > 0), "top-edge vertices not monotone in mesh y"

    def vertex_slip(self, slip_mm):
        """Slip on every PyLith fault vertex: the impulse vertices carry slip_mm, the locked ones zero."""
        slip_all = np.zeros(len(self.fault_xyz))
        slip_all[self.vertex_index] = slip_mm
        return slip_all

    def potency_per_triangle_m3(self, slip_mm):
        """Exact integral of the piecewise-linear (tent) slip over each PyLith fault triangle: area x mean of its three
        vertex slips (m^3; slip in mm, locked vertices = 0)."""
        return self.tri_area * self.vertex_slip(slip_mm)[self.fault_cells].mean(1) * 1e-3

    def potency_m3(self, slip_mm):
        """Potency (m^3) of the tent slip: sum over triangles of area x mean vertex slip."""
        slip_all = self.vertex_slip(slip_mm)
        return float(np.sum(self.tri_area * slip_all[self.fault_cells].mean(1)) * 1e-3)

    def potency_depth_m(self, slip_mm, fraction):
        """The depth (m) above which `fraction` of the potency lies (model_utils.potency_extent on the triangles: exact
        tent potency per triangle, triangle centroid depth). Register #22: depth50 / depth90."""
        pot_tri = self.potency_per_triangle_m3(slip_mm)
        assert abs(pot_tri.sum() - self.potency_m3(slip_mm)) <= 1e-9 * max(1.0, self.potency_m3(slip_mm))
        depth_m, _ = potency_extent(pot_tri, self.tri_depth_m, self.tri_km, fraction)
        return depth_m

    def moment_nm(self, slip_mm, mu_per_triangle):
        """Seismic moment (N m): sum over triangles of area x mean tent slip x the triangle's rigidity (Pa)."""
        slip_all = self.vertex_slip(slip_mm)
        return float(np.sum(self.tri_area * slip_all[self.fault_cells].mean(1) * 1e-3 * mu_per_triangle))


def tent_fault_template(tents, frame_lonlat):
    """csi.TriangularTents on the impulse vertices (km, depth +), in G's column order; faces = the PyLith fault
    triangles among them. Built once (its adjacency map is slow); each solve deep-copies it.
    Source: 08a [10] (CSI TriangularTents, Jolivet et al.)."""
    template = TriangularTents("tents", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    x_km, y_km = template.ll2xy(tents.imp_lon, tents.imp_lat)
    assert np.allclose(x_km * 1e3, tents.imp_xyz[:, 0], atol=1e-3) and np.allclose(y_km * 1e3, tents.imp_xyz[:, 1], atol=1e-3), \
        "CSI frame differs from the mesh CRS"
    template.Vertices = np.c_[x_km, y_km, -tents.imp_xyz[:, 2] / 1e3]
    template.Vertices_ll = np.c_[tents.imp_lon, tents.imp_lat, -tents.imp_xyz[:, 2] / 1e3]
    template.Faces = tents.sub_faces.copy()
    template.patch = [template.Vertices[face] for face in template.Faces]
    template.patchll = [template.Vertices_ll[face] for face in template.Faces]
    template.vertices2tents()
    template.initializeslip(n=tents.n_imp)
    assert len(template.tent) == tents.n_imp and template.N_slip == tents.n_imp
    return template


def data_cd(mode, samples, cov_p, nug_power, keep=None):
    """The data covariance (mm^2) of one treatment of the samples (n_px = the pixels each quadtree block averages, CSI's wgt):
      nugget -- diagonal, nugget2 / n_px**nug_power: the samples treated as independent. nug_power 0 = every sample the
                nugget variance (equal weights; THE LOCKED WEIGHTING, model_config.weighting / nugget_power, register #6);
                0.5 = the block-size weighting used until 2026-10-25.
      full   -- the full correlated Cd of the semivariogram (covariance_matrix with nug_power): Lohman and Simons (2005),
                Sudhaus and Jonsson (2009); the sensitivity row.
    The exponent is the `nug_power` ARGUMENT in both modes (until 2026-10-30 "nugget" silently took the exponent the
    samples had been built with). keep (2026-10-16): samples outside it (closer to the trace than the near-trace cut)
    are decoupled from every other sample and given a variance 1e8 times their own, so they take no part in the fit but
    stay in the products. Source: 08a [10] `data_cd` (written for this project)."""
    assert nug_power is not None and np.isfinite(float(nug_power)), "data_cd needs the nugget exponent explicitly"
    n_px = np.asarray(samples.wgt, float)
    assert n_px.shape == (samples.vel.size,) and np.all(n_px > 0), "the quadtree block pixel counts (wgt) are missing"
    if mode == "nugget":
        Cd = np.diag(float(cov_p[0]) / n_px ** float(nug_power))
    else:
        assert mode == "full", f"unknown Cd treatment {mode!r} (nugget | full)"
        Cd = covariance_matrix(np.asarray(samples.x, float), np.asarray(samples.y, float), n_px, cov_p, float(nug_power))
    assert np.all(np.diag(Cd) > 0)
    if keep is not None:
        keep = np.asarray(keep, bool)
        assert keep.shape == (samples.vel.size,) and keep.sum() > 0
        out = ~keep
        diag_out = np.diag(Cd)[out].copy()
        Cd[out, :] = 0.0
        Cd[:, out] = 0.0
        Cd[out, out] = 1e8 * diag_out
    return Cd


def invert_tents(template, samples, G_insar, Cd, surface_template, G_surface, sigma, lam, frame_lonlat, use_surface=False,
                 vel_insar=None, vel_surface=None, name="tents", ramp_spec="1,x,y"):
    """Right-lateral slip on the PyLith tents (mm) from the InSAR samples (and, with use_surface, the crossing
    constraint), the PyLith G entered by CSI's `Fault.setGFs`: exponential prior buildCmXY(sigma, lam) on the vertex
    positions, the reference frame as a CSI transformation (ramp_spec, CSI polynomial terms: "1,x,y" a plane solved
    jointly with slip = the LOCKED setting, register #7; "1" a constant = the sensitivity row; the surface data keep a
    constant), NNLS (slip >= 0, the frame terms free) with multisourcesolve. Works on deep copies of the template fault
    and the samples (Cd replaced by `Cd`); vel_insar replaces the data vector (the Monte Carlo). surface_template None =
    no surface data at all (the locked model); with surface data but use_surface False they are predicted, not fitted.
    Returns (fault, insar data, surface data or None), each data set with its synth and the fitted frame removed from vel.
    Source: 08a [10] `invert_tents` (CSI, Jolivet et al.; PyLith GFs as in the PyLith inversion example
    examples/subduction-3d/invert_slip.py)."""
    fault_model = copy.deepcopy(template)
    fault_model.name = name
    insar_data = copy.deepcopy(samples)
    if vel_insar is not None:
        insar_data.vel[:] = vel_insar
    assert Cd.shape == (insar_data.vel.size, insar_data.vel.size)
    insar_data.Cd = Cd.copy()
    fault_model.setGFs(insar_data, strikeslip=[G_insar.copy()])     # already right-lateral positive (build_G sign)
    surface = None
    if surface_template is not None:
        surface = copy.deepcopy(surface_template)
        if vel_surface is not None:
            surface.vel[:] = vel_surface
        fault_model.setGFs(surface, strikeslip=[G_surface])
    assert not (use_surface and surface is None), "use_surface needs the surface data"
    if use_surface:
        solve_datas = [insar_data, surface]
    else:
        solve_datas = [insar_data]
    trans = transform("ramps", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    trans.buildGFs(solve_datas, [str(ramp_spec)] + ["1"] * (len(solve_datas) - 1), verbose=False)
    fault_model.assembleGFs(solve_datas, slipdir="s", verbose=False)
    fault_model.assembled(solve_datas, verbose=False)
    fault_model.assembleCd(solve_datas, verbose=False)
    trans.assembleGFs(solve_datas, verbose=False)
    trans.assembled(solve_datas, verbose=False)
    trans.assembleCd(solve_datas, verbose=False)
    fault_model.buildCmXY(sigma, lam, verbose=False)
    trans.buildCm(1e4, verbose=False)
    solver = multisrc(f"{name} mean event", [fault_model, trans], verbose=False)
    solver.assembleGFs()
    solver.assembleCm()
    mprior = np.zeros(solver.Np)
    ramp_idx = []                                                     # the parameters of the reference frame
    for i, param_type in enumerate(solver.paramTypes):
        if param_type[0] == trans.name:
            ramp_idx.append(i)
    mprior[ramp_idx] = -1e3
    solver.ConstrainedLeastSquareSoln(method="nnls", mprior=mprior)
    solver.distributem()
    for dataset in solve_datas:
        dataset.buildsynth([fault_model], direction="s")
        dataset.removeTransformation(trans)
    if surface is not None and not use_surface:                       # not inverted, but still predicted (a check)
        surface.buildsynth([fault_model], direction="s")
    assert np.all(fault_model.slip[:, 0] >= -1e-9), "negative slip out of NNLS"
    return fault_model, insar_data, surface


def match_impulse_order(ref_fault_xyz, ref_vertex_index, fault_xyz, vertex_index, label="", tol_m=1e-3):
    """Column order that puts one GreensFns run's impulses into a reference run's order, matched by COORDINATES.
    PyLith numbers the fault vertices differently with a different number of MPI ranks, so two runs of the same mesh can
    list the same impulses in a different order. Asserts the same fault mesh (every vertex within tol_m, one to one)
    and the same impulse vertices; returns idx with fault_xyz[vertex_index][idx] == ref_fault_xyz[ref_vertex_index].
    Source: written for this project (2026-10-25)."""
    from scipy.spatial import cKDTree
    ref_fault = np.asarray(ref_fault_xyz, float)
    fault = np.asarray(fault_xyz, float)
    assert fault.shape == ref_fault.shape, f"fault mesh differs ({label}: {fault.shape} vs {ref_fault.shape})"
    d_fault, i_fault = cKDTree(fault).query(ref_fault)
    assert d_fault.max() < tol_m and np.unique(i_fault).size == i_fault.size, \
        f"fault mesh differs ({label}: vertex match up to {d_fault.max():.3g} m)"
    ref_imp = ref_fault[np.asarray(ref_vertex_index)]
    imp = fault[np.asarray(vertex_index)]
    assert imp.shape == ref_imp.shape, f"impulse count differs ({label}: {imp.shape[0]} vs {ref_imp.shape[0]})"
    d_imp, idx = cKDTree(imp).query(ref_imp)
    assert d_imp.max() < tol_m and np.unique(idx).size == idx.size, f"impulse vertices differ ({label}: up to {d_imp.max():.3g} m)"
    return idx


def _csi_triangle_chunk(lon_v, lat_v, depth_km_v, lon, lat, look, frame_lonlat):
    """CSI triangular-dislocation (Meade 2007) strike-slip Green's functions of one chunk of constant-slip triangles
    (corners lon_v, lat_v, depth_km_v, 3 per triangle, flattened) at the samples (lon, lat), projected on `look`:
    (n_samples, n_triangles), per unit slip, CSI's sign. A separate function so chunks can run in parallel processes."""
    from csi import TriangularPatches, insar
    line = insar("samples", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    line.read_from_binary(np.zeros(np.size(lon)), np.asarray(lon, float), np.asarray(lat, float), err=np.ones(np.size(lon)),
                          los=np.tile(np.asarray(look, float), (np.size(lon), 1)), remove_nan=False, remove_zeros=False)
    flt = TriangularPatches("analytic tents", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    x, y = flt.ll2xy(lon_v, lat_v)
    flt.Vertices = np.c_[x, y, depth_km_v]                       # km, depth positive
    flt.Faces = np.arange(np.size(lon_v)).reshape(-1, 3)
    flt.vertices2ll()
    flt.patch = [flt.Vertices[f] for f in flt.Faces]
    flt.patch2ll()
    flt.setdepth()
    flt.homogeneizeStrike(direction=0, sign=-1.0)                # one strike direction for every sub-triangle (as tent_check)
    flt.initializeslip()
    flt.buildGFs(line, vertical=True, slipdir="s", verbose=False)
    G_part = flt.G[line.name]["strikeslip"]
    assert G_part.shape == (np.size(lon), np.size(lon_v) // 3), f"CSI G {G_part.shape}"
    return G_part


def analytic_tent_G(tents, lon, lat, look, frame_lonlat, to_mesh, n_sub=8, verbose=False, n_workers=1):
    """Homogeneous half-space Green's functions of the PyLith tents (unit right-lateral slip at one impulse vertex,
    linear to zero at its neighbours), on the SAME fault mesh: every fault triangle touching an impulse vertex is split
    into n_sub^2 constant-slip sub-triangles, each sub-triangle's displacement comes from CSI's triangular dislocation
    (Meade 2007, Poisson solid, `TriangularPatches.buildGFs`), and is added to each corner tent with the tent's value at
    the sub-triangle centroid. Exact for a half-space up to the sub-division (n_sub 8 vs 16: 0.3 %, 2026-10-25
    tent check). Works for any dip (the triangles carry the geometry). Sign: CSI's strike-slip with the normals turned
    east (homogeneizeStrike(0, -1)) is left-lateral for this trace, so G = -G_csi (checked against PyLith, tent check;
    tests/test_model_utils.py); callers should still assert a uniform-slip east-minus-west step > 0. n_workers > 1 runs
    the chunks of 5000 sub-triangles in parallel processes (same result: the chunks are summed in order).
    Returns G (n_samples, tents.n_imp), m per m.
    Source: written for this project (2026-10-26), generalising order2_surface_tent_check_2026-10-25/tent_check.py."""
    from scipy.sparse import csr_matrix
    column_of_vertex = {int(v): c for c, v in enumerate(tents.vertex_index)}
    sub_corners = []                                            # the n_sub^2 sub-triangles, corners in barycentric (i, j)
    for i in range(n_sub):
        for j in range(n_sub - i):
            sub_corners.append(((i, j), (i + 1, j), (i, j + 1)))
            if i + j < n_sub - 1:
                sub_corners.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
    lams = np.array([[[a / n_sub, b / n_sub, 1 - (a + b) / n_sub] for a, b in sc] for sc in sub_corners])   # (n_sub^2, 3, 3)
    assert lams.shape[0] == n_sub ** 2
    w_corner = lams.mean(1)                                      # tent value of each corner at each sub-triangle centroid
    pts, rows, cols, vals = [], [], [], []
    for cell in tents.fault_cells:
        hit = [k for k, v in enumerate(cell) if int(v) in column_of_vertex]
        if not hit:
            continue
        corners = tents.fault_xyz[cell]                          # (3, 3) m, z up
        first = len(pts) * n_sub ** 2
        pts.append(np.einsum("sab,bc->sac", lams, corners))      # (n_sub^2, 3, 3) sub-triangle corners
        for k in hit:
            rows.append(first + np.arange(n_sub ** 2))
            cols.append(np.full(n_sub ** 2, column_of_vertex[int(cell[k])]))
            vals.append(w_corner[:, k])
    pts = np.concatenate(pts)                                    # (n_patch, 3, 3)
    n_patch = pts.shape[0]
    W = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_patch, tents.n_imp))
    assert np.all(np.asarray(W.sum(0)).ravel() > 0), "an impulse vertex touches no fault triangle"
    lon_all, lat_all = to_mesh.transform(pts[:, :, 0].ravel(), pts[:, :, 1].ravel(), direction="INVERSE")
    depth_all = -pts[:, :, 2].ravel() / 1e3                      # km, depth positive
    chunk = 5000                                                 # sub-triangles per CSI fault (memory: n_samples x chunk)
    starts = list(range(0, n_patch, chunk))
    jobs = [(lon_all[3 * s:3 * (s + chunk)], lat_all[3 * s:3 * (s + chunk)], depth_all[3 * s:3 * (s + chunk)],
             np.asarray(lon, float), np.asarray(lat, float), np.asarray(look, float), tuple(frame_lonlat)) for s in starts]
    G = np.zeros((np.size(lon), tents.n_imp))
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = pool.map(_csi_triangle_chunk, *zip(*jobs))
            for start, G_part in zip(starts, results):
                G += np.asarray((W[start:start + chunk].T @ G_part.T).T)
                if verbose:
                    print(f"  analytic tents: {min(start + chunk, n_patch):,} / {n_patch:,} sub-triangles", flush=True)
    else:
        for start, job in zip(starts, jobs):
            G_part = _csi_triangle_chunk(*job)
            G += np.asarray((W[start:start + chunk].T @ G_part.T).T)
            if verbose:
                print(f"  analytic tents: {min(start + chunk, n_patch):,} / {n_patch:,} sub-triangles", flush=True)
    return -G


def positions_fingerprint(lon, lat):
    """12-hex sha256 of a sample set's positions (lon, lat as float64): the key of every matrix computed AT the samples
    (a rewritten point file with the same positions keeps its key). Written for this project (2026-10-30)."""
    import hashlib
    return hashlib.sha256(np.c_[np.asarray(lon, np.float64), np.asarray(lat, np.float64)].tobytes()).hexdigest()[:12]


def analytic_tent_G_cached(cache_npz, tents, lon, lat, look, frame_lonlat, to_mesh, n_sub, run_name, points_fingerprint,
                           n_workers=1):
    """analytic_tent_G, cached in cache_npz and keyed by its query: the PyLith run whose tents it uses, n_sub, the
    fingerprint of the sample point set and the impulse vertices (all asserted on reading, so a changed question can
    never be served a stale matrix). Computed (slow: ~1 h per 100,000 sub-triangles) only when the file is missing.
    Returns G (n_samples, tents.n_imp), m per m, right-lateral positive. Written for this project (2026-10-30)."""
    if not os.path.exists(cache_npz):
        t0 = time.time()
        G = analytic_tent_G(tents, lon, lat, look, frame_lonlat, to_mesh, n_sub=n_sub, verbose=True, n_workers=n_workers)
        os.makedirs(os.path.dirname(cache_npz), exist_ok=True)
        np.savez(cache_npz, G_rl_m_per_m=G, n_sub=n_sub, run_name=run_name, points_fingerprint=points_fingerprint,
                 imp_xyz=tents.imp_xyz, sample_lon=np.asarray(lon, float), sample_lat=np.asarray(lat, float),
                 provenance=f"model_utils.analytic_tent_G (CSI Meade 2007 triangular dislocations, n_sub {n_sub}) on the tents of "
                            f"{run_name}, {time.strftime('%Y-%m-%d %H:%M')}, {(time.time() - t0) / 60:.0f} min")
        print(f"wrote {cache_npz} ({(time.time() - t0) / 60:.0f} min)")
    cached = np.load(cache_npz)
    assert int(cached["n_sub"]) == n_sub and str(cached["run_name"]) == run_name, f"{cache_npz}: another n_sub or run"
    assert str(cached["points_fingerprint"]) == points_fingerprint, f"{cache_npz}: made for another sample set"
    assert np.allclose(cached["imp_xyz"], tents.imp_xyz, atol=1e-3), f"{cache_npz}: another impulse order or mesh"
    assert np.array_equal(cached["sample_lon"], np.asarray(lon, float)), f"{cache_npz}: other sample positions"
    return cached["G_rl_m_per_m"]


def tent_model_setup(run_names, points_npz, strike_deg, medium_dirs, km_window, creep_lo_km, creep_hi_km, trace_xy_m,
                     frame_lonlat, crs, profile_lats, near_trace_cut_km, near_field_km, trace_km_pp, crossings=None,
                     surf_floor_mm=0.5, trace_south_north=None):
    """Everything an inversion on PyLith Green's functions needs, for one mesh and one sample set, built from the
    products alone: the G matrix of every medium (load_G; rebuilt from the cached PyLith output when it was projected
    on another strike), the tent geometry (TentFault; media matched by vertex COORDINATES because PyLith's vertex
    numbering depends on the MPI rank count, each medium's G columns reordered into the first medium's impulse order)
    and its CSI template; the signed distance (km, east +) of every sample from the model trace (`trace_xy_m`, m in the
    mesh CRS) and its along-strike km (`sample_km_pp`, nearest trace point of `trace_km_pp`); `keep` = the samples that
    enter the fit (|d| >= near_trace_cut_km, register #2); the near-field masks `west_near` / `east_near` = fitted
    samples within near_field_km west / east of the trace inside the SSE section (creep_lo_km-creep_hi_km), the
    near-field RMS of the metric policy (2026-10-28). With `crossings` (dict km, lat, mean_mm of 04's measured crossings)
    also the measured-crossing surface constraint (`surface`, `G_surface`; off in the locked model, register #17).
    run_names: medium -> PyLith GreensFns run; medium_dirs: medium -> (pylith_output dir, gf dir). Returns a dict.
    Source: 08a [5], [7], [10] (written for this project)."""
    import pyproj
    to_mesh = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    gnpz = {}
    G = {}
    for medium, run_name in run_names.items():
        pylith_output_dir, gf_dir = medium_dirs[medium]
        gnpz[medium] = load_G(run_name, points_npz, strike_deg, pylith_output_dir, gf_dir)
        G[medium] = gnpz[medium]["G_rl_m_per_m"]
    ref_medium = list(run_names)[0]
    impulse_order = {}
    for medium in run_names:
        impulse_order[medium] = match_impulse_order(gnpz[ref_medium]["fault_xyz"], gnpz[ref_medium]["vertex_index"],
                                                    gnpz[medium]["fault_xyz"], gnpz[medium]["vertex_index"], label=medium)
        G[medium] = G[medium][:, impulse_order[medium]]                    # columns in the reference impulse order
    tents = TentFault(gnpz[ref_medium], to_mesh, profile_lats)
    template = tent_fault_template(tents, frame_lonlat)
    points = np.load(points_npz)
    signed_d_km = signed_distance_m(trace_xy_m[0], trace_xy_m[1], points["x"], points["y"]) / 1e3
    keep = np.abs(signed_d_km) >= near_trace_cut_km                        # samples that enter the fit (register #2)
    trace_km_pp = np.asarray(trace_km_pp, float)
    assert trace_km_pp.size == np.asarray(trace_xy_m[0]).size, "trace_km_pp must match trace_xy_m"
    nearest = cKDTree(np.c_[trace_xy_m[0], trace_xy_m[1]]).query(np.c_[points["x"], points["y"]])[1]
    sample_km_pp = trace_km_pp[nearest]
    in_section = (sample_km_pp >= creep_lo_km) & (sample_km_pp <= creep_hi_km)
    west_near = keep & in_section & (signed_d_km < 0) & (signed_d_km >= -near_field_km)
    east_near = keep & in_section & (signed_d_km > 0) & (signed_d_km <= near_field_km)
    assert west_near.sum() >= 20 and east_near.sum() >= 20, f"near-field masks too small ({west_near.sum()}, {east_near.sum()})"
    out = dict(gnpz=gnpz, G=G, ref_medium=ref_medium, impulse_order=impulse_order, tents=tents, template=template,
               points=points, signed_d_km=signed_d_km, keep=keep, near_trace_cut_km=float(near_trace_cut_km),
               sample_km_pp=sample_km_pp, in_section=in_section, west_near=west_near, east_near=east_near,
               near_field_km=float(near_field_km), to_mesh=to_mesh, surface=None, G_surface=None)
    if crossings is not None:                                              # the measured-crossing surface constraint (sensitivity only)
        assert trace_south_north is not None, "the surface constraint needs the trace (south to north)"
        in_window = (crossings["km"] >= km_window[0]) & (crossings["km"] <= km_window[1])

        def trace_lon_of_lat(lat_values):
            return np.interp(lat_values, trace_south_north[:, 1], trace_south_north[:, 0])

        def lonlat_to_mesh_xy(lon_values, lat_values):
            return to_mesh.transform(lon_values, lat_values)

        constraint = crossing_constraint(crossings["km"][in_window], crossings["lat"][in_window], crossings["mean_mm"][in_window],
                                         trace_lon_of_lat, lonlat_to_mesh_xy, tents.imp_xyz[tents.top_order, :2], tents.top_order,
                                         tents.n_imp, floor_mm=surf_floor_mm)
        out.update(constraint=constraint, surface=surface_slip_data(constraint, surf_floor_mm, frame_lonlat), G_surface=constraint["G"])
    return out


def tent_numbers(slip_mm, insar_data, tents, keep, west_near, east_near, mu_ref, mu_per_triangle=None):
    """The numbers of one tent solution (the metric policy of 2026-10-28; no chi2/N anywhere): RMS of the residual
    (plane-removed data minus model) over the FITTED samples (`keep`) and over the near-field masks west / east;
    potency (exact tent integral, m^3); Mw at mu_ref (Pa) and with the per-triangle rigidity mu_per_triangle (Pa; NaN
    if None), mu_effective = M0 / potency (GPa); peak slip (mm) and its depth (m); the depths above which 50 % and
    90 % of the potency lie (m, TentFault.potency_depth_m). Mw: Hanks & Kanamori (1979). Written for this project."""
    keep = np.asarray(keep, bool)
    assert not np.any(np.asarray(west_near, bool) & ~keep) and not np.any(np.asarray(east_near, bool) & ~keep), \
        "a near-field sample is not a fitted sample"
    res = insar_data.vel - insar_data.synth
    pot = tents.potency_m3(slip_mm)
    if mu_per_triangle is None:
        m0_sfcvm = np.nan
        mw_sfcvm = np.nan
    else:
        m0_sfcvm = tents.moment_nm(slip_mm, mu_per_triangle)
        mw_sfcvm = float(moment_magnitude(m0_sfcvm))
    return dict(rms_mm=float(np.sqrt(np.mean(res[keep] ** 2))), rms_west_mm=float(np.sqrt(np.mean(res[west_near] ** 2))),
                rms_east_mm=float(np.sqrt(np.mean(res[east_near] ** 2))), n_fitted=int(keep.sum()), potency_m3=pot,
                Mw_30gpa=float(moment_magnitude(mu_ref * pot)), Mw_sfcvm=mw_sfcvm, mu_effective_gpa=m0_sfcvm / pot / 1e9,
                peak_slip_mm=float(slip_mm.max()), peak_depth_m=float(tents.imp_depth_km[np.argmax(slip_mm)] * 1e3),
                depth50_m=tents.potency_depth_m(slip_mm, 0.5), depth90_m=tents.potency_depth_m(slip_mm, 0.9))


def tent_lcurve(solve, sigma_grid, tents, mu_ref, label, keep):
    """One L-curve over sigma_grid: for each sigma the solution `solve(sigma)` -> (fault, insar data, surface), the
    weighted misfit chi2 of the FITTED samples (`keep`; the L-curve's x axis), the prior-weighted roughness
    sqrt(m^T (Cm / sigma^2)^-1 m) (its y axis), the RMS of the fitted samples, peak slip, Mw at mu_ref and the solve
    time (s). (Until 2026-10-30 the RMS included the samples of the near-trace cut.) Source: 08a [11] `lcurve`
    (Hansen 1992; written for this project)."""
    keep = np.asarray(keep, bool)
    rows = []
    for sigma in sigma_grid:
        t0 = time.time()
        fault_model, insar_data, _ = solve(sigma)
        res = (insar_data.vel - insar_data.synth)[keep]
        Cd_fit = insar_data.Cd[np.ix_(keep, keep)]
        slip_vec = fault_model.slip[:, 0]
        rows.append(dict(sigma=float(sigma), chi2_fitted=float(np.dot(res, np.linalg.solve(Cd_fit, res))),
                         roughness=float(np.sqrt(np.dot(slip_vec, np.linalg.solve(fault_model.Cm / sigma ** 2, slip_vec)))),
                         rms_mm=float(np.sqrt(np.mean(res ** 2))), peak_slip_mm=float(slip_vec.max()),
                         Mw_30gpa=float(moment_magnitude(mu_ref * tents.potency_m3(slip_vec))), seconds=time.time() - t0))
        print(f"  {label} sigma {sigma:>7.3f}: chi2 {rows[-1]['chi2_fitted']:.0f}, roughness {rows[-1]['roughness']:.1f}, "
              f"RMS {rows[-1]['rms_mm']:.3f} mm, Mw30 {rows[-1]['Mw_30gpa']:.3f} ({rows[-1]['seconds']:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def own_corner(solve, base_grid, refine_factors, tents, mu_ref, label, keep):
    """A model's own L-curve corner (register #16), in two steps on ONE declared rule: (1) the L-curve on base_grid and
    its corner (lcurve_corner: maximum curvature of log sqrt(chi2) vs log roughness); (2) the L-curve on base_grid
    plus base corner x refine_factors (new sigmas only) and its corner = the model's sigma, unsnapped.
    Returns (corner, base_corner, table of every solve, flagged `step` 1 or 2). Written for this project (2026-10-30)."""
    base = tent_lcurve(solve, base_grid, tents, mu_ref, f"{label} [base]", keep)
    base_corner, _, _ = lcurve_corner(base.sigma.values, np.sqrt(base.chi2_fitted.values), base.roughness.values)
    extra = [base_corner * f for f in refine_factors if np.min(np.abs(np.asarray(base_grid) / (base_corner * f) - 1)) > 1e-6]
    added = tent_lcurve(solve, extra, tents, mu_ref, f"{label} [refine]", keep)
    table = pd.concat([base.assign(step=1), added.assign(step=2)]).sort_values("sigma").reset_index(drop=True)
    corner, _, _ = lcurve_corner(table.sigma.values, np.sqrt(table.chi2_fitted.values), table.roughness.values)
    assert table.sigma.values[1] < corner < table.sigma.values[-2], f"{label}: corner {corner:.2f} at the edge of the grid"
    print(f"{label}: L-curve corner {base_corner:.3f} on the base grid, {corner:.3f} after refining", flush=True)
    return float(corner), float(base_corner), table


def uniform_slip_shape(tents, section_km, taper_km, depth_km, top_km=0.0):
    """The slip pattern of the fixed-depth scan (register #19, #22): 1 on every impulse vertex between top_km and depth_km
    deep (km) inside the SSE section, tapered linearly to 0 over taper_km at each end of the section (as 07b [9]), 0
    elsewhere. With the tents this is full slip down to D, falling to zero at the next vertex row (one mesh cell).
    Source: depth_scan_2026-10-28/depth_scan.py (written for this project)."""
    full_lo, full_hi = section_km[0] + taper_km, section_km[1] - taper_km
    along = np.clip(np.minimum(tents.imp_km - full_lo, full_hi - tents.imp_km) / taper_km + 1, 0, 1)
    return along * (tents.imp_depth_km <= depth_km + 1e-6) * (tents.imp_depth_km >= top_km - 1e-6)


def uniform_slip_fit(G, shape, data_mm, x_km, y_km, keep):
    """One slip amplitude (mm) for a fixed slip pattern `shape` plus a plane (1, x, y), by least squares over the fitted
    samples (`keep`) with equal weights (the locked weighting, register #6: every sample the nugget variance, so the
    weights cancel). Returns (amplitude_mm, residual_mm = data minus the fitted pattern and plane, at every sample).
    Source: depth_scan_2026-10-28/depth_scan.py (numpy least squares; written for this project)."""
    keep = np.asarray(keep, bool)
    design = np.c_[G @ shape, np.ones(data_mm.size), x_km, y_km]
    coef, *_ = np.linalg.lstsq(design[keep], data_mm[keep], rcond=None)
    return float(coef[0]), data_mm - design @ coef


def profile_samples(lon, lat, line_ends_xy, epsg, half_width_m=300.0):
    """The samples within half_width_m of a profile line (06's `prof_fp_<line>_ends`: x0, y0, x1, y1 in EPSG `epsg`, m)
    and their distance along it (m from its first end). Returns (selected mask, along_m for every sample).
    Source: scripts/model_plots/plot_run_summary.profile_selection (written for this project)."""
    import pyproj
    to_field = pyproj.Transformer.from_crs(4326, int(epsg), always_xy=True)
    xw, yw, xe, ye = np.asarray(line_ends_xy, float)
    unit = np.array([xe - xw, ye - yw])
    length = float(np.linalg.norm(unit))
    unit /= length
    sx, sy = to_field.transform(np.asarray(lon, float), np.asarray(lat, float))
    along = (sx - xw) * unit[0] + (sy - yw) * unit[1]
    across = -(sx - xw) * unit[1] + (sy - yw) * unit[0]
    return (np.abs(across) <= half_width_m) & (along >= 0) & (along <= length), along


def correlated_noise(Cd, n_draws, seed):
    """n_draws realisations (n_draws, n) of zero-mean Gaussian noise with covariance Cd (mm^2): Cholesky factor L of
    Cd, noise = L z, z ~ N(0, 1) from numpy's default_rng(seed). Written for this project (Monte Carlo, register #20)."""
    L = np.linalg.cholesky(Cd)
    z = np.random.default_rng(seed).standard_normal((n_draws, Cd.shape[0]))
    return z @ L.T


def materna_stress_drop_pa(mu_pa, slip_m, half_width_m):
    """Stress drop of a slipping patch (Pa): delta_tau = mu d / x, d the slip, x half the patch width (register #19:
    Materna et al. 2024, SI Eq. 2, applied with d = the best uniform slip and x = half the best depth of the
    fixed-depth scan). Written for this project; the formula as the register states it."""
    assert mu_pa > 0 and slip_m > 0 and half_width_m > 0
    return float(mu_pa * slip_m / half_width_m)


# =====================================================================================================================
# 9. PyLith forward runs, materials and the stress drop
# =====================================================================================================================
def read_forward_points(points_h5, points_xy):
    """3D displacement (mm; last time step) of a PyLith run at the given points (m, mesh CRS), matched by coordinates
    (< 1 mm asserted). Source: 08a [9] `read_forward` (PyLith OutputSolnPoints output; written for this project)."""
    with h5py.File(points_h5, "r") as h5:
        dist, order = cKDTree(h5["geometry/vertices"][:, :2]).query(points_xy)
        assert dist.max() < 1e-3, f"{os.path.basename(points_h5)}: output points are not the requested points"
        return h5["vertex_fields/displacement"][-1][order] * 1e3


def read_forward_slip(fault_h5, sign_rl):
    """Fault vertices (m), cells and the right-lateral slip (mm, = sign_rl x the left-lateral component, the sign
    build_G derived) of a PyLith FaultCohesiveKin run. Source: 08a [9] `read_forward` (written for this project)."""
    with h5py.File(fault_h5, "r") as h5:
        f_xyz = h5["geometry/vertices"][:]
        f_cells = np.asarray(h5["viz/topology/cells"][:], int)
        slip_rl = float(sign_rl) * h5["vertex_fields/slip"][-1][:, 1] * 1e3
    return f_xyz, f_cells, slip_rl


def superposition_check(G, imp_xyz, points_h5, fault_h5, points_xy, look, sign_rl):
    """G x (a forward run's own vertex slip) against that run's fault-parallel displacement at the samples: exact
    (< 1e-4 mm) when G and the forward run share mesh, medium and element order. Returns the max / rms difference
    and the rms signal (mm). Source: 08a [9] (c) and the driver's `superposition` (written for this project)."""
    u_fp = read_forward_points(points_h5, points_xy).dot(look)
    f_xyz, _, slip_rl = read_forward_slip(fault_h5, sign_rl)
    vertex_dist, vertex_match = cKDTree(f_xyz).query(imp_xyz)
    assert vertex_dist.max() < 1e-6, "forward-run fault vertices differ from the GF fault vertices"
    assert np.setdiff1d(np.nonzero(slip_rl > 1e-6)[0], vertex_match).size == 0, "forward slip on vertices without an impulse"
    superposed = G.dot(slip_rl[vertex_match])
    diff = superposed - u_fp
    return dict(max_abs_diff_mm=float(np.abs(diff).max()), rms_diff_mm=float(np.sqrt(np.mean(diff ** 2))),
                rms_signal_mm=float(np.sqrt(np.mean(superposed ** 2))))


def shear_modulus_pa(density_kg_m3, vs_m_s):
    """Shear modulus mu = rho vs^2 (Pa)."""
    return density_kg_m3 * vs_m_s ** 2


def mu_beside_fault(elastic_info_h5, grid_npz, tents, label):
    """The rigidity PyLith actually used beside the fault. Reads the per-cell shear modulus of a run
    (`<run>-elastic_info.h5`; materials basis order 0, or the cell mean of the vertex values for order 1), compares
    it with the grid it was read from (`<grid>.npz`, mu = rho vs^2, linear interpolation at the cell centroids of the
    fine grid box), and takes the nearest cell on each side of every fault triangle. Returns (mean of the two sides,
    west, east) per triangle (Pa) and the median |relative difference| from the grid.
    Source: 08a [6] and the driver's [5b] (written for this project)."""
    with h5py.File(elastic_info_h5, "r") as elastic_h5:
        cells = np.asarray(elastic_h5["viz/topology/cells"][:], int)
        cell_xyz = elastic_h5["geometry/vertices"][:][cells].mean(1)
        if "cell_fields" in elastic_h5 and "shear_modulus" in elastic_h5["cell_fields"]:
            cell_mu = elastic_h5["cell_fields/shear_modulus"][0, :, 0]                      # materials basis order 0
        else:
            cell_mu = elastic_h5["vertex_fields/shear_modulus"][0, :, 0][cells].mean(1)     # basis order 1: cell mean of the vertex values
    grid_data = np.load(grid_npz)
    grid_mu_values = shear_modulus_pa(grid_data["density"].astype(float), grid_data["vs"].astype(float))
    grid_mu = RegularGridInterpolator((grid_data["x"], grid_data["y"], grid_data["z"]), grid_mu_values)
    fine = (cell_xyz[:, 0] > -3500) & (cell_xyz[:, 0] < 2500) & (cell_xyz[:, 1] > -4000) & (cell_xyz[:, 1] < 3600) & (cell_xyz[:, 2] > -1500)
    rel = cell_mu[fine] / grid_mu(cell_xyz[fine]) - 1.0
    print(f"{label}: PyLith cell mu vs grid at centroids ({fine.sum()} cells in the fine box, top 1.5 km): median |rel diff| "
          f"{np.median(np.abs(rel)):.3f}, 90th pct {np.percentile(np.abs(rel), 90):.3f}; PyLith mu range "
          f"{cell_mu.min() / 1e9:.3f}-{cell_mu.max() / 1e9:.1f} GPa")
    assert cell_mu.min() >= 0.99 * float(np.min(grid_mu_values)), "mu below the floor"
    _, near = cKDTree(cell_xyz).query(tents.tri_cen, k=30)
    rel_xyz = cell_xyz[near] - tents.tri_cen[:, None, :]
    offset = np.einsum("tkc,tc->tk", rel_xyz, tents.tri_n)
    assert np.all((offset > 0).any(1)) and np.all((offset < 0).any(1)), "a fault triangle lacks a cell on one side"
    dist = np.linalg.norm(rel_xyz, axis=2)
    east_i = near[np.arange(len(near)), np.argmin(np.where(offset > 0, dist, np.inf), axis=1)]
    west_i = near[np.arange(len(near)), np.argmin(np.where(offset < 0, dist, np.inf), axis=1)]
    shallow = tents.tri_cen[:, 2] > -100
    print(f"{label}: fault-adjacent mu, top 100 m median: west {np.median(cell_mu[west_i][shallow]) / 1e9:.3f}, "
          f"east {np.median(cell_mu[east_i][shallow]) / 1e9:.3f} GPa")
    return 0.5 * (cell_mu[east_i] + cell_mu[west_i]), cell_mu[west_i], cell_mu[east_i], float(np.median(np.abs(rel)))


def harmonic_mean_rigidity(mu_a, mu_b):
    """Effective rigidity of a fault on a material interface: 2 mu_a mu_b / (mu_a + mu_b) (Vavrycuk 2013, GJI 194,
    395-400, as cited in 08a [15c])."""
    return 2.0 * mu_a * mu_b / (mu_a + mu_b)


def screw_dislocation_mm(slip_mm, locking_depth_m, x_m):
    """Surface displacement (mm) of a 2D screw dislocation, slip s from the surface to depth D:
    u(x) = (s / pi) arctan(D / x), x the signed distance from the fault (m). Source: 08a [8] (the standard 2D
    screw-dislocation solution; written for this project)."""
    return (slip_mm / np.pi) * np.arctan(locking_depth_m / x_m)


def median_edge_length_m(triangles_xyz):
    """Median length (m) of all the edges of a set of triangles (n, 3 vertices, xyz). Source: 08a [8] (written for this project)."""
    edge_01 = np.linalg.norm(triangles_xyz[:, 1] - triangles_xyz[:, 0], axis=1)
    edge_12 = np.linalg.norm(triangles_xyz[:, 2] - triangles_xyz[:, 1], axis=1)
    edge_20 = np.linalg.norm(triangles_xyz[:, 0] - triangles_xyz[:, 2], axis=1)
    return float(np.median(np.concatenate([edge_01, edge_12, edge_20])))


def vertex_areas_m2(vertices_xyz, cells):
    """Area (m^2) attached to every fault vertex: one third of the area of each triangle it belongs to.
    Source: 08a [15b] (written for this project)."""
    triangles = vertices_xyz[cells]
    triangle_area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    areas = np.zeros(len(vertices_xyz))
    np.add.at(areas, cells.ravel(), np.repeat(triangle_area / 3.0, 3))
    return areas


def slip_weighted_stress_drop_pa(dtau_pa, slip_m, vertex_area_m2):
    """Stress drop of a distributed slip model (Pa): sum(dtau s A) / sum(s A) over the fault vertices (Noda et al., 2013),
    negative slip counted as zero. dtau = PyLith's shear-traction change (a right-lateral drop is a positive
    left-lateral traction change, asserted). Source: 08a [15b]."""
    weight = np.clip(slip_m, 0.0, None) * vertex_area_m2
    assert np.sum(dtau_pa * weight) > 0, "right-lateral slip did not lower the right-lateral shear traction: sign convention broken"
    return float(np.sum(dtau_pa * weight) / np.sum(weight))


def potency_extent(sub_potency_m3, sub_depth_m, sub_km, fraction=0.9):
    """Depth extent w (m) and along-strike length L (km) that hold `fraction` of a slip model's potency: w = the depth
    above which that share of the potency lies; L = the span between the (1 - fraction)/2 and (1 + fraction)/2 potency
    quantiles along strike. Inputs are the sub-triangle potencies, depths and along-strike km of 08a [15e].
    Source: written for this project (2026-10-25)."""
    p = np.asarray(sub_potency_m3, float).ravel()
    assert (p >= 0).all() and p.sum() > 0, "potency must be non-negative with a positive total"

    def quantile(values, q):
        order = np.argsort(values)
        cum = np.cumsum(p[order]) / p.sum()
        return float(np.asarray(values).ravel()[order][np.searchsorted(cum, q)])
    depth, km = np.asarray(sub_depth_m, float).ravel(), np.asarray(sub_km, float).ravel()
    w_m = quantile(depth, fraction)
    l_km = quantile(km, 0.5 + fraction / 2) - quantile(km, 0.5 - fraction / 2)
    assert w_m > 0 and l_km > 0
    return w_m, l_km


# =====================================================================================================================
# 10. Provenance and running PyLith
# =====================================================================================================================
def field_fingerprint(fp_maps_npz):
    """The fingerprint of 06's fault-parallel field that a model fits, in the form every model product carries
    ("fp_maps.npz:<sha256 12>"; insar_utils.file_fingerprint). Never parsed as a number. Source: the driver (2026-10-05)."""
    return f"fp_maps.npz:{utils.file_fingerprint(fp_maps_npz)}"


def run_logged(cmd, cwd, env_extra=None):
    """Run a shell command in cwd, echo it, fail loudly (assert on the return code); returns the wall time (s).
    Source: 08a [1] `run_shell` (written for this project)."""
    print("$", cmd)
    t0 = time.time()
    env = dict(os.environ, **(env_extra or {}))
    out = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
    assert out.returncode == 0, f"command failed ({out.returncode}):\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    print(out.stdout[-1500:])
    return time.time() - t0


def ensure_product(path, recipe, cwd, allow_run, timings=None):
    """A product of the PyLith side (mesh, database, run output). It must exist; with allow_run (TULEPOND_RUN_PYLITH=1)
    a missing one is made by the shell command `recipe` run in cwd (the PyLith work directory, data_dir), else the
    notebook stops with the command that makes it. Returns path. Source: 08a [1] `cached` (written for this project)."""
    if not os.path.exists(path):
        assert allow_run, (f"missing PyLith product {path}\n  made once with (from {cwd}): {recipe}\n  set TULEPOND_RUN_PYLITH=1 "
                           f"to run it here (needs the PyLith 5.0.2 bundle, see pylith/README.md)")
        seconds = run_logged(recipe, cwd)
        if timings is not None:
            timings[os.path.basename(path)] = seconds
        assert os.path.exists(path), f"{recipe} did not write {path}"
    return path


def crossing_steps_like(values_mm, signed_d_km, sample_km_pp, crossing_km, inner_km=0.03, outer_km=0.15,
                        half_along_km=0.2, min_per_side=3):
    """The cross-fault step of a field at the samples, measured the way 03/04 measure the crossings (2026-10-16): for each
    crossing km, the median of the samples inner_km to outer_km east of the trace and within +/- half_along_km along
    strike, minus the same median west. NaN where a side has fewer than min_per_side samples. Used to compare the model
    with the measured crossings like for like (the model's top-edge slip is not what the crossings measure)."""
    values_mm = np.asarray(values_mm, float)
    d = np.asarray(signed_d_km, float)
    s_km = np.asarray(sample_km_pp, float)
    steps = np.full(np.asarray(crossing_km).size, np.nan)
    for i, km_c in enumerate(np.asarray(crossing_km, float)):
        along = np.abs(s_km - km_c) <= half_along_km
        east = along & (d >= inner_km) & (d <= outer_km) & np.isfinite(values_mm)
        west = along & (d <= -inner_km) & (d >= -outer_km) & np.isfinite(values_mm)
        if east.sum() >= min_per_side and west.sum() >= min_per_side:
            steps[i] = np.median(values_mm[east]) - np.median(values_mm[west])
    return steps
