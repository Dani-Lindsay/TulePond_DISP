#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

model_utils.py: the modelling functions of the Tule Pond slip models (notebooks 07 to 10, 14 and
scripts/sensitivity/*.py). TulePond_filepaths.py holds the paths and shared constants, model_config.py the settings
of the slip models, insar_utils.py the InSAR helpers, and the numbered notebooks the order of the analysis. Each
function states what it does, its inputs and outputs with units, and its source ("written for this project" where
it follows no published code). No function reads a path on its own: every file is passed as an argument.

Contents:
  1. Along-trace distance and the fault trace          km_from_pinole, model_trace, signed_distance_m
  2. The data: fault-parallel field, noise, quadtree   look_vector, decompose_fault_parallel, los_from_steps,
                                                       fault_parallel_from_steps, dist_to_trace, fault_parallel_dataset,
                                                       semivariogram_model, covariance, downsample, covariance_matrix,
                                                       build_cd, quadtree_block_corners, write_points_list,
                                                       samples_from_arrays
  3. Magnitudes and the L-curve                        moment_magnitude, lcurve_corner
  4. Surface constraint (sensitivity tests only)       arc_length_on_polyline, local_scatter_mm, tent_rows,
                                                       crossing_constraint, surface_slip_data
  5. PyLith Green's functions -> G                     read_gf_run, trace_top, build_G, load_G
  6. Tent (vertex) parameterisation and inversion      TentFault, tent_fault_template, data_cd, invert_tents,
                                                       match_impulse_order, analytic_tent_G,
                                                       positions_fingerprint, analytic_tent_G_cached,
                                                       tent_model_setup, tent_numbers, tent_lcurve, own_corner,
                                                       uniform_slip_shape, uniform_slip_fit, profile_samples, side_deflection_mm,
                                                       correlated_noise, strike_slip_stress_drop_pa
  7. PyLith materials, rigidity, extent of slip        shear_modulus_pa, mu_beside_fault, harmonic_mean_rigidity,
                                                       potency_extent
  8. Provenance                                        field_fingerprint
  9. Cross-fault steps at the crossings                crossing_steps_like
The settings of the paper's models are in model_config.py.
"""
import copy
import os
import time

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

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
    """Along-strike distance (km from Point Pinole) of each latitude (deg): the distance of the nearest vertex of the
    fault-profile trace (hayward_fault_profiles.csv, vertices 0.1 km apart, km = (n - 1 - index) x 0.1;
    `profile_lats` = its ft_lat column, deg). Processed in chunks of 20000 latitudes to limit memory.
    Returns an array of km. Source: written for this project (the 100 m index rule is tested in tests/test_filepaths.py)."""
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
    """The mapped trace of the slip models: the KML trace (lon, lat rows, deg) between lat_lo and lat_hi, sorted north
    to south, discretised every `every_km` (km) by CSI (`TriangularPatches.trace` + `discretize2`), with its arc length
    and distance from Point Pinole. frame_lonlat = (lon0, lat0) of the CSI local frame; profile_lats as km_from_pinole.
    Returns a dict: lonlat (n, 2), ref_fault (the CSI object: xi, yi in km of the CSI frame, loni, lati), x_km, y_km,
    arc_s (km), arc_total (km), km_pp (km from Point Pinole of every discretised point), every_km.
    Source: CSI (Jolivet et al.), following the CSI tutorial `tutoFaultMeshing`."""
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
    """Signed perpendicular distance (m, east positive) of points (px, py) from a densely sampled polyline
    (line_x, line_y), all in m of a projected CRS: distance to the nearest polyline sample, sign from the local
    tangent. Returns an array of m. (dist_to_trace gives the unsigned distance.) Source: written for this project."""
    tangent = np.c_[np.gradient(line_x), np.gradient(line_y)]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    dist, i = cKDTree(np.c_[line_x, line_y]).query(np.c_[px, py])
    normal = np.c_[tangent[i, 1], -tangent[i, 0]]
    normal[normal[:, 0] < 0] *= -1.0                                  # east +
    return np.einsum("ij,ij->i", np.c_[px - line_x[i], py - line_y[i]], normal)


# =====================================================================================================================
# 2. The data: the fault-parallel mean event, its noise model and the quadtree samples
# =====================================================================================================================
def look_vector(strike_deg):
    """Horizontal unit vector (east, north, up) along the strike bearing (deg clockwise from north): the 'look
    vector' of the fault-parallel field, with which right-lateral motion of the east side is positive.
    Source: written for this project."""
    return np.array([np.sin(np.deg2rad(strike_deg)), np.cos(np.deg2rad(strike_deg)), 0.0])


def decompose_fault_parallel(los_asc, los_desc, inc_both, azi_both, strike_deg, window_px):
    """Fault-parallel and vertical maps (mm) from the ascending and descending LOS maps, with MintPy's
    `asc_desc2horz_vert` (horizontal direction = the strike, horz_az_angle = -strike as MintPy counts anticlockwise;
    only pixels valid on both tracks; decomposition window window_px pixels). inc_both / azi_both: (2, nr, nc),
    ascending first. Returns (fault_parallel, up), mm.
    Source: MintPy asc_desc2horz_vert (Yunjun et al. 2019), called with the same arguments as notebook 06."""
    from mintpy.asc_desc2horz_vert import asc_desc2horz_vert
    valid_both = np.isfinite(los_asc) & np.isfinite(los_desc)
    los_stack = np.stack([los_asc, los_desc]).astype(np.float32)
    los_stack[:, ~valid_both] = np.nan
    fault_parallel, up = asc_desc2horz_vert(los_stack, inc_both, azi_both, horz_az_angle=-strike_deg, step=window_px)
    return fault_parallel.astype(float), up.astype(float)


def los_from_steps(steps_h5, mask_files=None):
    """Notebook 06's per-track LOS mean event (mm; MintPy convention, positive toward the satellite), with the incidence
    and azimuth angles (deg) of the box, from its saved `tulepond_steps_<track>.h5`, optionally masked at another temporal
    coherence (mask_files: track -> MintPy maskTempCoh file, cut to the box from the X_FIRST / Y_FIRST attributes, as
    notebook 06 does). steps_h5: track -> path, keys 'ascending' and 'descending'. Returns track -> dict(mean, inc, azi)
    on 06's box grid. Source: written for this project."""
    out = {}
    for track in ("ascending", "descending"):
        with h5py.File(steps_h5[track], "r") as h5:
            mean = h5["mean_event"][:].astype(float)
            inc = h5["incidenceAngle"][:]
            azi = h5["azimuthAngle"][:]
            box = dict(x_first=float(h5.attrs["X_FIRST"]), y_first=float(h5.attrs["Y_FIRST"]), nr=int(h5.attrs["LENGTH"]),
                       nc=int(h5.attrs["WIDTH"]), dx=float(h5.attrs["X_STEP"]), dy=float(h5.attrs["Y_STEP"]))
        if mask_files is not None:
            with h5py.File(mask_files[track], "r") as mask_h5:
                col_start = int(round((box["x_first"] - float(mask_h5.attrs["X_FIRST"])) / box["dx"]))
                row_start = int(round((box["y_first"] - float(mask_h5.attrs["Y_FIRST"])) / box["dy"]))
                mask_name = [name for name in mask_h5.keys() if name != "date"][0]
                mask = mask_h5[mask_name][row_start:row_start + box["nr"], col_start:col_start + box["nc"]].astype(bool)
            mean = np.where(mask, mean, np.nan)
        out[track] = dict(mean=mean, inc=inc, azi=azi)
    return out


def fault_parallel_from_steps(steps_h5, strike_deg, window_px, mask_files=None):
    """The fault-parallel mean event recomputed from notebook 06's saved per-track products (los_from_steps: the
    per-pixel mean of the kept events at the time-series coherence, with the incidence and azimuth of the box), optionally
    masked at another temporal coherence, and decomposed with decompose_fault_parallel (strike_deg in deg, window_px in
    pixels). Returns the fault-parallel map (mm) on 06's box grid.
    With the coherence-0.9 masks the result equals 06's fp_maps.npz `fp_event` (asserted in 07); without masks it is
    the field at coherence 0.7, the mask of the time-series fit (the coherence-0.7 sensitivity sample set).
    Source: written for this project."""
    tracks = los_from_steps(steps_h5, mask_files)
    inc_both = np.stack([tracks["ascending"]["inc"], tracks["descending"]["inc"]]).astype(np.float32)
    azi_both = np.stack([tracks["ascending"]["azi"], tracks["descending"]["azi"]]).astype(np.float32)
    fault_parallel, _ = decompose_fault_parallel(tracks["ascending"]["mean"], tracks["descending"]["mean"], inc_both, azi_both,
                                                 strike_deg, window_px)
    return fault_parallel


def dist_to_trace(sar, fault):
    """Horizontal distance (km) from each sample of a CSI data set `sar` to the nearest point of the fault's
    discretised trace (fault.xf, fault.yf), both in km of the CSI local frame. Source: written for this project."""
    trace_tree = cKDTree(np.c_[fault.xf, fault.yf])
    dist, _ = trace_tree.query(np.c_[sar.x, sar.y])
    return dist


def fault_parallel_dataset(fp_event, lon, lat, strike_deg, fault, frame_lonlat, swath_km, exclude_km, km_window,
                           profile_lats):
    """The fault-parallel field (mm) as one csi.insar data set whose look vector is the horizontal unit vector along
    the strike bearing: every finite pixel of fp_event, cut to swath_km of the trace (and beyond exclude_km of it) and
    to the along-strike km_window (km from Point Pinole). No plane or offset is removed from the field here; the
    reference frame is solved for in the inversion. Returns (sar, look).
    Source: written for this project (CSI insar.read_from_binary; Jolivet et al.)."""
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
    gamma(h) = n2 + s1 (1 - exp(-h / l1)) + s2 (1 - exp(-h / l2)), sills in mm^2 and ranges in km.
    Source: written for this project; far-field semivariogram noise model after Lohman and Simons (2005)."""
    short_term = sill_short * (1 - np.exp(-distance_km / range_short_km))
    long_term = sill_long * (1 - np.exp(-distance_km / range_long_km))
    return nugget2 + short_term + long_term


def covariance(sar, fault, swath_km, cov_mask_km, cov_distmax_km, npairs=300000, seed=0, return_bins=False):
    """Fit semivariogram_model to the far field: the pixels beyond cov_mask_km of the trace, a plane (1, x, y) removed;
    npairs random pixel pairs closer than cov_distmax_km, binned every 0.1 km (bins with more than 50 pairs). Returns
    (nugget2, sill_short, range_short_km, sill_long, range_long_km), in mm^2 and km; with return_bins also the binned
    semivariogram as a table (bin centre km, pairs, gamma mm^2, fitted model mm^2) and the number of pixels fitted.
    `sar` must hold pixels well beyond the mask: cov_mask_km <= 0.4 swath_km and cov_distmax_km <= 4 swath_km are
    asserted (the noise fit is therefore made on a wider swath than the model, model_config.noise_swath_km).
    Source: written for this project; semivariogram noise model after Lohman and Simons (2005)."""
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
    """Distance-based quadtree (csi.imagedownsampling): blocks of start_km (km) divided down to min_km (km) toward the
    trace; a regular grid when start_km == min_km. Returns the downsampled csi.insar (its `wgt` = pixels per block,
    `xycorner` the block corners). Source: CSI imagedownsampling (Jolivet et al.); quadtree sampling after Lohman and
    Simons (2005)."""
    sampler = imdown(sar.name, sar, fault, verbose=False)
    sampler.initialstate(start_km, min_km, tolerance=0.1, plot=False)
    if start_km > min_km + 1e-9:
        sampler.distanceBased(chardist=0.0, expodist=1, plot=False)
    return sampler.newimage


def covariance_matrix(x_km, y_km, n_pixels, cov_p, nug_power):
    """The data covariance of quadtree samples (mm^2): Cd_ij = s1 exp(-r_ij / l1) + s2 exp(-r_ij / l2), plus
    nugget2 / n_pixels**nug_power on the diagonal (n_pixels = the pixels each block averages, CSI's `wgt`).
    x_km, y_km: sample positions (km); cov_p = (nugget2, sill_short, range_short_km, sill_long, range_long_km).
    Returns Cd (n, n), mm^2. Source: written for this project; the covariance from the far-field semivariogram follows
    Lohman and Simons (2005) and Sudhaus and Jonsson (2009)."""
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
    Source: written for this project."""
    ds.Cd = covariance_matrix(ds.x, ds.y, ds.wgt, cov_p, nug_power)
    return ds


def quadtree_block_corners(ds):
    """The quadtree blocks as polygons (lon, lat of the four corners in deg, (n_samples, 4) each), for drawing each
    block at its true size. CSI keeps two opposite corners of every block in its local km frame (`xycorner`, one row
    per sample, in sample order); the four corners are converted to lon/lat with the image's xy2ll. Asserts that more
    than 99 % of the samples fall inside their own block. Returns (block_lon, block_lat).
    Source: written for this project."""
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


def write_points_list(stem, lon, lat, crs, header, extra):
    """Write a sample set as a PyLith PointsList (`<stem>.txt`: "name x y z", m in the mesh CRS `crs`, z = 0 on the flat
    free surface) and an npz (`<stem>.npz`: lon, lat, x, y, names, crs and the arrays of `extra`). lon, lat in deg;
    `header` is written as the first comment line. If the set exists already, its sample positions (lon, lat, x, y,
    and the coordinates of the .txt) must be unchanged, because the cached PyLith Green's functions are computed at
    them. Returns (x, y), m.
    Source: written for this project (PointsList format of PyLith OutputSolnPoints; see pylith/README.md)."""
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
    """A quadtree sample set rebuilt as a csi.insar data set from saved arrays (07's model data product or a sample-set
    npz): positions in lon/lat and in CSI's local km frame (set exactly as saved, not recomputed), data (mm), block
    pixel counts, the fault-parallel look vector, and its full covariance (build_cd with cov_p and nug_power).
    Returns the csi.insar data set. Source: written for this project."""
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
# 3. Magnitudes and the L-curve
# =====================================================================================================================
def moment_magnitude(moment_nm):
    """Moment magnitude from the seismic moment M0 in N m: Mw = 2/3 (log10 M0 - 9.1) (Hanks & Kanamori 1979)."""
    return (2.0 / 3.0) * (np.log10(moment_nm) - 9.1)


def lcurve_corner(sigmas, misfit, roughness):
    """The L-curve corner (Hansen 1992). x = log10 misfit (pass sqrt(chi2 total)) and y = log10 roughness as cubic
    splines of log10 sigma; curvature = |x' y'' - y' x''| / (x'^2 + y'^2)^1.5 on 400 points between the second and
    the second-last grid sigma (the spline end intervals are excluded). Returns the sigma of maximum curvature, the
    400 log10 sigmas and the curvature. Source: written for this project, after Hansen (1992)."""
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


# =====================================================================================================================
# 4. Surface constraint of the 3D model (measured crossings tied to the tent slip on the top edge; sensitivity only)
# =====================================================================================================================
def arc_length_on_polyline(px, py, vx, vy):
    """Arc length along the polyline (vx, vy) of the point on it nearest to each (px, py), that distance, and the arc
    length of every polyline vertex, in the units of the inputs (m in the mesh CRS). Source: written for this project."""
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
    """Local scatter (mm) of each crossing: 1.4826 x MAD of the values (mm) of the crossings within +/- half_window_km
    (km) of it, floored at floor_mm (2 x floor_mm with fewer than 3 neighbours). A 1e-6 km tolerance keeps the window
    edges (the crossings lie on a 0.1 km grid). Source: written for this project."""
    err = np.empty(len(km))
    for i, k in enumerate(km):
        nb = values[np.abs(km - k) <= half_window_km + 1e-6]
        err[i] = max(1.4826 * np.median(np.abs(nb - np.median(nb))), floor_mm) if nb.size >= 3 else 2.0 * floor_mm
    return err


def tent_rows(s, s_vert, top_columns, n_param):
    """G rows of the surface constraint: each datum at arc length s is linear between the two bracketing top-edge
    vertices (the order-1 PyLith tents), d = (1 - w) slip_j + w slip_(j+1), w = (s - s_j) / (s_(j+1) - s_j).
    s and s_vert in m; top_columns[i] = the parameter column of the i-th top-edge vertex (in arc-length order); n_param
    = number of model parameters. Returns G (n_data x n_param). Source: written for this project."""
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
    """Surface constraint of the 3D model (used in sensitivity tests only): every finite measured crossing that lies
    within the mesh top edge (ends excluded) is one datum, tied to the modelled slip at its own position on the top
    edge (tent_rows); error = the local scatter of the crossings (local_scatter_mm), floored at floor_mm (mm). The
    crossings are not smoothed or interpolated across data gaps. trace_lon_of_lat(lat) -> lon on the mapped trace;
    to_mesh_xy(lon, lat) -> (x, y) in the mesh CRS (m); top_xy = top-edge vertices (m) in arc-length order; crossings
    farther than max_dist_m (m) from the top edge fail an assertion. Returns a dict: vel, lon, lat, err (mm),
    G (rows x n_param), km, dist_m, n_outside. Source: written for this project."""
    ok = np.isfinite(mean_mm)
    km, lat, mean_mm = np.asarray(km)[ok], np.asarray(lat)[ok], np.asarray(mean_mm)[ok]
    lon = trace_lon_of_lat(lat)
    x, y = to_mesh_xy(lon, lat)
    s, dist, s_vert = arc_length_on_polyline(np.asarray(x, float), np.asarray(y, float), top_xy[:, 0], top_xy[:, 1])
    inside = (s > 1.0) & (s < s_vert[-1] - 1.0)                      # beyond the mesh ends a crossing projects onto an end vertex
    assert inside.sum() >= 10, f"only {inside.sum()} crossings fall within the mesh top edge"
    assert dist[inside].max() < max_dist_m, f"a crossing lies {dist[inside].max():.0f} m from the mesh top edge"
    err = local_scatter_mm(km, mean_mm, floor_mm=floor_mm)[inside]  # scatter computed from all crossings, kept for the inside ones
    G = tent_rows(s[inside], s_vert, np.asarray(top_columns), n_param)
    return dict(vel=mean_mm[inside], lon=lon[inside], lat=lat[inside], err=err, G=G, km=km[inside],
                dist_m=dist[inside], n_outside=int((~inside).sum()))


def surface_slip_data(constraint, floor_mm, frame_lonlat):
    """The crossing constraint (crossing_constraint) as a csi.surfaceslip data set: values (mm), positions, diagonal
    Cd from the errors (mm^2). Source: CSI surfaceslip (Jolivet et al.)."""
    surface_data = surfaceslip(f"crossings f{floor_mm:g}", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    surface_data.read_from_binary(constraint["vel"], constraint["lon"], constraint["lat"], err=constraint["err"])
    surface_data.buildCd()
    return surface_data


# =====================================================================================================================
# 5. PyLith Green's functions -> the fault-parallel G matrix
# =====================================================================================================================
def read_gf_run(pylith_output_dir, name):
    """Read one PyLith GreensFns run `name` from pylith_output_dir. Returns the fault vertices (m) and cells, the
    impulse slip (m; n_impulses, n_vertices, 3: opening, left-lateral, reverse), the output-point coordinates (m) and
    their displacement (m; n_impulses, n_points, 3).
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
    """(y, x) of the fault's top-edge vertices (z = 0), sorted by y: the mesh's own surface trace (m).
    Source: written for this project."""
    top = fault_xyz[np.abs(fault_xyz[:, 2]) < 1e-6]
    order = np.argsort(top[:, 1])
    return top[order, 1], top[order, 0]


def build_G(names, points_npz, strike_deg, pylith_output_dir, gf_dir):
    """Assemble the fault-parallel Green's-function matrix G (n_samples x n_impulses, m of fault-parallel displacement
    per m of right-lateral slip) from PyLith GreensFns runs (several = concatenated along the impulse axis) and write
    gf_dir/G_<names[0]>.npz. The strike is explicit: G is the 3D displacement projected on the horizontal unit vector
    along strike_deg (deg; the fault-parallel look vector of 07), so a G built at another strike must be rebuilt
    (load_G). points_npz: the sample set of write_points_list. Matching (each step asserted): impulse -> fault vertex
    (exactly one vertex with unit left-lateral slip); output point -> sample (nearest neighbour in the mesh CRS,
    < 1 mm); sign: the right-lateral sign relative to PyLith's left-lateral component is derived from the output
    (uniform slip over the impulse region and every surface impulse must move the east side along +look, as in the
    data). Returns the path of the written npz.
    Source: the reading pattern of PyLith examples/subduction-3d/invert_slip.py (readGreens: fault geometry/vertices,
    viz/topology/cells, vertex_fields/slip[:, :, 1]; responses vertex_fields/displacement) and its matchCoords step."""
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
    G_ll = np.hstack(columns)                              # m of fault-parallel displacement per m of left-lateral slip
    vertex_index = np.concatenate(vertex_index)
    assert np.unique(vertex_index).size == vertex_index.size, "a fault vertex received two impulses"
    # sign: right-lateral slip moves the east side along +look (the data convention of 07)
    # (a) uniform left-lateral slip over the whole impulse region (sum of all tents): east-minus-west step of the
    #     samples within 500 m of the trace along the region; (b) every surface-vertex impulse individually (a buried
    #     tent's fault-parallel field has lobes of both signs, so it is not used for this test)
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
    """The G matrix of one GreensFns run (gf_dir/G_<name>.npz), built with build_G from the cached PyLith outputs when
    it is missing or was projected on a strike other than strike_deg (deg); PyLith itself is not run. Asserts the
    strike and that G is on the points of points_npz. Returns the loaded npz. Source: written for this project."""
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
# 6. The tent (vertex) parameterisation of the PyLith fault, and the inversion
# =====================================================================================================================
class TentFault:
    """The PyLith fault of a G matrix (the npz of build_G) as used by the inversion: the fault vertices and triangles
    (m, z up), the impulse vertices (= the columns of G, order-1 tents), each triangle's centroid, unit normal (east
    positive) and area (m^2), the top-edge vertices and the km from Point Pinole of every impulse vertex and triangle.
    to_mesh: pyproj transformer lon/lat -> mesh CRS; profile_lats as km_from_pinole. Built once per mesh.
    Source: written for this project."""

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
        # along-strike km of every vertex and triangle: the km of the top-edge point it lies below. A dipping fault (the
        # pylith/build/generate_gmsh_dip.py meshes: the trace shifted by depth / tan(dip) along one horizontal normal) is
        # first unrolled: xy - depth * w, w = n_h n_z / |n_h|^2 from the median unit normal n of the fault triangles
        # (the horizontal shift per metre of depth). w = 0 for a vertical fault (|n_z| < 1e-3).
        median_normal = np.median(tri_n, axis=0)
        horizontal = median_normal[:2]
        self.shift_per_depth = np.zeros(2) if abs(median_normal[2]) < 1e-3 else horizontal * median_normal[2] / np.dot(horizontal, horizontal)

        def km_below(xyz):
            unrolled_y = xyz[:, 1] - (-xyz[:, 2]) * self.shift_per_depth[1]
            return np.interp(unrolled_y, self.top[:, 1], top_km)
        self.imp_km = km_below(self.imp_xyz)                                       # km from Point Pinole of every impulse vertex
        self.tri_km = km_below(self.tri_cen)                                       # ... and of every triangle centroid
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
        """Slip (mm) on every PyLith fault vertex: the impulse vertices carry slip_mm, the vertices without an
        impulse zero."""
        slip_all = np.zeros(len(self.fault_xyz))
        slip_all[self.vertex_index] = slip_mm
        return slip_all

    def potency_per_triangle_m3(self, slip_mm):
        """Exact integral of the piecewise-linear (tent) slip over each PyLith fault triangle: area x mean of its three
        vertex slips (m^3; slip in mm, vertices without an impulse = 0)."""
        return self.tri_area * self.vertex_slip(slip_mm)[self.fault_cells].mean(1) * 1e-3

    def potency_m3(self, slip_mm):
        """Potency (m^3) of the tent slip: sum over triangles of area x mean vertex slip."""
        slip_all = self.vertex_slip(slip_mm)
        return float(np.sum(self.tri_area * slip_all[self.fault_cells].mean(1)) * 1e-3)

    def potency_depth_m(self, slip_mm, fraction):
        """The depth (m) above which `fraction` of the potency lies (potency_extent on the triangles: exact tent
        potency per triangle, triangle centroid depth)."""
        pot_tri = self.potency_per_triangle_m3(slip_mm)
        assert abs(pot_tri.sum() - self.potency_m3(slip_mm)) <= 1e-9 * max(1.0, self.potency_m3(slip_mm))
        depth_m, _ = potency_extent(pot_tri, self.tri_depth_m, self.tri_km, fraction)
        return depth_m

    def moment_nm(self, slip_mm, mu_per_triangle):
        """Seismic moment (N m): sum over triangles of area x mean tent slip x the triangle's rigidity (Pa)."""
        slip_all = self.vertex_slip(slip_mm)
        return float(np.sum(self.tri_area * slip_all[self.fault_cells].mean(1) * 1e-3 * mu_per_triangle))


def tent_fault_template(tents, frame_lonlat):
    """csi.TriangularTents fault on the impulse vertices of `tents` (TentFault; km, depth positive), in the column order
    of G; faces = the PyLith fault triangles among them. Built once (its adjacency map is slow to compute); each solve
    deep-copies it. Source: CSI TriangularTents (Jolivet et al.)."""
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
    """The data covariance (mm^2) of the samples for one weighting `mode` (n_px = the pixels each quadtree block
    averages, CSI's wgt; cov_p as covariance_matrix):
      nugget: diagonal, nugget2 / n_px**nug_power, the samples treated as independent. nug_power 0 gives every
              sample the nugget variance (equal weights; the paper's setting, model_config.weighting / nugget_power);
              0.5 weights each sample by its block size.
      full:   the full correlated covariance of the semivariogram (covariance_matrix with nug_power), after Lohman and
              Simons (2005) and Sudhaus and Jonsson (2009); a sensitivity test.
    The exponent is the `nug_power` argument in both modes. keep (bool per sample, optional): samples outside it
    (closer to the trace than the near-trace cut) are decoupled from every other sample and given a variance 1e8
    times their own, so they do not influence the fit but remain in the products. Returns Cd (n, n).
    Source: written for this project."""
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
    constraint), with the Green's functions G_insar (m/m, right-lateral positive) entered by CSI's `Fault.setGFs`:
    exponential prior buildCmXY(sigma, lam) on the vertex positions (lam in km), the reference frame as a CSI
    transformation (ramp_spec, CSI polynomial terms: "1,x,y" a plane solved jointly with slip, the paper's setting;
    "1" a constant, a sensitivity test; the surface data keep a constant), NNLS (slip >= 0, the frame terms free) with
    multisourcesolve. Works on deep copies of the template fault and the samples (their covariance replaced by `Cd`,
    mm^2); vel_insar / vel_surface replace the data vectors (Monte Carlo). surface_template None: no surface data (the
    paper's setting); with surface data but use_surface False they are predicted, not fitted.
    Returns (fault, insar data, surface data or None), each data set with its synth and the fitted frame removed from vel.
    Source: CSI (Jolivet et al.); PyLith Green's functions used as in the PyLith inversion example
    examples/subduction-3d/invert_slip.py."""
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
    if surface is not None and not use_surface:                       # not inverted, but predicted
        surface.buildsynth([fault_model], direction="s")
    assert np.all(fault_model.slip[:, 0] >= -1e-9), "negative slip out of NNLS"
    return fault_model, insar_data, surface


def match_impulse_order(ref_fault_xyz, ref_vertex_index, fault_xyz, vertex_index, label="", tol_m=1e-3):
    """Column order that puts one GreensFns run's impulses into a reference run's order, matched by coordinates (m).
    PyLith numbers the fault vertices differently with a different number of MPI ranks, so two runs of the same mesh can
    list the same impulses in a different order. Asserts the same fault mesh (every vertex within tol_m, one to one)
    and the same impulse vertices; returns idx with fault_xyz[vertex_index][idx] == ref_fault_xyz[ref_vertex_index].
    `label` names the run in assertion messages. Source: written for this project."""
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
    (n_samples, n_triangles), m per m of slip, with CSI's sign convention. lon, lat in deg, depth in km; frame_lonlat
    = CSI frame origin. A module-level function so that chunks can run in parallel processes.
    Source: CSI TriangularPatches.buildGFs (Jolivet et al.; Meade 2007)."""
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
    flt.homogeneizeStrike(direction=0, sign=-1.0)                # one strike direction for every sub-triangle
    flt.initializeslip()
    flt.buildGFs(line, vertical=True, slipdir="s", verbose=False)
    G_part = flt.G[line.name]["strikeslip"]
    assert G_part.shape == (np.size(lon), np.size(lon_v) // 3), f"CSI G {G_part.shape}"
    return G_part


def analytic_tent_G(tents, lon, lat, look, frame_lonlat, to_mesh, n_sub=8, verbose=False, n_workers=1):
    """Homogeneous half-space Green's functions of the PyLith tents (unit right-lateral slip at one impulse vertex,
    linear to zero at its neighbours), on the same fault mesh: every fault triangle touching an impulse vertex is split
    into n_sub^2 constant-slip sub-triangles, the displacement of each sub-triangle comes from CSI's triangular
    dislocation (Meade 2007, Poisson solid, `TriangularPatches.buildGFs`), and is added to each corner tent with the
    tent's value at the sub-triangle centroid. The result converges to the half-space solution as n_sub increases. The
    geometry is carried by the triangles, so any dip is handled. Sign: CSI's strike slip with the normals turned east
    (homogeneizeStrike(0, -1)) is left-lateral for this trace, so G = -G_csi (tested in tests/test_model_utils.py);
    callers should assert a positive uniform-slip east-minus-west step. lon, lat: samples (deg); look: projection unit
    vector (east, north, up); to_mesh: pyproj transformer lon/lat -> mesh CRS. n_workers > 1 runs the chunks of 5000
    sub-triangles in parallel processes (the chunks are summed in order, so the result is the same).
    Returns G (n_samples, tents.n_imp), m per m.
    Source: written for this project; triangular dislocations of Meade (2007) as implemented in CSI (Jolivet et al.)."""
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
    """First 12 hex characters of the sha256 of a sample set's positions (lon, lat in deg, as float64): the cache key
    of every matrix computed at the samples (a rewritten point file with the same positions keeps its key).
    Source: written for this project."""
    import hashlib
    return hashlib.sha256(np.c_[np.asarray(lon, np.float64), np.asarray(lat, np.float64)].tobytes()).hexdigest()[:12]


def analytic_tent_G_cached(cache_npz, tents, lon, lat, look, frame_lonlat, to_mesh, n_sub, run_name, points_fingerprint,
                           n_workers=1):
    """analytic_tent_G, cached in cache_npz and keyed by its query: the PyLith run whose tents it uses, n_sub, the
    fingerprint of the sample point set (positions_fingerprint) and the impulse vertices (all asserted on reading, so a
    cached matrix is never used for different inputs). Computed only when the file is missing (about 1 h per 100,000
    sub-triangles). Returns G (n_samples, tents.n_imp), m per m, right-lateral positive.
    Source: written for this project."""
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
    archived products: the G matrix of every medium (load_G; rebuilt from the cached PyLith output when it was
    projected on another strike), the tent geometry (TentFault; media matched by vertex coordinates because PyLith's
    vertex numbering depends on the MPI rank count, each medium's G columns reordered into the first medium's impulse
    order) and its CSI template; the signed distance (km, east positive) of every sample from the model trace
    (`trace_xy_m`, m in the mesh CRS) and its along-strike km (`sample_km_pp`, nearest trace point of `trace_km_pp`);
    `keep` = the samples that enter the fit (|d| >= near_trace_cut_km); the near-field masks `west_near` / `east_near`
    = fitted samples within near_field_km west / east of the trace inside the SSE section (creep_lo_km to creep_hi_km),
    over which the near-field RMS is computed. With `crossings` (dict km, lat, mean_mm of 04's measured crossings) also
    the measured-crossing surface constraint (`surface`, `G_surface`; used in sensitivity tests only).
    run_names: medium -> PyLith GreensFns run; medium_dirs: medium -> (pylith_output dir, gf dir). Returns a dict.
    Source: written for this project."""
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
    keep = np.abs(signed_d_km) >= near_trace_cut_km                        # samples that enter the fit
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
    """Summary metrics of one tent solution (slip_mm on the impulse vertices; insar_data with synth and the fitted frame
    removed): RMS of the residual (data minus model, mm) over the fitted samples (`keep`) and over the near-field masks
    west / east; potency (exact tent integral, m^3); Mw at mu_ref (Pa) and with the per-triangle rigidity
    mu_per_triangle (Pa; NaN if None), mu_effective = M0 / potency (GPa); peak slip (mm) and its depth (m); the depths
    above which 50 % and 90 % of the potency lie (m, TentFault.potency_depth_m). Returns a dict.
    Source: written for this project; Mw after Hanks & Kanamori (1979)."""
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
    weighted misfit chi2 of the fitted samples (`keep`; the L-curve's x axis), the prior-weighted roughness
    sqrt(m^T (Cm / sigma^2)^-1 m) (its y axis), the RMS of the fitted samples (mm), peak slip (mm), Mw at mu_ref (Pa)
    and the solve time (s). `label` prefixes the progress lines. Returns a DataFrame, one row per sigma.
    Source: written for this project, after Hansen (1992)."""
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
    """A model's own L-curve corner, in two steps: (1) the L-curve on base_grid and its corner (lcurve_corner: maximum
    curvature of log sqrt(chi2) against log roughness); (2) the L-curve on base_grid plus base corner x refine_factors
    (new sigmas only) and its corner, which is the model's sigma (not rounded to a grid value). Arguments as
    tent_lcurve. Returns (corner, base_corner, table of every solve with `step` 1 or 2).
    Source: written for this project, after Hansen (1992)."""
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
    """Slip pattern of the fixed-depth scan: 1 on every impulse vertex between top_km and depth_km deep (km) inside the
    SSE section (section_km, km from Point Pinole), tapered linearly to 0 over taper_km (km) at each end of the section,
    0 elsewhere. With the tents this is full slip down to depth_km, falling to zero at the next vertex row (one mesh
    cell). Returns an array (n_imp,). Source: written for this project."""
    full_lo, full_hi = section_km[0] + taper_km, section_km[1] - taper_km
    along = np.clip(np.minimum(tents.imp_km - full_lo, full_hi - tents.imp_km) / taper_km + 1, 0, 1)
    return along * (tents.imp_depth_km <= depth_km + 1e-6) * (tents.imp_depth_km >= top_km - 1e-6)


def uniform_slip_fit(G, shape, data_mm, x_km, y_km, keep):
    """One slip amplitude (mm) for a fixed slip pattern `shape` (G in m/m, data in mm, sample positions x_km, y_km in
    km) plus a plane (1, x, y), by least squares over the fitted samples (`keep`) with equal weights (the paper's
    weighting gives every sample the nugget variance, so the weights cancel). Returns (amplitude_mm, residual_mm =
    data minus the fitted pattern and plane, at every sample). Source: written for this project (numpy least squares)."""
    keep = np.asarray(keep, bool)
    design = np.c_[G @ shape, np.ones(data_mm.size), x_km, y_km]
    coef, *_ = np.linalg.lstsq(design[keep], data_mm[keep], rcond=None)
    return float(coef[0]), data_mm - design @ coef


def profile_samples(lon, lat, line_ends_xy, epsg, half_width_m=300.0):
    """The samples (lon, lat in deg) within half_width_m (m) of a profile line (06's `prof_fp_<line>_ends`: x0, y0, x1,
    y1 in EPSG `epsg`, m) and their distance along it (m from its first end). Returns (selected mask, along_m for every
    sample). Source: written for this project."""
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


def side_deflection_mm(values_mm, signed_d_km, in_section, keep, side, near_km, far_km):
    """Near-fault deflection on one side (side -1 west, +1 east) of the trace: the mean of values_mm over the fitted
    samples (keep) in the SSE section (in_section) whose distance from the trace is within near_km (km, (min, max)), minus
    the mean over those within far_km. Returns (deflection mm, n near, n far). Source: written for this project."""
    distance_km = side * np.asarray(signed_d_km, float)
    usable = np.asarray(keep, bool) & np.asarray(in_section, bool)
    near = usable & (distance_km >= near_km[0]) & (distance_km <= near_km[1])
    far = usable & (distance_km >= far_km[0]) & (distance_km <= far_km[1])
    assert near.sum() >= 5 and far.sum() >= 5, f"too few samples (near {near.sum()}, far {far.sum()})"
    return float(np.mean(values_mm[near]) - np.mean(values_mm[far])), int(near.sum()), int(far.sum())


def correlated_noise(Cd, n_draws, seed):
    """n_draws realisations (n_draws, n) of zero-mean Gaussian noise with covariance Cd (mm^2): Cholesky factor L of
    Cd, noise = L z, z ~ N(0, 1) from numpy's default_rng(seed). Used for the Monte Carlo uncertainty.
    Source: written for this project."""
    L = np.linalg.cholesky(Cd)
    z = np.random.default_rng(seed).standard_normal((n_draws, Cd.shape[0]))
    return z @ L.T


def strike_slip_stress_drop_pa(mu_pa, slip_m, width_m):
    """Stress drop (Pa) of a long strike-slip fault that reaches the free surface: delta_sigma = (2 / pi) mu d / w, mu the
    rigidity (Pa), d the (uniform) slip (m), w the down-dip width of the slipping part (m).
    Source: Knopoff (1958), doi 10.1111/j.1365-246X.1958.tb00033.x; Kanamori & Anderson (1975), BSSA 65, Eq. 1 with
    C = 2 / pi."""
    assert mu_pa > 0 and slip_m > 0 and width_m > 0
    return float(2.0 / np.pi * mu_pa * slip_m / width_m)


# =====================================================================================================================
# 7. PyLith materials, rigidity and the extent of slip
# =====================================================================================================================
def shear_modulus_pa(density_kg_m3, vs_m_s):
    """Shear modulus mu = rho vs^2 (Pa) from density (kg/m^3) and S-wave speed (m/s)."""
    return density_kg_m3 * vs_m_s ** 2


def mu_beside_fault(elastic_info_h5, grid_npz, tents, label):
    """The rigidity used by PyLith beside the fault. Reads the per-cell shear modulus of a run
    (`<run>-elastic_info.h5`; materials basis order 0, or the cell mean of the vertex values for order 1), compares
    it with the grid it was read from (`<grid>.npz`, mu = rho vs^2, linear interpolation at the cell centroids of the
    fine grid box), and takes the nearest cell on each side of every fault triangle of `tents` (TentFault). `label`
    prefixes the printed summary. Returns (mean of the two sides, west, east) per triangle (Pa) and the median
    |relative difference| from the grid. Source: written for this project."""
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
    """Effective rigidity (units of the inputs) of a fault on a material interface: 2 mu_a mu_b / (mu_a + mu_b).
    Source: Vavrycuk (2013), GJI 194, 395-400."""
    return 2.0 * mu_a * mu_b / (mu_a + mu_b)


def potency_extent(sub_potency_m3, sub_depth_m, sub_km, fraction=0.9):
    """Depth extent w (m) and along-strike length L (km) that hold `fraction` of a slip model's potency: w = the depth
    above which that share of the potency lies; L = the span between the (1 - fraction)/2 and (1 + fraction)/2 potency
    quantiles along strike. Inputs: potencies (m^3), depths (m) and along-strike positions (km) of the fault elements.
    Source: written for this project."""
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
# 8. Provenance
# =====================================================================================================================
def field_fingerprint(fp_maps_npz):
    """The fingerprint of 06's fault-parallel field that a model fits, in the form every model product carries
    ("fp_maps.npz:<first 12 hex characters of the sha256>"; insar_utils.file_fingerprint). Returns a string.
    Source: written for this project."""
    return f"fp_maps.npz:{utils.file_fingerprint(fp_maps_npz)}"


# =====================================================================================================================
# 9. Cross-fault steps at the crossings
# =====================================================================================================================
def crossing_steps_like(values_mm, signed_d_km, sample_km_pp, crossing_km, inner_km=0.03, outer_km=0.15,
                        half_along_km=0.2, min_per_side=3):
    """Cross-fault step (mm) of a field values_mm at the samples, measured as notebooks 03 and 04 measure the crossings:
    for each crossing km (crossing_km, km from Point Pinole), the median of the samples inner_km to outer_km (km) east
    of the trace (signed_d_km, east positive) and within +/- half_along_km along strike (sample_km_pp), minus the same
    median west. NaN where a side has fewer than min_per_side samples. Used to compare the model with the measured
    crossings with the same measurement (the model's top-edge slip differs from the step the crossings measure).
    Source: written for this project."""
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
