# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # 06 The event field: per-pixel step maps, the mean event and its cross-fault shape
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13. Run time: about 3 min per track for the per-pixel fits; seconds for the rest.
# ```
#
# **What this notebook does.** Notebook 04 measures the events on cross-fault
# differences of the fault-parallel series. This notebook maps them: every pixel
# in a box around Tule Pond is fitted with its own time function, which gives
# the cross-fault shape of each event as well as its size. Each event's step is
# placed at the step epoch of notebook 04 (`onset_lo` in `creep_events.csv`, the
# start of the onset window).
#
# 1. **Per-pixel model** on each track: the time-function model of MintPy's
#    `timeseries2velocity` (Yunjun et al. 2019): intercept, linear rate, annual
#    cycle and one Heaviside step per event. It is solved per pixel by least
#    squares over the epochs with data, because `timeseries2velocity` does not
#    accept records with gaps.
# 2. **Burst-seam correction**: the offset across the one burst seam each track
#    has in the box is measured per event and removed. The seam orientation is
#    taken from the OPERA CSLC static bounding polygons (section 2); both tracks
#    are corrected in the same way.
# 3. **Mean event**: the mean of the five step maps per track and its per-pixel
#    scatter.
# 4. **Decomposition** of the inter-event rate and the mean event into
#    fault-parallel and vertical components with MintPy's `asc_desc2horz_vert`,
#    as in notebook 02, here with the mean local strike of the section hosting
#    the events.
# 5. **Profiles** X-X' (the Tule Pond peak) and Y-Y' (creepmeter CFW1) across the
#    fault-parallel, vertical and LOS maps: the pixels, the near-fault step (east
#    band minus west band, the aperture of notebook 04) and the near-fault
#    amplitude of each block against its own far field. No model is fitted to
#    the profiles; the slip models are in notebooks 07 and 08.
# 6. **Inputs for Fig 2**: the maps and profiles, written to one file that
#    notebook 09 draws after the slip model.
# 7. **Mean-event statistics**: the near-fault fault-parallel deflection of each
#    block at X-X' and Y-Y', the decay distance of the X-X' event profile on
#    each side of the trace, and the vertical displacement around Lake Elizabeth
#    east of the fault.
#
# Every near-fault step is the median of an east band minus the median of a
# west band, 30 to 150 m from the trace (`dist_min` to `dist_max`), the same
# aperture as notebook 04, so that both notebooks measure the same quantity.
#
# **Inputs**: per track, the MintPy stack `timeseries_tropo.h5`, `geometryGeo.h5`,
# `maskTempCoh_07.h5` (the fit mask, `coh_timeseries`) and `maskTempCoh_09.h5`
# (the map mask, `coh_maps`); notebook 04 `creep_events.csv` and
# `section_lengths.csv`; the fault trace `hayward_fault_profiles.csv`; the OPERA
# `CSLC-S1-STATIC` files of the bursts covering the box (`static_dir` per track).
#
# **Outputs** (`outputs/06_event_field_geometry/`): `tulepond_steps_<track>.h5`
# (every per-event LOS step map, before and after the burst-seam correction,
# each tagged with its `step_date` on that track and its `step_epoch` from
# notebook 04; the mean, scatter, rate and geometry; read by notebook 07),
# `csi_mean_event_<track>.npz` (the mean event as a flat table of pixels),
# `fp_maps.npz` (fault-parallel and vertical rate and mean event; read by
# notebooks 07 and 08), `profile_<XX|YY>_<field>[_<track>].csv`,
# `fig2_inputs.npz` (read by notebooks 08, 09 and 11),
# `timefunc_fit_<track>.npz` (the cached per-pixel fit),
# `06_event_field_stats.txt`, registry rows `06_*`.
#
# **Figures**: `figS_event_step_maps_coh0.9` (.png and .pdf, the step-map grid,
# supplement).
#
# **What to check.** The printed near-fault LOS steps on the two tracks
# (section 7, asserted: the mean event is negative on ascending and positive
# on descending, and every descending step is positive) and the per-event
# ascending/descending ratio printed beside the geometric ratio; the burst
# offsets (section 6, asserted below 2 mm); the fault-parallel profiles at X-X'
# and Y-Y' (section 10); the vertical step printed beside the fault-parallel
# one (section 9); the mean-event statistics (section 12).

# %% [markdown]
# ## 1. Imports

# %%
import glob
import os
import re

import insar_utils as utils
from TulePond_filepaths import (common_paths, insar_dic_list, box_lonlat,
                                coh_timeseries, az_smooth_n, dist_min, dist_max, paths_out, repo_dir,
                                lake_elizabeth_lonlat, lake_elizabeth_radius_m)   # the Lake Elizabeth area of section 12

import numpy as np
import pandas as pd
import pyproj
import h5py
import pygmt
import shapely
from scipy.spatial import cKDTree
from mintpy.utils import time_func                                    # the design matrix timeseries2velocity uses
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert              # the asc/desc decomposition (MintPy)

# %% [markdown]
# ## 2. Paths, parameters and the burst seams
#
# Every parameter the notebook uses, with its definition, and the position of
# the burst seam of each track in the box.
#
# **Burst seams.** The OPERA CSLC-S1-STATIC files carry the bounding polygon of
# each burst (`identification/bounding_polygon`, lon/lat). A seam is the shared
# edge of two consecutive bursts of one subswath; the polygons tile the frame
# without overlap (asserted). Across the box the edge is replaced by the
# straight line through its latitudes at the box's west and east longitudes,
#
# $$\varphi_s(\lambda) = b + s\,(\lambda + 122^\circ), \qquad s = \frac{\varphi_E - \varphi_W}{\lambda_E - \lambda_W}, \qquad b = \varphi_W - s\,(\lambda_W + 122^\circ),$$
#
# where $\lambda$ is longitude (deg), $\varphi_s$ the seam latitude (deg),
# $\lambda_W$ and $\lambda_E$ the box's west and east longitudes (deg),
# $\varphi_W$ and $\varphi_E$ the latitudes of the polygon edge there (deg),
# $s$ the slope (deg/deg) and $b$ the latitude at $122^\circ$ W (deg). The edge
# departs from this line by less than `burst_line_tol_deg` (about one 30 m
# pixel, asserted). Each track has one seam in the box (asserted).
#
# The displacement discontinuity in the DISP-S1 mosaic (OPERA 2025) does not lie
# on the polygon edge. The polygon line gives the seam's orientation, and the
# line is moved north by `burst_seam_shift_m` (m, per track, converted to
# latitude at 111.2 km per degree). The shifts were found by scanning the shift
# in 10 m steps on the mean of the five step maps before correction; section 6
# checks that the jump across the shifted line is larger than the jump across
# the polygon line. The shifts must be found again if the products change.

# %%
notebook = "06_event_field_geometry"
out_dir = paths_out[notebook]
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

unit = 1000                     # m -> mm
periodic = [1.0]                # annual cycle in the per-pixel model (years)
min_obs = 60                    # a pixel with gaps needs this many finite epochs to be fitted
geom_step = 20                  # MintPy's decomposition window (pixels), as notebook 02

# burst-seam correction (sections 2 and 6)
burst_band_deg = 0.0015         # width of the band on each side of the seam (deg of latitude)
burst_far_m = 400.0             # band pixels closer than this to the trace are excluded (m)
burst_line_tol_deg = 0.0003     # about one 30 m pixel: the polygon edge must lie this close to the straight line
burst_min_pixels = 200          # finite pixels each band needs for every event
                                # (the ascending seam runs along the box's northern edge, so its north band is clipped)
burst_corrected = ("ascending", "descending")
burst_seam_shift_m = {"ascending": 550.0, "descending": -690.0}     # seam shift from the polygon line (m, north positive)


def burst_seams(static_dir, box):
    """Find the burst seams that cross a box.

    A seam is the shared edge of two consecutive bursts of one subswath, read from the CSLC-S1-STATIC
    bounding polygons, and kept as the straight line lat = intercept + slope * (lon + 122).

    Parameters
    ----------
    static_dir : str
        Folder of the OPERA_L2_CSLC-S1-STATIC_*.h5 files of one track.
    box : tuple of float
        (W, S, E, N) in degrees.

    Returns
    -------
    seams : list of dict
        Per seam: the north and south burst ids, slope (deg/deg), intercept (deg, at 122 W), the latitudes at the
        box's west and east longitudes, and the largest departure of the edge from the line (deg).
    box_files : list of str
        The static files of the bursts that cover the box.
    """
    static_files = sorted(glob.glob(os.path.join(static_dir, "OPERA_L2_CSLC-S1-STATIC_*.h5")))
    assert static_files, f"no CSLC-S1-STATIC files in {static_dir} (see 'static_dir' in TulePond_filepaths)"
    box_polygon = shapely.box(*box)
    polygons = {}
    box_files = []
    for static_file in static_files:
        burst_id = re.search(r"STATIC_(T\d{3}-\d{6}-IW[1-3])_", os.path.basename(static_file)).group(1)
        assert burst_id not in polygons, f"two static files for burst {burst_id} in {static_dir}"
        with h5py.File(static_file, "r") as static_h5:
            polygon = shapely.from_wkt(static_h5["identification/bounding_polygon"][()].decode())
        if polygon.intersects(box_polygon):
            polygons[burst_id] = polygon
            box_files.append(static_file)
    assert polygons, f"no burst in {static_dir} covers the box {box}"
    seams = []
    for burst_id in sorted(polygons):
        track, burst_number, subswath = burst_id.split("-")
        next_id = f"{track}-{int(burst_number) + 1:06d}-{subswath}"
        if next_id not in polygons:
            continue
        pair = (burst_id, next_id)
        assert polygons[burst_id].intersection(polygons[next_id]).area < 1e-10, f"bursts {pair} overlap"
        shared_edge = shapely.line_merge(polygons[burst_id].boundary.intersection(polygons[next_id].boundary))
        assert shared_edge.geom_type == "LineString" and shared_edge.length > 0, f"bursts {pair} share no single edge"
        if not shared_edge.intersects(box_polygon):
            continue
        edge_lon, edge_lat = np.array(shared_edge.coords).T
        order = np.argsort(edge_lon)
        edge_lon = edge_lon[order]
        edge_lat = edge_lat[order]
        assert edge_lon[0] < box[0] and edge_lon[-1] > box[2], f"the {pair} seam does not span the box west to east"
        lat_west, lat_east = np.interp([box[0], box[2]], edge_lon, edge_lat)
        slope = (lat_east - lat_west) / (box[2] - box[0])
        intercept = lat_west - slope * (box[0] + 122.0)
        in_box = (edge_lon > box[0]) & (edge_lon < box[2])
        departure = np.abs(edge_lat[in_box] - (intercept + slope * (edge_lon[in_box] + 122.0))).max(initial=0.0)
        assert departure < burst_line_tol_deg, f"the {pair} seam is not straight across the box ({departure:.5f} deg)"
        north_id = max(pair, key=lambda b: polygons[b].centroid.y)
        south_id = min(pair, key=lambda b: polygons[b].centroid.y)
        seams.append(dict(north=north_id, south=south_id, slope=float(slope), intercept=float(intercept),
                          lat_west=float(lat_west), lat_east=float(lat_east), departure_deg=float(departure)))
    return seams, box_files


burst_seam_list = {}
burst_static_files = []
for dic in insar_dic_list:
    orbit = dic["ORBIT_DIRECTION"]
    burst_seam_list[orbit], seam_files = burst_seams(dic["static_dir"], box_lonlat)
    burst_static_files += seam_files
    for seam in burst_seam_list[orbit]:
        print(f"[{orbit}] burst seam {seam['north']} / {seam['south']}: lat {seam['lat_west']:.4f} at {box_lonlat[0]} E, "
              f"{seam['lat_east']:.4f} at {box_lonlat[2]} E (slope {seam['slope']:+.4f} deg/deg); "
              f"corrected: {orbit in burst_corrected}")
for orbit in ("ascending", "descending"):
    assert len(burst_seam_list[orbit]) == 1, f"expected one {orbit} burst seam in the box, found {burst_seam_list[orbit]}"
des_seam = burst_seam_list["descending"][0]
asc_seam = burst_seam_list["ascending"][0]
# check against the seam positions read from the polygons: descending 37.5384 N at -121.97 E; ascending
# T035-073249-IW2 / T035-073250-IW2, 37.6149 N at -122.10 E to 37.6504 N at -121.84 E (along the box's northern edge)
assert abs(des_seam["intercept"] + des_seam["slope"] * (-121.97 + 122.0) - 37.54) < 0.01, f"descending seam moved: {des_seam}"
assert abs(asc_seam["lat_west"] - 37.6149) < 0.01 and abs(asc_seam["lat_east"] - 37.6504) < 0.01, f"ascending seam moved: {asc_seam}"
zoa1_seam_lat = des_seam["intercept"] + des_seam["slope"] * (common_paths["ref_lon"] + 122.0)
print(f"ZOA1 ({common_paths['ref_lat']} N) is {abs(common_paths['ref_lat'] - zoa1_seam_lat) * 111.2e3:.0f} m "
      f"{'north' if common_paths['ref_lat'] > zoa1_seam_lat else 'south'} of the descending seam")
burst_lines = {}                # (slope, intercept) of each corrected seam, as sections 6 and 13 read them
for orbit in burst_corrected:
    burst_lines[orbit] = [(seam["slope"], seam["intercept"]) for seam in burst_seam_list[orbit]]

# profiles across the maps, perpendicular to the local strike
profile_pts = {"X": (-121.97816, 37.56270),      # X-X': the trace point 58.5 km from Point Pinole, inside the 90% band
               "Y": (-121.95233, 37.53214)}      # Y-Y': CFW1
cross_m = 2000.0                # profile half-length (m)
swath_m = 150.0                 # along-strike half-width of the profile swath (m)
coh_maps = 0.9                  # temporal-coherence mask of the maps and of the decomposed field the slip model inverts;
                                # the per-pixel fit and the LOS checks use coh_timeseries
far_m = (1500.0, 2000.0)        # far field of each block for the block amplitudes (m from the trace)
block_seed = 20260923           # each block-amplitude bootstrap starts its own generator from this seed
n_boot_blocks = 2000            # bootstrap resamples for the block amplitudes
event_far_m = 1500.0            # the event profile is zeroed on its pixels at |distance| >= this (m, both blocks)
rate_zero_m = 1000.0            # the rate profile is zeroed on the mid-point of the two block medians within this (m)
min_far_pixels = 20             # the event profile needs this many far-field pixels to be zeroed
xfault_step_m = (dist_min * 1e3, dist_max * 1e3)   # the near-fault band, the aperture of notebook 04 (m)
decay_bin_m = 150.0             # bin width of the decay distance (section 12); the running median of Fig 2
                                # (notebook 09, median_bin_m) uses the same bins

events_file = os.path.join(paths_out["04_event_detection"], "creep_events.csv")
lengths_file = os.path.join(paths_out["04_event_detection"], "section_lengths.csv")

for dic in insar_dic_list:
    assert os.path.exists(dic["ts_file"]), f"MintPy stack missing: {dic['ts_file']}"

map_mask_files = []
for dic in insar_dic_list:
    map_mask_files.append(dic["mask_file"][coh_maps])
REG = utils.Registry(notebook, common_paths, repo_dir)
stack_files = []
for dic in insar_dic_list:
    stack_files.append(dic["ts_file"])
    stack_files.append(dic["geo_file"])
    stack_files.append(dic["mask_file"][coh_timeseries])
REG.inputs(events_file, lengths_file, common_paths["fault_profiles"], *stack_files, *map_mask_files, *burst_static_files)

# %% [markdown]
# ## 3. Load the events, the trace and the box
#
# The step epoch of each Tule Pond event is `onset_lo` from notebook 04, the
# last acquisition before the step. The box is `box_lonlat` (W, S, E, N)
# converted to a row and column slice of the product grid. Both tracks are on
# the same grid (asserted), so one slice serves both and the two LOS maps line
# up pixel for pixel.
#
# The decomposition strike is the mean local strike of the section hosting the
# events (`tulepond_creep_length` in notebook 04 `section_lengths.csv`),
#
# $$\theta = \frac{1}{N}\sum_{j=1}^{N} \left(\beta_j + 90^\circ\right),$$
#
# where $\beta_j$ is the cross-fault azimuth of trace vertex $j$ (deg clockwise
# from north), smoothed by a running median over `az_smooth_n` vertices, and
# $N$ the number of trace vertices in the section. Each pixel gets a signed
# distance from the trace,
#
# $$d = (\mathbf{x} - \mathbf{x}_v) \cdot \hat{\mathbf{n}}, \qquad \hat{\mathbf{n}} = \left(\sin(\theta - 90^\circ),\ \cos(\theta - 90^\circ)\right),$$
#
# where $\mathbf{x}$ is the pixel centre (easting, northing in m, UTM),
# $\mathbf{x}_v$ the nearest trace vertex (m) and $\hat{\mathbf{n}}$ the unit
# normal to the strike, so that $d$ (m) is positive on the east (north-east)
# side of the fault.

# %%
events = pd.read_csv(events_file)
tulepond_events = events[events.segment == "TULEPOND"].sort_values("onset_lo").reset_index(drop=True)
# the step of notebook 04 is H(t > step epoch), with the step epoch the last acquisition before the event (onset_lo)
step_epochs = tulepond_events.onset_lo.to_numpy(float)
print(f"{len(tulepond_events)} Tule Pond events, step epochs (onset_lo): {step_epochs}")

x_fault = pd.read_csv(common_paths["fault_profiles"])
x_fault["az_smooth"] = x_fault["profile_az"].rolling(az_smooth_n, center=True, min_periods=3).median()
lengths = pd.read_csv(lengths_file).set_index("name")
band = lengths.loc["tulepond_band90"]          # the 90% band that Fig 1 stacks
# the decomposition strike: the mean local strike of the section hosting the events, not of the whole box
section = lengths.loc["tulepond_creep_length"]
in_section = (x_fault.ft_lat >= section.lat_lo) & (x_fault.ft_lat <= section.lat_hi)
strike = float(np.nanmean(x_fault.loc[in_section, "az_smooth"] + 90.0))
print(f"strike: SSE section (km {section.km_lo:.1f}-{section.km_hi:.1f}, {in_section.sum()} crossings) {strike:.2f} deg")
assert 130 < strike < 160

grid_files = []
for dic in insar_dic_list:
    grid_files.append(dic["ts_file"])
attrs = utils.load_h5_attrs(grid_files[0])
for grid_file in grid_files[1:]:
    other_attrs = utils.load_h5_attrs(grid_file)
    for attr_name in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP", "EPSG", "WIDTH", "LENGTH"):
        assert attrs[attr_name] == other_attrs[attr_name], f"tracks are on different grids ({attr_name}): {attrs[attr_name]} vs {other_attrs[attr_name]}"
epsg = int(attrs["EPSG"])
to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
to_ll = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)

# the box as a row/column slice of the product grid
box_west, box_south, box_east, box_north = box_lonlat
box_x, box_y = to_utm.transform([box_west, box_west, box_east, box_east], [box_south, box_north, box_south, box_north])
x0 = float(attrs["X_FIRST"])
dx = float(attrs["X_STEP"])
y0 = float(attrs["Y_FIRST"])
dy = float(attrs["Y_STEP"])
c0 = max(int(np.floor((min(box_x) - x0) / dx)), 0)
c1 = min(int(np.ceil((max(box_x) - x0) / dx)), int(attrs["WIDTH"]))
r0 = max(int(np.floor((max(box_y) - y0) / dy)), 0)                  # dy < 0: the northern edge is the first row
r1 = min(int(np.ceil((min(box_y) - y0) / dy)), int(attrs["LENGTH"]))
nr = r1 - r0
nc = c1 - c0
print(f"box rows {r0}:{r1} cols {c0}:{c1} = {nr} x {nc} pixels of {abs(dx):.0f} m")

# pixel-centre coordinates of the box
Xg, Yg = np.meshgrid(x0 + (np.arange(c0, c1) + 0.5) * dx, y0 + (np.arange(r0, r1) + 0.5) * dy)
lon_g, lat_g = to_ll.transform(Xg, Yg)
# the trace in UTM and the signed cross-fault distance of every pixel (m, positive on the north-east side)
trace_x, trace_y = to_utm.transform(x_fault.ft_lon.values, x_fault.ft_lat.values)
tree = cKDTree(np.c_[trace_x, trace_y])
d_trace, nearest_vertex = tree.query(np.c_[Xg.ravel(), Yg.ravel()])
normal = np.array([np.sin(np.deg2rad(strike - 90)), np.cos(np.deg2rad(strike - 90))])   # unit normal: strike - 90 is the NE side (+ east)
d_signed = ((np.c_[Xg.ravel(), Yg.ravel()] - np.c_[trace_x, trace_y][nearest_vertex]) @ normal).reshape(nr, nc)
d_trace = d_trace.reshape(nr, nc)

# %% [markdown]
# ## 4. The per-pixel time-function model
#
# Each pixel's LOS time series on one track is modelled with the time functions
# of MintPy's `timeseries2velocity` (Yunjun et al. 2019),
#
# $$d(t) = a + v\,(t - t_0) + c\,\cos\!\big(2\pi (t - t_0)\big) + s\,\sin\!\big(2\pi (t - t_0)\big) + \sum_{k=1}^{K} A_k\, H(t - \tau_k),$$
#
# where $d$ is the LOS displacement (mm, positive toward the satellite), $t$
# the acquisition time (decimal years), $t_0$ the first acquisition of the
# track (decimal years), $a$ the intercept (mm), $v$ the inter-event LOS rate
# (mm/yr), $c$ and $s$ the cosine and sine amplitudes of the annual cycle (mm),
# $A_k$ the step of event $k$ (mm), $K$ the number of events (five), and $H$ the
# Heaviside function, $H(x) = 1$ for $x > 0$ and 0 otherwise. This is the
# joint model of notebook 04 applied to one pixel. The step date $\tau_k$ is
# the last acquisition on this track on or before the step epoch of notebook 04
# (`onset_lo`, the last acquisition before the event on the two-track grid), so
# it counts as pre-event and the next acquisition is the first post-event
# sample.
#
# The design matrix $\mathbf{G}$ comes from `mintpy.utils.time_func`. Only the
# solver is replaced, by a per-pixel one that skips missing epochs. For each
# pixel the least-squares solution and its formal errors over the $N$ epochs
# with data are
#
# $$\hat{\mathbf{m}} = (\mathbf{G}^\top \mathbf{G})^{-1} \mathbf{G}^\top \mathbf{d}, \qquad \hat{\sigma}^2 = \frac{\lVert \mathbf{d} - \mathbf{G}\hat{\mathbf{m}} \rVert^2}{N - P}, \qquad \sigma_{m_j} = \sqrt{\hat{\sigma}^2 \left[(\mathbf{G}^\top \mathbf{G})^{-1}\right]_{jj}},$$
#
# where $\mathbf{d}$ is the vector of observed displacements (mm),
# $\hat{\mathbf{m}} = (a, v, c, s, A_1, \dots, A_K)$ the parameters, $P = 4 + K$
# their number, $\hat{\sigma}^2$ the residual variance (mm$^2$) and
# $\sigma_{m_j}$ the formal 1-sigma error of parameter $j$. The residual RMS is
# $\sqrt{\lVert \mathbf{d} - \mathbf{G}\hat{\mathbf{m}} \rVert^2 / N}$ (mm).
# Pixels with complete records are solved in one matrix operation; the others
# one at a time, and only if they have at least `min_obs` epochs.

# %%
def step_dates_for(dates_yyyymmdd, step_epochs_yr):
    """Place each step epoch of notebook 04 on one track's acquisitions.

    Parameters
    ----------
    dates_yyyymmdd : list of str
        The track's acquisition dates (YYYYMMDD).
    step_epochs_yr : array of float
        Step epochs of notebook 04 (decimal years, onset_lo).

    Returns
    -------
    list of tuple
        (step_date, step_epoch) per event: step_date (YYYYMMDD) is the last acquisition on this track on or before
        the step epoch, so with MintPy's H(t > t_step) it stays pre-event.
    """
    dts = pd.to_datetime(dates_yyyymmdd, format="%Y%m%d")
    yrs = np.array([utils.date_to_decimal_year(d) for d in dates_yyyymmdd])
    out = []
    for step_epoch in step_epochs_yr:
        on_or_before = np.nonzero(yrs <= step_epoch + 1e-4)[0]
        assert len(on_or_before) > 0, f"no acquisition on or before the step epoch {step_epoch:.4f}"
        i = int(on_or_before[-1])
        assert step_epoch - yrs[i] < 0.05, \
            f"the last acquisition before the step epoch {step_epoch:.4f} is {365.25 * (step_epoch - yrs[i]):.0f} d earlier"
        out.append((dts[i].strftime("%Y%m%d"), float(step_epoch)))
    return out


def fit_time_func(cube, dates, keep, step_dates):
    """Per-pixel least squares of MintPy's time-function model (linear, annual, one step per event).

    Parameters
    ----------
    cube : ndarray
        (n_dates, n_rows, n_cols) LOS displacement (mm), NaN where missing.
    dates : list of str
        Acquisition dates (YYYYMMDD).
    keep : ndarray of bool
        (n_rows, n_cols) pixels to fit.
    step_dates : list of str
        Step dates (YYYYMMDD).

    Returns
    -------
    out : dict
        Coefficient maps (mm, mm/yr), their formal errors ("<name>Std") and "residualRMS" (mm).
    G : ndarray
        The design matrix.
    names : list of str
        The coefficient names, in the column order of G.
    """
    model = dict(polynomial=1, periodic=periodic, stepDate=list(step_dates), polyline=[], exp={}, log={})
    G = time_func.get_design_matrix4time_func(dates, model).astype(float)
    names = ["intercept", "velocity"]
    for period in periodic:
        names.append(f"period{period}Cos")
        names.append(f"period{period}Sin")
    for step_date in step_dates:
        names.append(f"step{step_date}")
    assert G.shape[1] == len(names)
    n_dates, n_rows, n_cols = cube.shape
    series = cube.reshape(n_dates, -1)                          # one column per pixel
    n_par = G.shape[1]
    coef_maps = np.full((n_par, n_rows * n_cols), np.nan, np.float32)
    coef_std = np.full((n_par, n_rows * n_cols), np.nan, np.float32)
    rms = np.full(n_rows * n_cols, np.nan, np.float32)
    fitted_pixels = np.nonzero(keep.reshape(-1))[0]
    complete = np.isfinite(series[:, fitted_pixels]).all(0)
    print(f"  fitting {len(fitted_pixels)} pixels x {n_dates} dates, {n_par} parameters: {complete.sum()} complete records, "
          f"{(~complete).sum()} with gaps")
    if complete.any():                                          # complete records: one solve
        coef = np.linalg.lstsq(G, series[:, fitted_pixels[complete]], rcond=None)[0]
        residual = series[:, fitted_pixels[complete]] - G @ coef
        variance = (residual ** 2).sum(0) / (n_dates - n_par)
        covariance = np.linalg.inv(G.T @ G)
        coef_maps[:, fitted_pixels[complete]] = coef
        coef_std[:, fitted_pixels[complete]] = np.sqrt(np.outer(np.diag(covariance), variance))
        rms[fitted_pixels[complete]] = np.sqrt((residual ** 2).mean(0))
    for pixel in fitted_pixels[~complete]:                      # records with gaps: one at a time
        pixel_series = series[:, pixel]
        finite = np.isfinite(pixel_series)
        if finite.sum() < min_obs:
            continue
        G_pixel = G[finite]
        coef = np.linalg.lstsq(G_pixel, pixel_series[finite], rcond=None)[0]
        residual = pixel_series[finite] - G_pixel @ coef
        variance = (residual ** 2).sum() / max(finite.sum() - n_par, 1)
        coef_maps[:, pixel] = coef
        coef_std[:, pixel] = np.sqrt(np.diag(np.linalg.inv(G_pixel.T @ G_pixel)) * variance)
        rms[pixel] = np.sqrt((residual ** 2).mean())
    out = {}
    for i, name in enumerate(names):
        out[name] = coef_maps[i].reshape(n_rows, n_cols)
        out[name + "Std"] = coef_std[i].reshape(n_rows, n_cols)
    out["residualRMS"] = rms.reshape(n_rows, n_cols)
    return out, G, names

# %% [markdown]
# ## 5. Fit each track, or read the cached fit
#
# The per-pixel fit is the slow step, so it is cached in
# `timefunc_fit_<track>.npz`. The cache is reused only if it was made from the
# same step dates, input dates, box, fit mask and model settings (`periodic`,
# `min_obs`); otherwise the pixels are refitted and the reason is printed.

# %%
cache_meta_keys = ("step_dates", "names", "input_dates", "box_rows_cols", "periodic", "min_obs", "fit_mask")
fits = {}           # per track: the coefficient maps, the geometry, the step dates
for dic in insar_dic_list:
    orbit = dic["ORBIT_DIRECTION"]
    f_fit = os.path.join(out_dir, f"timefunc_fit_{orbit}.npz")
    with h5py.File(dic["ts_file"], "r") as ts_h5:
        dates = [date.decode() for date in ts_h5["date"][:]]
    steps = step_dates_for(dates, step_epochs)
    step_dates = [step_date for step_date, _ in steps]
    with h5py.File(dic["mask_file"][coh_timeseries], "r") as mask_h5:
        mask_name = [name for name in mask_h5.keys() if name != "date"][0]
        keep = mask_h5[mask_name][r0:r1, c0:c1].astype(bool)
    with h5py.File(dic["geo_file"], "r") as geo_h5:              # loaded as notebook 02 does (float32, as MintPy wrote it)
        inc = geo_h5["incidenceAngle"][r0:r1, c0:c1].astype(np.float32)
        azi = geo_h5["azimuthAngle"][r0:r1, c0:c1].astype(np.float32)
    # The cache is keyed on everything the fit depends on; the step dates alone are not enough, because
    # appending epochs leaves them unchanged. A cache that does not match is recomputed.
    box_rows_cols = np.array([r0, r1, c0, c1])
    cache_ok = False
    if os.path.exists(f_fit):
        cached = np.load(f_fit)
        mismatches = []
        if list(cached["step_dates"]) != step_dates:
            mismatches.append("step dates differ")
        if "input_dates" not in cached.files or list(cached["input_dates"]) != dates:
            mismatches.append(f"input dates differ (now {len(dates)})")
        if "box_rows_cols" not in cached.files or not np.array_equal(cached["box_rows_cols"], box_rows_cols):
            mismatches.append("box differs")
        if "fit_mask" not in cached.files or not np.array_equal(cached["fit_mask"], keep):
            mismatches.append("fit mask differs")
        if "periodic" not in cached.files or list(cached["periodic"]) != list(periodic):
            mismatches.append("periodic terms differ")
        if "min_obs" not in cached.files or int(cached["min_obs"]) != min_obs:
            mismatches.append("min_obs differs")
        cache_ok = len(mismatches) == 0
        if not cache_ok:
            print(f"[{orbit}] cached fit unusable ({'; '.join(mismatches)}); refitting")
    if cache_ok:
        coef = {}
        for name in cached.files:
            if name not in cache_meta_keys:
                coef[name] = cached[name]
        print(f"[{orbit}] per-pixel fit loaded from {os.path.basename(f_fit)}")
    else:
        with h5py.File(dic["ts_file"], "r") as ts_h5:
            cube = ts_h5["timeseries"][:, r0:r1, c0:c1].astype(np.float32) * unit     # m -> mm
        cube[cube == 0] = np.nan                                # MintPy no-data
        print(f"[{orbit}] {len(dates)} dates, coherent pixels {keep.mean():.0%} of the box")
        coef, _, names = fit_time_func(cube, dates, keep, step_dates)
        np.savez(f_fit, step_dates=np.array(step_dates), names=np.array(names), input_dates=np.array(dates),
                 box_rows_cols=box_rows_cols, periodic=np.array(periodic), min_obs=min_obs, fit_mask=keep, **coef)
        del cube
    fits[orbit] = dict(coef=coef, keep=keep, inc=inc, azi=azi, steps=steps, dates=dates)
    print(f"  {orbit}: steps at {step_dates} (04's step epochs {[round(step_epoch, 4) for _, step_epoch in steps]})")

# %% [markdown]
# ## 6. The step maps, the box median and the burst-seam correction
#
# The step maps $A_k(\mathbf{x})$ and the rate map $v(\mathbf{x})$ are masked
# with the fit mask, and the box median of each step map is removed,
# $A_k(\mathbf{x}) \leftarrow A_k(\mathbf{x}) - \operatorname{median}_{\text{box}} A_k$,
# because the datum of the InSAR series (GNSS station ZOA1) is an arbitrary
# constant here.
#
# **Burst-seam correction.** Each pixel at longitude $\lambda$ and latitude
# $\varphi$ (deg) is placed relative to the seam by
# $\eta = \varphi - s\,(\lambda + 122^\circ)$, and the seam is
# $\eta = \ell$, with $\ell = b + \Delta / 111.2\ \text{km}$, where $s$ and $b$
# are the slope and intercept of the polygon line (section 2) and $\Delta$ the
# shift `burst_seam_shift_m` (m). For each event $k$ the offset across the seam
# is
#
# $$O_k = \operatorname{median}_{\ell < \eta < \ell + w} A_k - \operatorname{median}_{\ell - w < \eta < \ell} A_k,$$
#
# where $O_k$ is the offset, north band minus south band (mm), and
# $w$ = `burst_band_deg` the band width (deg). Pixels within `burst_far_m` of
# the trace are left out of both bands, and each band needs `burst_min_pixels`
# finite pixels. The side of the seam holding most of the track's valid pixels
# is held fixed and $O_k$ is removed from the other side: $A_k \leftarrow A_k +
# O_k$ south of the seam, or $A_k \leftarrow A_k - O_k$ north of it.
#
# The box median is not removed again after the correction, so the choice of the
# fixed side shifts a whole map by a constant. Every value taken from these
# maps is a difference (east band minus west band, or a profile zeroed on its
# far field), and the slip models fit a constant per data set, so a constant
# does not change them.
#
# Check: on the mean of the five uncorrected step maps, the jump across the
# shifted seam (north band minus south band) must be larger in magnitude than
# the jump across the polygon line (asserted, printed). Every offset must be
# below 2 mm (asserted).

# %%
event_maps = {}
burst_log = []
for dic in insar_dic_list:
    orbit = dic["ORBIT_DIRECTION"]
    fit = fits[orbit]
    step_keys = [f"step{step_date}" for step_date, _ in fit["steps"]]
    step_maps = np.stack([fit["coef"][step_key] for step_key in step_keys]).astype(float)            # (n_events, nr, nc), mm
    step_maps_std = np.stack([fit["coef"][step_key + "Std"] for step_key in step_keys]).astype(float)
    rate = fit["coef"]["velocity"].astype(float)                                                      # mm/yr
    step_maps[:, ~fit["keep"]] = np.nan
    rate[~fit["keep"]] = np.nan
    step_maps -= np.nanmedian(step_maps, axis=(1, 2))[:, None, None]                                  # box median per event
    step_maps_raw = step_maps.copy()
    if orbit in burst_lines:
        far_from_trace = d_trace > burst_far_m
        # the seam: the polygon line shifted by burst_seam_shift_m, checked on the mean of the uncorrected maps
        # against the polygon line itself
        mean_raw = np.nanmean(step_maps, axis=0)
        located = []
        for line_slope, poly_lat in burst_lines[orbit]:
            above_line = lat_g - line_slope * (lon_g + 122.0)
            jumps = {}
            for name, shift_m in (("polygon", 0.0), ("seam", burst_seam_shift_m[orbit])):
                lat_line = poly_lat + shift_m / 111.2e3
                north = (above_line > lat_line) & (above_line < lat_line + burst_band_deg) & far_from_trace & np.isfinite(mean_raw)
                south = (above_line < lat_line) & (above_line > lat_line - burst_band_deg) & far_from_trace & np.isfinite(mean_raw)
                assert min(north.sum(), south.sum()) >= burst_min_pixels, f"[{orbit}] {name} line: too few band pixels"
                jumps[name] = float(np.median(mean_raw[north]) - np.median(mean_raw[south]))
            print(f"  [{orbit}] seam {burst_seam_shift_m[orbit]:+.0f} m along track from the polygon line {poly_lat:.4f} N (at -122 E): "
                  f"mean-map jump {jumps['seam']:+.2f} mm there, {jumps['polygon']:+.2f} mm at the polygon line")
            assert abs(jumps["seam"]) > abs(jumps["polygon"]), f"[{orbit}] the jump at the shifted seam does not exceed the jump at the polygon line: {jumps}"
            located.append(dict(slope=line_slope, line_lat=poly_lat + burst_seam_shift_m[orbit] / 111.2e3, polygon_line_lat=poly_lat,
                                seam_shift_m=burst_seam_shift_m[orbit], mean_jump_at_seam_mm=jumps["seam"], mean_jump_at_polygon_mm=jumps["polygon"]))
        burst_lines[orbit] = [(seam["slope"], seam["line_lat"]) for seam in located]      # the shifted seams, drawn in section 13
        for seam in located:
            line_slope, line_lat = seam["slope"], seam["line_lat"]
            above_line = lat_g - line_slope * (lon_g + 122.0)                    # > line_lat north of the burst line
            north_band = (above_line > line_lat) & (above_line < line_lat + burst_band_deg) & far_from_trace
            south_band = (above_line < line_lat) & (above_line > line_lat - burst_band_deg) & far_from_trace
            north_side = above_line > line_lat
            south_side = above_line < line_lat
            # hold fixed the side with most of the track's valid (fit-mask) pixels in the box; the other side moves
            n_valid_north = int((fit["keep"] & north_side).sum())
            n_valid_south = int((fit["keep"] & south_side).sum())
            if n_valid_north >= n_valid_south:
                fixed_side = "north"
            else:
                fixed_side = "south"
            print(f"  [{orbit}] lat {line_lat:.3f} burst line: valid pixels north {n_valid_north}, south {n_valid_south}; "
                  f"{fixed_side} side held fixed")
            for i, (_, step_epoch) in enumerate(fit["steps"]):
                n_north = int(np.isfinite(step_maps[i][north_band]).sum())
                n_south = int(np.isfinite(step_maps[i][south_band]).sum())
                assert min(n_north, n_south) >= burst_min_pixels, \
                    f"[{orbit}] event {step_epoch:.2f}: only {n_north} north / {n_south} south pixels in the burst bands"
                burst_offset = np.nanmedian(step_maps[i][north_band]) - np.nanmedian(step_maps[i][south_band])   # north minus south
                if fixed_side == "north":
                    step_maps[i][south_side] += burst_offset
                else:
                    step_maps[i][north_side] -= burst_offset
                burst_log.append(dict(track=orbit, line_lat=line_lat, step_epoch=step_epoch, offset_mm=burst_offset,
                                      n_north=n_north, n_south=n_south, fixed_side=fixed_side,
                                      polygon_line_lat=seam["polygon_line_lat"], seam_shift_m=seam["seam_shift_m"],
                                      mean_jump_at_seam_mm=seam["mean_jump_at_seam_mm"], mean_jump_at_polygon_mm=seam["mean_jump_at_polygon_mm"]))
                print(f"  [{orbit}] event {step_epoch:.2f}: offset (north - south) across the lat {line_lat:.3f} burst line "
                      f"{burst_offset:+.2f} mm removed ({n_north} / {n_south} pixels)")
    event_maps[orbit] = dict(ev=step_maps, ev_raw=step_maps_raw, ev_std=step_maps_std, rate=rate)
burst_df = pd.DataFrame(burst_log)
assert (burst_df.offset_mm.abs() < 2.0).all(), "a burst offset of 2 mm or more; inspect the step maps"
assert sorted(set(burst_df.track)) == sorted(burst_corrected), f"burst correction ran on {sorted(set(burst_df.track))}, not on {burst_corrected}"

# %% [markdown]
# ## 7. Check: the near-fault LOS step of every event on both tracks
#
# The near-fault step of a map $u(\mathbf{x})$ is
#
# $$S = \operatorname{median}_{\,d_1 < d < d_2} u - \operatorname{median}_{\,-d_2 < d < -d_1} u,$$
#
# where $S$ is the step (mm, or mm/yr for a rate map), $d$ the signed distance
# from the trace (m, section 3), and $d_1$, $d_2$ = 30 and 150 m
# (`xfault_step_m`, the aperture of notebook 04). Only pixels between the
# latitudes of the 90 % band of notebook 04 (the band Fig 1 stacks) are used.
#
# Right-lateral creep projects with opposite signs into the two geometries, so
# the ascending step is expected negative and the descending step positive. The
# expected ratio of the two is
#
# $$R = \frac{\hat{\mathbf{l}}_{\text{asc}} \cdot \hat{\mathbf{f}}}{\hat{\mathbf{l}}_{\text{desc}} \cdot \hat{\mathbf{f}}}, \qquad \hat{\mathbf{l}} = \left(-\sin i \sin \alpha,\ \sin i \cos \alpha,\ \cos i\right), \qquad \hat{\mathbf{f}} = (\sin\theta,\ \cos\theta,\ 0),$$
#
# where $\hat{\mathbf{l}}$ is the unit vector (east, north, up) from the ground
# to the satellite, $i$ the incidence angle (deg), $\alpha$ the LOS azimuth
# (deg, anticlockwise from north, MintPy convention), both the medians over the
# band, $\hat{\mathbf{f}}$ the unit vector along the strike $\theta$ (deg) and
# $R$ the ratio for pure strike slip (dimensionless).
#
# Check: the mean of the event steps is negative on ascending and positive on
# descending, and every descending step is positive (asserted). Ascending
# events with a step that is not negative are printed. The ratio of the
# observed steps is printed beside $R$ and is not asserted.

# %%
in_band = (lat_g >= band.lat_lo) & (lat_g <= band.lat_hi)
east = in_band & (d_signed > xfault_step_m[0]) & (d_signed < xfault_step_m[1])
west = in_band & (d_signed < -xfault_step_m[0]) & (d_signed > -xfault_step_m[1])
los_vector = {}
for orbit in event_maps:
    los_vector[orbit] = utils.los_unit_vector(np.nanmedian(fits[orbit]["inc"][in_band]), np.nanmedian(fits[orbit]["azi"][in_band]))
strike_vector = utils.horizontal_unit_vector(strike)
los_dot_strike_asc = float(los_vector["ascending"] @ strike_vector)
los_dot_strike_desc = float(los_vector["descending"] @ strike_vector)
geom_ratio = los_dot_strike_asc / los_dot_strike_desc
print(f"LOS . strike: asc {los_dot_strike_asc:+.3f}, desc {los_dot_strike_desc:+.3f}; geometric asc/desc ratio {geom_ratio:+.2f}")


def block_step(field_map):
    """Near-fault step of a map inside the 90% band.

    Parameters
    ----------
    field_map : ndarray
        (n_rows, n_cols) map on the box grid (mm or mm/yr).

    Returns
    -------
    float
        Median of the east band minus median of the west band, xfault_step_m from the trace.
    """
    return float(np.nanmedian(field_map[east]) - np.nanmedian(field_map[west]))


step_rows = []
for i, (_, step_epoch) in enumerate(fits["ascending"]["steps"]):
    row = dict(step_epoch=step_epoch)
    for orbit in event_maps:
        row[f"{orbit}_step_mm"] = block_step(event_maps[orbit]["ev"][i])
    row["ratio_asc_desc"] = row["ascending_step_mm"] / row["descending_step_mm"]
    step_rows.append(row)
steps_df = pd.DataFrame(step_rows)
print(steps_df.round(2).to_string(index=False))
mean_asc = steps_df.ascending_step_mm.mean()
mean_desc = steps_df.descending_step_mm.mean()
print(f"mean of the {len(steps_df)} events: asc {mean_asc:+.2f} mm, desc {mean_desc:+.2f} mm, ratio {mean_asc / mean_desc:+.2f} "
      f"(geometric {geom_ratio:+.2f})")
assert mean_asc < 0 < mean_desc, "the mean event has the wrong LOS sign: the geometry or the fault side is wrong"
assert (steps_df.descending_step_mm > 0).all(), "a descending step with the wrong sign"
ascending_not_negative = steps_df[steps_df.ascending_step_mm >= 0]
for _, step_row in ascending_not_negative.iterrows():
    print(f"Note {step_row.step_epoch:.2f}: ascending step {step_row.ascending_step_mm:+.2f} mm, ratio {step_row.ratio_asc_desc:+.2f}: "
          f"ascending step not negative (not asserted)")
print("Check passed: the mean event steps negative on ascending and positive on descending; every descending step is positive")

# %% [markdown]
# ## 8. The mean event per track
#
# At every pixel where all $K$ step maps are valid,
#
# $$\bar{A}(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K} A_k(\mathbf{x}), \qquad s_A(\mathbf{x}) = \sqrt{\frac{1}{K-1}\sum_{k=1}^{K}\left(A_k(\mathbf{x}) - \bar{A}(\mathbf{x})\right)^2}, \qquad \sigma_{\bar{A}}(\mathbf{x}) = \frac{1}{\sqrt{K}}\sqrt{\frac{1}{K}\sum_{k=1}^{K}\sigma_{A_k}^2(\mathbf{x})},$$
#
# where $\bar{A}$ is the mean event (mm), $s_A$ the per-pixel scatter of the
# events about it (1 s.d., mm), $\sigma_{A_k}$ the formal error of step $k$
# (section 4, mm) and $\sigma_{\bar{A}}$ the formal error of the mean (mm).
# Everything is written to `tulepond_steps_<track>.h5`, and the mean event as a
# flat table of pixels (lon, lat, LOS, error, incidence, azimuth, scatter) to
# `csi_mean_event_<track>.npz`.

# %%
def mean_event_error(step_std_maps, n_events):
    """Formal 1-sigma error of the mean of n_events step maps.

    Parameters
    ----------
    step_std_maps : ndarray
        (n_events, n_rows, n_cols) formal errors of the steps (mm).
    n_events : int
        Number of events.

    Returns
    -------
    ndarray
        The RMS of the per-pixel formal errors divided by sqrt(n_events) (mm).
    """
    return np.sqrt(np.nanmean(step_std_maps ** 2, 0)) / np.sqrt(n_events)


mean_event = {}
for dic in insar_dic_list:
    orbit = dic["ORBIT_DIRECTION"]
    fit = fits[orbit]
    maps = event_maps[orbit]
    n_kept = len(fit["steps"])
    kept_maps = maps["ev"]
    kept_std = maps["ev_std"]
    n_valid = np.isfinite(kept_maps).sum(0)
    mean = np.where(n_valid == n_kept, np.nanmean(kept_maps, 0), np.nan)
    scatter = np.where(n_valid == n_kept, np.nanstd(kept_maps, 0, ddof=1), np.nan)
    err_mean = mean_event_error(kept_std, n_kept)
    good = np.isfinite(mean)
    mean_event[orbit] = dict(mean=mean, scatter=scatter, err=err_mean, n=n_kept)
    f_h5 = os.path.join(out_dir, f"tulepond_steps_{orbit}.h5")
    with h5py.File(f_h5, "w") as out_h5:
        for i, (step_date, step_epoch) in enumerate(fit["steps"]):
            event_dataset = out_h5.create_dataset(f"event{i}", data=maps["ev"][i].astype(np.float32))
            event_dataset.attrs.update(dict(step_date=step_date, step_epoch=step_epoch))
            out_h5.create_dataset(f"event{i}Std", data=maps["ev_std"][i].astype(np.float32))
            out_h5.create_dataset(f"event{i}_raw", data=maps["ev_raw"][i].astype(np.float32))
        box_datasets = dict(mean_event=mean, event_scatter=scatter, mean_event_err=err_mean, rate=maps["rate"],
                            incidenceAngle=fit["inc"], azimuthAngle=fit["azi"], lon=lon_g, lat=lat_g, coh_mask=fit["keep"])
        for dataset_name, values in box_datasets.items():
            out_h5.create_dataset(dataset_name, data=np.asarray(values, dtype=np.float32))
        for attr_name, attr_value in attrs.items():
            out_h5.attrs[attr_name] = str(attr_value)
        out_h5.attrs.update(dict(X_FIRST=str(x0 + c0 * dx), Y_FIRST=str(y0 + r0 * dy), WIDTH=str(nc), LENGTH=str(nr), UNIT="mm",
                                 n_events=n_kept, coherence=str(coh_timeseries),
                                 note="LOS + toward satellite; box median removed per event; azimuthAngle = LOS ground->sat, anticlockwise from N",
                                 provenance=utils.provenance_stamp(notebook, repo_dir)))
    np.savez(os.path.join(out_dir, f"csi_mean_event_{orbit}.npz"), lon=lon_g[good], lat=lat_g[good], vel_mm=mean[good],
             err_mm=err_mean[good], incidence=fit["inc"][good], azimuth=fit["azi"][good], scatter_mm=scatter[good])
    print(f"[{orbit}] mean of {n_kept} events: {good.sum()} pixels, 2-98% {np.nanpercentile(mean, 2):+.1f} to {np.nanpercentile(mean, 98):+.1f} mm, "
          f"median scatter {np.nanmedian(scatter):.2f} mm; wrote {os.path.basename(f_h5)} and csi_mean_event_{orbit}.npz")

# %% [markdown]
# ## 9. Fault-parallel and vertical maps
#
# From here on the maps are masked at the stricter temporal coherence
# `coh_maps`; that mask lies inside the fit mask (asserted), so it only removes
# pixels. The inter-event rate, the mean event and each event are decomposed
# with MintPy's `asc_desc2horz_vert` (Yunjun et al. 2019), as in notebook 02.
# At each pixel valid on both tracks it solves
#
# $$\begin{pmatrix} d_{\text{asc}} \\ d_{\text{desc}} \end{pmatrix} = \begin{pmatrix} \sin i_{\text{asc}} \cos(\alpha_{\text{asc}} - \psi) & \cos i_{\text{asc}} \\ \sin i_{\text{desc}} \cos(\alpha_{\text{desc}} - \psi) & \cos i_{\text{desc}} \end{pmatrix} \begin{pmatrix} u_\parallel \\ u_z \end{pmatrix}, \qquad \psi = -\theta,$$
#
# where $d$ is the LOS displacement or rate of each track (mm or mm/yr,
# positive toward the satellite), $i$ the incidence angle (deg), $\alpha$ the
# LOS azimuth (deg, anticlockwise from north), $\psi$ the azimuth of the
# horizontal direction in MintPy's anticlockwise convention (deg), $\theta$ the
# strike of section 3 (deg clockwise from north), $u_\parallel$ the
# fault-parallel component (positive toward the strike direction, south-east)
# and $u_z$ the vertical component (positive up), in the units of $d$. The
# fault-perpendicular motion is assumed zero. MintPy takes $i$ and $\alpha$ as
# the medians over windows of `geom_step` x `geom_step` pixels and solves with
# the pseudo-inverse. For pure strike-slip creep the vertical step across the
# trace is expected near zero (printed).

# %%
def decompose(los_asc, los_desc, inc_both, azi_both):
    """Fault-parallel and vertical maps from the two LOS maps (MintPy asc_desc2horz_vert, Yunjun et al. 2019).

    Parameters
    ----------
    los_asc, los_desc : ndarray
        (n_rows, n_cols) LOS maps (mm or mm/yr, positive toward the satellite); only pixels valid on both are used.
    inc_both, azi_both : ndarray
        (2, n_rows, n_cols) incidence and azimuth angles (deg), ascending first.

    Returns
    -------
    fault_parallel, up : ndarray
        (n_rows, n_cols) components along the strike and vertical, in the units of the input.
    """
    valid_both = np.isfinite(los_asc) & np.isfinite(los_desc)
    los_stack = np.stack([los_asc, los_desc]).astype(np.float32)
    los_stack[:, ~valid_both] = np.nan
    fault_parallel, up = asc_desc2horz_vert(los_stack, inc_both, azi_both, horz_az_angle=-strike, step=geom_step)
    return fault_parallel.astype(float), up.astype(float)


# the maps from here on (decomposition, profiles, Fig 2) use the stricter coh_maps mask
if coh_maps != coh_timeseries:
    for dic in insar_dic_list:
        orbit = dic["ORBIT_DIRECTION"]
        with h5py.File(dic["mask_file"][coh_maps], "r") as mask_h5:
            mask_x_first = float(mask_h5.attrs["X_FIRST"])
            mask_y_first = float(mask_h5.attrs["Y_FIRST"])
            col_start = int(round((x0 + c0 * dx - mask_x_first) / dx))
            row_start = int(round((y0 + r0 * dy - mask_y_first) / dy))
            mask_name = [name for name in mask_h5.keys() if name != "date"][0]
            mask_maps = mask_h5[mask_name][row_start:row_start + nr, col_start:col_start + nc].astype(bool)
        assert not (mask_maps & ~fits[orbit]["keep"]).any(), f"{orbit}: coh_maps mask not inside the fit mask"
        for field in ("mean", "scatter", "err"):
            mean_event[orbit][field] = np.where(mask_maps, mean_event[orbit][field], np.nan)
        event_maps[orbit]["rate"] = np.where(mask_maps, event_maps[orbit]["rate"], np.nan)
        event_maps[orbit]["ev"] = np.where(mask_maps[None], event_maps[orbit]["ev"], np.nan)
        event_maps[orbit]["ev_raw"] = np.where(mask_maps[None], event_maps[orbit]["ev_raw"], np.nan)   # read after here only by section 13
        print(f"[{orbit}] maps masked at temporal coherence {coh_maps}: {mask_maps.mean():.0%} of the box "
              f"(fit mask {coh_timeseries}: {fits[orbit]['keep'].mean():.0%})")

inc_both = np.stack([fits["ascending"]["inc"], fits["descending"]["inc"]]).astype(np.float32)
azi_both = np.stack([fits["ascending"]["azi"], fits["descending"]["azi"]]).astype(np.float32)
fp_rate, up_rate = decompose(event_maps["ascending"]["rate"], event_maps["descending"]["rate"], inc_both, azi_both)
fp_evt, up_evt = decompose(mean_event["ascending"]["mean"], mean_event["descending"]["mean"], inc_both, azi_both)
fp_events = {}
for i, (_, step_epoch) in enumerate(fits["ascending"]["steps"]):
    fp_events[step_epoch], _ = decompose(event_maps["ascending"]["ev"][i], event_maps["descending"]["ev"][i], inc_both, azi_both)

print(f"inter-event FP rate step east - west ({xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m, 90% band): "
      f"{block_step(fp_rate):+.2f} mm/yr; Up {block_step(up_rate):+.2f} mm/yr")
print(f"mean event FP step: {block_step(fp_evt):+.2f} mm; Up {block_step(up_evt):+.2f} mm (expected near zero for pure strike-slip)")
for step_epoch, fp_event_map in fp_events.items():
    print(f"  event {step_epoch:.2f}: FP step {block_step(fp_event_map):+.2f} mm")
fp_maps = dict(fp_rate=fp_rate, up_rate=up_rate, fp_event=fp_evt, up_event=up_evt,
               lon=lon_g, lat=lat_g, X=Xg, Y=Yg, strike=strike, n_events=mean_event["ascending"]["n"])
for step_epoch, fp_event_map in fp_events.items():
    fp_maps[f"fp_event_{step_epoch:.2f}"] = fp_event_map                      # named by the event's step epoch
np.savez(os.path.join(out_dir, "fp_maps.npz"), **fp_maps)

# %% [markdown]
# ## 10. The cross-fault profiles X-X' and Y-Y'
#
# Each profile runs perpendicular to the local strike $\theta_\ell$ at its
# point $\mathbf{x}_0$ (the smoothed strike of the trace vertex nearest the
# point, not the box mean). Each pixel gets a cross-fault and an along-strike
# coordinate,
#
# $$p = (\mathbf{x} - \mathbf{x}_0) \cdot \hat{\mathbf{c}}, \qquad q = (\mathbf{x} - \mathbf{x}_0) \cdot \hat{\mathbf{a}}, \qquad \hat{\mathbf{c}} = \left(\sin(\theta_\ell - 90^\circ),\ \cos(\theta_\ell - 90^\circ)\right), \qquad \hat{\mathbf{a}} = \left(-\cos(\theta_\ell - 90^\circ),\ \sin(\theta_\ell - 90^\circ)\right),$$
#
# where $\mathbf{x}$ and $\mathbf{x}_0$ are UTM coordinates (m), $p$ the
# distance across the fault (m, positive east) and $q$ the distance along
# strike (m). The profile holds every finite pixel with $|q| \le$ `swath_m` and
# $|p| \le$ `cross_m`.
#
# The rate profile is zeroed at the fault,
# $v(p) \leftarrow v(p) - \tfrac{1}{2}\left[\operatorname{median}_{-r \le p < 0} v + \operatorname{median}_{0 < p \le r} v\right]$
# with $r$ = `rate_zero_m`, and the event profile on its
# far field, $e(p) \leftarrow e(p) - \operatorname{median}_{|p| \ge p_f} e$ with
# $p_f$ = `event_far_m`, where $v$ is the rate (mm/yr) and $e$ the mean event
# (mm). The near-fault step $S$ of section 7 is then taken on each profile.
#
# The near-fault amplitude of each block is measured against that block's own
# far field,
#
# $$B_W = \operatorname{median}_{d_1 \le -p < d_2} e - \operatorname{median}_{f_1 \le -p < f_2} e, \qquad B_E = \operatorname{median}_{d_1 \le p < d_2} e - \operatorname{median}_{f_1 \le p < f_2} e,$$
#
# where $B_W$ and $B_E$ are the west and east block amplitudes (mm), $d_1$,
# $d_2$ = 30 and 150 m the near-fault band and $f_1$, $f_2$ = 1500 and 2000 m
# the far field (`far_m`). Its standard error is the standard deviation of
# $B$ over `n_boot_blocks` bootstrap resamples, each drawing the near-band and
# far-field pixels with replacement [citation needed]. The bootstrap treats the
# pixels as independent.
#
# The same functions run on the fault-parallel maps (the values quoted in the
# text), on the vertical maps (section 11) and on each track's LOS maps (shown
# in Fig 2).

# %%
def local_strike(lonlat):
    """Smoothed local strike of the trace (deg, compass bearing) at the trace vertex nearest lonlat."""
    point_x, point_y = to_utm.transform(*lonlat)
    vertex = int(tree.query([float(point_x), float(point_y)])[1])
    return float(x_fault.az_smooth.iloc[vertex] + 90.0)


def profile(field_map, lonlat, strike_deg=None):
    """Cross-fault profile of a map through a point.

    Parameters
    ----------
    field_map : ndarray
        (n_rows, n_cols) map on the box grid.
    lonlat : tuple of float
        (lon, lat) of the profile point (deg).
    strike_deg : float, optional
        Strike to use (deg); the local strike at the point if None.

    Returns
    -------
    distance : ndarray
        Cross-fault distance of each pixel (m, positive east), sorted.
    values : ndarray
        The map values at those pixels.
    ends : tuple of float
        The line's end points in UTM (x0, y0, x1, y1), m.
    strike_deg : float
        The strike used (deg).
    """
    point_x, point_y = to_utm.transform(*lonlat)
    point_x = float(point_x)
    point_y = float(point_y)
    if strike_deg is None:
        strike_deg = local_strike(lonlat)
    cross_e = np.sin(np.deg2rad(strike_deg - 90.0))             # cross-fault unit vector (+ east)
    cross_n = np.cos(np.deg2rad(strike_deg - 90.0))
    along_e = -cross_n                                          # along-strike unit vector
    along_n = cross_e
    dist_across = (Xg - point_x) * cross_e + (Yg - point_y) * cross_n
    dist_along = (Xg - point_x) * along_e + (Yg - point_y) * along_n
    in_swath = (np.abs(dist_along) <= swath_m) & (np.abs(dist_across) <= cross_m) & np.isfinite(field_map)
    distance = dist_across[in_swath]
    values = field_map[in_swath]
    order = np.argsort(distance)
    ends = (point_x - cross_m * cross_e, point_y - cross_m * cross_n, point_x + cross_m * cross_e, point_y + cross_m * cross_n)
    return distance[order], values[order], ends, strike_deg


def block_amplitudes(distance, values):
    """Near-fault amplitude of each block of a profile against its own far field.

    The bootstrap uses its own generator seeded with block_seed, so the result does not depend on what was
    drawn before.

    Parameters
    ----------
    distance : ndarray
        Cross-fault distance (m, positive east).
    values : ndarray
        Profile values (mm).

    Returns
    -------
    dict
        Per block ("W", "E"): amp (median of the xfault_step_m band minus median of the far_m band, mm), se
        (bootstrap standard error, mm), far (far-field median, mm), n_near and n_far (pixel counts).
    """
    rng = np.random.default_rng(block_seed)
    out = {}
    for block_name, side in (("W", -1), ("E", 1)):
        near = (side * distance >= xfault_step_m[0]) & (side * distance < xfault_step_m[1])
        far = (side * distance >= far_m[0]) & (side * distance < far_m[1])
        assert near.sum() >= 10 and far.sum() >= 50, (block_name, near.sum(), far.sum())
        near_values = values[near]
        far_values = values[far]
        boot = np.empty(n_boot_blocks)
        for draw in range(n_boot_blocks):
            near_resampled = rng.choice(near_values, near_values.size)
            far_resampled = rng.choice(far_values, far_values.size)
            boot[draw] = np.median(near_resampled) - np.median(far_resampled)
        out[block_name] = dict(amp=float(np.median(near_values) - np.median(far_values)), se=float(boot.std()),
                               far=float(np.median(far_values)), n_near=int(near.sum()), n_far=int(far.sum()))
    return out


def near_fault_step(distance, values):
    """Near-fault step of a profile: median of the east band minus median of the west band, xfault_step_m from
    the trace.

    Parameters
    ----------
    distance : ndarray
        Cross-fault distance (m, positive east).
    values : ndarray
        Profile values (mm or mm/yr).

    Returns
    -------
    float
        The step, in the units of values.
    """
    band_lo, band_hi = xfault_step_m
    east_values = values[(distance > band_lo) & (distance < band_hi)]
    west_values = values[(distance < -band_lo) & (distance > -band_hi)]
    return float(np.median(east_values) - np.median(west_values))


def zeroed_profiles(rate_map, event_map, lonlat):
    """Rate and event profiles through a point, zeroed, with their near-fault steps.

    The rate profile is zeroed on the mid-point of the two block medians within rate_zero_m of the trace, the
    event profile on its median at |distance| >= event_far_m.

    Parameters
    ----------
    rate_map : ndarray
        (n_rows, n_cols) rate map (mm/yr).
    event_map : ndarray
        (n_rows, n_cols) event map (mm).
    lonlat : tuple of float
        (lon, lat) of the profile point (deg).

    Returns
    -------
    dict
        d, v (rate profile distance in m and values), de, ve (event profile), ends, lonlat, strike, and
        rate_step, event_step (near-fault steps).
    """
    rate_dist, rate_values, ends, strike_used = profile(rate_map, lonlat)
    near_trace = np.abs(rate_dist) <= rate_zero_m
    rate_values = rate_values - 0.5 * (np.median(rate_values[near_trace & (rate_dist < 0)]) + np.median(rate_values[near_trace & (rate_dist > 0)]))
    event_dist, event_values, _, _ = profile(event_map, lonlat, strike_used)
    far_field = np.abs(event_dist) >= event_far_m
    assert far_field.sum() >= min_far_pixels, f"only {far_field.sum()} far-field pixels to zero the event profile at {lonlat}"
    event_values = event_values - np.median(event_values[far_field])
    return dict(d=rate_dist, v=rate_values, de=event_dist, ve=event_values, ends=ends, lonlat=lonlat, strike=strike_used,
                rate_step=near_fault_step(rate_dist, rate_values), event_step=near_fault_step(event_dist, event_values))


# fault-parallel profiles (the values quoted in the text)
profile_names = list(profile_pts)
profiles = {}
for name, lonlat in profile_pts.items():
    fp_profile = zeroed_profiles(fp_rate, fp_evt, lonlat)
    profiles[name] = fp_profile
    pd.DataFrame(dict(d_m=fp_profile["d"], fp_rate_mm_yr=fp_profile["v"])).to_csv(os.path.join(out_dir, f"profile_{name}{name}_rate.csv"), index=False)
    pd.DataFrame(dict(d_m=fp_profile["de"], fp_event_mm=fp_profile["ve"])).to_csv(os.path.join(out_dir, f"profile_{name}{name}_event.csv"), index=False)
    print(f"{name}-{name}' FP (local strike {fp_profile['strike']:.1f} deg): {fp_profile['d'].size} rate pixels, {fp_profile['de'].size} event pixels; "
          f"near-fault steps ({xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m): "
          f"rate {fp_profile['rate_step']:+.2f} mm/yr, event {fp_profile['event_step']:+.2f} mm")

# LOS profiles per track, as shown in Fig 2 (positive toward the satellite, so right-lateral slip steps
# negative on ascending and positive on descending)
tracks = ["ascending", "descending"]
los_profiles = {}
for orbit in tracks:
    los_profiles[orbit] = {}
    for name, lonlat in profile_pts.items():
        los_profile = zeroed_profiles(event_maps[orbit]["rate"], mean_event[orbit]["mean"], lonlat)
        pd.DataFrame(dict(d_m=los_profile["d"], los_rate_mm_yr=los_profile["v"])).to_csv(
            os.path.join(out_dir, f"profile_{name}{name}_rate_{orbit}.csv"), index=False)
        pd.DataFrame(dict(d_m=los_profile["de"], los_event_mm=los_profile["ve"])).to_csv(
            os.path.join(out_dir, f"profile_{name}{name}_event_{orbit}.csv"), index=False)
        los_profile["blocks"] = block_amplitudes(los_profile["de"], los_profile["ve"])
        west_block = los_profile["blocks"]["W"]
        east_block = los_profile["blocks"]["E"]
        print(f"[{orbit[:3]}] {name}-{name}' LOS: rate step {los_profile['rate_step']:+.2f} mm/yr, event step {los_profile['event_step']:+.2f} mm; "
              f"event per block W {west_block['amp']:+.2f} +/- {west_block['se']:.2f}, E {east_block['amp']:+.2f} +/- {east_block['se']:.2f} mm")
        los_profiles[orbit][name] = los_profile
xx_ratio = los_profiles["ascending"]["X"]["event_step"] / los_profiles["descending"]["X"]["event_step"]
print(f"X-X' event asc/desc LOS step ratio {xx_ratio:+.2f} (geometric, pure strike-slip: {geom_ratio:+.2f})")
assert los_profiles["ascending"]["X"]["event_step"] < 0 < los_profiles["descending"]["X"]["event_step"], "the X-X' LOS event steps have the wrong signs"

# %% [markdown]
# ## 11. The maps and profiles for Fig 2
#
# Fig 2 is drawn in notebook 09 after the slip model, so that it can include
# the model row. It needs the per-track LOS maps (masked at `coh_maps`), the
# fault-parallel and vertical maps and the profiles, written to one file,
# `fig2_inputs.npz`. The fault-parallel and vertical profiles get their block
# amplitudes (section 10) here, and the vertical profiles are zeroed in the
# same way as the others.

# %%
for name in profile_names:
    profiles[name]["blocks"] = block_amplitudes(profiles[name]["de"], profiles[name]["ve"])
# the vertical field through the same lines (Fig 2's second column)
up_profiles = {}
for name, lonlat in profile_pts.items():
    up_profiles[name] = zeroed_profiles(up_rate, up_evt, lonlat)
    up_profiles[name]["blocks"] = block_amplitudes(up_profiles[name]["de"], up_profiles[name]["ve"])

fig2_in = dict(lon=lon_g, lat=lat_g, X=Xg, Y=Yg, epsg=epsg, strike=strike, coh_maps=coh_maps,
               n_events=mean_event["ascending"]["n"], cross_m=cross_m, names=np.array(profile_names),
               fp_rate=fp_rate, fp_event=fp_evt, up_rate=up_rate, up_event=up_evt)
for orbit in tracks:
    fig2_in[f"rate_{orbit}"] = event_maps[orbit]["rate"]
    fig2_in[f"event_{orbit}"] = mean_event[orbit]["mean"]
profile_sets = {"fp": profiles, "up": up_profiles}            # field name in the npz key -> its profiles
for orbit in tracks:
    profile_sets[orbit] = los_profiles[orbit]
for field, profile_set in profile_sets.items():
    for name in profile_names:
        one_profile = profile_set[name]
        for array_name in ("d", "v", "de", "ve", "ends"):
            fig2_in[f"prof_{field}_{name}_{array_name}"] = np.asarray(one_profile[array_name])
        for block_name in ("W", "E"):
            fig2_in[f"prof_{field}_{name}_{block_name}_amp"] = one_profile["blocks"][block_name]["amp"]
            fig2_in[f"prof_{field}_{name}_{block_name}_se"] = one_profile["blocks"][block_name]["se"]
f_fig2_in = os.path.join(out_dir, "fig2_inputs.npz")
np.savez(f_fig2_in, provenance=utils.provenance_stamp(notebook, repo_dir), **fig2_in)
print("wrote", f_fig2_in)

# %% [markdown]
# ## 12. Mean-event statistics: block deflections, decay distance and the Lake Elizabeth vertical
#
# Three quantities quoted in the text, all from the mean-event maps (registered
# in section 14).
#
# 1. **Near-fault deflection of each block** on the fault-parallel profiles X-X'
#    and Y-Y': the block amplitudes $B_W$ and $B_E$ of section 10 (the same
#    definition as the LOS blocks of Fig 2), with their bootstrap standard
#    errors.
# 2. **Decay distance** of the mean event on each side of X-X', the distance
#    from the trace beyond which the fault-parallel event profile $e(p)$
#    (zeroed on its far field, section 10) stays within the scatter of the far
#    field. With $r = \pm p$ the distance outward from the trace on the west
#    ($-$) or east ($+$) side (m), the far-field level and scatter of that side
#    are
#
#    $$L = \operatorname{median}_{f_1 \le r < f_2} e, \qquad \sigma_f = 1.4826\, \operatorname{median}_{f_1 \le r < f_2} \lvert e - L \rvert,$$
#
#    where $L$ is the far-field level (mm), $\sigma_f$ the robust standard
#    deviation from the median absolute deviation (mm) [citation needed], and
#    $f_1$, $f_2$ = 1500 and 2000 m (`far_m`). The profile is binned outward,
#    $r_b < r \le r_b + \delta$, with $\delta$ = `decay_bin_m` = 150 m and
#    $r_b = 0, \delta, \dots$ up to $f_1$, and $m_b$ is the median of $e$ in
#    each bin with at least 3 pixels (the running median Fig 2 draws). A bin is
#    within the far field if $\lvert m_b - L \rvert \le \sigma_f$. The decay
#    distance $D$ (m) is the inner edge $r_b$ of the first bin of the run of bins
#    that are all within the far field out to $f_1$; the outermost bin must be
#    within (asserted). The resolution is one bin, 150 m.
# 3. **The Lake Elizabeth vertical**, from the mean-event vertical map $u_z$
#    (section 9). The area $\mathcal{A}$ is every valid pixel within $R$ =
#    `lake_elizabeth_radius_m` (1 km) of the lake centre (`lake_elizabeth_lonlat`)
#    that lies east of the trace beyond the near-fault band and short of the far
#    field, $d_2 \le d < p_f$ (150 to 1500 m, $d$ from section 3). The lake is open
#    water with no data, so the area's pixels are its shores. The reference
#    $\mathcal{F}$ is the far field of the event profiles, every valid pixel with
#    $p_f \le \lvert d \rvert <$ `cross_m` (1500 to 2000 m) on both sides and
#    within $\pm R$ along strike of the lake centre. Then
#
#    $$U = \operatorname{median}_{\mathcal{A}} u_z - \operatorname{median}_{\mathcal{F}} u_z, \qquad U_{75} = P_{75,\mathcal{A}}(u_z) - \operatorname{median}_{\mathcal{F}} u_z, \qquad F_{75} = P_{75,\mathcal{F}}(u_z) - \operatorname{median}_{\mathcal{F}} u_z,$$
#
#    where $U$ is the vertical displacement of the area relative to the far
#    field (mm), $P_{75}$ the 75th percentile, $U_{75}$ the upper quartile of the
#    area relative to the far field (mm) and $F_{75}$ the same statistic of the
#    far field itself, the value for noise alone (mm). The standard errors of
#    $U$ and $U_{75}$ are the standard deviations over `n_boot_blocks` bootstrap
#    resamples of the pixels of $\mathcal{A}$ and $\mathcal{F}$ (own generator,
#    `block_seed`) [citation needed]. The difference against each side's far
#    field alone and against the box median is also printed, to show the
#    dependence on the reference.
#
# The bootstraps treat the pixels as independent. The pixels are correlated over
# about 1 km, so the standard errors are lower bounds.

# %%
def decay_distance(distance, values, side):
    """Decay distance of a zeroed event profile on one side of the trace.

    Parameters
    ----------
    distance : ndarray
        Cross-fault distance (m, positive east).
    values : ndarray
        Event profile, zeroed on its far field (mm).
    side : int
        -1 for the west side, +1 for the east side.

    Returns
    -------
    distance : float
        Inner edge of the first decay_bin_m bin from which every bin with >= 3 pixels, out to far_m[0], has a
        median within the far-field level +/- its robust scatter (m).
    far_level, far_scatter : float
        Median and 1.4826 x MAD of the far_m pixels of that side (mm).
    bin_inner, bin_median : ndarray
        Inner edges (m) and medians (mm) of the bins with >= 3 pixels.
    """
    outward = side * distance
    far_values = values[(outward >= far_m[0]) & (outward < far_m[1])]
    assert far_values.size >= 50, f"only {far_values.size} far-field pixels on side {side:+d}"
    far_level = float(np.median(far_values))
    far_scatter = float(1.4826 * np.median(np.abs(far_values - far_level)))
    edges = np.arange(0.0, far_m[0] + decay_bin_m / 2, decay_bin_m)
    bin_inner = []
    bin_median = []
    for inner, outer in zip(edges[:-1], edges[1:]):
        in_bin = (outward > inner) & (outward <= outer)
        if in_bin.sum() >= 3:
            bin_inner.append(float(inner))
            bin_median.append(float(np.median(values[in_bin])))
    bin_inner = np.array(bin_inner)
    bin_median = np.array(bin_median)
    within = np.abs(bin_median - far_level) <= far_scatter
    assert within[-1], f"side {side:+d}: even the last bin before the far field is outside its scatter; no decay distance"
    first_inside = len(within)
    while first_inside > 0 and within[first_inside - 1]:      # walk inward while the bins stay inside
        first_inside -= 1
    return float(bin_inner[first_inside]), far_level, far_scatter, bin_inner, bin_median


# 1 and 2: the block deflections of the fault-parallel profiles (section 11) and the decay distance at X-X'
mean_event_lines = ["mean event, fault-parallel near-fault deflection per block (near "
                    f"{xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m minus the same block's far field {far_m[0]:.0f}-{far_m[1]:.0f} m):"]
for name in profile_names:
    west_block = profiles[name]["blocks"]["W"]
    east_block = profiles[name]["blocks"]["E"]
    mean_event_lines.append(f"   {name}-{name}': W {west_block['amp']:+.2f} +/- {west_block['se']:.2f} mm, E {east_block['amp']:+.2f} +/- {east_block['se']:.2f} mm")
decay = {}
for side_name, side in (("W", -1), ("E", 1)):
    decay[side_name] = decay_distance(profiles["X"]["de"], profiles["X"]["ve"], side)
    distance_m, far_level, far_scatter, bin_inner, bin_median = decay[side_name]
    bin_text = []
    for inner, median in zip(bin_inner, bin_median):
        bin_text.append(f"{inner:.0f}:{median:+.2f}")
    mean_event_lines.append(f"   X-X' decay distance {side_name}: {distance_m:.0f} m (far field {far_level:+.2f} +/- {far_scatter:.2f} mm; "
                            f"bin medians, inner edge m: mm, {' '.join(bin_text)})")

# 3: the Lake Elizabeth vertical
lake_x, lake_y = (float(value) for value in to_utm.transform(*lake_elizabeth_lonlat))
lake_distance = np.hypot(Xg - lake_x, Yg - lake_y)
along_unit = np.array([np.sin(np.deg2rad(strike)), np.cos(np.deg2rad(strike))])      # along strike (east, north)
lake_along = (Xg - lake_x) * along_unit[0] + (Yg - lake_y) * along_unit[1]
lake_centre_vertex = int(tree.query([lake_x, lake_y])[1])
lake_centre_side = float((np.array([lake_x, lake_y]) - np.array([trace_x[lake_centre_vertex], trace_y[lake_centre_vertex]])) @ normal)
assert lake_centre_side > 0, f"the Lake Elizabeth centre {lake_elizabeth_lonlat} is not east of the trace ({lake_centre_side:.0f} m)"
valid_up = np.isfinite(up_evt)
lake_area = (lake_distance <= lake_elizabeth_radius_m) & (d_signed >= xfault_step_m[1]) & (d_signed < event_far_m) & valid_up
lake_far = (np.abs(d_signed) >= event_far_m) & (np.abs(d_signed) < cross_m) & (np.abs(lake_along) <= lake_elizabeth_radius_m) & valid_up
assert lake_area.sum() >= 20 and lake_far.sum() >= 100, f"Lake Elizabeth: {lake_area.sum()} area pixels, {lake_far.sum()} far-field pixels"
assert not (lake_area & lake_far).any(), "the Lake Elizabeth area overlaps the far field it is measured against"
lake_rng = np.random.default_rng(block_seed)                        # its own generator, as in block_amplitudes
area_values = up_evt[lake_area]
far_values = up_evt[lake_far]
lake_boot = np.empty(n_boot_blocks)
lake_boot_p75 = np.empty(n_boot_blocks)
for draw in range(n_boot_blocks):
    area_draw = lake_rng.choice(area_values, area_values.size)
    far_median_draw = np.median(lake_rng.choice(far_values, far_values.size))
    lake_boot[draw] = np.median(area_draw) - far_median_draw
    lake_boot_p75[draw] = np.percentile(area_draw, 75) - far_median_draw
far_median = float(np.median(far_values))
lake_up = dict(value=float(np.median(area_values) - far_median), se=float(lake_boot.std()),
               p75=float(np.percentile(area_values, 75) - far_median), p75_se=float(lake_boot_p75.std()),
               far_p75=float(np.percentile(far_values, 75) - far_median),
               n_area=int(lake_area.sum()), n_far=int(lake_far.sum()), centre_from_trace_m=lake_centre_side)
lake_alternatives = {"east far field only": lake_far & (d_signed > 0), "west far field only": lake_far & (d_signed < 0),
                     "the box median": valid_up}
alternative_text = []
for reference_name, reference in lake_alternatives.items():
    alternative_text.append(f"{reference_name} {np.median(area_values) - np.median(up_evt[reference]):+.2f} mm "
                            f"(upper quartile {np.percentile(area_values, 75) - np.median(up_evt[reference]):+.2f})")
mean_event_lines.append(f"Lake Elizabeth vertical (mean event): {lake_up['value']:+.2f} +/- {lake_up['se']:.2f} mm over {lake_up['n_area']} pixels within "
                        f"{lake_elizabeth_radius_m:.0f} m of {lake_elizabeth_lonlat} (centre {lake_centre_side:.0f} m east of the trace; pixels "
                        f">= {xfault_step_m[1]:.0f} m east of it) relative to {lake_up['n_far']} far-field pixels ({event_far_m:.0f}-{cross_m:.0f} m "
                        f"from the trace, both sides, +/- {lake_elizabeth_radius_m:.0f} m along strike); against " + ", ".join(alternative_text))
mean_event_lines.append(f"Lake Elizabeth upper quartile (upper quartile of the area minus the far-field median): {lake_up['p75']:+.2f} +/- "
                        f"{lake_up['p75_se']:.2f} mm; the far field's own upper quartile (noise alone) {lake_up['far_p75']:+.2f} mm")
print("\n".join(mean_event_lines))

# %% [markdown]
# ## 13. Supplementary figure: the step maps of every event on both tracks
#
# Three rows: ascending after the burst-seam correction (no uncorrected
# ascending row is drawn); descending before the correction; and descending
# after it. The corrected rows are the data the slip model inverts. Each
# track's shifted seam (section 6) is drawn dashed on its rows. Columns: the
# five events, each titled by its step epoch (`onset_lo` of notebook 04) as a
# date, the mean of the events, and the per-pixel scatter across them,
# computed as in section 8 for every row (asserted equal to the mean event of
# section 8 for the corrected rows). LOS in mm, positive toward the
# satellite, so ascending and descending have opposite colours across the
# fault. Drawn with pygmt, north up in lon/lat, at the 18.3 cm page width, with
# a diverging red-blue scale for the steps and magma for the scatter.

# %%
clim = 4.0                 # step colour range +/- this (mm)
scatter_lim = 2.0          # scatter colour range 0 to this (mm)
map_w_cm = 2.3             # each map's width
gap_x_cm = 0.15            # between maps in a row
gap_y_cm = 0.55            # between rows
row_label_cm = 0.75        # the row label sits this far left of the left column, clear of the lat annotations
n_events_fig = len(step_epochs)
n_col = n_events_fig + 2


def mean_and_scatter(step_stack):
    """Mean and 1-sd scatter of a stack of step maps, as the mean event of section 8.

    Parameters
    ----------
    step_stack : ndarray
        (n_events, n_rows, n_cols) step maps (mm).

    Returns
    -------
    mean, scatter : ndarray
        (n_rows, n_cols) per-pixel mean and sample standard deviation (mm), NaN where any map is missing.
    """
    kept_maps = step_stack
    n_valid = np.isfinite(kept_maps).sum(0)
    mean = np.where(n_valid == len(kept_maps), np.nanmean(kept_maps, 0), np.nan)
    scatter = np.where(n_valid == len(kept_maps), np.nanstd(kept_maps, 0, ddof=1), np.nan)
    return mean, scatter


# the rows: (label, track, step-map stack)
grid_rows = [("Ascending", "ascending", event_maps["ascending"]["ev"]),            # corrected, as the bottom row
             ("Descending", "descending", event_maps["descending"]["ev_raw"]),
             ("Des, Corrected", "descending", event_maps["descending"]["ev"])]
grid_stats = []
for row_label, orbit, step_stack in grid_rows:
    row_mean, row_scatter = mean_and_scatter(step_stack)
    assert len(step_stack) == mean_event[orbit]["n"]
    if step_stack is event_maps[orbit]["ev"]:                   # corrected rows: must equal the mean event of section 8
        assert np.allclose(row_mean, mean_event[orbit]["mean"], rtol=0, atol=1e-5, equal_nan=True), f"{row_label}: grid mean is not the mean event"
        assert np.allclose(row_scatter, mean_event[orbit]["scatter"], rtol=0, atol=1e-5, equal_nan=True), f"{row_label}: grid scatter differs"
    grid_stats.append((row_mean, row_scatter, len(step_stack)))

# each event's title: its step epoch (onset_lo of notebook 04) as a calendar date
onset_dates = []
for i, step_epoch in enumerate(step_epochs):
    for orbit in fits:
        assert abs(fits[orbit]["steps"][i][1] - step_epoch) < 1e-9, f"{orbit} event {i} is not 04's event {i}"
    onset_dates.append(utils.decimal_year_to_day(step_epoch).strftime("%Y-%m-%d"))
print("event titles (onset_lo):", onset_dates)
region_box = [box_lonlat[0], box_lonlat[2], box_lonlat[1], box_lonlat[3]]
x_corner = x0 + c0 * dx                                         # corner of the box's first pixel
y_corner = y0 + r0 * dy
_, map_h_cm = utils.gmt_map_size(region_box, f"M{map_w_cm}c")
proj_box = f"M{map_w_cm}c"
cpt_dir = os.path.join(out_dir, "cpt")
os.makedirs(cpt_dir, exist_ok=True)
cpt_step = utils.write_cpt(os.path.join(cpt_dir, "step_maps.cpt"), "RdBu_r", -clim, clim)
cpt_scatter = utils.write_cpt(os.path.join(cpt_dir, "step_scatter.cpt"), "magma", 0, scatter_lim)   # no blue against green
pygmt.config(FONT="7p,Helvetica", FONT_ANNOT_PRIMARY="6p,Helvetica", FONT_LABEL="6p,Helvetica", MAP_FRAME_TYPE="plain",
             MAP_FRAME_PEN="0.5p", MAP_TICK_LENGTH_PRIMARY="2p", MAP_ANNOT_OFFSET="2p", MAP_LABEL_OFFSET="2p", FORMAT_GEO_MAP="D")
fig = pygmt.Figure()
panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
assert len(grid_rows) * n_col <= len(panel_letters)
burst_lon = np.array([region_box[0], region_box[1]])
panel_number = 0
for row_number, (row_label, orbit, step_stack) in enumerate(grid_rows):
    row_mean, row_scatter, n_row_mean = grid_stats[row_number]
    panels = []                                                 # (map, title, colour table) for each column
    for i in range(len(fits[orbit]["steps"])):
        panel_title = onset_dates[i]
        panels.append((step_stack[i], panel_title, cpt_step))
    panels.append((row_mean, f"Mean of {n_row_mean}", cpt_step))
    panels.append((row_scatter, "Scatter", cpt_scatter))
    if row_number > 0:
        fig.shift_origin(xshift=f"-{(n_col - 1) * (map_w_cm + gap_x_cm)}c", yshift=f"-{map_h_cm + gap_y_cm}c")
    for col_number, (panel_map, panel_title, cpt) in enumerate(panels):
        if col_number > 0:
            fig.shift_origin(xshift=f"{map_w_cm + gap_x_cm}c")
        grid = utils.utm_to_lonlat_grid(panel_map, x_corner, y_corner, dx, dy, epsg, region_box, 0.00015, 0.00012)   # ~13 m, half the 30 m pixel, so no pixel is dropped or doubled
        if col_number == 0:
            sides = "W"
        else:
            sides = "w"
        if row_number == len(grid_rows) - 1:
            sides += "S"
        else:
            sides += "s"
        sides += "nr"
        fig.basemap(region=region_box, projection=proj_box, frame=[sides, "xa0.15f0.05", "ya0.05f0.025"])
        fig.grdimage(grid=grid, cmap=cpt, nan_transparent=True)
        if cpt == cpt_step:
            line_colour = "black"
        else:
            line_colour = "white"                               # white on the dark scatter map
        fig.plot(x=x_fault.ft_lon.values, y=x_fault.ft_lat.values, pen=f"0.5p,{line_colour}")
        for line_slope, line_lat in burst_lines.get(orbit, []):    # the burst seam, thin dashed, on every row of its track
            fig.plot(x=burst_lon, y=line_lat + line_slope * (burst_lon + 122.0), pen=f"0.35p,{line_colour},2p_1.5p")
        fig.basemap(region=region_box, projection=proj_box, frame=[sides, "xa0.15f0.05", "ya0.05f0.025"])
        fig.text(position="TL", text=f"@%Helvetica-Bold%{panel_letters[panel_number]}@%%  {panel_title}", justify="BL",
                 offset="0c/0.12c", font="6.5p,Helvetica", no_clip=True)
        if col_number == 0:                                     # the row label, rotated, left of the lat annotations
            fig.text(position="LM", text=row_label, angle=90, justify="BC", offset=f"-{row_label_cm}c/0c",
                     font="6.5p,Helvetica-Bold", no_clip=True)
        panel_number += 1
# colour bars, 0.22 cm tall, 0.75 cm under the bottom row
fig.shift_origin(xshift=f"-{(n_col - 1) * (map_w_cm + gap_x_cm)}c")
span_step = n_events_fig * (map_w_cm + gap_x_cm) + map_w_cm   # the event + mean columns
w_step = 2 * map_w_cm + gap_x_cm                               # bar two maps wide, centred under those columns
cb_pt = 6.0 / min(1.0, np.sqrt(w_step / 9.0))                  # GMT shrinks a short bar's fonts by ~sqrt(length / 9 cm): this prints ~6p
with pygmt.config(FONT_ANNOT_PRIMARY=f"{cb_pt:.1f}p,Helvetica", FONT_LABEL=f"{cb_pt:.1f}p,Helvetica"):
    fig.colorbar(cmap=cpt_step, position=f"x{0.5 * (span_step - w_step)}c/-0.75c+w{w_step}c/0.22c+h+jTL+e",
                 frame=["xa2f1+lLOS step (mm), positive toward the satellite"])
with pygmt.config(FONT_ANNOT_PRIMARY="12p,Helvetica", FONT_LABEL="12p,Helvetica"):    # GMT halves the fonts of a bar this short: 12p prints as ~6p
    fig.colorbar(cmap=cpt_scatter, position=f"x{(n_col - 1) * (map_w_cm + gap_x_cm)}c/-0.75c+w{map_w_cm}c/0.22c+h+jTL+ef",
                 frame=["xa1f0.5+lScatter, 1 sd (mm)"])
# no figure title; the caption gives the per-pixel steps, box median removed, burst offsets removed in the top
# (ascending) and bottom rows, coherence coh_maps
f_grid = os.path.join(fig_dir, f"figS_event_step_maps_coh{coh_maps}")
fig.savefig(f_grid + ".png", dpi=600, transparent=False, crop="+gwhite", anti_alias=True, show=False)
fig.savefig(f_grid + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
print("Saved figure to:", f_grid + ".png")

# %% [markdown]
# ## 14. Register
#
# The stats file `06_event_field_stats.txt` (every value above, with its
# definition) and the registry rows `06_*` quoted in the paper.

# %%
stats_lines = [f"06 EVENT FIELD GEOMETRY  (coherence {coh_timeseries}, box {box_lonlat}, {nr} x {nc} px)",
               f"local strike {strike:.2f} deg; LOS.strike asc {los_dot_strike_asc:+.3f} desc {los_dot_strike_desc:+.3f}, "
               f"geometric asc/desc ratio {geom_ratio:+.2f}",
               f"per-event near-fault LOS steps (east - west, {xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m, stacked band):"]
for _, step_row in steps_df.iterrows():
    stats_lines.append(f"   {step_row.step_epoch:.2f}: asc {step_row.ascending_step_mm:+.2f} mm, desc {step_row.descending_step_mm:+.2f} mm, "
                       f"ratio {step_row.ratio_asc_desc:+.2f}")
stats_lines.append(f"mean event = {mean_event['ascending']['n']} events")
stats_lines.append("burst seams in the box (CSLC-S1-STATIC bounding polygons, " + ", ".join(os.path.basename(f) for f in burst_static_files) + "):")
for orbit, seams in burst_seam_list.items():
    for seam in seams:
        stats_lines.append(f"   {orbit} {seam['north']} / {seam['south']}: lat {seam['lat_west']:.4f} at {box_lonlat[0]} E, "
                           f"{seam['lat_east']:.4f} at {box_lonlat[2]} E; corrected: {orbit in burst_corrected}")
for _, burst_row in burst_df.iterrows():
    stats_lines.append(f"   burst fix {burst_row.track} data seam {burst_row.line_lat:.4f} N at -122 E ({burst_row.seam_shift_m:+.0f} m along track from the "
                       f"polygon line {burst_row.polygon_line_lat:.4f}; mean-map jump {burst_row.mean_jump_at_seam_mm:+.2f} vs {burst_row.mean_jump_at_polygon_mm:+.2f} mm at the polygon line) "
                       f"event {burst_row.step_epoch:.2f}: {burst_row.offset_mm:+.2f} mm (north - south; {burst_row.n_north} / {burst_row.n_south} band pixels; {burst_row.fixed_side} side held fixed)")
stats_lines.append(f"FP inter-event step (stacked band) {block_step(fp_rate):+.2f} mm/yr, Up {block_step(up_rate):+.2f}; "
                   f"FP mean event {block_step(fp_evt):+.2f} mm, Up {block_step(up_evt):+.2f}")
stats_lines.append(f"profiles, near-fault steps east - west ({xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m), +/-{cross_m / 1e3:g} km, "
                   f"swath +/-{swath_m:.0f} m:")
for name in profile_names:
    fp_profile = profiles[name]
    stats_lines.append(f"   {name}-{name}' ({fp_profile['lonlat'][1]:.4f} N, local strike {fp_profile['strike']:.1f} deg): "
                       f"inter-event {fp_profile['rate_step']:+.2f} mm/yr, event {fp_profile['event_step']:+.2f} mm (n {fp_profile['d'].size})")
stats_lines.append("LOS profiles per track (Fig 2), near-fault steps east - west, LOS positive toward the satellite:")
for orbit in tracks:
    for name in profile_names:
        los_profile = los_profiles[orbit][name]
        west_block = los_profile["blocks"]["W"]
        east_block = los_profile["blocks"]["E"]
        stats_lines.append(f"   {orbit[:3]} {name}-{name}': inter-event {los_profile['rate_step']:+.2f} mm/yr, event {los_profile['event_step']:+.2f} mm; "
                           f"event per block (near {xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m over far {far_m[0]:.0f}-{far_m[1]:.0f} m) "
                           f"W {west_block['amp']:+.2f} +/- {west_block['se']:.2f}, E {east_block['amp']:+.2f} +/- {east_block['se']:.2f} mm")
stats_lines.append(f"X-X' event asc/desc LOS step ratio {xx_ratio:+.2f} (geometric {geom_ratio:+.2f})")
stats_lines += mean_event_lines
print("\n".join(stats_lines))
sidecar = REG.sidecar("06_event_field", stats_lines)

REG.register("06_strike_local_deg", strike, "deg", 2, where="M4", cell="strike", sidecar=sidecar)
REG.register("06_asc_desc_geom_ratio", geom_ratio, "", 2, where="M4", cell="los-steps", sidecar=sidecar)
REG.register("06_n_events_mean", mean_event["ascending"]["n"], "count", 0, where="M4", cell="mean-event", sidecar=sidecar)
for _, step_row in steps_df.iterrows():
    year_tag = f"{int(step_row.step_epoch)}"
    REG.register(f"06_los_step_asc_{year_tag}_mm", step_row.ascending_step_mm, "mm", 2, where="S", cell="los-steps", sidecar=sidecar)
    REG.register(f"06_los_step_desc_{year_tag}_mm", step_row.descending_step_mm, "mm", 2, where="S", cell="los-steps", sidecar=sidecar)
    REG.register(f"06_step_ratio_{year_tag}", step_row.ratio_asc_desc, "", 2, where="S", cell="los-steps", sidecar=sidecar)
for _, burst_row in burst_df.iterrows():
    REG.register(f"06_burst_{burst_row.track}_{int(burst_row.step_epoch)}_mm", burst_row.offset_mm, "mm", 2, where="S", cell="burst-fix", sidecar=sidecar)
for track_name, track_rows in burst_df.groupby("track"):      # the along-track seam shift of each track
    REG.register(f"06_burst_seam_shift_{track_name}_m", float(track_rows.seam_shift_m.iloc[0]), "m", 0, where="S", cell="burst-fix", sidecar=sidecar)
REG.register("06_fp_rate_step_mm_yr", block_step(fp_rate), "mm/yr", 2, where="P6", cell="decompose", sidecar=sidecar)
REG.register("06_up_rate_step_mm_yr", block_step(up_rate), "mm/yr", 2, where="S", cell="decompose", sidecar=sidecar)
REG.register("06_fp_event_step_mm", block_step(fp_evt), "mm", 2, where="P6", cell="decompose", sidecar=sidecar)
REG.register("06_up_event_step_mm", block_step(up_evt), "mm", 2, where="S", cell="decompose", sidecar=sidecar)
for name in profile_names:
    REG.register(f"06_{name}{name}_strike_local_deg", profiles[name]["strike"], "deg", 1, where="Fig2", cell="profiles", sidecar=sidecar)
    REG.register(f"06_{name}{name}_rate_step_mm_yr", profiles[name]["rate_step"], "mm/yr", 2, where="P6", cell="profiles", sidecar=sidecar)
    REG.register(f"06_{name}{name}_event_step_mm", profiles[name]["event_step"], "mm", 2, where="P6", cell="profiles", sidecar=sidecar)
for orbit in tracks:
    for name in profile_names:
        los_profile = los_profiles[orbit][name]
        REG.register(f"06_{name}{name}_rate_step_{orbit[:3]}_mm_yr", los_profile["rate_step"], "mm/yr", 2, where="Fig2", cell="los-profiles", sidecar=sidecar)
        REG.register(f"06_{name}{name}_event_step_{orbit[:3]}_mm", los_profile["event_step"], "mm", 2, where="Fig2", cell="los-profiles", sidecar=sidecar)
        for block_name in ("W", "E"):
            REG.register(f"06_{name}{name}_event_{block_name}_{orbit[:3]}_mm", los_profile["blocks"][block_name]["amp"], "mm", 2,
                         where="Fig2", cell="los-profiles", sidecar=sidecar)
            REG.register(f"06_{name}{name}_event_{block_name}_{orbit[:3]}_se_mm", los_profile["blocks"][block_name]["se"], "mm", 2,
                         where="Fig2", cell="los-profiles", sidecar=sidecar)
REG.register("06_XX_event_ratio_asc_desc", xx_ratio, "", 2, where="S2", cell="los-profiles", sidecar=sidecar)
# section 12: the fault-parallel block deflections, the decay distance at X-X' and the Lake Elizabeth vertical
for name in profile_names:
    for block_name in ("W", "E"):
        REG.register(f"06_{name}{name}_fp_event_{block_name}_mm", profiles[name]["blocks"][block_name]["amp"], "mm", 2,
                     where="P6", cell="mean-event numbers", sidecar=sidecar)
        REG.register(f"06_{name}{name}_fp_event_{block_name}_se_mm", profiles[name]["blocks"][block_name]["se"], "mm", 2,
                     where="P6", cell="mean-event numbers", sidecar=sidecar)
for side_name in ("W", "E"):
    REG.register(f"06_XX_fp_event_decay_{side_name}_m", decay[side_name][0], "m", 0, where="P6", cell="mean-event numbers", sidecar=sidecar)
REG.register("06_lake_elizabeth_up_event_mm", lake_up["value"], "mm", 2, where="P6", cell="mean-event numbers", sidecar=sidecar)
REG.register("06_lake_elizabeth_up_event_se_mm", lake_up["se"], "mm", 2, where="P6", cell="mean-event numbers", sidecar=sidecar)
REG.register("06_lake_elizabeth_n_px", lake_up["n_area"], "count", 0, where="P6", cell="mean-event numbers", sidecar=sidecar)
REG.register("06_lake_elizabeth_up_event_p75_mm", lake_up["p75"], "mm", 2, where="P6", cell="mean-event numbers", sidecar=sidecar)
REG.register("06_lake_elizabeth_up_event_p75_se_mm", lake_up["p75_se"], "mm", 2, where="P6", cell="mean-event numbers", sidecar=sidecar)
REG.register("06_lake_elizabeth_far_p75_mm", lake_up["far_p75"], "mm", 2, where="P6", cell="mean-event numbers", sidecar=sidecar)
REG.register("06_lake_elizabeth_radius_m", lake_elizabeth_radius_m, "m", 0, where="P6", cell="mean-event numbers", sidecar=sidecar)
for orbit in mean_event:
    REG.register(f"06_n_px_csi_{orbit[:3]}", int(np.isfinite(mean_event[orbit]["mean"]).sum()), "count", 0, where="M4", cell="mean-event", sidecar=sidecar)
REG.render()
print("registered; see", common_paths["paper_numbers"])
