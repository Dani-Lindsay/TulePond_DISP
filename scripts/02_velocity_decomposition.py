# %% [markdown]
# # [1] 02 Velocity decomposition: fault-parallel and vertical velocity maps
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13 (MintPy). Run time: seconds.
# ```
#
# **Purpose.** This notebook takes the GNSS-tied ascending and descending
# velocity mosaics (`mosaic_velfield/<orbit>_velocity_NA.h5`, written by
# `data_prep/run04` and `run05` and fingerprinted in notebook 01) and solves,
# pixel by pixel, for the fault-parallel (FP) and vertical (Up) velocity. The
# solve uses `asc_desc2horz_vert` from MintPy, with the mean strike of the
# mapped Hayward trace as the horizontal direction. All horizontal motion is
# assumed to be fault-parallel (the standard two-look assumption; Methods M2).
#
# **Geometry** (definitions from MintPy `utils0.enu2los`):
#
# ```
# inc_angle  - incidence angle of the LOS vector, measured from vertical
# az_angle   - azimuth of the LOS vector from the ground to the satellite, measured
#              from north, anticlockwise positive.  head_angle = 90 - az_angle.
#              Ascending  ~100 (heading N10W, looking east, satellite west of the pixel)
#              Descending ~260 (heading S10E, looking west, satellite east of the pixel)
# strike     - compass bearing clockwise from north = profile_az + 90 (SE).
#              horz_az_angle in MintPy is anticlockwise, so the strike enters as -strike
#              (tested in tests/test_insar_utils.py).
# ```
#
# **Outputs** (`outputs/02_velocity_decomposition/`):
# - `fp_velocity_map.h5` with datasets `fp_velocity` and `up_velocity` (mm/yr)
# - `fp_velocity_xfault_swath200m_dist1000m.csv`: the cross-fault transects at
#   the five creepmeters
# - `02_velocity_stats.txt` and the registry rows `02_*`
#
# The map product `fp_velocity_map.h5` is read by 04 (the Fig. 1 map), 05 and
# 09.
#
# **Checks.** Check 1 re-derives the decomposition at random pixels with an
# independent design matrix and compares it with the MintPy result (assert:
# maximum difference < 1e-3 mm/yr). Check 2 asserts a positive (right-lateral)
# fault-parallel step across the trace at all five creepmeters; the step at
# CFW1 is compared with the creepmeter in notebook 05.

# %%
# [2]
import os

import insar_utils as utils
from TulePond_filepaths import (common_paths, paths_mosaic, creepmeters,
                                unit, az_smooth_n, paths_out, repo_dir)

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
import h5py                                      # MintPy HDF5 products
import pyproj                                    # lon/lat -> UTM
import mintpy
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert   # the decomposition (MintPy)

# %% [markdown]
# ## [3] Paths and parameters

# %%
# [4]
notebook = "02_velocity_decomposition"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)


geom_step = 20        # window (pixels) over which MintPy takes the median geometry (MintPy default)
swath_m = 200.0       # transects: along-fault half-width of the swath (m)
max_dist_m = 1000.0   # transects: cross-fault half-extent (m)
n_check = 200         # pixels re-derived independently in check 1
rng = np.random.default_rng(0)

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(paths_mosaic["ascending"]["vel_NA_file"], paths_mosaic["ascending"]["geo_file"],
           paths_mosaic["descending"]["vel_NA_file"], paths_mosaic["descending"]["geo_file"],
           common_paths["fault_profiles"])

# %% [markdown]
# ## [5] Load the two velocity mosaics and their geometry
#
# Both orbits are on the same 30 m UTM grid (EPSG 32610), so no resampling is
# needed and a pixel in the ascending map is the same ground pixel in the
# descending map. Velocities are stored in m/yr and converted to mm/yr on
# loading (`unit = 1000`).

# %%
# [6]
insar = {}
for orbit in ("ascending", "descending"):
    print(f"Loading {orbit} mosaic")
    with h5py.File(paths_mosaic[orbit]["vel_NA_file"], "r") as hf:
        vel = hf["velocity"][:] * unit                                  # m/yr -> mm/yr
        attrs = {}
        for key, value in hf.attrs.items():                               # MintPy attributes, as text
            attrs[key] = value.decode() if isinstance(value, bytes) else str(value)
    with h5py.File(paths_mosaic[orbit]["geo_file"], "r") as hf:
        inc = hf["incidenceAngle"][:].astype(np.float32)               # as stored; MintPy handles the no-data
        azi = hf["azimuthAngle"][:].astype(np.float32)
    insar[orbit] = {"vel": vel, "inc": inc, "azi": azi, "attrs": attrs}
    print(f"   grid {attrs['LENGTH']} x {attrs['WIDTH']} px, {attrs['X_STEP']} m, EPSG {attrs['EPSG']}, "
          f"ref pixel ({attrs['REF_LON']}, {attrs['REF_LAT']}) UTM")
    print(f"   velocity valid {100 * np.isfinite(vel).mean():.1f}%, median {np.nanmedian(vel):+.2f} mm/yr, "
          f"incidence {np.nanmin(inc[inc > 0]):.1f} to {np.nanmax(inc):.1f} deg, azimuth median {np.nanmedian(azi[inc > 0]):.1f} deg")

# Check: the two grids must be identical
attrs = insar["ascending"]["attrs"]
for key in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP", "WIDTH", "LENGTH", "EPSG", "REF_LAT", "REF_LON"):
    assert attrs[key] == insar["descending"]["attrs"][key], f"asc/desc grids differ in {key}"
assert attrs["EPSG"] == "32610" and float(attrs["X_STEP"]) == 30.0, "expected a 30 m UTM zone 10 grid"
assert insar["ascending"]["vel"].shape == insar["ascending"]["inc"].shape, "geometry grid differs from velocity grid"

# Check: the ground -> satellite vector points west for ascending and east for descending
inc_asc, azi_asc = insar["ascending"]["inc"], insar["ascending"]["azi"]
inc_desc, azi_desc = insar["descending"]["inc"], insar["descending"]["azi"]
los_vector_asc = utils.los_unit_vector(np.nanmedian(inc_asc[inc_asc > 0]), np.nanmedian(azi_asc[inc_asc > 0]))
los_vector_desc = utils.los_unit_vector(np.nanmedian(inc_desc[inc_desc > 0]), np.nanmedian(azi_desc[inc_desc > 0]))
assert los_vector_asc[0] < 0 < los_vector_desc[0], "look directions are not ascending-west / descending-east: check the azimuth convention"
print("median LOS unit vectors (E, N, U): asc", los_vector_asc.round(3), " desc", los_vector_desc.round(3))

# %% [markdown]
# ## [7] Fault trace: the single strike used in the decomposition
#
# `hayward_fault_profiles.csv` is the mapped trace sampled every 100 m (built
# in notebook 01): `ft_lon, ft_lat, profile_az`, where `profile_az` is the
# cross-fault azimuth pointing to the east side. A 21-crossing (2.1 km) rolling
# median removes the digitising kinks and retains the bends of the trace (the
# same smoothing as in notebook 03). One strike, the mean of the smoothed local
# strike, is used for the whole map; notebook 03 uses the local strike at each
# crossing for the creep time series.

# %%
# [8]
x_fault = pd.read_csv(common_paths["fault_profiles"])
x_fault["az_smooth"] = x_fault["profile_az"].rolling(az_smooth_n, center=True, min_periods=3).median()
x_fault["strike"] = x_fault["az_smooth"] + 90.0                 # compass bearing, pointing SE along the fault
strike = float(np.nanmean(x_fault["strike"]))

print(f"trace: {len(x_fault)} crossings, latitude {x_fault.ft_lat.min():.3f} to {x_fault.ft_lat.max():.3f}")
print(f"local strike ranges {np.nanmin(x_fault.strike):.1f} to {np.nanmax(x_fault.strike):.1f} deg")
print(f"mean strike used for the map: {strike:.2f} deg (compass bearing, clockwise from north)")
assert 130 < strike < 160, "the Hayward Fault strikes ~N36W; a mean strike outside 130-160 deg indicates an error in the trace file"

# %% [markdown]
# ## [9] Two-look decomposition (MintPy)
#
# `asc_desc2horz_vert` solves, in windows of `geom_step` x `geom_step` pixels
# with the window-median geometry,
#
# $$ \begin{bmatrix} d_{asc} \\ d_{desc} \end{bmatrix} =
#    \begin{bmatrix} \sin\theta_a \cos(\alpha_a - \alpha_h) & \cos\theta_a \\
#                    \sin\theta_d \cos(\alpha_d - \alpha_h) & \cos\theta_d \end{bmatrix}
#    \begin{bmatrix} v_{FP} \\ v_{U} \end{bmatrix} $$
#
# with $\alpha_h$ = `-strike`. Pixels missing on either track are set to NaN on
# both, so that a valid value is never combined with a zero.

# %%
# [10]
vel_asc, vel_desc = insar["ascending"]["vel"], insar["descending"]["vel"]
both = np.isfinite(vel_asc) & np.isfinite(vel_desc)
print(f"pixels valid on both tracks: {both.sum()} ({100 * both.mean():.1f}% of the grid)")

los_vel = np.stack([vel_asc, vel_desc]).astype(np.float32)
los_vel[:, ~both] = np.nan
los_inc = np.stack([inc_asc, inc_desc]).astype(np.float32)
los_azi = np.stack([azi_asc, azi_desc]).astype(np.float32)

fp_vel, up_vel = asc_desc2horz_vert(los_vel, los_inc, los_azi, horz_az_angle=-strike, step=geom_step)
fp_vel, up_vel = fp_vel.astype(np.float64), up_vel.astype(np.float64)

print("")
print(f"FP velocity: {100 * np.isfinite(fp_vel).mean():.1f}% valid, median {np.nanmedian(fp_vel):+.2f} mm/yr, "
      f"5-95% {np.nanpercentile(fp_vel, 5):+.2f} to {np.nanpercentile(fp_vel, 95):+.2f}")
print(f"Up velocity: median {np.nanmedian(up_vel):+.2f} mm/yr, 5-95% {np.nanpercentile(up_vel, 5):+.2f} to {np.nanpercentile(up_vel, 95):+.2f}")

# %% [markdown]
# ## [11] Check 1: independent re-derivation of the decomposition at random pixels
#
# The solve is repeated at `n_check` random pixels with
# `utils.design_matrix_up_fp` (the design matrix of this repository, covered by
# the tests) and `numpy.linalg.solve`, using the same window-median geometry as
# MintPy. Check: the two agree to 1e-3 mm/yr (float32 storage), which verifies
# that the sign and azimuth conventions are the same.

# %%
# [12]
# MintPy returns NaN where a window has no valid geometry; report the count, then compare only where it returned a value
lost = both & ~np.isfinite(fp_vel)
print(f"pixels valid on both tracks but NaN after the solve: {lost.sum()} ({100 * lost.sum() / both.sum():.2f}%)")

rows, cols = np.where(both & np.isfinite(fp_vel))
pick = rng.choice(len(rows), size=n_check, replace=False)
diff_fp, diff_up = [], []
for row, col in zip(rows[pick], cols[pick]):
    # MintPy's geometry window for this pixel: the geom_step x geom_step block it falls in
    row_0, col_0 = (row // geom_step) * geom_step, (col // geom_step) * geom_step
    inc_window = np.nanmedian(los_inc[:, row_0:row_0 + geom_step, col_0:col_0 + geom_step], axis=(1, 2))
    azi_window = np.nanmedian(los_azi[:, row_0:row_0 + geom_step, col_0:col_0 + geom_step], axis=(1, 2))
    G = utils.design_matrix_up_fp(inc_window, azi_window, strike)               # columns [U, FP]
    up_hand, fp_hand = np.linalg.solve(G, [vel_asc[row, col], vel_desc[row, col]])
    diff_fp.append(fp_hand - fp_vel[row, col])
    diff_up.append(up_hand - up_vel[row, col])
diff_fp, diff_up = np.array(diff_fp), np.array(diff_up)

print(f"independent solve minus MintPy at {n_check} pixels: FP max |diff| {np.abs(diff_fp).max():.2e} mm/yr, "
      f"Up max |diff| {np.abs(diff_up).max():.2e} mm/yr")
assert np.abs(diff_fp).max() < 1e-3 and np.abs(diff_up).max() < 1e-3, "the independent design matrix and MintPy disagree"
print("Check passed: the MintPy decomposition equals the independent solve")

# %% [markdown]
# ## [13] Check 2: sign and size of the fault-parallel step at the creepmeters
#
# Right-lateral creep moves the east block south-east relative to the west
# block. FP is positive along the strike bearing (south-east), so the east side
# of the trace has the more positive FP velocity. At each creepmeter a
# cross-fault transect is sampled from the FP map: points every 30 m within
# +/- `swath_m` along the fault and +/- `max_dist_m` across it, centred on the
# trace point at the latitude of the creepmeter, each point taking the value of
# the pixel beneath it (so a pixel can be sampled twice). The transect is
# fitted with `v = a + b d + step H(d > 0)`; the step is the fault-crossing
# creep rate in the velocity field, compared with the creepmeter rate in
# notebook 05. Check: the step is positive at every creepmeter.

# %%
# [14]
def step_fit(dist, vel):
    """v = a + b*d + step*[d > 0]; the step is the fault-crossing creep rate.
    Solved by least squares for the three unknowns."""
    G = np.column_stack([np.ones_like(dist), dist, (dist > 0).astype(float)])
    a, b, step = np.linalg.lstsq(G, vel, rcond=None)[0]
    return step, a + b * dist + step * (dist > 0)


# UTM position of every crossing, and the pixel grid
utm = pyproj.Transformer.from_crs(4326, int(attrs["EPSG"]), always_xy=True)
x_fault["easting"], x_fault["northing"] = utm.transform(x_fault["ft_lon"].values, x_fault["ft_lat"].values)
x_first, x_step = float(attrs["X_FIRST"]), float(attrs["X_STEP"])
y_first, y_step = float(attrs["Y_FIRST"]), float(attrs["Y_STEP"])
n_rows, n_cols = fp_vel.shape
pixel_m = abs(x_step)

# the transects: walk a grid of points across (dist) and along (offset) the fault at each creepmeter
# and read the FP pixel under each point
transect_rows = []
for code, cm_lat, cm_lon, name in creepmeters:
    trace_point = x_fault.loc[(x_fault.ft_lat - cm_lat).abs().idxmin()]  # the trace point at the creepmeter's latitude
    cross_az = np.deg2rad(trace_point["az_smooth"])                      # cross-fault direction, toward the east side
    along_az = np.deg2rad(trace_point["strike"])                         # along the fault
    n_pixels = 0
    for dist in np.arange(-max_dist_m, max_dist_m + pixel_m, pixel_m):          # across the fault, + = east
        for offset in np.arange(-swath_m, swath_m + pixel_m, pixel_m):          # along the fault
            easting = trace_point["easting"] + dist * np.sin(cross_az) + offset * np.sin(along_az)
            northing = trace_point["northing"] + dist * np.cos(cross_az) + offset * np.cos(along_az)
            col = int(round((easting - x_first) / x_step))
            pixel_row = int(round((northing - y_first) / y_step))
            if 0 <= pixel_row < n_rows and 0 <= col < n_cols and np.isfinite(fp_vel[pixel_row, col]):
                transect_rows.append((code, float(dist), float(fp_vel[pixel_row, col])))
                n_pixels += 1
    print(f"  transect {code} ({name}, lat {cm_lat}): {n_pixels} finite samples")
x_fault_transects = pd.DataFrame(transect_rows, columns=["code", "dist_m", "fp_vel"])
transects_csv = os.path.join(out_dir, f"fp_velocity_xfault_swath{swath_m:.0f}m_dist{max_dist_m:.0f}m.csv")
x_fault_transects.to_csv(transects_csv, index=False)
print(f"Saved transects to: {transects_csv}")

steps = {}
print("")
print("fault-parallel velocity step across the trace (east minus west, mm/yr):")
for code, cm_lat, cm_lon, name in creepmeters:
    transect = x_fault_transects[x_fault_transects.code == code]
    step, _ = step_fit(transect.dist_m.values, transect.fp_vel.values)
    steps[code] = step
    print(f"  {code}: step fit {step:5.2f}")
    assert step > 0, f"{code}: a negative step means the east block moves NW: check the sign convention"
print("Check passed: the step is positive (right-lateral) at every creepmeter")

# %% [markdown]
# ## [15] Save the map product
#
# `fp_velocity_map.h5` carries the FP and Up datasets, the grid attributes of
# the mosaic and the strike used.

# %%
# [16]
provenance = {"PROVENANCE": utils.provenance_stamp(notebook, repo_dir), "FAULT_AZIMUTH_DEG": strike, "UNIT": "mm/year"}
map_h5 = os.path.join(out_dir, "fp_velocity_map.h5")
with h5py.File(map_h5, "w") as hf:
    hf.create_dataset("fp_velocity", data=fp_vel.astype(np.float32), compression="gzip")
    hf.create_dataset("up_velocity", data=up_vel.astype(np.float32), compression="gzip")
    for key in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP", "WIDTH", "LENGTH", "EPSG", "REF_LAT", "REF_LON"):
        hf.attrs[key] = attrs[key]
    for key, value in provenance.items():
        hf.attrs[key] = str(value)
print(f"Saved FP + Up maps to: {map_h5}")

# %% [markdown]
# ## [17] Register
#
# The sidecar `02_velocity_stats.txt` and the registry rows `02_*`.

# %%
# [18]
lines = [
    "02 VELOCITY DECOMPOSITION",
    f"inputs: GNSS-tied velocity mosaics, {attrs['LENGTH']} x {attrs['WIDTH']} px at {attrs['X_STEP']} m, EPSG {attrs['EPSG']}",
    f"mean fault strike used for the map: {strike:.2f} deg (clockwise from north)",
    f"pixels valid on both tracks: {both.sum()} ({100 * both.mean():.1f}%)",
    f"FP velocity median {np.nanmedian(fp_vel):+.2f} mm/yr, 5-95% {np.nanpercentile(fp_vel, 5):+.2f} to {np.nanpercentile(fp_vel, 95):+.2f}",
    f"Up velocity median {np.nanmedian(up_vel):+.2f} mm/yr, 5-95% {np.nanpercentile(up_vel, 5):+.2f} to {np.nanpercentile(up_vel, 95):+.2f}",
    f"independent re-derivation at {n_check} px: max |diff| FP {np.abs(diff_fp).max():.1e}, Up {np.abs(diff_up).max():.1e} mm/yr",
    "",
    "cross-fault FP velocity step at the creepmeters (east - west, mm/yr):",
] + [f"  {code}: {steps[code]:.2f}" for code in steps] + [
    "",
    f"MintPy {mintpy.__version__} (site-packages install in opera_disp_feb13)",
]
sidecar = REG.sidecar("02_velocity", lines)

REG.register("02_strike_mean_deg", strike, "deg", 2, where="M2", cell="strike", sidecar=sidecar)
REG.register("02_fp_valid_pct", 100 * np.isfinite(fp_vel).mean(), "pct", 1, where="M2", cell="decompose", sidecar=sidecar)
REG.register("02_fp_median_mm_yr", np.nanmedian(fp_vel), "mm/yr", 2, where="M2", cell="decompose", sidecar=sidecar)
for code in steps:
    REG.register(f"02_xfault_step_{code}_mm_yr", steps[code], "mm/yr", 2, where="M4", cell="transects", sidecar=sidecar)
REG.register("02_mintpy_version", mintpy.__version__, "1", 0, where="M1", cell="register", sidecar=sidecar)
REG.render()
