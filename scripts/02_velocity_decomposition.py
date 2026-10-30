# %% [markdown]
# # 02 Velocity decomposition: fault-parallel and vertical velocity maps
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13 (MintPy). Run time: seconds.
# ```
#
# **What this notebook does.**
#
# 1. Loads the GNSS-tied ascending and descending velocity mosaics and their
#    incidence and azimuth angles, and checks that both are on one grid and that
#    the look directions are as expected (section 3).
# 2. Computes one mean strike of the mapped Hayward Fault trace (section 4).
# 3. Decomposes the two line-of-sight (LOS) velocities, pixel by pixel, into
#    the fault-parallel (FP) and vertical (Up) velocity with MintPy, assuming
#    that all horizontal motion is parallel to the mean strike (section 5;
#    Methods M2).
# 4. Check 1 repeats the decomposition at random pixels with an independent
#    design matrix (section 6).
# 5. Check 2 samples cross-fault transects of the FP map at the five
#    creepmeters and fits the velocity step across the trace (section 7).
# 6. Saves the FP and Up maps and registers the numbers (sections 8 and 9).
#
# **Use of MintPy** (Yunjun et al. 2019). The input time series and velocities
# were made with MintPy in `data_prep/run04`. There the OPERA DISP-S1
# displacement stacks are converted to MintPy time series (`disp_nc_to_mintpy`
# from `opera-utils`), referenced at the pixel of GNSS station ZOA1
# (`reference_point.py`) and corrected with the OPERA TROPO layer (`diff.py`),
# and the velocity is fitted with `timeseries2velocity.py`. `data_prep/run05`
# then applies the GNSS tie and mosaics the frames of each orbit. In this
# notebook MintPy is used only through the function `asc_desc2horz_vert` (module
# `mintpy.asc_desc2horz_vert`), called on arrays in memory for the decomposition
# of section 5. The files are read with h5py, not with the MintPy readers. The
# MintPy version is registered (`02_mintpy_version`).
#
# **Inputs**: `mosaic_velfield/<orbit>_velocity_NA.h5` and
# `mosaic_velfield/<orbit>_geometry.h5` for both orbits (written by
# `data_prep/run04` and `run05`, fingerprinted in notebook 01), the fault
# crossings `hayward_fault_profiles.csv` (notebook 01) and the creepmeter list
# in `TulePond_filepaths.py`.
#
# **Outputs** (`outputs/02_velocity_decomposition/`):
# - `fp_velocity_map.h5` with datasets `fp_velocity` and `up_velocity` (mm/yr),
#   read by notebooks 04 (the Fig. 1 map) and 05
# - `fp_velocity_xfault_swath200m_dist1000m.csv`: the cross-fault transects at
#   the five creepmeters, read by notebook 05
# - `02_velocity_stats.txt` and the registry rows `02_*`
#
# **Figures**: none.
#
# **What to check.** The printed output of section 3 (grid, valid fraction and
# median LOS unit vectors of each track) and of section 4 (the mean strike).
# Section 6 asserts that the independent solve and MintPy agree to 1e-3 mm/yr.
# Section 7 prints the FP step at each creepmeter and asserts that every step is
# positive (right-lateral). The step at CFW1 is compared with the creepmeter in
# notebook 05.

# %% [markdown]
# ## 1. Imports and settings

# %%
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
# ## 2. Paths and parameters

# %%
notebook = "02_velocity_decomposition"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)


geom_step = 20        # block size (pixels) over which MintPy takes the median geometry (MintPy default)
swath_m = 200.0       # transects: along-fault half-width of the swath (m)
max_dist_m = 1000.0   # transects: cross-fault half-extent (m)
n_check = 200         # pixels solved independently in check 1 (section 6)
rng = np.random.default_rng(0)

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(paths_mosaic["ascending"]["vel_NA_file"], paths_mosaic["ascending"]["geo_file"],
           paths_mosaic["descending"]["vel_NA_file"], paths_mosaic["descending"]["geo_file"],
           common_paths["fault_profiles"])

# %% [markdown]
# ## 3. Load the two velocity mosaics and their geometry
#
# Both orbits are on the same 30 m UTM grid (EPSG 32610), so no resampling is
# needed and a pixel in the ascending map is the same ground pixel in the
# descending map (asserted). Velocities are stored in m/yr and converted to
# mm/yr on loading (`unit = 1000`).
#
# The geometry follows the MintPy convention (`utils0.enu2los`): the incidence
# angle $\theta$ is measured from the vertical, and the azimuth $\alpha$ is the
# azimuth of the LOS vector from the ground to the satellite, measured from
# north, anticlockwise positive. For Sentinel-1, $\alpha \approx 100^\circ$ on
# the ascending track (heading N10W, looking east, satellite west of the pixel)
# and $\alpha \approx 260^\circ$ on the descending track (heading S10E, looking
# west, satellite east of the pixel). The unit vector from the ground to the
# satellite, in east, north and up components, is
#
# $$ \hat{\mathbf{l}} = \left(-\sin\theta \sin\alpha,\ \sin\theta \cos\alpha,\ \cos\theta\right) $$
#
# (`utils.los_unit_vector`). Check: its east component, from the median
# geometry of each track, is negative for the ascending track and positive for
# the descending track.

# %%
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
# ## 4. Fault trace: the mean strike used in the decomposition
#
# `hayward_fault_profiles.csv` is the hand-mapped trace resampled every 100 m
# (notebook 01): `ft_lon, ft_lat, profile_az`, where `profile_az` is the
# cross-fault azimuth $\phi_k$ of crossing $k$ (compass bearing, toward the east
# side). A centred rolling median over 21 crossings (2.1 km; `az_smooth_n`,
# at least 3 values) removes the digitising kinks and keeps the bends of the
# trace, as in notebook 03. The local strike and the mean strike are
#
# $$ \Psi_k = \operatorname{median}\left(\phi_{k-10}, \dots, \phi_{k+10}\right) + 90^\circ, \qquad \Psi = \frac{1}{N}\sum_{k=1}^{N} \Psi_k, $$
#
# where $\Psi_k$ is a compass bearing (clockwise from north, pointing south-east
# along the fault) and $N$ the number of crossings. One strike, $\Psi$, is used
# for the whole map; notebook 03 uses the local strike $\Psi_k$ at each crossing
# for the creep time series. Check: $130^\circ < \Psi < 160^\circ$.

# %%
x_fault = pd.read_csv(common_paths["fault_profiles"])
x_fault["az_smooth"] = x_fault["profile_az"].rolling(az_smooth_n, center=True, min_periods=3).median()
x_fault["strike"] = x_fault["az_smooth"] + 90.0                 # compass bearing, pointing SE along the fault
strike = float(np.nanmean(x_fault["strike"]))

print(f"trace: {len(x_fault)} crossings, latitude {x_fault.ft_lat.min():.3f} to {x_fault.ft_lat.max():.3f}")
print(f"local strike ranges {np.nanmin(x_fault.strike):.1f} to {np.nanmax(x_fault.strike):.1f} deg")
print(f"mean strike used for the map: {strike:.2f} deg (compass bearing, clockwise from north)")
assert 130 < strike < 160, "the Hayward Fault strikes ~N36W; a mean strike outside 130-160 deg indicates an error in the trace file"

# %% [markdown]
# ## 5. Two-component decomposition with MintPy
#
# The LOS velocity, positive toward the satellite, is the projection of the
# ground velocity onto $\hat{\mathbf{l}}$. If all horizontal motion is
# parallel to the strike, the ground velocity is
# $v_{FP}\,\hat{\mathbf{s}} + v_U\,\hat{\mathbf{z}}$, with
# $\hat{\mathbf{s}} = (\sin\Psi, \cos\Psi, 0)$ and $\hat{\mathbf{z}} = (0, 0, 1)$,
# and
#
# $$ v_{LOS} = v_{FP} \sin\theta \cos(\alpha + \Psi) + v_U \cos\theta . $$
#
# The two tracks give, at each pixel,
#
# $$ \begin{bmatrix} v_{asc} \\ v_{desc} \end{bmatrix} =
#    \begin{bmatrix} \sin\theta_a \cos(\alpha_a + \Psi) & \cos\theta_a \\
#                    \sin\theta_d \cos(\alpha_d + \Psi) & \cos\theta_d \end{bmatrix}
#    \begin{bmatrix} v_{FP} \\ v_{U} \end{bmatrix}, $$
#
# where $v_{asc}$ and $v_{desc}$ are the ascending and descending LOS velocities
# (mm/yr), $\theta_a, \alpha_a$ and $\theta_d, \alpha_d$ the incidence and
# azimuth of each track (deg), $\Psi$ the mean strike of section 4, $v_{FP}$ the
# fault-parallel velocity (mm/yr, positive toward $\Psi$, south-east) and $v_U$
# the vertical velocity (mm/yr, positive up). The fault-normal horizontal
# component is assumed to be zero.
#
# `asc_desc2horz_vert` writes the horizontal term as
# $\sin\theta\cos(\alpha - \alpha_h)$, where $\alpha_h$ is the azimuth of the
# horizontal direction measured anticlockwise from north, so the strike enters
# as $\alpha_h = -\Psi$ (`horz_az_angle=-strike`; tested in
# `tests/test_insar_utils.py`). MintPy splits the grid into blocks of
# `geom_step` x `geom_step` pixels (20 x 20, 600 m), replaces $\theta$ and
# $\alpha$ by their medians over each block (NaN ignored), and solves
# $\mathbf{m} = G^{+}\mathbf{d}$ with the pseudo-inverse $G^{+}$, which for this
# 2 x 2 system is the inverse. Pixels missing on either track are set to NaN on
# both tracks before the solve.

# %%
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
# ## 6. Check 1: independent solve of the decomposition at random pixels
#
# The solve is repeated at `n_check` random pixels with
# `utils.design_matrix_up_fp`, the design matrix of this repository (covered by
# the tests), whose rows are
# $[\hat{\mathbf{l}}\cdot\hat{\mathbf{z}},\ \hat{\mathbf{l}}\cdot\hat{\mathbf{s}}]$
# for each track (columns U and FP), and `numpy.linalg.solve`, using the same
# block-median geometry as MintPy. This verifies that the sign and azimuth
# conventions agree. The number of pixels valid on both tracks for which MintPy
# returns NaN (blocks without valid geometry) is printed. Check: the two solves
# agree to 1e-3 mm/yr (float32 storage).

# %%
# MintPy returns NaN where a window has no valid geometry; report the count, then compare only where it returned a value
lost = both & ~np.isfinite(fp_vel)
print(f"pixels valid on both tracks but NaN after the solve: {lost.sum()} ({100 * lost.sum() / both.sum():.2f}%)")

rows, cols = np.where(both & np.isfinite(fp_vel))
pick = rng.choice(len(rows), size=n_check, replace=False)
diff_fp, diff_up = [], []
for row, col in zip(rows[pick], cols[pick]):
    # MintPy's geometry block for this pixel: the geom_step x geom_step block it falls in
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
# ## 7. Check 2: sign and size of the fault-parallel step at the creepmeters
#
# Right-lateral creep moves the east block south-east relative to the west
# block. FP is positive along the strike bearing (south-east), so the east side
# of the trace has the more positive FP velocity.
#
# At each creepmeter a cross-fault transect is sampled from the FP map, centred
# on the trace point nearest to the latitude of the creepmeter. The sample
# points are
#
# $$ \mathbf{x} = \mathbf{x}_0 + d\,\hat{\mathbf{n}} + o\,\hat{\mathbf{s}}_k , $$
#
# where $\mathbf{x}_0$ is the UTM position of the trace point, $\hat{\mathbf{n}}$
# the horizontal unit vector along the smoothed cross-fault azimuth (toward the
# east side), $\hat{\mathbf{s}}_k$ the unit vector along the local strike $\Psi_k$,
# $d$ the distance across the fault (m, positive to the east, from
# $-$`max_dist_m`) and $o$ the offset along the fault (m, from $-$`swath_m`),
# both in steps of one pixel (30 m). Each point takes the value of the pixel
# beneath it, so a pixel can be sampled twice. The transect is fitted by least
# squares with
#
# $$ v(d) = a + b\,d + \Delta v\,H(d), \qquad H(d) = \begin{cases} 1 & d > 0 \\ 0 & d \le 0 \end{cases} $$
#
# where $v$ is the FP velocity (mm/yr), $a$ an offset (mm/yr), $b$ a linear
# gradient across the fault (mm/yr per m) and $\Delta v$ the step across the
# trace (mm/yr), the fault-crossing creep rate in the velocity field. It is
# compared with the creepmeter rate in notebook 05. Check: $\Delta v > 0$ at
# every creepmeter.

# %%
def step_fit(dist, vel):
    """Fit an offset, a linear gradient and a step at d = 0 to a cross-fault transect.

    The model is v = a + b * d + step * H(d), with H(d) = 1 for d > 0 and 0
    otherwise, solved by least squares for a, b and step.

    Parameters
    ----------
    dist : numpy.ndarray
        Distance across the fault (m, positive to the east).
    vel : numpy.ndarray
        Fault-parallel velocity at each distance (mm/yr).

    Returns
    -------
    step : float
        Step across the trace (mm/yr), the fault-crossing creep rate.
    model : numpy.ndarray
        Fitted velocity at each distance (mm/yr).
    """
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
# ## 8. Save the map product
#
# `fp_velocity_map.h5` carries the FP and Up datasets, the grid attributes of
# the mosaic and the strike used.

# %%
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
# ## 9. Register
#
# The sidecar `02_velocity_stats.txt` and the registry rows `02_*`.

# %%
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
