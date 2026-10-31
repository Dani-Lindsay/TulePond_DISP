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
# # 01 Prep figures: the velocity chain (Fig. S1) and the GNSS tie (Fig. S2)
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13. Run time: about 5 minutes (the troposphere column reads each TROPO.h5 once).
# ```
#
# **What this notebook does.** It draws the two supplementary figures that
# document the prepared velocity fields, from the products of `data_prep/run04`
# and `run05` (acquisitions to 27 March 2026), and registers the numbers quoted
# in their captions and in Methods M1. No product is modified.
#
# 1. **Acquisitions**: the number of dates and the first and last date of each
#    frame (section 3).
# 2. **Fig. S1 layers per frame**: average coherence, data density, the LOS
#    velocity before the GNSS tie, the change in velocity due to the troposphere
#    correction, the GNSS ramp, and the final velocity (section 4). The
#    troposphere column is computed as minus the velocity of `TROPO.h5`, which
#    is valid because the run04 velocity is a straight-line fit and therefore
#    linear in the data; section 5 checks this.
# 3. **Mosaics**: the frames of each orbit are mosaicked, the two correction
#    layers are referenced at GNSS station ZOA1, and the colour limits and the
#    share of clipped pixels are computed (sections 6 and 7). Fig. S1 is drawn
#    in section 8.
# 4. **GNSS tie**: the misfit between GNSS and InSAR LOS velocities before and
#    after the ramp, per orbit, from the per-station table of run05 (section 9).
# 5. **Frame overlap**: the difference between the two ascending frames where
#    they overlap, before the tie (section 10). Fig. S2 is drawn in section 11.
#
# The time series and velocities were made with MintPy (Yunjun et al. 2019) in
# `data_prep/run04`; the GNSS tie was applied in `data_prep/run05`.
#
# **Inputs**: for each frame (09156, 09157, 11115), in `<frame>/mintpy/`:
# `avgSpatialCoh.h5`, `timeseries_density.h5`, `timeseries_tropo.h5`,
# `velocity_masked.h5`, `velocity_NA.h5`, `TROPO.h5`; the water mask of the
# rebased stack of the larger ascending frame (`<frame>_stack.nc`);
# `GPS/09157_enu_NA.csv` (the reference station position); the per-station tie
# table `GPS/gps_insar_validation.csv` from run05; and the grid functions of
# `data_prep/run04_velocity.py`.
#
# **Outputs**: `outputs/01_prep_figures/01_prep_figures_stats.txt` (layer
# ranges, clipping, GNSS tie, frame overlap) and the colour tables in
# `outputs/01_prep_figures/cpt/`. Registry rows `01_n_acquisitions_<frame>`,
# `01_first_date_<frame>`, `01_last_date_<frame>`, `01_tropo_rms_*`,
# `01_gnss_*` and `01_overlap_*`.
#
# **Figures** (Science style: Helvetica, bold capital panel letters, 7 pt text,
# 18.3 cm wide, colour-blind-safe colour maps):
# `Figures_<date>/figS_velocity_chain_coh<c>_vel<v>_ramp<r>.png/.pdf`
# (supplement) and `figS_gnss_tie_coh<c>_nasc<n>_ndesc<n>.png/.pdf` (supplement).
#
# **What to check.** Section 5 asserts that the straight-line refit matches the
# run04 velocity on every frame. The sidecar and the printed output of sections
# 7, 9 and 10 list the percentage of pixels clipped by the fixed colour limits,
# the GNSS RMSE before and after the ramp for each orbit, and the median and RMS
# of the 09156 minus 09157 overlap difference.

# %% [markdown]
# ## 1. Imports and settings

# %%
import glob
import os
import sys

import insar_utils as utils
from TulePond_filepaths import common_paths, coh_velocity, frames, paths_gps, repo_dir

import numpy as np                              # Matrix calculations
import pandas as pd                             # Tables
import h5py                                     # MintPy HDF5 products
import xarray as xr                             # rebased stack (water mask)
import pyproj                                   # lon/lat <-> UTM
import pygmt                                    # Fig. S1 maps
import matplotlib.pyplot as plt                 # Fig. S2
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.join(repo_dir, "data_prep"))
from run04_velocity import COH_VELOCITY, frame_grid, union_grid, place   # the grids used for the run04 mosaics

notebook = "01_prep_figures"
REG = utils.Registry(notebook, common_paths, repo_dir)
fig_dir = common_paths["fig_dir"]
os.makedirs(fig_dir, exist_ok=True)
MM = 1000.0

# %% [markdown]
# ## 2. Parameters and inputs
#
# - `ramp_lim`: one symmetric colour limit (mm/yr) for both correction columns
#   (troposphere and GNSS ramp), so that the two are drawn on the same scale.
# - `ref_half`: half-width (pixels) of the window for the value at the
#   reference station. The value is a median over 11 x 11 pixels (330 m),
#   because the reference pixel itself can be masked.
# - `nodata_rgb`: the grey background of the velocity and correction panels,
#   so that nodata is grey and zero is white.
# - `row_chunk`: rows of `TROPO.h5` read at a time. It limits memory use and
#   does not change the numbers.
# - `n_check_pixels`: pixels per frame refitted in section 5.
# - Fig. S2 colours: grey before the ramp, Okabe-Ito blue after the ramp; the
#   overlap map is drawn on a sea and land base.
#
# The reference station position is read from the screened GNSS table
# (velocities from the University of Nevada, Reno; Blewitt et al. 2018). The
# run04 coherence threshold must equal the one in `TulePond_filepaths.py`
# (asserted). Every input file is fingerprinted and recorded with each registry
# row.

# %%
ramp_lim = 1.0               # mm/yr, both correction columns
ref_half = 5                 # pixels
nodata_rgb = "204/204/204"   # 0.8 grey behind the velocity/correction panels
row_chunk = 200              # rows of TROPO.h5 read at a time (~0.7 GB)
n_check_pixels = 50          # pixels per frame refitted in section 5
c_before = "0.55"            # S2: before the ramp, grey
c_after = "#0072B2"          # S2: after the ramp, Okabe-Ito blue
sea_land = ["#cfe0ee", "#b0b0b0"]   # S2 overlap base: water_mask 0 = sea, 1 = land

assert COH_VELOCITY == coh_velocity, f"run04 masks velocities at {COH_VELOCITY}, TulePond_filepaths says {coh_velocity}"

ref_station = common_paths["ref_station"]
gnss_table_file = paths_gps["enu"].format(frame="09157")
gnss = pd.read_csv(gnss_table_file)
ref_row = gnss[gnss.StaID == ref_station]
assert len(ref_row) == 1, f"{ref_station} not in the screened GNSS table"
ref_lon = float(ref_row.Lon.iloc[0])
ref_lat = float(ref_row.Lat.iloc[0])
print(f"reference station {ref_station} at {ref_lon:.4f}, {ref_lat:.4f}")

input_files = [gnss_table_file, paths_gps["validation"]]
for frame_id, paths in frames.items():
    mintpy_dir = paths["mintpy_dir"]
    input_files.append(os.path.join(mintpy_dir, "avgSpatialCoh.h5"))
    input_files.append(os.path.join(mintpy_dir, "timeseries_density.h5"))
    input_files.append(os.path.join(mintpy_dir, "TROPO.h5"))
    input_files.append(paths["timeseries_tropo"])
    input_files.append(paths["velocity_masked"])
    input_files.append(paths["velocity_NA"])
REG.inputs(*input_files)

# %% [markdown]
# ## 3. Acquisitions per frame (Methods M1)
#
# The number of dates and the first and last date of each frame, read from the
# time series to which the velocities were fitted. The dates are kept for the
# `TROPO.h5` date check (section 4) and the refit (section 5).

# %%
def yyyymmdd_to_iso(date_string):
    """Convert a YYYYMMDD date string to ISO format.

    Parameters
    ----------
    date_string : str
        Date as 'YYYYMMDD', e.g. '20160729'.

    Returns
    -------
    str
        Date as 'YYYY-MM-DD', e.g. '2016-07-29'.
    """
    return f"{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}"


acq_rows = []
dates_by_frame = {}
for frame_id, paths in frames.items():
    with h5py.File(paths["timeseries_tropo"], "r") as f:
        dates = [d.decode() for d in f["date"][:]]
        orbit = str(f.attrs["ORBIT_DIRECTION"]).lower()
    dates_by_frame[frame_id] = dates
    acq_rows.append(dict(frame=frame_id, orbit=orbit, n=len(dates), first=dates[0], last=dates[-1]))
    REG.register(f"01_n_acquisitions_{frame_id}", len(dates), "count", 0, where="M1", cell="acquisitions")
    REG.register(f"01_first_date_{frame_id}", yyyymmdd_to_iso(dates[0]), "date", 0, where="M1", cell="acquisitions")
    REG.register(f"01_last_date_{frame_id}", yyyymmdd_to_iso(dates[-1]), "date", 0, where="M1", cell="acquisitions")
acquisitions = pd.DataFrame(acq_rows)
print(acquisitions.to_string(index=False))

# %% [markdown]
# ## 4. Fig. S1 layers, per frame
#
# Six layers on the grid of each frame:
# 1. OPERA average temporal coherence (`avgSpatialCoh.h5`)
# 2. data density, the fraction of epochs with a valid value
# 3. the LOS velocity before the tie, $v_{0}$ (`velocity_masked.h5`, coherence
#    0.7, zero at ZOA1)
# 4. the change in velocity due to the troposphere correction, $\Delta v_{trop}$
# 5. the GNSS ramp applied by run05, $\Delta v_{ramp} = v_{NA} - v_{0}$, where
#    $v_{NA}$ is the velocity after the tie (`velocity_NA.h5`); read from the
#    products, not refitted
# 6. (formed after the mosaic, section 6) the final velocity
#
# Exact zeros in a MintPy velocity file are nodata (the fit writes 0 where no
# data were available) and are set to NaN. The ramp layer is not set to zero per
# frame: run05 fits one quadratic per track plus one constant per frame, and the
# difference between the per-frame constants is an inter-frame offset.
#
# **Straight-line velocity.** The time of each acquisition is in decimal years,
# the MintPy convention,
#
# $$ t_i = Y_i + \frac{D_i - 1}{365.25}, $$
#
# where $Y_i$ is the year and $D_i$ the day of the year of acquisition $i$. The
# least-squares slope of a straight line $d_i = c + v\,t_i$ through all $n$
# dates is a weighted sum of the data,
#
# $$ v[d] = \sum_{i=1}^{n} w_i\,d_i, \qquad w_i = \frac{t_i - \bar t}{\sum_{j=1}^{n} (t_j - \bar t)^2}, $$
#
# where $d_i$ is the LOS displacement (mm) at date $i$, $c$ the intercept and
# $\bar t$ the mean of the $t_i$ (`slope_weights`).
#
# **Troposphere column.** run04 subtracts the OPERA TROPO delay $\tau_i$ from the
# time series before it fits the velocity with a straight line through every
# date (`polynomial 1`, no excluded dates). Because $v[\cdot]$ is linear in the
# data, the change in velocity due to the correction is
#
# $$ \Delta v_{trop} = v[d - \tau] - v[d] = -v[\tau], $$
#
# so the column is minus the straight-line slope of `TROPO.h5`, shown on the
# pixels of the velocity mask only. This requires `TROPO.h5` to carry the same
# dates as the time series (asserted in `tropo_velocity`) and the run04 velocity
# to be this straight-line fit (checked in section 5).

# %%
def read_velocity(path):
    """Read a MintPy velocity file in mm/yr.

    Parameters
    ----------
    path : str
        MintPy velocity file (HDF5, dataset `velocity` in m/yr).

    Returns
    -------
    numpy.ndarray
        Velocity (mm/yr), with the exact zeros (MintPy nodata) set to NaN.
    """
    with h5py.File(path, "r") as f:
        velocity = f["velocity"][:] * MM
    return np.where(velocity == 0, np.nan, velocity)


def decimal_years(date_strings):
    """Convert dates to decimal years with the MintPy convention.

    Parameters
    ----------
    date_strings : list of str
        Dates as 'YYYYMMDD'.

    Returns
    -------
    numpy.ndarray
        Decimal years, year + (day of year - 1) / 365.25.
    """
    dates = pd.to_datetime(date_strings, format="%Y%m%d")
    return np.asarray(dates.year + (dates.dayofyear - 1) / 365.25, dtype=float)


def slope_weights(times):
    """Weights that give the least-squares straight-line slope as a weighted sum.

    Parameters
    ----------
    times : numpy.ndarray
        Times of the samples (decimal years).

    Returns
    -------
    numpy.ndarray
        Weights w = (t - mean t) / sum((t - mean t)^2), so that sum(w * y) is
        the slope of the straight line fitted to y(times).
    """
    centred = times - times.mean()
    return centred / np.sum(centred ** 2)


def tropo_velocity(path, ts_dates):
    """Straight-line slope of the troposphere delay at every pixel.

    The fit is the same all-dates straight line as timeseries2velocity in
    run04. The cube is read in blocks of `row_chunk` rows.

    Parameters
    ----------
    path : str
        TROPO.h5 of one frame (dataset `timeseries` in m).
    ts_dates : list of str
        Dates ('YYYYMMDD') of the time series; TROPO.h5 must carry the same
        dates (asserted).

    Returns
    -------
    numpy.ndarray
        Slope of the delay (mm/yr), float32, on the frame grid.
    """
    with h5py.File(path, "r") as f:
        tropo_dates = [d.decode() for d in f["date"][:]]
        assert tropo_dates == ts_dates, (f"{path}: TROPO.h5 has {len(tropo_dates)} dates, the time series "
                                         f"{len(ts_dates)}; the troposphere column would differ from the applied correction")
        weights = slope_weights(decimal_years(tropo_dates))
        n_rows = f["timeseries"].shape[1]
        out = np.empty(f["timeseries"].shape[1:], dtype="float32")
        for row_start in range(0, n_rows, row_chunk):
            block = f["timeseries"][:, row_start:row_start + row_chunk, :]
            out[row_start:row_start + row_chunk] = np.tensordot(weights, block, axes=(0, 0)) * MM
    return out


layers_by_orbit = {}
for frame_id, paths in frames.items():
    mintpy_dir = paths["mintpy_dir"]
    grid = frame_grid(mintpy_dir)
    with h5py.File(os.path.join(mintpy_dir, "avgSpatialCoh.h5"), "r") as f:
        coh = f["avgSpatialCoh"][:]
    with h5py.File(os.path.join(mintpy_dir, "timeseries_density.h5"), "r") as f:
        density = f["timeseriesDensity"][:]
    orig = read_velocity(paths["velocity_masked"])
    # the change due to the troposphere correction is -velocity(TROPO), shown on the velocity-mask pixels only
    tropo_slope = tropo_velocity(os.path.join(mintpy_dir, "TROPO.h5"), dates_by_frame[frame_id])
    tropo = np.where(np.isfinite(orig), -tropo_slope, np.nan)
    # not set to zero per frame: run05 fits one quadratic per track plus one constant per
    # frame, and the difference between the per-frame constants is an inter-frame offset
    ramp = read_velocity(paths["velocity_NA"]) - orig
    if grid["orbit"] not in layers_by_orbit:
        layers_by_orbit[grid["orbit"]] = []
    layers_by_orbit[grid["orbit"]].append((grid, dict(coh=coh, density=density, orig=orig, tropo=tropo, ramp=ramp)))
    print(f"{frame_id} ({grid['orbit']}): {np.isfinite(orig).sum()} velocity pixels")

# %% [markdown]
# ## 5. Check: the straight-line fit reproduces the run04 velocity
#
# At `n_check_pixels` random pixels of each frame this section fits the straight
# line of section 4 (`slope_weights` on the time-series dates) to
# `timeseries_tropo.h5` and compares the result with `velocity_masked.h5`.
# Inside the velocity mask run04 fitted this series (the referenced time series
# minus TROPO), so the two must agree to float precision. Check: the largest
# $|v_{refit} - v_{0}|$ is below 1e-3 mm/yr on every frame.

# %%
check_rng = np.random.default_rng(0)
for frame_id, paths in frames.items():
    velocity_run04 = read_velocity(paths["velocity_masked"])
    weights = slope_weights(decimal_years(dates_by_frame[frame_id]))
    valid_rows, valid_cols = np.where(np.isfinite(velocity_run04))
    pick = check_rng.choice(len(valid_rows), size=n_check_pixels, replace=False)
    largest_diff = 0.0
    with h5py.File(paths["timeseries_tropo"], "r") as f:
        for row, col in zip(valid_rows[pick], valid_cols[pick]):
            series_mm = f["timeseries"][:, row, col] * MM
            assert np.all(np.isfinite(series_mm)), f"{frame_id}: NaN in the time series at a velocity pixel ({row}, {col})"
            refit = float(np.sum(weights * series_mm))
            largest_diff = max(largest_diff, abs(refit - float(velocity_run04[row, col])))
    print(f"{frame_id}: straight-line refit minus run04 velocity at {n_check_pixels} pixels, max |diff| {largest_diff:.2e} mm/yr")
    assert largest_diff < 1e-3, f"{frame_id}: the run04 velocity is not the straight-line fit of timeseries_tropo.h5"
print("Check passed: the run04 velocity is the straight-line fit, so -velocity(TROPO) is the applied correction")

# %% [markdown]
# ## 6. Mosaic the frames of each orbit and reference the corrections at ZOA1
#
# The two ascending frames are placed on the run04 union grid, the larger frame
# first. Each pixel is taken from the first frame whose coherence is finite
# there (its footprint), because density is a finite 0 outside the footprint.
#
# The two correction layers are then referenced at ZOA1,
#
# $$ x'(\mathbf{r}) = x(\mathbf{r}) - \operatorname{median}_{\mathbf{r}' \in W} x(\mathbf{r}'), $$
#
# where $x$ is $\Delta v_{trop}$ or $\Delta v_{ramp}$ (mm/yr), $\mathbf{r}$ a pixel,
# and $W$ the window of $(2\,\mathrm{ref\_half} + 1)^2$ pixels centred on the
# pixel of ZOA1 (NaN ignored). A uniform offset changes neither the velocity
# gradients nor the creep rates, and it is removed so that the corrections can
# be shown on the $\pm$1 mm/yr scale. The final velocity is
# $v_{final} = v_{0} + \Delta v'_{ramp}$.
#
# One velocity colour limit is used for both orbits,
# $v_{lim} = \lceil P_{98}(|v|) \rceil$, where $P_{98}$ is the 98th percentile
# of $|v|$ over all finite pixels of $v_{0}$ and $v_{final}$ of both orbits, and
# $\lceil\cdot\rceil$ rounds up to the next integer mm/yr.

# %%
def ref_rowcol(grid):
    """Row and column of the reference station's pixel on a UTM grid.

    Parameters
    ----------
    grid : dict
        UTM grid with `x0`, `y0` (first-pixel corner, m), `dx`, `dy` (pixel
        size, m; `dy` negative) and `epsg`.

    Returns
    -------
    tuple of int
        Row and column of the pixel that holds the reference station.
    """
    to_utm = pyproj.Transformer.from_crs(4326, grid["epsg"], always_xy=True)
    ref_x, ref_y = to_utm.transform(ref_lon, ref_lat)
    ref_row_index = int(round((ref_y - grid["y0"]) / grid["dy"]))
    ref_col_index = int(round((ref_x - grid["x0"]) / grid["dx"]))
    return ref_row_index, ref_col_index


def zero_at_ref(arr, grid):
    """Reference an array at the reference station.

    Parameters
    ----------
    arr : numpy.ndarray
        Layer on the grid `grid`.
    grid : dict
        UTM grid of `arr` (see `ref_rowcol`).

    Returns
    -------
    numpy.ndarray
        `arr` minus its median (NaN ignored) in the (2 ref_half + 1)^2 pixel
        window around the reference station.
    """
    ref_row_index, ref_col_index = ref_rowcol(grid)
    row_lo = max(ref_row_index - ref_half, 0)
    col_lo = max(ref_col_index - ref_half, 0)
    window = arr[row_lo:ref_row_index + ref_half + 1, col_lo:ref_col_index + ref_half + 1]
    assert np.isfinite(window).any(), "the reference station falls outside the valid data"
    return arr - np.nanmedian(window)


def frame_area(entry):
    """Number of pixels of one frame, used to put the larger frame first.

    Parameters
    ----------
    entry : tuple
        (grid, layers) of one frame.

    Returns
    -------
    int
        Width times length of the frame grid.
    """
    grid = entry[0]
    return grid["W"] * grid["L"]


def finite(a):
    """Finite values of an array.

    Parameters
    ----------
    a : numpy.ndarray
        Any array.

    Returns
    -------
    numpy.ndarray
        The finite values, flattened.
    """
    return a[np.isfinite(a)].ravel()


merged = {}
for orbit in ("descending", "ascending"):
    group = sorted(layers_by_orbit[orbit], key=frame_area, reverse=True)    # larger frame first
    grids = []
    for grid, _ in group:
        grids.append(grid)
    common_grid = union_grid(grids)
    mosaic = {}
    for key in group[0][1]:
        mosaic[key] = np.full((common_grid["L"], common_grid["W"]), np.nan, dtype="float32")
    covered = np.zeros((common_grid["L"], common_grid["W"]), bool)
    for grid, layers in group:
        take = np.isfinite(place(layers["coh"], grid, common_grid)) & ~covered
        for key, arr in layers.items():
            mosaic[key][take] = place(arr, grid, common_grid)[take]
        covered |= take
    mosaic["ramp"] = zero_at_ref(mosaic["ramp"], common_grid)
    mosaic["tropo"] = zero_at_ref(mosaic["tropo"], common_grid)
    mosaic["final"] = mosaic["orig"] + mosaic["ramp"]
    merged[orbit] = (common_grid, mosaic)

# one velocity colour limit for both orbits: the 98th percentile of |v| before and after the tie, rounded up
all_velocities = []
for common_grid, mosaic in merged.values():
    all_velocities.append(finite(mosaic["orig"]))
    all_velocities.append(finite(mosaic["final"]))
vel_lim = float(np.ceil(np.percentile(np.abs(np.concatenate(all_velocities)), 98)))
print(f"velocity colour limit +/-{vel_lim:g} mm/yr, correction limit +/-{ramp_lim:g} mm/yr")

# %% [markdown]
# ## 7. Pixels clipped by the fixed colour limits (sidecar)
#
# Per orbit and layer: the minimum, the maximum, the RMS,
# $\sqrt{\frac{1}{N}\sum_{k=1}^{N} x_k^2}$, and the percentage of pixels beyond
# the colour limit, $100\,N_{|x| > x_{lim}} / N$, where $x_k$ are the $N$ finite
# pixel values of the layer and $x_{lim}$ its colour limit. The troposphere RMS
# of each orbit is registered.

# %%
lines = ["FIGURE S1: layer ranges and clipping",
         f"coherence mask on the velocity columns : {COH_VELOCITY}",
         f"velocity colour limit                  : +/-{vel_lim:g} mm/yr",
         f"correction colour limit                : +/-{ramp_lim:g} mm/yr", "",
         f"{'orbit':<12}{'layer':<8}{'min':>9}{'max':>9}{'rms':>8}{'clipped':>9}"]
for orbit, (_, mosaic) in merged.items():
    for layer, lim in (("tropo", ramp_lim), ("ramp", ramp_lim), ("orig", vel_lim), ("final", vel_lim)):
        values = finite(mosaic[layer])
        lines.append(f"{orbit:<12}{layer:<8}{values.min():>9.2f}{values.max():>9.2f}{np.sqrt(np.mean(values ** 2)):>8.2f}"
                     f"{100 * np.mean(np.abs(values) > lim):>8.1f}%")
    REG.register(f"01_tropo_rms_{orbit[:3]}_mm_yr", float(np.sqrt(np.mean(finite(mosaic['tropo']) ** 2))), "mm/yr", 2,
                 where="M1, Fig S1", cell="figS1")
print("\n".join(lines))

# %% [markdown]
# ## 8. Fig. S1: the velocity chain, one row per orbit direction
#
# Drawn with pygmt, north up in longitude and latitude. Each mosaic is resampled
# to a regular longitude-latitude grid at about 100 m by nearest pixel
# (`utils.utm_to_lonlat_grid`, no interpolation), finer than a printed pixel of
# a 2.7 cm panel at 600 dpi (the PNG is written at 600 dpi). The velocity and
# correction panels have a grey background, so that nodata is grey and zero is
# white. ZOA1 is the black square, and a 10 km scale bar is drawn in panels C
# and I. There is one colour bar per column.
#
# GMT reduces the font size of a short colour bar (on a 2 cm bar a 12p font
# prints at about 6p), so the colour bars are drawn with 12p fonts to print at
# about 6p.

# %%
cpt_dir = os.path.join(REG.out_dir, "cpt")
os.makedirs(cpt_dir, exist_ok=True)

# the lon/lat box holding both orbit mosaics
region_s1 = None
for orbit, (common_grid, _) in merged.items():
    orbit_region = utils.utm_grid_region(common_grid["x0"], common_grid["y0"], common_grid["dx"], common_grid["dy"],
                                         common_grid["W"], common_grid["L"], common_grid["epsg"])
    if region_s1 is None:
        region_s1 = orbit_region
    else:
        region_s1 = [min(region_s1[0], orbit_region[0]), max(region_s1[1], orbit_region[1]),
                     min(region_s1[2], orbit_region[2]), max(region_s1[3], orbit_region[3])]
d_lon = 0.0012               # ~100 m: finer than a printed pixel at 600 dpi
d_lat = 0.001
map_width = 2.72             # cm, one panel
gap_x = 0.15                 # cm between panels
gap_y = 0.45                 # cm between rows
proj_s1 = f"M{map_width}c"
map_height = utils.gmt_map_size(region_s1, proj_s1)[1]
columns = [
    {"key": "coh", "title": "Coherence", "cmap": "magma", "lo": 0, "hi": 1, "grey": False,
     "cbar_label": "Coherence", "annot": "a0.5"},
    {"key": "density", "title": "Data density", "cmap": "cividis", "lo": 0, "hi": 1, "grey": False,
     "cbar_label": "Valid fraction", "annot": "a0.5"},
    {"key": "orig", "title": "Velocity before tie", "cmap": "RdBu_r", "lo": -vel_lim, "hi": vel_lim, "grey": True,
     "cbar_label": "LOS (mm/yr)", "annot": f"a{vel_lim:g}"},
    {"key": "tropo", "title": "Troposphere", "cmap": "PuOr_r", "lo": -ramp_lim, "hi": ramp_lim, "grey": True,
     "cbar_label": "Correction (mm/yr)", "annot": f"a{ramp_lim:g}"},
    {"key": "ramp", "title": "GNSS ramp", "cmap": "PuOr_r", "lo": -ramp_lim, "hi": ramp_lim, "grey": True,
     "cbar_label": "Correction (mm/yr)", "annot": f"a{ramp_lim:g}"},
    {"key": "final", "title": "Final velocity", "cmap": "RdBu_r", "lo": -vel_lim, "hi": vel_lim, "grey": True,
     "cbar_label": "LOS (mm/yr)", "annot": f"a{vel_lim:g}"},
]
row_width = (len(columns) - 1) * (map_width + gap_x)     # shift back from the last panel to the first
cpts = []
for column in columns:
    cpts.append(utils.write_cpt(os.path.join(cpt_dir, f"s1_{column['key']}.cpt"), column["cmap"], column["lo"], column["hi"]))
pygmt.config(FONT="7p,Helvetica", FONT_ANNOT_PRIMARY="6p,Helvetica", FONT_LABEL="6p,Helvetica", MAP_FRAME_TYPE="plain",
             MAP_FRAME_PEN="0.5p", MAP_TICK_LENGTH_PRIMARY="2p", MAP_ANNOT_OFFSET="2p", MAP_LABEL_OFFSET="2p")
fig = pygmt.Figure()
letters = "ABCDEFGHIJKL"
for i, orbit in enumerate(merged):
    common_grid, mosaic = merged[orbit]
    if i > 0:
        fig.shift_origin(xshift=f"-{row_width}c", yshift=f"-{map_height + gap_y}c")
    for j, column in enumerate(columns):
        if j > 0:
            fig.shift_origin(xshift=f"{map_width + gap_x}c")
        grid = utils.utm_to_lonlat_grid(mosaic[column["key"]], common_grid["x0"], common_grid["y0"], common_grid["dx"],
                                        common_grid["dy"], common_grid["epsg"], region_s1, d_lon, d_lat)
        if column["grey"]:
            fig.basemap(region=region_s1, projection=proj_s1, frame=[f"lrtb+g{nodata_rgb}"])
        else:
            fig.basemap(region=region_s1, projection=proj_s1, frame=["lrtb"])
        fig.grdimage(grid=grid, cmap=cpts[j], nan_transparent=True)
        if column["grey"]:
            fig.plot(x=[ref_lon], y=[ref_lat], style="s0.12c", fill="black", pen="0.3p,white")
        if j == 2:     # 10 km scale bar, on the velocity-before-tie panels (C and I)
            fig.basemap(region=region_s1, projection=proj_s1, frame=["lrtb"], map_scale="jBL+w10k+o0.15c/0.15c")
        else:
            fig.basemap(region=region_s1, projection=proj_s1, frame=["lrtb"])
        letter = letters[i * len(columns) + j]
        if i == 0:
            title = column["title"]
        else:
            title = ""
        fig.text(position="TL", text=f"@%Helvetica-Bold%{letter}@%%  " + title, justify="BL",
                 offset="0c/0.08c", font="6.5p,Helvetica", no_clip=True)
        if j == 0:
            fig.text(position="ML", text=orbit.capitalize(), angle=90, justify="BC", offset="-0.12c/0c", font="7p,Helvetica", no_clip=True)
fig.shift_origin(xshift=f"-{row_width}c")
with pygmt.config(FONT_ANNOT_PRIMARY="12p,Helvetica", FONT_LABEL="12p,Helvetica", MAP_TICK_LENGTH_PRIMARY="4p"):
    for j, column in enumerate(columns):
        fig.colorbar(cmap=cpts[j], position=f"x{j * (map_width + gap_x) + 0.1 * map_width}c/-0.2c+w{0.8 * map_width}c/0.18c+h+jTL",
                     frame=[f"x{column['annot']}+l{column['cbar_label']}"])
f_s1 = os.path.join(fig_dir, f"figS_velocity_chain_coh{COH_VELOCITY:g}_vel{vel_lim:g}_ramp{ramp_lim:g}")
fig.savefig(f_s1 + ".png", dpi=600, transparent=False, crop="+gwhite", anti_alias=True, show=False)
fig.savefig(f_s1 + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
print("wrote", f_s1 + ".png")

# %% [markdown]
# ## 9. Fig. S2 data: the GNSS tie per station
#
# The per-station table of run05, pooled by orbit. A station sampled in both
# ascending frames is averaged to one row (the two frames share a track and look
# geometry), so the panels count unique stations.
#
# In run05 the GNSS velocity of each station is projected onto the LOS of its
# own pixels, $g_i = \hat{\mathbf{l}}_i \cdot \mathbf{v}^{GNSS}_i$, where
# $\hat{\mathbf{l}}_i$ is the unit vector from the ground to the satellite and
# $\mathbf{v}^{GNSS}_i$ the east, north and up velocity of station $i$. The
# residual is $r_i = (g_i - g_{ZOA1}) - s_i$, where $g_{ZOA1}$ is the GNSS LOS
# velocity of ZOA1 and $s_i$ the InSAR LOS velocity at the station, before the
# ramp ($r^{b}_i$, `residual_before`) or after it ($r^{a}_i$, `residual_after`),
# all in mm/yr. This section references both
# axes to ZOA1: the GNSS value is $g_i - g_{ZOA1}$ (`gnss_los - gref`), and the
# InSAR value before or after the ramp is $g_i - g_{ZOA1} - r_i$. This shifts
# the axes only and leaves every residual unchanged. The misfit of each orbit is
#
# $$ \mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} r_i^2}, $$
#
# over its $n$ unique stations, before and after the ramp. It is in-sample; the
# run05 sidecar lists the leave-one-out values.

# %%
tie = pd.read_csv(paths_gps["validation"])
orbits = ["ascending", "descending"]
stations = {}
for orbit in orbits:
    stations[orbit] = tie[tie.orbit == orbit].groupby("StaID", as_index=False).mean(numeric_only=True)


def rmse(a, b=0.0):
    """Root-mean-square of a difference.

    Parameters
    ----------
    a : array_like
        Values.
    b : array_like or float, optional
        Values subtracted from `a` (default 0).

    Returns
    -------
    float
        sqrt(mean((a - b)^2)).
    """
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def referenced(table):
    """GNSS and InSAR LOS velocities of the stations, relative to ZOA1.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-station tie table of run05 (`gnss_los`, `gref`,
        `residual_before`, `residual_after`; mm/yr).

    Returns
    -------
    tuple of numpy.ndarray
        GNSS LOS velocity, InSAR LOS velocity before the ramp and InSAR LOS
        velocity after the ramp (mm/yr), all relative to ZOA1.
    """
    g_los = table.gnss_los.to_numpy() - table.gref.to_numpy()
    return g_los, g_los - table.residual_before.to_numpy(), g_los - table.residual_after.to_numpy()


tie_stats = {}
for orbit in orbits:
    g_los, before, after = referenced(stations[orbit])
    tie_stats[orbit] = dict(n=len(stations[orbit]), before=rmse(g_los, before), after=rmse(g_los, after))
    REG.register(f"01_gnss_n_stations_{orbit[:3]}", len(stations[orbit]), "count", 0, where="M1, Fig S2", cell="figS2")
    REG.register(f"01_gnss_rmse_before_{orbit[:3]}_mm_yr", tie_stats[orbit]["before"], "mm/yr", 1, where="M1, Fig S2", cell="figS2")
    REG.register(f"01_gnss_rmse_after_{orbit[:3]}_mm_yr", tie_stats[orbit]["after"], "mm/yr", 1, where="M1, Fig S2", cell="figS2")
common = set(stations["ascending"].StaID) & set(stations["descending"].StaID)
REG.register("01_gnss_n_stations_common", len(common), "count", 0, where="M1", cell="figS2")
print(pd.DataFrame(tie_stats).T.round(3))

# %% [markdown]
# ## 10. Fig. S2 data: the two ascending frames where they overlap
#
# An internal comparison independent of GNSS: the velocity difference
# $\Delta v = v_{0}^{09156} - v_{0}^{09157}$ (both before the tie, coherence
# 0.7, zero at ZOA1) on the grid of the larger frame, with the smaller frame
# placed into it by integer pixel offset. The water mask of the larger frame is
# the land and sea base. The number of overlapping pixels, and the median and
# RMS of $\Delta v$ over them, are registered.

# %%
def load_velocity(path):
    """Read a MintPy velocity file and its grid.

    Parameters
    ----------
    path : str
        MintPy velocity file (HDF5, dataset `velocity` in m/yr).

    Returns
    -------
    velocity : numpy.ndarray
        Velocity (mm/yr), with the exact zeros (MintPy nodata) set to NaN.
    grid : dict
        `x0`, `y0`, `dx`, `dy` (m), `W`, `L` (pixels) and `epsg` of the grid.
    """
    with h5py.File(path, "r") as h:
        attrs = h.attrs
        grid = dict(x0=float(attrs["X_FIRST"]), y0=float(attrs["Y_FIRST"]), dx=float(attrs["X_STEP"]),
                    dy=float(attrs["Y_STEP"]), W=int(attrs["WIDTH"]), L=int(attrs["LENGTH"]), epsg=int(attrs["EPSG"]))
        velocity = h["velocity"][:] * MM
    return np.where(velocity == 0, np.nan, velocity), grid


# the ascending frames in the tie table, in order of first appearance
asc_frames = []
for frame_number in tie.loc[tie.orbit == "ascending", "frame"]:
    frame_name = f"{int(frame_number):05d}"
    if frame_name not in asc_frames:
        asc_frames.append(frame_name)
assert len(asc_frames) == 2, f"expected two ascending frames, got {asc_frames}"

# the larger frame carries the grid; the smaller one is placed into it by integer pixel offset
vel_first, grid_first = load_velocity(frames[asc_frames[0]]["velocity_masked"])
vel_second, grid_second = load_velocity(frames[asc_frames[1]]["velocity_masked"])
if grid_first["W"] * grid_first["L"] < grid_second["W"] * grid_second["L"]:
    big_frame = asc_frames[1]
    vel_big, grid_big, vel_small, grid_small = vel_second, grid_second, vel_first, grid_first
else:
    big_frame = asc_frames[0]
    vel_big, grid_big, vel_small, grid_small = vel_first, grid_first, vel_second, grid_second
overlap = np.full(vel_big.shape, np.nan)
col_offset = round((grid_small["x0"] - grid_big["x0"]) / grid_big["dx"])
row_offset = round((grid_small["y0"] - grid_big["y0"]) / grid_big["dy"])
row_start = max(0, row_offset)
row_end = min(grid_big["L"], row_offset + grid_small["L"])
col_start = max(0, col_offset)
col_end = min(grid_big["W"], col_offset + grid_small["W"])
overlap[row_start:row_end, col_start:col_end] = (vel_big[row_start:row_end, col_start:col_end]
                                                 - vel_small[row_start - row_offset:row_end - row_offset,
                                                             col_start - col_offset:col_end - col_offset])
stack_files = sorted(glob.glob(os.path.join(frames[big_frame]["dir"], "*_stack*.nc")))
assert len(stack_files) == 1, f"expected one rebased stack in {frames[big_frame]['dir']}, found {stack_files}"
REG.inputs(stack_files[0])
water = xr.open_dataset(stack_files[0])["water_mask"].values
assert water.shape == overlap.shape, "water mask and velocity grid differ"
# the difference is (09156 - 09157) whichever frame is larger
if big_frame != "09156":
    overlap = -overlap
d_overlap = finite(overlap)
overlap_lim = float(np.nanpercentile(np.abs(d_overlap), 98))
REG.register("01_overlap_n_pixels", int(d_overlap.size), "count", 0, where="M1, Fig S2", cell="figS2")
REG.register("01_overlap_median_mm_yr", float(np.median(d_overlap)), "mm/yr", 2, where="M1, Fig S2", cell="figS2")
REG.register("01_overlap_rms_mm_yr", rmse(d_overlap), "mm/yr", 2, where="M1, Fig S2", cell="figS2")
lines += ["", "FIGURE S2: GNSS tie and frame overlap",
          "RMSE is in-sample over unique stations (frames averaged)"]
for orbit, orbit_stats in tie_stats.items():
    lines.append(f"{orbit:<12} n={orbit_stats['n']:<4} RMSE {orbit_stats['before']:.3f} -> {orbit_stats['after']:.3f} mm/yr")
lines.append(f"overlap 09156 - 09157: {d_overlap.size} pixels, median {np.median(d_overlap):+.3f}, rms {rmse(d_overlap):.3f} mm/yr")
REG.sidecar("01_prep_figures", lines)
print("\n".join(lines[-4:]))

# %% [markdown]
# ## 11. Fig. S2: GNSS against InSAR, residuals, and the frame overlap
#
# Drawn with matplotlib. Top row: GNSS against InSAR LOS velocity for each orbit
# (before the ramp, open grey; after the ramp, blue), and the overlap map with
# the GNSS stations. Bottom row: the residual distributions and the overlap
# distribution. The scatter span and the histogram bins are shared by both
# orbits.

# %%
utils.mpl_paper_style(base=7)

# one scatter span for both orbits, centred on each orbit's own range
spans = []
centre = {}
for orbit in orbits:
    all_los = np.concatenate(referenced(stations[orbit]))
    spans.append(np.nanmax(all_los) - np.nanmin(all_los))
    centre[orbit] = (np.nanmax(all_los) + np.nanmin(all_los)) / 2
span = max(spans) * 1.15

# one set of histogram bins and one count axis for both orbits
residual_parts = []
for orbit in orbits:
    residual_parts.append(stations[orbit].residual_before.to_numpy())
    residual_parts.append(stations[orbit].residual_after.to_numpy())
all_res = np.concatenate(residual_parts)
h_span = max(6.0, np.nanpercentile(np.abs(all_res), 98) * 1.1)
bins = np.linspace(-h_span, h_span, 21)
largest_count = 0
for orbit in orbits:
    for column_name in ("residual_before", "residual_after"):
        counts = np.histogram(stations[orbit][column_name], bins=bins)[0]
        largest_count = max(largest_count, counts.max())
h_max = int(largest_count * 1.15 + 1)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
for j, orbit in enumerate(orbits):
    g_los, before, after = referenced(stations[orbit])
    lim = (centre[orbit] - span / 2, centre[orbit] + span / 2)
    ax = axes[0, j]
    ax.plot(lim, lim, "--", lw=0.7, color="0.6", zorder=1)
    ax.scatter(before, g_los, s=16, facecolors="none", edgecolors=c_before, linewidths=0.8, label="Before the ramp", zorder=2)
    ax.scatter(after, g_los, s=12, color=c_after, label="After the ramp", zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", "box")
    ticks = []
    for tick in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(*lim):
        if lim[0] <= tick <= lim[1]:
            ticks.append(tick)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("InSAR LOS velocity (mm/yr)")
    ax.set_ylabel("GNSS LOS velocity (mm/yr)")
    orbit_stats = tie_stats[orbit]
    ax.text(0.04, 0.96, f"RMSE {orbit_stats['before']:.1f} $\\rightarrow$ {orbit_stats['after']:.1f} mm/yr\nn = {orbit_stats['n']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6)
    utils.panel_title(ax, "AB"[j], orbit.capitalize())
    if j == 0:
        ax.legend(loc="lower right", fontsize=6)
    ax = axes[1, j]
    ax.axvline(0, lw=0.7, color="0.6")
    ax.hist(stations[orbit].residual_before, bins=bins, color=c_before, alpha=0.55, label="Before the ramp")
    ax.hist(stations[orbit].residual_after, bins=bins, color=c_after, alpha=0.75, label="After the ramp")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, h_max)
    ax.set_xlabel("GNSS minus InSAR (mm/yr)")
    ax.set_ylabel("Stations")
    utils.panel_title(ax, "DE"[j], f"{orbit.capitalize()} residuals")

# the overlap map, on the larger frame's pixel grid, with the GNSS stations that fall on it
ax = axes[0, 2]
ax.imshow(water, cmap=ListedColormap(sea_land), vmin=-0.5, vmax=1.5, interpolation="none", zorder=0)
im = ax.imshow(overlap, cmap="RdBu_r", vmin=-overlap_lim, vmax=overlap_lim, interpolation="nearest", zorder=1)
all_stations = pd.concat(stations.values()).drop_duplicates("StaID")
to_utm_big = pyproj.Transformer.from_crs(4326, grid_big["epsg"], always_xy=True)
station_x, station_y = to_utm_big.transform(all_stations.Lon.values, all_stations.Lat.values)
station_rows = (station_y - grid_big["y0"]) / grid_big["dy"]
station_cols = (station_x - grid_big["x0"]) / grid_big["dx"]
inside = (station_rows >= 0) & (station_rows < grid_big["L"]) & (station_cols >= 0) & (station_cols < grid_big["W"])
ax.scatter(station_cols[inside], station_rows[inside], marker="^", s=10, c="k", edgecolors="white", linewidths=0.4, zorder=3)
bar_px = 10000 / abs(grid_big["dx"])
ax.plot([0.05 * grid_big["W"], 0.05 * grid_big["W"] + bar_px], [0.94 * grid_big["L"]] * 2, "-", color="k", lw=1.2,
        solid_capstyle="butt", zorder=4)
ax.text(0.05 * grid_big["W"] + bar_px / 2, 0.92 * grid_big["L"], "10 km", ha="center", va="bottom", fontsize=6, zorder=4)
ax.set_xticks([])
ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02)
cb.set_label("09156 minus 09157 (mm/yr)", fontsize=6)
cb.ax.tick_params(labelsize=6)
utils.panel_title(ax, "C", "Ascending frame overlap")
ax = axes[1, 2]
ax.axvline(0, lw=0.7, color="0.6")
ax.hist(d_overlap, bins=np.linspace(-overlap_lim, overlap_lim, 41), color=c_after, alpha=0.8)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_xlabel("09156 minus 09157 (mm/yr)")
ax.set_ylabel("Pixels")
ax.text(0.04, 0.96, f"median {np.median(d_overlap):+.2f} mm/yr\nRMS {rmse(d_overlap):.2f} mm/yr",
        transform=ax.transAxes, va="top", ha="left", fontsize=6)
utils.panel_title(ax, "F", "Overlap difference")
f_s2 = os.path.join(fig_dir, f"figS_gnss_tie_coh{COH_VELOCITY:g}_nasc{tie_stats['ascending']['n']}_ndesc{tie_stats['descending']['n']}.png")
fig.savefig(f_s2, dpi=300, facecolor="white")
fig.savefig(f_s2.replace(".png", ".pdf"), facecolor="white")
print("wrote", f_s2)

# %% [markdown]
# ## 12. Render the registry

# %%
REG.render()
