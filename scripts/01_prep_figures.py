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
# # [1] 01 Prep figures: the velocity chain (Fig. S1) and the GNSS tie (Fig. S2)
#
# ```
# Created on Tue Sep 29 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. About 5 minutes (the troposphere column reads each TROPO.h5 once).
# ```
#
# **What this notebook does.** Draws the two supplement figures that document
# the prepared velocity fields, from the products `data_prep/run04` and `run05`
# wrote on the current data (March 2026 refresh), and registers the numbers
# their captions and Methods M1 quote. Nothing here changes a product.
#
# Ported from `Hayward_DISP-S1/plotting/figS1_processing.py` and
# `figS2_gnss_tie.py` (July 2026 data), with two changes:
# - the troposphere column is the velocity of `TROPO.h5` itself. The old
#   script differenced `velocity_notropo.h5`, which run04 no longer writes (the
#   July copy was moved to `MISC_2026-09-28/` as stale). The velocity fit is a
#   straight line (run04: `polynomial 1`, no excluded dates), and a straight-line
#   fit is linear in the data, so velocity(ts - TROPO) - velocity(ts) =
#   -velocity(TROPO) exactly. Nothing is approximated.
# - Science style: Helvetica, bold capital panel letters, 7 pt text, 18.3 cm
#   wide, colour-blind-safe colour maps (cividis for the data density, not
#   viridis).
#
# Outputs: `Figures_<date>/figS1_velocity_chain_*.png/.pdf`,
# `figS2_gnss_tie_*.png/.pdf`, and the stats sidecar in `outputs/01_prep_figures/`.

# %%
# [2]
import glob
import os
import sys

import insar_utils as utils
from TulePond_filepaths import common_paths, data_dir, frames, paths_gps, repo_dir

import numpy as np                              # Matrix calculations
import pandas as pd                             # Tables
import h5py                                     # MintPy HDF5 products
import xarray as xr                             # the rebased stack (water mask)
import pyproj                                   # lon/lat <-> UTM

sys.path.insert(0, os.path.join(repo_dir, "data_prep"))
from run04_velocity import COH_VELOCITY, frame_grid, union_grid, place   # the grids run04 mosaicked on

notebook = "01_prep_figures"
REG = utils.Registry(notebook, common_paths, repo_dir)
fig_dir = common_paths["fig_dir"]
os.makedirs(fig_dir, exist_ok=True)
MM = 1000.0

# %% [markdown]
# ## [3] Parameters
#
# - `ramp_lim`: one symmetric colour limit for BOTH correction columns
#   (troposphere and GNSS ramp), so the two can be compared directly.
# - the velocity limit is the 98th percentile of |v| over both orbits, rounded
#   up, never per panel: ascending and descending differ in size and nothing may
#   autoscale.
# - `ref_half`: half-window (pixels) for the value at the reference station,
#   a median over ~330 m because the reference pixel itself can be masked.

# %%
# [4]
ramp_lim = 1.0          # mm/yr, both correction columns
ref_half = 5            # pixels
nodata_grey = "0.8"     # velocity/correction panels: nodata grey so white reads as zero
row_chunk = 200         # rows of TROPO.h5 read at a time (~0.7 GB)
c_before, c_after = "0.55", "#0072B2"          # S2: before the ramp grey, after blue
sea_land = ["#cfe0ee", "#b0b0b0"]               # S2 overlap base: water_mask 0 = sea, 1 = land

ref_station = common_paths["ref_station"]
gnss = pd.read_csv(paths_gps["enu"].format(frame="09157", ref_frame="NA"))
ref_row = gnss[gnss.StaID == ref_station]
assert len(ref_row) == 1, f"{ref_station} not in the screened GNSS table"
ref_lon, ref_lat = float(ref_row.Lon.iloc[0]), float(ref_row.Lat.iloc[0])
print(f"reference station {ref_station} at {ref_lon:.4f}, {ref_lat:.4f}")

# %% [markdown]
# ## [5] Acquisitions per frame (Methods M1)
#
# The number of dates and the first and last date of each frame, read from
# the time series the velocities were fitted to.

# %%
# [6]
acq_rows = []
for fid, paths in frames.items():
    with h5py.File(paths["timeseries_tropo"], "r") as f:
        dates = [d.decode() for d in f["date"][:]]
        orbit = str(f.attrs["ORBIT_DIRECTION"]).lower()
    acq_rows.append(dict(frame=fid, orbit=orbit, n=len(dates), first=dates[0], last=dates[-1]))
    iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"
    REG.register(f"01_n_acquisitions_{fid}", len(dates), "count", 0, where="M1", cell="acquisitions")
    REG.register(f"01_first_date_{fid}", iso(dates[0]), "date", 0, where="M1", cell="acquisitions")
    REG.register(f"01_last_date_{fid}", iso(dates[-1]), "date", 0, where="M1", cell="acquisitions")
acquisitions = pd.DataFrame(acq_rows)
print(acquisitions.to_string(index=False))

# %% [markdown]
# ## [7] Fig. S1 layers, per frame
#
# Six layers on each frame's own grid:
# 1. OPERA average temporal coherence (`avgSpatialCoh.h5`)
# 2. data density, the fraction of epochs with a valid value
# 3. the LOS velocity before the tie (`velocity_masked.h5`, coherence 0.7, ZOA1 datum)
# 4. the change the troposphere correction made to the velocity, -velocity(TROPO)
# 5. the GNSS ramp run05 applied, `velocity_NA - velocity_masked` (not refitted
#    here: this is exactly the correction that was applied)
# 6. (formed after the mosaic) the final velocity, 3 + 5
#
# Exact zeros in a MintPy velocity are nodata (the fit writes 0 where it had
# nothing to fit) and become NaN.

# %%
# [8]
def read_velocity(path):
    with h5py.File(path, "r") as f:
        v = f["velocity"][:] * MM
    return np.where(v == 0, np.nan, v)


def tropo_velocity(path):
    """Straight-line slope (mm/yr) of TROPO.h5 at every pixel, the same
    all-dates fit run04's timeseries2velocity makes, read in row chunks."""
    with h5py.File(path, "r") as f:
        dates = pd.to_datetime([d.decode() for d in f["date"][:]])
        t = np.asarray(dates.year + (dates.dayofyear - 1) / 365.25, dtype=float)
        tc = t - t.mean()
        weights = tc / np.sum(tc ** 2)
        n_rows = f["timeseries"].shape[1]
        out = np.empty(f["timeseries"].shape[1:], dtype="float32")
        for r0 in range(0, n_rows, row_chunk):
            block = f["timeseries"][:, r0:r0 + row_chunk, :]
            out[r0:r0 + row_chunk] = np.tensordot(weights, block, axes=(0, 0)) * MM
    return out


layers_by_orbit = {}
for fid, paths in frames.items():
    mdir = paths["mintpy_dir"]
    grid = frame_grid(mdir)
    with h5py.File(os.path.join(mdir, "avgSpatialCoh.h5"), "r") as f:
        coh = f["avgSpatialCoh"][:]
    with h5py.File(os.path.join(mdir, "timeseries_density.h5"), "r") as f:
        density = f["timeseriesDensity"][:]
    orig = read_velocity(paths["velocity_masked"])
    # velocity(ts - TROPO) - velocity(ts) = -velocity(TROPO); shown on the velocity-mask pixels only
    tropo = np.where(np.isfinite(orig), -tropo_velocity(os.path.join(mdir, "TROPO.h5")), np.nan)
    # not zeroed per frame: run05 fits one quadratic per track plus a constant per
    # frame, and the difference of the per-frame constants is a real inter-frame offset
    ramp = read_velocity(paths["velocity_NA"]) - orig
    layers_by_orbit.setdefault(grid["orbit"], []).append(
        (grid, dict(coh=coh, density=density, orig=orig, tropo=tropo, ramp=ramp)))
    print(f"{fid} ({grid['orbit']}): {np.isfinite(orig).sum()} velocity pixels")

# %% [markdown]
# ## [9] Mosaic same-orbit frames and reference the corrections at ZOA1
#
# The two ascending frames go onto run04's union grid, the larger frame first;
# a pixel is taken from the first frame whose COHERENCE is finite there (its
# footprint), since density is a finite 0 outside the footprint. The two
# correction columns are then referenced at ZOA1: a uniform offset changes no
# gradient and no creep rate, and would otherwise swamp the +/-1 mm/yr scale.

# %%
# [10]
to_utm = {}


def ref_rowcol(cg):
    if cg["epsg"] not in to_utm:
        to_utm[cg["epsg"]] = pyproj.Transformer.from_crs(4326, cg["epsg"], always_xy=True)
    x, y = to_utm[cg["epsg"]].transform(ref_lon, ref_lat)
    return int(round((y - cg["y0"]) / cg["dy"])), int(round((x - cg["x0"]) / cg["dx"]))


def zero_at_ref(arr, cg):
    r, c = ref_rowcol(cg)
    win = arr[max(r - ref_half, 0):r + ref_half + 1, max(c - ref_half, 0):c + ref_half + 1]
    assert np.isfinite(win).any(), "reference station falls outside the valid data"
    return arr - np.nanmedian(win)


merged = {}
for orbit in ("descending", "ascending"):
    group = sorted(layers_by_orbit[orbit], key=lambda gl: -gl[0]["W"] * gl[0]["L"])
    cg = union_grid([g for g, _ in group])
    lay = {k: np.full((cg["L"], cg["W"]), np.nan, dtype="float32") for k in group[0][1]}
    covered = np.zeros((cg["L"], cg["W"]), bool)
    for grid, layers in group:
        take = np.isfinite(place(layers["coh"], grid, cg)) & ~covered
        for k, arr in layers.items():
            lay[k][take] = place(arr, grid, cg)[take]
        covered |= take
    lay["ramp"] = zero_at_ref(lay["ramp"], cg)
    lay["tropo"] = zero_at_ref(lay["tropo"], cg)
    lay["final"] = lay["orig"] + lay["ramp"]
    merged[orbit] = (cg, lay)


def finite(a):
    return a[np.isfinite(a)].ravel()


vel_lim = float(np.ceil(np.nanpercentile(np.abs(np.concatenate(
    [np.concatenate([finite(l["orig"]), finite(l["final"])]) for _, l in merged.values()])), 98)))
print(f"velocity colour limit +/-{vel_lim:g} mm/yr, correction limit +/-{ramp_lim:g} mm/yr")

# %% [markdown]
# ## [11] What the fixed colour limits clip (sidecar)

# %%
# [12]
lines = ["FIGURE S1: layer ranges and clipping",
         f"coherence mask on the velocity columns : {COH_VELOCITY}",
         f"velocity colour limit                  : +/-{vel_lim:g} mm/yr",
         f"correction colour limit                : +/-{ramp_lim:g} mm/yr", "",
         f"{'orbit':<12}{'layer':<8}{'min':>9}{'max':>9}{'rms':>8}{'clipped':>9}"]
for orbit, (_, l) in merged.items():
    for k, lim in (("tropo", ramp_lim), ("ramp", ramp_lim), ("orig", vel_lim), ("final", vel_lim)):
        d = finite(l[k])
        lines.append(f"{orbit:<12}{k:<8}{d.min():>9.2f}{d.max():>9.2f}{np.sqrt(np.mean(d ** 2)):>8.2f}"
                     f"{100 * np.mean(np.abs(d) > lim):>8.1f}%")
    REG.register(f"01_tropo_rms_{orbit[:3]}_mm_yr", float(np.sqrt(np.mean(finite(l['tropo']) ** 2))), "mm/yr", 2,
                 where="M1, Fig S1", cell="figS1")
print("\n".join(lines))

# %% [markdown]
# ## [13] Fig. S1: the velocity chain, one row per orbit direction
#
# pygmt, north up in lon/lat (Dani 2026-09-29: pygmt for anything with a map).
# Each mosaic is resampled to a regular lon/lat grid by nearest pixel
# (`utils.utm_to_lonlat_grid`, no interpolation) at ~100 m, finer than a
# printed pixel of a 2.7 cm panel at 600 dpi (the PNG is 600 dpi). The velocity and correction panels sit
# on grey so nodata reads grey and white reads as zero; ZOA1 is the black
# square; a 10 km scale bar in C and I. One colour bar per column.
#
# GMT shrinks the fonts of a short colour bar (a 2 cm bar prints 12p as ~6p,
# tested 2026-09-29), so the bars are drawn with 12p to print at ~6p.

# %%
# [14]
import pygmt

cpt_dir = os.path.join(REG.out_dir, "cpt"); os.makedirs(cpt_dir, exist_ok=True)
region_s1 = None
for orbit, (cg, _) in merged.items():
    r_ = utils.utm_grid_region(cg["x0"], cg["y0"], cg["dx"], cg["dy"], cg["W"], cg["L"], cg["epsg"])
    region_s1 = r_ if region_s1 is None else [min(region_s1[0], r_[0]), max(region_s1[1], r_[1]), min(region_s1[2], r_[2]), max(region_s1[3], r_[3])]
d_lon, d_lat = 0.0012, 0.001                          # ~100 m: finer than a printed pixel at 600 dpi
Wm, gap_x, gap_y = 2.72, 0.15, 0.45
proj_s1 = f"M{Wm}c"
Hm = utils.gmt_map_size(region_s1, proj_s1)[1]
columns = [("coh", "Coherence", "magma", 0, 1, False, "Coherence", "a0.5"),
           ("density", "Data density", "cividis", 0, 1, False, "Valid fraction", "a0.5"),
           ("orig", "Velocity before tie", "RdBu_r", -vel_lim, vel_lim, True, "LOS (mm/yr)", f"a{vel_lim:g}"),
           ("tropo", "Troposphere", "PuOr_r", -ramp_lim, ramp_lim, True, "Correction (mm/yr)", f"a{ramp_lim:g}"),
           ("ramp", "GNSS ramp", "PuOr_r", -ramp_lim, ramp_lim, True, "Correction (mm/yr)", f"a{ramp_lim:g}"),
           ("final", "Final velocity", "RdBu_r", -vel_lim, vel_lim, True, "LOS (mm/yr)", f"a{vel_lim:g}")]
cpts = [utils.write_cpt(os.path.join(cpt_dir, f"s1_{k}.cpt"), cmap, lo, hi) for k, _, cmap, lo, hi, *_ in columns]
pygmt.config(FONT="7p,Helvetica", FONT_ANNOT_PRIMARY="6p,Helvetica", FONT_LABEL="6p,Helvetica", MAP_FRAME_TYPE="plain",
             MAP_FRAME_PEN="0.5p", MAP_TICK_LENGTH_PRIMARY="2p", MAP_ANNOT_OFFSET="2p", MAP_LABEL_OFFSET="2p")
fig = pygmt.Figure()
letters = iter("ABCDEFGHIJKL")
for i, orbit in enumerate(merged):
    cg, lay = merged[orbit]
    if i:
        fig.shift_origin(xshift=f"-{5 * (Wm + gap_x)}c", yshift=f"-{Hm + gap_y}c")
    for j, (key, head, _, _, _, grey, _, _) in enumerate(columns):
        if j:
            fig.shift_origin(xshift=f"{Wm + gap_x}c")
        grid = utils.utm_to_lonlat_grid(lay[key], cg["x0"], cg["y0"], cg["dx"], cg["dy"], cg["epsg"], region_s1, d_lon, d_lat)
        fig.basemap(region=region_s1, projection=proj_s1, frame=["lrtb+g204/204/204" if grey else "lrtb"])
        fig.grdimage(grid=grid, cmap=cpts[j], nan_transparent=True)
        if grey:
            fig.plot(x=[ref_lon], y=[ref_lat], style="s0.12c", fill="black", pen="0.3p,white")
        fig.basemap(region=region_s1, projection=proj_s1, frame=["lrtb"], **({"map_scale": "jBL+w10k+o0.15c/0.15c"} if j == 2 else {}))
        fig.text(position="TL", text=f"@%Helvetica-Bold%{next(letters)}@%%  " + (head if i == 0 else ""), justify="BL",
                 offset="0c/0.08c", font="6.5p,Helvetica", no_clip=True)
        if j == 0:
            fig.text(position="ML", text=orbit.capitalize(), angle=90, justify="BC", offset="-0.12c/0c", font="7p,Helvetica", no_clip=True)
fig.shift_origin(xshift=f"-{5 * (Wm + gap_x)}c")
with pygmt.config(FONT_ANNOT_PRIMARY="12p,Helvetica", FONT_LABEL="12p,Helvetica", MAP_TICK_LENGTH_PRIMARY="4p"):
    for j, (_, _, _, _, _, _, cb_label, annot) in enumerate(columns):
        fig.colorbar(cmap=cpts[j], position=f"x{j * (Wm + gap_x) + 0.1 * Wm}c/-0.2c+w{0.8 * Wm}c/0.18c+h+jTL",
                     frame=[f"x{annot}+l{cb_label}"])
f_s1 = os.path.join(fig_dir, f"figS1_velocity_chain_coh{COH_VELOCITY:g}_vel{vel_lim:g}_ramp{ramp_lim:g}")
fig.savefig(f_s1 + ".png", dpi=600, transparent=False, crop="+gwhite", anti_alias=True, show=False)
fig.savefig(f_s1 + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
print("wrote", f_s1 + ".png")

# %% [markdown]
# ## [15] Fig. S2 data: the GNSS tie per station
#
# run05's per-station table, pooled by orbit. A station sampled in both
# ascending frames is averaged to one row (the two share a track and look
# geometry), so the panels count unique STATIONS. Both axes are referenced to
# ZOA1 (its absolute LOS is subtracted); this shifts the axes only, and leaves
# every residual and RMSE unchanged. The RMSE is in-sample, before and after
# the ramp; run05's sidecar carries the leave-one-out numbers.

# %%
# [16]
REG.inputs(paths_gps["validation"])
tie = pd.read_csv(paths_gps["validation"])
orbits = ["ascending", "descending"]
stations = {o: tie[tie.orbit == o].groupby("StaID", as_index=False).mean(numeric_only=True) for o in orbits}


def rmse(a, b=0.0):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def referenced(g):
    """GNSS LOS, and InSAR LOS before and after the ramp, all relative to ZOA1."""
    g_los = g.gnss_los.to_numpy() - g.gref.to_numpy()
    return g_los, g_los - g.residual_before.to_numpy(), g_los - g.residual_after.to_numpy()


tie_stats = {}
for o in orbits:
    g_los, before, after = referenced(stations[o])
    tie_stats[o] = dict(n=len(stations[o]), before=rmse(g_los, before), after=rmse(g_los, after))
    REG.register(f"01_gnss_n_stations_{o[:3]}", len(stations[o]), "count", 0, where="M1, Fig S2", cell="figS2")
    REG.register(f"01_gnss_rmse_before_{o[:3]}_mm_yr", tie_stats[o]["before"], "mm/yr", 1, where="M1, Fig S2", cell="figS2")
    REG.register(f"01_gnss_rmse_after_{o[:3]}_mm_yr", tie_stats[o]["after"], "mm/yr", 1, where="M1, Fig S2", cell="figS2")
common = set(stations["ascending"].StaID) & set(stations["descending"].StaID)
REG.register("01_gnss_n_stations_common", len(common), "count", 0, where="M1", cell="figS2")
print(pd.DataFrame(tie_stats).T.round(3))

# %% [markdown]
# ## [17] Fig. S2 data: the two ascending frames where they overlap
#
# An internal check that uses no GNSS: 09156 minus 09157 velocity (both
# untied, coherence 0.7, ZOA1 datum) on the larger frame's grid, with that
# frame's water mask as a land/sea base.

# %%
# [18]
def load_velocity(path):
    with h5py.File(path, "r") as h:
        a = h.attrs
        g = dict(x0=float(a["X_FIRST"]), y0=float(a["Y_FIRST"]), dx=float(a["X_STEP"]), dy=float(a["Y_STEP"]),
                 W=int(a["WIDTH"]), L=int(a["LENGTH"]), epsg=int(a["EPSG"]))
        return h["velocity"][:] * MM, g


asc_frames = [f"{int(f):05d}" for f in dict.fromkeys(tie.loc[tie.orbit == "ascending", "frame"])]
assert len(asc_frames) == 2, f"expected two ascending frames, got {asc_frames}"
va, ga = load_velocity(frames[asc_frames[0]]["velocity_masked"])
vb, gb = load_velocity(frames[asc_frames[1]]["velocity_masked"])
big_dir = frames[asc_frames[0]]["dir"]
name_a, name_b = asc_frames
if ga["W"] * ga["L"] < gb["W"] * gb["L"]:              # a must be the larger frame
    va, ga, vb, gb, big_dir, name_a, name_b = vb, gb, va, ga, frames[asc_frames[1]]["dir"], name_b, name_a
va, vb = np.where(va == 0, np.nan, va), np.where(vb == 0, np.nan, vb)
overlap = np.full(va.shape, np.nan)
ox, oy = round((gb["x0"] - ga["x0"]) / ga["dx"]), round((gb["y0"] - ga["y0"]) / ga["dy"])
r0, r1, c0, c1 = max(0, oy), min(ga["L"], oy + gb["L"]), max(0, ox), min(ga["W"], ox + gb["W"])
overlap[r0:r1, c0:c1] = va[r0:r1, c0:c1] - vb[r0 - oy:r1 - oy, c0 - ox:c1 - ox]
water = xr.open_dataset(sorted(glob.glob(os.path.join(big_dir, "*_stack*.nc")))[0])["water_mask"].values
assert water.shape == overlap.shape, "water mask and velocity grid differ"
# the difference is (09156 - 09157) whichever frame is larger
if name_a != "09156":
    overlap = -overlap
d_overlap = finite(overlap)
overlap_lim = float(np.nanpercentile(np.abs(d_overlap), 98))
REG.register("01_overlap_n_pixels", int(d_overlap.size), "count", 0, where="M1, Fig S2", cell="figS2")
REG.register("01_overlap_median_mm_yr", float(np.median(d_overlap)), "mm/yr", 2, where="M1, Fig S2", cell="figS2")
REG.register("01_overlap_rms_mm_yr", rmse(d_overlap), "mm/yr", 2, where="M1, Fig S2", cell="figS2")
lines += ["", "FIGURE S2: GNSS tie and frame overlap",
          "RMSE is in-sample over unique stations (frames averaged)"]
lines += [f"{o:<12} n={s['n']:<4} RMSE {s['before']:.3f} -> {s['after']:.3f} mm/yr" for o, s in tie_stats.items()]
lines += [f"overlap 09156 - 09157: {d_overlap.size} pixels, median {np.median(d_overlap):+.3f}, rms {rmse(d_overlap):.3f} mm/yr"]
sidecar = REG.sidecar("01_prep_figures", lines)
print("\n".join(lines[-4:]))

# %% [markdown]
# ## [19] Fig. S2: GNSS against InSAR, residuals, and the frame overlap
#
# matplotlib (Dani 2026-09-29: S2 may stay matplotlib). Top row: GNSS against
# InSAR LOS velocity for each orbit (before the ramp open grey, after blue),
# and the overlap map with the GNSS stations. Bottom row: the residual
# distributions and the overlap distribution. One scatter span and one set of
# histogram bins for both orbits.

# %%
# [20]
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

utils.mpl_paper_style(base=7)
spans, centre = [], {}
for o in orbits:
    v = np.concatenate(referenced(stations[o]))
    spans.append(np.nanmax(v) - np.nanmin(v))
    centre[o] = (np.nanmax(v) + np.nanmin(v)) / 2
span = max(spans) * 1.15
all_res = np.concatenate([np.r_[stations[o].residual_before, stations[o].residual_after] for o in orbits])
h_span = max(6.0, np.nanpercentile(np.abs(all_res), 98) * 1.1)
bins = np.linspace(-h_span, h_span, 21)
h_max = int(max(np.histogram(stations[o][col], bins=bins)[0].max()
                for o in orbits for col in ("residual_before", "residual_after")) * 1.15 + 1)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
for j, o in enumerate(orbits):
    g_los, before, after = referenced(stations[o])
    lim = (centre[o] - span / 2, centre[o] + span / 2)
    ax = axes[0, j]
    ax.plot(lim, lim, "--", lw=0.7, color="0.6", zorder=1)
    ax.scatter(before, g_los, s=16, facecolors="none", edgecolors=c_before, linewidths=0.8, label="Before the ramp", zorder=2)
    ax.scatter(after, g_los, s=12, color=c_after, label="After the ramp", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", "box")
    ticks = [t for t in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(*lim) if lim[0] <= t <= lim[1]]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel("InSAR LOS velocity (mm/yr)"); ax.set_ylabel("GNSS LOS velocity (mm/yr)")
    s_ = tie_stats[o]
    ax.text(0.04, 0.96, f"RMSE {s_['before']:.1f} $\\rightarrow$ {s_['after']:.1f} mm/yr\nn = {s_['n']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6)
    utils.panel_title(ax, "AB"[j], o.capitalize())
    if j == 0:
        ax.legend(loc="lower right", fontsize=6)
    ax = axes[1, j]
    ax.axvline(0, lw=0.7, color="0.6")
    ax.hist(stations[o].residual_before, bins=bins, color=c_before, alpha=0.55, label="Before the ramp")
    ax.hist(stations[o].residual_after, bins=bins, color=c_after, alpha=0.75, label="After the ramp")
    ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(0, h_max)
    ax.set_xlabel("GNSS minus InSAR (mm/yr)"); ax.set_ylabel("Stations")
    utils.panel_title(ax, "DE"[j], f"{o.capitalize()} residuals")

ax = axes[0, 2]
ax.imshow(water, cmap=ListedColormap(sea_land), vmin=-0.5, vmax=1.5, interpolation="none", zorder=0)
im = ax.imshow(overlap, cmap="RdBu_r", vmin=-overlap_lim, vmax=overlap_lim, interpolation="nearest", zorder=1)
sx, sy = pyproj.Transformer.from_crs(4326, ga["epsg"], always_xy=True).transform(
    pd.concat(stations.values()).drop_duplicates("StaID").Lon.values,
    pd.concat(stations.values()).drop_duplicates("StaID").Lat.values)
rows, cols = (sy - ga["y0"]) / ga["dy"], (sx - ga["x0"]) / ga["dx"]
inside = (rows >= 0) & (rows < ga["L"]) & (cols >= 0) & (cols < ga["W"])
ax.scatter(cols[inside], rows[inside], marker="^", s=10, c="k", edgecolors="white", linewidths=0.4, zorder=3)
bar_px = 10000 / abs(ga["dx"])
ax.plot([0.05 * ga["W"], 0.05 * ga["W"] + bar_px], [0.94 * ga["L"]] * 2, "-", color="k", lw=1.2, solid_capstyle="butt", zorder=4)
ax.text(0.05 * ga["W"] + bar_px / 2, 0.92 * ga["L"], "10 km", ha="center", va="bottom", fontsize=6, zorder=4)
ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02)
cb.set_label("09156 minus 09157 (mm/yr)", fontsize=6); cb.ax.tick_params(labelsize=6)
utils.panel_title(ax, "C", "Ascending frame overlap")
ax = axes[1, 2]
ax.axvline(0, lw=0.7, color="0.6")
ax.hist(d_overlap, bins=np.linspace(-overlap_lim, overlap_lim, 41), color=c_after, alpha=0.8)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_xlabel("09156 minus 09157 (mm/yr)"); ax.set_ylabel("Pixels")
ax.text(0.04, 0.96, f"median {np.median(d_overlap):+.2f} mm/yr\nRMS {rmse(d_overlap):.2f} mm/yr",
        transform=ax.transAxes, va="top", ha="left", fontsize=6)
utils.panel_title(ax, "F", "Overlap difference")
f_s2 = os.path.join(fig_dir, f"figS2_gnss_tie_coh{COH_VELOCITY:g}_nasc{tie_stats['ascending']['n']}_ndesc{tie_stats['descending']['n']}.png")
fig.savefig(f_s2, dpi=300, facecolor="white")
fig.savefig(f_s2.replace(".png", ".pdf"), facecolor="white")
print("wrote", f_s2)

# %%
# [21]
REG.render()
