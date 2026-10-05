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
# # 07c Fig 2: the event field and the 3D slip model
#
# ```
# Created on Mon Sep 28 2026
# @author: daniellelindsay
# Run in opera_disp_feb13, after 06, 07a and 08a (run_all.sh runs 08a and 08b before 07c). Seconds.
# ```
#
# **What this notebook does.** Draws Fig 2 from notebook 06's maps and profiles
# (`fig2_inputs.npz`), notebook 04's crossings (panel K's grey dots and black line) and
# the model row of `08a_3d_model`: the selected 3D model `fig2_model_row_pylith_nuggetcd.npz`
# (panels I-L and the vermillion lines on G, H) with the homogeneous PyLith model of the
# same inversion `fig2_model_row_pylith_nuggetcd_homog.npz` (the blue lines on G, H). It
# registers nothing: every number on the figure was registered where it was made. The
# only computation here is for display: the Gaussian K line (profile_smooth_km, as Fig
# 1C), the per-side running medians on the profiles, and the colour limits.
#
# Fig 2 is the fault-parallel and vertical version (Dani, 2026-09-28); the per-track LOS
# version is the supplement figure.
# PREPUB: Moved out of `06_event_field_geometry` on 2026-09-28 so the model row is the real model.
#
# **Inputs**: `06_event_field_geometry/fig2_inputs.npz`;
# `08a_3d_model/fig2_model_row_pylith_nuggetcd.npz` and `..._nuggetcd_homog.npz`;
# `07b_uniform_model_tests/fig2_model_row.npz` (07a's product: the quadtree samples the 08a
# products must be on); `04_event_detection/event_fraction_by_crossing.csv` and
# `creep_events.csv`; `hayward_fault_profiles.csv`; the alignment-array summary;
# the GHCN file of the Fremont rain gauge; the GNSS station list of the descending frame
# (`GPS/11115_StaList.txt`, the reference station ZOA1); the CFW1 position
# (`TulePond_filepaths.creepmeters`).
#
# **Figures**: `fig2_fault_parallel_vertical_coh<c>_<n>ev_pylith_nuggetcd` (Fig 2) and
# `figS_los_maps_profiles_coh<c>_<n>ev` (the LOS version, supplement). Small GMT helper
# files (CPTs, legend specs, block polygons) go to `outputs/07c_fig2/cpt/`.
#
# **What to check**
# - [8] the model row is read from `08a_3d_model/fig2_model_row_pylith_nuggetcd.npz`, and
#   both 08a products are on 07a's current quadtree samples (asserted).
# - [9] the K line's crossing count.
# - [13] the per-block amplitudes printed for Fig 2 equal 06's registry rows `06_XX_*`, `06_YY_*`.
# - [10] the model product's crossings are 04's five-event means (asserted).
# - On the figures: the model row (I-L) uses the fonts of the map and profile panels; panel L
#   uses magma (no blue/green); figS and Fig 2 draw the model in the same colour.

# %% [markdown]
# ## [1] Imports

# %%
## [1]
import os

import insar_utils as utils
from TulePond_filepaths import (common_paths, creepmeters, paths_out, paths_gps, des11115_dic,
                                fig2_map_centre_km, fig2_map_half_km, fig2_map_extra_km, profile_smooth_km)

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import pygmt

# %% [markdown]
# ## [2] Constants
#
# Colours (Okabe-Ito), fonts, the running-median bin of the profiles, the map window, the
# named alignment arrays and the input files.

# %%
## [2]
notebook = "07c_fig2"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
cpt_dir = os.path.join(out_dir, "cpt")                 # GMT helper files: CPTs, legend specs, block polygons
os.makedirs(cpt_dir, exist_ok=True)

# the running median drawn on the profiles
median_bin_m = 150.0            # running-median bin (five pixels), taken on each side of the fault separately
median_col = "black"            # per-side running median
grey = "gray55"                 # pixels
homog_col = "#0072B2"           # the homogeneous PyLith model (Okabe-Ito blue)
hetero_col = "#D55E00"          # the 3D (aligned SFCVM) model, the paper's model (Okabe-Ito vermillion)
model_col = hetero_col

# fonts: the model row (I-L) uses the same sizes as the map and profile panels
panel_letter_font = "9p,Helvetica-Bold"     # A-L (Science: bold capital panel letters)
heading_font = "9p,Helvetica-Bold"          # column headings, row labels, the I/J titles
ylabel_font = "9p,Helvetica"                # the y-axis labels of the profile and model panels
annot_font = "7p,Helvetica"                 # tick annotations, x-axis labels, map labels, the Mw text
legend_font = "6p,Helvetica"                # the line legends (G and K)
profile_label_font = "7p,Helvetica-Bold"    # X, X', Y, Y' on every map and profile panel (one size throughout)
map_bar_font_pt = 13.0                      # the map colour bars ask GMT for 13 pt (GMT shrinks it on the page, see [6])

# Fig 2 map window (unrotated, north up): centred on the trace at fig2_map_centre_km from Point Pinole,
# +/- fig2_map_half_km east-west, +/- (fig2_map_half_km + fig2_map_extra_km) north-south (10.4 x 12 km).
# The values live in TulePond_filepaths so Fig 1 (notebook 04) draws the same box.
aa_labelled = ["HGIL", "HRKT"]  # the arrays named in the paper (notebook 09_alignment_arrays); the other arrays of the section are not drawn
aa_label_pos = {"HGIL": ("RM", "-0.15c/0c"),     # HGIL left of its symbol
                "HRKT": ("LM", "0.15c/0.12c")}   # (justify, offset) of each label about its symbol

f_fig2_in = os.path.join(paths_out["06_event_field_geometry"], "fig2_inputs.npz")
aa_file = common_paths["aa_summary"]
pylith_model = os.path.join(paths_out["08a_3d_model"], "fig2_model_row_pylith_nuggetcd.npz")             # the paper's model
pylith_compare = os.path.join(paths_out["08a_3d_model"], "fig2_model_row_pylith_nuggetcd_homog.npz")    # its homogeneous twin
row07a_file = os.path.join(paths_out["07b_uniform_model_tests"], "fig2_model_row.npz")                   # 07a writes it there

# %% [markdown]
# ## [3] Load: the maps and profiles from notebook 06, the trace
#
# The per-track maps and profiles were made in 06; here they are only unpacked into
# dictionaries by field (fault-parallel, vertical, ascending, descending) and line (X, Y).
# The grid's pixel centres are on a regular UTM lattice (asserted), so a pixel is found by
# its index.

# %%
## [3]
fig2_in = np.load(f_fig2_in)
print(str(fig2_in["provenance"]))
grid_x = fig2_in["X"]
grid_y = fig2_in["Y"]
n_rows, n_cols = grid_x.shape
strike = float(fig2_in["strike"])
coh_maps = float(fig2_in["coh_maps"])
cross_m = float(fig2_in["cross_m"])
line_names = [str(name) for name in fig2_in["names"]]
tracks = ["ascending", "descending"]
n_mean = int(fig2_in["n_events"])
rate_los = {}
event_los = {}
for track in tracks:
    rate_los[track] = fig2_in[f"rate_{track}"]
    event_los[track] = fig2_in[f"event_{track}"]


def unpack(field):
    """The profiles of one field (fp, up, ascending, descending) as 06 held them: for each
    line, the pixel distances and values of the rate ("d", "v") and of the event ("de", "ve"),
    the line ends, and each block's near-fault amplitude and its error."""
    out = {}
    for line in line_names:
        profile = {}
        for key in ("d", "v", "de", "ve", "ends"):
            profile[key] = fig2_in[f"prof_{field}_{line}_{key}"]
        profile["blocks"] = {}
        for block in ("W", "E"):
            profile["blocks"][block] = dict(amp=float(fig2_in[f"prof_{field}_{line}_{block}_amp"]),
                                            se=float(fig2_in[f"prof_{field}_{line}_{block}_se"]))
        out[line] = profile
    return out


fp_profiles = unpack("fp")
up_profiles = unpack("up")
los_profiles = {}
for track in tracks:
    los_profiles[track] = unpack(track)

epsg = int(fig2_in["epsg"])
to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
to_ll = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
dx = float(grid_x[0, 1] - grid_x[0, 0])
dy = float(grid_y[1, 0] - grid_y[0, 0])
assert np.allclose(np.diff(grid_x, axis=1), dx) and np.allclose(np.diff(grid_y, axis=0), dy), "the 06 grid is not regular"
x_origin = float(grid_x[0, 0]) - 0.5 * dx           # outer corner of the first pixel
y_origin = float(grid_y[0, 0]) - 0.5 * dy

# the trace, and the unit normal pointing to the east (NE) block, as in 06
prof = pd.read_csv(common_paths["fault_profiles"])
trace_x, trace_y = to_utm.transform(prof.ft_lon.values, prof.ft_lat.values)
trace_tree = cKDTree(np.c_[trace_x, trace_y])
east_normal = np.array([np.sin(np.deg2rad(strike - 90)), np.cos(np.deg2rad(strike - 90))])


def running_median(dist, values, bin_m=median_bin_m):
    """Binned median on each side of the fault separately: the bin edges start at
    the trace and run outward, so no bin (and no drawn segment) straddles the
    fault. Bins with fewer than 3 pixels are skipped. Returns a list of
    (bin centres, medians) arrays, one per side (west, east)."""
    out = []
    for side_sign in (-1, 1):
        edges = np.sort(side_sign * np.arange(0.0, cross_m + bin_m, bin_m))
        bin_index = np.digitize(dist, edges)
        centres = []
        medians = []
        for i in range(1, len(edges)):
            in_bin = (bin_index == i) & (side_sign * dist > 0)
            if in_bin.sum() >= 3:
                centres.append(0.5 * (edges[i - 1] + edges[i]))
                medians.append(np.median(values[in_bin]))
        out.append((np.array(centres), np.array(medians)))
    return out


# %% [markdown]
# ## [4] The map window and the regular lon/lat grid
#
# A lon/lat box about the trace point fig2_map_centre_km from Point Pinole; the 06 maps are
# resampled to a regular lon/lat grid (the nearest 30 m pixel). `to_grid` optionally
# balances a map so the fault sits at zero (the mid-point of the two block medians).

# %%
## [4]
km_of = utils.lat_to_km_map(common_paths["fault_profiles"])
centre_index = int(np.argmin(np.abs(km_of(prof.ft_lat.values) - fig2_map_centre_km)))
centre_x, centre_y = to_utm.transform(prof.ft_lon[centre_index], prof.ft_lat[centre_index])
centre_x = float(centre_x)
centre_y = float(centre_y)
half_ns_km = fig2_map_half_km + fig2_map_extra_km
lon_w, lat_s = to_ll.transform(centre_x - fig2_map_half_km * 1e3, centre_y - half_ns_km * 1e3)
lon_e, lat_n = to_ll.transform(centre_x + fig2_map_half_km * 1e3, centre_y + half_ns_km * 1e3)
region_map = [lon_w, lon_e, lat_s, lat_n]
print(f"map window: {2 * fig2_map_half_km:g} x {2 * half_ns_km:g} km about {fig2_map_centre_km:g} km "
      f"({prof.ft_lat[centre_index]:.4f} N)")

# the maps on a regular lon/lat grid (nearest 30 m pixel)
lat_q = np.arange(lat_s - 0.002, lat_n + 0.002, 0.00025)
lon_q = np.arange(lon_w - 0.002, lon_e + 0.002, 0.0003)
lon_grid, lat_grid = np.meshgrid(lon_q, lat_q)
east_q, north_q = to_utm.transform(lon_grid, lat_grid)
col_q = np.round((east_q - x_origin) / dx - 0.5).astype(int)
row_q = np.round((north_q - y_origin) / dy - 0.5).astype(int)
on_06_grid = (col_q >= 0) & (col_q < n_cols) & (row_q >= 0) & (row_q < n_rows)
_, nearest_trace = trace_tree.query(np.c_[east_q.ravel(), north_q.ravel()])
offset_from_trace = np.c_[east_q.ravel(), north_q.ravel()] - np.c_[trace_x, trace_y][nearest_trace]
side_q = np.dot(offset_from_trace, east_normal).reshape(east_q.shape)    # + east of the trace


def to_grid(field_map, balance):
    """One 06 map on the regular lon/lat grid (xarray), balanced (fault at zero: the
    mid-point of the two block medians in the window) when `balance`."""
    values = np.full(lon_grid.shape, np.nan)
    values[on_06_grid] = field_map[row_q[on_06_grid], col_q[on_06_grid]]
    if balance:
        values -= 0.5 * (np.nanmedian(values[side_q < 0]) + np.nanmedian(values[side_q > 0]))
    return xr.DataArray(values, coords={"lat": lat_q, "lon": lon_q}, dims=("lat", "lon"))


def gmt_map_size(region, projection):
    """Width and height (cm) GMT gives a region/projection pair (mapproject -W)."""
    size_file = os.path.join(cpt_dir, "dims.txt")
    with pygmt.clib.Session() as lib:
        lib.call_module("mapproject", f"-R{'/'.join(str(r) for r in region)} -J{projection} -W ->{size_file}")
    with open(size_file) as handle:
        width_cm, height_cm = handle.read().split()
    return float(width_cm), float(height_cm)


lines_ll = {}
for line, profile in fp_profiles.items():
    x_west, y_west, x_east, y_east = profile["ends"]
    lines_ll[line] = to_ll.transform(np.array([x_west, x_east]), np.array([y_west, y_east]))
cfw_rows = []
for meter in creepmeters:
    if meter[0] == "CFW1":
        cfw_rows.append(meter)
assert len(cfw_rows) == 1, "CFW1 must appear once in TulePond_filepaths.creepmeters"
cfw = cfw_rows[0]                                       # (code, lat, lon, name)
aa = utils.load_aa_summary(aa_file)
aa = aa[(aa.long_deg > lon_w) & (aa.long_deg < lon_e) & (aa.lat_deg > lat_s) & (aa.lat_deg < lat_n)]
print("alignment arrays in the window:", ", ".join(aa.site_code))

# %% [markdown]
# ## [5] The sites on the maps: rain gauge, GNSS reference station, named places
#
# On panel B: the Fremont rain gauge, the one gauge the rainfall statistics use over the InSAR
# period (position from its GHCN file; Dani 2026-10-03: Fremont only), and the GNSS station
# ZOA1, the reference of every InSAR and GNSS product (`common_paths['ref_station']`; position
# from the GNSS station list, asserted inside the map). The two ACWD wells are no longer drawn
# (Dani 2026-10-05). ZOA1 and the gauge are ~0.1 km apart, under one symbol width on the map, so
# ZOA1 (a black star) is drawn on the gauge's white diamond, ZOA1 labelled above and Fremont right.
# The places named on the vertical event map (panel F).

# %%
## [5]
rain_gauges = {}
for station_id, name in (("USC00043244", "Fremont"),):            # the gauge of the InSAR-period rainfall statistics
    head = pd.read_csv(os.path.join(common_paths["ghcn_dir"], f"{station_id}.csv"), nrows=1)
    rain_gauges[name] = (float(head.LATITUDE.iloc[0]), float(head.LONGITUDE.iloc[0]))
for name, (gauge_lat, gauge_lon) in rain_gauges.items():
    print(f"rain gauge {name} {gauge_lat:.4f} {gauge_lon:.4f}")
gauge_col = "white"             # gauges: white diamonds (no blue/green pair)
# the GNSS reference station, from the station list of the descending frame (data_prep/run00 writes one per frame)
gnss_name = common_paths["ref_station"]
sta_list_file = paths_gps["StaList"].format(frame=des11115_dic["Frame"])
sta_list = pd.read_csv(sta_list_file, sep=r"\s+", header=None, names=["site", "lat", "lon", "start_yr", "end_yr"])
gnss_rows = sta_list[sta_list.site == gnss_name]
assert len(gnss_rows) == 1, f"{gnss_name} must appear once in {sta_list_file}"
gnss_lat = float(gnss_rows.lat.iloc[0])
gnss_lon = float(gnss_rows.lon.iloc[0])
assert lon_w < gnss_lon < lon_e and lat_s < gnss_lat < lat_n, f"{gnss_name} ({gnss_lat}, {gnss_lon}) is outside the Fig 2 map window {region_map}"
gnss_x, gnss_y = to_utm.transform(gnss_lon, gnss_lat)
for name, (gauge_lat, gauge_lon) in rain_gauges.items():
    gauge_x, gauge_y = to_utm.transform(gauge_lon, gauge_lat)
    print(f"GNSS {gnss_name} {gnss_lat:.4f} {gnss_lon:.4f} ({os.path.basename(sta_list_file)}), "
          f"{np.hypot(gnss_x - gauge_x, gnss_y - gauge_y):.0f} m from the {name} gauge")
gnss_col = "black"              # GNSS: a black star (arrays are black squares, CFW1 a yellow triangle, gauges white diamonds)
# places named on the vertical event map (panel F): (label lines, lon, lat, leader dx, dy in cm); the label sits at the
# leader's end, two lines, above it for an up leader and below it for a down leader, left of it for a leader to the
# left (clear of X' and the frame). Tule Pond (Dani 2026-10-02): the pond itself, labelled bottom left of the fault
# between X and Y.
map_places = [(("Alameda", "Creek"), -121.980765, 37.572280, 0.3, 0.45),
              (("Lake", "Elizabeth"), -121.962053, 37.549023, 0.4, 0.12),
              (("Tule", "Pond"), -121.975545, 37.558686, -0.35, -0.45)]

# %% [markdown]
# ## [6] Panel sizes and GMT settings
#
# A double-column Science figure (<= 18.3 cm). The maps are made exactly as tall as a
# stacked pair of profile panels. GMT shrinks colour-bar text on short bars (by the square
# root of the bar length, measured with GMT 6.6 on 2026-10-03), so `colour_bar_font_pt`
# asks for a larger size on a shorter bar: every colour bar then shows its text at the size
# of the map colour bars.

# %%
## [6]
Wp, Hp = 3.4, 1.85                                  # one profile panel, width and height (cm)
gap_pair, gap_col, gap_row = 0.25, 0.65, 1.25       # within a stacked pair; between the two profile columns (room for the +/-2000 labels); between rows
Hpair = 2 * Hp + gap_pair                           # a stacked pair; the maps are made exactly this tall
map_w_4cm, map_h_4cm = gmt_map_size(region_map, "M4c")
Wm = 4.0 * Hpair / map_h_4cm                        # one map, width and height (cm)
Hm = Hpair
proj_map = f"M{Wm:.3f}c"
gap_map, cb_w, gap_cb = 0.25, 0.3, 0.3              # between the two maps; colour-bar width; map-to-bar gap
map_bar_length = Hm - 0.6                           # the map colour bars
x_prof = 2 * Wm + gap_map + gap_cb + cb_w + 1.35    # left edge of the profile columns (room for the bar label and the y annotation)
W_model = 2 * Wp + gap_col                          # the slip-model panels span both profile columns
print(f"map panels {Wm:.2f} x {Hm:.2f} cm; figure ~{x_prof + W_model + 0.7:.1f} x {3 * Hpair + 2 * gap_row + 0.9:.1f} cm")
lim_prof_los = {"rate": 6.0, "event": 6.0}          # LOS profile axes (mm/yr, mm)
lim_prof_fpup = {"rate": 10.0, "event": 10.0}       # fault-parallel and vertical profile axes (mm/yr, mm)
pygmt.config(FONT="8p,Helvetica", FONT_ANNOT_PRIMARY=annot_font, FONT_LABEL=annot_font, MAP_FRAME_TYPE="plain",
             MAP_FRAME_PEN="0.6p", MAP_TICK_LENGTH_PRIMARY="2p", MAP_LABEL_OFFSET="4p", MAP_ANNOT_OFFSET="3p", FORMAT_GEO_MAP="ddd.xx")


def colour_bar_font_pt(bar_length_cm):
    """The font size (pt) to ask GMT for on a colour bar of this length so its text shows at
    the size of the map colour bars (map_bar_font_pt asked on bars of map_bar_length cm).
    GMT shrinks colour-bar text by the square root of the bar length, so the asked size
    grows by sqrt(map_bar_length / bar_length_cm)."""
    return map_bar_font_pt * np.sqrt(map_bar_length / bar_length_cm)

# %% [markdown]
# ## [7] The drawing helpers: maps, colour bars, profile pairs
#
# Each helper draws into the figure `fig` it is given, from the current origin, and says
# where it leaves the origin.

# %%
## [7]
def profile_line_labels(fig):
    """X-X' and Y-Y' labels at the ends of the profile lines."""
    for line, (line_lon, line_lat) in lines_ll.items():
        fig.text(x=line_lon[0], y=line_lat[0], text=line, justify="RM", offset="-0.08c/0c", font=profile_label_font)
        fig.text(x=line_lon[1], y=line_lat[1], text=line + "'", justify="LM", offset="0.08c/0c", font=profile_label_font)


def map_base(fig, grid, cpt, label, stations=True, places=False):
    """One map, north up, in lon/lat: the field, the trace, the profile lines, and
    (stations) CFW1, the named alignment arrays, the rain gauges and GNSS ZOA1; (places)
    the named places with leader lines. No colour bar."""
    fig.basemap(region=region_map, projection=proj_map, frame=["lrtb"])
    fig.grdimage(grid=grid, cmap=cpt, nan_transparent=True)
    fig.plot(x=prof.ft_lon.values, y=prof.ft_lat.values, pen="0.7p,black")
    for line_lon, line_lat in lines_ll.values():
        fig.plot(x=line_lon, y=line_lat, pen="1.0p,black")
    profile_line_labels(fig)
    if places:                                     # a leader line from each place to its label (GMT text -D...+v)
        line_h = 0.27                              # cm between the two label lines (7 pt)
        for (first, second), place_lon, place_lat, leader_dx, leader_dy in map_places:
            leader_up = leader_dy > 0
            if leader_up:                          # the line nearer the point carries the leader
                near_line, far_line = second, first
                far_dy = leader_dy + line_h
                justify = "B"
            else:
                near_line, far_line = first, second
                far_dy = leader_dy - line_h
                justify = "T"
            if leader_dx >= 0:
                justify += "L"
            else:
                justify += "R"
            fig.text(x=place_lon, y=place_lat, text=near_line, justify=justify,
                     offset=f"{leader_dx}c/{leader_dy}c+v0.6p,black", font=annot_font)   # the leader
            for text, text_dy in ((near_line, leader_dy), (far_line, far_dy)):            # both lines on a translucent white box
                utils.text_backed(fig, x=place_lon, y=place_lat, text=text, justify=justify,
                                  offset=f"{leader_dx}c/{text_dy}c", font=annot_font)
    if stations:
        used = aa.site_code.isin(aa_labelled)                                    # only the arrays the paper uses are drawn
        fig.plot(x=aa.long_deg.values[used], y=aa.lat_deg.values[used], style="s0.18c", fill="black", pen="0.4p,white")
        for _, array_row in aa[used].iterrows():                                 # HGIL left of its symbol, HRKT right and nudged up off the Y-Y' line
            justify, offset = aa_label_pos[array_row.site_code]
            utils.text_backed(fig, x=array_row.long_deg, y=array_row.lat_deg, text=array_row.site_code,
                              justify=justify, offset=offset, font=annot_font)
        fig.plot(x=[cfw[2]], y=[cfw[1]], style="i0.22c", fill="#F0E442", pen="0.5p,black")
        utils.text_backed(fig, x=cfw[2], y=cfw[1], text="CFW1", justify="RM", offset="-0.15c/0c", font=annot_font)
        # the Fremont rain gauge (label right) and, on it, GNSS ZOA1 (label above, nudged left off the X label: left of it runs off the map)
        for name, (gauge_lat, gauge_lon) in rain_gauges.items():
            fig.plot(x=[gauge_lon], y=[gauge_lat], style="d0.22c", fill=gauge_col, pen="0.6p,black")
            utils.text_backed(fig, x=gauge_lon, y=gauge_lat, text=name, justify="LM", offset="0.16c/0c",
                              font="6.5p,Helvetica")
        fig.plot(x=[gnss_lon], y=[gnss_lat], style="a0.17c", fill=gnss_col, pen="0.3p,white")
        utils.text_backed(fig, x=gnss_lon, y=gnss_lat, text=gnss_name, justify="BC", offset="-0.1c/0.16c", font="6.5p,Helvetica")
    fig.text(position="TL", text=label, justify="BL", offset="0c/0.12c", font=panel_letter_font, no_clip=True)
    fig.basemap(region=region_map, projection=proj_map, frame=["lrtb"], map_scale="jBL+w2k+o0.3c/0.55c")      # frame on top, scale bar


def colour_bar(fig, cpt, lim, cb_label, x_cm, y_cm, length_cm):
    """A vertical colour bar from (x_cm, y_cm), its text at the size of the map colour bars."""
    font_pt = colour_bar_font_pt(length_cm)
    with pygmt.config(FONT_ANNOT_PRIMARY=f"{font_pt:.3g}p", FONT_LABEL=f"{font_pt:.3g}p"):
        fig.colorbar(cmap=cpt, position=f"x{x_cm}c/{y_cm}c+w{length_cm}c/{cb_w}c+jBL",
                     frame=f"xa{max(1, round(lim / 2)):g}+l{cb_label}")


def los_maps(fig, field, balance, tag, labels, cb_label, row_label, cols, headings=False, stations_on=(), places_on=()):
    """The two maps of one row (cols: the two fields) from the current origin
    (bottom-left of the first map), one shared colour bar right of the second map,
    the row label left of the first map. Leaves the origin where it started.
    Returns the colour limit."""
    grids = {}
    lim = 0.3
    for col in cols:
        grids[col] = to_grid(field[col], balance)
        finite = grids[col].values[np.isfinite(grids[col].values)]
        lim = max(lim, float(np.nanpercentile(np.abs(finite), 98)))
    lim = float(np.ceil(lim * 2) / 2)                        # rounded up to 0.5, so the bar ends are round numbers
    cpt = utils.write_cpt(os.path.join(cpt_dir, f"los_{tag}.cpt"), "RdBu_r", -lim, lim)
    for k, col in enumerate(cols):
        if k:
            fig.shift_origin(xshift=f"{Wm + gap_map}c")
        map_base(fig, grids[col], cpt, labels[k], stations=labels[k] in stations_on,   # all map labels sit on the quiet vertical maps,
                 places=labels[k] in places_on)                                       # so the fault-parallel maps stay easy to read
        if headings:
            fig.text(position="TC", text=col.capitalize(), justify="BC", offset="0c/0.5c", font=heading_font, no_clip=True)
    colour_bar(fig, cpt, lim, cb_label, Wm + gap_cb, 0.3, map_bar_length)
    fig.shift_origin(xshift=f"-{Wm + gap_map}c")
    fig.text(position="ML", text=row_label, angle=90, justify="BC", offset="-0.3c/0c", font=heading_font, no_clip=True)
    return lim


def write_line_legend(path, entries):
    """A GMT legend spec: one short line symbol and its label per (pen, label) entry."""
    with open(path, "w") as handle:
        for pen, label in entries:
            handle.write(f"S 0.15c - 0.3c - {pen} 0.35c {label}\n")


def los_profile_pairs(fig, value_key, dist_key, lim, labels, ylab, prof_set, cols, headings=False, xlabel=False,
                      model_lines=None):
    """For each field in cols a stacked pair (X above Y) of profiles from prof_set, drawn
    from the current origin (bottom-left of the first column's TOP panel): pixels in grey,
    the model lines (model_lines: a list of (lines by (col, line), colour, label)), the
    per-side running median on top. The first column carries the y annotation, the
    column pair the y label. Leaves the origin at the first column's LOWER panel."""
    cross_km = cross_m / 1e3                                                 # distances are stored in m; drawn in km
    for k, col in enumerate(cols):
        if k:
            fig.shift_origin(xshift=f"{Wp + gap_col}c", yshift=f"{Hp + gap_pair}c")     # next column, back at the top panel
        for j, line in enumerate(line_names):
            profile = prof_set[col][line]
            dist = profile[dist_key]
            values = profile[value_key]
            if j > 0:
                fig.shift_origin(yshift=f"-{Hp + gap_pair}c")
            if j == 1 and xlabel:
                x_frame = "xa1f0.2+lCross-fault distance (km)"                     # km, ticks every 0.2 km
            else:
                x_frame = "xa1f0.2"
            if k == 0:
                sides = "W"
            else:
                sides = "w"
            if j == 1:
                sides += "Srt"
            else:
                sides += "srt"
            fig.basemap(region=[-cross_km, cross_km, -lim, lim], projection=f"X{Wp}c/{Hp}c", frame=[sides, x_frame, f"ya{lim / 2:g}"])
            fig.plot(x=np.asarray(dist) / 1e3, y=values, style="c0.03c", fill=grey, transparency=30)   # the pixels
            if model_lines is not None:
                for lines_by_key, line_col, _ in model_lines:                        # the comparison (homogeneous) first, the model on top
                    if (col, line) not in lines_by_key:
                        continue
                    model_dist, model_values = lines_by_key[(col, line)]
                    for side_sign in (-1, 1):                                        # each side on its own: the model steps at the trace
                        on_side = side_sign * model_dist > 0
                        fig.plot(x=model_dist[on_side] / 1e3, y=model_values[on_side], pen="1.2p," + line_col)
            for median_x, median_values in running_median(dist, values):            # the per-side running median, on top of the models
                fig.plot(x=np.asarray(median_x) / 1e3, y=median_values, pen="1.0p," + median_col)
            fig.plot(x=[0, 0], y=[-lim, lim], pen="0.5p,gray50")
            if model_lines is not None and len(model_lines) > 1 and k == 0 and j == 1:   # legend: bottom centre of the first column's Y-Y' panel,
                legend_file = os.path.join(cpt_dir, "legend_models.txt")                 # on a white box, drawn after the trace line so it sits on top
                legend_entries = []
                for _, line_col, line_label in model_lines:
                    legend_entries.append(("1.2p," + line_col, line_label))
                write_line_legend(legend_file, legend_entries)
                with pygmt.config(FONT_ANNOT_PRIMARY=legend_font):
                    fig.legend(spec=legend_file, position="jBC+o0c/0.08c", box="+gwhite+c0.05c/0.03c")      # entries stacked
            fig.text(position="TL", text=line, font=profile_label_font, offset="0.1c/-0.08c")
            fig.text(position="TR", text=line + "'", font=profile_label_font, offset="-0.1c/-0.08c")
            if j == 0:
                fig.text(x=-cross_km, y=lim, text=labels[k], justify="BL", offset="0c/0.12c", font=panel_letter_font, no_clip=True)
                if headings:
                    fig.text(position="TC", text=col.capitalize(), justify="BC", offset="0c/0.5c", font=heading_font, no_clip=True)
    fig.text(position="MR", text=ylab, angle=90, justify="TC", offset=f"0.2c/{0.5 * (Hp + gap_pair)}c", font=ylabel_font, no_clip=True)
    fig.shift_origin(xshift=f"-{Wp + gap_col}c")                                     # back to the first column (lower panel)


# %% [markdown]
# ## [8] The model products of notebook 08a
#
# The selected 3D model (aligned SFCVM, nugget-only data covariance) draws panels I-L and
# the vermillion lines on G and H; the homogeneous PyLith model of the same inversion draws
# the blue lines on G and H only. Both must be on 07a's current quadtree samples, i.e. 08a
# must have run after 07a (asserted).

# %%
## [8]
for path in (pylith_model, pylith_compare, row07a_file):
    assert os.path.exists(path), f"missing {path}: run 07a_uniform_model and 08a_3d_model first"
model = np.load(pylith_model)
compare = np.load(pylith_compare)
row07a = np.load(row07a_file)
assert str(model["model_tag"]) == "pylith_nuggetcd", f"unexpected model tag {model['model_tag']}"
assert str(compare["model_tag"]) == "pylith_nuggetcd_homog", f"unexpected comparison tag {compare['model_tag']}"
for product, product_file in ((model, pylith_model), (compare, pylith_compare)):
    same_samples = (np.array_equal(product["sample_lon"], row07a["sample_lon"])
                    and np.array_equal(product["sample_lat"], row07a["sample_lat"]))
    assert same_samples, f"{product_file} is not on 07a's current quadtree samples: re-run 08a_3d_model after 07a"
model_tag = "_" + str(model["model_tag"])               # in the Fig 2 file name
print(f"Fig 2 model row from {pylith_model} (tag {model_tag!r}); comparison {pylith_compare}")


def profile_lines(product):
    """A model product's profiles along X-X' and Y-Y', keyed by (field, line): (distance m, mm)."""
    lines_by_key = {}
    for col, key in (("fault-parallel", "fp"), ("vertical", "up")):
        for line in line_names:
            lines_by_key[(col, line)] = (product[f"prof_{line}_{key}_d"], product[f"prof_{line}_{key}_mm"])
    return lines_by_key


model_lines = [(profile_lines(compare), homog_col, "Uniform"),          # no commas: GMT legend labels stop at one
               (profile_lines(model), hetero_col, "3D (SFSVM)")]

# %% [markdown]
# ## [9] Panel K's black line
#
# K's black line (Dani 2026-10-02): the five-event mean at every crossing (04's table, the
# grey dots), smoothed exactly as Fig 1C (Gaussian, profile_smooth_km). It is a display line
# only: the 3D model is fitted to the crossings themselves.

# %%
## [9]
crossings04 = pd.read_csv(os.path.join(paths_out["04_event_detection"], "event_fraction_by_crossing.csv"))
crossings04 = crossings04.sort_values("km").reset_index(drop=True)
events04 = pd.read_csv(os.path.join(paths_out["04_event_detection"], "creep_events.csv"))
cols04 = [f"amp_{onset:.3f}" for onset in events04[events04.segment == "TULEPOND"].onset_lo]
assert len(cols04) == n_mean, f"04 has {len(cols04)} Tule Pond events, 06's mean event has {n_mean}"
for column in cols04:
    assert column in crossings04.columns, f"04's crossing table lacks the Tule Pond event {column}"
crossing_mean04 = crossings04[cols04].mean(axis=1).values
k_line_km = crossings04.km.values
k_line_mm = utils.gaussian_smooth_km(crossing_mean04, crossings04.lat.values, profile_smooth_km)
print(f"K line: {np.isfinite(crossing_mean04).sum()} crossings, Gaussian sigma {profile_smooth_km} km")

# %% [markdown]
# ## [10] The model row (I-L)
#
# I the fault-parallel model and J the residual, each quadtree sample filled over its own
# block (shared colour bar right of J, on the event row's scale); K surface slip along
# strike, the crossings (04) and the model's top-edge slip; L slip with depth on the
# unrolled fault, top 1 km (PyLith tents: slip at the vertices, linear between them).
# Fonts as the map and profile panels: panel letters, I/J titles, axis labels, tick
# labels, the K legend, the Mw text and L's colour bar.

# %%
## [10]
def model_row(fig, product, lim):
    """Row 3 of the figure from the current origin (bottom-left of I). Leaves the origin
    where it started."""
    cpt = utils.write_cpt(os.path.join(cpt_dir, "model_fp.cpt"), "RdBu_r", -lim, lim)
    # the block area on the page is the area each sample averages (equal dots over-weight
    # the many small blocks at the trace); largest first so the small near-fault blocks are on top
    block_area = np.ptp(product["block_lon"], axis=1) * np.ptp(product["block_lat"], axis=1)
    for k, (letter, values, title) in enumerate((("I", product["model_mm"], "Model"),
                                                 ("J", product["residual_mm"], "Residual"))):
        if k:
            fig.shift_origin(xshift=f"{Wm + gap_map}c")
        fig.basemap(region=region_map, projection=proj_map, frame=["lrtb"])
        blocks_file = os.path.join(cpt_dir, f"blocks_{letter}.txt")
        with open(blocks_file, "w") as handle:
            for block in np.argsort(-block_area):
                handle.write(f"> -Z{values[block]:.4f}\n")
                for corner_lon, corner_lat in zip(product["block_lon"][block], product["block_lat"][block]):
                    handle.write(f"{corner_lon:.6f} {corner_lat:.6f}\n")
        fig.plot(data=blocks_file, fill="+z", cmap=cpt, close=True)
        fig.plot(x=prof.ft_lon.values, y=prof.ft_lat.values, pen="0.7p,black")
        for line_lon, line_lat in lines_ll.values():
            fig.plot(x=line_lon, y=line_lat, pen="1.0p,black")
        profile_line_labels(fig)                   # the same X-X' / Y-Y' labels as the data maps
        fig.text(position="TL", text=letter, justify="BL", offset="0c/0.12c", font=panel_letter_font, no_clip=True)
        fig.text(position="TC", text=title, justify="BC", offset="0c/0.12c", font=heading_font, no_clip=True)
        fig.basemap(region=region_map, projection=proj_map, frame=["lrtb"], map_scale="jBL+w2k+o0.3c/0.55c")
    colour_bar(fig, cpt, lim, "mm", Wm + gap_cb, 0.3, map_bar_length)
    fig.shift_origin(xshift=f"-{Wm + gap_map}c")
    fig.text(position="ML", text="Model (fault-parallel)", angle=90, justify="BC", offset="-0.3c/0c",
             font=heading_font, no_clip=True)

    # K: surface slip along strike, the crossings (04) and the model's top edge
    k_indent = 0.7                                       # room for the I/J bar label and the depth label
    Wk = W_model - k_indent
    Hk = 0.38 * Hm
    gap_kl = 0.5
    Hl = Hm - Hk - gap_kl
    km_view = [54.0, 65.0]
    fig.shift_origin(xshift=f"{x_prof + k_indent}c", yshift=f"{Hl + gap_kl}c")
    product_crossings = np.sort(np.asarray(product["crossing_mm"])[np.isfinite(product["crossing_mm"])])
    crossings_04 = np.sort(crossing_mean04[np.isfinite(crossing_mean04)])
    assert product_crossings.size == crossings_04.size and np.allclose(product_crossings, crossings_04, atol=1e-6), \
        "the model product's crossings are not 04's five-event means: re-run 07a / 08a"
    k_view = (k_line_km >= km_view[0]) & (k_line_km <= km_view[1])
    highest_mm = max(np.nanmax(k_line_mm[k_view]), np.nanmax(product["surface_model_mm"]))
    slip_top = float(np.ceil(highest_mm / 2 + 1) * 2)            # the next even mm above the highest value, plus 2
    region_k = km_view + [0.0, slip_top]
    with pygmt.config(FONT_LABEL=ylabel_font):                   # the y label at the size of the profiles' y labels
        fig.basemap(region=region_k, projection=f"X{Wk}c/{Hk}c", frame=["Wsrt", "xa2f1", f"ya{slip_top / 2:g}+lmm"])
    fig.plot(x=product["crossing_km"], y=product["crossing_mm"], style="c0.03c", fill=grey, transparency=30)
    fig.plot(x=k_line_km, y=k_line_mm, pen="1.0p," + median_col)       # NaN in data gaps: GMT breaks the line there
    fig.plot(x=product["surface_km"], y=product["surface_model_mm"], pen="1.0p," + model_col)
    legend_file = os.path.join(cpt_dir, "legend_k.txt")
    write_line_legend(legend_file, [("1.0p," + median_col, "Crossings (smoothed)"),   # no commas: GMT labels stop at one
                                    ("1.0p," + model_col, "3D (SFSVM)")])
    with pygmt.config(FONT_ANNOT_PRIMARY=legend_font):
        fig.legend(spec=legend_file, position="jTL+o0.15c/0.05c", box=False)          # top left, clear of the lines (Dani 2026-10-03)
    fig.text(position="TL", text="K", justify="BL", offset="0c/0.12c", font=panel_letter_font, no_clip=True)
    # no +/- in the label: the bootstrap 2-sigma understates the uncertainty, which is set by the sigma
    # choice and the SFCVM alignment; quote the range in the caption (08a registry
    # 08a_mw_30gpa_range_lo, 08a_mw_30gpa_range_hi)
    fig.text(position="TR", text=f"M@-w@- {float(product['mw_30gpa']):.2f} (30 GPa), {float(product['mw_sfcvm']):.2f} "
                                 f"({float(product['mu_effective_gpa']):.1f} GPa, USGS SFSVM)",
             justify="BR", offset="0c/0.12c", font=annot_font, no_clip=True)

    # L: slip on the unrolled fault, sampled on a regular grid by linear interpolation on the
    # vertices' own Delaunay triangulation (the PyLith triangles overlap slightly once the curved
    # fault is unrolled onto km-along x depth, so they cannot be used directly)
    fig.shift_origin(yshift=f"-{Hl + gap_kl}c")
    z_max = 1.0
    z_axis = np.arange(0.00625, z_max, 0.0125)
    nodes_sz = product["nodes_kmz"]
    s_axis = np.arange(nodes_sz[:, 0].min(), nodes_sz[:, 0].max(), 0.025)
    s_mesh, z_mesh = np.meshgrid(s_axis, z_axis)
    slip_grid = griddata(nodes_sz[:, :2], product["slip_vertex_mm"], (s_mesh, z_mesh), method="linear")
    km_axis = float(product["s0_km_pp"]) + s_axis              # s grows southward, as km from Point Pinole
    in_view = (km_axis >= km_view[0]) & (km_axis <= km_view[1])
    grid_l = xr.DataArray(slip_grid[:, in_view], coords=dict(depth=z_axis, km=km_axis[in_view]), dims=("depth", "km"))
    slip_max = float(np.nanmax(slip_grid))
    cpt_l = utils.write_cpt(os.path.join(cpt_dir, "slip.cpt"), "magma", 0.0, slip_max)    # magma: no blue/green
    region_l = km_view + [0.0, z_max]
    proj_l = f"X{Wk}c/-{Hl}c"
    fig.basemap(region=region_l, projection=proj_l, frame=["WSrt", "xa2f1+lkm from Point Pinole", "ya0.5f0.25"])
    fig.grdimage(grid=grid_l, cmap=cpt_l, nan_transparent=True)
    with pygmt.config(FONT_LABEL=ylabel_font):                   # frame on top, with the y label at the profiles' size
        fig.basemap(region=region_l, projection=proj_l, frame=["WSrt", "xa2f1", "ya0.5f0.25+lDepth (km)"])
    fig.text(position="TL", text="L", justify="BL", offset="0c/0.12c", font=panel_letter_font, no_clip=True)
    font_pt = colour_bar_font_pt(Hl)
    with pygmt.config(FONT_ANNOT_PRIMARY=f"{font_pt:.3g}p", FONT_LABEL=f"{font_pt:.3g}p"):
        fig.colorbar(cmap=cpt_l, position=f"x{Wk + gap_cb}c/0c+w{Hl}c/{cb_w}c+jBL",
                     frame=f"xa{max(1, round(slip_max / 3)):g}+lSlip (mm)")
    fig.shift_origin(xshift=f"-{x_prof + k_indent}c")


# %% [markdown]
# ## [11] The whole figure
#
# Row 1 the inter-event rate (A, B maps; C, D profiles), row 2 the mean event (E, F maps;
# G, H profiles), row 3 the model (I-L). The four maps of rows 1 and 2 show the same pixels
# (a common mask); I and J share E and F's colour scale.

# %%
## [11]
def common_mask(fields):
    """True where every map of every given field (dicts of maps by column) is finite."""
    mask = None
    for field in fields:
        for field_map in field.values():
            if mask is None:
                mask = np.isfinite(field_map)
            else:
                mask = mask & np.isfinite(field_map)
    return mask


def masked(field, mask):
    """The maps of one field with NaN outside the mask."""
    out = {}
    for col, field_map in field.items():
        out[col] = np.where(mask, field_map, np.nan)
    return out


def draw_fig2(rate_maps, event_maps, profile_set, event_label, stem, cols, ylabs, lims):
    """The whole figure for the two columns `cols` (fields of the map and profile
    dictionaries). Saved as <stem>.png/.pdf."""
    fig = pygmt.Figure()
    # row 1: the inter-event rate. A, B the maps; C, D the profiles (arrays, CFW1, the gauge and ZOA1 on B)
    los_maps(fig, rate_maps, True, "rate", ["A", "B"], "mm/yr", "Inter-event rate", cols, headings=True, stations_on=("B",))
    fig.shift_origin(xshift=f"{x_prof}c", yshift=f"{Hp + gap_pair}c")
    los_profile_pairs(fig, "v", "d", lims["rate"], ["C", "D"], ylabs[0], profile_set, cols, headings=True)
    # row 2: the event. E, F the maps (Alameda Creek, Lake Elizabeth, Tule Pond on F); G, H the profiles
    fig.shift_origin(xshift=f"-{x_prof}c", yshift=f"-{gap_row + Hpair}c")
    lim_event = los_maps(fig, event_maps, False, "event", ["E", "F"], "mm", event_label, cols, places_on=("F",))
    fig.shift_origin(xshift=f"{x_prof}c", yshift=f"{Hp + gap_pair}c")
    if cols == tracks:
        lines_here = None                                   # the model lines are fault-parallel and vertical: none on LOS
    else:
        lines_here = model_lines
    los_profile_pairs(fig, "ve", "de", lims["event"], ["G", "H"], ylabs[1], profile_set, cols, xlabel=True,
                      model_lines=lines_here)
    # row 3: the model of notebook 08a, I and J on the same colour scale as E and F
    fig.shift_origin(xshift=f"-{x_prof}c", yshift=f"-{gap_row + Hpair}c")
    model_row(fig, model, lim_event)
    fig.savefig(stem + ".png", dpi=600, transparent=False, crop="+gwhite", anti_alias=True, show=False)   # +gwhite: opaque page
    fig.savefig(stem + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
    print("Saved to:", stem + ".png")
    fig.show()


mask_los = common_mask([rate_los, event_los])           # A, B, E, F show the same pixels
print(f"common pixel mask for the four LOS maps: {mask_los.sum()} pixels ({mask_los.mean():.0%} of the box)")

# %% [markdown]
# ## [12] Fig S: the LOS observations on each track, and the model row
#
# Layout (Dani's sketch, 2026-09-23). Row 1, the inter-event rate: the ascending and
# descending LOS maps sharing one colour bar, then the X-X' and Y-Y' profiles of each track
# (pixels in grey, a per-side running median in black). Row 2, the mean event (the mean, not
# the median, Dani 2026-09-23: it is the field 07a and 08a invert), the same way. Row 3, the
# model row of the 3D model (08a), as in Fig 2. The maps are LOS, positive toward the
# satellite, so the same right-lateral step is blue-over-red on ascending and red-over-blue
# on descending. This LOS version is the supplementary figure; Fig 2 (next section) is the
# fault-parallel and vertical version.

# %%
## [12]
f_fig2_los = os.path.join(fig_dir, f"figS_los_maps_profiles_coh{coh_maps:g}_{n_mean}ev")
draw_fig2(masked(rate_los, mask_los), masked(event_los, mask_los), los_profiles, f"Mean event ({n_mean} events)",
          f_fig2_los, tracks, ("LOS rate (mm/yr)", "LOS (mm)"), lim_prof_los)

# %% [markdown]
# ## [13] Fig 2: the fault-parallel and vertical fields, and the model row
#
# The paper figure (Dani, 2026-09-28). The same layout as the LOS figure above, with the
# two columns the decomposed fault-parallel (positive right-lateral: the east block moving
# south-east) and vertical (positive up) fields from notebook 06, in place of the two
# tracks. The profiles and block amplitudes are 06's, the ones its registry holds
# (`06_XX_*`, `06_YY_*`). G and H carry the two model lines of [8].

# %%
## [13]
fpup_rate = {"fault-parallel": fig2_in["fp_rate"], "vertical": fig2_in["up_rate"]}
fpup_event = {"fault-parallel": fig2_in["fp_event"], "vertical": fig2_in["up_event"]}
fpup_profiles = {"fault-parallel": fp_profiles, "vertical": up_profiles}
for col in fpup_profiles:
    for line, profile in fpup_profiles[col].items():
        west = profile["blocks"]["W"]
        east = profile["blocks"]["E"]
        print(f"[{col}] {line}-{line}': per block W {west['amp']:+.2f} +/- {west['se']:.2f}, "
              f"E {east['amp']:+.2f} +/- {east['se']:.2f} mm")
f_fig2_fp = os.path.join(fig_dir, f"fig2_fault_parallel_vertical_coh{coh_maps:g}_{n_mean}ev{model_tag}")
draw_fig2(fpup_rate, fpup_event, fpup_profiles, f"Mean event ({n_mean} events)", f_fig2_fp,
          list(fpup_rate), ("Rate (mm/yr)", "Displacement (mm)"), lim_prof_fpup)
