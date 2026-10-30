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
# # 09 Fig 2: the event field and the 3D slip model
#
# ```
# Author: Danielle Lindsay
# Environment: environment.yml. Runs after 06, 07 and 08. Seconds.
# ```
#
# **What this notebook does**
#
# 1. Reads the products of notebooks 06, 07 and 08, and therefore runs after notebook 08: the maps and profiles of
#    notebook 06 (`fig2_inputs.npz`), the preferred slip model of notebook 08 (`selected_model.npz`: the PyLith run
#    `model_config.gf_run`, 3D SFSVM elastic structure, fault dipping 70° to the west), and the homogeneous model and
#    the samples of notebook 07 (`homogeneous_model.npz`, `model_data.npz`: the same fault with analytic half-space
#    Green's functions).
# 2. Draws Fig 2: the inter-event rate and the mean event as fault-parallel and vertical maps and profiles, and a model
#    row from the preferred slip model, with the homogeneous model as the comparison on panel G.
# 3. Draws the same figure with the ascending and descending LOS fields, a supplementary figure.
#
# No number is registered here; each number on the figure is registered by the notebook that computes it. The
# computations here are for display only: the per-side running medians on the profiles, the colour limits and the slip
# grid of panel L.
#
# The model row and lines:
# - G (fault-parallel profiles): each model evaluated at the quadtree samples within 300 m of each profile line
#   ($G\mathbf{s}$ plus the fitted plane, with no PyLith forward run), drawn as lines through the samples sorted by
#   distance and broken at the trace: 3D (vermillion) and homogeneous (blue). The vertical column (H) has no model,
#   because the inversion uses the fault-parallel field only.
# - I: the slip model plus its fitted plane $(1, x, y)$, so that J = E - I at every sample (asserted). Blocks inside the
#   60 m near-trace cut are left blank.
# - K: the cross-fault step 60 to 150 m from the trace at each crossing of notebook 04, for the data (black) and the
#   preferred model (vermillion), measured in the same way on the samples; Mw at 30 GPa and at the harmonic-mean SFSVM
#   rigidity.
# - L: the slip on the fault in the top 1 km (tents: slip at the vertices, linear between them).
#
# **Inputs**: `06_event_field_geometry/fig2_inputs.npz`; `08_3d_model/selected_model.npz`;
# `07_model_data/homogeneous_model.npz` and `model_data.npz`; `04_event_detection/event_fraction_by_crossing.csv`;
# `hayward_fault_profiles.csv`; the alignment-array summary; the GHCN file of the Fremont rain gauge; the GNSS station
# list of the descending frame (`GPS/11115_StaList.txt`, with the reference station ZOA1); the CFW1 position.
#
# **Figures**: `fig2_fault_parallel_vertical_coh<c>_<n>ev_selected3d` (Fig 2) and `figS_los_maps_profiles_coh<c>_<n>ev`
# (the LOS version, supplement). Small GMT helper files go to `outputs/09_fig2/cpt/`.
#
# **What to check**
# - Section 8: both model products are on the samples of notebook 07, and I = model + plane and J = E - I (asserted).
# - Section 9: the printed number of crossings drawn on K (where both sides have samples 60 to 150 m from the trace).
# - Section 13: the per-block amplitudes printed for Fig 2 equal the registry rows `06_XX_*` and `06_YY_*` of
#   notebook 06.

# %% [markdown]
# ## 1. Imports

# %%
import os

import insar_utils as utils
import model_utils
from TulePond_filepaths import (alameda_creek_trace_lonlat, lake_elizabeth_trace_lonlat, common_paths, creepmeters, paths_out, paths_gps, des11115_dic,
                                fig2_map_centre_km, fig2_map_half_km, fig2_map_extra_km, profile_smooth_km)

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import pygmt

# %% [markdown]
# ## 2. Constants
#
# Colours (Okabe-Ito), fonts, the running-median bin of the profiles, the map window, the named alignment arrays and the
# input files.

# %%
notebook = "09_fig2"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
cpt_dir = os.path.join(out_dir, "cpt")                 # GMT helper files: CPTs, legend specs, block polygons
os.makedirs(cpt_dir, exist_ok=True)

# the running median drawn on the profiles
median_bin_m = 150.0            # running-median bin (five pixels), taken on each side of the fault separately
median_col = "black"            # per-side running median
grey = "gray55"                 # pixels
homog_col = "#0072B2"           # 07's homogeneous model (Okabe-Ito blue)
hetero_col = "#D55E00"          # the 3D (SFSVM) model, the preferred model (Okabe-Ito vermillion)
model_col = hetero_col

# fonts: the model row (I-L) uses the same sizes as the map and profile panels
panel_letter_font = "9p,Helvetica-Bold"     # A-L (Science: bold capital panel letters)
heading_font = "9p,Helvetica-Bold"          # column headings, row labels, the I/J titles
ylabel_font = "9p,Helvetica"                # the y-axis labels of the profile and model panels
annot_font = "7p,Helvetica"                 # tick annotations, x-axis labels, map labels, the Mw text
legend_font = "6p,Helvetica"                # the line legends (G and K)
profile_label_font = "7p,Helvetica-Bold"    # X, X', Y, Y' on every map and profile panel (one size throughout)
map_bar_font_pt = 13.0                      # the map colour bars ask GMT for 13 pt (GMT shrinks it on the page, see section 6)

# Fig 2 map window (unrotated, north up): centred on the trace at fig2_map_centre_km from Point Pinole,
# +/- fig2_map_half_km east-west, +/- (fig2_map_half_km + fig2_map_extra_km) north-south (10.4 x 12 km).
# The values live in TulePond_filepaths so Fig 1 (notebook 04) draws the same box.
aa_labelled = ["HGIL", "HRKT"]  # the arrays used in the paper (notebook 11_alignment_arrays); the other arrays of the section are not drawn
aa_label_pos = {"HGIL": ("RM", "-0.15c/0c"),     # HGIL left of its symbol
                "HRKT": ("LM", "0.15c/0.12c")}   # (justify, offset) of each label about its symbol

f_fig2_in = os.path.join(paths_out["06_event_field_geometry"], "fig2_inputs.npz")
aa_file = common_paths["aa_summary"]
selected_file = os.path.join(paths_out["08_3d_model"], "selected_model.npz")          # the preferred model (notebook 08, section 13)
homogeneous_file = os.path.join(paths_out["07_model_data"], "homogeneous_model.npz")  # the homogeneous model of 07
model_data_file = os.path.join(paths_out["07_model_data"], "model_data.npz")         # 07's samples
model_tag = "_selected3d"                                                                 # in the Fig 2 file name
profile_half_width_m = 300.0    # G: model samples within this distance of each profile line (06's pixels: +/- 150 m)

# %% [markdown]
# ## 3. Load the maps and profiles of notebook 06 and the trace
#
# The maps and profiles are computed in notebook 06. Here they are unpacked into dictionaries by field (fault-parallel,
# vertical, ascending, descending) and by line (X, Y). The pixel centres lie on a regular UTM grid (asserted), so a
# pixel is found by its index.

# %%
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
    """Unpack the profiles of one field from notebook 06.

    Parameters
    ----------
    field : str
        "fp", "up", "ascending" or "descending".

    Returns
    -------
    dict
        For each line: the pixel distances and values of the rate ("d", "v") and of the event ("de", "ve"), the line
        ends ("ends"), and the near-fault amplitude and its standard error of each block ("blocks", "W" and "E").
    """
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
    """Binned median on each side of the fault.

    The bin edges start at the trace and run outward, so no bin or drawn segment crosses the fault. Bins with fewer
    than 3 pixels are skipped.

    Parameters
    ----------
    dist : numpy.ndarray
        Cross-fault distance of each pixel (m, east positive).
    values : numpy.ndarray
        Pixel values.
    bin_m : float
        Bin width (m).

    Returns
    -------
    list of tuple
        (bin centres, medians) for the west side, then the east side.
    """
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
# ## 4. The map window and the regular lon/lat grid
#
# The map window is a lon/lat box about the trace point `fig2_map_centre_km` from Point Pinole. The maps of notebook 06
# are resampled to a regular lon/lat grid (nearest 30 m pixel). `to_grid` can balance a map so that the fault is at
# zero, by subtracting the mid-point of the medians of the two sides.

# %%
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
    """Resample one map of notebook 06 to the regular lon/lat grid.

    Parameters
    ----------
    field_map : numpy.ndarray
        Map on the grid of notebook 06.
    balance : bool
        If True, subtract the mid-point of the medians west and east of the trace in the window, so the fault is at
        zero.

    Returns
    -------
    xarray.DataArray
        The map on (lat, lon).
    """
    values = np.full(lon_grid.shape, np.nan)
    values[on_06_grid] = field_map[row_q[on_06_grid], col_q[on_06_grid]]
    if balance:
        values -= 0.5 * (np.nanmedian(values[side_q < 0]) + np.nanmedian(values[side_q > 0]))
    return xr.DataArray(values, coords={"lat": lat_q, "lon": lon_q}, dims=("lat", "lon"))


def gmt_map_size(region, projection):
    """Map size that GMT gives a region and projection (`mapproject -W`).

    Parameters
    ----------
    region : list of float
        [west, east, south, north] (deg).
    projection : str
        GMT projection, e.g. "M4c".

    Returns
    -------
    tuple of float
        Width and height (cm).
    """
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
# ## 5. The sites on the maps: rain gauge, GNSS reference station, named places
#
# Panel B shows the Fremont rain gauge, used for the rainfall statistics over the InSAR period (position from its GHCN
# file), and the GNSS station ZOA1, the reference of the InSAR and GNSS products (`common_paths['ref_station']`; position
# from the GNSS station list, asserted inside the map). ZOA1 and the gauge are about 0.1 km apart, less than one symbol
# width on the map, so ZOA1 (a black star) is drawn on the white diamond of the gauge, with ZOA1 labelled above and
# Fremont to the right. Panel F shows the named places of the vertical event map.

# %%
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
# left (clear of X' and the frame). Tule Pond: the pond itself, labelled bottom left of the fault between X and Y.
map_places = [(("Alameda", "Creek"), -121.980765, 37.572280, 0.3, 0.45),
              (("Lake", "Elizabeth"), -121.962053, 37.549023, 0.4, 0.12),
              (("Tule", "Pond"), -121.975545, 37.558686, -0.35, -0.45)]

# %% [markdown]
# ## 6. Panel sizes and GMT settings
#
# The figure is a double-column Science figure (at most 18.3 cm wide). Each map is as tall as a stacked pair of profile
# panels. GMT 6.6 scales colour-bar text with the square root of the bar length, so `colour_bar_font_pt` asks for a
# larger size on a shorter bar,
#
# $$f = f_{map}\sqrt{\frac{\ell_{map}}{\ell}} ,$$
#
# where $f$ is the font size to ask for (pt), $f_{map} = 13$ pt the size asked on the map colour bars, $\ell_{map}$ the
# length of the map colour bars (cm) and $\ell$ the length of the bar being drawn (cm). Every colour bar then shows its
# text at the size of the map colour bars.

# %%
Wp, Hp = 3.4, 1.85                                  # one profile panel, width and height (cm)
gap_pair, gap_col, gap_row = 0.25, 0.65, 1.25       # within a stacked pair; between the two profile columns (room for the +/-2000 labels); between rows
Hpair = 2 * Hp + gap_pair                           # a stacked pair; the maps are made this tall
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
    """Font size to ask GMT for on a colour bar so its text shows at the size of the map colour bars.

    GMT shrinks colour-bar text by the square root of the bar length, so the size asked on the map bars
    (`map_bar_font_pt`, bars of `map_bar_length` cm) is scaled by sqrt(map_bar_length / bar_length_cm).

    Parameters
    ----------
    bar_length_cm : float
        Length of the colour bar (cm).

    Returns
    -------
    float
        Font size (pt).
    """
    return map_bar_font_pt * np.sqrt(map_bar_length / bar_length_cm)

# %% [markdown]
# ## 7. The drawing helpers: maps, colour bars, profile pairs
#
# Each helper draws into the figure `fig` it is given, from the current origin, and its docstring states where it
# leaves the origin.

# %%
def profile_line_labels(fig):
    """Write the X-X' and Y-Y' labels at the ends of the profile lines.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure, with a map frame active.
    """
    for line, (line_lon, line_lat) in lines_ll.items():
        fig.text(x=line_lon[0], y=line_lat[0], text=line, justify="RM", offset="-0.08c/0c", font=profile_label_font)
        fig.text(x=line_lon[1], y=line_lat[1], text=line + "'", justify="LM", offset="0.08c/0c", font=profile_label_font)


def map_base(fig, grid, cpt, label, stations=True, places=False):
    """Draw one map, north up, in lon/lat, without a colour bar.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure.
    grid : xarray.DataArray
        The field on the regular lon/lat grid.
    cpt : str
        Path of the colour table.
    label : str
        Panel letter.
    stations : bool
        Draw CFW1, the named alignment arrays, the rain gauge and the GNSS station ZOA1.
    places : bool
        Draw the named places with leader lines.
    """
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
    """Draw a vertical colour bar with its text at the size of the map colour bars.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure.
    cpt : str
        Path of the colour table.
    lim : float
        Half-range of the colour scale (sets the annotation interval).
    cb_label : str
        Label of the bar.
    x_cm, y_cm : float
        Bottom-left corner of the bar from the current origin (cm).
    length_cm : float
        Length of the bar (cm).
    """
    font_pt = colour_bar_font_pt(length_cm)
    with pygmt.config(FONT_ANNOT_PRIMARY=f"{font_pt:.3g}p", FONT_LABEL=f"{font_pt:.3g}p"):
        fig.colorbar(cmap=cpt, position=f"x{x_cm}c/{y_cm}c+w{length_cm}c/{cb_w}c+jBL",
                     frame=f"xa{max(1, round(lim / 2)):g}+l{cb_label}")


def los_maps(fig, field, balance, tag, labels, cb_label, row_label, cols, headings=False, stations_on=(), places_on=()):
    """Draw the two maps of one row, a shared colour bar and the row label.

    Drawn from the current origin (bottom-left of the first map); the colour bar is right of the second map and the
    row label left of the first. Leaves the origin where it started.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure.
    field : dict
        Maps by column name.
    balance : bool
        Balance each map so the fault is at zero (`to_grid`).
    tag : str
        Name of the colour-table file.
    labels : list of str
        Panel letters of the two maps.
    cb_label : str
        Label of the colour bar.
    row_label : str
        Label of the row.
    cols : list of str
        The two columns (keys of `field`).
    headings : bool
        Write the column names above the maps.
    stations_on, places_on : tuple of str
        Panel letters that show the stations and the named places.

    Returns
    -------
    float
        Half-range of the colour scale.
    """
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
        map_base(fig, grids[col], cpt, labels[k], stations=labels[k] in stations_on,   # the sites are labelled on B and the places on F only
                 places=labels[k] in places_on)
        if headings:
            fig.text(position="TC", text=col.capitalize(), justify="BC", offset="0c/0.5c", font=heading_font, no_clip=True)
    colour_bar(fig, cpt, lim, cb_label, Wm + gap_cb, 0.3, map_bar_length)
    fig.shift_origin(xshift=f"-{Wm + gap_map}c")
    fig.text(position="ML", text=row_label, angle=90, justify="BC", offset="-0.3c/0c", font=heading_font, no_clip=True)
    return lim


def write_line_legend(path, entries):
    """Write a GMT legend file with one short line symbol and its label per entry.

    Parameters
    ----------
    path : str
        Output file.
    entries : list of tuple
        (pen, label) pairs.
    """
    with open(path, "w") as handle:
        for pen, label in entries:
            handle.write(f"S 0.15c - 0.3c - {pen} 0.35c {label}\n")


def los_profile_pairs(fig, value_key, dist_key, lim, labels, ylab, prof_set, cols, headings=False, xlabel=False,
                      model_lines=None):
    """Draw a stacked pair of profiles (X above Y) for each column.

    Drawn from the current origin (bottom-left of the top panel of the first column): pixels in grey, the model lines,
    and the per-side running median on top. The first column carries the y annotation, the pair of columns the y
    label. Leaves the origin at the lower panel of the first column.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure.
    value_key, dist_key : str
        Keys of the values and distances in each profile ("v", "d" for the rate; "ve", "de" for the event).
    lim : float
        Half-range of the y axis.
    labels : list of str
        Panel letters of the two columns.
    ylab : str
        y-axis label.
    prof_set : dict
        Profiles by column and line (`unpack`).
    cols : list of str
        The two columns.
    headings : bool
        Write the column names above the top panels.
    xlabel : bool
        Label the x axis of the lower panels.
    model_lines : list of tuple, optional
        (lines by (column, line), colour, label) for each model.
    """
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
                    if (col, line) not in lines_by_key:                              # the vertical column has no model
                        continue
                    model_dist, model_values = lines_by_key[(col, line)]
                    for side_sign in (-1, 1):                                        # each side on its own: no line across the trace
                        on_side = side_sign * model_dist > 0
                        order = np.argsort(model_dist[on_side])
                        fig.plot(x=model_dist[on_side][order] / 1e3, y=model_values[on_side][order], pen="1.0p," + line_col)   # model at the samples
            for median_x, median_values in running_median(dist, values):            # the per-side running median, on top of the models
                fig.plot(x=np.asarray(median_x) / 1e3, y=median_values, pen="1.0p," + median_col)
            fig.plot(x=[0, 0], y=[-lim, lim], pen="0.5p,gray50")
            if model_lines is not None and len(model_lines) > 1 and k == 0 and j == 1:   # legend: bottom centre of the first column's Y-Y' panel,
                legend_file = os.path.join(cpt_dir, "legend_models.txt")                 # on a white box, drawn after the trace line so it sits on top
                with open(legend_file, "w") as handle:                                  # one line per model (no commas: GMT labels stop at one)
                    for _, line_col, line_label in model_lines:
                        handle.write(f"S 0.15c - 0.3c - 1.0p,{line_col} 0.35c {line_label}\n")
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
# ## 8. The model products of notebooks 08 and 07
#
# The preferred 3D model (notebook 08) gives panels I to L and the vermillion line on G. The homogeneous model of
# notebook 07 (the same fault, samples, weighting, plane, prior and corner rule, with analytic Green's functions on the
# same tents) gives the blue line on G. Panel I is the slip model plus its fitted plane,
# $G\mathbf{s} + B\mathbf{a}$, so that J = E - I at every sample. CSI removes the fitted plane from the data, so
# notebook 08 stores the plane separately.
#
# Check: both models are on the samples of notebook 07, and J = E - I (both asserted).

# %%
for path in (selected_file, homogeneous_file, model_data_file):
    assert os.path.exists(path), f"missing {path}: run 07_model_data and 08_3d_model first"
model = np.load(selected_file)
compare = np.load(homogeneous_file)
model_data = np.load(model_data_file)
for product, product_file in ((model, selected_file), (compare, homogeneous_file)):
    assert np.array_equal(product["sample_lon"], model_data["sample_lon"]) and np.array_equal(product["sample_lat"], model_data["sample_lat"]), \
        f"{product_file} is not on the current samples of 07: re-run 07 and 08"
assert np.array_equal(model["keep"], compare["keep"]), "the two models have different fitted samples"
model_with_plane_mm = model["model_mm"] + model["plane_mm"]
assert np.allclose(model_data["data_mm"] - model_with_plane_mm, model["residual_mm"], atol=1e-6), "E - I != J at the samples"
print(f"panel I: slip model + fitted plane ({float(model['plane_offset_mm']):+.2f} mm, {float(model['plane_dx_mm_per_km']):+.3f} mm/km x, "
      f"{float(model['plane_dy_mm_per_km']):+.3f} mm/km y); check passed: J = E - I at every sample")
print(f"panel K: Mw {float(model['Mw_30gpa']):.2f} (30 GPa), {float(model['Mw_sfcvm']):.2f} (harmonic SFSVM, "
      f"{float(model['mu_effective_gpa']):.1f} GPa effective)")


def profile_dots(product):
    """A model's prediction of the data at the samples near each profile line (fault-parallel only).

    Parameters
    ----------
    product : numpy.lib.npyio.NpzFile
        A model product of notebook 07 or 08 (`model_mm`, `plane_mm`).

    Returns
    -------
    dict
        Keyed by ("fault-parallel", line): (cross-fault distance in m, measured as in the profiles of notebook 06 from
        the centre of the line on the trace with east positive; the prediction G x slip + plane in mm) of the fitted
        samples within `profile_half_width_m` of the line.
    """
    dots = {}
    prediction = product["model_mm"] + product["plane_mm"]
    for line in line_names:
        ends = fp_profiles[line]["ends"]
        on_line, along_m = model_utils.profile_samples(model_data["sample_lon"], model_data["sample_lat"], ends, epsg,
                                                       profile_half_width_m)
        on_line &= model["keep"]
        half_length_m = 0.5 * np.hypot(ends[2] - ends[0], ends[3] - ends[1])
        dots[("fault-parallel", line)] = (along_m[on_line] - half_length_m, prediction[on_line])
        print(f"G {line}-{line}': {on_line.sum()} samples within {profile_half_width_m:.0f} m of the line")
    return dots


model_lines = [(profile_dots(compare), homog_col, "Homogeneous"),     # no commas: GMT legend labels stop at one
               (profile_dots(model), hetero_col, "3D (SFSVM)")]

# %% [markdown]
# ## 9. Panel K: the 60 to 150 m steps of the data and the model
#
# The steps are computed in notebook 08, section 6 (`model_utils.crossing_steps_like`). At each crossing $c$ of
# notebook 04, for the data and for the model,
#
# $$\Delta u_c = \operatorname{median}_{i \in E_c} u_i - \operatorname{median}_{i \in W_c} u_i ,$$
#
# where $u_i$ is the plane-removed data or the model at sample $i$ (mm), and $E_c$ and $W_c$ are the samples 60 to
# 150 m east and west of the trace within 200 m along strike of the crossing. These are samples outside the 60 m
# near-trace cut, so the model fits them. The steps are drawn without smoothing, and a crossing is drawn only where
# both sides have at least 3 samples.

# %%
crossings04 = pd.read_csv(os.path.join(paths_out["04_event_detection"], "event_fraction_by_crossing.csv"))
assert np.array_equal(np.asarray(model["crossing_km"], float), crossings04.km.values.astype(float)), "08's crossings are not 04's"
assert float(model["step_inner_km"]) == 0.06 and float(model["step_outer_km"]) == 0.15, "panel K steps must be 60-150 m"
k_line_km = np.asarray(model["crossing_km"], float)
k_data_mm = np.asarray(model["data_step_mm"], float)
k_model_mm = np.asarray(model["model_step_mm"], float)
finite_k = np.isfinite(k_data_mm) & np.isfinite(k_model_mm)
print(f"K: {finite_k.sum()} of {k_line_km.size} crossings have a 60-150 m step on both sides; band step data - model "
      f"{float(model['band_step_mean_mm']):+.2f} mm (mean), {float(model['band_step_rms_mm']):.2f} mm (RMS)")

# %% [markdown]
# ## 10. The model row (I to L)
#
# - I: the fault-parallel model (the slip model plus the fitted plane, so J = E - I).
# - J: the residual.
# - K: the 60 to 150 m cross-fault step along strike, data and model.
# - L: the slip with depth on the unrolled fault, top 1 km (PyLith tents: slip at the vertices, linear between them),
#   sampled on a regular grid by linear interpolation on the Delaunay triangulation of the impulse vertices.
#
# On I and J each quadtree sample fills its own block, with a shared colour bar right of J on the colour scale of the
# event row. Panel letters, I and J titles, axis labels, tick labels, the K legend, the Mw text and the colour bar of L
# use the font sizes of the map and profile panels.

# %%
def model_row(fig, product, lim, model_mm):
    """Draw row 3 of the figure (panels I to L).

    Drawn from the current origin (bottom-left of I). Leaves the origin where it started.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure.
    product : numpy.lib.npyio.NpzFile
        The preferred model of notebook 08 (`selected_model.npz`).
    lim : float
        Half-range of the colour scale of I and J (the scale of the event row).
    model_mm : numpy.ndarray
        Values of panel I at the samples: the slip model plus the fitted plane (mm, section 8).
    """
    cpt = utils.write_cpt(os.path.join(cpt_dir, "model_fp.cpt"), "RdBu_r", -lim, lim)
    # the block area on the page is the area each sample averages (equal dots over-weight
    # the many small blocks at the trace); largest first so the small near-fault blocks are on top
    block_area = np.ptp(product["block_lon"], axis=1) * np.ptp(product["block_lat"], axis=1)
    for k, (letter, values, title) in enumerate((("I", model_mm, "Model"),
                                                 ("J", product["residual_mm"], "Residual"))):
        if k:
            fig.shift_origin(xshift=f"{Wm + gap_map}c")
        fig.basemap(region=region_map, projection=proj_map, frame=["lrtb"])
        blocks_file = os.path.join(cpt_dir, f"blocks_{letter}.txt")
        with open(blocks_file, "w") as handle:
            for block in np.argsort(-block_area):
                if "keep" in product.files and not bool(product["keep"][block]):
                    continue                                   # inside the near-trace cut: not fitted, left blank
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

    # K: the cross-fault step 60-150 m from the trace at the crossings of 04, data and preferred model measured the same
    # way on the samples (section 9); unsmoothed
    k_indent = 0.7                                       # room for the I/J bar label and the depth label
    Wk = W_model - k_indent
    Hk = 0.38 * Hm
    gap_kl = 0.75                                        # room under K for the Alameda Creek / Lake Elizabeth bars
    Hl = Hm - Hk - gap_kl
    km_view = [54.0, 65.0]
    fig.shift_origin(xshift=f"{x_prof + k_indent}c", yshift=f"{Hl + gap_kl}c")
    k_view = (k_line_km >= km_view[0]) & (k_line_km <= km_view[1]) & finite_k
    highest_mm = max(np.nanmax(k_data_mm[k_view]), np.nanmax(k_model_mm[k_view]))
    lowest_mm = min(0.0, np.nanmin(k_data_mm[k_view]), np.nanmin(k_model_mm[k_view]))
    slip_top = float(np.ceil(highest_mm / 2 + 1) * 2)            # the next even mm above the highest value, plus 2
    slip_bottom = float(np.floor(lowest_mm / 2) * 2)
    region_k = km_view + [slip_bottom, slip_top]
    with pygmt.config(FONT_LABEL=ylabel_font):                   # the y label at the size of the profiles' y labels
        fig.basemap(region=region_k, projection=f"X{Wk}c/{Hk}c", frame=["Wsrt", "xa2f1", f"ya{slip_top / 2:g}+lmm"])
    fig.plot(x=km_view, y=[0, 0], pen="0.4p,gray60")
    k_plot = np.where(finite_k, 1.0, np.nan)                     # NaN where a side has no samples: GMT breaks the lines there
    fig.plot(x=k_line_km[finite_k], y=k_data_mm[finite_k], style="c0.05c", fill=median_col)
    fig.plot(x=k_line_km, y=k_model_mm * k_plot, pen="1.0p," + model_col)
    legend_file = os.path.join(cpt_dir, "legend_k.txt")
    with open(legend_file, "w") as handle:                       # no commas: GMT labels stop at one
        handle.write(f"S 0.15c c 0.08c {median_col} - 0.35c InSAR (60-150 m)\n")
        handle.write(f"S 0.15c - 0.3c - 1.0p,{model_col} 0.35c 3D (SFSVM)\n")
    with pygmt.config(FONT_ANNOT_PRIMARY=legend_font):
        fig.legend(spec=legend_file, position="jTL+o0.1c/0.05c", box=False)   # top left, at the north end
    fig.text(position="TL", text="K", justify="BL", offset="0c/0.12c", font=panel_letter_font, no_clip=True)
    # Alameda Creek and Lake Elizabeth on the trace (TulePond_filepaths): sky-blue bars under the x axis of K, named below
    # them, in the gap between K and L
    mm_per_cm = (slip_top - slip_bottom) / Hk
    landmark_km = {}
    for name, ends in (("Alameda Creek", alameda_creek_trace_lonlat), ("Lake Elizabeth", lake_elizabeth_trace_lonlat)):
        ends_km = km_of(np.array([lat for _, lat in ends]))
        nearest_m = [111e3 * np.min(np.hypot((prof.ft_lon - lon) * np.cos(np.radians(lat)), prof.ft_lat - lat)) for lon, lat in ends]
        assert max(nearest_m) < 100, f"{name}: a point is {max(nearest_m):.0f} m from the fault-profile trace"
        landmark_km[name] = ends_km
        bar_cm = abs(ends_km[1] - ends_km[0]) * Wk / (km_view[1] - km_view[0])
        fig.plot(data=np.array([[float(np.mean(ends_km)), slip_bottom - 0.10 * mm_per_cm, bar_cm, 0.07]]), style="r", fill="#56B4E9",
                 no_clip=True)                               # a rectangle symbol: GMT clips lines outside the frame, not symbols
        fig.text(x=float(np.mean(ends_km)), y=slip_bottom - 0.22 * mm_per_cm, text=name, font=annot_font, justify="CT", no_clip=True)
    print("K landmarks (km from Point Pinole): " + "; ".join(f"{n} {k[0]:.1f}-{k[1]:.1f}" for n, k in landmark_km.items()))
    # the uncertainty (Monte Carlo, sensitivity range) is given in the caption (08 / 10 registry), not on the panel
    fig.text(position="TR", text=f"M@-w@- {float(product['Mw_30gpa']):.2f} (30 GPa), {float(product['Mw_sfcvm']):.2f} "
                                 f"({float(product['mu_effective_gpa']):.1f} GPa, USGS SFSVM)",
             justify="BR", offset="0c/0.12c", font=annot_font, no_clip=True)

    # L: slip on the fault against km from Point Pinole and depth, sampled on a regular grid by linear interpolation on the
    # impulse vertices' own Delaunay triangulation (the PyLith triangles overlap slightly once the curved fault is unrolled)
    fig.shift_origin(yshift=f"-{Hl + gap_kl}c")
    z_max = 1.0
    z_axis = np.arange(0.00625, z_max, 0.0125)
    km_axis = np.arange(km_view[0], km_view[1] + 1e-9, 0.025)
    km_mesh, z_mesh = np.meshgrid(km_axis, z_axis)
    slip_grid = griddata(np.c_[product["imp_km"], product["imp_depth_km"]], product["slip_mm"], (km_mesh, z_mesh), method="linear")
    grid_l = xr.DataArray(slip_grid, coords=dict(depth=z_axis, km=km_axis), dims=("depth", "km"))
    slip_max = float(np.nanmax(slip_grid))
    cpt_l = utils.write_cpt(os.path.join(cpt_dir, "slip.cpt"), "magma", 0.0, slip_max)    # magma: no blue/green
    region_l = km_view + [0.0, z_max]
    proj_l = f"X{Wk}c/-{Hl}c"
    fig.basemap(region=region_l, projection=proj_l, frame=["WSrt", "xa2f1+lkm from Point Pinole", "ya0.5f0.25"])
    fig.grdimage(grid=grid_l, cmap=cpt_l, nan_transparent=True)
    with pygmt.config(FONT_LABEL=ylabel_font):                   # frame on top, with the y label at the profiles' size
        fig.basemap(region=region_l, projection=proj_l, frame=["WSrt", "xa2f1", "ya0.5f0.25+lDepth (km)"])
    fig.text(position="TL", text="L", justify="BL", offset="0c/0.12c", font=panel_letter_font, no_clip=True)
    # orientation in white on the dark slip panel: km from Point Pinole increases southward
    fig.text(position="BL", text="North", justify="BL", offset="0.12c/0.12c", font=f"{annot_font.split(',')[0]},Helvetica,white")
    fig.text(position="BR", text="South", justify="BR", offset="-0.12c/0.12c", font=f"{annot_font.split(',')[0]},Helvetica,white")
    font_pt = colour_bar_font_pt(Hl)
    with pygmt.config(FONT_ANNOT_PRIMARY=f"{font_pt:.3g}p", FONT_LABEL=f"{font_pt:.3g}p"):
        fig.colorbar(cmap=cpt_l, position=f"x{Wk + gap_cb}c/0c+w{Hl}c/{cb_w}c+jBL",
                     frame=f"xa{max(1, round(slip_max / 3)):g}+lSlip (mm)")
    fig.shift_origin(xshift=f"-{x_prof + k_indent}c")


# %% [markdown]
# ## 11. The whole figure
#
# Row 1 is the inter-event rate (maps A and B, profiles C and D), row 2 the mean event (maps E and F, profiles G and H),
# and row 3 the model (I to L). The four maps of rows 1 and 2 show the same pixels (a common mask), and I and J share
# the colour scale of E and F.

# %%
def common_mask(fields):
    """Mask of the pixels that are finite in every map.

    Parameters
    ----------
    fields : list of dict
        Fields, each a dict of maps by column.

    Returns
    -------
    numpy.ndarray of bool
        True where every map of every field is finite.
    """
    mask = None
    for field in fields:
        for field_map in field.values():
            if mask is None:
                mask = np.isfinite(field_map)
            else:
                mask = mask & np.isfinite(field_map)
    return mask


def masked(field, mask):
    """Set the maps of one field to NaN outside a mask.

    Parameters
    ----------
    field : dict
        Maps by column.
    mask : numpy.ndarray of bool
        Pixels to keep.

    Returns
    -------
    dict
        The masked maps by column.
    """
    out = {}
    for col, field_map in field.items():
        out[col] = np.where(mask, field_map, np.nan)
    return out


def draw_fig2(rate_maps, event_maps, profile_set, event_label, stem, cols, ylabs, lims):
    """Draw the whole figure for two columns and save it as <stem>.png and <stem>.pdf.

    Parameters
    ----------
    rate_maps, event_maps : dict
        Maps of the inter-event rate and of the mean event, by column.
    profile_set : dict
        Profiles by column and line (`unpack`).
    event_label : str
        Label of the event row.
    stem : str
        Output path without extension.
    cols : list of str
        The two columns.
    ylabs : tuple of str
        y-axis labels of the rate and event profiles.
    lims : dict
        Half-ranges of the profile axes ("rate", "event").
    """
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
        lines_here = None                                   # the model lines are fault-parallel: none on the LOS figure
    else:
        lines_here = model_lines
    los_profile_pairs(fig, "ve", "de", lims["event"], ["G", "H"], ylabs[1], profile_set, cols, xlabel=True,
                      model_lines=lines_here)
    # row 3: the model of notebook 08, I and J on the same colour scale as E and F
    fig.shift_origin(xshift=f"-{x_prof}c", yshift=f"-{gap_row + Hpair}c")
    model_row(fig, model, lim_event, model_with_plane_mm)
    fig.savefig(stem + ".png", dpi=600, transparent=False, crop="+gwhite", anti_alias=True, show=False)   # +gwhite: opaque page
    fig.savefig(stem + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
    print("Saved to:", stem + ".png")
    fig.show()


mask_los = common_mask([rate_los, event_los])           # A, B, E, F show the same pixels
print(f"common pixel mask for the four LOS maps: {mask_los.sum()} pixels ({mask_los.mean():.0%} of the box)")

# %% [markdown]
# ## 12. Fig S: the LOS observations on each track, and the model row
#
# Row 1 is the inter-event rate: the ascending and descending LOS maps with one shared colour bar, then the X-X' and
# Y-Y' profiles of each track (pixels in grey, a per-side running median in black). Row 2 is the mean event, the field
# inverted in notebooks 07 and 08, drawn in the same way. Row 3 is the model row of the 3D model (notebook 08), as in
# Fig 2. The maps are LOS, positive toward the satellite, so a right-lateral step has opposite colours on the ascending
# and descending tracks. This LOS version is a supplementary figure; Fig 2 (section 13) is the fault-parallel and
# vertical version.

# %%
f_fig2_los = os.path.join(fig_dir, f"figS_los_maps_profiles_coh{coh_maps:g}_{n_mean}ev")
draw_fig2(masked(rate_los, mask_los), masked(event_los, mask_los), los_profiles, f"Mean event ({n_mean} events)",
          f_fig2_los, tracks, ("LOS rate (mm/yr)", "LOS (mm)"), lim_prof_los)

# %% [markdown]
# ## 13. Fig 2: the fault-parallel and vertical fields, and the model row
#
# Fig 2 has the layout of the LOS figure of section 12, with the two tracks replaced by the fault-parallel field
# (positive right-lateral, the east block moving south-east) and the vertical field (positive up) of notebook 06. The
# profiles and block amplitudes are those of notebook 06, as in its registry (`06_XX_*`, `06_YY_*`). Panel G has the two
# model lines of section 8.

# %%
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
