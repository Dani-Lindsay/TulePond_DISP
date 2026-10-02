"""The surface constraint of the 3D slip model (Dani 2026-10-02, option B).

Each measured fault crossing (the five-event mean step at the crossing, notebook 04; one every 0.1 km,
each averaging +/- 200 m along strike) is one datum, tied to the modelled slip AT ITS OWN POSITION on the
fault's top edge. With PyLith order-1 tents the slip along the top edge is linear between neighbouring
top-edge vertices, so the forward operator of a crossing at arc length s between vertices j and j+1 is

    d = (1 - w) * slip_j + w * slip_(j+1),   w = (s - s_j) / (s_(j+1) - s_j)

No running median, no resampling to the vertices, nothing invented across data gaps (a vertex with no
crossing nearby is left to the InSAR data and the smoothing prior). Error per crossing = the local robust
scatter of the crossings (1.4826 x MAD within +/- 0.2 km), floored at floor_mm.

Replaces (2026-10-02) 07a's running median (0.4 km window every 0.2 km) interpolated to every vertex, whose
window test lost the edge crossings to float roundoff and dropped half its points.
"""
import numpy as np


def arc_length_on_polyline(px, py, vx, vy):
    """Arc length along the polyline (vx, vy) of the point on it nearest to each (px, py), and that distance.
    Units of the inputs (metres in the mesh CRS)."""
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
    """1.4826 x MAD of the crossings within +/- half_window_km of each crossing, floored at floor_mm.
    The 1e-6 km tolerance keeps the window edges (crossings sit on a 0.1 km grid; float roundoff)."""
    err = np.empty(len(km))
    for i, k in enumerate(km):
        nb = values[np.abs(km - k) <= half_window_km + 1e-6]
        err[i] = max(1.4826 * np.median(np.abs(nb - np.median(nb))), floor_mm) if nb.size >= 3 else 2.0 * floor_mm
    return err


def tent_rows(s, s_vert, top_columns, n_param):
    """G rows: each datum at arc length s is linear between the two bracketing top-edge vertices.
    top_columns[i] = the parameter column of the i-th top-edge vertex (in arc-length order)."""
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
    """The surface constraint: values, lon, lat, errors and G rows for every finite crossing that lies
    within the mesh top edge (ends excluded). trace_lon_of_lat(lat) -> lon on the mapped trace;
    to_mesh_xy(lon, lat) -> (x, y) in the mesh CRS (m); top_xy = top-edge vertices in arc-length order."""
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
