#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

Tie the InSAR velocity fields to GNSS and validate the tie:

1) Sample the InSAR velocity within 150 m of every screened GNSS station and
   project that station's ENU velocity into LOS with the incidence/azimuth of
   its own pixels (not a frame-mean geometry)
2) Reference the GNSS LOS to ZOA1, the station the InSAR is referenced at, so
   GNSS and InSAR share a datum and the fit targets ramp + scatter only
3) Per track group (frames sharing a track share look geometry): fit a shared
   quadratic surface + one constant per frame
4) Apply the ramp only if it passes two gates: leave-one-out RMSE improves by
   >= 10%, and it does not add an inter-frame offset beyond what the frame
   overlap allows
5) Write velocity_NA.h5 per frame (zero at ZOA1) and mosaic per orbit:
   mosaic_velfield/<orbit>_velocity_NA.h5 is the decomposition's input and the
   archived (Zenodo) product

The tie is primarily a datum; the GNSS is used here mainly for validation.
The ramp model, per track group,

    r(x, y) = a.x^2 + b.y^2 + c.xy + d.x + e.y  +  k_frame

with (x, y) centred geographic coordinates scaled to km. There is no global
constant: with one, the design matrix is structurally rank-deficient (the
frame indicators sum to the constant column) and lstsq would return a
minimum-norm answer without warning; `check_full_rank` enforces this.

The per-frame constants are a diagnostic, not a correction: with GNSS
referenced to ZOA1 they are each frame's residual offset after referencing
and are expected to be ~0. The inter-frame offset (difference of constants
within a track) is preserved in the output, not flattened: the correction is
re-zeroed at ZOA1 with one scalar per track group (`corr_ref`).

Output:  <frame>/mintpy/velocity_NA.h5, mosaic_velfield/<orbit>_velocity_NA.h5,
         <data_dir>/GPS/gps_insar_validation.csv (per-station residuals)

Run in the conda environment.
Source: written for this project; GNSS velocities from run00 (UNR, NA frame).
"""
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import pyproj

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from TulePond_filepaths import data_dir, common_paths, frames as FRAME_PATHS, paths_gps, mosaic_dir
import insar_utils as utils
from run04_velocity import frame_grid, merge_orbit

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
SAMPLE_RADIUS_M = 150.0   # InSAR averaging disc at each station (~78 30-m pixels)
MIN_PIXELS = 10           # a station backed by fewer valid pixels is dropped
VU_MAX_SUBSIDENCE = -5.0  # drop stations subsiding faster than this (mm/yr), so
#                           localized subsidence is not absorbed into the ramp
RAMP_MIN_IMPROVEMENT = 0.10   # LOO gate: fractional RMSE improvement required
OVERLAP_TOL = 0.5             # overlap gate: max inter-frame offset the ramp may add
REF_STATION = common_paths['ref_station']    # ZOA1
ref_frame = 'NA'
MM = 1000.0               # velocities are stored m/yr, fitted in mm/yr

QUAD_TERMS = ("x2", "y2", "xy", "x", "y")


# ---------------------------------------------------------------------------
# Grid + sampling helpers
# ---------------------------------------------------------------------------
def read_grid(path):
    with h5py.File(path, "r") as f:
        a = f.attrs
        return dict(epsg=int(a["EPSG"]),
                    x0=float(a["X_FIRST"]), y0=float(a["Y_FIRST"]),
                    dx=float(a["X_STEP"]), dy=float(a["Y_STEP"]),
                    width=int(a["WIDTH"]), length=int(a["LENGTH"]),
                    orbit=str(a["ORBIT_DIRECTION"]))


def pixel_lonlat(grid):
    """(lon, lat) of every pixel centre, shape (length, width)."""
    x = grid["x0"] + (np.arange(grid["width"]) + 0.5) * grid["dx"]
    y = grid["y0"] + (np.arange(grid["length"]) + 0.5) * grid["dy"]
    xx, yy = np.meshgrid(x, y)
    tf = pyproj.Transformer.from_crs(grid["epsg"], 4326, always_xy=True)
    return tf.transform(xx, yy)


def station_rowcol(lons, lats, grid):
    """Row/col of each station on this frame's grid; -1 where outside."""
    tf = pyproj.Transformer.from_crs(4326, grid["epsg"], always_xy=True)
    sx, sy = tf.transform(np.asarray(lons), np.asarray(lats))
    col = np.floor((sx - grid["x0"]) / grid["dx"]).astype(int)
    row = np.floor((sy - grid["y0"]) / grid["dy"]).astype(int)
    outside = (col < 0) | (col >= grid["width"]) | (row < 0) | (row >= grid["length"])
    row[outside] = -1
    col[outside] = -1
    return row, col


def disc(row, col, radius_px, shape):
    """Index arrays for a disc of pixels, clipped to the grid."""
    r = int(np.ceil(radius_px))
    rr, cc = np.mgrid[row - r:row + r + 1, col - r:col + r + 1]
    keep = ((rr - row) ** 2 + (cc - col) ** 2 <= radius_px ** 2) \
        & (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    return rr[keep], cc[keep]


def track_of(frame_id):
    """Track of a frame, from its own burst IDs (not a hardcoded table)."""
    from opera_utils import get_burst_ids_for_frame
    tracks = sorted({b.split("_")[0] for b in get_burst_ids_for_frame(int(frame_id))})
    return tracks[0]


def sample_frame(frame_id, paths, gnss):
    """GNSS LOS, InSAR LOS and their residual at every station this frame covers.

    The GNSS is projected to LOS with the incidence and azimuth of the
    station's own pixels, not a frame mean.
    """
    grid = read_grid(paths['velocity_masked'])
    with h5py.File(paths['velocity_masked'], "r") as f:
        vel = f["velocity"][:] * MM              # mm/yr
        vel_std = f["velocityStd"][:] * MM
    with h5py.File(paths['geometry'], "r") as f:
        inc = f["incidenceAngle"][:]
        azi = f["azimuthAngle"][:]

    radius_px = SAMPLE_RADIUS_M / abs(grid["dx"])
    row, col = station_rowcol(gnss["Lon"], gnss["Lat"], grid)

    rows = []
    for i, sta in enumerate(gnss.itertuples()):
        if row[i] < 0:
            continue                              # station outside this frame
        rr, cc = disc(row[i], col[i], radius_px, vel.shape)
        v = vel[rr, cc]
        ok = np.isfinite(v)
        if ok.sum() < MIN_PIXELS:
            continue                              # too little valid InSAR
        L = utils.los_unit_vector(np.nanmean(inc[rr, cc][ok]), np.nanmean(azi[rr, cc][ok]))
        if not np.all(np.isfinite(L)):
            continue    # velocity valid but geometry missing here: a single
            #             NaN gnss_los would make the whole group's fit/RMS NaN
        rows.append(dict(
            StaID=sta.StaID, frame=frame_id, orbit=grid["orbit"].lower(),
            Lon=sta.Lon, Lat=sta.Lat, n_px=int(ok.sum()),
            gnss_los=float(L[0] * sta.Ve + L[1] * sta.Vn + L[2] * sta.Vz),
            insar_los=float(np.nanmean(v[ok])),
            insar_std=float(np.nanmean(vel_std[rr, cc][ok])),
            Vz=float(sta.Vz)))
    return pd.DataFrame(rows), grid, vel


def ref_lonlat(vel_path):
    """(lon, lat) of the run04 InSAR reference pixel. MintPy's REF_LAT/REF_LON
    attrs hold UTM northing/easting; converted to geographic."""
    with h5py.File(vel_path, "r") as f:
        a = f.attrs
        northing, easting, epsg = float(a["REF_LAT"]), float(a["REF_LON"]), int(a["EPSG"])
    return pyproj.Transformer.from_crs(epsg, 4326, always_xy=True).transform(
        easting, northing)


# ---------------------------------------------------------------------------
# The ramp fit
# ---------------------------------------------------------------------------
def quad_columns(lon, lat, lon0, lat0):
    """Quadratic design columns in centred geographic coordinates scaled to km.
    A quadratic is closed under affine changes of variables, so the choice
    only affects conditioning, not the fitted surface."""
    x = (np.asarray(lon) - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (np.asarray(lat) - lat0) * 110.57
    return np.column_stack([x * x, y * y, x * y, x, y])


def check_full_rank(G, label):
    """Abort on a structurally rank-deficient design matrix (columns dependent
    however many stations are added, i.e. a parametrisation error). lstsq
    returns a minimum-norm answer with no warning otherwise."""
    n_obs, n_par = G.shape
    if np.linalg.matrix_rank(G) != n_par:
        raise SystemExit(f"ABORT: {label} design matrix rank-deficient with "
                         f"{n_obs} stations -- structural; see module docstring.")


def _design(obs, group, center):
    """Shared quadratic + one indicator per frame, no global constant
    (the indicators already span it; see check_full_rank)."""
    Q = quad_columns(obs["Lon"], obs["Lat"], *center)
    ind = np.column_stack([(obs["frame"] == f).to_numpy(float) for f in group])
    return np.hstack([Q, ind])


def loo_residuals(obs, group, center, y):
    """Leave-one-out out-of-sample residual at each station.

    In-sample residuals understate the misfit (the fit absorbs its own
    targets); LOO refits without each station and predicts it.
    """
    n, p = len(obs), _design(obs, group, center).shape[1]
    out = np.full(n, np.nan)
    for k in range(n):
        keep = np.ones(n, bool); keep[k] = False
        Gk = _design(obs.iloc[keep], group, center)
        if np.linalg.matrix_rank(Gk) != p:
            continue
        coef, *_ = np.linalg.lstsq(Gk, y[keep], rcond=None)
        gk = _design(obs.iloc[[k]], group, center)
        out[k] = y[k] - float((gk @ coef)[0])
    return out


def fit_group(track, group, obs):
    """Solve one track group: shared quadratic + one constant per frame, with
    the GNSS referenced to ZOA1 so the constants are expected near 0.

    Returns (quad, consts, gref, center, obs); gref is ZOA1's absolute GNSS
    LOS in this group's look geometry (the datum removed before fitting).
    """
    center = (obs["Lon"].mean(), obs["Lat"].mean())

    # 1) Reference the GNSS to ZOA1: the fit targets ramp + scatter only
    gref = float(obs.loc[obs["StaID"] == REF_STATION, "gnss_los"].mean())
    y = ((obs["gnss_los"] - gref) - obs["insar_los"]).to_numpy()   # residual_before

    # 2) Solve the shared quadratic + per-frame constants
    G = _design(obs, group, center)
    print(f"\n=== track {track}: frames {', '.join(group)} ===")
    print(f"    {len(obs)} stations, {G.shape[1]} parameters, "
          f"ref {REF_STATION} (gref {gref:+.2f} mm/yr)")
    check_full_rank(G, f"track {track}")
    p, *_ = np.linalg.lstsq(G, y, rcond=None)
    quad, consts = p[:5], dict(zip(group, p[5:]))
    after = y - G @ p
    after_loo = loo_residuals(obs, group, center, y)

    # 3) Report. Constants are expected near 0 (residual per-frame offset after the
    #    shared reference); far from zero means the run04 referencing did not take
    for name, c in zip(QUAD_TERMS, quad):
        print(f"    {name:>3} {c:+9.4f} mm/yr per km^n")
    for f, c in consts.items():
        flag = "   <-- non-zero; check the run04 referencing" if abs(c) >= 1.0 else ""
        print(f"    constant {f}: {c:+7.3f} mm/yr{flag}")
    print(f"    residual rms: before {np.sqrt(np.mean(y ** 2)):.3f}  "
          f"after {np.sqrt(np.mean(after ** 2)):.3f}  "
          f"after(LOO) {np.sqrt(np.nanmean(after_loo ** 2)):.3f} mm/yr")

    obs = obs.assign(gref=gref, residual_before=y,
                     residual_after=after, residual_after_loo=after_loo,
                     insar_los_deramp=obs["gnss_los"] - after)
    return quad, consts, gref, center, obs


def write_tie_report(all_obs, group_stats, n_pool, n_subsiding, outdir=None):
    """Write the tie statistics (settings, station counts per orbit and frame,
    per-track fit coefficients and constants, residual RMSE before / after /
    leave-one-out, mm/yr) to common_paths['validation_dir']/gps_tie_stats.txt.
    Returns the path."""
    obs = pd.concat(all_obs, ignore_index=True)
    L = []
    def w(x=""):
        L.append(x)
    w("=" * 78)
    w("GNSS - InSAR VELOCITY TIE  (run05)")
    w("=" * 78)
    w(f"Reference station        : {REF_STATION}")
    w(f"InSAR sampling radius    : {SAMPLE_RADIUS_M:.0f} m")
    w(f"Min valid pixels/station : {MIN_PIXELS}")
    w(f"Subsidence cut           : Vz < {VU_MAX_SUBSIDENCE:.1f} mm/yr dropped")
    w(f"Ramp gates               : LOO improvement >= {RAMP_MIN_IMPROVEMENT:.0%}, "
      f"inter-frame offset <= {OVERLAP_TOL:.1f} mm/yr")
    w("")
    w("-" * 78)
    w("STATION COUNTS")
    w("-" * 78)
    w(f"screened GNSS pool            : {n_pool}")
    w(f"dropped as fast-subsiding     : {n_subsiding}")
    for orbit in sorted(obs["orbit"].unique()):
        g = obs[obs["orbit"] == orbit]
        w(f"{orbit:<30}: {g.StaID.nunique()} unique stations "
          f"({len(g)} frame-station samples)")
        for fr in sorted(g["frame"].unique()):
            w(f"    frame {fr:<22}: {len(g[g['frame'] == fr])}")
    sets = {o: set(obs.loc[obs.orbit == o, "StaID"]) for o in obs["orbit"].unique()}
    if len(sets) == 2:
        a_, b_ = sets.values()
        w(f"common to both geometries     : {len(a_ & b_)}")
        w(f"union                         : {len(a_ | b_)}")
    w(f"fewest pixels at any station  : {int(obs.n_px.min())}")
    w("")
    w("-" * 78)
    w("PER-TRACK FIT")
    w("-" * 78)
    for st in group_stats:
        w(f"track {st['track']}  frames {', '.join(st['frames'])}  "
          f"{st['n_obs']} station samples")
        w(f"    gref at {REF_STATION}: {st['gref']:+.3f} mm/yr")
        for name, c in zip(QUAD_TERMS, st["quad"]):
            w(f"    {name:>3} {c:+9.4f} mm/yr per km^n")
        for fr, c in st["consts"].items():
            w(f"    constant {fr}: {c:+7.3f} mm/yr")
        w(f"    ramp applied: {st['apply_ramp']}  ({st['reason']})")
        w(f"    correction at {REF_STATION}: {st['corr_ref']:+.3f} mm/yr")
    w("")
    w("-" * 78)
    w("RESIDUAL RMSE BY ORBIT (mm/yr)  -- the numbers quoted for Fig. S2")
    w("-" * 78)
    w(f"{'orbit':<14}{'n':>5}{'before':>10}{'after':>10}{'after LOO':>12}")
    for orbit in sorted(obs["orbit"].unique()):
        g = obs[obs["orbit"] == orbit]
        w(f"{orbit:<14}{len(g):>5}"
          f"{np.sqrt(np.mean(g.residual_before ** 2)):>10.3f}"
          f"{np.sqrt(np.mean(g.residual_after ** 2)):>10.3f}"
          f"{np.sqrt(np.nanmean(g.residual_after_loo ** 2)):>12.3f}")
    w("")
    outdir = outdir or common_paths["validation_dir"]
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "gps_tie_stats.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"tie statistics -> {p}")
    return p


def decide_ramp(obs, consts, n_frames):
    """Whether to apply the fitted ramp. Two gates, because a marginal GNSS
    improvement can coincide with an inter-frame degradation:

    - LOO gate: the ramp must reduce the out-of-sample residual over
      datum-only by at least RAMP_MIN_IMPROVEMENT.
    - overlap gate (multi-frame groups): the shared quadratic cancels in the
      overlap, so the spread of the constants is the inter-frame offset the
      ramp would introduce. The overlap's pixels constrain the inter-frame
      offset more tightly than the sparse stations.
    """
    before = np.sqrt(np.mean(obs["residual_before"].to_numpy() ** 2))
    loo_rms = np.sqrt(np.nanmean(obs["residual_after_loo"].to_numpy() ** 2))
    improve = (before - loo_rms) / before
    if improve < RAMP_MIN_IMPROVEMENT:
        return False, f"LOO improves {improve*100:+.1f}% < " \
                      f"{RAMP_MIN_IMPROVEMENT*100:.0f}% -> datum tie only"
    if n_frames > 1:
        spread = max(consts.values()) - min(consts.values())
        if spread > OVERLAP_TOL:
            return False, f"ramp adds {spread:.2f} mm/yr inter-frame offset " \
                          f"> {OVERLAP_TOL:.2f} -> datum tie only"
    return True, f"LOO improves {improve*100:+.1f}%, overlap ok -> apply ramp"


def ramp_at(lon0, lat0, quad, const, center):
    """The fitted correction evaluated at one point (mm/yr)."""
    return float((quad_columns([lon0], [lat0], *center) @ quad)[0]) + const


def write_na_velocity(paths, frame_id, grid, quad, const, corr_ref, center, apply_ramp):
    """Write velocity_NA.h5: the InSAR velocity tied to GNSS, referenced at ZOA1.

    Per pixel: v = v_insar + [quad(x,y) + const - corr_ref] if the ramp is
    applied, else v = v_insar unchanged.

    `corr_ref` is the correction evaluated at ZOA1, one scalar per track
    group, so the output field is zero at ZOA1: the same datum as
    velocity_masked.h5 and timeseries_tropo.h5, which the creep analysis and
    the figures use. Because it is one scalar for the group, the offset
    between frames (difference of their `const`) is preserved, not flattened.
    `gref` is not added back: the field is NA-frame differential, zero at the
    reference station.
    """
    if apply_ramp:
        lon, lat = pixel_lonlat(grid)
        corr = (quad_columns(lon.ravel(), lat.ravel(), *center) @ quad).reshape(lon.shape)
        corr = corr + const - corr_ref
    else:
        corr = 0.0                    # already referenced by run04; nothing to add
    out = paths['velocity_NA']
    with h5py.File(paths['velocity_masked'], "r") as src, h5py.File(out, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for name in src:
            d = src[name][:]
            if name == "velocity":
                d = (d * MM + corr) / MM           # fitted in mm/yr, stored m/yr
            dst.create_dataset(name, data=d.astype("float32"))
        dst.attrs["REF_FRAME"] = ref_frame
        dst.attrs["TIE"] = (f"GNSS ramp, {ref_frame} frame, zero at {REF_STATION}"
                            if apply_ramp else "no ramp; identical to velocity_masked.h5")
    print(f"    {frame_id}: -> {os.path.basename(out)}"
          + ("" if apply_ramp else "  (no ramp; copy of velocity_masked)"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # 1) Screened GNSS velocities from run00 (any frame's CSV: the station
    #    list is AOI-wide), minus fast-subsiding stations
    gnss_path = paths_gps['enu'].format(frame="09157", ref_frame=ref_frame)
    gnss = pd.read_csv(gnss_path)
    print(f"GNSS: {len(gnss)} stations from {gnss_path}")
    subsiding = gnss["Vz"] < VU_MAX_SUBSIDENCE
    if subsiding.any():
        print(f"    dropping {subsiding.sum()} subsiding station(s): "
              f"{list(gnss.loc[subsiding, 'StaID'])}")
        gnss = gnss[~subsiding]

    # 2) Sample every frame; group frames by track (shared look geometry)
    groups, samples, grids = {}, {}, {}
    for frame, paths in FRAME_PATHS.items():
        df, grid, _ = sample_frame(frame, paths, gnss)
        samples[frame], grids[frame] = df, grid
        track = track_of(frame)
        groups.setdefault(track, []).append(frame)
        print(f"  {frame}  track {track}  {grid['orbit'].lower()}  "
              f"{len(df)} stations sampled")

    # 3) One independent solve per track group; gates decide the ramp
    all_obs, group_stats = [], []
    for track, group in groups.items():
        obs = pd.concat([samples[f] for f in group], ignore_index=True)
        quad, consts, gref, center, obs = fit_group(track, group, obs)
        apply_ramp, reason = decide_ramp(obs, consts, len(group))
        print(f"    ramp: {reason}")
        all_obs.append(obs.assign(track=track, ramp_applied=apply_ramp))
        group_stats.append(dict(track=track, frames=list(group), n_obs=len(obs),
                                quad=quad, consts=consts, gref=gref,
                                apply_ramp=apply_ramp, reason=reason))

        # 4) The correction at ZOA1, one scalar for the whole group, evaluated
        #    with the constant of the frame backing ZOA1 with the most pixels
        #    (a deterministic choice; other frames differ only by their const,
        #    which is the inter-frame offset, kept rather than averaged away)
        rlon, rlat = ref_lonlat(FRAME_PATHS[group[0]]['velocity_masked'])
        at_ref = obs[obs["StaID"] == REF_STATION]
        ref_frame_id = (str(at_ref.loc[at_ref["n_px"].idxmax(), "frame"])
                        if len(at_ref) else group[0])
        corr_ref = ramp_at(rlon, rlat, quad, consts.get(ref_frame_id), center)
        group_stats[-1]["corr_ref"] = corr_ref
        if apply_ramp:
            print(f"    correction at {REF_STATION}: {corr_ref:+.3f} mm/yr "
                  f"(removed, so the field is zero there)")
        for f in group:
            write_na_velocity(FRAME_PATHS[f], f, grids[f],
                              quad, consts[f], corr_ref, center, apply_ramp)

    # 5) Mosaic the tied fields per orbit, the same way run04 mosaics the
    #    untied ones. This is the decomposition's input and the archived
    #    product: one clean field per direction, no frame edge inside it.
    mdir = mosaic_dir
    os.makedirs(mdir, exist_ok=True)
    print(f"\n=== merge tied velocities -> {mdir} ===")
    by_orbit = {}
    for frame, paths in FRAME_PATHS.items():
        g = frame_grid(paths['mintpy_dir'])
        by_orbit.setdefault(g["orbit"], []).append(paths['dir'])
    for orbit, dirs in by_orbit.items():
        out = os.path.join(mdir, f"{orbit}_velocity_{ref_frame}.h5")
        if len(dirs) == 1:
            shutil.copy2(os.path.join(dirs[0], "mintpy", f"velocity_{ref_frame}.h5"), out)
        else:
            merge_orbit(dirs, f"velocity_{ref_frame}.h5", out)
        with h5py.File(out, "r") as h:
            fin = np.isfinite(h["velocity"][:])
        print(f"    {os.path.basename(out)}  {len(dirs)} frame(s)  "
              f"{100*fin.mean():.1f}% valid")

    # 6) Per-station residuals, as fitted (Fig. S2)
    out = os.path.join(data_dir, "GPS", "gps_insar_validation.csv")     # beside the GNSS series it was tied to
    pd.concat(all_obs, ignore_index=True).to_csv(out, index=False)
    print(f"\nstations -> {out}")
    write_tie_report(all_obs, group_stats, len(gnss) + int(subsiding.sum()),
                     int(subsiding.sum()))
    print("\ndone.")
