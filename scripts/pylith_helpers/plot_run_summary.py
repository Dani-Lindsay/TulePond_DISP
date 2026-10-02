#!/usr/bin/env python3
"""
One-page diagnostic figure per PyLith model run (mesh preset, dip, medium, Cd mode, surface constraint), made from
the cached products of a run alone:
  <gf_dir>/inversion_products<tag>.npz, <gf_dir>/results_comparison<tag>.csv, <gf_dir>/lcurve_<medium><tag>.csv
Panels: (a-c) data / model / residual at the quadtree blocks, (d) X-X' profile data vs model, (e) slip on the fault,
(f) L-curve with the selected sigma, (g) text: settings and numbers.
Output: <fig_dir>/<prefix>_<run>_<medium>.png (run = "default" + tag), 300 dpi, white, Helvetica/Arial, Okabe-Ito.

Ported from drafts/pylith (2026-09-30) for notebook 08b: `main(tag, media, gf_dir, fig_dir, prefix)`; gf_dir defaults
to the provided products (<model_sfcvm_dir>/gf), fig_dir to the repo's figure folder, prefix to "qc08b_summary".
Run in opera_disp_feb13:  python scripts/pylith_helpers/plot_run_summary.py [<tag> [<medium> ...]]
Old products (before 2026-09-30) lack settings / signed distance / block polygons: settings are inferred from the tag,
the signed distance is recomputed from 07a's trace and the block polygons come from fig2_model_row.npz (samples asserted equal).
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyproj
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from TulePond_filepaths import paths_sfcvm, paths_out, common_paths   # noqa: E402

OKABE = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", purple="#CC79A7", black="#000000", grey="#999999")
PROFILE_LAT = 37.5612                                       # X-X' (06), for the title only; the line comes from fig2_inputs
SSE_KM = None                                               # filled from 04's section_lengths.csv


def signed_distance_m(line_x, line_y, px, py):
    """Signed perpendicular distance (m, east +) from a densely sampled polyline: nearest sample, sign from the
    local tangent. (07a's dist_to_trace signs by the x-difference to the nearest ~100 m trace VERTEX, which flips
    the sign of samples within ~50 m of the trace between vertices; not used for sides here.)"""
    tangent = np.c_[np.gradient(line_x), np.gradient(line_y)]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    dist, i = cKDTree(np.c_[line_x, line_y]).query(np.c_[px, py])
    normal = np.c_[tangent[i, 1], -tangent[i, 0]]
    normal[normal[:, 0] < 0] *= -1.0                                  # east +
    return np.einsum("ij,ij->i", np.c_[px - line_x[i], py - line_y[i]], normal)


def trace_07():
    """07a's trace (cell [2]): the KML trace between lat 37.47-37.65, north to south, CSI-discretised every 20 m.
    Returns lon, lat, x_m, y_m in the CSI/mesh frame."""
    from insar_utils import load_trace
    from csi import TriangularPatches
    tl = load_trace(common_paths["fault_trace"])
    tl = tl[(tl[:, 1] >= 37.47) & (tl[:, 1] <= 37.65)]
    tl = tl[np.argsort(-tl[:, 1])]
    ref = TriangularPatches("trace frame", lon0=-121.968, lat0=37.557, verbose=False)
    ref.trace(tl[:, 0], tl[:, 1])
    ref.discretize2(every=0.02)
    return np.asarray(ref.loni), np.asarray(ref.lati), np.asarray(ref.xi) * 1e3, np.asarray(ref.yi) * 1e3


def sse_section_km():
    """The SSE section (04's section_lengths.csv), km from Point Pinole."""
    s = pd.read_csv(os.path.join(paths_out["04_event_detection"], "section_lengths.csv")).set_index("name")
    return float(s.loc["tulepond_creep_length", "km_lo"]), float(s.loc["tulepond_creep_length", "km_hi"])


def run_time_from_log(run_name):
    """Wall time (s) of the PyLith run from <model_sfcvm_dir>/runs/run_gf.log (last start/done pair for that name);
    None if the log is not provided (the Zenodo copy carries the products, not the run logs)."""
    log = os.path.join(paths_sfcvm["root"], "runs", "run_gf.log")
    if not os.path.exists(log):
        return None
    start = done = None
    for line in open(log):
        m = re.match(r"(\S+ \S+) (start|done) (\S+)", line.strip())
        if m and m.group(3) == run_name:
            t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            if m.group(2) == "start":
                start, done = t, None
            else:
                done = t
    return None if start is None or done is None else done - start


def settings_from_tag(tag):
    """Settings of a product made before settings were saved (inferred from its suffix)."""
    mesh = "shallow" if "_shallow" in tag else "default"
    cd = "diag" if "_diagcd" in tag else "full"
    return dict(mesh=mesh, dip="90 (vertical)", cd_mode=cd, cd_floor=None, use_surface="_nosurf" not in tag,
                points="default (07a quadtree, swath 4 km, 2386)", run_names={}, inferred_from_tag=True)


def load_run(tag, gf_dir):
    """Everything the figure needs, from the cached products (with the fallbacks for old products)."""
    prod = np.load(os.path.join(gf_dir, f"inversion_products{tag}.npz"), allow_pickle=True)
    results = pd.read_csv(os.path.join(gf_dir, f"results_comparison{tag}.csv")).set_index("model")
    media = [k[5:-3] for k in prod.files if k.startswith("slip_") and k.endswith("_mm")]
    lon, lat = prod["sample_lon"], prod["sample_lat"]
    if "settings" in prod.files:
        settings = json.loads(str(prod["settings"]))
    else:
        settings = settings_from_tag(tag)
    if "signed_d_km" in prod.files:
        signed_d_km = prod["signed_d_km"]
    else:
        tlon, tlat, tx, ty = trace_07()
        to_mesh = pyproj.Transformer.from_crs("EPSG:4326", str(np.load(os.path.join(paths_sfcvm["points"], "insar_points.npz"))["crs"]),
                                              always_xy=True)
        px, py = to_mesh.transform(lon, lat)
        signed_d_km = signed_distance_m(tx, ty, px, py) / 1e3
    if "block_lon" in prod.files:
        block_lon, block_lat = prod["block_lon"], prod["block_lat"]
    else:
        row = np.load(os.path.join(paths_out["07b_uniform_model_tests"], "fig2_model_row.npz"))
        assert np.array_equal(row["sample_lon"], lon) and np.array_equal(row["sample_lat"], lat), \
            "old products without block polygons must be on 07a's 2386 samples"
        block_lon, block_lat = row["block_lon"], row["block_lat"]
    if "trace_lon" in prod.files:
        trace_lon, trace_lat = prod["trace_lon"], prod["trace_lat"]
    else:
        trace_lon, trace_lat = trace_07()[:2]
    data = prod["data_mm"] if "data_mm" in prod.files else None
    out = dict(prod=prod, results=results, media=media, lon=lon, lat=lat, signed_d_km=signed_d_km, block_lon=block_lon,
               block_lat=block_lat, trace_lon=trace_lon, trace_lat=trace_lat, settings=settings, data=data,
               imp_xyz=prod["impulse_vertex_xyz"], imp_km=prod["impulse_km"], sub_faces=prod["sub_faces"])
    return out


def profile_selection(lon, lat):
    """Samples within 300 m of 06's X-X' line (fig2_inputs prof_fp_X_ends, EPSG of 06), and the 06 pixel profile."""
    fig2_in = np.load(os.path.join(paths_out["06_event_field_geometry"], "fig2_inputs.npz"))
    to_field = pyproj.Transformer.from_crs(4326, int(fig2_in["epsg"]), always_xy=True)
    xw, yw, xe, ye = fig2_in["prof_fp_X_ends"]
    u = np.array([xe - xw, ye - yw]); length = np.linalg.norm(u); u /= length
    sx, sy = to_field.transform(lon, lat)
    along = (sx - xw) * u[0] + (sy - yw) * u[1]
    perp = -(sx - xw) * u[1] + (sy - yw) * u[0]
    sel = (np.abs(perp) <= 300) & (along >= 0) & (along <= length)
    return sel, fig2_in["prof_fp_X_d"] / 1e3, fig2_in["prof_fp_X_v"]


def main(tag="", media=None, gf_dir=None, fig_dir=None, prefix="qc08b_summary"):
    global SSE_KM
    gf_dir = gf_dir or paths_sfcvm["gf"]
    fig_dir = fig_dir or common_paths["fig_dir"]
    # slide-sized QC page: 13 pt text, scoped to this call (rc_context) so the journal-sized figures drawn
    # afterwards in the same kernel keep their 7 pt axis labels
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 13,
                                "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                                "savefig.facecolor": "white", "figure.facecolor": "white"}):
        return _main(tag, media, gf_dir, fig_dir, prefix)


def _main(tag, media, gf_dir, fig_dir, prefix):
    global SSE_KM
    run = load_run(tag, gf_dir)
    if SSE_KM is None:
        SSE_KM = sse_section_km()
    media = media or run["media"]
    run_label = "default" + tag
    sel, prof_d, prof_v = profile_selection(run["lon"], run["lat"])
    signed = run["signed_d_km"]
    paths = []
    for medium in media:
        prod, res = run["prod"], run["results"].loc[f"PyLith {medium}"]
        model, resid = prod[f"model_{medium}_mm"], prod[f"residual_{medium}_mm"]
        data = run["data"] if run["data"] is not None else model + resid
        slip = prod[f"slip_{medium}_mm"]
        lc = pd.read_csv(os.path.join(gf_dir, f"lcurve_{medium}{tag}.csv"))
        st = run["settings"]
        run_name = st.get("run_names", {}).get(medium, f"gf_{st['mesh']}_{medium}")
        wall = run_time_from_log(run_name)
        n_imp = slip.size

        fig = plt.figure(figsize=(22, 12.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], height_ratios=[0.95, 1])
        # (a-c) maps
        pad = 0.005
        xlim = (run["block_lon"].min() - pad, run["block_lon"].max() + pad)
        ylim = (run["block_lat"].min() - pad, run["block_lat"].max() + pad)
        block_poly = np.stack([run["block_lon"], run["block_lat"]], axis=-1)
        for c, (vals, title) in enumerate(((data, "(a) data"), (model, "(b) model"),
                                           (resid, f"(c) residual, rms {np.sqrt(np.mean(resid ** 2)):.2f} mm"))):
            ax = fig.add_subplot(gs[0, c])
            pc = PolyCollection(block_poly, array=vals, cmap="RdBu_r", edgecolors="none")
            pc.set_clim(-4, 4); ax.add_collection(pc)
            ax.plot(run["trace_lon"], run["trace_lat"], "k-", lw=0.8)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect(1 / np.cos(np.deg2rad(37.557)))
            ax.set_title(title, loc="left")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=11)
            if c == 2:
                fig.colorbar(pc, ax=ax, shrink=0.8, label="fault-parallel (mm)")
        # (g) text panel
        ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
        cd_txt = {"full": "07a correlated Cd (nugget / n_px^0.5)", "diag": "diagonal, equal weights (median nugget)",
                  "floor": f"07a correlated Cd, n_px floored at {st.get('cd_floor')}",
                  "nugget": "diagonal, 07a's nugget / n_px^0.5 (no correlation)"}.get(st["cd_mode"], st["cd_mode"])
        lines = [f"run: {run_label}   medium: {medium}",
                 f"PyLith run: {run_name}",
                 f"mesh: {st['mesh']}",
                 f"dip: {st['dip']}",
                 f"points: {st['points']}",
                 f"Cd: {cd_txt}",
                 f"surface constraint: {'ON (crossing stack)' if st['use_surface'] else 'OFF (InSAR only)'}",
                 "",
                 f"sigma: {res.sigma:g} (L-curve corner {lc.corner.iloc[0]:.1f})",
                 f"rms: all {res.rms_mm:.2f}   W<=300 m {res.rms_west300_mm:.2f}   E<=300 m {res.rms_east300_mm:.2f} mm",
                 f"chi2 (InSAR, with the Cd above): {res.chi2:.0f}   n = {data.size}",
                 f"surface slip / crossings: {res.surface_pct:.0f} %",
                 f"potency: {res.potency_m3:.0f} m3   Mw(30 GPa) {res.Mw_30gpa:.2f}   Mw(SFSVM mu) {res.Mw_sfcvm:.2f}",
                 f"peak slip {res.get('peak_slip_mm', np.nan):.1f} mm at {abs(res.get('peak_depth_m', np.nan)):.0f} m depth",
                 "",
                 f"impulses: {n_imp}   run time: " + (f"{wall / 60:.0f} min ({wall / n_imp:.2f} s per impulse, 8 ranks)"
                                                    if wall else "(run log not provided; ~33 min, 8 ranks)"),
                 f"made {time.strftime('%Y-%m-%d %H:%M')} from inversion_products{tag}.npz"]
        if st.get("inferred_from_tag"):
            lines.append("(settings inferred from the product suffix)")
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=12.5, family="monospace",
                transform=ax.transAxes, linespacing=1.45)
        ax.set_title("(g) settings and numbers", loc="left")
        # (d) X-X' profile
        ax = fig.add_subplot(gs[1, 0:2])
        order = np.argsort(signed[sel])
        ax.plot(prof_d, prof_v, "-", color=OKABE["grey"], lw=1, label="06 X-X' pixel profile")
        ax.plot(signed[sel], data[sel], "o", color=OKABE["black"], ms=5, label="data (samples within 300 m of X-X')")
        ax.plot(signed[sel][order], model[sel][order], "-", color=OKABE["vermillion"], lw=2.2, label=f"model ({medium})")
        ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-2.5, 2.5)
        ax.axvspan(-0.3, 0, color=OKABE["orange"], alpha=0.12); ax.axvspan(0, 0.3, color=OKABE["blue"], alpha=0.10)
        ax.text(-0.32, ax.get_ylim()[1] * 0.92, f"W<=300 m rms {res.rms_west300_mm:.2f} mm", ha="right", va="top", fontsize=12)
        ax.text(0.32, ax.get_ylim()[1] * 0.92, f"E<=300 m rms {res.rms_east300_mm:.2f} mm", ha="left", va="top", fontsize=12)
        ax.set_xlabel("distance from the trace, east + (km)"); ax.set_ylabel("fault-parallel (mm)")
        ax.set_title(f"(d) X-X' (lat {PROFILE_LAT} N): data vs model; rms bands = all samples within 300 m of the trace", loc="left")
        ax.legend(loc="lower right")
        # (e) slip
        ax = fig.add_subplot(gs[1, 2])
        tri = mtri.Triangulation(run["imp_km"], run["imp_xyz"][:, 2] / 1e3, run["sub_faces"])
        tp = ax.tripcolor(tri, slip, shading="gouraud", cmap="magma", vmin=0, vmax=max(1.0, slip.max()))
        ax.set_xlim(54, 65); ax.set_ylim(-1.5, 0); ax.invert_xaxis()
        ax.hlines(-0.02, SSE_KM[0], SSE_KM[1], color=OKABE["orange"], lw=6)
        ax.set_xlabel("km from Point Pinole (orange bar: SSE section)"); ax.set_ylabel("z (km)")
        ax.set_title(f"(e) slip, sigma {res.sigma:g}: peak {slip.max():.1f} mm", loc="left")
        fig.colorbar(tp, ax=ax, shrink=0.8, label="right-lateral slip (mm)")
        # (f) L-curve
        ax = fig.add_subplot(gs[1, 3])
        ax.plot(np.sqrt(lc.chi2_total), lc.prior_norm, "o-", color=OKABE["blue"])
        for s, xv, yv in zip(lc.sigma, np.sqrt(lc.chi2_total), lc.prior_norm):
            ax.annotate(f"{s:g}", (xv, yv), xytext=(4, 2), textcoords="offset points", fontsize=10)
        pick = lc.sigma == res.sigma
        ax.plot(np.sqrt(lc.chi2_total[pick]), lc.prior_norm[pick], "s", ms=13, mfc="none", mec=OKABE["vermillion"], mew=2,
                label=f"selected sigma {res.sigma:g}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend()
        ax.set_xlabel(r"weighted misfit $\sqrt{\chi^2}$ (InSAR + surface)"); ax.set_ylabel(r"roughness $\sqrt{m^T C^{-1} m}$")
        ax.set_title(f"(f) L-curve (corner {lc.corner.iloc[0]:.1f})", loc="left")
        fig.suptitle(f"PyLith {medium} | {run_label} | mesh {st['mesh']}, dip {st['dip']}, Cd {st['cd_mode']}, "
                     f"surface {'on' if st['use_surface'] else 'off'}", fontsize=15, x=0.01, ha="left")
        os.makedirs(fig_dir, exist_ok=True)
        path = os.path.join(fig_dir, f"{prefix}_{run_label}_{medium}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    main(tag, sys.argv[2:] or None)
