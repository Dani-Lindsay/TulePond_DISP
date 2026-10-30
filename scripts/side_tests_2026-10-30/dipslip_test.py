"""dipslip_test.py (2026-10-30, for Roland Burgmann via Dani): does letting the fault DIP-SLIP (rake free) improve the slip
model of the Tule Pond creep events, in the HOMOGENEOUS elastic half-space?

Locked inversion settings (sensitivity_2026-10-27/sensitivity_rows.py, depth_scan_2026-10-28/depth_scan.py):
  60 m near-trace cut (closer samples get no weight); equal weights (every sample = the nugget variance of the 5 km-mask
  semivariogram fitted on a 15 km swath, model_utils.covariance); plane (1, x, y) per dataset solved jointly; CSI exponential
  prior buildCmXY(sigma, lam_xz = (0.1, 0.05) km); each model at its own L-curve corner (tent_lcurve / lcurve_corner rule:
  sqrt(chi2) vs prior norm over the 3D sigma grid); no surface constraint; strike-slip right-lateral >= 0.
Data: 06's mean event fields (fp_maps.npz): fault-parallel fp_event (07a's quadtree samples, insar_points.npz) and vertical
  up_event on the SAME quadtree blocks (asserted), look (0, 0, 1), its own 5 km / 15 km semivariogram nugget.
Forward model: exact half-space, Meade triangular dislocations (CSI), on the tents of the standard mesh fault
  (gf_default_sfcvm_order2), model_utils.analytic_tent_G's sub-division (n_sub 4), strike-slip AND dip-slip, both looks.
Free tents: depth <= 1.5 km and km 55-63 from Point Pinole (every other tent zero), the same for every model.
Models: A strike-slip only, fp + vertical; B strike-slip (>= 0) + dip-slip (free sign), fp + vertical; C strike-slip only, fp
  alone (the paper's homogeneous setup).
Solver (own, short): the augmented least squares of CSI's nnls branch, [W G; R] m = [W d; 0] with R = chol(Cm)^-1, solved
  exactly by eliminating the UNBOUNDED parameters (dip-slip, planes) with an orthogonal projection, NNLS on the strike-slip,
  then back-substitution (variable projection; exact for a linear problem). Checked: (1) KKT optimality of every corner
  solution; (2) all-tent fault-parallel-only solve at the rescore_dips corner reproduces the CSI inversion (rms, Mw).
Run: see README.md. Imports ONLY from the frozen repo_snapshot/ next to this script.
"""
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from scipy.optimize import nnls
from scipy.linalg import solve_triangular

OUT = os.path.expanduser("~/work/proj_TulePond_DISP/dipslip_2026-10-30")
REPO = os.path.join(OUT, "repo_snapshot")             # frozen copy of the repo (never the live repo: being rewired)
DATA = os.path.expanduser("~/work/proj_TulePond_DISP")
BK = os.path.join(DATA, "modeling_SFCVM")
sys.path.insert(0, os.path.join(REPO, "scripts", "model_plots"))
sys.path.insert(0, REPO)
t_start = time.time()

# ---- 07a cells [1]-[2] (paths, trace, CSI fault) and 08c cells [1]-[3] (samples, crossings, trace): read only
src07 = open(os.path.join(REPO, "scripts", "07a_uniform_model.py")).read()
exec(compile(src07.split("# ## [3] Data")[0], "07a_cells_1_2", "exec"))
import model_utils                                    # noqa: E402
import TulePond_filepaths                             # noqa: E402
for mod in (model_utils, TulePond_filepaths):
    assert os.path.realpath(mod.__file__).startswith(os.path.realpath(REPO)), f"{mod.__name__} not from the snapshot: {mod.__file__}"
src08 = open(os.path.join(REPO, "scripts", "08c_3d_model_sensitivity.py")).read()
# 2026-10-30: the live pipeline is being rewired (07a's products moved to outputs/07a_uniform_model/retired_2026-10-28/,
# insar_points.npz rewritten 11:38 with the same positions and values, asserted below). Read frozen copies in inputs_frozen/.
FROZEN = os.path.join(OUT, "inputs_frozen")
for name in ("model_data_07a.npz", "fig2_model_row.npz"):
    old_ = f'os.path.join(dir_07a, "{name}")'
    assert src08.count(old_) == 1, name
    src08 = src08.replace(old_, repr(os.path.join(FROZEN, name)))
exec(compile(src08.split("## [4]")[0], "08c_cells_1_3", "exec"))
points_npz["default"] = os.path.join(FROZEN, "insar_points.npz")     # frozen copy of the 07a default samples
_pf = np.load(points_npz["default"])
assert np.array_equal(_pf["lon"], model_data["sample_lon"]) and np.array_equal(_pf["vel_mm"], model_data["data_mm"])
assert str(_pf["field_fingerprint"]) == field_fingerprint
from TulePond_filepaths import model_crs              # noqa: E402
import pyproj                                         # noqa: E402

RUN, CUT_KM, N_SUB = "gf_default_sfcvm_order2", 0.06, 4
FREE_DEPTH_KM, FREE_KM = 1.5, (55.0, 63.0)
MU = 30e9
assert tuple(np.round(lam_xz, 6)) == (0.1, 0.05), lam_xz
fp = np.load(fp_file)
strike = float(fp["strike"])
assert abs(strike - strike_deg) < 1e-6
p = np.load(points_npz["default"])
assert float(p["swath_km"]) == swath_km == 4.0
LOOK_FP = np.asarray(look, float)
LOOK_UP = np.array([0.0, 0.0, 1.0])
assert np.allclose(LOOK_FP, model_utils.look_vector(strike_deg))

# =====================================================================================================================
# 1. Data: the vertical field on the SAME quadtree blocks; the two noise models (5 km mask, 15 km swath)
# =====================================================================================================================
assert np.array_equal(np.isfinite(fp["fp_event"]), np.isfinite(fp["up_event"])), "fp and up fields differ in pixel validity"
ds = {}
for comp, field in (("fp", "fp_event"), ("up", "up_event")):
    # fault_parallel_dataset only masks (swath, exclusion, km window); its look vector is not used for the vertical set
    sar, _ = model_utils.fault_parallel_dataset(fp[field], fp["lon"], fp["lat"], strike_deg, fault, model_frame_lonlat,
                                                swath_km, exclude_km, km_window, profile_lats)
    ds[comp] = model_utils.downsample(sar, fault, start_block_km, min_block_km)
    print(f"quadtree {comp}: {ds[comp].vel.size} blocks", flush=True)
d_fp = ds["fp"]
# the rebuilt fp quadtree must be 07a's samples (positions and values)
assert d_fp.vel.size == p["lon"].size, (d_fp.vel.size, p["lon"].size)
assert np.max(np.hypot(d_fp.x - p["x_km"], d_fp.y - p["y_km"])) * 1e3 < 1.0
assert np.allclose(d_fp.vel, p["vel_mm"], atol=1e-6), "rebuilt fp quadtree values differ from insar_points.npz"
# the vertical blocks: same centres (< 1 m) -> use them; else fall back to averaging up_event over 07a's block corners
same_blocks = ds["up"].vel.size == p["lon"].size and np.max(np.hypot(ds["up"].x - p["x_km"], ds["up"].y - p["y_km"])) * 1e3 < 1.0
assert same_blocks, "vertical quadtree blocks differ from insar_points: average up_event over block_lon/block_lat instead"
centre_mismatch_m = float(np.max(np.hypot(ds["up"].x - p["x_km"], ds["up"].y - p["y_km"])) * 1e3)
assert np.allclose(ds["up"].wgt, p["wgt"])
data = {"fp": p["vel_mm"].astype(float), "up": np.asarray(ds["up"].vel, float).copy()}
assert np.isfinite(data["up"]).all()
print(f"vertical samples on 07a's blocks: max centre mismatch {centre_mismatch_m:.2e} m", flush=True)

nug = {}
cov_fit = {}
for comp, field in (("fp", "fp_event"), ("up", "up_event")):
    sar_wide, _ = model_utils.fault_parallel_dataset(fp[field], fp["lon"], fp["lat"], strike_deg, fault, model_frame_lonlat,
                                                     15.0, exclude_km, km_window, profile_lats)
    cp = np.asarray(model_utils.covariance(sar_wide, fault, 15.0, 5.0, cov_distmax_km), float)
    cov_fit[comp] = cp
    nug[comp] = float(np.sqrt(cp[0]))
    print(f"semivariogram {comp} (5 km mask, 15 km swath): nugget {nug[comp]:.3f} mm, short {np.sqrt(cp[1]):.3f} mm / {cp[2]:.2f} km, "
          f"long {np.sqrt(cp[3]):.3f} mm / {cp[4]:.1f} km", flush=True)

# =====================================================================================================================
# 2. Tents of the standard mesh, the free set, the analytic Green's functions (strike-slip and dip-slip, both looks)
# =====================================================================================================================
gpath = os.path.join(BK, "gf", f"G_{RUN}.npz")
assert os.path.exists(gpath) and abs(float(np.load(gpath)["strike_deg"]) - strike_deg) < 1e-6   # load_G will not rebuild/write
s = model_utils.tent_model_setup({"m": RUN}, points_npz["default"], strike_deg, {"m": (os.path.join(BK, "pylith_output"), os.path.join(BK, "gf"))},
                                 crossings, km_window, creep_lo_km, creep_hi_km, surf_floor_mm, trace_south_north, trace_xy_m, frame_lonlat,
                                 model_crs, profile_lats, near_trace_cut_km=CUT_KM, trace_km_pp=trace["km_pp"])
t = s["tents"]
keep = np.asarray(s["keep"], bool)
d_km, km_pp = s["signed_d_km"], s["sample_km_pp"]          # d_km < 0 = west of the trace
in_section = (km_pp >= creep_lo_km) & (km_pp <= creep_hi_km)
w500 = keep & in_section & (d_km < 0) & (d_km >= -0.5)
e500 = keep & in_section & (d_km > 0) & (d_km <= 0.5)
free = (t.imp_depth_km <= FREE_DEPTH_KM + 1e-6) & (t.imp_km >= FREE_KM[0]) & (t.imp_km <= FREE_KM[1])
cols = np.nonzero(free)[0]
nf = cols.size
print(f"{RUN}: {t.n_imp} tents, {nf} free (depth <= {FREE_DEPTH_KM} km, km {FREE_KM}); {keep.sum()} fitted samples of {keep.size} "
      f"(cut {CUT_KM * 1e3:.0f} m); W500 {w500.sum()}, E500 {e500.sum()}; SSE section km {creep_lo_km:.2f}-{creep_hi_km:.2f}", flush=True)


def analytic_tent_G_sd(tents, cols_free, lon, lat, looks, frame_lonlat, to_mesh, n_sub):
    """model_utils.analytic_tent_G (n_sub^2 constant-slip Meade sub-triangles per fault triangle, tent-weighted at the
    sub-triangle centroid), restricted to the tents cols_free, for several look vectors at once (the samples repeated, one
    block of rows per look) and BOTH slip directions. Returns (G_ss, G_ds), (len(looks) * n_samples, n_free), m per m;
    G_ss = -G_csi (right-lateral positive, as analytic_tent_G); G_ds = CSI's dip-slip as it comes (sign fixed later from
    the predictions)."""
    from csi import TriangularPatches, insar as csi_insar
    from scipy.sparse import csr_matrix
    n = np.size(lon)
    looks = np.asarray(looks, float)
    line = csi_insar("samples", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
    line.read_from_binary(np.zeros(n * len(looks)), np.tile(lon, len(looks)), np.tile(lat, len(looks)), err=np.ones(n * len(looks)),
                          los=np.repeat(looks, n, axis=0), remove_nan=False, remove_zeros=False)
    assert line.vel.size == n * len(looks)
    column_of_vertex = {int(tents.vertex_index[c]): k for k, c in enumerate(cols_free)}
    sub_corners = []
    for i in range(n_sub):
        for j in range(n_sub - i):
            sub_corners.append(((i, j), (i + 1, j), (i, j + 1)))
            if i + j < n_sub - 1:
                sub_corners.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
    lams = np.array([[[a / n_sub, b / n_sub, 1 - (a + b) / n_sub] for a, b in sc] for sc in sub_corners])
    assert lams.shape[0] == n_sub ** 2
    w_corner = lams.mean(1)
    pts, rows, cols_, vals = [], [], [], []
    for cell in tents.fault_cells:
        hit = [k for k, v in enumerate(cell) if int(v) in column_of_vertex]
        if not hit:
            continue
        first = len(pts) * n_sub ** 2
        pts.append(np.einsum("sab,bc->sac", lams, tents.fault_xyz[cell]))
        for k in hit:
            rows.append(first + np.arange(n_sub ** 2))
            cols_.append(np.full(n_sub ** 2, column_of_vertex[int(cell[k])]))
            vals.append(w_corner[:, k])
    pts = np.concatenate(pts)
    n_patch = pts.shape[0]
    W = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols_))), shape=(n_patch, len(cols_free)))
    assert np.all(np.asarray(W.sum(0)).ravel() > 0)
    print(f"  analytic tents: {n_patch:,} sub-triangles, {line.vel.size} rows, ss + ds", flush=True)
    G_ss = np.zeros((line.vel.size, len(cols_free)))
    G_ds = np.zeros_like(G_ss)
    n_flipped = [0]
    chunk = 2000
    for start in range(0, n_patch, chunk):
        part = pts[start:start + chunk]
        lon_v, lat_v = to_mesh.transform(part[:, :, 0].ravel(), part[:, :, 1].ravel(), direction="INVERSE")
        flt = TriangularPatches("analytic tents", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
        x, y = flt.ll2xy(lon_v, lat_v)
        flt.Vertices = np.c_[x, y, -part[:, :, 2].ravel() / 1e3]
        flt.Faces = np.arange(3 * part.shape[0]).reshape(-1, 3)
        flt.vertices2ll()
        flt.patch = [flt.Vertices[f] for f in flt.Faces]
        flt.patch2ll()
        flt.setdepth()
        flt.homogeneizeStrike(direction=0, sign=-1.0)
        flt.initializeslip()
        flt.buildGFs(line, vertical=True, slipdir="sd", verbose=False)
        # 2026-10-30 (run 1 caught it: uniform dip-slip gave ~0 vertical step): CSI's Meade code (triangularDisp.displacement)
        # re-orders every triangle so that its normal points UP (normal z >= 0) whatever homogeneizeStrike did. On a vertical
        # fault the normal z is +/- round-off, so the normal (the moving side) flips between sub-triangles. Strike-slip is
        # unaffected (strike vector and moving side flip together), but the dip vector stays up, so the dip-slip SENSE flips.
        # Undo it: sub-triangles that triangularDisp flips (normal z < 0 in the homogenised vertex order) get G_ds x -1.
        tri_ = np.array(flt.patch)
        nz = np.cross(tri_[:, 1] - tri_[:, 0], tri_[:, 2] - tri_[:, 0])[:, 2]
        ds_flip = np.where(nz < 0, -1.0, 1.0)
        n_flipped[0] += int((nz < 0).sum())
        for key, Gacc, fac in (("strikeslip", G_ss, np.ones(part.shape[0])), ("dipslip", G_ds, ds_flip)):
            Gp = flt.G[line.name][key]
            assert Gp.shape == (line.vel.size, part.shape[0])
            Gacc += np.asarray((W[start:start + chunk].T @ (Gp * fac[None, :]).T).T)
        print(f"  analytic tents: {min(start + chunk, n_patch):,} / {n_patch:,} ({(time.time() - t_start) / 60:.1f} min)", flush=True)
    print(f"  dip-slip sense made consistent: {n_flipped[0]:,} of {n_patch:,} sub-triangles flipped back", flush=True)
    return -G_ss, G_ds


to_mesh_ll = pyproj.Transformer.from_crs("EPSG:4326", model_crs, always_xy=True)
gcache = os.path.join(OUT, f"G_dipslip_{RUN}_nsub{N_SUB}.npz")
if os.path.exists(gcache):
    gc = np.load(gcache)
    assert np.array_equal(gc["cols"], cols) and np.array_equal(gc["sample_lon"], p["lon"])
    G_ss_all, G_ds_all = gc["G_ss"], gc["G_ds"]
else:
    G_ss_all, G_ds_all = analytic_tent_G_sd(t, cols, p["lon"], p["lat"], [LOOK_FP, LOOK_UP], frame_lonlat, to_mesh_ll, N_SUB)
    np.savez(gcache, G_ss=G_ss_all, G_ds=G_ds_all, cols=cols, sample_lon=p["lon"], sample_lat=p["lat"],
             provenance=f"dipslip_test.py {time.strftime('%Y-%m-%d')}: Meade/CSI analytic tents of {RUN}, n_sub {N_SUB}; rows = "
                        "[fault-parallel look; vertical look] x samples; G_ss right-lateral +; G_ds CSI sign")
ns = p["lon"].size
G = {("fp", "ss"): G_ss_all[:ns], ("up", "ss"): G_ss_all[ns:], ("fp", "ds"): G_ds_all[:ns], ("up", "ds"): G_ds_all[ns:]}
# check against the cached all-tent fault-parallel strike-slip G of dip_checks_2026-10-26
Ga = np.load(os.path.join(DATA, "dip_checks_2026-10-26", f"G_analytic_{RUN}_nsub{N_SUB}.npy"))
assert Ga.shape == (ns, t.n_imp)
rel_check = float(np.max(np.abs(G[("fp", "ss")] - Ga[:, cols])) / np.max(np.abs(Ga[:, cols])))
print(f"fp strike-slip G vs cached dip_checks G: max relative difference {rel_check:.2e}", flush=True)
assert rel_check < 1e-6, rel_check
# right-lateral check: uniform strike-slip in the section -> east-minus-west fault-parallel step > 0 (as analytic_tent_G)
uni = (t.imp_km[cols] >= creep_lo_km) & (t.imp_km[cols] <= creep_hi_km)
pred = G[("fp", "ss")] @ uni.astype(float)
step_ss = float(np.mean(pred[e500]) - np.mean(pred[w500]))
assert step_ss > 0, step_ss
# dip-slip sense from the predictions: uniform +1 (CSI) dip-slip in the section, vertical east-minus-west within 500 m
pred = G[("up", "ds")] @ uni.astype(float)
step_ds_csi = float(np.mean(pred[e500]) - np.mean(pred[w500]))
DS_SIGN = 1.0 if step_ds_csi > 0 else -1.0
# physics check: a vertical fault slipping uniformly in dip-slip from the surface puts a full-slip vertical offset at the trace
# (+/- 1/2 on each side), decaying away from it: the 60-200 m bins must be near +/- 0.4, the 500 m means well above 0.1
prof = [float(np.mean(pred[keep & in_section & (d_km >= lo) & (d_km < hi)])) * DS_SIGN
        for lo, hi in ((-1, -0.5), (-0.5, -0.2), (-0.2, -0.06), (0.06, 0.2), (0.2, 0.5), (0.5, 1))]
print("  uniform dip-slip (east side up), vertical in bins -1..-0.5, -0.5..-0.2, -0.2..-0.06, 0.06..0.2, 0.2..0.5, 0.5..1 km: "
      + ", ".join(f"{v:+.3f}" for v in prof), flush=True)
assert abs(step_ds_csi) > 0.2 and prof[2] < -0.2 and prof[3] > 0.2, "dip-slip Green's functions not consistent in sense"
G[("fp", "ds")] = DS_SIGN * G[("fp", "ds")]
G[("up", "ds")] = DS_SIGN * G[("up", "ds")]
print(f"uniform unit dip-slip (CSI sign) -> vertical east minus west {step_ds_csi:+.3f} (m/m): CSI + = "
      f"{'east' if step_ds_csi > 0 else 'west'} side up. Reported dip-slip + = EAST SIDE UP (G_ds x {DS_SIGN:+.0f})", flush=True)
pred = G[("up", "ss")] @ uni.astype(float)
print(f"  (uniform strike-slip vertical east minus west {np.mean(pred[e500]) - np.mean(pred[w500]):+.4f}; "
      f"uniform dip-slip fault-parallel east minus west {np.mean((G[('fp', 'ds')] @ uni)[e500]) - np.mean((G[('fp', 'ds')] @ uni)[w500]):+.4f})")

# prior: CSI buildCmXY on the full template (lam0 from the whole fault, as the locked runs), sub-block of the free tents
tmpl = copy.deepcopy(s["template"])
tmpl.slipdir = "s"
tmpl.buildCmXY(1.0, lam_xz, verbose=False)
Cm1 = np.asarray(tmpl.Cm)[np.ix_(cols, cols)]          # sigma = 1; Cm(sigma) = sigma^2 Cm1
Lc = np.linalg.cholesky(Cm1)                           # Cm1 = Lc Lc^T -> Cm1^-1 = Lc^-T Lc^-1, prior rows R = Lc^-1 / sigma
Linv = solve_triangular(Lc, np.eye(nf), lower=True)
assert np.allclose(Linv @ Cm1 @ Linv.T, np.eye(nf), atol=1e-6)
plane = np.c_[np.ones(ns), p["x_km"], p["y_km"]]


# =====================================================================================================================
# 3. The solver (augmented least squares; strike-slip >= 0, dip-slip and planes free) and the L-curve
# =====================================================================================================================
def solve(sigma, comps, dirs, Gd=None, Linv_=None, keep_=None):
    """min sum_c ||(G_c m + P_c a_c - d_c) / nug_c||^2 (fitted samples) + ||Linv m_dir / sigma||^2 per slip direction,
    strike-slip >= 0, dip-slip and plane coefficients a_c free. Unbounded parameters eliminated exactly (orthogonal
    projection), NNLS on the strike-slip, back-substitution. Returns dict(ss, ds, planes, A, b, x, bounded mask)."""
    Gd = G if Gd is None else Gd
    Linv_ = Linv if Linv_ is None else Linv_
    keep_ = keep if keep_ is None else keep_
    n_t = Linv_.shape[0]
    blocks, rhs = [], []
    n_par = n_t * len(dirs) + 3 * len(comps)
    for ic, c in enumerate(comps):
        rows = np.zeros((keep_.sum(), n_par))
        for idr, dr in enumerate(dirs):
            rows[:, idr * n_t:(idr + 1) * n_t] = Gd[(c, dr)][keep_] / nug[c]
        rows[:, n_t * len(dirs) + 3 * ic: n_t * len(dirs) + 3 * ic + 3] = plane[keep_] / nug[c]
        blocks.append(rows)
        rhs.append(data[c][keep_] / nug[c])
    for idr in range(len(dirs)):                                    # prior rows, the same Cm for every slip direction
        rows = np.zeros((n_t, n_par))
        rows[:, idr * n_t:(idr + 1) * n_t] = Linv_ / sigma
        blocks.append(rows)
        rhs.append(np.zeros(n_t))
    A, b = np.vstack(blocks), np.concatenate(rhs)
    bounded = np.zeros(n_par, bool)
    bounded[:n_t] = True                                            # strike-slip (first direction) >= 0
    assert dirs[0] == "ss"
    Q, _ = np.linalg.qr(A[:, ~bounded])                             # orthonormal basis of the unbounded columns
    Ab = A[:, bounded] - Q @ (Q.T @ A[:, bounded])
    bb = b - Q @ (Q.T @ b)
    ss, _ = nnls(Ab, bb, maxiter=50 * n_t)
    unb, *_ = np.linalg.lstsq(A[:, ~bounded], b - A[:, bounded] @ ss, rcond=None)
    x = np.zeros(n_par)
    x[bounded], x[~bounded] = ss, unb
    out = dict(ss=ss, ds=x[n_t:2 * n_t] if "ds" in dirs else np.zeros(n_t), A=A, b=b, x=x, bounded=bounded,
               planes={c: x[n_t * len(dirs) + 3 * ic: n_t * len(dirs) + 3 * ic + 3] for ic, c in enumerate(comps)})
    return out


def kkt_check(sol, label):
    """Optimality of the convex bounded LS: gradient g = A^T (A x - b) ~ 0 for free / positive parameters, >= 0 at the bound."""
    g = sol["A"].T @ (sol["A"] @ sol["x"] - sol["b"])
    scale = np.linalg.norm(sol["A"].T @ sol["b"]) / np.sqrt(g.size)
    active = sol["bounded"] & (sol["x"] <= 1e-10)
    g_free = np.max(np.abs(g[~active])) / scale
    g_bound = np.min(g[active]) / scale if active.any() else 0.0
    print(f"  KKT {label}: max |grad| free {g_free:.1e}, min grad at bound {g_bound:.1e} (relative)", flush=True)
    assert g_free < 1e-5 and g_bound > -1e-5, (g_free, g_bound)


def score(sol, comps, sigma, label):
    """Residuals against the plane-removed data (every component, also one that was not fitted: its own plane then)."""
    r = dict(model=label, sigma=sigma)
    for c in ("fp", "up"):
        m = G[(c, "ss")] @ sol["ss"] + G[(c, "ds")] @ sol["ds"]
        if c in sol["planes"]:
            a = sol["planes"][c]
        else:                                                         # not fitted: best plane to (data - slip prediction)
            a, *_ = np.linalg.lstsq(plane[keep], (data[c] - m)[keep], rcond=None)
        res = data[c] - m - plane @ a
        r[f"{c}_fitted"] = c in comps
        r[f"{c}_rms_mm"] = float(np.sqrt(np.mean(res[keep] ** 2)))
        r[f"{c}_rms_W500_mm"] = float(np.sqrt(np.mean(res[w500] ** 2)))
        r[f"{c}_rms_E500_mm"] = float(np.sqrt(np.mean(res[e500] ** 2)))
        r[f"{c}_mean_res_W500_mm"] = float(np.mean(res[w500]))
        r[f"{c}_mean_res_E500_mm"] = float(np.mean(res[e500]))
        r[f"{c}_chi2"] = float(np.sum((res[keep] / nug[c]) ** 2))
        r[f"_{c}_res"], r[f"_{c}_model"], r[f"_{c}_plane"] = res, m, plane @ a
    full_ss = np.zeros(t.n_imp); full_ss[cols] = sol["ss"]
    full_ds = np.zeros(t.n_imp); full_ds[cols] = sol["ds"]
    pot_ss, pot_ds = t.potency_m3(full_ss), t.potency_m3(np.abs(full_ds))
    r.update(potency_ss_m3=pot_ss, Mw30_ss=float(model_utils.moment_magnitude(MU * pot_ss)),
             potency_ds_abs_m3=pot_ds, potency_ds_net_m3=t.potency_m3(full_ds),
             Mw30_ds=float(model_utils.moment_magnitude(MU * pot_ds)) if pot_ds > 0 else np.nan,
             ds_over_ss_potency=pot_ds / pot_ss, peak_ss_mm=float(sol["ss"].max()),
             peak_ss_km=float(t.imp_km[cols][np.argmax(sol["ss"])]), peak_ss_depth_m=float(1e3 * t.imp_depth_km[cols][np.argmax(sol["ss"])]))
    k = int(np.argmax(np.abs(sol["ds"])))
    r.update(peak_ds_mm=float(sol["ds"][k]), peak_ds_km=float(t.imp_km[cols][k]), peak_ds_depth_m=float(1e3 * t.imp_depth_km[cols][k]),
             ds_sense=("none" if not np.any(sol["ds"]) else ("east side up" if sol["ds"][k] > 0 else "west side up")))
    # effective number of parameters: trace of the data resolution (hat) matrix with the active set held
    A, nd = sol["A"], sum(keep.sum() for _ in comps)
    act = ~(sol["bounded"] & (sol["x"] <= 1e-10))
    Af = A[:, act]
    Gf = Af[:nd]
    r["dof_eff"] = float(np.trace(Gf @ np.linalg.solve(Af.T @ Af, Gf.T)))
    r["n_data_fitted"] = int(nd)
    r["_ss"], r["_ds"] = full_ss, full_ds
    return r


def lcurve(comps, dirs, label):
    rows = []
    for sg in sigma_grid:
        sol = solve(sg, comps, dirs)
        chi2 = 0.0
        for c in comps:
            m = sum(G[(c, dr)] @ (sol["ss"] if dr == "ss" else sol["ds"]) for dr in dirs) + plane @ sol["planes"][c]
            chi2 += float(np.sum(((data[c] - m)[keep] / nug[c]) ** 2))
        rough = float(np.sqrt(np.sum((Linv @ sol["ss"]) ** 2) + np.sum((Linv @ sol["ds"]) ** 2)))
        rows.append(dict(sigma=sg, chi2=chi2, prior_norm=rough))
    lc = pd.DataFrame(rows)
    corner, _, _ = model_utils.lcurve_corner(lc.sigma.values, np.sqrt(lc.chi2.values), lc.prior_norm.values)
    lc.to_csv(os.path.join(OUT, f"lcurve_{label}.csv"), index=False)
    print(f"  L-curve {label}: corner sigma {corner:.2f} ({(time.time() - t_start) / 60:.1f} min)", flush=True)
    return float(corner), lc


# ---- solver check: all tents free, fault-parallel only, at the rescore_dips corner -> the CSI inversion's numbers
ref = pd.read_csv(os.path.join(DATA, "dip_checks_2026-10-26", "rescore_dips.csv"))
ref = ref[(ref.geometry == "vertical") & (ref.family == "homogeneous analytic N^0")].iloc[0]
tmpl_all = np.asarray(tmpl.Cm)
Linv_all = solve_triangular(np.linalg.cholesky(tmpl_all), np.eye(t.n_imp), lower=True)
sol = solve(float(ref.sigma), ["fp"], ["ss"], Gd={("fp", "ss"): Ga}, Linv_=Linv_all)
res = data["fp"] - Ga @ sol["ss"] - plane @ sol["planes"]["fp"]
chk = dict(rms=float(np.sqrt(np.mean(res[keep] ** 2))), W500=float(np.sqrt(np.mean(res[w500] ** 2))),
           Mw=float(model_utils.moment_magnitude(MU * t.potency_m3(sol["ss"]))))
print(f"solver check (all tents, fp only, sigma {ref.sigma:.2f}): rms {chk['rms']:.4f} vs CSI {ref.rms_all_mm:.4f}; W500 {chk['W500']:.4f} "
      f"vs {ref.rms_W500_mm:.4f}; Mw {chk['Mw']:.4f} vs {ref.Mw_30gpa:.4f}", flush=True)
assert abs(chk["rms"] - ref.rms_all_mm) < 2e-3 and abs(chk["Mw"] - ref.Mw_30gpa) < 2e-3, chk
del sol, Linv_all, tmpl_all

# =====================================================================================================================
# 4. The three models, each at its own L-curve corner
# =====================================================================================================================
MODELS = [("A", "Strike-slip only, FP + vertical", ["fp", "up"], ["ss"]),
          ("B", "Strike-slip + dip-slip, FP + vertical", ["fp", "up"], ["ss", "ds"]),
          ("C", "Strike-slip only, FP alone", ["fp"], ["ss"])]
results = {}
for key, label, comps, dirs in MODELS:
    corner, lc = lcurve(comps, dirs, key)
    sol = solve(corner, comps, dirs)
    kkt_check(sol, key)
    r = score(sol, comps, corner, key)
    r["description"] = label
    results[key] = r
    print(f"done {key} ({label}): sigma {corner:.2f}, Mw30 ss {r['Mw30_ss']:.3f}, ds/ss potency {r['ds_over_ss_potency']:.3f}; "
          f"FP rms {r['fp_rms_mm']:.3f} (W {r['fp_rms_W500_mm']:.3f}, E {r['fp_rms_E500_mm']:.3f}); "
          f"UP rms {r['up_rms_mm']:.3f} (W {r['up_rms_W500_mm']:.3f}, E {r['up_rms_E500_mm']:.3f}); dof {r['dof_eff']:.1f}", flush=True)

# ---- does dip-slip buy more than its extra parameters? (B vs A, same data, same weights)
A_, B_ = results["A"], results["B"]
chi2_A = A_["fp_chi2"] + A_["up_chi2"]
chi2_B = B_["fp_chi2"] + B_["up_chi2"]
d_chi2, d_dof = chi2_A - chi2_B, B_["dof_eff"] - A_["dof_eff"]
N = A_["n_data_fitted"]
F = (d_chi2 / d_dof) / (chi2_B / (N - B_["dof_eff"]))
from scipy.stats import f as f_dist                   # noqa: E402
p_F = float(f_dist.sf(F, d_dof, N - B_["dof_eff"]))
compare = dict(chi2_A=chi2_A, chi2_B=chi2_B, d_chi2=d_chi2, dof_A=A_["dof_eff"], dof_B=B_["dof_eff"], d_dof=d_dof, N=N, F=F, p_F=p_F)
print(f"B vs A: chi2 {chi2_A:.0f} -> {chi2_B:.0f} (drop {d_chi2:.0f}), effective parameters {A_['dof_eff']:.1f} -> {B_['dof_eff']:.1f} "
      f"(+{d_dof:.1f}); F = {F:.2f}, p = {p_F:.2g} (independent-noise assumption)", flush=True)

# ---- is the vertical signal B fits fault-related? Variance of the vertical data (B's plane removed) and of B's vertical
#      prediction, near (<= 0.5 km) vs far (> 1 km), in the SSE section
dataB_up = data["up"] - B_["_up_plane"]
near = keep & in_section & (np.abs(d_km) <= 0.5)
far = keep & in_section & (np.abs(d_km) > 1.0)
vert = dict(up_data_rms_near_mm=float(np.sqrt(np.mean(dataB_up[near] ** 2))), up_data_rms_far_mm=float(np.sqrt(np.mean(dataB_up[far] ** 2))),
            up_predB_rms_near_mm=float(np.sqrt(np.mean(B_["_up_model"][near] ** 2))),
            up_predB_rms_far_mm=float(np.sqrt(np.mean(B_["_up_model"][far] ** 2))),
            up_data_step_EminusW_500_mm=float(np.mean(dataB_up[e500]) - np.mean(dataB_up[w500])),
            up_predB_step_EminusW_500_mm=float(np.mean(B_["_up_model"][e500]) - np.mean(B_["_up_model"][w500])),
            up_data_step_EminusW_100_mm=float(np.mean(dataB_up[keep & in_section & (d_km > 0) & (d_km <= 0.15)])
                                              - np.mean(dataB_up[keep & in_section & (d_km < 0) & (d_km >= -0.15)])),
            up_predB_step_EminusW_100_mm=float(np.mean(B_["_up_model"][keep & in_section & (d_km > 0) & (d_km <= 0.15)])
                                               - np.mean(B_["_up_model"][keep & in_section & (d_km < 0) & (d_km >= -0.15)])))
for k_, v_ in vert.items():
    print(f"  {k_}: {v_:+.3f}")

# ---- 4b (added after run 2): is the dip-slip gain more than dip-slip buys on NOISE with the vertical field's own statistics?
# The F-test above assumes independent noise (the nugget); the vertical residual is dominated by the correlated part of
# the semivariogram (0.80 mm over ~1 km), which dip-slip tents can also fit. Calibrated test, vertical data alone, at B's
# sigma: chi2 drop (equal nugget weights, fitted samples) of [dip-slip + plane] over [plane only], for (a) the real vertical
# data minus B's strike-slip vertical prediction, (b) N_MC realisations of Gaussian noise with the vertical semivariogram
# (cov_fit["up"], nugget on every sample = N^0, correlated part at the block centres; model_utils.covariance_matrix).
# The dip-slip fit is linear here (dip-slip free, prior Cm, plane free), so one operator serves every realisation.
N_MC = 500
sig_B = float(B_["sigma"])
Aw = np.vstack([np.c_[G[("up", "ds")][keep] / nug["up"], plane[keep] / nug["up"]], np.c_[Linv / sig_B, np.zeros((nf, 3))]])
Kop = np.linalg.solve(Aw.T @ Aw, Aw[:keep.sum()].T)                 # x = Kop @ (d_w) (the prior rows have zero data)
Gw = Aw[:keep.sum()]
Pw = plane[keep] / nug["up"]
Kp = np.linalg.pinv(Pw)


def chi2_drop(dw):
    """chi2 (plane only) - chi2 (dip-slip + plane), data part only; dw = weighted vertical data on the fitted samples."""
    r_plane = dw - Pw @ (Kp @ dw)
    r_ds = dw - Gw @ (Kop @ dw)
    return np.sum(r_plane ** 2, 0) - np.sum(r_ds ** 2, 0)


d_real = (data["up"] - G[("up", "ss")] @ results["B"]["_ss"][cols])[keep] / nug["up"]
drop_real = float(chi2_drop(d_real))
Cd_up = model_utils.covariance_matrix(p["x_km"][keep], p["y_km"][keep], np.ones(keep.sum()), cov_fit["up"], 0.0)
noise = np.linalg.cholesky(Cd_up) @ np.random.default_rng(30).standard_normal((keep.sum(), N_MC)) / nug["up"]
drop_mc = chi2_drop(noise)
# the same for the near-trace east-minus-west vertical step (60-500 m) in the SSE section
e_k, w_k = e500[keep], w500[keep]
step_mc = (noise[e_k].mean(0) - noise[w_k].mean(0)) * nug["up"]
mc = dict(N_MC=N_MC, mc_sigma=sig_B, drop_real=drop_real, drop_mc_median=float(np.median(drop_mc)), drop_mc_p95=float(np.percentile(drop_mc, 95)),
          drop_mc_max=float(drop_mc.max()), drop_real_rank=float(np.mean(drop_mc >= drop_real)),
          step_real_mm=vert["up_data_step_EminusW_500_mm"], step_mc_sd_mm=float(np.std(step_mc)),
          step_real_rank=float(np.mean(np.abs(step_mc) >= abs(vert["up_data_step_EminusW_500_mm"]))))
print(f"noise test (vertical alone, sigma {sig_B:.1f}, {N_MC} realisations of the vertical semivariogram): chi2 drop from dip-slip "
      f"real {drop_real:.0f}; noise median {mc['drop_mc_median']:.0f}, 95 % {mc['drop_mc_p95']:.0f}, max {mc['drop_mc_max']:.0f}; "
      f"fraction of noise >= real {mc['drop_real_rank']:.3f}", flush=True)
print(f"  E-W vertical step (60-500 m, SSE section): real {mc['step_real_mm']:+.2f} mm, noise sd {mc['step_mc_sd_mm']:.2f} mm, "
      f"fraction |noise| >= |real| {mc['step_real_rank']:.3f}", flush=True)
# ---- 4c: does the vertical step follow the fault-parallel step along strike (decomposition leakage / fault-related) or not?
fp_c = data["fp"] - results["C"]["_fp_plane"]
bins_km = np.arange(56.5, 62.01, 0.5)
along = []
for lo in bins_km:
    m_e = e500 & (km_pp >= lo) & (km_pp < lo + 0.5)
    m_w = w500 & (km_pp >= lo) & (km_pp < lo + 0.5)
    if m_e.sum() >= 5 and m_w.sum() >= 5:
        along.append(dict(km=lo + 0.25, up_step_mm=float(np.mean(dataB_up[m_e]) - np.mean(dataB_up[m_w])),
                          fp_step_mm=float(np.mean(fp_c[m_e]) - np.mean(fp_c[m_w])),
                          upB_step_mm=float(np.mean(B_["_up_model"][m_e]) - np.mean(B_["_up_model"][m_w]))))
along = pd.DataFrame(along)
along.to_csv(os.path.join(OUT, "dipslip_alongstrike_steps.csv"), index=False)
mc["r_upstep_fpstep_alongstrike"] = float(np.corrcoef(along.up_step_mm, along.fp_step_mm)[0, 1])
mc["up_step_alongstrike_min_mm"], mc["up_step_alongstrike_max_mm"] = float(along.up_step_mm.min()), float(along.up_step_mm.max())
print(along.round(2).to_string(index=False))
print(f"  along-strike correlation of the vertical and fault-parallel steps: r = {mc['r_upstep_fpstep_alongstrike']:+.2f}", flush=True)

# ---- tables
table = pd.DataFrame([{k_: v_ for k_, v_ in r.items() if not k_.startswith("_")} for r in results.values()])
table["nugget_fp_mm"], table["nugget_up_mm"] = nug["fp"], nug["up"]
table["n_free_tents"] = nf
table["provenance"] = (f"dipslip_test.py {time.strftime('%Y-%m-%d')}: homogeneous half-space (Meade/CSI analytic tents of {RUN}, n_sub {N_SUB}); "
                       f"free tents depth <= {FREE_DEPTH_KM} km, km {FREE_KM[0]:g}-{FREE_KM[1]:g}; 60 m cut; equal weights = 5 km/15 km "
                       "semivariogram nugget per component; plane per dataset jointly; buildCmXY lam (0.1, 0.05) km, same prior for "
                       "dip-slip; own L-curve corners on the 3D sigma grid; no surface constraint; dip-slip + = east side up")
table.to_csv(os.path.join(OUT, "dipslip_results.csv"), index=False)
pd.DataFrame([dict(**compare, **vert, **mc, nugget_fp_mm=nug["fp"], nugget_up_mm=nug["up"], centre_mismatch_m=centre_mismatch_m,
                   G_check_rel=rel_check, step_ds_csi=step_ds_csi, DS_SIGN=DS_SIGN, solver_check_rms=chk["rms"],
                   solver_check_ref_rms=float(ref.rms_all_mm), solver_check_Mw=chk["Mw"], solver_check_ref_Mw=float(ref.Mw_30gpa))]
             ).to_csv(os.path.join(OUT, "dipslip_checks.csv"), index=False)
np.savez(os.path.join(OUT, "dipslip_models.npz"), **{f"{k_}_{v_}": results[k_][f"_{v_}"] for k_ in results
                                                     for v_ in ("ss", "ds", "fp_model", "up_model", "fp_res", "up_res", "fp_plane", "up_plane")},
         data_fp=data["fp"], data_up=data["up"], keep=keep, d_km=d_km, km_pp=km_pp, imp_km=t.imp_km, imp_depth_km=t.imp_depth_km,
         sub_faces=t.sub_faces, cols=cols)
pd.set_option("display.width", 250)
print(table.drop(columns=["provenance"]).round(3).T.to_string())

# =====================================================================================================================
# 5. Draft figure (slide fonts, Okabe-Ito, no blue against green)
# =====================================================================================================================
plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11})
fig = plt.figure(figsize=(18, 10.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
tri = mtri.Triangulation(t.imp_km, t.imp_depth_km, triangles=t.sub_faces)
ss_max = max(results["A"]["_ss"].max(), results["B"]["_ss"].max())
ds_lim = max(np.abs(results["B"]["_ds"]).max(), 1e-3)
for i, (key, comp, title) in enumerate([("A", "_ss", "A  Strike-slip, model A (no dip-slip)"),
                                        ("B", "_ss", "B  Strike-slip, model B"),
                                        ("B", "_ds", "C  Dip-slip, model B (+ = east side up)")]):
    ax = fig.add_subplot(gs[0, i])
    v = results[key][comp]
    if comp == "_ss":
        im = ax.tripcolor(tri, v, shading="gouraud", cmap="magma_r", vmin=0, vmax=ss_max)
        lab = "Right-lateral slip (mm)"
    else:
        im = ax.tripcolor(tri, v, shading="gouraud", cmap="PuOr_r", vmin=-ds_lim, vmax=ds_lim)
        lab = "Dip-slip (mm)"
    ax.set_xlim(FREE_KM); ax.set_ylim(FREE_DEPTH_KM, 0)
    ax.axvline(creep_lo_km, color="0.4", ls="--", lw=1); ax.axvline(creep_hi_km, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Along strike from Point Pinole (km)"); ax.set_ylabel("Depth (km)")
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, orientation="horizontal", label=lab, shrink=0.85, pad=0.02)
# RMS bars
ax = fig.add_subplot(gs[1, 0])
groups = [("fp_rms_mm", "FP all"), ("fp_rms_W500_mm", "FP W500"), ("fp_rms_E500_mm", "FP E500"),
          ("up_rms_mm", "Up all"), ("up_rms_W500_mm", "Up W500"), ("up_rms_E500_mm", "Up E500")]
colors = {"A": "#0072B2", "B": "#D55E00", "C": "#E69F00"}
xg = np.arange(len(groups))
for j, key in enumerate(["A", "B", "C"]):
    vals = [results[key][g] for g, _ in groups]
    hatch = [None if results[key][f"{g[:2]}_fitted"] else "//" for g, _ in groups]
    bars = ax.bar(xg + (j - 1) * 0.27, vals, 0.27, color=colors[key], label=f"{key}: {results[key]['description']}")
    for bar_, h in zip(bars, hatch):
        if h:
            bar_.set_hatch(h); bar_.set_facecolor("white"); bar_.set_edgecolor(colors[key])
ax.axhline(nug["fp"], color="0.3", ls=":", lw=1.5, xmax=0.5)
ax.axhline(nug["up"], color="0.3", ls="--", lw=1.5, xmin=0.5)
ax.set_xticks(xg); ax.set_xticklabels([g[1] for g in groups], rotation=30, ha="right")
ax.set_ylabel("RMS residual (mm)")
ax.set_title("D  Misfit (lines: nuggets; hatched: not fitted)", loc="left", fontsize=13)
ax.legend(loc="upper left", fontsize=10, frameon=False)
ax.set_ylim(0, 1.45 * max(results[k_][g] for k_ in results for g, _ in groups))
# vertical across-fault profile in the SSE section
ax = fig.add_subplot(gs[1, 1])
edges = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.35, -0.25, -0.15, -0.06, 0.06, 0.15, 0.25, 0.35, 0.5, 0.75, 1, 1.5, 2, 3])
sel = keep & in_section
cen, med_d, med_a, med_b, q1, q3 = [], [], [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m_ = sel & (d_km >= lo) & (d_km < hi)
    if m_.sum() < 3:
        continue
    cen.append(0.5 * (lo + hi)); med_d.append(np.median(dataB_up[m_])); q1.append(np.percentile(dataB_up[m_], 25))
    q3.append(np.percentile(dataB_up[m_], 75))
    med_b.append(np.median(B_["_up_model"][m_])); med_a.append(np.median(A_["_up_model"][m_]))
ax.fill_between(cen, q1, q3, color="0.85", lw=0, label="Data, 25-75 %")
ax.plot(cen, med_d, "o-", color="k", lw=2, label="Data (B's plane removed), median")
ax.plot(cen, med_b, "s-", color=colors["B"], lw=2, label="Model B")
ax.plot(cen, med_a, "^--", color=colors["A"], lw=2, label="Model A")
ax.axvline(0, color="0.5", lw=1); ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlabel("Distance from trace, west - / east + (km)"); ax.set_ylabel("Vertical (mm)")
ax.set_title(f"E  Vertical across the fault, km {creep_lo_km:.1f}-{creep_hi_km:.1f}", loc="left")
ax.legend(loc="best", fontsize=10, frameon=False)
# vertical data map (B's plane removed) on the quadtree blocks
ax = fig.add_subplot(gs[1, 2])
polys = [np.c_[p["block_lon"][k_], p["block_lat"][k_]] for k_ in range(ns)]
vlim = np.percentile(np.abs(dataB_up[keep]), 98)
pc = PolyCollection(polys, array=dataB_up, cmap="PuOr_r", clim=(-vlim, vlim), edgecolors="none")
ax.add_collection(pc)
ax.plot(trace_south_north[:, 0], trace_south_north[:, 1], color="k", lw=1)
ax.set_xlim(p["block_lon"].min(), p["block_lon"].max()); ax.set_ylim(p["block_lat"].min(), p["block_lat"].max())
ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(p["lat"]))))
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.tick_params(labelsize=10)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
ax.set_title("F  Vertical data, B's plane removed", loc="left")
fig.colorbar(pc, ax=ax, label="Vertical (mm)", shrink=0.85)
fig.suptitle(f"DRAFT  Dip-slip test, homogeneous half-space. Vertical chi$^2$ gain from dip-slip: data {mc['drop_real']:.0f}, "
             f"correlated noise median {mc['drop_mc_median']:.0f} (95 % {mc['drop_mc_p95']:.0f}). Nuggets FP {nug['fp']:.2f}, vertical {nug['up']:.2f} mm",
             fontsize=14)
fig.savefig(os.path.join(OUT, "dipslip_draft.png"), dpi=150, facecolor="white")
print(f"wrote {OUT} in {(time.time() - t_start) / 60:.1f} min")
