#!/usr/bin/env python3
"""
DRAFT (2026-09-29): does the USGS SFCVM v26 rigidity structure explain the
west/east asymmetry of the Tule Pond mean event? (Dani: the homogeneous model
leaves a west near-field residual; route A before any PyLith work.)

2D antiplane cross-section across the fault. For strike-slip on a vertical
fault the fault-parallel displacement u(x, z) obeys
    d/dx(mu du/dx) + d/dz(mu du/dz) = 0
with a free surface (du/dz = 0 at z = 0), u -> 0 far away, and a jump
u_E - u_W = s(z) across the fault where it slips. Solved by finite volumes on a
graded grid (10 m cells at the fault and surface, growing to 60 km), the fault
on a cell face; the jump enters as a source on the two cells either side. One
solve per depth patch gives the 2D Green's functions (splu, all patches at once).

mu(x, z): SFCVM v26 sampled by drafts/07/sfcvm_extract.py
(outputs/07_slip_model_tests/sfcvm_creep_section_profiles_v26-0.csv): median
over 57.5-61.5 km along strike, per side, at offsets 0.2, 0.5, 1.5, 3 km and
depths 0-3 km; held constant beyond those (|x| < 0.2 km takes the 0.2 km value).

Data: every pixel of 06's fault-parallel mean event within +/-3 km of the
trace in the 90% band (58.2-59.8 km), > 30 m from the trace (as the 3D model),
one free constant. Slip is non-negative, second-difference smoothed, the
smoothing weight at the corner of each medium's own L-curve.

CHECK first: with uniform mu the solver must reproduce the analytic screw
dislocation u(x) = (s/pi)[atan(z2/x) - atan(z1/x)] (x > 0; odd in x).

Output: outputs/07_slip_model_tests/antiplane_sfcvm_2d.csv,
        Figures_<date>/test_antiplane_sfcvm_2d.png
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import nnls
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import pyproj
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
import insar_utils as utils
from TulePond_filepaths import common_paths

out_dir = os.path.join(common_paths["out_dir"], "07_slip_model_tests")
fig_dir = common_paths["fig_dir"]
band_km = (58.2, 59.8)          # 04's 90% band
data_half_km = 3.0
exclude_m = 30.0

# ---------------------------------------------------------------- grid
def graded(d0, d_fine_to, growth, extent):
    """Cell widths: d0 until d_fine_to, then growing by `growth` to `extent` (m)."""
    w = []
    while sum(w) < d_fine_to:
        w.append(d0)
    while sum(w) < extent:
        w.append(w[-1] * growth)
    return np.array(w)

wx = graded(10.0, 400.0, 1.08, 60e3)
x_edges = np.r_[-np.cumsum(wx)[::-1], 0.0, np.cumsum(wx)]          # fault on the face x = 0
wz = graded(10.0, 1500.0, 1.08, 60e3)
z_edges = np.r_[0.0, np.cumsum(wz)]
xc, zc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (z_edges[1:] + z_edges[:-1])
dx, dz = np.diff(x_edges), np.diff(z_edges)
nx, nz = xc.size, zc.size
i_fault = int(np.searchsorted(xc, 0.0))           # first east cell; the fault is between i_fault-1 and i_fault
print(f"grid {nx} x {nz} = {nx * nz} cells")


def assemble(mu):
    """Finite-volume operator for d(mu du) = 0; mu is (nz, nx) at cell centres.
    Far x and bottom: u = 0 on the boundary face; top: zero flux."""
    idx = np.arange(nx * nz).reshape(nz, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * nz)
    # x faces between neighbours
    dxc = xc[1:] - xc[:-1]
    t_x = (2 * mu[:, 1:] * mu[:, :-1] / (mu[:, 1:] + mu[:, :-1])) * dz[:, None] / dxc[None, :]
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    rows += [a, b]; cols += [b, a]; vals += [t_x.ravel(), t_x.ravel()]
    np.add.at(diag, a, -t_x.ravel()); np.add.at(diag, b, -t_x.ravel())
    # z faces
    dzc = zc[1:] - zc[:-1]
    t_z = (2 * mu[1:, :] * mu[:-1, :] / (mu[1:, :] + mu[:-1, :])) * dx[None, :] / dzc[:, None]
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    rows += [a, b]; cols += [b, a]; vals += [t_z.ravel(), t_z.ravel()]
    np.add.at(diag, a, -t_z.ravel()); np.add.at(diag, b, -t_z.ravel())
    # Dirichlet u = 0 on the left, right and bottom boundary faces (half-cell distance)
    np.add.at(diag, idx[:, 0], -mu[:, 0] * dz / (0.5 * dx[0]))
    np.add.at(diag, idx[:, -1], -mu[:, -1] * dz / (0.5 * dx[-1]))
    np.add.at(diag, idx[-1, :], -mu[-1, :] * dx / (0.5 * dz[-1]))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * nz,) * 2).tocsr()
    A = A + sp.diags(diag)
    t_fault = t_x[:, i_fault - 1]                     # transmissibility of the fault faces, per depth row
    return A.tocsc(), t_fault, idx


def greens(mu, patches):
    """Surface displacement (top-row cell centres) for unit slip on each depth patch."""
    A, t_fault, idx = assemble(mu)
    lu = splu(A)
    B = np.zeros((nx * nz, len(patches)))
    for k, (z1, z2) in enumerate(patches):
        rows_k = np.where((zc > z1) & (zc < z2))[0]
        B[idx[rows_k, i_fault - 1], k] = t_fault[rows_k]        # west cell: +T s moved to the right-hand side
        B[idx[rows_k, i_fault], k] = -t_fault[rows_k]           # east cell
    U = lu.solve(B)                                           # A u = b: +T s on the west cell, -T s on the east
    return U.reshape(nz, nx, -1)[0]                           # (nx, n_patches) at the surface


def sample(surface, x_m):
    """Surface values at x (m), each side from its own cells so the jump is kept."""
    out = np.empty((x_m.size, surface.shape[1]))
    west, east = x_m < 0, x_m >= 0
    for k in range(surface.shape[1]):
        out[west, k] = np.interp(x_m[west], xc[:i_fault], surface[:i_fault, k])
        out[east, k] = np.interp(x_m[east], xc[i_fault:], surface[i_fault:, k])
    return out

# ---------------------------------------------------------------- CHECK: uniform mu against the analytic solution
mu_uniform = np.full((nz, nx), 30.0)
test_patches = [(0.0, 1000.0), (200.0, 800.0)]
g_test = greens(mu_uniform, test_patches)
x_chk = np.r_[-np.geomspace(3000, 30, 40), np.geomspace(30, 3000, 40)]
g_chk = sample(g_test, x_chk)
for k, (z1, z2) in enumerate(test_patches):
    analytic = (np.arctan(z2 / x_chk) - np.arctan(z1 / x_chk)) / np.pi
    err = np.max(np.abs(g_chk[:, k] - analytic))
    print(f"CHECK uniform, slip {z1:.0f}-{z2:.0f} m: max |FV - analytic| = {err:.4f} (per unit slip)")
    assert err < 0.02, "finite-volume solver does not reproduce the analytic screw dislocation"

# ---------------------------------------------------------------- mu from SFCVM v26
sfcvm = pd.read_csv(os.path.join(out_dir, "sfcvm_creep_section_profiles_v26-0.csv"))
sfcvm = sfcvm[(sfcvm.km >= 57.5) & (sfcvm.km <= 61.5)]
mu_sfcvm = np.empty((nz, nx))
for side, sel_x in (("west", xc < 0), ("east", xc > 0)):
    part = sfcvm[sfcvm.offset_km < 0] if side == "west" else sfcvm[sfcvm.offset_km > 0]
    tab = part.assign(off=part.offset_km.abs()).groupby(["off", "depth_m"]).mu_gpa.median().unstack()
    interp = RegularGridInterpolator((tab.index.values * 1e3, tab.columns.values.astype(float)), tab.values)
    X, Z = np.meshgrid(np.clip(np.abs(xc[sel_x]), tab.index.min() * 1e3, tab.index.max() * 1e3),
                       np.clip(zc, 0, float(tab.columns.max())))
    mu_sfcvm[:, sel_x] = interp(np.c_[X.ravel(), Z.ravel()]).reshape(X.shape)
    print(f"{side}: mu at 5 m {np.median(mu_sfcvm[0, sel_x][np.abs(xc[sel_x]) < 1000]):.2f} GPa, "
          f"at 500 m {np.median(mu_sfcvm[np.argmin(np.abs(zc - 500)), sel_x][np.abs(xc[sel_x]) < 1000]):.2f} GPa")

# ---------------------------------------------------------------- data: the band's fault-parallel mean event
f2 = np.load(os.path.join(common_paths["out_dir"], "06_event_field_geometry", "fig2_inputs.npz"))
prof = pd.read_csv(common_paths["fault_profiles"])
to_utm = pyproj.Transformer.from_crs(4326, int(f2["epsg"]), always_xy=True)
fx, fy = to_utm.transform(prof.ft_lon.values, prof.ft_lat.values)
tx, ty = np.gradient(fx), np.gradient(fy)
nrm = np.c_[ty, -tx] / np.hypot(tx, ty)[:, None]         # points east (right of a northward trace)
km_of = utils.lat_to_km_map(common_paths["fault_profiles"])
X_, Y_, V_ = f2["X"].ravel(), f2["Y"].ravel(), f2["fp_event"].ravel()
ok = np.isfinite(V_)
X_, Y_, V_ = X_[ok], Y_[ok], V_[ok]
dist, j = cKDTree(np.c_[fx, fy]).query(np.c_[X_, Y_])
signed = np.sum((np.c_[X_, Y_] - np.c_[fx, fy][j]) * nrm[j], axis=1)
km_pix = km_of(prof.ft_lat.values[j])
keep = (km_pix >= band_km[0]) & (km_pix <= band_km[1]) & (np.abs(signed) <= data_half_km * 1e3) & (np.abs(signed) > exclude_m)
x_data, d_data = signed[keep], V_[keep]
assert np.median(d_data[(x_data < 0) & (x_data > -300)]) < 0 < np.median(d_data[(x_data > 0) & (x_data < 300)]), \
    "sign convention: right-lateral should be negative west, positive east"
print(f"data: {keep.sum()} pixels in {band_km} km, |x| {exclude_m:.0f} m to {data_half_km:g} km")

# ---------------------------------------------------------------- inversion, per medium
patches = [(z, z + 50.0) for z in np.arange(0, 1000, 50)] + [(z, z + 250.0) for z in np.arange(1000, 3000, 250)] \
          + [(z, z + 1000.0) for z in np.arange(3000, 6000, 1000)]
n_p = len(patches)
L = np.zeros((n_p - 2, n_p))
for k in range(n_p - 2):
    L[k, k:k + 3] = [1, -2, 1]
alphas = np.geomspace(0.3, 300, 25)


def invert(G, alpha):
    """NNLS on slip, one free constant (split into +/- columns so NNLS allows either sign)."""
    Gc = np.c_[G, np.ones(G.shape[0]), -np.ones(G.shape[0])]
    Lc = np.c_[L, np.zeros((L.shape[0], 2))]
    m, _ = nnls(np.r_[Gc, alpha * Lc], np.r_[d_data, np.zeros(L.shape[0])], maxiter=20000)
    return m[:n_p], m[n_p] - m[n_p + 1]


def corner(G):
    rows = []
    for a in alphas:
        s_, c_ = invert(G, a)
        rows.append((a, np.linalg.norm(G @ s_ + c_ - d_data), np.linalg.norm(L @ s_)))
    t = np.array(rows)
    lx, ly = np.log(t[:, 1]), np.log(np.maximum(t[:, 2], 1e-9))
    d1x, d1y, d2x, d2y = np.gradient(lx), np.gradient(ly), np.gradient(np.gradient(lx)), np.gradient(np.gradient(ly))
    curv = np.abs(d1x * d2y - d1y * d2x) / (d1x ** 2 + d1y ** 2) ** 1.5
    return float(t[np.nanargmax(curv[1:-1]) + 1, 0])


results, curves = [], {}
x_plot = np.r_[np.linspace(-3000, -5, 600), np.linspace(5, 3000, 600)]
for name, mu in (("Homogeneous", mu_uniform), ("SFCVM v26", mu_sfcvm)):
    surf = greens(mu, patches)
    G = sample(surf, x_data)
    a_sel = corner(G)
    slip, const = invert(G, a_sel)
    pred = G @ slip + const
    resid = d_data - pred
    near = np.abs(x_data) <= 300
    row = dict(medium=name, alpha=a_sel, rms_mm=np.sqrt(np.mean(resid ** 2)),
               rms_west_300m_mm=np.sqrt(np.mean(resid[near & (x_data < 0)] ** 2)),
               rms_east_300m_mm=np.sqrt(np.mean(resid[near & (x_data > 0)] ** 2)),
               surface_slip_mm=slip[0], peak_slip_mm=slip.max(),
               potency_per_m=float(np.sum(slip * np.diff(np.array(patches), axis=1).ravel()) / 1e3),   # mm*m -> m*m (per m along strike)
               depth_90pct_m=float(np.array(patches)[np.searchsorted(np.cumsum(slip * np.diff(np.array(patches), axis=1).ravel())
                                                                    / np.sum(slip * np.diff(np.array(patches), axis=1).ravel()), 0.9), 1]))
    results.append(row)
    curves[name] = dict(slip=slip, pred_plot=sample(surf, x_plot) @ slip + const, resid=resid)
    print(f"{name}: alpha {a_sel:.2g}; rms {row['rms_mm']:.2f} mm (west<=300 m {row['rms_west_300m_mm']:.2f}, "
          f"east {row['rms_east_300m_mm']:.2f}); surface slip {slip[0]:.1f} mm, 90% of potency above {row['depth_90pct_m']:.0f} m")
summary = pd.DataFrame(results)
summary.to_csv(os.path.join(out_dir, "antiplane_sfcvm_2d.csv"), index=False)
print(summary.round(3).to_string(index=False))

# ---------------------------------------------------------------- figure
utils.mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True, width_ratios=[1.6, 1.0, 1.0])
colours = {"Homogeneous": "#0072B2", "SFCVM v26": "#D55E00"}
ax = axes[0]
bins = np.linspace(-3000, 3000, 121)
idx = np.digitize(x_data, bins)
med = np.array([np.median(d_data[idx == k]) if (idx == k).sum() > 5 else np.nan for k in range(1, bins.size)])
ax.plot(x_data, d_data, ".", ms=0.8, color="0.75", rasterized=True)
ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "-", color="0.2", lw=1.0, label="Data, 50 m median")
for name, c in curves.items():
    ax.plot(x_plot, c["pred_plot"], "-", color=colours[name], lw=1.1, label=name)
ax.axvline(0, color="0.5", lw=0.5)
ax.set_xlim(-3000, 3000); ax.set_ylim(-9, 6)
ax.set_xlabel("Distance from the trace (m, + east)"); ax.set_ylabel("Fault-parallel displacement (mm)")
ax.legend(loc="lower right")
utils.panel_title(ax, "A", f"Mean event, {band_km[0]}-{band_km[1]} km, 2D antiplane fits")
ax = axes[1]
zmid = np.array([0.5 * (a + b) for a, b in patches])
for name, c in curves.items():
    ax.step(np.r_[c["slip"], c["slip"][-1]], np.r_[[p[0] for p in patches], patches[-1][1]] / 1e3, where="post", color=colours[name], lw=1.1, label=name)
ax.set_ylim(2.0, 0); ax.set_xlabel("Slip (mm)"); ax.set_ylabel("Depth (km)")
utils.panel_title(ax, "B", "Slip with depth")
ax = axes[2]
for side, sel_x in (("West", (xc < 0) & (xc > -1000)), ("East", (xc > 0) & (xc < 1000))):
    ax.plot(np.median(mu_sfcvm[:, sel_x], axis=1), zc / 1e3, "-" if side == "West" else "--", color="#D55E00", lw=1.1, label=f"SFCVM {side.lower()}")
ax.axvline(30, color="#0072B2", lw=1.1, label="Homogeneous (30 GPa)")
ax.set_xscale("log"); ax.set_ylim(2.0, 0); ax.set_xlim(0.02, 60)
ax.set_xlabel("Rigidity (GPa)"); ax.set_ylabel("Depth (km)")
ax.legend(loc="lower left")
utils.panel_title(ax, "C", "Rigidity within 1 km of the trace")
f_fig = os.path.join(fig_dir, "test_antiplane_sfcvm_2d.png")
fig.savefig(f_fig, dpi=300, facecolor="white")
print("wrote", f_fig)
