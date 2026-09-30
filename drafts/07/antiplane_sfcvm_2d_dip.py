#!/usr/bin/env python3
"""
DRAFT (2026-09-29): the aligned SFCVM structure with a DIPPING fault that
follows the model's own fault-block boundary, and the west-sediment rigidity
alternative to thickening it (Dani).

- Dip: SFCVM's Hayward fault (block 22 | 7 boundary), aligned on the trace at
  the surface, steps east with depth by ~0.44 m per m in 58.2-59.8 km, a dip
  of ~66 deg E (relocated seismicity north of the section: 75-80 deg E). Cases
  90, 75 and 66 deg. The fault in the 2D solver is the staircase of cell faces
  between the cells east and west of x_f(z) = z / tan(dip); the slip jump is
  applied across every such face (vertical and horizontal), at the face's depth.
- Sediment: either its base moved +100 m (the previous draft), or its rigidity
  scaled by f (thickness unchanged): west cells with mu < 1 GPa in the top
  500 m (the Quaternary unit of the aligned model).

CHECK: a dipping fault from the surface to depth D in a uniform half-space
must give u(x) = (s/pi)[pi/2 sgn(x) - atan((x - D/tan(dip)) / D)].

Output: outputs/07_slip_model_tests/antiplane_sfcvm_2d_dip.csv,
        Figures_<date>/test_antiplane_sfcvm_2d_dip.png
"""
import os

source = open(os.path.join("drafts", "07", "antiplane_sfcvm_2d_aligned.py")).read()
exec(source.split("# ---------------------------------------------------------------- 2. fits")[0])


def greens_dip(mu, patches, dip_deg):
    """Surface Green's functions for a fault dipping east at dip_deg, from the
    surface trace at x = 0; unit slip on each depth patch."""
    A, _, idx = assemble(mu)
    lu = splu(A)
    x_fault = zc / np.tan(np.deg2rad(dip_deg)) if dip_deg < 89.99 else np.zeros_like(zc)
    east = xc[None, :] > x_fault[:, None]                     # (nz, nx) side of every cell
    dxc, dzc = xc[1:] - xc[:-1], zc[1:] - zc[:-1]
    t_x = (2 * mu[:, 1:] * mu[:, :-1] / (mu[:, 1:] + mu[:, :-1])) * dz[:, None] / dxc[None, :]
    t_z = (2 * mu[1:, :] * mu[:-1, :] / (mu[1:, :] + mu[:-1, :])) * dx[None, :] / dzc[:, None]
    B = np.zeros((nx * nz, len(patches)))
    for k, (z1, z2) in enumerate(patches):
        # vertical faces: west cell on the left, east cell on the right, at the row depth
        jj, ii = np.where(~east[:, :-1] & east[:, 1:] & (zc[:, None] > z1) & (zc[:, None] < z2))
        np.add.at(B[:, k], idx[jj, ii], t_x[jj, ii]); np.add.at(B[:, k], idx[jj, ii + 1], -t_x[jj, ii])
        # horizontal faces of the staircase: east cell above, west cell below, at the face depth
        z_face = z_edges[1:-1]
        jj, ii = np.where(east[:-1, :] & ~east[1:, :] & (z_face[:, None] > z1) & (z_face[:, None] < z2))
        np.add.at(B[:, k], idx[jj + 1, ii], t_z[jj, ii]); np.add.at(B[:, k], idx[jj, ii], -t_z[jj, ii])
    U = lu.solve(B)
    return U.reshape(nz, nx, -1)[0]

# ---------------------------------------------------------------- CHECK: dipping fault, uniform mu, against the analytic solution
for dip_chk in (66.0, 75.0):
    g_ = sample(greens_dip(mu_uniform, [(0.0, 800.0)], dip_chk), x_chk)[:, 0]
    xi = 800.0 / np.tan(np.deg2rad(dip_chk))
    analytic = (0.5 * np.pi * np.sign(x_chk) - np.arctan((x_chk - xi) / 800.0)) / np.pi
    err = np.max(np.abs(g_ - analytic))
    print(f"CHECK dip {dip_chk:g}: max |FV - analytic| = {err:.4f} (per unit slip)")
    assert err < 0.03, "dipping-fault solver does not reproduce the analytic solution"

# ---------------------------------------------------------------- fits
near = np.abs(x_data) <= 300
bins = np.linspace(-3000, 3000, 121)
idx_b = np.digitize(x_data, bins)
med = np.array([np.median(d_data[idx_b == k]) if (idx_b == k).sum() > 5 else np.nan for k in range(1, bins.size)])
tab_al = tables["SFCVM aligned"]
mu_al = mu_from(tab_al)
sediment = (xc[None, :] < 0) & (zc[:, None] < 500) & (mu_al < 1.0)


def fit(mu, dip_deg):
    surf = greens_dip(mu, patches, dip_deg)
    G = sample(surf, x_data)
    a_sel = corner(G)
    slip, const = invert(G, a_sel)
    resid = d_data - (G @ slip + const)
    widths = np.diff(np.array(patches), axis=1).ravel() / np.sin(np.deg2rad(dip_deg))    # down-dip width of each patch
    return dict(rms_mm=np.sqrt(np.mean(resid ** 2)), rms_west_300m_mm=np.sqrt(np.mean(resid[near & (x_data < 0)] ** 2)),
                rms_east_300m_mm=np.sqrt(np.mean(resid[near & (x_data > 0)] ** 2)), surface_slip_mm=slip[0],
                potency_per_m=float(np.sum(slip * widths) / 1e3)), dict(slip=slip, pred_plot=sample(surf, x_plot) @ slip + const, mu=mu)


rows, fields = [], {}
cases = [("Homogeneous, vertical", mu_uniform, 90.0, "", np.nan)]
for dip in (90.0, 75.0, 66.0):
    cases.append((f"Aligned SFCVM, dip {dip:g}", mu_al, dip, "none", 1.0))
for dip in (90.0, 66.0):
    cases.append((f"Aligned, dip {dip:g}, sediment base +100 m", mu_from(tab_al, 100.0), dip, "thickness", np.nan))
    for f in (0.25, 0.5, 0.75, 1.5, 2.0):
        cases.append((f"Aligned, dip {dip:g}, sediment mu x{f:g}", np.where(sediment, mu_al * f, mu_al), dip, "rigidity", f))
for name, mu, dip, change, f in cases:
    row, field = fit(mu, dip)
    rows.append(dict(case=name, dip_deg=dip, change=change, mu_factor=f, **row))
    fields[name] = field
    print(f"{name:42s} rms {row['rms_mm']:.3f} (W {row['rms_west_300m_mm']:.2f}, E {row['rms_east_300m_mm']:.2f}); "
          f"surface {row['surface_slip_mm']:.1f} mm; potency {row['potency_per_m']:.2f}")
result = pd.DataFrame(rows)
result.to_csv(os.path.join(out_dir, "antiplane_sfcvm_2d_dip.csv"), index=False)

# what the preferred rigidity factor means for the sediment's shear-wave speed (rho unchanged: Vs scales as sqrt(f))
sed_rows = dense[(dense.offset_aligned_km < -0.2) & (dense.offset_aligned_km > -1.0) & (dense.depth_m < 200) & dense.zone.str.startswith("QT")]
vs_med = float(sed_rows.vs.median())
for dip in (90.0, 66.0):
    r_ = result[(result.change.isin(["rigidity", "none"])) & (result.dip_deg == dip)].sort_values("rms_mm").iloc[0]
    print(f"dip {dip:g}: best sediment rigidity factor {r_.mu_factor:g} -> Vs x{np.sqrt(r_.mu_factor):.2f} "
          f"(SFCVM sediment Vs, top 200 m, median {vs_med:.0f} m/s -> {vs_med * np.sqrt(r_.mu_factor):.0f} m/s)")

# ---------------------------------------------------------------- figure
utils.mpl_paper_style(base=7)
fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[1.5, 1.0, 1.0])
shown = ["Homogeneous, vertical", "Aligned SFCVM, dip 90", "Aligned SFCVM, dip 66"]
colours = dict(zip(shown, ["#0072B2", "#E69F00", "#000000"]))
ax = fig.add_subplot(grid[0, 0])
ax.plot(x_data, d_data, ".", ms=0.8, color="0.82", rasterized=True)
ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "-", color="0.5", lw=1.0, label="Data, 50 m median")
for name in shown:
    ax.plot(x_plot, fields[name]["pred_plot"], "-", color=colours[name], lw=1.0, label=name)
ax.axvline(0, color="0.5", lw=0.5)
ax.set_xlim(-1500, 1500); ax.set_ylim(-9, 6)
ax.set_xlabel("Distance from the trace (m, + east)"); ax.set_ylabel("Fault-parallel displacement (mm)")
ax.legend(loc="upper left", fontsize=6)
utils.panel_title(ax, "A", f"Mean event, {band_km[0]}-{band_km[1]} km")
ax = fig.add_subplot(grid[0, 1])
for name in shown:
    s_ = fields[name]["slip"]
    ax.step(np.r_[s_, s_[-1]], np.r_[[p[0] for p in patches], patches[-1][1]] / 1e3, where="post", color=colours[name], lw=1.0)
ax.set_ylim(2.0, 0); ax.set_xlabel("Slip (mm)"); ax.set_ylabel("Depth (km)")
utils.panel_title(ax, "B", "Slip with depth")
ax = fig.add_subplot(grid[0, 2])
for dip, c, mk in ((90.0, "#E69F00", "s"), (66.0, "#000000", "o")):
    sub = result[(result.dip_deg == dip) & result.change.isin(["rigidity", "none"])].sort_values("mu_factor")
    ax.plot(sub.mu_factor, sub.rms_mm, mk + "-", color=c, ms=3, label=f"Dip {dip:g}")
ax.axhline(result.set_index("case").loc["Homogeneous, vertical", "rms_mm"], color="#0072B2", lw=0.8, label="Homogeneous")
ax.set_xscale("log"); ax.set_xticks([0.25, 0.5, 1, 2]); ax.set_xticklabels(["0.25", "0.5", "1", "2"]); ax.minorticks_off()
ax.set_xlabel("West sediment rigidity factor"); ax.set_ylabel("RMS misfit (mm)")
ax.legend(fontsize=6)
utils.panel_title(ax, "C", "Sediment rigidity instead of thickness")
from matplotlib.colors import LogNorm
for k, (name, dip) in enumerate((("Aligned SFCVM, dip 90", 90.0), ("Aligned SFCVM, dip 66", 66.0))):
    ax = fig.add_subplot(grid[1, k])
    im = ax.pcolormesh(xc, zc / 1e3, fields[name]["mu"], cmap="cividis", norm=LogNorm(0.03, 30), shading="nearest", rasterized=True)
    ax.plot(zc / np.tan(np.deg2rad(dip)) if dip < 90 else 0 * zc, zc / 1e3, "--", color="white", lw=1.0)
    ax.set_xlim(-1500, 1500); ax.set_ylim(1.0, 0)
    ax.set_xlabel("Distance from the trace (m, + east)")
    if k == 0:
        ax.set_ylabel("Depth (km)")
    utils.panel_title(ax, "DE"[k], f"Aligned SFCVM, fault dip {dip:g} (rigidity)")
fig.colorbar(im, ax=fig.axes[-2:], label="Rigidity (GPa)", shrink=0.9)
ax = fig.add_subplot(grid[1, 2])
tbl = result[result.case.isin(shown + ["Aligned, dip 66, sediment base +100 m"])][["case", "rms_mm", "rms_west_300m_mm", "rms_east_300m_mm", "surface_slip_mm", "potency_per_m"]]
ax.axis("off")
ax.table(cellText=[[r.case.replace("Aligned SFCVM", "SFCVM").replace("Aligned", "SFCVM").replace("Homogeneous, vertical", "Homogeneous"),
                    f"{r.rms_mm:.2f}", f"{r.rms_west_300m_mm:.2f}", f"{r.rms_east_300m_mm:.2f}", f"{r.surface_slip_mm:.1f}"] for r in tbl.itertuples()],
         colLabels=["Case", "RMS", "W", "E", "Surf."], loc="center", fontsize=5.5).auto_set_font_size(False)
utils.panel_title(ax, "F", "RMS (mm) and surface slip (mm)")
f_fig = os.path.join(fig_dir, "test_antiplane_sfcvm_2d_dip.png")
fig.savefig(f_fig, dpi=300, facecolor="white")
print("wrote", f_fig)
