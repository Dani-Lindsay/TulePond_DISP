#!/usr/bin/env python3
"""
DRAFT (2026-09-29): SFCVM v26 aligned on the creeping trace, then only the
west-side sediment thickness adjusted within the model's plausible
uncertainty (Dani: align the model's fault with the mapped trace BEFORE any
tweaking; change sediment depth by ~50-100 m, not rigidity, not 600 m).

1. Alignment (no fitting). In SFCVM the Hayward fault is the boundary between
   fault blocks 22 and 7; its material contrast sits on that boundary, which
   is 25-225 m east of the hand-mapped creeping trace at the surface and dips
   east. At every sampled km the structure is shifted so the block boundary
   at the surface lies on the trace (its dip is kept), then the median over
   the 90% band is taken.
2. Adjustment: the base of the west-side Quaternary sediment moved by
   dh_west in (-100, -50, 0, +50, +100) m. The rigidity values are unchanged;
   only the depth at which the sediment gives way to the stiffer rock moves
   (the sediment column is stretched or squeezed above its base, the rock
   below is shifted by dh). The east side (Valley Sequence rock at the surface,
   no sediment layer in SFCVM) is not changed.

Machinery, data and inversion: drafts/07/antiplane_sfcvm_2d.py (exec'd up to
its figure). Each case at its own L-curve corner.

Output: outputs/07_slip_model_tests/antiplane_sfcvm_2d_aligned.csv,
        Figures_<date>/test_antiplane_sfcvm_2d_aligned.png
"""
import os

source = open(os.path.join("drafts", "07", "antiplane_sfcvm_2d.py")).read()
exec(source.split("# ---------------------------------------------------------------- figure")[0])

dense = pd.read_csv(os.path.join(out_dir, "sfcvm_creep_section_dense_v26-0.csv"))
dense = dense[(dense.km >= band_km[0]) & (dense.km <= band_km[1] + 1e-6)].copy()

# ---------------------------------------------------------------- 1. align SFCVM's fault (block boundary) on the trace
boundary = {}
for km, g in dense[dense.depth_m == 0].groupby("km"):
    g = g.sort_values("offset_km")
    change = np.where(g.fault_block.values[1:] != g.fault_block.values[:-1])[0]
    near = [0.5 * (g.offset_km.values[i] + g.offset_km.values[i + 1]) for i in change if abs(g.offset_km.values[i]) < 1.0]
    assert len(near) == 1, f"{km} km: expected one fault-block boundary within 1 km, got {near}"
    boundary[km] = near[0]
dense["offset_aligned_km"] = np.round(dense.offset_km - dense.km.map(boundary), 3)
print("SFCVM block boundary at the surface, east of the trace (m):",
      ", ".join(f"{k:.2f} km {1e3 * v:+.0f}" for k, v in boundary.items()))


def table_of(frame, offset_col):
    tab = frame.groupby([offset_col, "depth_m"]).mu_gpa.median().unstack()
    tab = tab.loc[:, tab.notna().all()]                     # depths present at every offset
    tab = tab[tab.notna().all(axis=1)]
    return tab


tables = {"SFCVM v26 as is": table_of(dense, "offset_km"), "SFCVM aligned": table_of(dense, "offset_aligned_km")}

# west sediment base in the aligned model: deepest depth where the Quaternary unit is the majority, 0.2-1 km west
west = dense[(dense.offset_aligned_km < -0.2) & (dense.offset_aligned_km > -1.0)]
qt_frac = west.assign(qt=west.zone.str.startswith("QT")).groupby("depth_m").qt.mean()
sed_base_m = float(qt_frac[qt_frac > 0.5].index.max())
print(f"west Quaternary sediment base in the aligned model: {sed_base_m:.0f} m")


def mu_from(tab, dh_west=0.0):
    """mu on the solver grid from an (offset, depth) table; west of the trace the
    sediment base is moved by dh_west (rigidity values unchanged)."""
    off_m, dep_m = tab.index.values * 1e3, tab.columns.values.astype(float)
    interp = RegularGridInterpolator((off_m, dep_m), tab.values, method="nearest")
    z_model = np.tile(zc[:, None], (1, nx))
    if dh_west:
        new_base = sed_base_m + dh_west
        z_w = np.where(zc < new_base, zc * sed_base_m / new_base, zc - dh_west)
        z_model[:, xc < 0] = z_w[:, None]
    X = np.tile(np.clip(xc, off_m.min(), off_m.max())[None, :], (nz, 1))
    Z = np.clip(z_model, 0, dep_m.max())
    return interp(np.c_[X.ravel(), Z.ravel()]).reshape(nz, nx)

# ---------------------------------------------------------------- 2. fits
near = np.abs(x_data) <= 300
rows, fields = [], {}
cases = [("Homogeneous", None, 0.0), ("SFCVM v26 as is", "SFCVM v26 as is", 0.0), ("SFCVM aligned", "SFCVM aligned", 0.0)] \
        + [(f"Aligned, west sediment {dh:+.0f} m", "SFCVM aligned", dh) for dh in (-100.0, -50.0, 50.0, 100.0)]
for name, tab_name, dh in cases:
    mu = mu_uniform if tab_name is None else mu_from(tables[tab_name], dh)
    surf = greens(mu, patches)
    G = sample(surf, x_data)
    a_sel = corner(G)
    slip, const = invert(G, a_sel)
    resid = d_data - (G @ slip + const)
    widths = np.diff(np.array(patches), axis=1).ravel()
    rows.append(dict(case=name, dh_west_m=dh, alpha=a_sel, rms_mm=np.sqrt(np.mean(resid ** 2)),
                     rms_west_300m_mm=np.sqrt(np.mean(resid[near & (x_data < 0)] ** 2)),
                     rms_east_300m_mm=np.sqrt(np.mean(resid[near & (x_data > 0)] ** 2)),
                     surface_slip_mm=slip[0], potency_per_m=float(np.sum(slip * widths) / 1e3),
                     depth_90pct_m=float(np.array(patches)[np.searchsorted(np.cumsum(slip * widths) / np.sum(slip * widths), 0.9), 1])))
    fields[name] = dict(mu=mu, slip=slip, pred_plot=sample(surf, x_plot) @ slip + const)
    r = rows[-1]
    print(f"{name:34s} rms {r['rms_mm']:.3f} (W {r['rms_west_300m_mm']:.2f}, E {r['rms_east_300m_mm']:.2f}); "
          f"surface {r['surface_slip_mm']:.1f} mm; potency {r['potency_per_m']:.2f}; 90% above {r['depth_90pct_m']:.0f} m")
result = pd.DataFrame(rows)
result.to_csv(os.path.join(out_dir, "antiplane_sfcvm_2d_aligned.csv"), index=False)
adjusted = result[result.case.str.startswith("Aligned, west")].sort_values("rms_mm").iloc[0].case
best_name = adjusted if result.set_index("case").loc[adjusted, "rms_mm"] < result.set_index("case").loc["SFCVM aligned", "rms_mm"] else "SFCVM aligned"
print("best of the aligned cases:", best_name)

# ---------------------------------------------------------------- figure: fits, slip, misfit; rigidity cross-sections below
bins = np.linspace(-3000, 3000, 121)
idx = np.digitize(x_data, bins)
med = np.array([np.median(d_data[idx == k]) if (idx == k).sum() > 5 else np.nan for k in range(1, bins.size)])
utils.mpl_paper_style(base=7)
fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[1.5, 1.0, 1.0])
shown = ["Homogeneous", "SFCVM v26 as is", "SFCVM aligned", best_name] if best_name != "SFCVM aligned" else ["Homogeneous", "SFCVM v26 as is", "SFCVM aligned"]
colours = dict(zip(shown, ["#0072B2", "#E69F00", "#D55E00", "#000000"]))
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
sub = result[result.case.str.startswith("Aligned, west") | (result.case == "SFCVM aligned")].sort_values("dh_west_m")
ax.plot(sub.dh_west_m, sub.rms_mm, "o-", color="#000000", ms=3, label="All pixels")
ax.plot(sub.dh_west_m, sub.rms_west_300m_mm, "s--", color="#D55E00", ms=3, label="West, 300 m")
ax.plot(sub.dh_west_m, sub.rms_east_300m_mm, "^:", color="#0072B2", ms=3, label="East, 300 m")
for col, c in (("rms_mm", "#000000"), ("rms_west_300m_mm", "#D55E00"), ("rms_east_300m_mm", "#0072B2")):
    ax.axhline(result.set_index("case").loc["Homogeneous", col], color=c, lw=0.6, alpha=0.5)
ax.set_xlabel("West sediment base change (m)"); ax.set_ylabel("RMS misfit (mm)")
ax.legend(fontsize=6, loc="upper right")
utils.panel_title(ax, "C", "Misfit (faint lines: homogeneous)")
from matplotlib.colors import LogNorm
sections = ["SFCVM v26 as is", "SFCVM aligned", best_name] if best_name != "SFCVM aligned" else ["SFCVM v26 as is", "SFCVM aligned"]
for k, name in enumerate(sections):
    ax = fig.add_subplot(grid[1, k])
    im = ax.pcolormesh(xc, zc / 1e3, fields[name]["mu"], cmap="cividis", norm=LogNorm(0.03, 30), shading="nearest", rasterized=True)
    ax.axvline(0, color="white", lw=1.0, ls="--")
    ax.set_xlim(-1500, 1500); ax.set_ylim(1.0, 0)
    ax.set_xlabel("Distance from the trace (m, + east)")
    if k == 0:
        ax.set_ylabel("Depth (km)")
    utils.panel_title(ax, "DEF"[k], name.replace("Aligned, ", "Aligned, ") + " (rigidity)")
fig.colorbar(im, ax=fig.axes[-len(sections):], label="Rigidity (GPa)", shrink=0.9)
f_fig = os.path.join(fig_dir, "test_antiplane_sfcvm_2d_aligned.png")
fig.savefig(f_fig, dpi=300, facecolor="white")
print("wrote", f_fig)
