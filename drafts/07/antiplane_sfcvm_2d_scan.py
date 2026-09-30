#!/usr/bin/env python3
"""
DRAFT (2026-09-29): starting from SFCVM v26, which rigidity structure does the
Tule Pond mean event prefer? (Dani: a regional model can be off by a little;
solve for it on the 2D profile.)

Machinery, data and inversion of drafts/07/antiplane_sfcvm_2d.py (exec'd up to
its figure). mu(x, z) here comes from the DENSE SFCVM sample
(sfcvm_extract_dense.py: 50 m across the fault within 1 km, 25 m in depth to
500 m; median over 57.5-61.5 km along strike), perturbed by two parameters:
  shift_m    the whole structure moved across the fault (+ = east). SFCVM puts
             the soft Quaternary / Valley Sequence boundary ~175 m east of the
             creeping trace; the model's horizontal grid is 100 m.
  thick_x    the soft near-surface section (top 500 m) stretched in depth by
             this factor: the depth to the stiffer rock beneath the sediment.
Each pair is inverted at a fixed smoothing weight (the SFCVM L-curve corner of
the parent script), so misfits compare like for like; the best pair is then
re-inverted at its own corner.

Output: outputs/07_slip_model_tests/antiplane_sfcvm_2d_scan.csv,
        Figures_<date>/test_antiplane_sfcvm_2d_scan.png
"""
import os

source = open(os.path.join("drafts", "07", "antiplane_sfcvm_2d.py")).read()
exec(source.split("# ---------------------------------------------------------------- figure")[0])

dense = pd.read_csv(os.path.join(out_dir, "sfcvm_creep_section_dense_v26-0.csv"))
dense = dense[(dense.km >= 57.5) & (dense.km <= 61.5)]
tab = dense.groupby(["offset_km", "depth_m"]).mu_gpa.median().unstack()
off_m, dep_m = tab.index.values * 1e3, tab.columns.values.astype(float)
mu_interp = RegularGridInterpolator((off_m, dep_m), tab.values, method="nearest")   # nearest: keep the model's sharp block boundaries


def mu_field(shift_m, thick_x, z_soft=500.0):
    """SFCVM mu on the solver grid, moved across the fault by shift_m and with
    the top z_soft metres stretched in depth by thick_x."""
    z_model = np.where(zc < z_soft * thick_x, zc / thick_x, zc - z_soft * (thick_x - 1.0))
    X, Z = np.meshgrid(np.clip(xc - shift_m, off_m.min(), off_m.max()), np.clip(z_model, 0, dep_m.max()))
    return mu_interp(np.c_[X.ravel(), Z.ravel()]).reshape(X.shape)


alpha_fixed = float(summary.loc[summary.medium == "SFCVM v26", "alpha"].iloc[0])
near = np.abs(x_data) <= 300
rows = []
for shift_m in np.arange(-250, 101, 25):
    for thick_x in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        G = sample(greens(mu_field(shift_m, thick_x), patches), x_data)
        slip, const = invert(G, alpha_fixed)
        resid = d_data - (G @ slip + const)
        rows.append(dict(shift_m=shift_m, thick_x=thick_x, rms_mm=np.sqrt(np.mean(resid ** 2)),
                         rms_west_300m_mm=np.sqrt(np.mean(resid[near & (x_data < 0)] ** 2)),
                         rms_east_300m_mm=np.sqrt(np.mean(resid[near & (x_data > 0)] ** 2)),
                         surface_slip_mm=slip[0],
                         potency_per_m=float(np.sum(slip * np.diff(np.array(patches), axis=1).ravel()) / 1e3)))
    print(f"shift {shift_m:+4.0f} m done; best so far rms {min(r['rms_mm'] for r in rows):.3f}")
scan = pd.DataFrame(rows)
scan.to_csv(os.path.join(out_dir, "antiplane_sfcvm_2d_scan.csv"), index=False)
ref = scan[(scan.shift_m == 0) & (scan.thick_x == 1.0)].iloc[0]
best = scan.loc[scan.rms_mm.idxmin()]
print(f"\nSFCVM as is : rms {ref.rms_mm:.3f} (W {ref.rms_west_300m_mm:.2f}, E {ref.rms_east_300m_mm:.2f}), surface {ref.surface_slip_mm:.1f} mm, potency {ref.potency_per_m:.2f}")
print(f"best        : shift {best.shift_m:+.0f} m, thickness x{best.thick_x:g}: rms {best.rms_mm:.3f} (W {best.rms_west_300m_mm:.2f}, E {best.rms_east_300m_mm:.2f}), "
      f"surface {best.surface_slip_mm:.1f} mm, potency {best.potency_per_m:.2f}")
hom = summary[summary.medium == "Homogeneous"].iloc[0]
print(f"homogeneous : rms {hom.rms_mm:.3f} (W {hom.rms_west_300m_mm:.2f}, E {hom.rms_east_300m_mm:.2f}), surface {hom.surface_slip_mm:.1f} mm, potency {hom.potency_per_m:.2f}")

# the best structure at its own L-curve corner, for the figure
mu_best = mu_field(best.shift_m, best.thick_x)
surf_best = greens(mu_best, patches)
G_best = sample(surf_best, x_data)
a_best = corner(G_best)
slip_best, const_best = invert(G_best, a_best)
resid_best = d_data - (G_best @ slip_best + const_best)
print(f"best at its own corner (alpha {a_best:.2g}): rms {np.sqrt(np.mean(resid_best ** 2)):.3f}, "
      f"W {np.sqrt(np.mean(resid_best[near & (x_data < 0)] ** 2)):.2f}, E {np.sqrt(np.mean(resid_best[near & (x_data > 0)] ** 2)):.2f}, "
      f"surface slip {slip_best[0]:.1f} mm")
curves["Adjusted SFCVM"] = dict(slip=slip_best, pred_plot=sample(surf_best, x_plot) @ slip_best + const_best, resid=resid_best)

# figure: A fits, B slip with depth, C misfit over the two parameters
bins = np.linspace(-3000, 3000, 121)
idx = np.digitize(x_data, bins)
med = np.array([np.median(d_data[idx == k]) if (idx == k).sum() > 5 else np.nan for k in range(1, bins.size)])
utils.mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True, width_ratios=[1.6, 1.0, 1.1])
colours = {"Homogeneous": "#0072B2", "SFCVM v26": "#D55E00", "Adjusted SFCVM": "#000000"}
ax = axes[0]
ax.plot(x_data, d_data, ".", ms=0.8, color="0.8", rasterized=True)
ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "-", color="0.45", lw=1.0, label="Data, 50 m median")
for name, c in curves.items():
    ax.plot(x_plot, c["pred_plot"], "-", color=colours[name], lw=1.1,
            label=name if name != "Adjusted SFCVM" else f"SFCVM, shift {best.shift_m:+.0f} m, sediment x{best.thick_x:g}")
ax.axvline(0, color="0.5", lw=0.5)
ax.set_xlim(-1500, 1500); ax.set_ylim(-9, 6)
ax.set_xlabel("Distance from the trace (m, + east)"); ax.set_ylabel("Fault-parallel displacement (mm)")
ax.legend(loc="lower right", fontsize=6)
utils.panel_title(ax, "A", "Mean event, 2D antiplane fits")
ax = axes[1]
for name, c in curves.items():
    ax.step(np.r_[c["slip"], c["slip"][-1]], np.r_[[p[0] for p in patches], patches[-1][1]] / 1e3, where="post", color=colours[name], lw=1.1)
ax.set_ylim(2.0, 0); ax.set_xlabel("Slip (mm)"); ax.set_ylabel("Depth (km)")
utils.panel_title(ax, "B", "Slip with depth")
ax = axes[2]
piv = scan.pivot(index="thick_x", columns="shift_m", values="rms_mm")
im = ax.pcolormesh(piv.columns, np.arange(len(piv.index)), piv.values, cmap="magma_r", shading="nearest")
ax.set_yticks(np.arange(len(piv.index))); ax.set_yticklabels([f"x{v:g}" for v in piv.index])
ax.plot(best.shift_m, list(piv.index).index(best.thick_x), "*", color="white", mec="k", ms=8)
ax.plot(0, list(piv.index).index(1.0), "o", mfc="none", mec="#D55E00", ms=6, mew=1.2)
ax.set_xlabel("Structure shift (m, + east)"); ax.set_ylabel("Sediment thickness")
fig.colorbar(im, ax=ax, label="RMS misfit (mm)", shrink=0.9)
utils.panel_title(ax, "C", "Misfit over the structure")
f_fig = os.path.join(fig_dir, "test_antiplane_sfcvm_2d_scan.png")
fig.savefig(f_fig, dpi=300, facecolor="white")
print("wrote", f_fig)
