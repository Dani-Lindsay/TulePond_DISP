#!/usr/bin/env python3
"""
DRAFT (2026-09-29): figure for Dani's email to Charles Williams. The 2D
antiplane fits of drafts/07/antiplane_sfcvm_2d_dip_top600.py (exec'd up to its
figure): homogeneous half-space vs SFCVM v26 aligned on the creeping trace,
vertical fault and 71 deg E (SFCVM's fault over the top 600 m).
Output: Figures_<date>/email_williams_2d_rigidity.png
"""
import os

source = open(os.path.join("drafts", "07", "antiplane_sfcvm_2d_dip_top600.py")).read()
exec(source.split("utils.mpl_paper_style(base=7)")[0])
from matplotlib.colors import LogNorm

# ---- uniform slip from the surface to depth D (Dani 2026-09-29: cleaner for the email than distributed slip)
# every patch edge above 3 km is a trial D; slip and a constant by least squares (slip >= 0)
edges_top = np.array([p[0] for p in patches] + [patches[-1][1]])
trial_D = edges_top[(edges_top > 0) & (edges_top <= 3000)]
uniform = {}
for key, mu, dip in (("Homogeneous, vertical", mu_uniform, 90.0), ("Aligned SFCVM, dip 90", mu_al, 90.0),
                     (f"Aligned SFCVM, dip {dip_top:.0f}", mu_al, dip_top)):
    surf = greens_dip(mu, patches, dip)
    G, Gp = sample(surf, x_data), sample(surf, x_plot)
    best = None
    for D in trial_D:
        cols = np.array([p[1] <= D + 1e-6 for p in patches])
        g1 = G[:, cols].sum(axis=1)
        A_ = np.c_[g1, np.ones_like(g1)]
        (s_, c_), *_ = np.linalg.lstsq(A_, d_data, rcond=None)
        if s_ <= 0:
            continue
        rss = float(np.sum((d_data - A_ @ [s_, c_]) ** 2))
        if best is None or rss < best[0]:
            best = (rss, D, s_, c_, cols)
    rss, D, s_, c_, cols = best
    resid = d_data - (G[:, cols].sum(axis=1) * s_ + c_)
    near_w, near_e = (np.abs(x_data) <= 300) & (x_data < 0), (np.abs(x_data) <= 300) & (x_data > 0)
    width = D / np.sin(np.deg2rad(dip))
    uniform[key] = dict(D=D, slip=s_, pred_plot=Gp[:, cols].sum(axis=1) * s_ + c_,
                        rms_mm=np.sqrt(np.mean(resid ** 2)), rms_west_300m_mm=np.sqrt(np.mean(resid[near_w] ** 2)),
                        rms_east_300m_mm=np.sqrt(np.mean(resid[near_e] ** 2)), potency_per_m=s_ * width / 1e3)
    print(f"{key:28s} uniform slip {s_:.1f} mm to {D:.0f} m: rms {uniform[key]['rms_mm']:.3f} "
          f"(W {uniform[key]['rms_west_300m_mm']:.2f}, E {uniform[key]['rms_east_300m_mm']:.2f}), potency {uniform[key]['potency_per_m']:.2f}")
pd.DataFrame({k: {c: v for c, v in u.items() if c != "pred_plot"} for k, u in uniform.items()}).T.to_csv(
    os.path.join(out_dir, "email_williams_uniform_slip.csv"))

cases = {"Homogeneous half-space, vertical fault": ("Homogeneous, vertical", "#0072B2", "-"),
         "SFCVM v26 aligned, vertical fault": ("Aligned SFCVM, dip 90", "#000000", "-"),
         f"SFCVM v26 aligned, fault dip {dip_top:.0f}$^\\circ$E": (f"Aligned SFCVM, dip {dip_top:.0f}", "#E69F00", "--")}
res = result.set_index("case")
utils.mpl_paper_style(base=7)
fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])

ax = fig.add_subplot(gs[0, 0])
ax.plot(x_data, d_data, ".", ms=0.8, color="0.82", rasterized=True)
ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "-", color="0.45", lw=1.0, label="InSAR median")
for label, (key, c, ls) in cases.items():
    ax.plot(x_plot, uniform[key]["pred_plot"], ls, color=c, lw=1.1,
            label=label.replace("half-space, vertical fault", "").replace("v26 aligned, vertical fault", "aligned, vertical").replace("v26 aligned, fault dip", "aligned, dip").strip(", "))
ax.axvline(0, color="0.5", lw=0.5)
ax.set_xlim(-1500, 1500); ax.set_ylim(-9, 6)
ax.set_xlabel("Distance from the fault trace (m, + east)"); ax.set_ylabel("Fault-parallel displacement (mm)")
ax.legend(loc="lower right", fontsize=6)   # the empty quadrant: east of the trace, below the data
utils.panel_title(ax, "A", f"Mean slow slip event, {band_km[0]}-{band_km[1]} km along the fault")

ax = fig.add_subplot(gs[0, 1])
for label, (key, c, ls) in cases.items():
    u = uniform[key]
    ax.plot([u["slip"], u["slip"], 0], [0, u["D"] / 1e3, u["D"] / 1e3], ls, color=c, lw=1.3)
    ax.text(u["slip"] + 0.3, u["D"] / 1e3, f"{u['slip']:.1f} mm to {u['D']:.0f} m", color=c, fontsize=6, va="center")
ax.set_ylim(2.0, 0); ax.set_xlim(0, None); ax.set_xlabel("Slip (mm)"); ax.set_ylabel("Depth (km)")
utils.panel_title(ax, "B", "Best-fitting uniform slip, surface to depth D")
ax = fig.add_subplot(gs[1, 0])
im = ax.pcolormesh(xc, zc / 1e3, mu_al, cmap="cividis", norm=LogNorm(0.03, 30), shading="nearest", rasterized=True)
ax.plot(0 * zc, zc / 1e3, "-", color="white", lw=1.2, label="Vertical fault")
ax.plot(zc / np.tan(np.deg2rad(dip_top)), zc / 1e3, "--", color="#E69F00", lw=1.2, label=f"Fault dip {dip_top:.0f}$^\\circ$E")
ax.set_xlim(-1500, 1500); ax.set_ylim(1.5, 0)
ax.set_xlabel("Distance from the fault trace (m, + east)"); ax.set_ylabel("Depth (km)")
ax.legend(loc="lower left", fontsize=6)
fig.colorbar(im, ax=ax, label="Shear modulus (GPa)", shrink=0.9)
utils.panel_title(ax, "C", "SFCVM v26, shifted so its Hayward fault sits on the creeping trace")

ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
hom_pot = uniform["Homogeneous, vertical"]["potency_per_m"]
rows_t = []
for label, (key, c, ls) in cases.items():
    r = pd.Series(uniform[key])
    rows_t.append([{"Homogeneous, vertical": "Homogeneous", "Aligned SFCVM, dip 90": "SFCVM, vertical"}.get(key, f"SFCVM, dip {dip_top:.0f}°E"),
                   f"{r.rms_mm:.2f}", f"{r.rms_west_300m_mm:.2f}", f"{r.rms_east_300m_mm:.2f}", f"{r.potency_per_m / hom_pot:.1f}x"])
t = ax.table(cellText=rows_t, colLabels=["Model", "RMS", "West\n<300 m", "East\n<300 m", "Potency"],
             colWidths=[0.36, 0.14, 0.17, 0.17, 0.16], bbox=[0.0, 0.42, 1.0, 0.5])
t.auto_set_font_size(False); t.set_fontsize(6.5)
ax.text(0.0, 0.05, "RMS misfit in mm (7,922 pixels within 3 km).\nPotency per metre along strike, relative to homogeneous.\n2D antiplane finite-volume model; uniform slip from the\nsurface to D, slip and D fitted, plus one constant.",
        transform=ax.transAxes, fontsize=6, va="bottom", color="0.3")
utils.panel_title(ax, "D", "Fit and size")
f_fig = os.path.join(fig_dir, "email_williams_2d_rigidity.png")
fig.savefig(f_fig, dpi=300, facecolor="white")
print("wrote", f_fig)
