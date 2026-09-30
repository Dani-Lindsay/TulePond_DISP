#!/usr/bin/env python3
"""
DRAFT (2026-09-29): the aligned SFCVM structure with the fault dip fitted to
SFCVM's own fault-block boundary over the top 600 m only (Dani: the top
~600 m is still steep). Boundary points per km in the band:
outputs/07_slip_model_tests/sfcvm_fault_boundary_band.csv (offset of the
block 22 | 7 boundary from the trace, aligned at the surface). Straight-line
fit over 0-600 m: 0.339 m east per m depth = 71.3 deg E (68-74 per km).
Cases 90, 85, 80, 71.3 deg, each at its own L-curve corner.

Output: outputs/07_slip_model_tests/antiplane_sfcvm_2d_dip_top600.csv,
        Figures_<date>/test_antiplane_sfcvm_2d_dip_top600.png
"""
import os

source = open(os.path.join("drafts", "07", "antiplane_sfcvm_2d_dip.py")).read()
exec(source.split("rows, fields = [], {}")[0])

boundary_pts = pd.read_csv(os.path.join(out_dir, "sfcvm_fault_boundary_band.csv"))
top = boundary_pts[boundary_pts.z <= 600]
slope = np.polyfit(top.z, top.off_al, 1)[0]
dip_top = float(np.degrees(np.arctan(1 / slope)))
print(f"SFCVM fault, top 600 m: dip {dip_top:.1f} deg E")

rows, fields = [], {}
for name, mu, dip in [("Homogeneous, vertical", mu_uniform, 90.0)] + \
        [(f"Aligned SFCVM, dip {d:.0f}", mu_al, d) for d in (90.0, 85.0, 80.0, dip_top)]:
    row, field = fit(mu, dip)
    rows.append(dict(case=name, dip_deg=dip, **row)); fields[name] = field
    print(f"{name:32s} rms {row['rms_mm']:.3f} (W {row['rms_west_300m_mm']:.2f}, E {row['rms_east_300m_mm']:.2f}); "
          f"surface {row['surface_slip_mm']:.1f} mm; potency {row['potency_per_m']:.2f}")
result = pd.DataFrame(rows)
result.to_csv(os.path.join(out_dir, "antiplane_sfcvm_2d_dip_top600.csv"), index=False)

utils.mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True, width_ratios=[1.0, 1.5, 1.0])
ax = axes[0]
from matplotlib.colors import LogNorm
im = ax.pcolormesh(xc, zc / 1e3, mu_al, cmap="cividis", norm=LogNorm(0.03, 30), shading="nearest", rasterized=True)
ax.plot(boundary_pts.off_al, boundary_pts.z / 1e3, ".", ms=2, color="white", label="SFCVM fault, per km")
zz = np.linspace(0, 1000, 50)
ax.plot(zz * slope, zz / 1e3, "-", color="#E69F00", lw=1.2, label=f"Fit, top 600 m: {dip_top:.0f}$^\\circ$E")
ax.plot(0 * zz, zz / 1e3, "--", color="white", lw=0.8, label="Vertical")
ax.set_xlim(-600, 900); ax.set_ylim(1.0, 0)
ax.set_xlabel("Distance from the trace (m, + east)"); ax.set_ylabel("Depth (km)")
ax.legend(fontsize=5.5 + 0.5, loc="lower left")
fig.colorbar(im, ax=ax, label="Rigidity (GPa)", shrink=0.85)
utils.panel_title(ax, "A", "Aligned SFCVM and its fault")
ax = axes[1]
ax.plot(x_data, d_data, ".", ms=0.8, color="0.82", rasterized=True)
ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "-", color="0.5", lw=1.0, label="Data, 50 m median")
colours = {"Homogeneous, vertical": "#0072B2", "Aligned SFCVM, dip 90": "#000000", f"Aligned SFCVM, dip {dip_top:.0f}": "#E69F00"}
for name, c in colours.items():
    ax.plot(x_plot, fields[name]["pred_plot"], "-", color=c, lw=1.0, label=name)
ax.axvline(0, color="0.5", lw=0.5); ax.set_xlim(-1500, 1500); ax.set_ylim(-9, 6)
ax.set_xlabel("Distance from the trace (m, + east)"); ax.set_ylabel("Fault-parallel displacement (mm)")
ax.legend(loc="upper left", fontsize=6)
utils.panel_title(ax, "B", "Fits")
ax = axes[2]
sub = result[result.case.str.startswith("Aligned")]
ax.plot(sub.dip_deg, sub.rms_mm, "o-", color="#000000", ms=3, label="All pixels")
ax.plot(sub.dip_deg, sub.rms_west_300m_mm, "s--", color="#D55E00", ms=3, label="West, 300 m")
ax.plot(sub.dip_deg, sub.rms_east_300m_mm, "^:", color="#0072B2", ms=3, label="East, 300 m")
hom = result.iloc[0]
for col, c in (("rms_mm", "#000000"), ("rms_west_300m_mm", "#D55E00"), ("rms_east_300m_mm", "#0072B2")):
    ax.axhline(hom[col], color=c, lw=0.6, alpha=0.5)
ax.set_xlabel("Fault dip (deg E)"); ax.set_ylabel("RMS misfit (mm)"); ax.legend(fontsize=6)
utils.panel_title(ax, "C", "Misfit (faint: homogeneous)")
f_fig = os.path.join(fig_dir, "test_antiplane_sfcvm_2d_dip_top600.png")
fig.savefig(f_fig, dpi=300, facecolor="white")
print("wrote", f_fig)
