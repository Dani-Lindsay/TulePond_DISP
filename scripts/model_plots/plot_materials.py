#!/usr/bin/env python3
"""
Author: Danielle Lindsay

Material figures for the SFSVM grids of the slip models (no Poisson cap; written by pylith/build/sfcvm_grid.py):
(a) map of the shear modulus mu = rho vs^2 at 50 m depth, as delivered and aligned on the mapped trace; (b) cross-section
of mu through the Tule Pond section at X-X' (lat 37.5612 N), as delivered and aligned, with the two published fault
dips of the section: 70.5 +/- 2 deg SW where the CFW1 creepmeter borehole crosses the fault at ~21 m depth (Bilham &
Whitehead 1997) and ~70 deg E at 100-650 m depth on the Alameda Creek seismic line (Williams et al. 2005).
Grid values are interpolated linearly (as PyLith's SimpleGridDB with query_type = linear).

`main(spatialdb_dir, fig_dir)` reads the two grids from spatialdb_dir (modeling_SFCVM/spatialdb,
TulePond_filepaths.paths_sfcvm) and writes figS_pylith_materials_sfcvm_v26.png (journal size, 7 pt) into fig_dir.
Run (repo root, environment.yml):  python scripts/model_plots/plot_materials.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
import pyproj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from TulePond_filepaths import paths_sfcvm, common_paths   # noqa: E402
from insar_utils import mpl_paper_style                     # noqa: E402

lat_xx = 37.5612
# published dips of the Tule Pond section: (dip deg, -1 west / +1 east, top and bottom depth m drawn)
PUBLISHED_DIPS = ((70.5, -1, 0.0, 1500.0),     # CFW1 borehole, ~21 m depth (Bilham & Whitehead 1997); drawn as the model fault
                  (70.0, +1, 100.0, 650.0))    # Alameda Creek seismic line, 100-650 m depth (Williams et al. 2005)
VERMILLION = "#D55E00"   # Okabe-Ito


def main(spatialdb_dir, fig_dir, name_materials="figS_pylith_materials_sfcvm_v26"):
    """Write the materials figure into fig_dir; return its path."""
    os.makedirs(fig_dir, exist_ok=True)
    grids = {tag: np.load(os.path.join(spatialdb_dir, f"sfcvm_v26-0_{tag}_nocap_grid.npz")) for tag in ("asis", "aligned")}
    g0 = grids["aligned"]
    tx, ty, tkm, normal = g0["trace_x"], g0["trace_y"], g0["trace_km"], g0["trace_normal"]
    mu = {tag: (g["density"].astype(float) * g["vs"].astype(float) ** 2 / 1e9) for tag, g in grids.items()}
    interp = {tag: RegularGridInterpolator((g["x"], g["y"], g["z"]), mu[tag], method="linear") for tag, g in grids.items()}

    # X-X': the trace point nearest lat 37.5612 (tmerc y of that latitude on the trace)
    to_mesh = pyproj.Transformer.from_crs("EPSG:4326", str(g0["crs"]), always_xy=True)
    _, y_xx = to_mesh.transform(-121.97, lat_xx)
    i_xx = int(np.argmin(np.abs(ty - y_xx)))
    print(f"X-X': trace point km {tkm[i_xx]:.1f} at ({tx[i_xx]:.0f}, {ty[i_xx]:.0f}) m; boundary offset b = "
          f"{g0['boundary_smooth_m'][i_xx]:.0f} m")

    mpl_paper_style(base=7)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    norm = LogNorm(0.03, 30)
    xs, ys = np.linspace(-6000, 5000, 441), np.linspace(-7000, 7000, 561)
    XS, YS = np.meshgrid(xs, ys, indexing="ij")
    for ax, tag, letter in zip(axes[0], ("asis", "aligned"), "AB"):
        vals = interp[tag](np.c_[XS.ravel(), YS.ravel(), np.full(XS.size, -50.0)]).reshape(XS.shape)
        pc = ax.pcolormesh(xs / 1e3, ys / 1e3, vals.T, norm=norm, cmap="cividis", shading="auto")
        ax.plot(tx / 1e3, ty / 1e3, color=VERMILLION, lw=1.0, label="Mapped trace")
        ax.axhline(ty[i_xx] / 1e3, color="k", ls=":", lw=0.7)
        ax.set_aspect("equal"); ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
        ax.set_title(r"$\bf{" + letter + r"}$  " + f"50 m depth, {'as delivered' if tag == 'asis' else 'aligned on the trace'}", loc="left", fontsize=7)
        ax.legend(loc="lower left", fontsize=6)
    fig.colorbar(pc, ax=axes[0], label="Shear modulus (GPa)", shrink=0.8)
    offs, deps = np.linspace(-3000, 3000, 601), np.linspace(0, 1500, 151)
    O, D = np.meshgrid(offs, deps, indexing="ij")
    for ax, tag, letter in zip(axes[1], ("asis", "aligned"), "CD"):
        px, py = tx[i_xx] + O * normal[i_xx, 0], ty[i_xx] + O * normal[i_xx, 1]
        vals = interp[tag](np.c_[px.ravel(), py.ravel(), -D.ravel()]).reshape(O.shape)
        pc2 = ax.pcolormesh(offs, deps, vals.T, norm=norm, cmap="cividis", shading="auto")
        for (dip, sign, z_top, z_bottom), colour, style, label in zip(PUBLISHED_DIPS, (VERMILLION, "k"), ("-", "--"),
                                                                    ("Bilham & Whitehead (1997), 70.5\N{DEGREE SIGN} SW",
                                                                     "Williams et al. (2005), 70\N{DEGREE SIGN} E")):
            run = sign / np.tan(np.radians(dip))                 # m east per m depth: west dip negative
            ax.plot([z_top * run, z_bottom * run], [z_top, z_bottom], color=colour, ls=style, lw=1.2, label=label)
        ax.set_ylim(1500, 0); ax.set_xlabel("Distance from the trace, east positive (m)"); ax.set_ylabel("Depth (m)")
        ax.set_title(r"$\bf{" + letter + r"}$  " + f"X-X' (km {tkm[i_xx]:.1f}), {'as delivered' if tag == 'asis' else 'aligned on the trace'}", loc="left", fontsize=7)
        ax.legend(loc="lower left", fontsize=6)
    fig.colorbar(pc2, ax=axes[1], label="Shear modulus (GPa)", shrink=0.8)
    path_a = os.path.join(fig_dir, f"{name_materials}.png")
    fig.savefig(path_a, dpi=300)
    plt.close(fig)

    print(f"wrote {path_a}")
    return path_a


if __name__ == "__main__":
    main(paths_sfcvm["spatialdb"], common_paths["fig_dir"])
