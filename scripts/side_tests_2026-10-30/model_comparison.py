"""model_comparison.py: SM figure comparing three slip models (Dani 2026-10-30: "three columns, model, residual,
distributed fault slip", with the best-fitting dipping homogeneous model). Replaces figS_model_slip in the SM.

Rows: the 3D SFSVM model with a vertical fault (08a selected_model.npz of 2026-10-30, now in outputs/08a_3d_model/
retired_2026-10-30/), the homogeneous model with a vertical fault (07a's, outputs/07a_uniform_model/retired_2026-10-30/) and the best-fitting dipping homogeneous model, 60 W (dip_deflection_2026-10-30/
homog60w_model.npz, made by fig2_homog60w.py from scripts/sensitivity/dip_rows.py at its recorded corner).
Columns: model prediction at the quadtree samples (plane removed), residual (plane-removed data minus model), slip
against along-strike distance and depth. Numbers in the row labels come from the products (Mw 30 GPa, RMS of the
fitted samples) and from model_utils.side_deflection_mm (the M7 deflection metric).
Science style (insar_utils.mpl_paper_style(base=7)), 18.3 cm wide, png + pdf into Figures_2026-10-28.
Run: cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. ~/miniforge3/envs/opera_disp_feb13/bin/python <this file>
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from cmcrameri import cm as cmc
import insar_utils as utils
import model_config as cfg
import model_utils
from TulePond_filepaths import paths_out, common_paths, data_dir

utils.mpl_paper_style(base=7)
cm = 1 / 2.54
model_data = np.load(os.path.join(paths_out["07a_uniform_model"], "model_data.npz"))
# 2026-10-30 late: the preferred model is now the 70 W one; the VERTICAL rows read the retired vertical products explicitly
selected = np.load(os.path.join(paths_out["08a_3d_model"], "retired_2026-10-30", "selected_model.npz"))      # 3D, vertical fault
homog = np.load(os.path.join(paths_out["07a_uniform_model"], "retired_2026-10-30", "homogeneous_model.npz"))  # homogeneous, vertical
assert "gf_shallow_sfcvm_order2" in str(selected["run"]), "the vertical-fault row must be the vertical run"
homog60w = np.load(os.path.join(data_dir, "dip_deflection_2026-10-30", "homog60w_model.npz"))
for product in (selected, homog, homog60w):
    assert np.array_equal(product["sample_lon"], model_data["sample_lon"]), "products are on different samples"
    assert np.array_equal(product["keep"], selected["keep"]), "products fit different samples"
keep = selected["keep"]
signed_d_km, sample_km = selected["signed_d_km"], selected["sample_km_pp"]
lo_km, hi_km = cfg.sse_section_km()
in_section = (sample_km >= lo_km) & (sample_km <= hi_km)


def row(name, product, slip_triangulation, slip_mm):
    data = np.asarray(product["data_plane_removed_mm"], float)
    model = np.asarray(product["model_mm"], float)
    residual = data - model
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    assert abs(rms - float(product["rms_mm"])) < 1e-3, f"{name}: RMS {rms:.4f} vs product {float(product['rms_mm']):.4f}"
    west = model_utils.side_deflection_mm(model, signed_d_km, in_section, keep, -1, (0.06, 0.25), (1.5, 3.0))[0]
    east = model_utils.side_deflection_mm(model, signed_d_km, in_section, keep, 1, (0.06, 0.25), (1.5, 3.0))[0]
    label = (f"{name}\nMw {float(product['Mw_30gpa']):.2f}, RMS {rms:.2f} mm\n"
             f"Deflection {west:+.1f} / {east:+.1f} mm")
    return dict(label=label, model=model, residual=residual, tri=slip_triangulation, slip=slip_mm)


rows = [row("3D SFSVM, vertical fault", selected,
            mtri.Triangulation(selected["imp_km"], selected["imp_depth_km"], selected["sub_faces"]), selected["slip_mm"]),
        row("Homogeneous, vertical fault", homog,
            mtri.Triangulation(selected["imp_km"], selected["imp_depth_km"], selected["sub_faces"]), homog["slip_mm"]),
        row("Homogeneous, 60\N{DEGREE SIGN}W fault", homog60w,
            mtri.Triangulation(homog60w["imp_km"], homog60w["imp_depth_km"], homog60w["sub_faces"]), homog60w["slip_mm"])]
FOUR_ROWS = os.environ.get("FOUR_ROWS") == "1"
if FOUR_ROWS:
    # Dani 2026-10-30: top two rows vertical (3D, homogeneous), bottom two rows the same pair with a 70 W fault
    # (3D: fine mesh, dip_deflection_2026-10-30/product_3d_dip70w.py; homogeneous: 250 m mesh, product_homog_dip70w.py)
    dip70w_3d = np.load(os.path.join(data_dir, "dip_deflection_2026-10-30", "dip70w_3d_model.npz"))
    dip70w_homog = np.load(os.path.join(data_dir, "dip_deflection_2026-10-30", "homog70w_model.npz"))
    for product in (dip70w_3d, dip70w_homog):
        assert np.array_equal(product["keep"], keep)
    # order (Dani 2026-10-30): homogeneous vertical, 3D vertical, homogeneous 70 W, 3D 70 W
    rows = [rows[1], rows[0],
            row("Homogeneous, 70\N{DEGREE SIGN}W fault", dip70w_homog,
                mtri.Triangulation(dip70w_homog["imp_km"], dip70w_homog["imp_depth_km"], dip70w_homog["sub_faces"]), dip70w_homog["slip_mm"]),
            row("3D SFSVM, 70\N{DEGREE SIGN}W fault", dip70w_3d,
                mtri.Triangulation(dip70w_3d["imp_km"], dip70w_3d["imp_depth_km"], dip70w_3d["sub_faces"]), dip70w_3d["slip_mm"])]
data_w = model_utils.side_deflection_mm(np.asarray(selected["data_plane_removed_mm"], float), signed_d_km, in_section, keep, -1,
                                        (0.06, 0.25), (1.5, 3.0))[0]
data_e = model_utils.side_deflection_mm(np.asarray(selected["data_plane_removed_mm"], float), signed_d_km, in_section, keep, 1,
                                        (0.06, 0.25), (1.5, 3.0))[0]
print(f"data deflection (3D plane removed): {data_w:+.2f} / {data_e:+.2f} mm")

# map window: the fitted samples around the SSE section
window = keep & (sample_km >= lo_km - 1.0) & (sample_km <= hi_km + 1.0) & (np.abs(signed_d_km) <= 3.0)
lon_lim = (model_data["sample_lon"][window].min() - 0.005, model_data["sample_lon"][window].max() + 0.005)
lat_lim = (model_data["sample_lat"][window].min() - 0.005, model_data["sample_lat"][window].max() + 0.005)
polygons = [np.c_[lon_row, lat_row] for lon_row, lat_row in zip(model_data["block_lon"], model_data["block_lat"])]
trace = model_data["trace_lonlat"]
aspect = 1.0 / np.cos(np.radians(np.mean(lat_lim)))
map_limit_mm = 4.0
slip_max = max(float(np.max(r["slip"])) for r in rows)

n_rows = len(rows)
fig = plt.figure(figsize=(18.3 * cm, (17.0 if n_rows == 3 else 22.0) * cm), constrained_layout=True)
grid = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 1.9])
letters = iter("ABCDEFGHIJKL")
for i, r in enumerate(rows):
    for j, (values, title) in enumerate(((r["model"], "Model"), (r["residual"], "Residual"))):
        ax = fig.add_subplot(grid[i, j])
        shown = np.where(keep, values, np.nan)
        collection = PolyCollection(polygons, array=shown, cmap=cmc.vik, edgecolors="none")
        collection.set_clim(-map_limit_mm, map_limit_mm)
        ax.add_collection(collection)
        ax.plot(trace[:, 0], trace[:, 1], color="k", lw=0.5)
        ax.set_xlim(lon_lim)
        ax.set_ylim(lat_lim)
        ax.set_aspect(aspect)
        ax.set_xticks([])
        ax.set_yticks([])
        utils.panel_title(ax, next(letters), title if i == 0 else "")
        if i == 0 and j == 0:                                   # 2 km scale bar (maps are north up, unrotated)
            bar_deg = 2.0 / (111.32 * np.cos(np.radians(np.mean(lat_lim))))
            x0 = lon_lim[0] + 0.06 * (lon_lim[1] - lon_lim[0])
            y0 = lat_lim[0] + 0.06 * (lat_lim[1] - lat_lim[0])
            ax.plot([x0, x0 + bar_deg], [y0, y0], color="k", lw=1.0)
            ax.text(x0 + bar_deg / 2, y0 + 0.012 * (lat_lim[1] - lat_lim[0]), "2 km", ha="center", va="bottom", fontsize=6.5)
        if j == 0:
            ax.set_ylabel(r["label"], fontsize=6.5)
        if i == n_rows - 1 and j == 1:
            fig.colorbar(collection, ax=ax, orientation="horizontal", label="Fault-parallel displacement (mm)", shrink=0.9)
    ax = fig.add_subplot(grid[i, 2])
    image = ax.tripcolor(r["tri"], r["slip"], shading="gouraud", cmap=cmc.lajolla, vmin=0, vmax=slip_max)
    for section_end_km in (lo_km, hi_km):
        ax.axvline(section_end_km, color="0.85", ls="--", lw=0.6)
    ax.set_ylim(1.5, 0)
    ax.set_xlim(54.5, 64.5)
    ax.set_ylabel("Depth (km)")
    utils.panel_title(ax, next(letters), "Slip on the fault" if i == 0 else "")
    if i == n_rows - 1:
        ax.set_xlabel("Distance along the fault from Point Pinole (km)")
        fig.colorbar(image, ax=ax, orientation="horizontal", label="Right-lateral slip (mm)", shrink=0.6)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(common_paths["fig_dir"], f"figS_model_comparison{'_4rows' if FOUR_ROWS else ''}.{ext}"), dpi=300, facecolor="white")
print("saved", "figS_model_comparison" + ("_4rows" if FOUR_ROWS else ""))
