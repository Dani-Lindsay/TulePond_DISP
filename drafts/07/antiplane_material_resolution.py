#!/usr/bin/env python3
"""
DIAGNOSTIC (2026-09-30, home-made 2D antiplane solver, NOT a validated tool):
why is the SFCVM effect much smaller in the 3D PyLith model (potency x1.15,
west 300 m rms 2.52 -> 2.22 mm) than in the 2D test (x1.8-2.4)? PyLith holds
one rigidity per cell, read at the cell centroid, so with 60-100 m cells the
soft top 25-100 m of the west sediment (mu 0.04-0.5 GPa in SFCVM) is never
seen: PyLith's fault-adjacent mu in the top 100 m came out west 0.50-0.59 GPa,
east 1.1-1.3 GPa. Here the 2D model is re-run with mu sampled the way PyLith
saw it (block-averaged to a cell size h, value at the block centre).
Machinery: drafts/07/antiplane_sfcvm_2d_aligned.py exec'd up to its fits.
Output: outputs/07_slip_model_tests/antiplane_material_resolution.csv
"""
import os

source = open(os.path.join("drafts", "07", "antiplane_sfcvm_2d_aligned.py")).read()
exec(source.split("# ---------------------------------------------------------------- 2. fits")[0])

tab_al = tables["SFCVM aligned"]
mu_fine = mu_from(tab_al)                                     # the 10 m-cell field the 2D test used
near = np.abs(x_data) <= 300


def blocky(mu, h):
    """mu on the solver grid, replaced by its value at the centre of the h x h block each cell falls in."""
    out = np.empty_like(mu)
    for j, z in enumerate(zc):
        zb = (np.floor(z / h) + 0.5) * h
        jz = int(np.argmin(np.abs(zc - zb)))
        for side in (xc < 0, xc >= 0):
            xs = xc[side]
            xb = np.sign(xs) * (np.floor(np.abs(xs) / h) + 0.5) * h
            ix = np.searchsorted(xc, xb).clip(0, nx - 1)
            out[j, side] = mu[jz, ix]
    return out


rows = []
for name, mu in (("Homogeneous", mu_uniform), ("SFCVM aligned, 10 m cells (2D test)", mu_fine),
                 ("SFCVM aligned, 60 m cells", blocky(mu_fine, 60.0)), ("SFCVM aligned, 100 m cells", blocky(mu_fine, 100.0)),
                 ("SFCVM aligned, 250 m cells", blocky(mu_fine, 250.0))):
    surf = greens(mu, patches)
    G = sample(surf, x_data)
    a_sel = corner(G)
    slip, const = invert(G, a_sel)
    resid = d_data - (G @ slip + const)
    widths = np.diff(np.array(patches), axis=1).ravel()
    w_top = mu[np.argmin(np.abs(zc - 50)), (xc < 0) & (xc > -100)].mean() / 1e0
    e_top = mu[np.argmin(np.abs(zc - 50)), (xc > 0) & (xc < 100)].mean()
    rows.append(dict(case=name, mu_west_top100_gpa=w_top, mu_east_top100_gpa=e_top, rms_mm=np.sqrt(np.mean(resid ** 2)),
                     rms_west_300m_mm=np.sqrt(np.mean(resid[near & (x_data < 0)] ** 2)),
                     rms_east_300m_mm=np.sqrt(np.mean(resid[near & (x_data > 0)] ** 2)),
                     surface_slip_mm=slip[0], potency_per_m=float(np.sum(slip * widths) / 1e3)))
    r = rows[-1]
    print(f"{name:38s} mu top 100 m W {w_top:.2f} E {e_top:.2f} GPa; rms {r['rms_mm']:.3f} (W {r['rms_west_300m_mm']:.2f}, "
          f"E {r['rms_east_300m_mm']:.2f}); potency {r['potency_per_m']:.2f}")
res = pd.DataFrame(rows)
res["potency_ratio"] = res.potency_per_m / res.potency_per_m.iloc[0]
res.to_csv(os.path.join(out_dir, "antiplane_material_resolution.csv"), index=False)
print(res.round(3).to_string(index=False))
