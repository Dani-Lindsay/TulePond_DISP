#!/usr/bin/env python3
"""
DIAGNOSTIC (2026-09-30, home-made 2D antiplane solver, NOT a validated tool):
WEST-dipping fault in the aligned SFCVM structure and in a homogeneous
half-space. The 3D homogeneous suite improves towards west dips (still at
75 W, the scan edge). Dips 90, 80 W, 70 W, 60 W, 45 W; each at its own
L-curve corner. The staircase fault handles either dip direction here.
CHECK: uniform mu against the analytic dipping screw dislocation.
Output: outputs/07_slip_model_tests/antiplane_sfcvm_2d_dip_west.csv
"""
import os

source = open(os.path.join("drafts", "07", "antiplane_sfcvm_2d_dip.py")).read()
exec(source.split("# ---------------------------------------------------------------- CHECK: dipping fault")[0])


def greens_dip(mu, patches, dip_deg):
    """Surface Green's functions for a fault from the trace at x = 0 dipping at dip_deg
    (< 90: east, > 90: west); unit slip on each depth patch. Slip jump = east side minus west side."""
    A, _, idx = assemble(mu)
    lu = splu(A)
    x_fault = zc / np.tan(np.deg2rad(dip_deg)) if abs(dip_deg - 90) > 1e-6 else np.zeros_like(zc)
    east = xc[None, :] > x_fault[:, None]
    dxc, dzc = xc[1:] - xc[:-1], zc[1:] - zc[:-1]
    t_x = (2 * mu[:, 1:] * mu[:, :-1] / (mu[:, 1:] + mu[:, :-1])) * dz[:, None] / dxc[None, :]
    t_z = (2 * mu[1:, :] * mu[:-1, :] / (mu[1:, :] + mu[:-1, :])) * dx[None, :] / dzc[:, None]
    B = np.zeros((nx * nz, len(patches)))
    z_face = z_edges[1:-1]
    for k, (z1, z2) in enumerate(patches):
        jj, ii = np.where(~east[:, :-1] & east[:, 1:] & (zc[:, None] > z1) & (zc[:, None] < z2))   # vertical faces: west | east
        np.add.at(B[:, k], idx[jj, ii], t_x[jj, ii]); np.add.at(B[:, k], idx[jj, ii + 1], -t_x[jj, ii])
        in_patch = (z_face[:, None] > z1) & (z_face[:, None] < z2)
        jj, ii = np.where(east[:-1, :] & ~east[1:, :] & in_patch)      # east above, west below (east dip)
        np.add.at(B[:, k], idx[jj + 1, ii], t_z[jj, ii]); np.add.at(B[:, k], idx[jj, ii], -t_z[jj, ii])
        jj, ii = np.where(~east[:-1, :] & east[1:, :] & in_patch)      # west above, east below (west dip)
        np.add.at(B[:, k], idx[jj, ii], t_z[jj, ii]); np.add.at(B[:, k], idx[jj + 1, ii], -t_z[jj, ii])
    U = lu.solve(B)
    return U.reshape(nz, nx, -1)[0]


for dip_chk in (66.0, 114.0, 135.0):
    g_ = sample(greens_dip(mu_uniform, [(0.0, 800.0)], dip_chk), x_chk)[:, 0]
    xi = 800.0 / np.tan(np.deg2rad(dip_chk))
    analytic = (0.5 * np.pi * np.sign(x_chk) - np.arctan((x_chk - xi) / 800.0)) / np.pi
    err = np.max(np.abs(g_ - analytic))
    print(f"CHECK dip {dip_chk:g}: max |FV - analytic| = {err:.4f} (per unit slip)")
    assert err < 0.03, "dipping-fault solver does not reproduce the analytic solution"

tab_al = tables["SFCVM aligned"]
mu_al = mu_from(tab_al)
near = np.abs(x_data) <= 300
rows = []
for medium, mu in (("Homogeneous", mu_uniform), ("SFCVM aligned", mu_al)):
    for dip in (90.0, 100.0, 110.0, 120.0, 135.0):
        surf = greens_dip(mu, patches, dip)
        G = sample(surf, x_data)
        a_sel = corner(G)
        slip, const = invert(G, a_sel)
        resid = d_data - (G @ slip + const)
        widths = np.diff(np.array(patches), axis=1).ravel() / abs(np.sin(np.deg2rad(dip)))
        rows.append(dict(medium=medium, dip=dip, dip_west_deg=180 - dip if dip > 90 else 90, rms_mm=np.sqrt(np.mean(resid ** 2)),
                         rms_west_300m_mm=np.sqrt(np.mean(resid[near & (x_data < 0)] ** 2)),
                         rms_east_300m_mm=np.sqrt(np.mean(resid[near & (x_data > 0)] ** 2)),
                         surface_slip_mm=slip[0], potency_per_m=float(np.sum(slip * widths) / 1e3)))
        r = rows[-1]
        print(f"{medium:14s} dip {r['dip_west_deg']:>3g} W: rms {r['rms_mm']:.3f} (W {r['rms_west_300m_mm']:.2f}, E {r['rms_east_300m_mm']:.2f}); "
              f"surface {r['surface_slip_mm']:.1f} mm; potency {r['potency_per_m']:.2f}")
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "antiplane_sfcvm_2d_dip_west.csv"), index=False)
