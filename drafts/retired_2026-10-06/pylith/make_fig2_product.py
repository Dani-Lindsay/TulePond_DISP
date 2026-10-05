#!/usr/bin/env python3
"""
Fig-2 product of a PyLith + SFCVM slip model, readable by scripts/07_slip_model_fig2.py in place of
outputs/07_slip_model_tests/fig2_model_row.npz (07 cell [13] keys and conventions; scripts/ untouched).

  python make_fig2_product.py <cd_tag> [<set_tag>]   cd_tag "_nuggetcd" (SELECTED) or "" (full Cd); set_tag "" or "_coh07"
  -> gf/fig2_model_row_pylith_<nuggetcd|full>.npz

Contents (run in opera_disp_feb13; PyLith steps shelled out to the PyLith env as the notebook does):
- the 2386 samples (order of fig2_model_row.npz, asserted): sample_lon/lat, block_lon/lat (copied), data_mm, model_mm,
  residual_mm from gf/inversion_products<cd_tag>.npz (medium sfcvm);
- the fault for panel L: nodes_kmz (km from Point Pinole, depth km +) of every impulse vertex, faces, slip_vertex_mm,
  s0_km_pp = 0 (07's km = s0 + s applies directly); surface_km / surface_model_mm at the top-edge vertices;
- crossing_* copied from fig2_model_row.npz (same data);
- prof_{X,Y}_{fp,up}_d / _mm from ONE PyLith kinematic forward run of the inverted vertex slip (SimpleDB with the
  vertex slip, "nearest", same mesh and material as the GF run; cfg/fwd_kin.cfg pattern) with OutputSolnPoints at
  07's profile points (fig2_inputs prof_fp_{X,Y}_ends, every 10 m over +/- cross_m, |d| <= 5 m dropped) and at the
  2386 samples; fp = displacement . 07's look, up = vertical; zeroed on the median of |d| >= 1500 m as 07;
  validation: forward displacement at the samples vs the inversion's model_mm (= G x slip), must be < 0.05 mm rms;
- scalars: mw_30gpa, mw_sfcvm, mu_effective_gpa, sfcvm_version, rms_mm, surf_ratio_pct, sigma, strike_deg,
  tie_depth_km, provenance; bootstrap 2-sigma (mw_30gpa_2sig, mw_sfcvm_2sig, surf_ratio_pct_2sig) when
  gf/uncertainty_<run>_sfcvm.csv exists (rerun after pylith_sfcvm_tests.py to add them).
"""
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from TulePond_filepaths import common_paths
CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"
cd_tag = sys.argv[1] if len(sys.argv) > 1 else "_nuggetcd"
set_tag = sys.argv[2] if len(sys.argv) > 2 else ""                # "" (07's 2386 samples) or "_coh07" (coherence-0.7 field)
medium = sys.argv[3] if len(sys.argv) > 3 else "sfcvm"             # sfcvm (aligned SFCVM) or homog (30 GPa); same inversion
assert cd_tag in ("", "_nuggetcd") and set_tag in ("", "_coh07") and medium in ("sfcvm", "homog")
label = set_tag.lstrip("_") + ("_" if set_tag else "") + ("nuggetcd" if cd_tag else "full") + ("" if medium == "sfcvm" else "_homog")
run = "default" + set_tag + cd_tag
cd_tag = set_tag + cd_tag                                          # the products' suffix
n_procs = 4


def sh(cmd, cwd=HERE):
    print("$", cmd)
    out = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    assert out.returncode == 0, f"command failed ({out.returncode}):\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    return out.stdout


# ---------------------------------------------------------------- inputs
prod = np.load(os.path.join(HERE, "gf", f"inversion_products{cd_tag}.npz"), allow_pickle=True)
results = pd.read_csv(os.path.join(HERE, "gf", f"results_comparison{cd_tag}.csv")).set_index("model").loc[f"PyLith {medium}"]
settings = json.loads(str(prod["settings"])) if "settings" in prod.files else {}
gf_name = settings.get("run_names", {}).get(medium, f"gf_default_{medium}")
gnpz = np.load(os.path.join(HERE, "gf", f"G_{gf_name}.npz"))
row07 = np.load(os.path.join(common_paths["out_dir"], "07_slip_model_tests", "fig2_model_row.npz"))
fig2_in = np.load(os.path.join(common_paths["out_dir"], "06_event_field_geometry", "fig2_inputs.npz"))
lon, lat = prod["sample_lon"], prod["sample_lat"]
same_samples = not set_tag                                        # 07's samples: blocks copied from fig2_model_row.npz (asserted equal)
if same_samples:
    assert np.array_equal(row07["sample_lon"], lon) and np.array_equal(row07["sample_lat"], lat), "samples differ from fig2_model_row.npz"
else:                                                              # another sample set: its own blocks, saved by the notebook
    assert "block_lon" in prod.files and prod["block_lon"].shape == (lon.size, 4)
assert np.array_equal(gnpz["sample_lon"], lon)
model_mm, residual_mm = prod[f"model_{medium}_mm"], prod[f"residual_{medium}_mm"]
data_mm = prod["data_mm"] if "data_mm" in prod.files else model_mm + residual_mm
slip_mm = prod[f"slip_{medium}_mm"]                          # per impulse vertex, right-lateral, mm
imp_xyz, imp_km, faces = prod["impulse_vertex_xyz"], prod["impulse_km"], prod["sub_faces"]
fault_xyz, vertex_index, sign_rl = gnpz["fault_xyz"], gnpz["vertex_index"], float(gnpz["sign_rl"])
G = gnpz["G_rl_m_per_m"]
assert np.allclose(fault_xyz[vertex_index], imp_xyz)
assert np.allclose(G.dot(slip_mm), model_mm, atol=1e-6), "products' model is not G x slip"
strike = float(row07["strike_deg"])
look = np.array([np.sin(np.deg2rad(strike)), np.cos(np.deg2rad(strike)), 0.0])

# ---------------------------------------------------------------- the kinematic slip db (all fault vertices; locked = 0)
slip_all = np.zeros(len(fault_xyz)); slip_all[vertex_index] = slip_mm * 1e-3
np.savez(os.path.join(HERE, "gf", f"fig2_slip_{label}.npz"), xyz=fault_xyz, slip_rl_m=slip_all, sign_rl=sign_rl)
slip_db = f"spatialdb/fig2_slip_{label}.spatialdb"
print(sh(f"bash pylith_env.sh {HERE} python3 make_vertex_slip_db.py gf/fig2_slip_{label}.npz {slip_db}").strip())

# ---------------------------------------------------------------- output points: 07's profile points + the samples
to_mesh_from_field = pyproj.Transformer.from_crs(int(fig2_in["epsg"]), CRS, always_xy=True)
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
cross_m, far_m = float(fig2_in["cross_m"]), 1500.0
d_line = np.arange(-cross_m, cross_m + 1.0, 10.0)
d_line = d_line[np.abs(d_line) > 5.0]                       # 07 [13]: the trace itself is a discontinuity
frac = (d_line + cross_m) / (2 * cross_m)
names = [str(n) for n in fig2_in["names"]]
pts, pt_names = [], []
for name in names:
    xw, yw, xe, ye = fig2_in[f"prof_fp_{name}_ends"]
    px, py = to_mesh_from_field.transform(xw + frac * (xe - xw), yw + frac * (ye - yw))
    pts.append(np.c_[px, py]); pt_names += [f"{name}{k:04d}" for k in range(d_line.size)]
sx, sy = to_mesh.transform(lon, lat)
pts.append(np.c_[sx, sy]); pt_names += [f"s{k:04d}" for k in range(lon.size)]
pts = np.vstack(pts)
points_file = f"points/fig2_points_{label}.txt"
with open(os.path.join(HERE, points_file), "w") as fh:
    fh.write(f"# 07 [13] profile points ({', '.join(names)}: {d_line.size} each, +/- {cross_m:.0f} m every 10 m, |d| <= 5 m dropped) "
             f"+ the {lon.size} quadtree samples; make_fig2_product.py {time.strftime('%Y-%m-%d %H:%M')}\n# name x y z (m) in {CRS}\n")
    for n, (x, y) in zip(pt_names, pts):
        fh.write(f"{n} {x:.3f} {y:.3f} 0.0\n")

# ---------------------------------------------------------------- the forward run (same mesh and material as the GF run)
fwd_name = f"fwd_fig2_{label}"
mat_default = {"sfcvm": "cfg/mat_sfcvm.cfg", "homog": "cfg/mat_homog.cfg"}[medium]
mat_cfg = settings.get("media", {}).get(medium, mat_default) if isinstance(settings.get("media"), dict) else mat_default
mesh_file = "mesh/" + settings.get("mesh_file", "mesh_vertical_tet_default.msh")
if not os.path.exists(os.path.join(HERE, "output", f"{fwd_name}-points.h5")):
    t0 = time.time()
    sh(f"bash pylith_env.sh {HERE} caffeinate -ims pylith cfg/pylithapp.cfg {mat_cfg} cfg/fwd_kin_simpledb.cfg "
       f"--problem.defaults.name={fwd_name} "
       f"--problem.mesh_initializer.phases.read_mesh.reader.filename={mesh_file} "
       f"--problem.solution_observers.points.reader.filename={points_file} "
       f"--problem.interfaces.fault.eq_ruptures.rupture.db_auxiliary_field.iohandler.filename={slip_db} "
       f"--nodes={n_procs}")
    print(f"forward run {fwd_name}: {time.time() - t0:.0f} s")
with h5py.File(os.path.join(HERE, "output", f"{fwd_name}-points.h5"), "r") as h:
    dist, order = cKDTree(h["geometry/vertices"][:, :2]).query(pts)
    assert dist.max() < 1e-3, "output points do not match the points file"
    u = h["vertex_fields/displacement"][-1][order] * 1e3       # mm, 3 components
with h5py.File(os.path.join(HERE, "output", f"{fwd_name}-fault.h5"), "r") as h:
    f_xyz = h["geometry/vertices"][:]
    slip_fwd = sign_rl * h["vertex_fields/slip"][-1][:, 1] * 1e3
d_v, i_v = cKDTree(f_xyz).query(fault_xyz)
assert d_v.max() < 1e-6
slip_check = float(np.abs(slip_fwd[i_v] - slip_all * 1e3).max())
print(f"imposed slip vs inverted slip: max |diff| {slip_check:.2e} mm")
assert slip_check < 1e-4, "the forward run did not reproduce the inverted vertex slip"   # ASCII db precision ~1e-5 mm

# validation at the samples: forward (fault only) vs the inversion's model (= G x slip)
u_samples = u[-lon.size:]
fp_samples = u_samples.dot(look)
d_val = fp_samples - model_mm
val = dict(rms_diff_mm=float(np.sqrt(np.mean(d_val ** 2))), max_abs_diff_mm=float(np.abs(d_val).max()),
           rms_model_mm=float(np.sqrt(np.mean(model_mm ** 2))))
print(f"validation: forward at the samples vs G x slip: rms {val['rms_diff_mm']:.4f} mm (max {val['max_abs_diff_mm']:.3f}), "
      f"model rms {val['rms_model_mm']:.3f} mm")
assert val["rms_diff_mm"] < 0.05, "forward run and Green's-function prediction disagree at the samples"

# profiles
profiles = {}
k0 = 0
for name in names:
    u_line = u[k0:k0 + d_line.size]; k0 += d_line.size
    for component, vec in (("fp", look), ("up", np.array([0.0, 0.0, 1.0]))):
        vals = u_line.dot(vec)
        vals = vals - np.median(vals[np.abs(d_line) >= far_m])   # 07 [13]: zeroed on the far field
        profiles[f"prof_{name}_{component}_d"] = d_line
        profiles[f"prof_{name}_{component}_mm"] = vals
    step = profiles[f"prof_{name}_fp_mm"]
    print(f"{name}-{name}' model fault-parallel: west {step[d_line < 0].min():+.2f} mm, east {step[d_line > 0].max():+.2f} mm")

# ---------------------------------------------------------------- fault panel, surface profile, scalars
nodes_kmz = np.c_[imp_km, -imp_xyz[:, 2] / 1e3]
top = np.abs(imp_xyz[:, 2]) < 1e-6
order_top = np.argsort(imp_km[top])
potency = float(results.potency_m3)
m0_sfcvm = 10 ** (1.5 * float(results.Mw_sfcvm) + 9.1)
mu_eff = m0_sfcvm / potency / 1e9
out = dict(sample_lon=lon, sample_lat=lat, block_lon=(row07["block_lon"] if same_samples else prod["block_lon"]),
           block_lat=(row07["block_lat"] if same_samples else prod["block_lat"]),
           data_mm=data_mm, model_mm=model_mm, residual_mm=residual_mm,
           nodes_kmz=nodes_kmz, faces=faces, slip_vertex_mm=slip_mm, s0_km_pp=0.0,
           surface_km=imp_km[top][order_top], surface_model_mm=slip_mm[top][order_top],
           crossing_km=row07["crossing_km"], crossing_mm=row07["crossing_mm"], crossing_profile_km=row07["crossing_profile_km"],
           crossing_profile_mm=row07["crossing_profile_mm"], crossing_profile_err_mm=row07["crossing_profile_err_mm"],
           mw_30gpa=float(results.Mw_30gpa), mw_sfcvm=float(results.Mw_sfcvm), mu_effective_gpa=mu_eff,
           sfcvm_version="v26-0 aligned", rms_mm=float(results.rms_mm), surf_ratio_pct=float(results.surface_pct),
           sigma=float(results.sigma), strike_deg=strike, tie_depth_km=0.0, potency_m3=potency,
           rms_west300_mm=float(results.rms_west300_mm), rms_east300_mm=float(results.rms_east300_mm), chi2=float(results.chi2),
           forward_validation=json.dumps(val), **profiles)
# Mw30 range: the selected sigma and its L-curve grid neighbours (this run's lcurve csv), plus the +/- 50 m SFCVM
# alignment runs where they exist for this sample set and Cd (results_comparison_{e50,w50}<cd>.csv)
lc = pd.read_csv(os.path.join(HERE, "gf", f"lcurve_{medium}{cd_tag}.csv")).sort_values("sigma").reset_index(drop=True)
i_sel = int(np.nonzero(np.isclose(lc.sigma.values, float(results.sigma)))[0][0])
nb = lc.iloc[max(i_sel - 1, 0):i_sel + 2]
range_vals, basis = list(nb.Mw_30.values), [f"sigma {', '.join(f'{v:g}' for v in nb.sigma)} (L-curve grid neighbours)"]
cd_part = "_nuggetcd" if cd_tag.endswith("_nuggetcd") else ""
if not set_tag:
    for off in ("e50", "w50"):
        f_off = os.path.join(HERE, "gf", f"results_comparison_{off}{cd_part}.csv")
        if os.path.exists(f_off):
            r_off = pd.read_csv(f_off).set_index("model").loc[f"PyLith sfcvm_{off}"]
            range_vals.append(float(r_off.Mw_30gpa)); basis.append(f"SFCVM boundary 50 m {off[0].upper()} ({r_off.Mw_30gpa:.2f})")
mw_range = [float(min(range_vals)), float(max(range_vals))]
assert mw_range[0] <= float(results.Mw_30gpa) + 1e-9 <= mw_range[1] + 2e-9
unc = os.path.join(HERE, "gf", f"uncertainty_{run}_{medium}.csv")
boot_txt = "bootstrap 2-sigma not yet available"
if os.path.exists(unc):
    u2 = pd.read_csv(unc); s2 = u2[u2.kind == "boot_2sd"].iloc[0]
    out.update(mw_30gpa_2sig=float(s2.Mw_30gpa), mw_sfcvm_2sig=float(s2.Mw_sfcvm), surf_ratio_pct_2sig=float(s2.surface_pct),
               potency_2sig_m3=float(s2.potency_m3))
    boot_txt = f"bootstrap 2-sigma: Mw30 {s2.Mw_30gpa:.2f}, Mw_sfcvm {s2.Mw_sfcvm:.2f}, surface {s2.surface_pct:.0f} %"
out["mw_30gpa_range"] = np.array(mw_range)
out["mw_30gpa_range_basis"] = "; ".join(basis)
out["provenance"] = (f"make_fig2_product.py {time.strftime('%Y-%m-%d %H:%M')}: run {run}, medium {medium}, PyLith GF {gf_name}, "
                     f"Cd {settings.get('cd_mode', 'full' if not cd_tag else 'nugget')}, surface constraint ON, forward run {fwd_name}; {boot_txt}")
path = os.path.join(HERE, "gf", f"fig2_model_row_pylith_{label}.npz")
np.savez(path, **out)
summary = (f"{os.path.basename(path)}: {run}/{medium}, sigma {out['sigma']:g}, rms {out['rms_mm']:.2f} (W {out['rms_west300_mm']:.2f} / "
           f"E {out['rms_east300_mm']:.2f}) mm, surface {out['surf_ratio_pct']:.0f} %, Mw30 {out['mw_30gpa']:.2f}, Mw_sfcvm {out['mw_sfcvm']:.2f} "
           f"(mu_eff {mu_eff:.2f} GPa), X-X' fp west {profiles['prof_X_fp_mm'][d_line < 0].min():+.2f} / east "
           f"{profiles['prof_X_fp_mm'][d_line > 0].max():+.2f} mm, forward-vs-G rms {val['rms_diff_mm']:.3f} mm; "
           f"Mw30 range {mw_range[0]:.2f}-{mw_range[1]:.2f} ({out['mw_30gpa_range_basis']}); {boot_txt}")
print(summary)
with open(os.path.join(HERE, "gf", "fig2_products_summary.txt"), "a") as fh:
    fh.write(time.strftime("%Y-%m-%d %H:%M ") + summary + "\n")
