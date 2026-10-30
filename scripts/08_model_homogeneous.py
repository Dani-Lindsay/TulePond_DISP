# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 08 Homogeneous slip model
#
# ```
# Author: Danielle Lindsay
# Environment: environment.yml. Runs after 07. About 10 min with the analytic Green's functions cached, about 2 h
# without.
# ```
#
# **What this notebook does**
#
# 1. **Homogeneous Green's functions** (section 3). The fault of the PyLith run `model_config.gf_run` (dipping 70° to
#    the west) in a homogeneous elastic half-space, with CSI triangular dislocations (Meade 2007).
# 2. **Inversion** (sections 4 to 6). The inversion of notebook 09 on these Green's functions: the same samples,
#    weights, plane, prior and corner rule. Only the elastic structure differs.
# 3. **Products** (section 7). The homogeneous comparison model of notebook 09 and Fig 2 (notebook 14).
#
# The settings are in `model_config.py` and the modelling functions in `model_utils.py`. No notebook runs PyLith: the
# fault geometry is read from the stored products of the run `model_config.gf_run` (`pylith/README.md`).
#
# **Inputs**: notebook 07 `model_data.npz`; the point list `modeling_inputs/points/insar_points.npz`; the stored G matrix
# of the PyLith run `model_config.gf_run` (its fault vertices); `hayward_fault_profiles.csv`.
#
# **Outputs** (`outputs/08_model_homogeneous/`): `homogeneous_lcurve.csv`, `homogeneous_model.npz` (slip, prediction
# and plane at the samples), `08_homogeneous_stats.txt`; the analytic Green's functions
# `modeling_uniform/gf/G_analytic_<run>_nsub<n>.npz` (cache). **Registry keys** `08_*`.
#
# **What to check**
# - Section 3: the n_sub table. The largest relative difference of the 20 test tents sets n_sub.
# - Section 5: the L-curve corner lies inside the sigma grid (asserted).
# - Section 6: the printed RMS of the fitted samples, the near-field RMS west and east, Mw at 30 GPa, and depth50 and
#   depth90.

# %% [markdown]
# ## 1. Imports, settings and the number registry
#
# The settings are printed from `model_config.summary()` so that the executed notebook records them.

# %%
import os
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyproj

warnings.filterwarnings("ignore", category=DeprecationWarning)   # library noise only; numerical warnings stay visible
warnings.filterwarnings("ignore", category=FutureWarning)
from TulePond_filepaths import common_paths, repo_dir, paths_out, paths_inputs, model_crs
import insar_utils as utils
import model_config as cfg                          # the settings of the slip models
import model_utils                                  # the modelling functions

notebook = "08_model_homogeneous"
out_dir = paths_out[notebook]
points_dir = paths_inputs["points"]
os.makedirs(out_dir, exist_ok=True)
made = time.strftime("%Y-%m-%d %H:%M")
model_data_file = os.path.join(paths_out["07_model_data"], "model_data.npz")
creep_lo_km, creep_hi_km = cfg.sse_section_km()
REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(model_data_file, common_paths["fault_profiles"], cfg.gf_files["G"])
print("\n".join(cfg.summary()))

# %% [markdown]
# ## 2. The samples and the trace
#
# The samples, the look vector, the strike and the noise model are read from notebook 07 (`model_data.npz`). The trace
# is rebuilt from the vertices saved there (`model_utils.model_trace`), as in notebook 09.

# %%
model_data = np.load(model_data_file)
model_frame_lonlat = tuple(model_data["frame_lonlat"])
strike_deg = float(model_data["strike_deg"])
look = model_data["look"]
cov_p = model_data["cov_p"]
positions_key = str(model_data["positions_fingerprint"])
profile_lats = pd.read_csv(common_paths["fault_profiles"]).ft_lat.values
trace = model_utils.model_trace(model_data["trace_lonlat"], -90.0, 90.0, model_frame_lonlat, profile_lats,
                                every_km=float(model_data["trace_every_km"]))
trace_xy_m = (trace["x_km"] * 1e3, trace["y_km"] * 1e3)
points_old = np.load(os.path.join(points_dir, "insar_points.npz"))
assert np.array_equal(points_old["lon"], model_data["sample_lon"]), "the PyLith point list is not on 07's samples"
print(f"{model_data['sample_lon'].size} samples, strike {strike_deg:.2f} deg, nugget {np.sqrt(cov_p[0]):.3f} mm")

# %% [markdown]
# ## 3. Homogeneous Green's functions on the PyLith fault
#
# The PyLith run `model_config.gf_run` defines the fault: its vertices, its triangles and its impulse vertices. Each
# impulse is a "tent", unit slip at one vertex that decreases linearly to zero at the neighbouring vertices.
# `model_utils.analytic_tent_G` computes the same tents in a homogeneous elastic half-space (Poisson solid) with CSI
# triangular dislocations (Meade 2007; Jolivet 2024). Each fault triangle is split into $n_{sub}^2$ sub-triangles of
# constant slip, and the Green's function of tent $j$ is
#
# $$\mathbf{g}_j = \sum_{t \ni j} \sum_{q=1}^{n_{sub}^2} \phi_j(\mathbf{c}_{tq})\, \mathbf{g}_{tq} ,$$
#
# where $t$ runs over the fault triangles that touch vertex $j$, $q$ over their sub-triangles, $\mathbf{c}_{tq}$ is the
# centroid of sub-triangle $q$ of triangle $t$, $\phi_j$ the tent function of vertex $j$ (1 at the vertex, 0 at the
# other two corners, linear in between), and $\mathbf{g}_{tq}$ the fault-parallel displacement at the samples (m) for
# 1 m of right-lateral slip on the sub-triangle.
#
# Choice of $n_{sub}$: 20 test tents in the SSE section (5 at the surface, 15 spread over depth) are computed with
# $n_{sub} = 4$ and $8$, and their relative difference is
#
# $$\varepsilon_j = \frac{\lVert \mathbf{g}_j^{(8)} - \mathbf{g}_j^{(4)} \rVert_2}{\lVert \mathbf{g}_j^{(8)} \rVert_2} ,$$
#
# where the norms run over the samples. If any $\varepsilon_j$ exceeds 1 %, $n_{sub} = 8$ is used, otherwise 4. The
# full matrix is cached and keyed by run, $n_{sub}$, sample positions and impulse vertices (`analytic_tent_G_cached`).
#
# Check: uniform right-lateral slip on every tent moves the east side along $+\hat{\mathbf{e}}$, as in the PyLith
# Green's functions.

# %%
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", model_crs, always_xy=True)
gf_npz = model_utils.load_G(cfg.gf_run, os.path.join(points_dir, "insar_points.npz"), strike_deg, cfg.gf_pylith_output_dir, cfg.gf_gf_dir)
tents = model_utils.TentFault(gf_npz, to_mesh, profile_lats)
in_section_tents = np.nonzero((tents.imp_km > creep_lo_km + 0.5) & (tents.imp_km < creep_hi_km - 0.5))[0]
surface_tents = in_section_tents[tents.imp_depth_km[in_section_tents] < 1e-6]
buried_tents = in_section_tents[tents.imp_depth_km[in_section_tents] > 1e-6]
buried_by_depth = buried_tents[np.argsort(tents.imp_depth_km[buried_tents])]
n_check = cfg.analytic_nsub_check_tents
test_tents = np.r_[surface_tents[np.linspace(0, surface_tents.size - 1, 5).astype(int)],
                   buried_by_depth[np.linspace(0, buried_by_depth.size - 1, n_check - 5).astype(int)]]
test_fault = SimpleNamespace(fault_xyz=tents.fault_xyz, fault_cells=tents.fault_cells, vertex_index=tents.vertex_index[test_tents],
                             n_imp=test_tents.size)
G_test = {}
for n_sub in cfg.analytic_nsub_options:
    G_test[n_sub] = model_utils.analytic_tent_G(test_fault, model_data["sample_lon"], model_data["sample_lat"], look, model_frame_lonlat, to_mesh, n_sub=n_sub)
n_coarse, n_fine = cfg.analytic_nsub_options
nsub_rel_diff = np.linalg.norm(G_test[n_fine] - G_test[n_coarse], axis=0) / np.linalg.norm(G_test[n_fine], axis=0)
nsub_table = pd.DataFrame(dict(tent_depth_m=np.round(tents.imp_depth_km[test_tents] * 1e3), tent_km=np.round(tents.imp_km[test_tents], 2),
                               rel_diff=nsub_rel_diff))
print(nsub_table.to_string(index=False))
n_sub = n_fine if nsub_rel_diff.max() > cfg.analytic_nsub_tolerance else n_coarse
print(f"largest n_sub {n_coarse} vs {n_fine} difference {nsub_rel_diff.max():.3f} "
      f"({'>' if n_sub == n_fine else '<='} {cfg.analytic_nsub_tolerance:g}): n_sub = {n_sub}")
analytic_cache = os.path.join(cfg.analytic_G_dir, f"G_analytic_{cfg.gf_run}_nsub{n_sub}.npz")
G_homogeneous = model_utils.analytic_tent_G_cached(analytic_cache, tents, model_data["sample_lon"], model_data["sample_lat"], look, model_frame_lonlat, to_mesh,
                                                   n_sub, cfg.gf_run, positions_key, n_workers=cfg.analytic_n_workers)
assert G_homogeneous.shape == gf_npz["G_rl_m_per_m"].shape, "analytic and PyLith G differ in shape"
signed_d_km = model_utils.signed_distance_m(trace_xy_m[0], trace_xy_m[1], points_old["x"], points_old["y"]) / 1e3
near_band = np.abs(signed_d_km) < 0.3
for label, G_check in (("analytic", G_homogeneous), ("PyLith 3D", gf_npz["G_rl_m_per_m"])):
    uniform = G_check.sum(axis=1)
    step = uniform[near_band & (signed_d_km > 0)].mean() - uniform[near_band & (signed_d_km < 0)].mean()
    print(f"{label}: uniform unit slip on every tent, east-minus-west within 300 m = {step:+.3f} m/m")
    assert step > 0, f"{label}: right-lateral slip does not move the east side along +look"
print(f"G homogeneous {G_homogeneous.shape} from {analytic_cache}")

# %% [markdown]
# ## 4. Inversion set-up (as in notebook 09)
#
# `model_utils.tent_model_setup` gives the tent fault, its CSI template (`TriangularTents` on the impulse vertices), each
# sample's signed distance from the trace and position along strike, `keep` (the fitted samples, 60 m or more from the
# trace) and the near-field masks (fitted samples within 500 m west or east of the trace in the SSE section). The samples
# are read from `model_data.npz` (`model_utils.samples_from_arrays`), as in notebook 09.
#
# The inversion (`model_utils.invert_tents`, CSI; Jolivet 2024) solves for the slip and a plane together:
#
# $$\min_{\mathbf{s} \ge 0,\ \mathbf{a}} \;
#   (\mathbf{d} - G\mathbf{s} - B\mathbf{a})^{T} C_d^{-1} (\mathbf{d} - G\mathbf{s} - B\mathbf{a})
#   + \mathbf{s}^{T} C_m^{-1} \mathbf{s}
#   + (\mathbf{a} - \mathbf{a}_0)^{T} C_a^{-1} (\mathbf{a} - \mathbf{a}_0) ,$$
#
# where $\mathbf{d}$ is the vector of fault-parallel samples (mm), $G$ the Green's functions (fault-parallel
# displacement per unit right-lateral slip, m/m), $\mathbf{s}$ the right-lateral slip at the impulse vertices (mm),
# $B$ the plane design matrix with rows $(1, x_i, y_i)$ ($x_i$, $y_i$ the sample position in km in the CSI frame),
# and $\mathbf{a}$ the plane coefficients (mm, mm/km, mm/km).
#
# - $C_d$ is the data covariance (`model_utils.data_cd`, mode "nugget" with exponent 0): diagonal, with the nugget
#   variance $n^2$ of notebook 07 (section 4) for every fitted sample (equal weights). Samples closer than 60 m to the trace are given
#   the variance $10^8 n^2$, so they stay in the products but do not influence the fit.
# - $C_m$ is the CSI exponential prior covariance of the slip (`Fault.buildCmXY`; Radiguet et al. 2011),
#
#   $$C_{m,jk} = \sigma^2 \,\frac{\lambda_0^2}{\lambda_h \lambda_v}\,
#     \exp\!\left(-\frac{h_{jk}}{2\lambda_h}\right) \exp\!\left(-\frac{v_{jk}}{2\lambda_v}\right) ,$$
#
#   where $\sigma$ is the regularisation strength (mm), $h_{jk}$ and $v_{jk}$ the horizontal and vertical distances
#   between impulse vertices $j$ and $k$ (km), $\lambda_h = 0.1$ km and $\lambda_v = 0.05$ km the correlation lengths
#   along strike and down dip (`model_config.lam_xz_km`), and $\lambda_0$ a normalising length that CSI computes from
#   the vertex spacing (km). CSI divides the distances by $2\lambda$, so the e-folding lengths are 0.2 km and 0.1 km.
# - $C_a = 10^4 I$ and $\mathbf{a}_0 = -10^3$ form a weak prior on the plane. CSI solves the system with
#   non-negative least squares on $\mathbf{m} - \mathbf{m}_0$, with $\mathbf{m} = (\mathbf{s}, \mathbf{a})$ and
#   $\mathbf{m}_0 = (\mathbf{0}, \mathbf{a}_0)$, so the slip is non-negative and the plane coefficients can take either
#   sign.
#
# No surface constraint is used (`model_config.use_surface`).

# %%
samples = model_utils.samples_from_arrays("fault-parallel samples", model_data["sample_lon"], model_data["sample_lat"],
                                          model_data["sample_x_km"], model_data["sample_y_km"], model_data["data_mm"],
                                          model_data["wgt"], look, model_frame_lonlat, cov_p, cfg.nugget_power)
setup = model_utils.tent_model_setup({"sfcvm": cfg.gf_run}, str(model_data["points_default"]), strike_deg,
                                     {"sfcvm": (cfg.gf_pylith_output_dir, cfg.gf_gf_dir)}, cfg.km_window, creep_lo_km, creep_hi_km,
                                     trace_xy_m, model_frame_lonlat, model_crs, profile_lats, cfg.near_trace_cut_km,
                                     cfg.near_field_km, trace["km_pp"])
assert np.array_equal(setup["tents"].imp_xyz, tents.imp_xyz), "the set-up's tents are not the tents of the analytic G"
keep = setup["keep"]
Cd = model_utils.data_cd(cfg.weighting, samples, cov_p, cfg.nugget_power, keep=keep)
print(f"{keep.sum()} fitted samples of {keep.size} (cut {cfg.near_trace_cut_km * 1e3:.0f} m); near field: "
      f"{setup['west_near'].sum()} west, {setup['east_near'].sum()} east; every fitted sample sd {np.sqrt(Cd[keep, keep][0]):.3f} mm")


def solve_homogeneous(sigma):
    """Solve the homogeneous model at one regularisation strength.

    Parameters
    ----------
    sigma : float
        Regularisation strength of the CSI exponential prior (mm).

    Returns
    -------
    tuple
        (fault, InSAR data set, None) from `model_utils.invert_tents`.
    """
    return model_utils.invert_tents(setup["template"], samples, G_homogeneous, Cd, None, None, sigma, cfg.lam_xz_km,
                                    model_frame_lonlat, use_surface=cfg.use_surface, name=f"homogeneous sigma {sigma:.3f}",
                                    ramp_spec=cfg.ramp_spec)

# %% [markdown]
# ## 5. L-curve and corner
#
# The regularisation strength is the corner of the model's own L-curve (`model_utils.own_corner`; Hansen 1992). For
# each $\sigma$ the L-curve takes the weighted misfit of the fitted samples and the prior-weighted roughness of the slip
# (`model_utils.tent_lcurve`),
#
# $$\chi^2 = \mathbf{r}_K^{T} C_{d,K}^{-1} \mathbf{r}_K , \qquad
#   R = \sqrt{\mathbf{s}^{T} \left(C_m / \sigma^2\right)^{-1} \mathbf{s}} ,$$
#
# where $\mathbf{r}_K = \mathbf{d} - G\mathbf{s} - B\mathbf{a}$ is the residual over the fitted samples $K$ (mm),
# $C_{d,K}$ the data covariance of those samples, and $C_m/\sigma^2$ the prior covariance of section 4 without
# $\sigma^2$. With $x = \log_{10}\sqrt{\chi^2}$ and $y = \log_{10} R$ as cubic splines of $\log_{10}\sigma$
# (`model_utils.lcurve_corner`), the curvature is
#
# $$\kappa = \frac{\lvert x' y'' - y' x'' \rvert}{\left(x'^2 + y'^2\right)^{3/2}} ,$$
#
# where the primes are derivatives with respect to $\log_{10}\sigma$. $\kappa$ is evaluated at 400 points between the
# second and the second-last $\sigma$ of the grid, and the corner is the $\sigma$ of maximum curvature. The corner is
# found first on the base grid (`model_config.sigma_grid_base`), then on the base grid plus nine values
# $\sigma_c \, 2^{k/16}$ ($k = -4, \dots, 4$) around the first corner $\sigma_c$. The model is solved at the second
# corner.

# %%
t0 = time.time()
sigma_hom, sigma_hom_base, lcurve_hom = model_utils.own_corner(solve_homogeneous, cfg.sigma_grid_base, cfg.sigma_refine_factors, tents,
                                                               cfg.mu_ref_pa, "homogeneous", keep)
lcurve_file = os.path.join(out_dir, "homogeneous_lcurve.csv")
lcurve_hom.assign(corner=sigma_hom, base_corner=sigma_hom_base).to_csv(lcurve_file, index=False)
print(f"homogeneous corner {sigma_hom:.2f} (base grid {sigma_hom_base:.2f}); {len(lcurve_hom)} solves in {(time.time() - t0) / 60:.1f} min; "
      f"wrote {lcurve_file}")

# %% [markdown]
# ## 6. The homogeneous model and its numbers
#
# The model is solved at the corner $\sigma$, and `model_utils.tent_numbers` gives its numbers.
#
# - RMS of the residual $r_i = d_i - (G\mathbf{s})_i - (B\mathbf{a})_i$ over a set of samples $S$,
#   $\mathrm{RMS}_S = \sqrt{\frac{1}{|S|}\sum_{i \in S} r_i^2}$ (mm), where $S$ is the fitted samples, or the fitted
#   samples within 500 m west or east of the trace in the SSE section.
# - Potency, $P = \sum_t A_t \bar{s}_t$ (m$^3$), where $A_t$ is the area of fault triangle $t$ (m$^2$) and $\bar{s}_t$
#   the mean of the slip at its three vertices (m). Because the slip is linear within each triangle, this is the
#   integral of the slip over the fault.
# - Seismic moment and moment magnitude (Hanks & Kanamori 1979), $M_0 = \mu P$ and
#   $M_w = \frac{2}{3}\left(\log_{10} M_0 - 9.1\right)$, where $\mu = 30$ GPa is the reference rigidity
#   (`model_config.mu_ref_pa`) and $M_0$ is in N m.
# - Peak slip, $\max_j s_j$ (mm), and the depth of its vertex (m).
# - depth50 and depth90, the depths above which 50 % and 90 % of the potency lie (`model_utils.potency_extent`):
#   $z_q = \min\{\, z_t : \sum_{t' :\, z_{t'} \le z_t} A_{t'} \bar{s}_{t'} \ge q P \,\}$, where $z_t$ is the depth of
#   the centroid of triangle $t$ (m) and $q$ is 0.5 or 0.9.
#
# CSI removes the fitted plane from the data, so the plane at each sample is the data minus the plane-removed data. It
# is checked to be a plane in $(x, y)$, and the CSI prediction is checked to equal $G\mathbf{s}$.

# %%
fault_hom, data_hom, _ = solve_homogeneous(sigma_hom)
slip_hom = fault_hom.slip[:, 0].copy()
numbers_hom = model_utils.tent_numbers(slip_hom, data_hom, tents, keep, setup["west_near"], setup["east_near"], cfg.mu_ref_pa)
plane_hom = samples.vel - data_hom.vel                                   # the fitted plane at every sample (CSI removes it from vel)
design = np.c_[np.ones(samples.vel.size), samples.x, samples.y]
plane_coef = np.linalg.lstsq(design, plane_hom, rcond=None)[0]
assert np.allclose(design @ plane_coef, plane_hom, atol=1e-6), "the removed reference frame is not a plane in (x, y)"
assert np.allclose(G_homogeneous @ slip_hom, data_hom.synth, atol=1e-6), "CSI's prediction is not G x slip"
for key, value in numbers_hom.items():
    print(f"  {key:18s} {value:.4g}")
print(f"  fitted plane: {plane_coef[0]:+.3f} mm {plane_coef[1]:+.3f} mm/km (x) {plane_coef[2]:+.3f} mm/km (y)")

# %% [markdown]
# ## 7. Products and registry
#
# `homogeneous_model.npz` holds the slip on the tents, the model prediction $G\mathbf{s}$ at every sample, the fitted
# plane, the plane-removed data, `keep` and $\sigma$. Notebook 09 reads it, and notebook 14 draws it in Fig 2. The
# numbers of the paper are registered with the stats file `08_homogeneous_stats.txt`.

# %%
model_file = os.path.join(out_dir, "homogeneous_model.npz")
np.savez(model_file, slip_mm=slip_hom, imp_xyz=tents.imp_xyz, model_mm=data_hom.synth.copy(), plane_mm=plane_hom,
         data_plane_removed_mm=data_hom.vel.copy(), keep=keep, sigma=sigma_hom, sigma_base_corner=sigma_hom_base, n_sub=n_sub,
         sample_lon=model_data["sample_lon"], sample_lat=model_data["sample_lat"], positions_fingerprint=positions_key,
         analytic_G=analytic_cache, **{k: v for k, v in numbers_hom.items()},
         provenance=f"{notebook} {made}: analytic (CSI Meade, n_sub {n_sub}) tents of {cfg.gf_run}; settings of model_config")
stats = REG.sidecar("08_homogeneous", cfg.summary() + [f"n_sub check: {nsub_table.to_dict('list')}",
                                                         f"sigma corner {sigma_hom:.3f} (base grid {sigma_hom_base:.3f})"]
                    + [f"{k} = {v}" for k, v in numbers_hom.items()])
where = "M8"
REG.register("08_analytic_nsub", n_sub, "count", 0, where, "section 3", stats)
REG.register("08_hom_sigma", sigma_hom, "mm", 1, where, "section 5", stats)
REG.register("08_hom_mw_30gpa", numbers_hom["Mw_30gpa"], "Mw", 2, where, "section 6", stats)
REG.register("08_hom_potency_m3", numbers_hom["potency_m3"], "m3", 0, where, "section 6", stats)
REG.register("08_hom_peak_slip_mm", numbers_hom["peak_slip_mm"], "mm", 1, where, "section 6", stats)
REG.register("08_hom_peak_depth_m", numbers_hom["peak_depth_m"], "m", 0, where, "section 6", stats)
REG.register("08_hom_depth50_m", numbers_hom["depth50_m"], "m", 0, where, "section 6", stats)
REG.register("08_hom_depth90_m", numbers_hom["depth90_m"], "m", 0, where, "section 6", stats)
REG.register("08_hom_rms_mm", numbers_hom["rms_mm"], "mm", 2, where, "section 6", stats)
REG.register("08_hom_rms_west500_mm", numbers_hom["rms_west_mm"], "mm", 2, where, "section 6", stats)
REG.register("08_hom_rms_east500_mm", numbers_hom["rms_east_mm"], "mm", 2, where, "section 6", stats)
REG.render()
print(f"wrote {model_file}")
