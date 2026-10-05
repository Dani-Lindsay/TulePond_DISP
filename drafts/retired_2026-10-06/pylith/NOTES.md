# PyLith 3D SFCVM Green's functions: running log

Overnight job started 2026-09-29 ~23:15 (unattended agent). Entries are timestamped (local time).

## 2026-09-29 23:15 setup
- Read PYLITH_PLAN.md, generate_gmsh_vertical.py, scripts/07_slip_model.py, drafts/07/sfcvm_extract_dense.py,
  drafts/07/antiplane_sfcvm_2d_aligned.py, PyLith examples (subduction-3d step07a + invert_slip.py,
  crustal-strikeslip-3d generate_gmsh.py/pylithapp.cfg, strikeslip-2d step05_greensfns.cfg).
- Machine: 10 cores, 24 GB RAM. PyLith env python3: numpy 1.26.4, gmsh 4.15.2, h5py 3.16.0, matplotlib 3.10.9;
  NO scipy, NO pyproj in the PyLith env (so CRS transforms and KD-trees are done in opera_disp_feb13).
- Trace in tmerc: km 52 (north end) at (x,y)=(-4597, 5919); km 56 (-2140, 2776); km 63 (1618, -3084);
  km 67 (south end) (3813, -6397). km from Point Pinole increases SOUTHWARD.
- 07 baseline (fig2_model_row.npz): 2386 samples, sigma 8, rms 1.952 mm, Mw 3.573 (30 GPa), strike 146.40 deg.
- Constraints from coordinator: use validated tools (PyLith gmsh_utils, spatialdata SimpleGridDB, PyLith GreensFns,
  CSI classes for the inversion); custom code = thin glue, listed in MORNING_SUMMARY.md. Deliverable notebook:
  fresh standalone drafts/pylith/pylith_sfcvm_gf.py (percent) paired to .ipynb; 07 not modified.

## 2026-09-29 23:20-23:35 Step 1: mesh (generate_gmsh_vertical.py)
- Diagnosis extended: not only the gmsh "Box" field hangs the 1D stage; ANY MathEval that references another
  MathEval (e.g. a separate cap field "Min(F3, 6000)") hangs it too (gmsh 4.15.2; mesh/debug/debug_fields2.py:
  single / Min-field-of-self-contained-MathEvals = 0.1 s; nested MathEval = hung > 60 s).
- Fix: three self-contained MathEval size fields (each capped at dx_far) combined by a gmsh "Min" field:
  whole fault (gmsh_utils get_math_progression, dx_fault), Tule Pond patch (dx_fine for y in [-3100, 2800] m =
  km 56-63, top 1.5 km, linear growth off the patch box), free surface over the data area (dx_surface). All
  linear growth size = dx + (bias-1)*distance, as get_math_progression is.
- Presets (`--preset`), times include improve_quality (gmsh_utils, 8 Relocate3D/optimize passes):
  | preset | dx fine/fault/surf/far (m), bias | tets | nodes | fault verts | patch edge median | min SICN | time |
  | coarse | 200/400/400/8000, 1.20 | 58,811 | 11,641 | 744 | 202 m | 0.434 | 3.5 s |
  | default | 100/250/250/6000, 1.15 | 192,634 | 36,094 | 2,364 | 101 m (74-125) | 0.318 | 11 s |
  | fine | 60/200/200/6000, 1.12 | 445,053 | 80,462 | 5,503 | 61 m | 0.336 | 33 s |
  Reports in mesh/mesh_vertical_tet_<preset>_report.txt. No inverted tets.

## 2026-09-29 23:35-23:48 PyLith GF pipeline tests
- make_points.py (opera env): 2386 samples -> points/insar_points.txt (tmerc, z=0), round-trip < 1e-9 deg.
- make_impulse_db.py (PyLith env, spatialdata SimpleGridAscii writer): impulse amplitude 1 m LL in a y window x depth.
- cfg/pylithapp.cfg (Dirichlet zero on all 3 components on sides+bottom, free top; length_scale 100 m;
  shear_modulus scale 10 GPa; P1 displacement), cfg/mat_homog.cfg (UniformDB rho 2700, vs 3333.33, vp 5773.50:
  mu 30 GPa, nu 0.25), cfg/gf.cfg (GreensFns, FaultCohesiveImpulses dof 1 (left-lateral), slip basis order 1,
  OutputSolnPoints at the samples).
- metadata block needed `arguments` etc. or pylith refuses to run.
- Timing (homogeneous, GMRES+GAMG defaults, ~11-13 KSP iterations per impulse):
  coarse mesh serial 1.4 s/impulse; default mesh serial 4.7 s/impulse (33 impulses 4:25 incl. ~60 s setup);
  default mesh `--nodes=4` 1.45 s/impulse (1:38 total). Serial vs np4 G identical to 3.6e-9 (max |G| 0.39).
  Profile (coarse, log_view): KSPSolve only ~0.07 s/impulse; the cost is residual evaluation / poststep
  (SNESFunctionEval ~1.2 s per impulse) and a one-off DMLocatePoints for the 2386 output points (27 s coarse).
- build_G.py (opera env): impulse->vertex by argmax (asserts single unit tent), points matched by KD-tree (< 1 mm),
  projection on 07's look = (sin 146.40, cos 146.40, 0), sign derived: right-lateral = -1 x PyLith left-lateral
  (uniform-slip east-minus-west step positive; all surface-vertex impulses positive individually).
  Note: a buried single-vertex tent's surface fault-parallel field has lobes of both signs, so per-impulse local
  east/west means are NOT a valid sign test (first version of the check failed on 2/19 buried impulses).
- 23:48 launched full homogeneous run: default mesh, impulses km 54-65 (y -4713..4380 m), full 3 km depth
  (bottom edge buried -> auto zero): 2105 impulses, --nodes=8 (run_gf.sh gf_default_homog).

## 2026-09-29 23:50-00:05 Step 2: materials (sfcvm_grid.py, write_sfcvm_griddb.py, plot_materials.py)
- Grid 165 x 197 x 80 = 2.60 M points: 50 m in x [-3500, 2500], y [-4000, 3600] (km ~55.3-63.6), growing x1.4 per
  step (max 4 km) to +/-50 km; z 25 m to 500 m, 50 m to 1.5 km, 100 m to 3 km, 250 m to 5 km, 1 km to 10 km,
  3 km to 40 km, plus a +100 m level. Sampling 78 s (one slab read per SFCVM block, nearest grid point as
  sfcvm_extract_dense.get_model_values). SimpleGridDB written by spatialdata's writer in 3 s (221 MB ASCII).
- Hayward boundary: first try used only blocks 22|7 -> found at 150/191 trace points. South of km ~65 the west
  block is 23 (Evergreen), so the boundary is {22 or 23}|7 -> found at 191/191 (1 point with 2 crossings, nearest
  used). b(km) east of trace: km 57.5-61.5 35..215 m (median 165); km 50-69 -295..215 m (west of the trace
  around km 55-57 and south of km 64). X-X' (km 58.7): b = 135 m.
- Shift: full within 3 km of the trace, linear taper to 0 at 6 km (far field = model as is); columns shifted
  whole (dip kept). Water/air/no-data (vs<=0): 2,485 points filled from the first valid value below.
  vs floor 150 m/s at 2,502 points; vp capped for nu <= 0.40 at 354,555 points (vs kept).
  mu 0.033..64.5 GPa; nu 0.215..0.400.
- Also wrote the unshifted grid (--no-align, for the figure). Bug fixed: the --no-align run had overwritten the
  boundary-offset csv with zeros; csv regenerated from the aligned npz and the script fixed (b_used).
- Figures: figures/fig_materials_sfcvm_v26.png (map at 50 m + X-X' section, as is vs aligned),
  figures/fig_materials_boundary_offset.png. The aligned section puts the soft-west / stiff-east contrast on the
  trace in the top ~250 m; below ~560 m the WEST side is the stiffer one (SFCVM west basement ~20 GPa vs
  ~5 GPa east), with the contrast east of the trace by ~100-500 m at 0.6-1.5 km depth (dip kept).
- SFCVM test (33 impulses, serial, default mesh): converges in 11-14 GMRES iterations (no solver trouble with the
  ~2000x rigidity contrast); G sign identical; ||G_sfcvm|| / ||G_homog|| = 1.27 for these shallow impulses.

## 00:05 notebook drafted: drafts/pylith/pylith_sfcvm_gf.py (percent format), cells [1]-[12].

## 00:20 homogeneous GF run done
- gf_default_homog: 2105 impulses, --nodes=8, 23:47:56 -> 00:19:42 (31.8 min incl. setup; ~0.9 s/impulse).
  GMRES iterations per impulse 10-14 (mostly 12). build_G: RL = -1 x LL; 90/90 surface impulses correct sign.
- 00:20 launched gf_default_sfcvm (aligned SFCVM), same mesh/impulses, --nodes=8.

## 00:25-00:45 validation (notebook cells [7], [7b]; homogeneous PyLith)
- Side check: with a proper perpendicular signed distance, 6 samples lie on opposite sides of the mesh fault and of
  07's trace, all within 15 m of the mesh fault (the mesh fault is a polyline through gmsh top-edge vertices on a
  spline through every 3rd trace point; median |difference in signed distance| 0.9 m, max 81 m).
  NOTE: 07's dist_to_trace(signed=True) signs by the x-offset to the nearest ~100 m trace VERTEX, which is wrong for
  samples within ~50 m of the trace between vertices (59 apparent mismatches before I noticed). Not used for sides here.
- (a) PyLith homog vs CSI Meade TDEs, 10 mm on km 57-61 top 1 km (PyLith vertex slip interpolated to CSI centroids):
  rms diff 0.265 mm of 2.18 mm signal (12%), BUT dominated by the 6 wrong-side samples (max 10.1 mm = a full step);
  farther than 300 m: rms diff 0.016 mm (0.7%). Potency 39,022 vs 39,259 m^3.
- (c) superposition: G x (forward-run vertex slip) = FaultCohesiveKin forward run to 1e-6 mm (exact).
- (d) forward runs vs CSI with identical slip, excluding the 6 wrong-side samples: default mesh rms 0.091 mm
  (within 300 m 0.101, farther 0.017, max 1.71 mm); fine mesh (60 m) rms 0.077 (0.086 / 0.011, max 1.21 mm);
  fine - default rms 0.083 mm. -> default mesh adequate; residual near-trace difference is P1 discretisation
  + trace geometry at the ~0.1 mm level, well below the 1.9 mm data rms.
- (b) vs 2D screw dislocation (km 54-65 top 1 km, samples km 57-62 within 2 km): rms diff 0.216 mm of 4.48 (4.8%);
  PyLith lower at 1-2 km as expected for an 11 km long patch (3D end effect), so this is only a sanity check.
- Homogeneous PyLith inversion (harness run, tent parameters): L-curve corner 9.6 -> sigma 12; rms 1.980 mm
  (07 CSI: 1.952), west 300 m 2.518 (07 2.467), east 300 m 1.619 (1.624), surface 75% (79%), Mw 3.51 (3.57).
  Like-for-like agreement with 07 -> the tent parameterisation + PyLith G reproduce the CSI baseline closely.
- 00:45 queued sensitivity run gf_default_sfcvmasis (SFCVM not shifted) after the aligned run.

## 00:53 SFCVM aligned GF run done; first comparison
- gf_default_sfcvm: 2105 impulses, 00:19:50 -> 00:53:08 (33 min), GMRES 11-17 its (mostly 14-15). Sign check
  90/90. Material gate: PyLith cell mu = grid mu at the cell centroids (median |rel diff| 0.000, 90th pct 0.003,
  40,630 cells in the fine box); PyLith cell mu range 0.072-64.5 GPa (the 0.033 GPa surface floor values are not
  reached at cell centroids). Fault-adjacent cells, top 100 m: west 0.50 GPa, east 1.32 GPa.
- CAVEAT (mesh): cells are basis-order-0 (one mu per cell, at the centroid). With 100 m fault cells and 250 m+
  surface cells, the SFCVM near-surface layering (25 m steps; west sediment ~200-550 m thick) is only coarsely
  represented away from the fault. -> 01:10 added mesh preset "shallow" (60 m band within 1 km of the fault, top
  600 m, along the patch): 484,753 tets, 87,566 nodes, 3,640 fault vertices, min SICN 0.356, 51 s to mesh.
  Queued gf_shallow_sfcvm then gf_shallow_homog (np 8) after the as-is sensitivity run.
- First inversion comparison (notebook harness, homog + aligned SFCVM):
  | model | sigma | rms | rms W300 | rms E300 | chi2 | surf % | potency m3 | Mw30 | Mw SFCVM mu |
  | CSI homog (07) | 8 | 1.952 | 2.467 | 1.624 | 3890 | 79 | - | 3.573 | 2.706 (07's) |
  | PyLith homog | 12 | 1.980 | 2.518 | 1.619 | 3981 | 75 | 7823 | 3.514 | 2.687 |
  | PyLith SFCVM aligned | 12 | 1.724 | 2.220 | 1.276 | 3954 | 74 | 8973 | 3.553 | 2.837 |
  The unweighted rms drops 13% with SFCVM, but chi2 (full data covariance) only 0.7%: the improvement is in
  directions the correlated-noise model down-weights. Both L-curve corners ~9.5 -> sigma 12.

## 01:27 as-is sensitivity run done; 01:40-02:00 notebook executed end to end (default mesh, 3 media)
- gf_default_sfcvmasis: 00:53:29 -> 01:26:30 (33 min), GMRES 11-17. Fault-adjacent mu top 100 m: west 0.52,
  east 0.48 GPa (as is, the contrast sits 35-215 m east of the fault, so both sides are soft next to the fault).
- 01:26 started gf_shallow_sfcvm (3373 impulses on the 60 m band mesh, np 8; ~2 s/impulse), gf_shallow_homog queued.
- Notebook pylith_sfcvm_gf.py paired (jupytext ipynb,py:percent) and executed with nbconvert (10.3 min, all PyLith
  products cached). Bugs fixed on the way: loop variable `tag` in [3] clobbered the output suffix (products were
  written as *asis.png; deleted, renamed to out_tag); sfcvm_grid.py path when rebuilding grids.
- Results (gf/results_comparison.csv; each at its own L-curve corner, all corners 9.2-9.6 -> sigma 12):
  | model | rms | rms W300 | rms E300 | chi2 | surf % | potency m3 | Mw30 | Mw(SFCVM mu) |
  | CSI homog (07) | 1.952 | 2.467 | 1.624 | 3890 | 79 | - | 3.573 | 2.706 |
  | PyLith homog | 1.980 | 2.518 | 1.619 | 3981 | 75 | 7823 | 3.514 | 2.687 |
  | PyLith SFCVM aligned | 1.724 | 2.220 | 1.276 | 3954 | 74 | 8973 | 3.553 | 2.837 |
  | PyLith SFCVM as is | 1.950 | 2.535 | 1.458 | 4141 | 74 | 9914 | 3.582 | 2.875 |

## 03:37 shallow-mesh SFCVM GF run done
- gf_shallow_sfcvm: 3373 impulses (60 m fault cells in the top ~600 m band), 01:26:39 -> 03:37:48 (2 h 11 min, np 8,
  ~2.3 s/impulse); GMRES 13-19 its (mostly 16-17). build_G: 144/144 surface impulses correct sign.
- 03:38 gf_shallow_homog started (same mesh/impulses), expected ~05:50. Then the notebook is executed with
  PYLITH_MESH_PRESET=shallow PYLITH_MEDIA=homog,sfcvm (products suffixed _shallow).
- 03:38-03:49 default notebook re-executed (tick-label fix): identical numbers (reproducible from cached outputs).
- 05:42 gf_shallow_homog done (03:37:29 -> 05:42:52, 2 h 05 min); build_G: 144/144 surface impulses correct sign,
  max |G| 0.55 (SFCVM shallow 0.77).
- 05:45 the Bash safety classifier returned no verdict several times in a row (transient infrastructure error);
  the shallow-mesh notebook run was delayed until it recovered.

## 2026-09-30 08:00-08:30 daytime session start (agent; Dani wants a dip decision today)
- Tasks: (1) one-page diagnostic per run, (2) Cd n_px floor, (3) 7 km swath sample set + GFs, (4) east-dipping faults
  80/70 in PyLith + SFCVM aligned to the dipping fault, (5) final table + recommendation. STATUS.md kept current.
- make_points_swath.py: 07's cells [1]-[4] exec'd with only the swath_km line replaced. Gate: swath 4 reproduces
  07's 2386 samples exactly (positions; data equal up to 07's fitted constant, which fig2_model_row.npz has removed).
- Pixel counts (07's km 52-67 window applied; the task's 104,994/132,447/143,936 were without it): within 4 km
  74,940; 6 km 99,595; 7 km 106,985; 8 km 111,048. West side has all the extra far field (east side ends at ~1-2 km
  beyond 4 km). Swath 7 -> 106,828 pixels (after the 30 m exclusion) -> 2396 quadtree samples: only 10 more than
  swath 4 (2386), because the quadtree starts from 2 km blocks, so the extra 32k pixels fill ~10 far blocks with
  large n_px (small nugget). 2382 of the swath-7 samples lie within 4 km, but the block grid differs from 07's
  (quadtree initial grid depends on the image extent), so it is not a superset of the 2386.
- Noise model with swath 7: nugget 0.809, short 0.856 mm @ 1.38 km, long 0 (swath 4: 0.797, 0.824 @ 1.22 km, long 0).
- run_gf.sh: optional 6th argument = points file. 08:25 launched queue_swath7.sh: gf_default_homog_swath7 then
  gf_default_sfcvm_swath7 (vertical, default mesh, np 8, ~33 min each).

## 08:15-08:30 step 1 done; dip meshes; SFCVM dip alignment
- plot_run_summary.py written and run for the 4 cached runs (8 one-pagers). Notebook now saves data_mm,
  signed_d_km, block polygons, trace, settings json in inversion_products*.npz and calls the plot at the end.
- generate_gmsh_dip.py: trace spline extruded along (h n_x, h n_y, -3000) with h = 3000/tan(dip) along the mean
  trace normal (07's wrap_fault construction); fault depth 3 km VERTICAL (down-dip length 3000/sin(dip)). Dip
  check from the fault triangles' normals: 80.0 E (193,784 tets, min SICN 0.278) and 70.1 E (196,858 tets, min SICN
  0.357), 11 s each. First version of the check reported 90-dip (arccos|nz| IS the dip); fixed.
- sfcvm_grid.py --dip: per-depth-level alignment (boundary found at each depth, nearest the fault's offset
  depth/tan(dip); shift = b(km,z) - f(z); below 3 km the 3 km shift; same 3-6 km horizontal taper). cfg/mat_sfcvm_dip80e/70e.cfg.
- 08:25 build_dip_grids.sh running (grids + griddbs); queue_dips.sh waits for the swath-7 queue then runs
  dip80e homog/sfcvm, dip70e homog/sfcvm at the DEFAULT 2386 points (decision: swath 7 adds only 10 samples; the
  dips must be comparable with the 8 cached vertical runs and 07's baseline).
- floor-15 notebook running (started 08:24).

## 08:25-08:40 step 2: Cd options on cached GFs (default mesh, vertical, surface ON unless stated)
- floor 15 (nugget / max(n_px,15)^0.5): chi2 x2.3, L-curve corner 6.8 -> sigma 8; at sigma 8 sfcvm rms 1.84 / W 2.44 /
  E 1.25; at sigma 12 (same as full) rms 1.74 vs 1.72 full -> the nugget floor does NOT change the fit; it only
  rescales chi2 and shifts the corner. floor 4: corner 8.6 -> sigma 12, sfcvm 1.73 / 2.23 / 1.27 (= full).
- diag Cd (equal weights, median variance) WITH the surface constraint: sfcvm sigma 12, rms 1.48 / W 1.76 / E 1.29,
  surface 94 %, Mw30 3.66, peak 11.1 mm at 59 m depth; homog 1.70 / 1.98 / 1.60, 90 %, Mw30 3.65.
  -> the CORRELATED part of 07's Cd (s1 0.82 mm at 1.2 km range) is what suppresses the west lobe, not the surface
  constraint and not the per-block nugget.
- 08:40 running PYLITH_CD=nugget: 07's per-block nugget variances, no correlation (diagonal), surface ON.
- The 4 dip GF runs are queued at the default 2386 points (queue_dips.sh, after the swath-7 queue).

## 08:45-08:55 step 2 conclusion; queues
- nugget-only diagonal Cd + surface ON: sfcvm rms 1.36 / W 1.50 / E 1.39, surface 98 %, Mw30 3.76, potency 18,073 m3
  (x2.0 full); homog 1.66 / 1.84 / 1.73, Mw30 3.70. Reduced chi2 ~5.6 (variances too small without the correlated
  terms). Conclusion: the correlated part of Cd suppresses the west lobe; recommend keeping 07's full Cd as
  primary and reporting nugget-only as the sensitivity. plot_run_summary.py: Cd label for 'nugget' (a KeyError had
  killed the one-pager after the products were written; pages regenerated).
- 80W mesh (80.0 W, min SICN ~0.3) + cfg made; 80W grid building; queue_dips_w.sh runs 80W after 70E.
- queue_inversions.sh: per config (swath7, dip80e, dip70e, dip80w) the notebook with full Cd then nugget Cd.

## 09:17-09:35 swath-7 inversions: lon bug, relaunch
- gf_default_sfcvm_swath7 done 09:17 (33 min). queue_inversions.sh started the swath-7 pair, both died in 30 s:
  CSI's insar.read_from_binary stores lon in 0-360 (237.9) while the quadtree samples / points files use -122, so
  the samples-vs-points assert failed. Fix: set samples.lon/lat back to the points' values after read_from_binary
  (x, y unaffected; asserted equal to the points' mesh coordinates). Relaunched the pair (queue_swath7_inv.sh);
  the dip pairs in queue_inversions.sh will use the patched notebook.
- Coordinator note (network outage) received; no run was interrupted (all queues nohup'd; floor-15/floor-4 done).

## 09:40 alignment check of the dip grids (largest log-mu jump along the normal at km 58.7 vs the fault position)
- vertical 'aligned' (columns shifted whole): contrast on the trace to ~600 m, then +385 m at 1 km, +765 m at 2 km
  east of the fault (the SFCVM boundary's own ~68 E dip).
- dip80e: jump +75/+195/+375 m at 0.4/1/2 km vs fault +71/+176/+353; dip70e: +155/+385/+745 vs +146/+364/+728;
  dip80w: -75/-175/-355 vs -71/-176/-353 -> the contrast follows the dipping fault to within the 50 m grid.
  (The detector picks other jumps at a few depths, e.g. dip80e z200; a display issue of the check, not of the grid.)
- swath-7 full-Cd inversion done 09:43 (sigma 12 both media).

## 09:50-10:00 step 3 done; control run queued
- swath-7 nugget Cd: sfcvm 1.37 / 1.51 / 1.39, potency 18,175 m3 (+0.6 % vs default nugget); homog 1.66 / 1.86 / 1.73.
  Step 3 conclusion: the 7 km swath changes nothing (10 extra samples, all far west); dips stay at the 2386 points.
- final_comparison.py written (Markdown by hand: no tabulate in the env, no installs); gf/final_comparison.csv.
- sfcvm_grid.py --dip 90v (vertical, per-depth alignment, f(z) = 0) + cfg/mat_sfcvm_dip90v.cfg + medium 'sfcvmv' in
  the notebook (control for the dip reading). queue_v.sh runs gf_default_sfcvmv after the 80W sensitivity.

## 10:25-10:40 order-2 (Charles Williams via coordinator; revised: keep slip order 1)
- PyLith 5.0.2 examples pair basis_order 2 (displacement, lagrange_multiplier_fault) with defaults.quadrature_order 2
  (strikeslip-2d step01c/step04, subduction-3d step08a); docs: quadrature order must be the same for all subfields.
  The docs recommend slip order 1 for GreensFns (order 0 = jumps at cell boundaries); Charles' revision keeps order 1.
- test_coarse_o2 (slip order 0, P2): PyLith accepts order-0 impulses (15 cell impulses, slip written as
  cell_fields/slip) -- not used after the revision. test_default_o2 (slip 0, P2, homog, np 2, alongside the 8-rank
  job): 12.7 s/impulse, setup ~5 min; killed after 16 impulses (timing obtained).
- cfg/order2.cfg: quadrature 2, displacement/lagrange order 2, materials order 1, slip order 1. run_gf_o2.sh.
  sfcvm_grid.py --no-cap (vs floor kept, vp as given) -> spatialdb/mat_sfcvm_v26-0_aligned_nocap.spatialdb (nu up
  to ~0.5). cfg/mat_sfcvm_nocap.cfg. Notebook: PYLITH_ORDER=2 (run names *_order2, sfcvm -> nocap cfg/grid;
  elastic_info read from vertex_fields when materials are order 1).
- Queue re-planned (queue_after_dips.sh; queue_dips_w.sh / queue_v.sh killed): after 70E -> order-2 sfcvm nocap ->
  order-2 homog -> 80W homog/sfcvm -> sfcvmv control.
- 10:40 test_default_o2_nocap: 20-impulse production-setup test (order2.cfg + nocap SFCVM, np 2).

## 10:42-10:52 dip 80E results; order-2 test failure
- dip80e full: homog 2.09 / 2.65 / 1.73, chi2 3952, potency 8,091; sfcvm 1.78 / 2.30 / 1.32, chi2 3959, potency
  8,308, Mw30 3.53 (2124 impulses, 34 min each). East dip: west lobe slightly worse for both (as expected).
- dip80e nugget: sfcvm 1.51 / 1.78 / 1.35 (sigma 8), homog 1.85 / 2.22 / 1.70.
- test_default_o2_nocap died in setup: SimpleGridDB 'could not find values for density at (23750, -50000, -4881)':
  materials order 1 -> db queried at the boundary vertices, which sit exactly on the grid's outermost node (+/-50 km).
  sfcvm_grid.py: grid_half 51 km, z to -41 km (new grids only; the cached capped/dip grids keep 50 km, fine for
  order-0 materials queried at centroids). Rebuilding the nocap grid/db and rerunning the test.

## 10:55 order-2 production-setup test passes
- test_default_o2_nocap (order2.cfg + mat_sfcvm_nocap, default mesh, np 2 alongside the 8-rank job): 33 impulses,
  setup ~5.5 min (P2 point location), then ~7 s/impulse; GMRES 23-26 its. elastic_info has vertex_fields
  (density, shear_modulus, bulk_modulus) -> the notebook reads the cell mean of the vertex values. fault/points
  outputs as order 1 (vertex slip tents, 2386 points) -> build_G unchanged.

## 11:00-11:40 order-2 tests (33 impulses, np 2 alongside the 8-rank job; G matched by impulse coordinates because
## the fault vertex numbering differs between serial and MPI runs)
| variant | ||G|| / P1 capped | surface / buried | / homog same order (surface) |
| P1 capped mat0 (production so far) | 1.000 | 1.00 / 1.00 | 1.27 (1.31) |
| P1 no-cap mat0 | 1.130 | 1.09 / 1.40 | 1.43 (1.42) |
| P2 capped mat0 | 0.608 | 0.57 / 0.83 | 1.09 (1.18) |
| P2 no-cap mat0 | 0.621 | 0.59 / 0.81 | 1.11 (1.22) |
| P2 no-cap mat1 (Charles' spec) | 0.647 | 0.60 / 0.90 | 1.16 (1.25) |
| P2 homog vs P1 homog | 0.708 | 0.63 / 1.00 | - |
- The material order is not the cause (mat0 vs mat1 at P2: 0.62 vs 0.65). The P1/P2 gap is mostly in the response
  of SURFACE tents at samples 0.3-1 km from the trace (0.70), and exists for the homogeneous medium too.
- Production order-2 runs on hold (queue_after_dips.sh killed); P2 forward validation (fwd_default_o2_homog,
  cfg/order2_fwd.cfg = order2.cfg without the impulse slip line, which FaultCohesiveKin rejects) running.
- 11:40 fwd_default_o2_homog (P2, cfg/order2_fwd.cfg): vs P1 forward rms 0.092 mm (|x|<300 m 0.056, 0.3-1 km 0.076),
  vs fine-P1 0.080 mm; signal 2.2 mm. P2 kinematics OK. Order-2 production relaunched (queue_after_dips.sh);
  fwd_default_o2_sfcvm (no-cap, mat1) started for the superposition check of the order-2 SFCVM GFs.
- 11:45 sum test on the 33 test tents (homog): all tents summed P2/P1 = 0.984 (rms diff 11 %); surface tents summed
  0.70, buried tents summed 1.23; per surface tent 0.5-1.0 with correlation 0.86-0.94. => P1 and P2 agree for a
  smooth slip patch (as the forward runs do) and differ in how a SINGLE 100 m tent at the surface vs just below
  it maps to the surface: a sub-cell discretisation effect (a one-cell tent is not resolved by 250 m surface cells
  at P1). Expected consequence: order-2 inversions may redistribute slip between the top row and the row beneath
  and change the surface-% and W/E numbers; the fitted field should change little. To be quantified by the run.

## 11:52-12:15 dip 70E; step 5 written
- dip70e full: homog 2.21 / 2.79 / 1.88, chi2 3954, potency 8,285; sfcvm 1.87 / 2.42 / 1.39, chi2 3950, potency 8,038
  (2227 impulses, 38 min each). Nugget: sfcvm 1.60 / 1.92 / 1.41 (sigma 8), homog 1.98 / 2.45 / 1.73.
- Dip trend monotonic: east dip hurts the west fit for both media; SFCVM keeps ~0.35 mm advantage; chi2 flat.
- STATUS.md step 5: full table (29 rows) + recommendation (homogeneous primary; vertical SFCVM as variant; no
  realistic-dip heterogeneous model fits the west lobe under 07's Cd). gf/final_comparison.csv.
- Order-2 SFCVM production: 5.2 s/impulse at np 8 (with a notebook running alongside), ETA ~14:50.

## 12:10 regression run failed; 16:32 resumed (API spend-limit stop ~11:50-16:32; PyLith queue kept running)
- notebook_default_regression (12:03): ValueError at the order-2 superposition line: `order` (the PYLITH_ORDER
  switch) was clobbered by `dist, order = cKDTree(...).query(...)` in the forward-run loop of [7b]. Renamed the
  switch `elem_order`. Runs that skip that loop (dips, swath 7) were unaffected; the default products predate the switch.
- Notebook now runs without a homog medium (ref_medium = first medium; validations (a)/(b)/(d) need homog and
  say so; superposition per medium against fwd_default[_o2]_<medium>).
- Logs: gf_default_sfcvm_order2 11:43 -> 14:27 (2 h 44 min, 2105 impulses, 4.7 s/impulse np 8);
  gf_default_homog_order2 started 14:27, ETA ~16:51. queue_after_dips.sh killed (80W / 90v not started, per coordinator).
- 16:42 order-2 SFCVM inversions launched (full, then nugget).
- 16:36 order-2 SFCVM inversion running: elastic_info (vertex materials) cell means vs grid: median 0.3 %, 90th pct
  37 % (smearing at the contrast); fault-adjacent mu top 100 m W 0.670 / E 1.094 GPa (order 0: 0.50 / 1.32).
- 16:45 order-2 superposition (G[sfcvm order2] x P2-forward vertex slip vs the P2 forward run fwd_default_o2_sfcvm):
  rms 0.085 mm (4 % of 2.1 mm), max 2.76 mm; NOT exact as at order 1. Split: within 300 m of a patch END (y)
  0.249 mm rms, inside the patch span 0.022 mm rms (0.5 %), max 0.056 mm. So the P2 forward run's patch ends are
  sharper than the tent sum (its "nearest" slip db is evaluated at the P2 edge nodes too), and the order-2 G is
  consistent with the forward solution away from those ends. Gate made explicit: order 1 exact (1e-4 mm), order 2
  < 5 % relative with a printed WARNING and the numbers recorded in the products. Inversions relaunched 16:38.
- 16:52 order-2 SFCVM full Cd: 1.77 / 2.27 / 1.32, chi2 4017, surf 72 %, potency 9,764, Mw30 3.58 (sigma 12).
- gf_default_homog_order2: 14:27 -> 16:50:33 (2105 impulses, np 8). Two-media order-2
  inversions launched (queue_o2_both.sh: full now, nugget after the SFCVM-only nugget run finishes).

## 17:35-17:55 new tasks (selected model = default_nuggetcd / sfcvm): tests companion, offset GFs, Fig-2 products
- Coordinator: order-2 nugget pair done by them (17:34), plot_test_summary.py added (now called at the end of every
  notebook run); 80W / 90v GF runs cancelled (queue_after_dips, queue_inversions, queue_o2_both killed).
- pylith_sfcvm_tests.py written and launched 17:37 (bootstrap n_boot 24 from 07, seeds 11 / 7 as 07 and its tests;
  recovery depths and scan depths parsed from scripts/07_slip_model_tests.py).
- sfcvm_grid.py --offset +/-50 -> aligned_e50 / aligned_w50 grids (51 km extent), dbs; gf_default_sfcvm_e50 started
  17:43 (np 8), w50 follows (queue_offset.sh).
- make_vertex_slip_db.py + make_fig2_product.py (+ cfg/fwd_kin_simpledb.cfg: fwd_kin.cfg's SimpleGridDB 'filename'
  property is rejected by SimpleDB, which uses iohandler.filename). Forward runs fwd_fig2_nuggetcd / _full (116 s,
  np 4): imposed slip = inverted slip to 5e-6 mm (ASCII db precision; gate 1e-4), forward at the samples = G x slip
  to 0.0000 mm rms. Products gf/fig2_model_row_pylith_{nuggetcd,full}.npz; summary lines in gf/fig2_products_summary.txt.

## 18:15-18:25 pylith_sfcvm_tests.py results (exit 0; 4 runs x 24 bootstrap realisations ~5.5 min each)
- Reference rows reproduce the tables (sfcvm nugget Mw30 3.756, full 3.553): the exec of the notebook is faithful.
- Bootstrap mean vs reference: nugget sfcvm 3.738 vs 3.756 (potency 0.94x), homog 3.718 vs 3.698; full sfcvm 3.411 vs
  3.553 (potency 0.61x), homog 3.397 vs 3.514. 07's own CSI bootstrap: 3.488 +/- 0.013 vs 3.573 (same sign, same cause:
  re-inverting a regularised prediction plus correlated noise loses deep moment).
- Recovery (seed 7): full Cd recovers 0.56 -> 0.12 of the moment for D 0.3 -> 2 km, Mw out 3.58-3.67 (07 CSI: 0.70 ->
  0.12, 3.63-3.69); nugget Cd 1.09 -> 0.68 (sfcvm), Mw within 0.11. West residual bias none for nugget (<= 0.11 mm vs
  noise 0.71 mm); full Cd: west -0.42 (sfcvm) / -0.92 (homog, D 1 km), east +0.6-1.6 mm (under-recovered deep slip).
- Depth scan: full sfcvm best 0.85 km, bracket 0.7-1.1 (rescaled by reduced chi2 1.71: 0.7-1.25 km), Mw 3.84-4.01;
  07 CSI smooth12 0.65-1.25, 3.82-4.00. Nugget: single-depth bracket (reduced chi2 7.1; diagonal Cd overstates information).
- make_fig2_product.py rerun: 2-sigma keys added (nugget: mw_30gpa_2sig 0.016, mw_sfcvm_2sig 0.071, surf 3.5 %;
  full: 0.026, 0.111, 4.2 %). final_comparison.csv refreshed (33 rows, bootstrap 2sd columns).
- Recovery figure panel (b): dotted input profiles drawn with slanted steps (bin-to-bin); fixed via plot_tests_figures.py.

## 18:15-18:30 alignment sensitivity (boundary 50 m E / W), inversions
- gf_default_sfcvm_e50 done ~18:19 (slowed by the tests companion); w50 follows (~18:53).
- Notebook: align_tag (media sfcvm_<tag> -> product suffix _e50 / _w50) so offset runs never overwrite the default
  products; the plot_test_summary.py call is wrapped so its failure prints a WARNING instead of discarding a run.
- e50 full (18:19-18:23): fault-adjacent mu top 100 m W 0.357 / E 0.476 GPa (both sides soft, the contrast is 50 m
  east); sigma 12.
- plot_test_summary.py (coordinator's) failed on medium 'sfcvm_e50' (KeyError in the marker dict). Minimal fix: markers
  for sfcvm_e50/w50 (diamond), the 4 offset tags added to the ladder order, one legend entry. Original kept in the
  scratchpad (plot_test_summary.py.orig). Redrawn: 33 runs.
- plot_tests_figures.py: recovery figure from the CSVs (panel b input drawn as an exact step); the tests script calls it.

## 19:10 end of session: alignment sensitivity done, STATUS final
- w50 GF 18:19-18:52; inversions: w50 fault-adjacent mu top 100 m W 1.326 / E 1.374 GPa (both stiff: the fault sits in
  the east block). full: 1.67 / 2.09 / 1.33, chi2 3835, Mw30 3.49 (sigma 12); nugget: 1.38 / 1.54 / 1.37, chi2 13968,
  Mw30 3.66 (sigma 8). e50 full 1.81 / 2.35 / 1.34, chi2 4051, 3.58; nugget 1.35 / 1.42 / 1.45, chi2 12703, 3.81 (sigma 16).
- Full Cd prefers W, nugget prefers E -> no robust preference; +/- 50 m moves the selected Mw30 by -0.10 / +0.05.
- final_comparison.csv 37 rows; summary_all_tests.png 36 runs; STATUS.md rewritten (Now, alignment section, table).
- Nothing running. Output dir output/ holds all PyLith HDF5 (several GB); delete test_*/fwd_* when no longer needed.

## 18:55-19:20 task 6: coherence-0.7 sensitivity (the east-side gap at km 60.2-60.8 is Lake Elizabeth, Central Park, Fremont)
- coh07/make_fp_coh.py: 06 exec'd to its fp_maps save with asserted substitutions (coh_maps; out/fig dir -> coh07/cohNN;
  fit cache READ from outputs/06 and asserted valid; Registry -> no-op). Gate: coh 0.9 rebuild = 06 fig2_inputs fp_event
  exactly (149,115 px, max diff 0). coh 0.7: 201,620 px (06 uses the coh_timeseries 0.7 fit mask directly). outputs/06
  and registry.csv mtimes unchanged (checked before/after). MintPy mode (stacks present).
- make_points_swath.py 4 coh07/coh07/fp_maps.npz coh07: 86,699 px -> 3148 samples; noise nugget 0.798, short 0.838 mm
  @ 0.42 km, LONG 1.404 mm @ 13.1 km (coh 0.9: 0.797, 0.824 @ 1.22 km, long 0): the coh-0.7 far field is noisier.
- 19:02 queue_coh07.sh: gf_default_sfcvm_coh07 then gf_default_homog_coh07 (np 8). Notebook: PYLITH_POINTS=coh07;
  make_fig2_product.py takes a set tag (_coh07; blocks from the products).
- coh07/strip_stats.py (east of trace 0-200 m, km 60.2-60.8; per-track implied fp = LOS / G_h from MintPy's
  get_design_matrix4horz_vert, no-vertical assumption; check: near-field correlation with 06's decomposed fp 0.988 / 0.984):
  coh 0.9: asc 2 px, desc 4 px, decomposed 2 px (median fp +2.95); coh 0.7: asc 27 px (LOS median +0.53, implied fp
  -2.0, 16-84 % -8.0..+0.3: wrong sign for the east block), desc 35 px (LOS +2.23, implied fp +4.76, 1.6..6.3),
  decomposed 21 px (median +2.0, -3.0..+3.8). Tracks disagree in the strip -> not clean signal (desc plausible).
  Samples east <= 300 m, km 60.2-60.8: coh 0.9 1 (0.69 mm at 254 m); coh 0.7 16 (median 0.68), of which 3 within
  40 m of the trace carry WEST-block values (-4.6, -6.7, -5.9 mm at 9-38 m): trace-location error of tens of metres.
  (Two medians equal across coherences were checked on the raw values: coincidences.)
- 19:20 the Bash safety classifier returned no verdict twice (transient); notes appended with the edit tool.
- 19:35 gf_default_sfcvm_coh07 done (33 min); homog_coh07 started. Early inversion coh07 sfcvm nugget (sigma 12; corner
  8.9): spike km 60.6-61.2 mean 10.09 / max 11.44 mm (coh 0.9: 10.20 / 11.95; crossings median 8.0, 6 crossings,
  7.4-9.5); Mw30 3.77, Mw_sfcvm 3.12, surface 95 %; native rms 1.71 / 1.80 / 1.83 on the 3148 samples.
- coh07/compare_coh.py: every model also scored on the SAME 2386 coh-0.9 samples (G at those points x the model's slip,
  mean constant removed): coh07 nugget sfcvm 1.35 / W 1.48 / E 1.35 vs coh-0.9 nugget sfcvm 1.35 / 1.48 / 1.37 ->
  equivalent models. The surface constraint has no running-median point in km 60.6-61.2 (nearest km 60.5 9.1 mm,
  km 61.5 5.0 mm; interpolated), so the spike is not pinned by it there.
- make_fig2_product.py: mw_30gpa_range (+ basis) = min/max over the L-curve neighbour sigmas and the +/- 50 m alignment
  runs (only for 07's sample set): selected 3.66-3.81, primary 3.46-3.62.

## 20:25 task 6 done
- homog_coh07 GF 19:35-20:08; inversions both media full (20:08) and nugget (20:13); Fig-2 coh07 nugget (20:18, exit 0,
  forward = G x slip exactly, Mw30 range 3.72-3.80 over sigma 8/12/16).
- coh07 full sfcvm: spike 6.41 / 7.58 mm, 2.03 / 2.48 / 1.70 own, 1.74 / 2.20 / 1.35 on 09 samples, Mw30 3.49; homog full
  6.44 / 7.81, Mw30 3.45; homog nugget (sigma 8) 8.57 / 9.60, Mw30 3.68.
- Conclusion: the spike goes with the nugget Cd, not the east gap (coh07 nugget sfcvm 10.1 / 11.4 vs 10.2 / 12.0; full Cd
  6.4-7.2 at either coherence; crossings 8.0). The coh07 selected-setup model is equivalent on the common samples.
- plot_test_summary.py: coh07 tags added to the ladder order (40 runs). final_comparison.csv 41 rows. STATUS rewritten.
- Nothing running (a leftover process count of 1 was the checking grep itself).
