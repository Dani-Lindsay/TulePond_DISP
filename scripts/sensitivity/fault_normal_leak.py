"""fault_normal_leak.py: how much of the west-east asymmetry of the mean event survives if the second unknown of the
decomposition is fault-normal motion instead of vertical motion.

Author: Danielle Lindsay

The ascending and descending tracks give two numbers per place, so the decomposition of 06 solves for two unknowns,
fault-parallel (FP) and vertical (U), and assumes no fault-normal (FN) motion (Methods M2, M7). At the Tule Pond
geometry FN motion maps strongly into the returned FP (most of the strike vector points north, which Sentinel-1
barely sees), so the assumption matters for the asymmetry of the near-fault deflections (main text; Fig 2G). This
script measures, for each side of the trace in the SSE section, the near-fault LOS deflection of each track (the
mean over 0.06-0.25 km from the trace minus the mean over 1.5-3 km, model_config.near/far_deflection_km, as M7) and
solves the two deflections of that side twice:
  (FP, U)   the paper's assumption, FN = 0;
  (FP, FN)  the alternative, U = 0.
The (FP, U) solution must reproduce the deflection measured directly on 06's fault-parallel field (asserted, gate).
Uncertainties: a bootstrap over 0.5 km along-strike bins of the section (the noise is spatially correlated, so
pixels are not resampled one by one). The geometric leak gain (FP returned by the (FP, U) solution per mm of FN) is
reported. The same is done per 0.5 km bin along the section.

Inputs: 06 `tulepond_steps_<track>.h5` (per-track LOS mean event, masked at coherence 0.9 as the inverted field) and
`fp_maps.npz` (pixel positions, the fault-parallel field, the strike), the mapped trace, 04's SSE section.
Outputs (model_config.sensitivity_dir): fault_normal_leak.csv (per side and assumption), fault_normal_leak_bins.csv
(per 0.5 km bin). Notebook 10 draws and registers them.
Run from the repository root (seconds):
    PYTHONPATH=. python scripts/sensitivity/fault_normal_leak.py
"""
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

import common
from common import cfg, model_utils, utils
from TulePond_filepaths import common_paths, insar_dic_list, paths_out

SCRIPT = os.path.abspath(__file__)
BIN_KM = 0.5                    # along-strike bins: the bootstrap unit and the along-strike profile
N_BOOT = 2000
GATE_MM = 0.2                   # the (FP, U) solution must match the deflection of 06's fault-parallel field to this

# ---- 1. the per-track LOS mean event (coherence 0.9) on the pixels of 06's fault-parallel field
field_dir = paths_out["06_event_field_geometry"]
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
strike_deg = float(fp["strike"])
steps = {d["ORBIT_DIRECTION"]: os.path.join(field_dir, f"tulepond_steps_{d['ORBIT_DIRECTION']}.h5") for d in insar_dic_list}
masks = {d["ORBIT_DIRECTION"]: d["mask_file"][0.9] for d in insar_dic_list}
los_maps = model_utils.los_from_steps(steps, masks)
valid = np.isfinite(fp["fp_event"])
for track in ("ascending", "descending"):
    assert np.isfinite(los_maps[track]["mean"][valid]).all(), f"{track}: a pixel of the fault-parallel field has no LOS value"

# ---- 2. signed distance from the trace (east +) and km from Point Pinole of every pixel
trace = pd.read_csv(common_paths["fault_profiles"])                   # 791 rows, south to north, 100 m apart
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
tx, ty = to_utm.transform(trace.ft_lon.values, trace.ft_lat.values)
_, nearest = cKDTree(np.c_[tx, ty]).query(np.c_[fp["X"][valid], fp["Y"][valid]])
toward_east = np.deg2rad(trace.profile_az.values[nearest])           # bearing of the profile toward the east side
signed_d_km = ((fp["X"][valid] - tx[nearest]) * np.sin(toward_east) + (fp["Y"][valid] - ty[nearest]) * np.cos(toward_east)) / 1e3
km_pp = (len(trace) - 1 - nearest) * 0.1
section = cfg.sse_section_km()
in_section = (km_pp >= section[0]) & (km_pp <= section[1])
bins = np.floor((km_pp - section[0]) / BIN_KM).astype(int)
n_bins = int(np.ceil((section[1] - section[0]) / BIN_KM))

los = {track: los_maps[track]["mean"][valid] for track in los_maps}
unit = {track: utils.los_unit_vector(los_maps[track]["inc"][valid], los_maps[track]["azi"][valid]) for track in los_maps}   # (3, n): E, N, U
fp_pixels = fp["fp_event"][valid]

# ---- 3. the two decompositions of the deflections of one side
s = np.deg2rad(strike_deg)
DIRECTIONS = {"FP": np.array([np.sin(s), np.cos(s), 0.0]),                    # right-lateral: the east block toward the strike (SE)
              "U": np.array([0.0, 0.0, 1.0]),
              "FN": np.array([np.sin(s - np.pi / 2), np.cos(s - np.pi / 2), 0.0])}   # the east block away from the fault (NE)
ASSUMPTIONS = {"FP+U (FN = 0, the paper)": ("FP", "U"), "FP+FN (U = 0)": ("FP", "FN")}


def side_masks(side, use):
    """Near and far pixels of one side (-1 west, +1 east) among `use`."""
    distance = side * signed_d_km
    near = use & (distance >= cfg.near_deflection_km[0]) & (distance <= cfg.near_deflection_km[1])
    far = use & (distance >= cfg.far_deflection_km[0]) & (distance <= cfg.far_deflection_km[1])
    return near, far


def deflections(use):
    """Per side: the LOS deflection of each track, the mean LOS unit vector of its near pixels, and the deflection of
    06's fault-parallel field. Returns side -> dict, or None where a side has too few pixels."""
    out = {}
    for side in (-1, 1):
        near, far = side_masks(side, use)
        if near.sum() < 20 or far.sum() < 20:
            return None
        out[side] = dict(d=np.array([los[t][near].mean() - los[t][far].mean() for t in ("ascending", "descending")]),
                         unit=np.array([unit[t][:, near].mean(axis=1) for t in ("ascending", "descending")]),
                         fp_direct=fp_pixels[near].mean() - fp_pixels[far].mean())
    return out


def solve(side_values, pair):
    """The two components of `pair` from the two LOS deflections of one side."""
    G = side_values["unit"] @ np.c_[DIRECTIONS[pair[0]], DIRECTIONS[pair[1]]]
    return np.linalg.solve(G, side_values["d"])


# ---- 4. the section: both assumptions, the gate, the bootstrap over along-strike bins
whole = deflections(in_section)
for side in (-1, 1):
    fp_u = solve(whole[side], ASSUMPTIONS["FP+U (FN = 0, the paper)"])[0]
    assert abs(fp_u - whole[side]["fp_direct"]) < GATE_MM, \
        f"side {side}: (FP, U) gives {fp_u:.2f} mm, 06's field {whole[side]['fp_direct']:.2f} mm"
rng = np.random.default_rng(0)
boot = {(name, side): [] for name in ASSUMPTIONS for side in (-1, 1)}
for _ in range(N_BOOT):
    chosen = rng.integers(0, n_bins, n_bins)
    use = np.zeros_like(in_section)
    weight = np.bincount(chosen, minlength=n_bins)
    # a bin drawn k times enters k times: repeat its pixels by index
    index = np.concatenate([np.repeat(np.flatnonzero(in_section & (bins == b)), weight[b]) for b in range(n_bins) if weight[b]])
    values = {}
    for side in (-1, 1):
        distance = side * signed_d_km[index]
        near = index[(distance >= cfg.near_deflection_km[0]) & (distance <= cfg.near_deflection_km[1])]
        far = index[(distance >= cfg.far_deflection_km[0]) & (distance <= cfg.far_deflection_km[1])]
        values[side] = dict(d=np.array([los[t][near].mean() - los[t][far].mean() for t in ("ascending", "descending")]),
                            unit=np.array([unit[t][:, near].mean(axis=1) for t in ("ascending", "descending")]))
    for name, pair in ASSUMPTIONS.items():
        for side in (-1, 1):
            boot[(name, side)].append(solve(values[side], pair))

# the leak gain: FP returned by the (FP, U) solution per mm of FN, at the near-field geometry of each side
leak_gain = {}
for side in (-1, 1):
    G_fpu = whole[side]["unit"] @ np.c_[DIRECTIONS["FP"], DIRECTIONS["U"]]
    leak_gain[side] = float(np.linalg.solve(G_fpu, whole[side]["unit"] @ DIRECTIONS["FN"])[0])

rows = []
stamp = common.stamp(SCRIPT)
for name, pair in ASSUMPTIONS.items():
    for side, side_name in ((-1, "west"), (1, "east")):
        value = solve(whole[side], pair)
        draws = np.array(boot[(name, side)])
        rows.append(dict(assumption=name, side=side_name, fp_mm=value[0], fp_q05=np.quantile(draws[:, 0], 0.05),
                         fp_q95=np.quantile(draws[:, 0], 0.95), second=pair[1], second_mm=value[1],
                         second_q05=np.quantile(draws[:, 1], 0.05), second_q95=np.quantile(draws[:, 1], 0.95),
                         los_asc_mm=whole[side]["d"][0], los_desc_mm=whole[side]["d"][1], fp_direct_mm=whole[side]["fp_direct"],
                         leak_gain=leak_gain[side], provenance=stamp))
    ratio = np.abs(np.array(boot[(name, -1)])[:, 0] / np.array(boot[(name, 1)])[:, 0])
    west, east = solve(whole[-1], pair)[0], solve(whole[1], pair)[0]
    rows.append(dict(assumption=name, side="west/east", fp_mm=abs(west / east), fp_q05=np.quantile(ratio, 0.05),
                     fp_q95=np.quantile(ratio, 0.95), second=pair[1], provenance=stamp))
table = pd.DataFrame(rows)
os.makedirs(common.OUT, exist_ok=True)
table.to_csv(os.path.join(common.OUT, "fault_normal_leak.csv"), index=False, float_format="%.4f")

# ---- 5. along strike, per 0.5 km bin
bin_rows = []
for b in range(n_bins):
    sides = deflections(in_section & (bins == b))
    if sides is None:
        continue
    row = dict(km_lo=section[0] + b * BIN_KM, km_hi=min(section[0] + (b + 1) * BIN_KM, section[1]), provenance=stamp)
    for name, pair in ASSUMPTIONS.items():
        tag = "fpu" if pair[1] == "U" else "fpfn"
        for side, side_name in ((-1, "west"), (1, "east")):
            value = solve(sides[side], pair)
            row[f"{tag}_fp_{side_name}_mm"] = value[0]
            row[f"{tag}_{pair[1].lower()}_{side_name}_mm"] = value[1]
    bin_rows.append(row)
pd.DataFrame(bin_rows).to_csv(os.path.join(common.OUT, "fault_normal_leak_bins.csv"), index=False, float_format="%.4f")

print(f"strike {strike_deg:.2f} deg; SSE section km {section[0]}-{section[1]}; near {cfg.near_deflection_km} km, far "
      f"{cfg.far_deflection_km} km; bootstrap over {n_bins} bins of {BIN_KM} km")
print(f"leak gain (FP returned per mm of FN, FN = 0 decomposition): west {leak_gain[-1]:.2f}, east {leak_gain[1]:.2f}")
print(table.drop(columns="provenance").round(2).to_string(index=False))
print(f"{len(bin_rows)} along-strike bins written; outputs in {common.OUT}")
