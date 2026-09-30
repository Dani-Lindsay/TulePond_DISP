#!/usr/bin/env python3
"""
The fault-parallel mean event of notebook 06 at a different map coherence (sensitivity, 2026-09-30).

scripts/06_event_field_geometry.py is exec'd from its start up to and including the `fp_maps.npz` save, with these
text substitutions (each asserted to occur exactly once; scripts/06 and outputs/06 are not modified):
  - coh_maps = 0.9                       -> the requested coherence
  - out_dir / fig_dir                    -> drafts/pylith/coh07/coh<NN>/ (every file this span writes goes there)
  - the time-function fit cache          -> READ from 06's own outputs (asserted valid, never refitted or written there)
  - the saved-steps / burst-offset paths -> read from 06's own outputs (saved-products mode only)
  - REG = utils.Registry(...)            -> a no-op (nothing is written to outputs/registry.csv)
06 applies coh_maps only when it differs from coh_timeseries (0.7); at coh_maps 0.7 the maps therefore use the
fit mask itself, i.e. every pixel with temporal coherence >= 0.7 in both tracks.

Gate: run with 0.9 first; its fp_event must equal outputs/06_event_field_geometry/fig2_inputs.npz fp_event
(NaN pattern identical, values to 1e-6 mm).
Run in opera_disp_feb13 from the repo root:  python drafts/pylith/coh07/make_fp_coh.py 0.9   then   ... 0.7
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)
coh = float(sys.argv[1])
assert coh in (0.7, 0.9)
out = os.path.join(HERE, f"coh{int(round(coh * 10)):02d}")
os.makedirs(out, exist_ok=True)

src = open(os.path.join(REPO, "scripts", "06_event_field_geometry.py")).read()
end_key = 'np.savez(os.path.join(out_dir, "fp_maps.npz")'
assert src.count(end_key) == 1
cut = src.index("# %% [markdown]", src.index(end_key))                # the end of the fp_maps cell
head = src[:cut]


class _NoRegistry:
    """Stands in for utils.Registry: this sensitivity run registers nothing."""
    def __getattr__(self, name):
        return lambda *a, **k: None


subs = [
    ("coh_maps = 0.9 ", f"coh_maps = {coh} "),
    ('out_dir = paths_out[notebook]\n', f'out_dir = paths_out[notebook]\nin_dir = out_dir\nout_dir = {out!r}\n'),
    ('fig_dir = common_paths["fig_dir"]\n', f'fig_dir = {out!r}\n'),
    ('saved_steps = {o: os.path.join(out_dir, f"tulepond_steps_{o}.h5")', 'saved_steps = {o: os.path.join(in_dir, f"tulepond_steps_{o}.h5")'),
    ('    burst_df = pd.read_csv(os.path.join(out_dir, "burst_offsets.csv"))', '    burst_df = pd.read_csv(os.path.join(in_dir, "burst_offsets.csv"))'),
    ('    f_fit = os.path.join(out_dir, f"timefunc_fit_{orbit}.npz")', '    f_fit = os.path.join(in_dir, f"timefunc_fit_{orbit}.npz")'),
    ('    if cache_ok:\n        coef = ', '    assert cache_ok, "06 time-function cache unusable: refusing to refit into outputs/06"\n    if cache_ok:\n        coef = '),
    ('REG = utils.Registry(notebook, common_paths, repo_dir)', 'REG = _NoRegistry()'),
]
for a, b in subs:
    assert head.count(a) == 1, f"06 text not found exactly once: {a!r}"
    head = head.replace(a, b)
t0 = time.time()
ns = {"_NoRegistry": _NoRegistry, "__name__": "__coh_sensitivity__"}
exec(compile(head, "06_event_field_geometry.py (coh sensitivity)", "exec"), ns)
assert ns["out_dir"] == out and ns["coh_maps"] == coh
fp = np.load(os.path.join(out, "fp_maps.npz"))
n_fp = int(np.isfinite(fp["fp_event"]).sum())
# the per-track LOS mean event AFTER the coh_maps masking, with the geometry (for per-track statistics)
me, fits = ns["mean_event"], ns["fits"]
np.savez(os.path.join(out, "los_mean_event.npz"), lon=ns["lon_g"], lat=ns["lat_g"], strike=ns["strike"], coh_maps=coh,
         **{f"{o}_{k}": np.asarray(v) for o in ("ascending", "descending")
            for k, v in (("los_mm", me[o]["mean"]), ("inc", fits[o]["inc"]), ("azi", fits[o]["azi"]))})
print(f"coh {coh}: fp_event finite pixels {n_fp} ({time.time() - t0:.0f} s); wrote {out}/fp_maps.npz")

if coh == 0.9:                                                          # the reproduction gate
    ref = np.load(os.path.join(REPO, "outputs", "06_event_field_geometry", "fig2_inputs.npz"))["fp_event"]
    a, b = fp["fp_event"], ref
    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), "NaN pattern differs from 06's fig2_inputs fp_event"
    diff = float(np.nanmax(np.abs(a - b)))
    assert diff < 1e-6, f"fp_event differs from 06's by {diff} mm"
    print(f"PASS: coh 0.9 rebuild reproduces outputs/06 fig2_inputs fp_event ({n_fp} pixels, max |diff| {diff:.1e} mm)")
