"""read_list.py: the list of input files a run of the notebooks read, from the log of the read audit.

Author: Danielle Lindsay

The read audit (sitecustomize.py in this folder) logs every file under the data root that a Python process opens. This
script keeps the files opened for reading that the run did not write itself (notebook outputs, figures and executed
copies are left out) and writes them, relative to the data root, one per line. scripts/zenodo_package.py packs the
inputs of the notebooks from this list.

Audited run (repo root; the hook is loaded from PYTHONPATH and is active only when TULEPOND_READ_LOG is set):
    PYTHONPATH=scripts/read_audit:. TULEPOND_READ_LOG=<log> TULEPOND_AUDIT_ROOT=<data root> TULEPOND_SUPERSEDE=1 ./run_all.sh
Then:
    python scripts/read_audit/read_list.py <log> <data root> <out list>
"""
import os
import sys

log, root, out = sys.argv[1], sys.argv[2].rstrip(os.sep) + os.sep, sys.argv[3]
GENERATED = ("outputs/", "Figures_", "executed/")      # written by the run itself
reads, writes = set(), set()
for line in open(log):
    parts = line.rstrip("\n").split(" ", 2)
    if len(parts) < 3 or not parts[2].startswith(root):
        continue
    rel = parts[2][len(root):]
    (writes if parts[1] == "write" else reads if parts[1] == "read" else set()).add(rel)
inputs = sorted(r for r in reads - writes if not r.startswith(GENERATED) and os.path.isfile(root + r))
with open(out, "w") as fh:
    fh.write("\n".join(inputs) + "\n")
print(f"{len(inputs)} input files read ({sum(os.path.getsize(root + r) for r in inputs) / 1e9:.2f} GB) -> {out}")
