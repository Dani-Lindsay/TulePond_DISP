"""Data S1 workbook (Science: one multi-sheet Excel file, a sheet per figure panel or table, P values included).
Inputs: the csv files and README.csv in OUT_DIR/Data_S1/ (written by build_data_S1.py, unchanged here), the five SM
tables parsed from the manuscript .tex (values exactly as printed), and the best storm rule's search-paid P from the
registry of the data root (TULEPOND_DATA_DIR; read only). Output: OUT_DIR/Data_S1.xlsx.

Run (repo root; needs pandas 2.2 and openpyxl, in environment.yml):
    TULEPOND_DATA_DIR=<data root> python scripts/data_S1/make_workbook.py OUT_DIR <manuscript .tex>
"""
import os, re, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from TulePond_filepaths import common_paths          # noqa: E402

OUT_DIR, TEX = os.path.realpath(sys.argv[1]), sys.argv[2]
for forbidden in ("FROZEN_", "SUBMITTED_"):
    assert forbidden not in OUT_DIR, f"refusing to write into {OUT_DIR}"
CSV = os.path.join(OUT_DIR, "Data_S1")
OUT = os.path.join(OUT_DIR, "Data_S1.xlsx")
tex = open(TEX).read()
reg = pd.read_csv(common_paths["registry_csv"], dtype=str).drop_duplicates("key", keep="last").set_index("key")

def strip_tex(s):
    s = s.replace("\\%", "%").replace("--", "-").replace("~", " ").replace("$", "")
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\^\{?\\?(circ|\*)\}?", lambda m: "°" if m.group(1) == "circ" else "*", s)
    s = s.replace("\\times", "x").replace("\\pm", "+/-").replace("\\alpha", "alpha").replace("\\mathrm{wet}", "wet")
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\footnotesize\s*", "", s)
    s = re.sub(r"[{}]", "", s).replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()

def table(label):
    """Rows of the tabular in the float carrying this label: header rows (before the 2nd \\hline) and data rows."""
    i = tex.index("\\label{" + label + "}")
    a = tex.rfind("\\begin{table}", 0, i); b = tex.index("\\end{table}", i)
    body = tex[a:b]
    cap = strip_tex(re.search(r"\\caption\{(.*?)\}\s*(\\label|\n)", body, re.S).group(1))
    tab = body[body.index("\\begin{tabular}"):body.index("\\end{tabular}")]
    tab = tab[tab.index("}", tab.index("{tabular}") + 9) + 1:]          # drop the column spec
    rows = [r for r in re.split(r"\\\\", tab)]
    parts, cur = [], []
    for r in rows:
        r = r.strip()
        if not r: continue
        segs = re.split(r"\\hline", r)
        for k, sgm in enumerate(segs):
            if k > 0: parts.append(cur); cur = []
            sgm = sgm.strip()
            if sgm: cur.append([strip_tex(c) for c in sgm.split("&")])
    parts.append(cur)
    parts = [p for p in parts if p]
    header = parts[0]; data = [r for p in parts[1:] for r in p]
    ncol = max(len(r) for r in header + data)
    cols = [" ".join(h[j] for h in header if j < len(h) and h[j]).strip() or f"col{j+1}" for j in range(ncol)]
    df = pd.DataFrame([r + [""] * (ncol - len(r)) for r in data], columns=cols)
    for c in df.columns:                                                 # numbers as numbers where the whole column is numeric
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() == (df[c] != "").sum() and conv.notna().any(): df[c] = conv
    return cap, df

readme = pd.read_csv(os.path.join(CSV, "README.csv"))
sheets = []
for f in readme.file:
    name = os.path.splitext(f)[0]
    assert len(name) <= 31 and not re.search(r"[\[\]:*?/\\]", name), name
    sheets.append((name, pd.read_csv(os.path.join(CSV, f))))
missing = sorted(set(os.listdir(CSV)) - set(readme.file) - {"README.csv"})
assert not missing, f"csv files without a README row: {missing}"

tabs = [("tableS1", "sup_tab:step_ci"), ("tableS2", "tab:cfw1_gittins"), ("tableS3", "sup_tab:sensitivity"),
        ("tableS4", "tab:onset_windows"), ("tableS5", "tab:forecast")]
extra = []
for name, lab in tabs:
    cap, df = table(lab)
    sheets.append((name, df)); extra.append(dict(file="", sheet=name, figure_panel="Table " + name[5:].upper(),
                  contents=cap, source_file="manuscript table (values as printed)", notebook=""))

pv = pd.read_csv(os.path.join(CSV, "figS12B_storm_rule_p_values.csv"))
pv.insert(0, "test", "storm rule: >= Z mm in 3 d within X d before the InSAR onset (one-sided, 20,000 season-preserving draws)")
# the best rule and its search-paid P, as registered (values as rendered in the registry: Z 40 mm, X 30 d, P 0.501)
n_rules = int(reg.loc["12_rain_sweep_n_settings", "rendered"])
assert n_rules == len(pv), f"{len(pv)} rules in the sweep table, the registry says {n_rules}"
search = pd.DataFrame([dict(test=f"best of the {n_rules} storm rules, accounting for the search (search-paid p)",
                            Z_mm=int(reg.loc["12_rain_sweep_best_Z_mm", "rendered"]), X_d=int(reg.loc["12_rain_sweep_best_X_d", "rendered"]),
                            p=float(reg.loc["12_rain_sweep_p_search_paid", "rendered"]))])
P = pd.concat([pv, search], ignore_index=True)
sheets.append(("P_values", P)); extra.append(dict(file="", sheet="P_values", figure_panel="fig. S12B; M9",
              contents="P values of the rainfall coincidence tests, one row per rule, and the search-paid P for the best rule "
                       "(registry 12_rain_sweep_p_search_paid)", source_file="figS12B_storm_rule_p_values.csv; registry", notebook="12_rain_seasonality"))

rd = readme.copy(); rd.insert(1, "sheet", [os.path.splitext(f)[0] for f in rd.file])
rd = pd.concat([rd, pd.DataFrame(extra)], ignore_index=True).drop(columns="file")
head = pd.DataFrame({"sheet": ["Data S1", "", "Paper", "Data and code", "Units", ""],
                     "figure_panel": ["", "", "", "", "", ""],
                     "contents": ["Data underlying the figures and tables of Lindsay and Bürgmann, Predictable slow slip events on the Hayward Fault? (Science, submitted 2026). One sheet per figure panel or table; this README lists every sheet.",
                                  "", "Sheet names give the figure and panel (for example Fig1E_peak_zone_stack or figS9_dip_sweep) or the table (tableS1).",
                                  "Full data, code and gridded maps: doi:10.5281/zenodo.23850500 (data and results v1.0), doi:10.5281/zenodo.23934127 (code v1.0.0).",
                                  "Units are in the column names; distances along the fault are km from Point Pinole; dates are decimal years or UTC dates.", ""],
                     "source_file": [""] * 6, "notebook": [""] * 6})
maps = pd.DataFrame({  # gridded maps are not tabulated; the Data S1 caption points here for their files in the data deposit
    "sheet": ["Gridded maps"] * 5,
    "figure_panel": ["Figs. 1A, S5A", "Figs. 2A, 2B, 2E, 2F; fig. S11A", "fig. S7", "fig. S1", "fig. S2C"],
    "contents": ["Fault-parallel velocity map: outputs.zip / outputs/02_velocity_decomposition/fp_velocity_map.h5",
                 "Inter-event and mean-event fault-parallel and vertical fields: outputs.zip / outputs/06_event_field_geometry/fig2_inputs.npz",
                 "Per-event line-of-sight step maps: outputs.zip / outputs/06_event_field_geometry/tulepond_steps_ascending.h5 and tulepond_steps_descending.h5",
                 "Coherence, data density and velocity maps per track: the frame stacks 09156_stack.nc, 09157_stack.nc, 11115_stack.nc and gnss_velocity.zip / mosaic_velfield/ascending_velocity_NA.h5 and descending_velocity_NA.h5 (maps drawn by notebook 01_prep_figures)",
                 "Ascending frame-overlap difference: computed from 09156_stack.nc and 09157_stack.nc by notebook 01_prep_figures"],
    "source_file": ["doi:10.5281/zenodo.23850500"] * 5, "notebook": ["02", "06", "06", "01", "01"]})
rd = pd.concat([head, rd, pd.DataFrame({c: [""] for c in rd.columns}), maps], ignore_index=True)

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    rd.to_excel(xw, sheet_name="README", index=False)
    for name, df in sheets:
        df.to_excel(xw, sheet_name=name, index=False)
    for ws in xw.book.worksheets:
        ws.freeze_panes = "A2"
        if ws.title == "README":
            for col, w in zip("ABCDE", (28, 14, 110, 50, 28)): ws.column_dimensions[col].width = w
print(f"wrote {OUT}: {len(sheets) + 1} sheets, {os.path.getsize(OUT) / 1e6:.1f} MB")
for name, lab in tabs:
    print(name, dict(sheets)[name].shape, list(dict(sheets)[name].columns)[:8])
