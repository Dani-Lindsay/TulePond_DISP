"""Number every cell of a jupytext percent-format script in order: the first heading of a markdown cell
becomes '## [n] Title', and code cells get a first line '# [n]'. Existing [n] tags are replaced, so the
script can be re-run. Usage: python number_cells.py <script.py> [...]. Source: written for this project."""
import re, sys
for path in sys.argv[1:]:
    lines = open(path).read().split("\n")
    out, n, i = [], 0, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# %%"):
            n += 1
            out.append(line)
            i += 1
            if "[markdown]" in line:
                # copy the lines of the markdown cell
                while i < len(lines) and not lines[i].startswith("# %%"):
                    m = re.match(r"^(# #+ )(\[\d+\] )?(.*)$", lines[i])
                    if m and not any(l.startswith("# %%") or re.match(r"^# #+ \[\d+\] ", l) for l in out[len(out) - (i - (i)):]):
                        pass
                    out.append(lines[i])
                    i += 1
                # tag the first heading of the cell
                start = len(out) - 1
                while start >= 0 and not out[start].startswith("# %%"):
                    start -= 1
                for j in range(start + 1, len(out)):
                    m = re.match(r"^(# #+ )(?:\[\d+\] )?(.*)$", out[j])
                    if m:
                        out[j] = f"{m.group(1)}[{n}] {m.group(2)}"
                        break
            else:
                # replace an existing tag line with the new one
                if i < len(lines) and re.match(r"^# \[\d+\]$", lines[i]):
                    i += 1
                out.append(f"# [{n}]")
        else:
            out.append(line)
            i += 1
    open(path, "w").write("\n".join(out))
    print(f"{path}: {n} cells numbered")
