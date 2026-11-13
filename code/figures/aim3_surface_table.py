"""Chapter 4: grades of the ocular surface at each session, test and control eye (Table of Sec. 4.3.5).

Output (figures/aim3/):
  surface-grades-table.tex   median (range) of the grades of animals 2-6 before and after each cooling session and
                             at the terminal session: bulbar redness, limbal redness, corneal staining
Input: figures/aim3/sources/master_scores.csv (grades of record; the terminal grades sit in the Pre columns)
  Session 3 of animal 3 is left out, as in every measure of the chapter.

Run:  python code/figures/aim3_surface_table.py
"""
import numpy as np
import pandas as pd

from provenance import THESIS, record
from table_centring import centre   # 2026-11-12: header rows and first column centred

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "master_scores.csv"
OUT = FIG3 / "surface-grades-table.tex"
EXCLUDED = {(3, 3)}
MEAS = [("bulbar", "Bulbar redness"), ("limbal", "Limbal redness"), ("stain", "Corneal staining")]


def g(v):
    return f"{v:g}"


def cell(x):
    x = pd.Series(x, dtype=float).dropna()
    if x.empty:
        return "--"
    lo, hi, md = x.min(), x.max(), x.median()
    return g(md) if lo == hi else f"{g(md)} ({g(lo)}--{g(hi)})"


def main():
    d = pd.read_csv(SRC)
    d = d[d.Pig > 1]
    sn = d.Session.str.extract(r"Session (\d)", expand=False).astype(int)
    d = d.assign(S=sn)
    d = d[[(int(p), int(s)) not in EXCLUDED for p, s in zip(d.Pig, d.S)]]
    rows = []
    for s, lab in ((1, "1 (30 min)"), (2, "2 (60 min)"), (3, "3 (120 min)"), (4, "Terminal")):
        for when in (("Pre", "Post") if s < 4 else ("Pre",)):
            n = None
            cells = []
            for key, _t in MEAS:
                q = d[(d.S == s) & d.Measure.str.lower().str.contains(key)]
                for eye in ("Test", "Control"):
                    v = q[f"{when}_{eye}"]
                    cells.append(cell(v))
                    n = max(n or 0, int(v.notna().sum()))
            first = lab if (when == "Pre" or s == 4) else ""
            tl = "" if s == 4 else ("Before" if when == "Pre" else "After")
            rows.append(f"{first} & {tl} & {n} & " + " & ".join(cells) + " \\\\")
    L = ["% Written by code/figures/aim3_surface_table.py; do not edit.",
         "\\begin{tabular}{L{1.9cm}L{1.0cm}R{0.4cm}" + "L{1.45cm}" * 6 + "}", "\\hline",
         " & & & \\multicolumn{2}{l}{\\textbf{Bulbar redness}} & \\multicolumn{2}{l}{\\textbf{Limbal redness}} & "
         "\\multicolumn{2}{l}{\\textbf{Corneal staining}} \\\\",
         "\\textbf{Session} & & \\textit{n} & Test & Control & Test & Control & Test & Control \\\\", "\\hline"]
    L += rows + ["\\hline", "\\end{tabular}"]
    OUT.write_text(centre("\n".join(L) + "\n"), encoding="utf-8")
    print("\n".join(L))
    record(__file__, [OUT], [SRC], notes="Ocular surface grades, median (range), animals 2-6, by session.",
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
