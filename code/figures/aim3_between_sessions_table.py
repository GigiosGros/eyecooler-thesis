"""Chapter 4: start of each later session against session 1, test eye relative to the control eye (Table of Sec. 4.3.2).

Output (figures/aim3/):
  between-sessions-table.tex   mean change from session 1 of the Jet recording at the start of each later session, test
                               minus control, with its 90 % confidence interval, and the smallest Holm-corrected p of the
                               paired t test over the later sessions (b-wave implicit time and amplitude, summed OPs;
                               flashes 25 and 8)
Input: figures/aim3/sources/erg/erg_cohort_between_sessions.csv (copy of the LivePigs-Tools erg_cohort output)
  Animals 2-6 (Animal_set "primary"); session 3 of animal 3 is left out, as in every measure of the chapter.
  The tests are those of code/checks/ch4/between_sessions_test.py (one-sample t of test minus control, Holm across
  the later sessions within each measure and flash).

Run:  python code/figures/aim3_between_sessions_table.py
"""
import numpy as np
import pandas as pd
from scipy import stats

from provenance import THESIS, record
from table_centring import centre   # 2026-11-12: header rows and first column centred

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "erg" / "erg_cohort_between_sessions.csv"
OUT = FIG3 / "between-sessions-table.tex"
SESSIONS = [("Session 2 - 1hr", "Session 2"), ("Session 3 - 2hr", "Session 3"), ("Session 4 - End of Life", "Terminal")]
ROWS = [("b_t", 25.0, "Implicit time (ms)", 1), ("b_t", 8.0, "", 1),
        ("b_amp", 25.0, "Amplitude ($\\mu$V)", 0), ("b_amp", 8.0, "", 0),
        ("op_amp", 25.0, "Summed OPs ($\\mu$V)", 0), ("op_amp", 8.0, "", 0),
        ("b_width", 25.0, "Width at 80\\% (ms)", 1), ("b_width", 8.0, "", 1)]   # 2026-11-08: width added


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def sg(v, d):
    return f"${v:+.{d}f}$"


def main():
    d = pd.read_csv(SRC)
    d = d[(d.Animal_set == "primary") & (d.Electrode == "Jet")]
    ns = {s: d[d.Session == s].Pig.nunique() for s, _ in SESSIONS}
    body = []
    for meas, fl, lab, dig in ROWS:
        cells, ps = [], []
        for s, _ in SESSIONS:
            x = d[(d.Measure == meas) & (d.Flash == fl) & (d.Session == s)].Test_minus_Control.dropna().to_numpy(float)
            n, m = len(x), x.mean()
            sd = x.std(ddof=1) if n > 1 else np.nan
            if n > 1 and sd > 0:
                h = stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n)
                cells.append(f"{sg(m, dig)} ({sg(m - h, dig)}, {sg(m + h, dig)})")
                ps.append(stats.ttest_1samp(x, 0).pvalue)
            else:
                cells.append(sg(m, dig))
                ps.append(np.nan)
        ph = holm(ps)
        pmin = np.nanmin(ph) if np.isfinite(ph).any() else np.nan
        pc = "" if not np.isfinite(pmin) else f"{pmin:.2f}"
        body.append(f"{lab} & {fl:g} & " + " & ".join(cells) + f" & {pc} \\\\")
    L = ["% Written by code/figures/aim3_between_sessions_table.py; do not edit.",
         "\\begin{tabular}{L{3.0cm}R{0.9cm}L{2.9cm}L{2.9cm}L{2.9cm}R{1.1cm}}", "\\hline",
         "\\textbf{Measure} & \\textbf{Flash} & " + " & ".join(f"\\textbf{{{lab}}}\\newline (\\textit{{n}} = {ns[s]})" for s, lab in SESSIONS)
         + " & \\textit{p}\\textsubscript{Holm} \\\\", "\\hline"]
    L += body + ["\\hline", "\\end{tabular}"]
    OUT.write_text(centre("\n".join(L) + "\n"), encoding="utf-8")
    print("\n".join(L))
    record(__file__, [OUT], [SRC], notes="Between sessions: change from session 1, test minus control, 90 % CI, Holm p.",
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
