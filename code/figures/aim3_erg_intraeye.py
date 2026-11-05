"""Chapter 4: change of each eye, test vs test, control vs control, and the two compared.

Output (figures/aim3/):
  erg_intraeye.csv   per set, comparison, flash, measure, duration: mean, SD, 95 % CI and paired
                     t test of the test eye's change, of the control eye's change and of their
                     difference, as values and in percent of the eye's own earlier value
  erg-effects-table.tex, erg-power-table.tex   the two tables of Sec. 4.3 and 4.2
  erg_power.csv      power of the paired t test at n animals, from the SDs over the sessions, for
                     amplitude, implicit time and width: test vs test, control vs control, and
                     test vs test against control vs control

Input: figures/aim3/sources/erg/erg_cohort_effects.csv (aim3_erg_figures.py --refresh)
  Sets: ring about 9.5 C (animals 2-4, animal 5 session 1), ring about 4 C (animal 5 from session 2,
  animal 6), and all of animals 2-6 without and with animal 1.
  Recovery recordings only >= 10 min after switch-off (Recovery_eligible).

Power: two-sided alpha 0.05, noncentral t. The effect assumed is 30 % for the amplitude and, for the
times, the observed mean change of the test eye over all sessions of the set. The SD is across
sessions, one per animal and duration, not across animals.

Run:  python code/figures/aim3_erg_intraeye.py
"""
import numpy as np
import pandas as pd
from scipy import stats

from provenance import THESIS, record

FIG3 = THESIS / "figures" / "aim3"
EFF = FIG3 / "sources" / "erg" / "erg_cohort_effects.csv"
IOP = FIG3 / "sources" / "master_iop.csv"
# 2026-11-04 Holm/effect sizes: p and Holm-adjusted p of the test-against-control comparison
# (paired t test; Holm across the three durations within each measure), columns added to the
# cooling summary by aim3_erg_stats.py; the set is the one of the table's first block
SUMM = FIG3 / "sources" / "erg" / "erg_cohort_cooling_summary.csv"
SUMM_SET = "Ring about 9.5 C (Pigs 2-4, Pig 5 S1)"
CONTACT_CAL = {5: 2, 6: 1}


def cold(d):
    return d.apply(lambda r: int(r.Session.split()[1]) >= CONTACT_CAL.get(r.Pig, 99), axis=1)


SETS = {"ring 9.5": lambda d: (d.Pig > 1) & ~cold(d), "ring 4": lambda d: (d.Pig > 1) & cold(d),
        "animals 2-6": lambda d: d.Pig > 1, "animals 1-6": lambda d: d.Pig > 0}
MEAS = ["b_t", "b_amp", "b_width", "b_peaks", "a_t", "a_amp", "op_amp"]
COMPS = {"Cooling": "cooling", "Jet restoration": "recovery", "RM restoration": "session", "Rewarming": "rewarming"}


def stat(x):
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    if n < 2:
        return dict(n=n, mean=x.mean() if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    m, s = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    p = stats.ttest_1samp(x, 0).pvalue if s > 0 else np.nan
    return dict(n=n, mean=m, sd=s, lo=m - h, hi=m + h, p=p)


def power(n, d, a=0.05):
    df = n - 1
    nc = abs(d) * np.sqrt(n)
    tc = stats.t.ppf(1 - a / 2, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)


def n80(d):
    return next((n for n in range(3, 300) if power(n, d) >= 0.8), np.nan)


def changes(d, pct):
    t, c = d.dTest.astype(float), d.dControl.astype(float)
    if pct:
        t, c = 100 * t / d.Test_earlier.abs(), 100 * c / d.Control_earlier.abs()
    return t, c


def sg(v, d=1):
    return f"${v:+.{d}f}$"


def cell(d, st, comp, fl, m, dur, who):
    """Value (95 % CI); percent, one cell of the effects table."""
    q = d[(d.set == st) & (d.comparison == comp) & (d.flash == fl) & (d.measure == m) & (d.duration == str(dur)) & (d.who == who)]
    a, p = q[q.unit == "abs"].iloc[0], q[q.unit == "pct"].iloc[0]
    dig = 0 if m in ("b_amp", "op_amp") else 1
    ci = f" ({sg(a.lo, dig)}, {sg(a.hi, dig)})" if a.n > 1 and not np.isnan(a.lo) else ""
    return f"{sg(a['mean'], dig)}{ci}; {sg(p['mean'], 0)}\\%"


def pcell(v):
    return "" if v is None or not np.isfinite(v) else ("$<$0.001" if v < 0.001 else f"{v:.3f}")


def pvals(summ, m, dur):
    """p and p_Holm of the test-against-control comparison (b-wave only, the primary measures)."""
    q = summ[(summ.Set == SUMM_SET) & summ.Comparison.str.startswith("Cooling") & (summ.Flash == 25)
             & (summ.Measure == m) & (summ.Duration_min == dur)]
    if m not in ("b_t", "b_amp") or q.empty:
        return " & "
    return f"{pcell(q.p_test_vs_control.iloc[0])} & {pcell(q.p_holm_test_vs_control.iloc[0])}"


def effects_table(d, iop, summ):
    names = [("b_t", "Implicit time (ms)"), ("b_amp", "Amplitude ($\\mu$V)"),
             ("b_width", "Width at 80\\% (ms)"), ("op_amp", "Oscillatory potentials ($\\mu$V)")]
    L = ["% Written by code/figures/aim3_erg_intraeye.py; do not edit.",
         "\\begin{tabular}{L{2.2cm}R{0.6cm}L{2.9cm}L{2.9cm}L{3.0cm}R{0.85cm}R{0.85cm}}", "\\hline",
         "\\textbf{Measure} & \\textbf{Min} & \\textbf{Test eye} & \\textbf{Control eye} & \\textbf{Test against control} & \\textit{p} & \\textit{p}\\textsubscript{Holm} \\\\", "\\hline",
         "\\multicolumn{7}{l}{\\emph{Ring about 9.5~$^\\circ$C}} \\\\"]
    for m, lab in names:
        for k, dur in enumerate((30, 60, 120)):
            L.append(f"{lab if k == 0 else ''} & {dur} & " + " & ".join(cell(d, "ring 9.5", "cooling", 25.0, m, dur, w) for w in ("TvT", "CvC", "TvT-CvC")) + " & " + pvals(summ, m, dur) + " \\\\")
    g = iop[(iop.Pig > 1) & (iop.Duration_min > 0) & ~((iop.Pig == 6) | ((iop.Pig == 5) & iop.Session.str.contains("Session [2-3]")))]
    for k, dur in enumerate((30, 60, 120)):
        cs = []
        for col in ("dTest", "dControl", "Test_minus_Control"):
            s = stat(g[g.Duration_min == dur][col])
            cs.append(f"{sg(s['mean'])} ({sg(s['lo'])}, {sg(s['hi'])})")
        L.append(f"{'Intraocular pressure (mmHg)' if k == 0 else ''} & {dur} & " + " & ".join(cs) + " & & \\\\")
    # 2026-11-05: durations and session counts of the 4 C group from the data (was 30 and 60 min, "one session each",
    # before animals 5 and 6 had their 120-min sessions)
    q4 = d[(d.set == "ring 4") & (d.comparison == "cooling") & (d.flash == 25.0) & (d.measure == "b_t")
           & (d.unit == "abs") & (d.who == "TvT")]
    n4 = {dur: int(q4[q4.duration == str(dur)].n.iloc[0]) for dur in (30, 60, 120) if (q4.duration == str(dur)).any()}
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    lo, hi = min(n4.values()), max(n4.values())
    nsess = f"{words[lo]} session{'s' if lo > 1 else ''} each" if lo == hi else f"{words[lo]} or {words[hi]} sessions at each duration"
    L += ["\\hline", f"\\multicolumn{{7}}{{l}}{{\\emph{{Ring about 4~$^\\circ$C, {nsess}}}}} \\\\"]
    for m, lab in names:
        for k, dur in enumerate(n4):
            L.append(f"{lab if k == 0 else ''} & {dur} & " + " & ".join(cell(d, "ring 4", "cooling", 25.0, m, dur, w) for w in ("TvT", "CvC", "TvT-CvC")) + " & & \\\\")
    L += ["\\hline", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def power_table(p):
    p = p[p.set == "ring 9.5"]
    L = ["% Written by code/figures/aim3_erg_intraeye.py; do not edit.",
         "\\begin{tabular}{L{3.0cm}L{2.4cm}R{1.1cm}R{0.8cm}R{1.2cm}R{1.2cm}R{1.2cm}R{1.2cm}}", "\\hline",
         "\\textbf{Measure} & \\textbf{Comparison} & \\textbf{Mean change} & \\textbf{SD} & \\textbf{Change assumed} & \\textbf{Power, n=4} & \\textbf{Power, n=6} & \\textbf{n for 80\\%} \\\\", "\\hline"]
    who = {"TvT": "Test vs test", "CvC": "Control vs control", "TvT-CvC": "Test vs control"}
    pc = lambda v: "$>$99\\%" if not np.isfinite(v) or v > 0.99 else f"{100 * v:.0f}\\%"
    for lab, tex in (("implicit time (ms)", "b-wave implicit time (ms)"), ("implicit time (%)", "b-wave implicit time (\\%)"),
                     ("amplitude (%)", "b-wave amplitude (\\%)"), ("width at 80 % (ms)", "b-wave width at 80\\% (ms)"),
                     ("width at 80 % (%)", "b-wave width at 80\\% (\\%)")):
        for k, (w, wt) in enumerate(who.items()):
            r = p[(p.measure == lab) & (p.who == w)].iloc[0]
            L.append(f"{tex if k == 0 else ''} & {wt} & {sg(r.mean_change)} & {r.sd:.1f} & {r.effect:.1f} & {pc(r.power_n4)} & {pc(r.power_n6)} & {r.n_for_80:.0f} \\\\")
    L += ["\\hline", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def main():
    e = pd.read_csv(EFF)
    e = e[e.Recovery_eligible.astype(bool)].dropna(subset=["dTest", "dControl"])
    rows, pw = [], []
    for sname, keep in SETS.items():
        s = e[keep(e)]
        for key, cname in COMPS.items():
            for fl in sorted(s.Flash.unique(), reverse=True):
                for m in MEAS:
                    for dur in (30, 60, 120, "all"):
                        d = s[s.Comparison.str.startswith(key) & (s.Flash == fl) & (s.Measure == m)]
                        if dur != "all":
                            d = d[d.Duration_min == dur]
                        if len(d) == 0:
                            continue
                        for unit, pct in (("abs", False), ("pct", True)):
                            t, c = changes(d, pct)
                            for who, x in (("TvT", t), ("CvC", c), ("TvT-CvC", t - c)):
                                rows.append(dict(set=sname, comparison=cname, flash=fl, measure=m, unit=unit,
                                                 duration=dur, who=who, **stat(x)))
        c = s[s.Comparison.str.startswith("Cooling") & (s.Flash == 25)]
        for m, lab, pct, eff in (("b_amp", "amplitude (%)", True, 30.0), ("b_amp", "amplitude (uV)", False, None),
                                 ("b_t", "implicit time (ms)", False, None), ("b_t", "implicit time (%)", True, None),
                                 ("b_width", "width at 80 % (ms)", False, None), ("b_width", "width at 80 % (%)", True, None)):
            t, cc = changes(c[c.Measure == m], pct)
            size = eff if eff else abs(t.mean())
            for who, x in (("TvT", t), ("CvC", cc), ("TvT-CvC", t - cc)):
                sd = x.std(ddof=1)
                pw.append(dict(set=sname, measure=lab, who=who, n_sessions=len(x), mean_change=x.mean(), sd=sd,
                               effect=size, power_n4=power(4, size / sd), power_n6=power(6, size / sd), n_for_80=n80(size / sd)))
    o1, o2 = FIG3 / "erg_intraeye.csv", FIG3 / "erg_power.csv"
    d = pd.DataFrame(rows)
    d["duration"] = d["duration"].astype(str)
    p = pd.DataFrame(pw)
    d.to_csv(o1, index=False)
    p.to_csv(o2, index=False)
    iop = pd.read_csv(IOP)
    t1, t2 = FIG3 / "erg-effects-table.tex", FIG3 / "erg-power-table.tex"
    summ = pd.read_csv(SUMM)
    t1.write_text(effects_table(d, iop, summ), encoding="utf-8")
    t2.write_text(power_table(p), encoding="utf-8")
    print(pd.DataFrame(pw).round(2).to_string(index=False))
    record(__file__, [o1, o2, t1, t2], [EFF, IOP, SUMM], origins={"figures/aim3/sources/erg/erg_cohort_effects.csv": "aim3_erg_figures.py --refresh"},
           notes="Within-eye changes and power by ring group.", manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
