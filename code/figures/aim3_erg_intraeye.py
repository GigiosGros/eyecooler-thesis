"""Chapter 4: change of each eye, test vs test, control vs control, and the two compared.

Output (figures/aim3/):
  erg_intraeye.csv   per set, comparison, flash, measure, duration: mean, SD, 95 % CI and paired
                     t test of the test eye's change, of the control eye's change and of their
                     difference, as values and in percent of the eye's own earlier value
  erg-effects-table.tex, erg-effects-4c-table.tex, erg-power-table.tex   the tables of Sec. 4.3 and 4.2
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
import re

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
# 2026-11-08: the main results are over both ring temperatures (animals 2-6, "primary" in the summary)
SUMM_SET = "primary"
MAIN = "animals 2-6"
CONTACT_CAL = {5: 2, 6: 1}


# 2026-11-08: session 3 of animal 3 (systemic change in both eyes; the animal died in recovery) is left
# out of every measure; = EXCLUDED_SESSIONS of erg_cohort.py
EXCLUDED = {(3, 3)}


def drop_excluded(d):
    sn = d.Session.str.extract(r"Session (\d+)", expand=False).astype(float)
    bad = pd.Series([(int(p), int(s)) in EXCLUDED if s == s else False for p, s in zip(d.Pig, sn)], index=d.index)
    return d[~bad]


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


def cell(d, st, comp, fl, m, dur, who, with_ci=None):
    """Value (95 % CI); percent, one cell of the effects table.
    2026-11-07 (simplified tables): the CI only on the test-against-control comparison, and not in the
    ring-about-4-C block (one or two sessions)."""
    q = d[(d.set == st) & (d.comparison == comp) & (d.flash == fl) & (d.measure == m) & (d.duration == str(dur)) & (d.who == who)]
    a, p = q[q.unit == "abs"].iloc[0], q[q.unit == "pct"].iloc[0]
    dig = 0 if m in ("b_amp", "op_amp") else 1
    if with_ci is None:
        with_ci = who == "TvT-CvC"
    ci = f" ({sg(a.lo, dig)}, {sg(a.hi, dig)})" if with_ci and a.n > 1 and not np.isnan(a.lo) else ""
    # 2026-11-08: the tables give the change in the unit of the measure only; the percentages stay in the text
    return f"{sg(a['mean'], dig)}{ci}"


def pcell(v):
    return "" if v is None or not np.isfinite(v) else ("$<$0.001" if v < 0.001 else f"{v:.3f}")


def pvals(summ, m, dur, fl=25):
    """p and p_Holm of the test-against-control comparison (the primary measures: b-wave and OPs, 8 and 25)."""
    q = summ[(summ.Set == SUMM_SET) & summ.Comparison.str.startswith("Cooling") & (summ.Flash == fl)
             & (summ.Measure == m) & (summ.Duration_min == dur)]
    if m not in ("b_t", "b_amp", "op_amp", "b_width") or q.empty:   # 2026-11-08: width too
        return ""
    return f"{pcell(q.p_holm_test_vs_control.iloc[0])}"


def gval(summ, m, dur, fl):
    """Hedges' g of the paired test-minus-control differences (g_paired of the cohort summary: d_z times the
    small-sample correction), as a point estimate (2026-11-08)."""
    q = summ[(summ.Set == SUMM_SET) & summ.Comparison.str.startswith("Cooling") & (summ.Flash == fl)
             & (summ.Measure == m) & (summ.Duration_min == dur)]
    if q.empty or not np.isfinite(q.g_paired.iloc[0]):
        return ""
    return sg(q.g_paired.iloc[0], 2)


def effects_table(d, iop, summ):
    """2026-11-07: three tables, each with both ring levels -- the primary measures (b-wave implicit time and
    amplitude, summed OPs) at 25 and at 8 cd.s/m2, and the secondary measures (b-wave width at 25 and 8, intraocular
    pressure). The CI is given on the test-against-control difference with the ring at about 9.5 C; the ring-about-4-C
    rows (one or two sessions) give the means only. p_Holm only for the primary measures at about 9.5 C."""
    head = "% Written by code/figures/aim3_erg_intraeye.py; do not edit."
    colspec = "L{2.4cm}R{0.6cm}L{1.8cm}L{1.8cm}L{4.0cm}R{1.0cm}R{0.9cm}"   # 2026-11-08: g column added
    th = ("\\textbf{Measure} & \\textbf{Min} & \\textbf{Test eye} & \\textbf{Control eye} & "
          "\\textbf{Test against control} & \\textit{p}\\textsubscript{Holm} & \\textit{g} \\\\")
    # 2026-11-08: one block, both ring temperatures together (animals 2-6); the ring-level split stays in
    # erg_intraeye.csv and is reported in the text only where it changes a result
    sections = [(MAIN, None, (30, 60, 120))]

    def erg_rows(st, durs, measures, with_p):
        out = []
        for m, fl, lab in measures:
            for k, dur in enumerate(durs):
                pv = pvals(summ, m, dur, fl) if with_p else ""
                out.append(f"{lab if k == 0 else ''} & {dur} & "
                           + " & ".join(cell(d, st, "cooling", float(fl), m, dur, w) for w in ("TvT", "CvC", "TvT-CvC"))
                           + f" & {pv} & {gval(summ, m, dur, fl)} \\\\")
        return out

    def table(measures, with_p, extra=None):
        L = [head, f"\\begin{{tabular}}{{{colspec}}}", "\\hline", th, "\\hline"]
        for k, (st, title, durs) in enumerate(sections):
            if k:
                L.append("\\hline")
            if title:
                L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\")
            L += erg_rows(st, durs, measures, with_p)
            if extra:
                L += extra(st, durs)
        return "\n".join(L + ["\\hline", "\\end{tabular}"]) + "\n"

    prim = lambda fl: [("b_t", fl, "Implicit time (ms)"), ("b_amp", fl, "Amplitude ($\\mu$V)"),
                       ("op_amp", fl, "Oscillatory potentials ($\\mu$V)")]

    cold_iop = (iop.Pig == 6) | ((iop.Pig == 5) & iop.Session.str.contains("Session [2-3]"))

    def iop_rows(st, durs):
        g = iop[(iop.Pig > 1) & (iop.Duration_min > 0)]
        if st == "ring 4":
            g = g[cold_iop]
        elif st == "ring 9.5":
            g = g[~cold_iop]
        out = []
        # 2026-11-08: p_Holm of the paired t test of the test eye's change against the control eye's, Holm across
        # the three session lengths, as for the ERG measures
        ph = {}
        if st != "ring 4":
            ps = {d: stats.ttest_rel(g[g.Duration_min == d].dTest, g[g.Duration_min == d].dControl).pvalue for d in durs}
            run = 0.0
            for i, d in enumerate(sorted(ps, key=ps.get)):
                run = max(run, min(1.0, (len(ps) - i) * ps[d]))
                ph[d] = run
        for k, dur in enumerate(durs):
            cs = []
            for col in ("dTest", "dControl", "Test_minus_Control"):
                x = stat(g[g.Duration_min == dur][col])
                ci = (st != "ring 4" and col == "Test_minus_Control" and np.isfinite(x["lo"]))
                cs.append(f"{sg(x['mean'])} ({sg(x['lo'])}, {sg(x['hi'])})" if ci else sg(x["mean"]))
            x = g[g.Duration_min == dur].Test_minus_Control.astype(float).dropna()
            n = len(x)
            gi = x.mean() / x.std(ddof=1) * (1 - 3 / (4 * (n - 1) - 1)) if n > 2 and x.std(ddof=1) > 0 else np.nan
            out.append(f"{'Intraocular pressure (mmHg)' if k == 0 else ''} & {dur} & " + " & ".join(cs)
                       + f" & {pcell(ph[dur]) if dur in ph else ''} & {sg(gi, 2) if np.isfinite(gi) else ''} \\\\")
        return out

    # 2026-11-08: the width goes with the other ERG measures of its flash; intraocular pressure has its own
    # table, with the IOP results (the file name erg-effects-secondary-table.tex is kept)
    width = lambda fl: [("b_width", fl, "Width at 80\\% (ms)")]
    t25 = table(prim(25) + width(25), True)
    t8 = table(prim(8) + width(8), True)
    sec = table([], True, iop_rows)
    return t25, t8, sec


def power_table(p):
    p = p[p.set == MAIN]   # 2026-11-08: both ring temperatures together, as the main results
    L = ["% Written by code/figures/aim3_erg_intraeye.py; do not edit.",
         "\\begin{tabular}{L{3.4cm}L{2.8cm}R{1.4cm}R{1.0cm}R{1.6cm}R{1.6cm}}", "\\hline",
         "\\textbf{Measure} & \\textbf{Comparison} & \\textbf{Mean change} & \\textbf{SD} & \\textbf{Change assumed} & \\textbf{Power, 5 animals} \\\\", "\\hline"]
    # 2026-11-07: one power column, five animals (the number the study treats as adequate); n=4, n=6 and the
    # n for 80 % stay in erg_power.csv
    who = {"TvT": "Test vs test", "CvC": "Control vs control", "TvT-CvC": "Test vs control"}
    pc = lambda v: "$>$99\\%" if not np.isfinite(v) or v > 0.99 else f"{100 * v:.0f}\\%"
    for lab, tex in (("implicit time (ms)", "b-wave implicit time (ms)"), ("implicit time (%)", "b-wave implicit time (\\%)"),
                     ("amplitude (%)", "b-wave amplitude (\\%)"), ("width at 80 % (ms)", "b-wave width at 80\\% (ms)"),
                     ("width at 80 % (%)", "b-wave width at 80\\% (\\%)")):
        for k, (w, wt) in enumerate(who.items()):
            r = p[(p.measure == lab) & (p.who == w)].iloc[0]
            L.append(f"{tex if k == 0 else ''} & {wt} & {sg(r.mean_change)} & {r.sd:.1f} & {r.effect:.1f} & {pc(r.power_n5)} \\\\")
    L += ["\\hline", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def main():
    e = pd.read_csv(EFF)
    e = drop_excluded(e[e.Recovery_eligible.astype(bool)].dropna(subset=["dTest", "dControl"]))
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
                               effect=size, power_n4=power(4, size / sd), power_n5=power(5, size / sd), power_n6=power(6, size / sd), n_for_80=n80(size / sd)))
    o1, o2 = FIG3 / "erg_intraeye.csv", FIG3 / "erg_power.csv"
    d = pd.DataFrame(rows)
    d["duration"] = d["duration"].astype(str)
    p = pd.DataFrame(pw)
    d.to_csv(o1, index=False)
    p.to_csv(o2, index=False)
    iop = drop_excluded(pd.read_csv(IOP))
    t1, t2 = FIG3 / "erg-effects-table.tex", FIG3 / "erg-power-table.tex"
    summ = pd.read_csv(SUMM)
    t1b, t1c = FIG3 / "erg-effects-8-table.tex", FIG3 / "erg-effects-secondary-table.tex"
    e1, e2, e3 = effects_table(d, iop, summ)
    t1.write_text(e1, encoding="utf-8")
    t1b.write_text(e2, encoding="utf-8")
    t1c.write_text(e3, encoding="utf-8")
    t2.write_text(power_table(p), encoding="utf-8")
    print(pd.DataFrame(pw).round(2).to_string(index=False))
    record(__file__, [o1, o2, t1, t1b, t1c, t2], [EFF, IOP, SUMM], origins={"figures/aim3/sources/erg/erg_cohort_effects.csv": "aim3_erg_figures.py --refresh"},
           notes="Within-eye changes and power by ring group.", manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
