"""Chapter 4: Chapter-2-style statistics for the in-vivo study (Holm adjustment and effect sizes, 2026-11-04).

1. Adds to figures/aim3/sources/erg/erg_cohort_cooling_summary.csv the columns written by
   LivePigs-Tools erg_cohort.add_ch2_stats (CI90_low/high, g_paired, g_CI95_low/high_approx,
   p_holm_test_vs_control, Holm_family_size), computed from the columns already there
   (n_animals, Mean_T_minus_C, SD, p_test_vs_control). The rows are NOT re-copied from the study
   folder: the thesis sources deliberately hold out Pig 5 S3 and Pig 6 S2. Every earlier column is
   checked to be unchanged.
2. IOP test minus control per cooling duration from figures/aim3/sources/master_iop.csv:
   mean, 95 % and 90 % CI, paired t test, Hedges-corrected paired g (approx. 95 % CI).
3. Writes figures/aim3/aim3_stats.csv: every number of this kind quoted in Ch4
   (Cooling b_t / b_amp / op_amp at 8 and 25 cd.s/m2: p, p_Holm, g; Jet and RM restoration: 90 % CI;
   IOP: 90 % CI), for the primary set (animals 2-6) and the ring-about-9.5-C group (Table 4.x).

Holm: across the three cooling durations within Set x Comparison x Flash x Measure, on the paired
t test of the test eye's change against the control eye's change. g = d_z x J, d_z = mean / SD of
the paired differences, J = 1 - 3/(4(n-1)-1); 95 % CI from the large-sample normal approximation
SE(d_z) = sqrt(1/n + d_z^2/(2n)), only approximate at n = 3-5.

Run:  python code/figures/aim3_erg_stats.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from provenance import THESIS, record

TOOLS = os.environ.get("LIVE_PIG_TOOLS", str(THESIS / "code" / "ch4-live-pigs"))
sys.path.insert(0, TOOLS)
import erg_cohort as EC  # noqa: E402  (LivePigs-Tools: the same functions as the study analysis)

FIG3 = THESIS / "figures" / "aim3"
SUMM = FIG3 / "sources" / "erg" / "erg_cohort_cooling_summary.csv"
IOP = FIG3 / "sources" / "master_iop.csv"
OUT = FIG3 / "aim3_stats.csv"
SETS = {"primary": "primary", "ring 9.5": "Ring about 9.5 C (Pigs 2-4, Pig 5 S1)"}
NEW = ["CI90_low", "CI90_high", "g_paired", "g_CI95_low_approx", "g_CI95_high_approx",
       "p_holm_test_vs_control", "Holm_family_size"]


def augment():
    with open(SUMM, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        head = [h for h in rd.fieldnames if h not in NEW]
        rows = list(rd)
    old = [{k: r[k] for k in head} for r in rows]
    EC.add_ch2_stats(rows)
    assert all({k: r[k] for k in head} == o for r, o in zip(rows, old)), "earlier columns changed"
    with open(SUMM, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, head + NEW)
        w.writeheader()
        w.writerows(rows)
    return rows


def iop_rows():
    m = pd.read_csv(IOP)
    m = m[(m.Duration_min > 0) & (m.Pig > 1)].dropna(subset=["Test_minus_Control"])
    # 2026-11-08: session 3 of animal 3 left out of every measure (erg_cohort.EXCLUDED_SESSIONS)
    m = m[~((m.Pig == 3) & m.Session.str.startswith("Session 3"))]
    keep = {"primary": m,
            "ring 9.5": m[~((m.Pig == 6) | ((m.Pig == 5) & m.Session.str.contains("Session [2-3]")))]}
    out = []
    for sname, d in keep.items():
        for dur in (30, 60, 120):
            x = d[d.Duration_min == dur].Test_minus_Control.to_numpy(float)
            n, mean, sd, lo, hi = EC.ci95(list(x))
            t, p = EC.paired_t(list(d[d.Duration_min == dur].dTest), list(d[d.Duration_min == dur].dControl))
            l9, h9 = EC.ci90(n, mean, sd)
            g, gl, gh = EC.hedges_gz(n, mean, sd)
            out.append(dict(set=sname, comparison="IOP (Post - Pre)", measure="IOP_mmHg", flash="", duration=dur,
                            n=n, pigs=", ".join(map(str, d[d.Duration_min == dur].Pig)), mean_T_minus_C=mean,
                            CI95_low=lo, CI95_high=hi, CI90_low=l9, CI90_high=h9, p=p, p_holm=None,
                            g=g, g_CI95_low_approx=gl, g_CI95_high_approx=gh))
    return out


def main():
    rows = augment()
    out = []
    for sname, sval in SETS.items():
        # 2026-11-05: the primary measures are the b-wave implicit time and amplitude and the summed OPs,
        # each at 8 and 25 cd.s/m2 (Holm per measure and flash across the three durations); was b_t, b_amp at 25 only
        for comp in ("Cooling", "Jet restoration", "RM restoration"):
            for fl, meas in [(fl, m) for fl in ("25", "8") for m in ("b_t", "b_amp", "op_amp")]:
                for r in rows:
                    if r["Set"] == sval and r["Comparison"].startswith(comp) and r["Flash"] == fl \
                            and r["Measure"] == meas:
                        f = EC._f
                        out.append(dict(set=sname, comparison=r["Comparison"], measure=meas, flash=fl,
                                        duration=int(f(r["Duration_min"])), n=int(f(r["n_animals"])), pigs=r["Pigs"],
                                        mean_T_minus_C=f(r["Mean_T_minus_C"]), CI95_low=f(r["CI95_low"]),
                                        CI95_high=f(r["CI95_high"]), CI90_low=f(r["CI90_low"]),
                                        CI90_high=f(r["CI90_high"]), p=f(r["p_test_vs_control"]),
                                        p_holm=f(r["p_holm_test_vs_control"]), g=f(r["g_paired"]),
                                        g_CI95_low_approx=f(r["g_CI95_low_approx"]),
                                        g_CI95_high_approx=f(r["g_CI95_high_approx"])))
    out += iop_rows()
    d = pd.DataFrame(out)
    d.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(d.drop(columns=["pigs"]).round(3).to_string(index=False))
    record(__file__, [SUMM, OUT], [SUMM, IOP],
           origins={"figures/aim3/sources/erg/erg_cohort_cooling_summary.csv":
                    "aim3_erg_figures.py --refresh (rows of 2026-11-04 12:15, Pig 5 S3 and Pig 6 S2 held out); "
                    "columns CI90_*, g_*, p_holm_*, Holm_family_size added in place by this script with "
                    "LivePigs-Tools erg_cohort.add_ch2_stats"},
           notes="2026-11-04 Holm/effect sizes: Chapter-2-style statistics for Ch4.",
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
