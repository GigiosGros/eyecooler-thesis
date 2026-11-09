"""Chapter 4: the ERG at the start of each later session against session 1 (2026-11-08).

For each animal of animals 2-6 and each later session, the change of the test eye's Jet recording before cooling
from that of session 1, minus the same change of the control eye (Test_minus_Control of
figures/aim3/sources/erg/erg_cohort_between_sessions.csv, Electrode Jet, Animal_set primary), with the mean and 95 %
confidence interval over the animals. Rows: 25 and 8 cd.s/m2; columns: b-wave implicit time, b-wave amplitude, summed
oscillatory potentials. Session 3 of animal 3 is left out (Animal_set "excluded").

Output: figures/aim3/erg-between-sessions.png
Run:  python code/figures/aim3_between_sessions_figure.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from provenance import THESIS, record
from aim3_erg_figures import PIG_COLORS

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "erg" / "erg_cohort_between_sessions.csv"
OUT = FIG3 / "erg-between-sessions.png"
MEAS = [("b_t", "b-wave implicit time,\ntest minus control (ms)"), ("b_amp", "b-wave amplitude,\ntest minus control (µV)"),
        ("op_amp", "Oscillatory potentials (sum),\ntest minus control (µV)")]
SESS = [(2, "Session 2"), (3, "Session 3"), (4, "Terminal")]


def main():
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["Animal_set"] == "primary" and r["Electrode"] == "Jet"
            and r["Test_minus_Control"] not in ("", "None")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for i, fl in enumerate((25.0, 8.0)):
        for j, (m, lab) in enumerate(MEAS):
            ax = axes[i][j]
            ax.axhline(0, color="0.6", lw=0.7)
            for k, (sn, _) in enumerate(SESS):
                pts = [(int(r["Pig"]), float(r["Test_minus_Control"])) for r in rows
                       if float(r["Flash"]) == fl and r["Measure"] == m and int(r["Session"].split()[1]) == sn]
                for p, v in pts:
                    ax.plot(k - 0.08 + (p - 4) * 0.03, v, "o", ms=3.5, color=PIG_COLORS[(p - 1) % 6])
                vals = [v for _, v in pts]
                if len(vals) > 2:
                    mu = np.mean(vals)
                    h = stats.t.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals))
                    ax.errorbar(k + 0.18, mu, yerr=h, fmt="s", color="k", ms=3.5, capsize=2, lw=0.9)
                elif len(vals) == 2:   # two animals: a 95 % CI (t 12.7) says nothing, so the mean is drawn with its range
                    mu = np.mean(vals)
                    ax.errorbar(k + 0.18, mu, yerr=[[mu - min(vals)], [max(vals) - mu]], fmt="D", color="k", ms=3,
                                capsize=2, lw=0.9, mfc="white")
            ax.set_xticks(range(len(SESS)))
            ax.set_xticklabels([s for _, s in SESS], fontsize=7.5)
            ax.set_xlim(-0.5, len(SESS) - 0.5)
            ax.set_title(f"{fl:g} cd·s/m²", fontsize=8.5)
            if j == 0 or True:
                ax.set_ylabel(lab, fontsize=7.5)
            ax.tick_params(labelsize=7.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", color=PIG_COLORS[p - 1], label=f"Animal {p}") for p in range(2, 7)]
    h.append(Line2D([], [], marker="s", ls="none", color="k", label="Mean, 95% CI"))
    h.append(Line2D([], [], marker="D", ls="none", color="k", mfc="white", ms=3, label="Mean, range (two animals)"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    record(__file__, [OUT], [SRC], notes="2026-11-08: session 1 against the later sessions", manifest=FIG3 / "MANIFEST.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
