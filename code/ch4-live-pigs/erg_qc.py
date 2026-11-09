#!/usr/bin/env python3
"""
Trace-by-trace quality check of every ERG used in the cohort analysis.

Reads ERG Cohort Analysis/erg_traces.npz (written by erg_cohort.py: our averaged trace through
the 75 Hz low-pass) and the recordings table, and writes to ERG Cohort Analysis/:

  erg_qc.csv                      one row per trace: the checks below, and a verdict
  fig_qc_pigN_<jet|rm>_flashF.png contact sheets - every trace in its own panel,
                                  flagged panels red;
                                  at flashes 25 and 8 the OP trace under each (2026-10-28)

Checks (each flags; none changes a value). Only our own values are used (2026-11-09: the RETeval's
reported trace and cursor values are not read, so there is no comparison with the device):
  baseline     |mean of the 20 ms before the flash| > 10 uV       (offset error)
  pre-noise    SD of the 20 ms before the flash > 15 uV
  late level   |mean 150-200 ms| > 150 uV                          (drift / slow potential)
  pick         a- or b-wave pick on the edge of its search window  (from the recordings table)

    python erg_qc.py
"""
from pathlib import Path
import csv
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parent
STUDY = TOOLS.parent
D = STUDY / "ERG Cohort Analysis"
LIM = dict(baseline=10.0, pre_noise=15.0, late=150.0)
EXPLORATORY_PIGS = {1}          # = erg_cohort.EXPLORATORY_PIGS
OP_FLASHES = ("8", "25")        # = erg_all.OP_FLASHES


def main():
    sys.path.insert(0, str(TOOLS))
    z = np.load(D / "erg_traces.npz", allow_pickle=False)
    grid, keys, ours = z["grid"], list(z["keys"]), z["ours"]
    with open(D / "erg_cohort_recordings.csv", encoding="utf-8") as fh:
        rec = {(r["Pig"], r["Session"], r["Group"], {"Test": "RE", "Control": "LE"}[r["Eye"]], r["Flash"]): r
               for r in csv.DictReader(fh)}
    pre = (grid >= -20) & (grid < 0)
    late = (grid >= 150) & (grid <= 200)
    rows = []
    for k, y in zip(keys, ours):
        pig, sess, grp, eye, fl = k.split("|")
        r = rec.get((pig, sess, grp, eye, fl), {})
        flags = []
        base, noise, lv = float(np.mean(y[pre])), float(np.std(y[pre])), float(np.mean(y[late]))
        if abs(base) > LIM["baseline"]:
            flags.append(f"baseline {base:+.0f} uV")
        if noise > LIM["pre_noise"]:
            flags.append(f"pre-flash noise {noise:.0f} uV")
        if abs(lv) > LIM["late"]:
            flags.append(f"late level {lv:+.0f} uV")
        if r.get("Pick_ok") == "False":
            flags.append("pick on window edge")
        rows.append(dict(Pig=int(pig), Session=sess, Group=grp, Eye=("Test" if eye == "RE" else "Control"),
                         Flash=fl, Baseline_uV=round(base, 1), Pre_noise_uV=round(noise, 1),
                         Late_level_uV=round(lv, 1),
                         Drift_flag=r.get("Drift_flag", ""), QC=("; ".join(flags) if flags else "ok")))
    with open(D / "erg_qc.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(); w.writerows(rows)
    sheets = contact_sheets(grid, keys, ours, rows,
                            z["ops"] if "ops" in z.files else None,
                            z["op_window"] if "op_window" in z.files else None)
    bad = [r for r in rows if r["QC"] != "ok"]
    print(f"{len(rows)} traces, {len(bad)} flagged -> erg_qc.csv; {len(sheets)} contact sheets")
    for r in bad:
        print(f"  P{r['Pig']} {r['Session'][:9]} {r['Group']:9s} {r['Eye']:7s} fl{r['Flash']:>4}: {r['QC']}")


def contact_sheets(grid, keys, ours, rows, ops=None, op_win=None):
    """One contact sheet per pig x electrode x flash. At flashes 25 and 8 each trace has
    its OP trace (75-300 Hz, from erg_traces.npz) in a panel under it, same time axis,
    OP window shaded. Pig 1 is labelled exploratory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qc = {(str(r["Pig"]), r["Session"], r["Group"], {"Test": "RE", "Control": "LE"}[r["Eye"]], r["Flash"]): r["QC"]
          for r in rows}
    made = []
    idx = {tuple(k.split("|")): i for i, k in enumerate(keys)}
    groups = {"jet": ["Pre Jet", "Intra Jet", "Post Jet"], "rm": ["Pre RM", "Post RM"]}
    for pig in sorted({k[0] for k in idx}, key=int):
        sess = sorted({k[1] for k in idx if k[0] == pig})
        expl = int(pig) in EXPLORATORY_PIGS
        for el, gl in groups.items():
            for fl in ("25", "8", "0.06"):
                with_op = ops is not None and fl in OP_FLASHES
                cols = [(g, e) for g in gl for e in ("RE", "LE")]
                kinds = [(s, k) for s in sess for k in (("erg", "op") if with_op else ("erg",))]
                ratios = [1.0 if k == "erg" else 0.55 for _s, k in kinds]
                fig = plt.figure(figsize=(2.3 * len(cols), 1.9 * sum(ratios) + 0.65))
                gs = fig.add_gridspec(len(kinds), len(cols), height_ratios=ratios, hspace=0.15, wspace=0.08)
                ref = {}
                for i, (s, kind) in enumerate(kinds):
                    for j, (g, e) in enumerate(cols):
                        ax = fig.add_subplot(gs[i, j], sharex=ref.get("x"), sharey=ref.get(kind))
                        ref.setdefault("x", ax); ref.setdefault(kind, ax)
                        k = (pig, s, g, e, fl)
                        if i == 0:
                            ax.set_title(f"{g.split()[0]} {'test' if e == 'RE' else 'control'}", fontsize=8)
                        if j == 0:
                            ax.set_ylabel(s.replace("Session ", "S").replace(" - ", " ") if kind == "erg"
                                          else "OPs", fontsize=7)
                        ax.tick_params(labelsize=6)
                        if i < len(kinds) - 1:
                            ax.tick_params(labelbottom=False)
                        if j > 0:
                            ax.tick_params(labelleft=False)
                        if k not in idx:
                            ax.set_axis_off(); continue
                        n = idx[k]
                        ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
                        if kind == "op":
                            if np.isfinite(ops[n]).any():
                                ax.plot(grid, ops[n], color="#7b3294", lw=0.7)
                                if op_win is not None and np.isfinite(op_win[n]).all():
                                    ax.axvspan(op_win[n][0], op_win[n][1], color="#7b3294", alpha=0.12, lw=0)
                            continue
                        ax.plot(grid, ours[n], color="k", lw=0.8)
                        q = qc.get(k, "ok")
                        if q != "ok":
                            for sp in ax.spines.values():
                                sp.set_color("#c00000"); sp.set_linewidth(1.6)
                            ax.text(0.02, 0.02, q[:60], transform=ax.transAxes, fontsize=5.5, color="#c00000",
                                    va="bottom", wrap=True)
                fig.suptitle(f"Pig {pig}{' (exploratory)' if expl else ''} {el.upper()} flash {fl}: our average (black); red frame = QC flag"
                             + ("; under each: OPs 75-300 Hz (purple, shaded = OP window)" if with_op else ""),
                             fontsize=9)
                fig.subplots_adjust(top=1 - 0.6 / (1.9 * sum(ratios) + 0.5), bottom=0.05, left=0.07, right=0.99)
                name = f"fig_qc_pig{pig}_{el}_flash{fl.replace('.', 'p')}.png"
                fig.savefig(D / name, dpi=110)
                plt.close(fig)
                made.append(name)
    return made


if __name__ == "__main__":
    main()
