"""Chapter 4: electroretinogram results of the in-vivo study.

Outputs (figures/aim3/):
  erg-waveforms.png     one session per cooling duration (among sessions with a recovery
                        recording, the animal with the median b-wave slowing): test and
                        control eye, Jet electrode, before /
                        during cooling / recovery, flash 25 cd.s/m2 (a-, b-wave) and the
                        oscillatory potentials at flash 8 (75-300 Hz)
  erg-cooling-effect.png  change of each eye from its own recording before cooling (test vs
                        test, control vs control; 2026-11-01), per animal and cooling
                        duration, with the mean and its 95 % CI: b-wave implicit time,
                        amplitude and width at the three flashes, summed OPs at 8 and 25
  erg-dose-response.png b-wave implicit time and summed OPs, test minus control, against
                        the ring temperature at the recording (Jet; before, during,
                        recovery)
  erg_results_summary.csv  every number quoted in the ERG results (printed on each run)
  MANIFEST.json         provenance (provenance.record)

Inputs, copied by --refresh from the study folder (read only) into figures/aim3/sources/erg/:
  erg_cohort_recordings.csv, erg_cohort_cooling_summary.csv, erg_cohort_restoration.csv,
  erg_cohort_effects.csv, erg_traces.npz, MANIFEST.json   -- ERG Cohort Analysis (LivePigs-Tools erg_cohort.py)
  Summary statistics are those of Set SET; the sessions at the colder ring are drawn with triangles
  and animal 1 is exploratory. Ring temperatures are the corrected ones (Ring_temp_corrected_C).

Definitions are those of erg_cohort.py (thesis Sec. 4.2 Data Analysis): change = later
minus earlier for each eye; restored = the test eye's own change within 2 x SD of the control
eyes' change (pooled); recovery recordings only >= 10 min after switch-off. Test = right
(cooled) eye.

Run:  python code/figures/aim3_erg_figures.py            (from the copies)
      python code/figures/aim3_erg_figures.py --refresh  (re-copy the inputs first)
"""
import csv
import json
import os
import pathlib
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from provenance import THESIS, record

STUDY = pathlib.Path(os.environ.get("LIVE_PIG_STUDY", THESIS / "data" / "Live Pigs Safety Study"))
COHORT = STUDY / "ERG Cohort Analysis"
FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "erg"
FILES = ["erg_cohort_recordings.csv", "erg_cohort_cooling_summary.csv", "erg_cohort_restoration.csv",
         "erg_cohort_effects.csv", "erg_traces.npz", "MANIFEST.json"]
SET = "Ring about 9.5 C (Pigs 2-4, Pig 5 S1)"      # the group with enough animals for statistics
SHOWN = {1, 2, 3, 4, 5, 6}
CONTACT_CAL = {5: 2, 6: 1}      # sessions in the ring-about-4-C group
DURS = [30, 60, 120]
PIG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
PHASES = [("Pre Jet", "Before cooling", "#1f77b4"), ("Intra Jet", "During cooling", "#d62728"),
          ("Post Jet", "Recovery", "#2ca02c")]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(name):
    with open(SRC / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def refresh():
    SRC.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(COHORT / f, SRC / f)


def summary_row(summ, comp, fl, meas, dur):
    return next((r for r in summ if r["Comparison"].startswith(comp) and r["Flash"] == fl
                 and r["Measure"] == meas and num(r["Duration_min"]) == dur), None)


# ------------------------------------------------------------------ figures
def cold(r):
    return int(r["Session"].split()[1]) >= CONTACT_CAL.get(int(r["Pig"]), 99)


def cooling_effect_figure(eff, out):
    panels = [("b_t", "25", "b-wave implicit time,\nchange (ms)"), ("b_t", "8", None), ("b_t", "0.06", None),
              ("b_amp", "25", "b-wave amplitude,\nchange (µV)"), ("b_amp", "8", None), ("b_amp", "0.06", None),
              ("b_width", "25", "b-wave width at 80%,\nchange (ms)"), ("b_width", "8", None), ("b_width", "0.06", None),
              ("op_amp", "25", "Oscillatory potentials,\nsum OP1-4, change (µV)"), ("op_amp", "8", None), (None, None, None)]
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 8.2))
    for ax, (meas, fl, ylab) in zip(axes.flat, panels):
        if meas is None:
            ax.axis("off")
            continue
        ax.axhline(0, color="0.6", lw=0.7)
        for k, d in enumerate(DURS):
            rs = [r for r in eff if r["Comparison"].startswith("Cooling") and float(r["Flash"]) == float(fl)
                  and r["Measure"] == meas and num(r["Duration_min"]) == d and int(r["Pig"]) in SHOWN]
            for col, off, mfc in (("dTest", -0.2, None), ("dControl", 0.2, "white")):
                pts = [(int(r["Pig"]), num(r[col]), cold(r)) for r in rs if num(r[col]) is not None]
                for p, v, cd in pts:
                    ax.plot(k + off + (p - 3.5) * 0.03, v, "^" if cd else "o", ms=3.5, color=PIG_COLORS[(p - 1) % 6], mfc=mfc)
                vals = [v for p, v, cd in pts if p != 1 and not cd]
                if len(vals) > 1:
                    m, s = np.mean(vals), np.std(vals, ddof=1)
                    h = stats.t.ppf(0.975, len(vals) - 1) * s / np.sqrt(len(vals))
                    ax.errorbar(k + off + 0.13, m, yerr=h, fmt="s", color="k", ms=3.5, capsize=2, lw=0.9, mfc=mfc or "k")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{d} min" for d in DURS])
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(f"{fl} cd·s/m²", fontsize=8.5)
        if ylab:
            ax.set_ylabel(ylab)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", color=PIG_COLORS[p - 1], label=f"Animal {p}" + (" (exploratory)" if p == 1 else ""))
         for p in sorted(SHOWN)]
    h += [Line2D([], [], marker="o", ls="none", color="0.3", label="Filled: test eye"),
          Line2D([], [], marker="o", ls="none", color="0.3", mfc="white", label="Open: control eye"),
          Line2D([], [], marker="^", ls="none", color="0.3", label="Triangle: ring about 4 °C"),
          Line2D([], [], marker="s", ls="none", color="k", label="Mean at ring about 9.5 °C, 95% CI")]
    axes[3][2].legend(handles=h, loc="center", frameon=False, fontsize=6.5, title="Change of each eye,\nbefore to during cooling")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def dose_response_figure(recs, out):
    by = {}
    for r in recs:
        if r["Group"].endswith("Jet"):
            by.setdefault((int(r["Pig"]), r["Session"], r["Group"], r["Flash"]), {})[r["Eye"]] = r
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    pooled = {}
    for ax, (col, fl, ylab) in zip(axes, (("b_t_ms", "25", "b-wave implicit time,\ntest minus control (ms)"),
                                          ("OP_sum_uV", "8", "Oscillatory potentials (sum OP1-4),\n"
                                                             "test minus control (µV)"))):
        xs, ys = [], []
        for (p, s) in sorted({(k[0], k[1]) for k in by}):
            pts = []
            for g, _gl, _c in PHASES:
                e = by.get((p, s, g, fl), {})
                t, c = e.get("Test"), e.get("Control")
                if t and c and num(t["Ring_temp_corrected_C"]) is not None and num(t[col]) is not None \
                        and num(c[col]) is not None:
                    pts.append((g, num(t["Ring_temp_corrected_C"]), num(t[col]) - num(c[col])))
            if len(pts) < 2:
                continue
            ax.plot([x for _g, x, _y in pts], [y for _g, _x, y in pts], "-", color=PIG_COLORS[(p - 1) % 6],
                    lw=0.8, alpha=0.6)
            for g, x, y in pts:
                mk = {"Pre Jet": "o", "Intra Jet": "v", "Post Jet": "s"}[g]
                ax.plot(x, y, mk, ms=4.5, color=PIG_COLORS[(p - 1) % 6], mfc="white" if g == "Post Jet" else None)
                xs.append(x); ys.append(y)
        ax.axhline(0, color="0.7", lw=0.7)
        ax.set_xlabel("Ring temperature at the recording, corrected (°C)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{fl} cd·s/m²", fontsize=8.5)
        if len(xs) > 2:
            b1, b0 = np.polyfit(xs, ys, 1)
            pooled[(col, fl)] = dict(slope_per_C=b1, r=float(np.corrcoef(xs, ys)[0, 1]), n=len(xs))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=m, ls="none", color="0.3", mfc=f, label=l)
         for m, f, l in (("o", "0.3", "Before cooling"), ("v", "0.3", "During cooling"),
                         ("s", "white", "After switch-off"))]
    axes[1].legend(handles=h, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return pooled


def waveform_figure(recs, summ, rest, out):
    z = np.load(SRC / "erg_traces.npz", allow_pickle=False)
    grid, keys = z["grid"], [str(k) for k in z["keys"]]
    idx = {k: i for i, k in enumerate(keys)}
    has_ops = "ops" in z.files
    # sessions whose post Jet recording is a recovery recording (>= 10 min after switch-off)
    eligible = {(int(r["Pig"]), int(float(r["Duration_min"]))) for r in rest
                if r["Comparison"].startswith("Jet") and r["Measure"] == "b_t" and r["Flash"] == "25"
                and r["Verdict"] in ("restored", "NOT restored")}
    # per duration: among those sessions, the animal whose b-wave slowing (flash 25) is the median
    chosen = []
    for d in DURS:
        r = summary_row(summ, "Cooling", "25", "b_t", d)
        if not r:
            continue
        pigs = [int(float(p)) for p in r["Pigs"].split(",") if p.strip()]
        vals = [float(v) for v in r["Per_animal"].split(",") if v.strip()]
        order = sorted((v, p) for v, p in zip(vals, pigs) if (p, d) in eligible)
        if not order:
            continue
        chosen.append((d, order[(len(order) - 1) // 2][1]))
    sess_of = {(int(r["Pig"]), int(float(r["Duration_min"]))): r["Session"] for r in recs}
    ncol = 4 if has_ops else 2
    fig, axes = plt.subplots(len(chosen), ncol, figsize=(7.0, 1.9 * len(chosen) + 0.5), squeeze=False)
    for i, (d, p) in enumerate(chosen):
        s = sess_of.get((p, d))
        for j, (eye, role) in enumerate((("RE", "Test"), ("LE", "Control"))):
            for kind, col, fl, arr, xlim in (("erg", j, "25", "ours", (-10, 120)),
                                             ("ops", 2 + j, "8", "ops", (0, 60))):
                if kind == "ops" and not has_ops:
                    continue
                ax = axes[i][col]
                for g, gl, c in PHASES:
                    k = f"{p}|{s}|{g}|{eye}|{fl}"
                    if k in idx:
                        y = z[arr][idx[k]]
                        if np.isfinite(y).any():
                            ax.plot(grid, y, color=c, lw=0.9, label=gl)
                ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
                ax.set_xlim(*xlim)
                if i == 0:
                    ax.set_title(f"{role} eye\n" + ("a- and b-wave, 25" if kind == "erg" else "OPs, 8") + " cd·s/m²",
                                 fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{d} min (animal {p})\nµV")
                if i == len(chosen) - 1:
                    ax.set_xlabel("Time from flash (ms)")
    for row in axes:                       # same scale for the two eyes of each kind
        for a, b in ((0, 1), (2, 3)):
            if b < ncol:
                lo = min(row[a].get_ylim()[0], row[b].get_ylim()[0]); hi = max(row[a].get_ylim()[1], row[b].get_ylim()[1])
                row[a].set_ylim(lo, hi); row[b].set_ylim(lo, hi)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return chosen


# ------------------------------------------------------------------ numbers quoted
def numbers(summ, rest, recs, pooled, chosen):
    rows = []
    add = lambda key, val, note="": rows.append(dict(key=key, value=val, note=note))
    for comp in ("Cooling", "Jet restoration", "Rewarming", "RM restoration"):
        for fl in ("25", "8", "0.06"):
            for meas in ("b_t", "b_amp", "op_amp", "a_t", "a_amp"):
                for d in DURS:
                    r = summary_row(summ, comp, fl, meas, d)
                    if r and num(r["Mean_T_minus_C"]) is not None:
                        base = f"{comp}|{meas}|{fl}|{d}"
                        add(base + "|mean", num(r["Mean_T_minus_C"]))
                        add(base + "|ci", f"{num(r['CI95_low'])} to {num(r['CI95_high'])}"
                            if num(r["CI95_low"]) is not None else "")
                        add(base + "|p", num(r["p_test_vs_control"]))
                        add(base + "|n", int(num(r["n_animals"])), r["Per_animal"])
    for r in summ:
        if r["Comparison"].startswith("Taken"):
            add(f"taken_together|{r['Flash']}|{int(num(r['Duration_min']))}", r["Count"], r["Per_animal"])
    for comp in ("Jet restoration", "RM restoration"):
        for meas in ("b_t", "b_amp", "op_amp"):
            for fl in ("25", "8", "0.06"):
                rs = [r for r in rest if r["Comparison"].startswith(comp) and r["Measure"] == meas and r["Flash"] == fl]
                if not rs:
                    continue
                ok = sum(r["Verdict"] == "restored" for r in rs)
                nr = [f"animal {r['Pig']} S{r['Session'][8]} ({num(r['Test_minus_Control']):+.1f} vs "
                      f"{num(r['Limit']):.1f})" for r in rs if r["Verdict"] == "NOT restored"]
                add(f"{comp}|{meas}|{fl}|restored", f"{ok} of {ok + len(nr)}", "; ".join(nr))
                add(f"{comp}|{meas}|{fl}|limit", num(rs[0]["Limit"]))
                sysf = sorted({f"animal {r['Pig']} S{r['Session'][8]}" for r in rs if r["Eye_check"].startswith("BOTH")})
                if sysf:
                    add(f"{comp}|{meas}|{fl}|systemic", "; ".join(sysf))
    post = [r for r in rest if r["Comparison"].startswith("Jet") and r["Measure"] == "b_t" and r["Flash"] == "25"
            and r["Verdict"] in ("restored", "NOT restored")]
    mins = [num(r["Post_min_after_switch_off"]) for r in post if num(r["Post_min_after_switch_off"]) is not None]
    fr = [num(r["Post_ring_rewarmed_fraction"]) for r in post if num(r["Post_ring_rewarmed_fraction"]) is not None]
    add("recovery|minutes_after_off", f"{min(mins):g} to {max(mins):g}")
    add("recovery|ring_rewarmed_fraction", f"{min(fr):.2f} to {max(fr):.2f}")
    intra = [num(r["Ring_temp_C"]) for r in recs if r["Group"] == "Intra Jet" and r["Eye"] == "Test"
             and r["Flash"] == "25" and num(r["Ring_temp_C"]) is not None]
    add("intra|ring_C", f"{min(intra):.1f} to {max(intra):.1f}")
    for (col, fl), v in pooled.items():
        add(f"dose|{col}|{fl}", f"slope {v['slope_per_C']:+.2f} per degC, r {v['r']:+.2f}, n {v['n']}")
    add("waveform_examples", "; ".join(f"{d} min: animal {p}" for d, p in chosen))
    return rows


def main():
    if "--refresh" in sys.argv:
        refresh()
    recs = [r for r in load("erg_cohort_recordings.csv") if int(r["Pig"]) in SHOWN]
    summ = [r for r in load("erg_cohort_cooling_summary.csv") if r["Set"] == SET]
    rest = [r for r in load("erg_cohort_restoration.csv") if int(r["Pig"]) in SHOWN]
    eff = load("erg_cohort_effects.csv")
    f1, f2, f3 = FIG3 / "erg-waveforms.png", FIG3 / "erg-cooling-effect.png", FIG3 / "erg-dose-response.png"
    chosen = waveform_figure(recs, summ, rest, f1)
    cooling_effect_figure(eff, f2)
    pooled = dose_response_figure(recs, f3)
    rows = numbers(summ, rest, recs, pooled, chosen)
    out_csv = FIG3 / "erg_results_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["key", "value", "note"]); w.writeheader(); w.writerows(rows)
    for r in rows:
        if r["key"].split("|")[0] in ("taken_together", "recovery", "intra", "dose", "waveform_examples") \
                or r["key"].endswith(("restored", "systemic")):
            print(r["key"], "=", r["value"], r["note"])
    man = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    inputs = [SRC / f for f in FILES]
    origins = {f"figures/aim3/sources/erg/{f}": str(COHORT / f) for f in FILES}
    record(__file__, [f1, f2, f3, out_csv], inputs, origins=origins,
           notes=(f"ERG Cohort Analysis generated {man.get('generated')} by LivePigs-Tools {man.get('git_head')}"
                  f"{' (uncommitted)' if man.get('uncommitted_changes') else ''}; animals with ERG data: 1-4. "
                  "UPDATE-PIGS56: re-run with --refresh after the cohort analysis includes animals 5 and 6."),
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
