"""Chapter 5: one synthesis figure that sets the three aims side by side.

Output (figures/discussion/):
  discussion-synthesis.png        (A) ex vivo, Aim 1: cooling at five sites in human donor
                                  and fresh porcine eyes (original ring);
                                  (B) model, Aim 2: the same sites in the model, without
                                  perfusion (enucleated eye in the 37 C oil bath) and with
                                  perfusion (human eye in the orbit), beside the human donor
                                  eyes; (C) in vivo, Aim 3: ring hold temperature of each
                                  session, and (D) b-wave implicit-time delay (test minus
                                  control eye, 25 cd s/m2 flash), by cooling duration
  discussion_synthesis_values.csv every plotted number, with its source file and row
  MANIFEST.json                   provenance (provenance.record)

Inputs, copied by --refresh into figures/discussion/sources/ (no value is computed here
except reading and selecting rows; SD and n are taken as stored):
  exvivo_all_groups_summary.csv   <- thesis-analysis/results/02_groups/all_groups_summary.csv
                                     (the source of Tables 2.4 and 2.5; dT_mean, dT_sd, n)
  model_vs_exvivo.csv             <- figures/aim2/sources/data/python/corrected/
                                     model_vs_exvivo.csv (the corrected model; the snapshot
                                     behind Figure 3.x model-vs-exvivo; column "dT model (C)",
                                     group "Human Fresh Aluminum Original", perfused corrected
                                     Python model v1, eye at rest before cooling, read at 35 min)
  replica_exvivo_compare_all.csv  <- scr-heat-replica/results/figures_2026-10-22/exvivo_original/
                                     exvivo_compare_all.csv (Section 3.3 "Model versus Ex-Vivo
                                     Measurements": case F1_v1_Al_exvivo, oil coefficient
                                     30 W m-2 K-1 assumed in advance, no perfusion; quantity
                                     "cooling", column "model")
  ring_temperature_summary.csv    <- figures/aim3/ring_temperature_summary.csv (hold_mean_C)
  erg_cohort_cooling_summary.csv  <- figures/aim3/sources/erg/erg_cohort_cooling_summary.csv
                                     (Cooling (Intra Jet - Pre Jet), flash 25, b_t: per-animal
                                     values and Mean_T_minus_C, the means quoted in Sec. 4.3)

Animal 1 is exploratory: drawn grey and hollow and labelled so. The bars in the b-wave
panel are the cohort means as quoted in Chapter 4, which still include animal 1 at 30 and
60 min (UPDATE-PIGS56: re-run with --refresh after animals 5 and 6 and the restatement of
the ERG statistics on animals 2 onward).

Run:  python code/figures/discussion_synthesis_figure.py            (from the copies)
      python code/figures/discussion_synthesis_figure.py --refresh  (re-copy the inputs first)
"""
import csv
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from provenance import THESIS, record  # noqa: E402

BOX = THESIS.parents[2]  # .../Eye Cooler
CODE = BOX / "03 - Experiments & Data" / "Analysis Code"
OUT = THESIS / "figures" / "discussion"
SRC = OUT / "sources"
ORIGINS = {
    "exvivo_all_groups_summary.csv": CODE / "thesis-analysis" / "results" / "02_groups" / "all_groups_summary.csv",
    "model_vs_exvivo.csv": THESIS / "figures" / "aim2" / "sources" / "data" / "python" / "corrected"
    / "model_vs_exvivo.csv",
    "replica_exvivo_compare_all.csv": CODE / "scr-heat-replica" / "results" / "figures_2026-10-22"
    / "exvivo_original" / "exvivo_compare_all.csv",
    "ring_temperature_summary.csv": THESIS / "figures" / "aim3" / "ring_temperature_summary.csv",
    "erg_cohort_cooling_summary.csv": THESIS / "figures" / "aim3" / "sources" / "erg"
    / "erg_cohort_cooling_summary.csv",
}

# Sites from anterior to posterior, as in Tables 2.4 and 2.5. Names used by each source.
SITES = ["Anterior sclera", "Equatorial sclera", "Vitreous", "Posterior sclera", "Proximal optic nerve"]
NAME_EXVIVO = dict(zip(SITES, ["Anterior Sclera", "Equatorial Sclera", "Vitreous", "Posterior Sclera", "Optic Nerve"]))
NAME_MODEL = dict(zip(SITES, ["Anterior Sclera", "Equatorial Sclera", "Vitreous Humor", "Posterior Sclera",
                              "Optic Nerve Head"]))
NAME_REPLICA = NAME_EXVIVO
DURATIONS = [30, 60, 120]
SESSION_OF = {30: "1", 60: "2", 120: "3"}
PIGS = [1, 2, 3, 4, 5]  # animal 5: session 1 only, the ring-about-9.5-C group
EXPLORATORY = {1}

# Okabe-Ito (colour-blind safe) for the series; animal colours as in the Chapter 4 figures.
C_HUMAN, C_PIG, C_BATH, C_PERF = "#0072B2", "#E69F00", "#CC79A7", "#009E73"
ANIMAL = {2: "#d62728", 3: "#2ca02c", 4: "#9467bd", 5: "#8c564b"}
EXPL = "0.55"

plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def refresh():
    SRC.mkdir(parents=True, exist_ok=True)
    for name, src in ORIGINS.items():
        shutil.copy2(src, SRC / name)
    print("sources refreshed")


def rows(name):
    with open(SRC / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    if "--refresh" in sys.argv or not all((SRC / n).exists() for n in ORIGINS):
        refresh()
    values = []  # (panel, series, site_or_duration, animal, value, sd, n, source, selector)

    # ---------------------------------------------------------------- (A) ex vivo
    ex = rows("exvivo_all_groups_summary.csv")
    def exv(species, status, site):
        r = [x for x in ex if x["species"] == species and x["status"] == status and x["ring_material"] == "Aluminum"
             and x["ring"] == "Original" and x["channel"] == NAME_EXVIVO[site]]
        assert len(r) == 1, (species, site)
        return float(r[0]["dT_mean"]), float(r[0]["dT_sd"]), int(r[0]["n"])
    human = {s: exv("Human", "Fresh", s) for s in SITES}
    pig = {s: exv("Porcine", "Fresh", s) for s in SITES}
    for s in SITES:
        values.append(("A", "human donor eyes", s, "", *human[s], "exvivo_all_groups_summary.csv",
                       f"Human/Fresh/Aluminum/Original/{NAME_EXVIVO[s]}: dT_mean, dT_sd, n"))
        values.append(("A", "fresh porcine eyes", s, "", *pig[s], "exvivo_all_groups_summary.csv",
                       f"Porcine/Fresh/Aluminum/Original/{NAME_EXVIVO[s]}: dT_mean, dT_sd, n"))

    # ---------------------------------------------------------------- (B) model
    mv = rows("model_vs_exvivo.csv")
    perf = {}
    for s in SITES:
        r = [x for x in mv if x["group"] == "Human Fresh Aluminum Original" and x["site"] == NAME_MODEL[s]]
        assert len(r) == 1, s
        perf[s] = float(r[0]["dT model (C)"])
        # the measured value stored beside it must be the Table 2.5 value used in panel A
        assert abs(float(r[0]["dT exp (C)"]) - human[s][0]) < 1e-9, s
        values.append(("B", "model, perfused eye", s, "", perf[s], "", "", "model_vs_exvivo.csv",
                       f"Human Fresh Aluminum Original/{NAME_MODEL[s]}: dT model (C)"))
    rp = rows("replica_exvivo_compare_all.csv")
    bath = {}
    for s in SITES:
        r = [x for x in rp if x["case"] == "F1_v1_Al_exvivo" and x["row_type"] == "site"
             and x["site"] == NAME_REPLICA[s] and x["quantity"] == "cooling"]
        assert len(r) == 1 and float(r[0]["h_oil"]) == 30.0 and r[0]["choroid_perfusion"] == "False", s
        bath[s] = float(r[0]["model"])
        assert abs(float(r[0]["measured_mean"]) - human[s][0]) < 1e-9, s
        values.append(("B", "model, eye in the bath without perfusion", s, "", bath[s], "", "",
                       "replica_exvivo_compare_all.csv", f"F1_v1_Al_exvivo/{NAME_REPLICA[s]}/cooling: model"))

    # ---------------------------------------------------------------- (C) in vivo
    rt = rows("ring_temperature_summary.csv")
    hold = {}
    for r in rt:
        if r["hold_mean_C"]:
            if int(r["pig"]) == 6 or (int(r["pig"]) == 5 and int(r["session"]) > 1):
                continue            # ring about 4 C; panel C is the group at about 9.5 C
            hold[(int(r["pig"]), int(r["duration_planned_min"]))] = float(r["hold_mean_C"]) + 5.5   # corrected
    for (p, d), v in sorted(hold.items()):
        values.append(("C", "ring hold temperature", d, p, v, "", "", "ring_temperature_summary.csv",
                       f"pig {p}, session {SESSION_OF[d]}: hold_mean_C"))
    eg = rows("erg_cohort_cooling_summary.csv")
    delay, mean_delay = {}, {}
    for d in DURATIONS:
        r = [x for x in eg if x["Comparison"].startswith("Cooling") and x["Flash"] == "25" and x["Measure"] == "b_t" and x["Set"].startswith("Ring about 9.5")
             and int(x["Duration_min"]) == d]
        assert len(r) == 1, d
        pigs = [int(p) for p in r[0]["Pigs"].split(",")]
        vals = [float(v) for v in r[0]["Per_animal"].split(",")]
        for p, v in zip(pigs, vals):
            delay[(p, d)] = v
            values.append(("C", "b-wave delay, test minus control", d, p, v, "", "", "erg_cohort_cooling_summary.csv",
                           f"Cooling/25/b_t/{d}: Per_animal"))
        mean_delay[d] = float(r[0]["Mean_T_minus_C"])
        values.append(("C", "b-wave delay, cohort mean as quoted", d, r[0]["Pigs"], mean_delay[d], r[0]["SD"],
                       r[0]["n_animals"], "erg_cohort_cooling_summary.csv", f"Cooling/25/b_t/{d}: Mean_T_minus_C"))

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(6.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.5, wspace=0.32)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC1, axC2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    xs = range(len(SITES))
    labels = ["Anterior\nsclera", "Equatorial\nsclera", "Vitreous", "Posterior\nsclera", "Proximal\noptic nerve"]
    w = 0.38
    for k, (series, col, off) in enumerate([(human, C_HUMAN, -w / 2), (pig, C_PIG, w / 2)]):
        axA.bar([x + off for x in xs], [series[s][0] for s in SITES], w, color=col,
                yerr=[series[s][1] for s in SITES], error_kw={"lw": 0.8, "capsize": 2})
    axA.set_title("(A) Ex vivo, Aim 1", loc="left", fontsize=9.5, fontweight="bold")
    axA.legend(handles=[Patch(color=C_HUMAN, label="Human donor eyes"), Patch(color=C_PIG, label="Fresh porcine eyes")],
               frameon=False, fontsize=7.5, loc="upper right")
    w3 = 0.27
    axB.bar([x - w3 for x in xs], [human[s][0] for s in SITES], w3, facecolor="white", edgecolor=C_HUMAN, lw=1.0,
            yerr=[human[s][1] for s in SITES], error_kw={"lw": 0.8, "capsize": 2, "ecolor": C_HUMAN})
    axB.bar(list(xs), [bath[s] for s in SITES], w3, color=C_BATH)
    axB.bar([x + w3 for x in xs], [perf[s] for s in SITES], w3, color=C_PERF)
    axB.set_title("(B) Model, Aim 2", loc="left", fontsize=9.5, fontweight="bold")
    axB.legend(handles=[Patch(facecolor="white", edgecolor=C_HUMAN, label="Human donor eyes (as in A)"),
                        Patch(color=C_BATH, label="Model, in the bath, no perfusion"),
                        Patch(color=C_PERF, label="Model, perfused eye")],
               frameon=False, fontsize=7.5, loc="upper right")
    for ax in (axA, axB):
        ax.set_xticks(list(xs), labels, fontsize=7.5)
        ax.set_ylabel("Cooling, fall from the start (°C)")
        ax.set_ylim(0, 31)

    # (C) left: ring hold temperature
    for p in PIGS:
        pts = [(d, hold[(p, d)]) for d in DURATIONS if (p, d) in hold]
        jit = (p - 2.5) * 4
        if p in EXPLORATORY:
            axC1.plot([d + jit for d, _ in pts], [v for _, v in pts], "o", color=EXPL, mfc="white", ms=4.5)
        else:
            axC1.plot([d + jit for d, _ in pts], [v for _, v in pts], "o", color=ANIMAL[p], ms=4.5)
    axC1.axhline(9.5, color="0.3", ls=":", lw=0.8)
    axC1.text(12, 9.53, "set point", fontsize=7, color="0.3", ha="left", va="bottom")
    axC1.set_ylim(8.5, 10.5)
    axC1.set_ylabel("Ring hold temperature (°C)")
    axC1.set_title("(C) In vivo, Aim 3: ring", loc="left", fontsize=9.5, fontweight="bold")
    # (C) right: b-wave delay
    axC2.bar(DURATIONS, [mean_delay[d] for d in DURATIONS], 16, color="0.85", zorder=0)
    for p in PIGS:
        pts = [(d, delay[(p, d)]) for d in DURATIONS if (p, d) in delay]
        jit = (p - 2.5) * 3
        if p in EXPLORATORY:
            axC2.plot([d + jit for d, _ in pts], [v for _, v in pts], "o", color=EXPL, mfc="white", ms=4.5)
        else:
            axC2.plot([d + jit for d, _ in pts], [v for _, v in pts], "o", color=ANIMAL[p], ms=4.5)
    axC2.axhline(0, color="0.3", lw=0.6)
    axC2.set_ylim(-1, 9)
    axC2.set_ylabel("b-wave delay, cooled − fellow eye (ms)")
    axC2.set_title("(D) In vivo, Aim 3: ERG", loc="left", fontsize=9.5, fontweight="bold")
    for ax in (axC1, axC2):
        ax.set_xticks(DURATIONS, ["30", "60", "120"])
        ax.set_xlim(10, 140)
        ax.set_xlabel("Cooling duration (min)")
    handles = [Line2D([], [], color=ANIMAL[p], marker="o", ms=4.5, ls="none", label=f"Animal {p}") for p in (2, 3, 4)]
    handles.append(Line2D([], [], color=EXPL, marker="o", mfc="white", ms=4.5, ls="none",
                          label="Animal 1 (exploratory)"))
    handles.append(Patch(color="0.85", label="Mean, as in Chapter 4"))
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.14)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "discussion-synthesis.png"
    fig.savefig(png, facecolor="white")
    plt.close(fig)

    out_csv = OUT / "discussion_synthesis_values.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["panel", "series", "site_or_duration_min", "animal", "value", "sd", "n", "source", "selector"])
        wr.writerows(values)
    for v in values:
        print(v[0], v[1], v[2], v[3], f"{v[4]:.2f}", v[5], v[6])
    record(__file__, [png, out_csv], [SRC / n for n in ORIGINS],
           origins={f"figures/discussion/sources/{n}": str(p) for n, p in ORIGINS.items()},
           notes=("Chapter 5 synthesis figure. Cooling = fall from the start of cooling (dT): ex vivo over the "
                  "20-min record (minutes 16-20), perfused model from its own start (ring at 25 C) to 35 min, bath "
                  "model as in the replica comparison. Animal 1 exploratory (grey, hollow). UPDATE-PIGS56."),
           manifest=OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
