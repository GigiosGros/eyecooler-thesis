"""Chapter 5: one synthesis figure that sets the three aims side by side.

Output (figures/discussion/):
  discussion-synthesis.png        (A) ex vivo, Aim 1: cooling at five sites in human donor
                                  and fresh porcine eyes (original ring);
                                  (B) rings, Aim 2 (2026-11-12): the same sites in the
                                  human donor eyes with the original, reduced-bulk and
                                  ceramic rings (until 2026-11-12: the perfused model);
                                  (C) in vivo, Aim 3, safety (2026-11-12): test minus
                                  control eye, per session, for the change in intraocular
                                  pressure by session length and the change in the surface
                                  grades (bulbar and limbal redness, corneal staining);
                                  (D) in vivo, Aim 3, retinal function (2026-11-12):
                                  b-wave implicit time, test minus control change from the
                                  Jet baseline, during cooling and after rewarming (25 cd s/m2),
                                  each animal joined, by session length
                                  (until 2026-11-12: (C) ring hold temperature, (D) the
                                  cooling change only)
  discussion_synthesis_values.csv every plotted number, with its source file and row
  MANIFEST.json                   provenance (provenance.record)

Inputs, copied by --refresh into figures/discussion/sources/ (no value is computed here
except reading and selecting rows; SD and n are taken as stored):
  exvivo_all_groups_summary.csv   <- thesis-analysis/results/02_groups/all_groups_summary.csv
                                     (the source of Tables 2.4 and 2.5 and of the ring groups of
                                     Chapter 3; dT_mean, dT_sd, n)
  (model_vs_exvivo.csv, the perfused model of the old panel B, is no longer read: 2026-11-12;
   replica_exvivo_compare_all.csv, the bath model, since 2026-11-11; the stored copies in
   sources/ are kept)
  erg_cohort_cooling_summary.csv  <- figures/aim3/sources/erg/erg_cohort_cooling_summary.csv
                                     (Set primary, flash 25, b_t; "Cooling (Intra Jet - Pre Jet)"
                                     and "Jet restoration (Post Jet - Pre Jet)": Per_animal and
                                     Mean_T_minus_C, the values of Table 4.3 and Sec. 4.3)
  master_iop.csv                  <- figures/aim3/sources/master_iop.csv (Animal_set primary:
                                     Test_minus_Control; the per-length means are Table 4.5)
  master_scores.csv               <- figures/aim3/sources/master_scores.csv (Change_T_minus_C of
                                     the three surface grades, cooling sessions of animals 2-6,
                                     session 3 of animal 3 left out, as in Table 4.6)
  (ring_temperature_summary.csv, the old panel C, is no longer read: 2026-11-12; the stored copy
   in sources/ is kept)

Panels C and D show animals 2 to 6 (the primary set of Chapter 4); animal 1, exploratory, and
session 3 of animal 3 are left out, as in Chapter 4.

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
    "erg_cohort_cooling_summary.csv": THESIS / "figures" / "aim3" / "sources" / "erg"
    / "erg_cohort_cooling_summary.csv",
    "master_iop.csv": THESIS / "figures" / "aim3" / "sources" / "master_iop.csv",
    "master_scores.csv": THESIS / "figures" / "aim3" / "sources" / "master_scores.csv",
}
RINGS = [("original", "Aluminum", "Original"), ("reduced", "Aluminum", "Reduced Bulk"), ("ceramic", "Ceramic", "Original")]
RING_LABEL = {"original": "original ring (aluminum)", "reduced": "reduced-bulk ring (aluminum)",
              "ceramic": "ceramic ring (original geometry)"}
GRADES = [("bulbar redness (0-4)", "Bulbar"), ("limbal redness (0-4)", "Limbal"), ("corneal staining (0-4)", "Staining")]

# Sites from anterior to posterior, as in Tables 2.4 and 2.5. Names used by each source.
SITES = ["Anterior sclera", "Equatorial sclera", "Vitreous", "Posterior sclera", "Proximal optic nerve"]
NAME_EXVIVO = dict(zip(SITES, ["Anterior Sclera", "Equatorial Sclera", "Vitreous", "Posterior Sclera", "Optic Nerve"]))
DURATIONS = [30, 60, 120]
SESSION_OF = {30: "1", 60: "2", 120: "3"}
PIGS = [2, 3, 4, 5, 6]  # the primary set of Chapter 4, both ring temperatures (2026-11-12: animal 1 not drawn)

# Okabe-Ito (colour-blind safe) for the series; animal colours as in the Chapter 4 figures.
C_HUMAN, C_PIG = "#0072B2", "#E69F00"
C_RING = {"original": C_HUMAN, "reduced": "#56B4E9", "ceramic": "#CC79A7"}   # Okabe-Ito; original = human blue of A
ANIMAL = {2: "#d62728", 3: "#2ca02c", 4: "#9467bd", 5: "#ff7f0e", 6: "#8c564b"}   # 2026-11-12: one color per animal across Chapter 4 and 5 (COLORS of aim3_invivo_figures.py)

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

    # ---------------------------------------------------------------- (B) rings, Aim 2 (2026-11-12: the three
    # rings measured in the human donor eyes at the probe sites, instead of the model)
    def exv_ring(material, ring, site):
        r = [x for x in ex if x["species"] == "Human" and x["status"] == "Fresh" and x["ring_material"] == material
             and x["ring"] == ring and x["channel"] == NAME_EXVIVO[site]]
        assert len(r) == 1, (material, ring, site)
        return float(r[0]["dT_mean"]), float(r[0]["dT_sd"]), int(r[0]["n"])
    rings = {key: {s: exv_ring(mat, ring, s) for s in SITES} for key, mat, ring in RINGS}
    assert rings["original"] == human
    for key, mat, ring in RINGS:
        for s in SITES:
            values.append(("B", f"human donor eyes, {RING_LABEL[key]}", s, "", *rings[key][s],
                           "exvivo_all_groups_summary.csv", f"Human/Fresh/{mat}/{ring}/{NAME_EXVIVO[s]}: dT_mean, dT_sd, n"))

    # ---------------------------------------------------------------- (C) in vivo, safety
    keep = lambda pig, session: pig in PIGS and not (pig == 3 and session.startswith("Session 3"))
    iop = {}
    for r in rows("master_iop.csv"):
        if r["Animal_set"] == "primary" and int(r["Duration_min"]) in DURATIONS and r["Test_minus_Control"]:
            assert keep(int(r["Pig"]), r["Session"])
            iop[(int(r["Pig"]), int(r["Duration_min"]))] = float(r["Test_minus_Control"])
    iop_mean = {d: sum(v for (p, dd), v in iop.items() if dd == d) / sum(1 for (p, dd) in iop if dd == d) for d in DURATIONS}
    for (p, d), v in sorted(iop.items()):
        values.append(("C", "IOP change, test minus control (mmHg)", d, p, v, "", "", "master_iop.csv",
                       f"pig {p}, {d} min: Test_minus_Control"))
    for d in DURATIONS:
        values.append(("C", "IOP change, mean over animals (Table 4.5)", d, "2-6", iop_mean[d], "", "", "master_iop.csv",
                       "mean of Test_minus_Control, Animal_set primary"))
    grade = {}
    for r in rows("master_scores.csv"):
        p, s = int(r["Pig"]), r["Session"]
        if keep(p, s) and int(r["Duration_min"]) in DURATIONS and r["Change_T_minus_C"]:
            grade.setdefault(r["Measure"], []).append((p, int(r["Duration_min"]), float(r["Change_T_minus_C"])))
    for m, _ in GRADES:
        for p, d, v in grade[m]:
            values.append(("C", f"{m}: change, test minus control (grade steps)", d, p, v, "", "", "master_scores.csv",
                           f"{m}, pig {p}, {d} min: Change_T_minus_C"))

    # ---------------------------------------------------------------- (D) in vivo, retinal function
    eg = rows("erg_cohort_cooling_summary.csv")
    erg, erg_mean = {}, {}
    for key, comp in (("cooling", "Cooling"), ("after", "Jet restoration")):
        for d in DURATIONS:
            r = [x for x in eg if x["Comparison"].startswith(comp) and x["Flash"] == "25" and x["Measure"] == "b_t"
                 and x["Set"] == "primary" and int(x["Duration_min"]) == d]
            assert len(r) == 1, (comp, d)
            for p, v in zip([int(p) for p in r[0]["Pigs"].split(",")], [float(v) for v in r[0]["Per_animal"].split(",")]):
                erg[(key, p, d)] = v
                values.append(("D", f"b-wave implicit time, test minus control, {key}", d, p, v, "", "",
                               "erg_cohort_cooling_summary.csv", f"{comp}/25/b_t/{d}: Per_animal"))
            erg_mean[(key, d)] = float(r[0]["Mean_T_minus_C"])
            values.append(("D", f"b-wave implicit time, mean, {key}", d, r[0]["Pigs"], erg_mean[(key, d)], r[0]["SD"],
                           r[0]["n_animals"], "erg_cohort_cooling_summary.csv", f"{comp}/25/b_t/{d}: Mean_T_minus_C"))

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(6.5, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.55, wspace=0.3)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    gC = gs[1, 0].subgridspec(1, 2, wspace=0.62, width_ratios=[0.8, 1.2])
    axI, axG = fig.add_subplot(gC[0, 0]), fig.add_subplot(gC[0, 1])
    axD = fig.add_subplot(gs[1, 1])
    xs = range(len(SITES))
    labels = ["Anterior\nsclera", "Equatorial\nsclera", "Vitreous", "Posterior\nsclera", "Proximal\noptic nerve"]
    ekw = {"lw": 0.7, "capsize": 1.8}
    TITLE = dict(loc="left", fontsize=9.5, fontweight="bold")

    # (A) ex vivo, human against porcine
    w = 0.38
    for series, col, off in [(human, C_HUMAN, -w / 2), (pig, C_PIG, w / 2)]:
        axA.bar([x + off for x in xs], [series[s][0] for s in SITES], w, color=col,
                yerr=[series[s][1] for s in SITES], error_kw=ekw)
    axA.set_title("(A) Ex vivo, Aim 1", **TITLE)
    axA.legend(handles=[Patch(color=C_HUMAN, label="Human donor eyes"), Patch(color=C_PIG, label="Fresh porcine eyes")],
               frameon=False, fontsize=7.8, loc="upper right")

    # (B) the three rings in the human donor eyes
    w3 = 0.27
    for k, (key, _, _) in enumerate(RINGS):
        axB.bar([x + (k - 1) * w3 for x in xs], [rings[key][s][0] for s in SITES], w3, color=C_RING[key],
                yerr=[rings[key][s][1] for s in SITES], error_kw=ekw)
    axB.set_title("(B) Rings in donor eyes, Aim 2", **TITLE)
    axB.legend(handles=[Patch(color=C_RING["original"], label="Original (aluminum; as in A)"),
                        Patch(color=C_RING["reduced"], label="Reduced bulk (aluminum)"),
                        Patch(color=C_RING["ceramic"], label="Ceramic (original geometry)")],
               frameon=False, fontsize=7.8, loc="upper right")
    for ax in (axA, axB):
        ax.set_xticks(list(xs), labels, fontsize=7.6)
        ax.set_ylabel("Cooling, fall from the start (°C)")
        ax.set_ylim(0, 31)

    # (C) safety: intraocular pressure and surface grades, test minus control, per session
    jit = {p: (p - 4) * 0.09 for p in PIGS}
    for i, d in enumerate(DURATIONS):
        for p in PIGS:
            if (p, d) in iop:
                axI.plot(i + jit[p], iop[(p, d)], "o", color=ANIMAL[p], ms=4)
        axI.plot([i - 0.3, i + 0.3], [iop_mean[d]] * 2, color="k", lw=1.4)
    axI.axhline(0, color="0.3", lw=0.6)
    axI.set_xticks(range(3), [str(d) for d in DURATIONS])
    axI.set_xlim(-0.6, 2.6)
    axI.set_xlabel("Session length (min)")
    axI.set_ylabel("Intraocular pressure change,\ntest − control (mmHg)")
    axI.set_title("(C) In vivo, Aim 3: safety", **TITLE)
    for i, (m, lab) in enumerate(GRADES):
        by_val = {}
        for p, d, v in sorted(grade[m], key=lambda t: (t[0], t[1])):
            by_val.setdefault(v, []).append(p)
        for v, ps in by_val.items():
            n = len(ps)
            for j, p in enumerate(ps):          # ties side by side, centred on the endpoint
                axG.plot(i + (j - (n - 1) / 2) * 0.062, v, "o", color=ANIMAL[p], ms=3.3)
    axG.axhline(0, color="0.3", lw=0.6)
    axG.set_xticks(range(3), [lab for _, lab in GRADES], fontsize=7.6)
    axG.set_xlim(-0.55, 2.55)
    axG.set_xlabel("Surface grade")
    # limits from the data, so no value is clipped (corneal staining reaches +4: animal 2, session 3)
    gv = [v for m, _ in GRADES for _, _, v in grade[m]]
    axG.set_ylim(min(gv) - 0.5, max(gv) + 0.5)
    axG.set_yticks(range(int(min(gv)), int(max(gv)) + 1))
    axG.set_ylabel("Grade change, test − control\n(steps of the 0–4 scale)")

    # (D) retinal function: during cooling, then after rewarming, each animal joined
    pos = {(key, d): 2.6 * i + (0 if key == "cooling" else 1) for i, d in enumerate(DURATIONS) for key in ("cooling", "after")}
    for d in DURATIONS:
        for p in PIGS:
            a, b = erg.get(("cooling", p, d)), erg.get(("after", p, d))
            xa, xb = pos[("cooling", d)] + jit[p] * 0.6, pos[("after", d)] + jit[p] * 0.6
            if a is not None and b is not None:
                axD.plot([xa, xb], [a, b], "-", color=ANIMAL[p], lw=0.8, alpha=0.8)
            if a is not None:
                axD.plot(xa, a, "o", color=ANIMAL[p], ms=4)
            if b is not None:
                axD.plot(xb, b, "o", color=ANIMAL[p], ms=4, mfc="white")
        for key in ("cooling", "after"):
            x = pos[(key, d)]
            axD.plot([x - 0.28, x + 0.28], [erg_mean[(key, d)]] * 2, color="k", lw=1.4)
    axD.axhline(0, color="0.3", lw=0.6)
    axD.set_xticks([pos[(k, d)] for d in DURATIONS for k in ("cooling", "after")],
                   ["cooling", "after"] * 3, fontsize=7.8)
    for i, d in enumerate(DURATIONS):
        axD.text(2.6 * i + 0.5, -0.11, f"{d} min", transform=axD.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
    axD.set_xlim(-0.6, 2.6 * 2 + 1.6)
    axD.set_ylim(-5, 9)
    axD.set_ylabel("b-wave implicit time change,\ntest − control (ms)")
    axD.set_title("(D) In vivo, Aim 3: retinal function", **TITLE)

    handles = [Line2D([], [], color=ANIMAL[p], marker="o", ms=4.5, ls="none", label=f"Animal {p}") for p in PIGS]
    handles.append(Line2D([], [], color="k", lw=1.4, label="Mean"))
    handles.append(Line2D([], [], color="0.4", marker="o", mfc="white", ms=4.5, ls="none", label="After rewarming (D)"))
    fig.legend(handles=handles, frameon=False, fontsize=7.8, ncol=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), handlelength=1.4, columnspacing=0.9, handletextpad=0.4)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.95, bottom=0.12)
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
           notes=("Chapter 5 synthesis figure (2026-11-12 redesign): (A) ex vivo human vs fresh porcine, (B) the three "
                  "rings in the human donor eyes at the probe sites, (C) in-vivo safety (IOP and surface grades, test minus "
                  "control), (D) b-wave implicit time during cooling and after rewarming, animals 2-6."),
           manifest=OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
