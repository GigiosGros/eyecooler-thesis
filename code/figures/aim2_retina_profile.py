"""Chapter 3: temperature along the retina in the model, redrawn from the data.

The values are COMSOL's own: the solution stored inside Ring Comparison.mph,
read without COMSOL along the model's "Retina Edge" line by
extract_comsol_retina.py in 03 - Experiments & Data/Analysis Code/scr-heat-replica
(see that script and the copied extraction manifest).  This replaces the
kelvin screenshot taken from the model report (aim2_report_images.py, which
still writes retina-profile.png; that file is no longer used in the thesis).

(A) the profile with v1 before cooling and after 15 and 35 min;
(B) how the other two rings differ from v1 after 35 min, which cannot be seen
    in (A) because the three lines coincide.

Source: Analysis Code/scr-heat-replica/results/comsol_retina/
Run:    python code/figures/aim2_retina_profile.py
"""
import sys
import csv
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from provenance import FIG, SOURCES, THESIS, record, written_by_other

BOX = THESIS.parents[2]
ORIGIN = BOX / "03 - Experiments & Data" / "Analysis Code" / "scr-heat-replica" / "results" / "comsol_retina"
FILES = {f"retina_profile_{v}.csv": f"comsol_retina__retina_profile_{v}.csv" for v in ("v1", "v2", "v3")}
FILES["MANIFEST.json"] = "comsol_retina__extraction_manifest.json"
TIMES = {0: ("Before cooling", "#8a8a8a", "-"), 35: ("35 min", "#08519c", "-"), 15: ("15 min", "#6baed6", (0, (4, 3)))}
RINGS = {"v2": ("#D55E00", "-"), "v3": ("#009E73", (0, (5, 2)))}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    d["s"] = d["arc_mm"] - d["arc_mm"][int(np.argmin(d["z_mm"]))]            # distance from the posterior pole
    return d


def main():
    other = written_by_other("retina-temperature-profile.png", __file__)
    if other and "--force" not in sys.argv:
        print(f"retina-temperature-profile.png was last written by {other} (the figure the thesis uses); not overwritten. Use --force to replace it.")
        return
    dst_dir = SOURCES / "data" / "model"
    dst_dir.mkdir(parents=True, exist_ok=True)
    snaps = []
    for src, name in FILES.items():
        if (ORIGIN / src).exists():                      # refresh the snapshot when the source is at hand
            shutil.copyfile(ORIGIN / src, dst_dir / name)
        snaps.append(dst_dir / name)
    d = {v: load(dst_dir / FILES[f"retina_profile_{v}.csv"]) for v in ("v1", "v2", "v3")}
    v1 = d["v1"]

    fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True, constrained_layout=True,
                               gridspec_kw=dict(height_ratios=[2.2, 1]))
    for t, (label, color, ls) in TIMES.items():
        a.plot(v1["s"], v1[f"T_comsol_{t}min_C"], color=color, ls=ls, lw=2.0, label=label)
    a.axhline(35, color="0.35", lw=0.8, ls=":")
    a.text(v1["s"].min(), 35.25, "35 °C", fontsize=8, color="0.35", va="bottom")
    a.set_ylabel("Temperature (°C)", fontsize=9)
    a.set_ylim(5, 38)
    a.set_title("(A) Original ring (v1)", loc="left", fontsize=10)
    h, l = a.get_legend_handles_labels()
    o = [l.index(TIMES[t][0]) for t in sorted(TIMES)]
    a.legend([h[i] for i in o], [l[i] for i in o], frameon=False, fontsize=9, loc="lower center")
    a.grid(axis="y", color="0.9", lw=0.6)
    a.annotate("anterior edge\nof the retina", (v1["s"][0], v1["T_comsol_35min_C"][0]), xytext=(v1["s"][0] + 3.2, 9.5),
               fontsize=8, color="0.3", va="center", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    a.annotate("posterior pole", (0, v1["T_comsol_35min_C"][int(np.argmin(np.abs(v1["s"])))]), xytext=(0, 29.5), ha="center",
               fontsize=8, color="0.3", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))

    out = {}
    for v, (color, ls) in RINGS.items():
        diff = np.interp(v1["s"], d[v]["s"], d[v]["T_comsol_35min_C"]) - v1["T_comsol_35min_C"]
        b.plot(v1["s"], diff, color=color, ls=ls, lw=1.8, label=f"{v} − v1")
        out[v] = (float(diff.min()), float(diff.max()), float(diff[int(np.argmin(np.abs(v1["s"])))]))
    b.axhline(0, color="0.35", lw=0.8)
    b.set_ylabel("Difference from v1\nafter 35 min (°C)", fontsize=9)
    b.set_xlabel("Distance along the retina from the posterior pole (mm)", fontsize=9)
    b.set_title("(B) Other rings against v1", loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
    b.grid(axis="y", color="0.9", lw=0.6)
    dst = FIG / "retina-temperature-profile.png"
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    notes = {"line": "model selection 'Retina Edge' (plane of symmetry), %.1f mm long" % (v1["arc_mm"][-1])}
    for t in TIMES:
        T = v1[f"T_comsol_{t}min_C"]
        notes[f"v1_{t}min_C"] = "ends %.2f and %.2f, maximum %.2f, at the posterior pole %.2f" % (
            T[0], T[-1], T.max(), T[int(np.argmin(np.abs(v1["s"])))])
    for v, (lo, hi, pole) in out.items():
        notes[f"{v}_minus_v1_35min_C"] = "from %+.2f to %+.2f, at the posterior pole %+.2f" % (lo, hi, pole)
    rel = str(ORIGIN.relative_to(BOX)).replace("\\", "/")
    record(__file__, [dst], snaps, origins={s.name: f"{rel}/{src}" for s, src in zip(snaps, FILES)}, notes=notes)
    for k, v in notes.items():
        print(k, ":", v)
    print("wrote", dst.name)


if __name__ == "__main__":
    main()
