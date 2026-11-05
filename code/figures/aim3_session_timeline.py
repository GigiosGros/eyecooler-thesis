"""Chapter 4: Figure fig:aim3-session-timeline (sequence of a cooling session).

Output (figures/aim3/):
  session-timeline.png   drawn diagram of one session: baseline assessments, dark
                         adaptation and baseline RM/Jet ERG, placement, switch-on, cooling
                         (30, 60 or 120 min) with the Jet ERG during it, switch-off, Jet ERG
                         after switch-off, removal, RM ERG, and the post-cooling assessments;
                         the dark period (room lights off, red light) is marked
  MANIFEST.json          provenance (provenance.record)

Input:
  figures/aim3/session_times.csv  (aim3_session_times_table.py; Table tab:aim3-session-times)
    The device and ERG times drawn are the MEDIANS over the sessions of the primary animals
    (2 to 4; animal 1 is exploratory) of: wait (placement -> switch-on), jet_before,
    jet_during (mid-session, one recording from animal 3 on), jet_after, removal_min,
    rm_after. They are printed on every run.
  Times NOT in that table are taken from the Methods text (Sec. subsec:aim3-assessments and
  subsec:retinal-function-assessment) and drawn as approximate (open symbols for the eye assessments):
    BASELINE_RM_BEFORE_PLACEMENT  baseline RM recording a few minutes before placement
                                  (bench sheets, e.g. animal 2 session 3: 9:53 -> 9:59)
    DARK_ADAPT_MIN = 20           dark adaptation planned before the baseline ERG
    STAIN_BEFORE_ERG = (24, 37)   staining preceded the baseline ERG by 24 to 37 min
    photographs and tonometry come before the staining (order of the Methods)
  The three phases are drawn as one continuous axis (2026-11-01): the phases share the
  switch-on and switch-off lines, each phase keeps its own scale, and the dark bar runs unbroken.
  After removal the post assessments are drawn with the same open markers; the staining is done in the dark and the lights go on after it (2026-10-31).

Run:  python code/figures/aim3_session_timeline.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from provenance import THESIS, record

FIG3 = THESIS / "figures" / "aim3"
TIMES = FIG3 / "session_times.csv"
OUT = FIG3 / "session-timeline.png"
PRIMARY = [2, 3, 4, 5, 6]
BASELINE_RM_BEFORE_PLACEMENT = 5
DARK_ADAPT_MIN = 20
STAIN_BEFORE_ERG = (24, 37)

plt.rcParams.update({"font.size": 8})


def medians():
    d = pd.read_csv(TIMES)
    d = d[d["pig"].isin(PRIMARY)]
    m = {k: float(np.nanmedian(pd.to_numeric(d[k], errors="coerce")))
         for k in ("wait_min", "jet_before_min", "jet_after_min", "removal_min", "rm_after_min")}
    m["n_sessions"] = len(d)
    return m


C_DEV, C_ERG, C_ASSESS, C_DARK, C_COOL = "#1f4e79", "#c0392b", "#2e7d32", "0.15", "#4a90d9"
# y positions of the rows (top to bottom)
Y = {"dark": 6.0, "assess": 5.0, "erg": 3.9, "cool": 2.6, "device": 1.2}


def point(ax, x, row, color, approx=False, marker="o"):
    ax.plot([x], [Y[row]], marker=marker, ms=6, color=color,
            mfc="white" if approx else color, mew=1.2, zorder=4, clip_on=False)


def label(ax, x, row, text, color, dy, ha="center"):
    ax.annotate(text, (x, Y[row]), xytext=(0, dy), textcoords="offset points",
                ha=ha, va="bottom" if dy > 0 else "top", fontsize=6.8, color=color)


def main():
    m = medians()
    wait, jb, ja = m["wait_min"], m["jet_before_min"], m["jet_after_min"]
    rem, rma = m["removal_min"], m["rm_after_min"]
    place = -wait
    rm_base = place - BASELINE_RM_BEFORE_PLACEMENT
    dark0 = rm_base - DARK_ADAPT_MIN
    stain = rm_base - np.mean(STAIN_BEFORE_ERG)
    iop, photo = stain - 5, stain - 10
    post0 = rma + 4                     # post assessments start after the RM recording
    post_len = 16
    stain_len = 6                       # post-cooling staining, in the dark (2026-10-31)
    lights_on = post0 + stain_len       # the lights go on after the staining; tonometry and photos follow
    lights_off = stain - 2              # the room goes dark for the staining, before dark adaptation

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 0.85], wspace=0)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    xA = (photo - 6, 0)
    xB = (0, 120)
    xC = (0, post0 + post_len + 4)
    for ax, xl in ((axA, xA), (axB, xB), (axC, xC)):
        ax.set_xlim(*xl)
        ax.set_ylim(0.3, 6.6)
        ax.patch.set_visible(False)     # markers on the shared lines show from both sides
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
        for r in ("assess", "erg", "device"):
            ax.axhline(Y[r], color="0.88", lw=0.6, zorder=0)
    axA.set_yticks([Y["dark"], Y["assess"], Y["erg"], Y["cool"], Y["device"]],
                   ["Room", "Eye\nassessments", "ERG\n(test eye)", "Cooling", "Devices"])
    axA.tick_params(axis="y", length=0)

    # dark period: from the baseline staining to the end of the post-cooling staining
    for ax, (a, b) in ((axA, (lights_off, xA[1])), (axB, xB), (axC, (xC[0], lights_on))):
        ax.add_patch(Rectangle((a, Y["dark"] - 0.17), b - a, 0.34, color=C_DARK, lw=0, zorder=1))
    axA.text(lights_off + 1, Y["dark"], "lights off, red light", color="white", fontsize=6.6,
             ha="left", va="center", zorder=2)
    axB.text(np.mean(xB), Y["dark"], "dark", color="white", fontsize=6.6, ha="center",
             va="center", zorder=2)
    axC.text(lights_on, Y["dark"] + 0.25, "lights on", fontsize=6.6, ha="center", va="bottom")

    # ---- A: before switch-on
    point(axA, photo, "assess", C_ASSESS, approx=True)
    label(axA, photo, "assess", "photos", C_ASSESS, 8)
    point(axA, iop, "assess", C_ASSESS, approx=True)
    label(axA, iop, "assess", "tonometry", C_ASSESS, -9)
    point(axA, stain, "assess", C_ASSESS, approx=True)
    label(axA, stain, "assess", "staining", C_ASSESS, 8, ha="left")
    axA.annotate("", xy=(rm_base, Y["erg"] + 0.55), xytext=(dark0, Y["erg"] + 0.55),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35"))
    axA.text((rm_base + dark0) / 2, Y["erg"] + 0.62, f"dark adaptation\n{DARK_ADAPT_MIN} min",
             fontsize=6.4, ha="center", va="bottom", color="0.25")
    point(axA, rm_base, "erg", C_ERG)                # same marker as the RM after cooling (2026-11-01); its time stays approximate
    label(axA, rm_base, "erg", "RM", C_ERG, -9)
    point(axA, -jb, "erg", C_ERG)
    label(axA, -jb, "erg", "Jet", C_ERG, -9)
    point(axA, place, "device", C_DEV, marker="v")
    label(axA, place, "device", "placement", C_DEV, -9)
    point(axB, 0, "device", C_DEV, marker="v")     # on the later axes, above their row line
    label(axA, 0, "device", "switch-on", C_DEV, 8, ha="right")
    axA.axvline(0, color=C_DEV, lw=0.8, ls="--", zorder=5, clip_on=False)
    axA.set_xticks([-60, -40, -20, 0])

    # ---- B: cooling, the three durations
    for i, dur in enumerate((30, 60, 120)):
        y = Y["cool"] + 0.45 - i * 0.45
        axB.add_patch(Rectangle((0, y - 0.14), dur, 0.28, color=C_COOL, alpha=0.35 + 0.25 * i,
                                lw=0))
        axB.text(dur - 2, y, f"{dur} min", fontsize=6.8, ha="right", va="center")
        point(axB, dur / 2, "erg", C_ERG)
        label(axB, dur / 2, "erg", f"{dur // 2}", C_ERG, 8)
    axB.text(62, Y["erg"] - 0.3, "Jet during cooling, at mid-session", ha="center",
             va="top", fontsize=6.6, color=C_ERG)
    axB.text(62, Y["device"], "ring held at its set point", ha="center", va="center",
             fontsize=6.8, color=C_DEV)
    axB.set_xticks([30, 60, 120])

    # ---- C: after switch-off
    point(axC, 0, "device", C_DEV, marker="v")
    label(axC, 0, "device", "switch-off", C_DEV, 8, ha="left")
    point(axC, rem, "device", C_DEV, marker="v")
    label(axC, rem, "device", "removal", C_DEV, -9)
    point(axC, ja, "erg", C_ERG)
    label(axC, ja, "erg", "Jet", C_ERG, -9)
    point(axC, rma, "erg", C_ERG)
    label(axC, rma, "erg", "RM", C_ERG, 8)
    # the same open markers as before cooling: staining in the dark, then tonometry and photos with the lights on
    pstain, piop, pphoto = post0 + stain_len / 2, lights_on + 3, lights_on + 8
    point(axC, pstain, "assess", C_ASSESS, approx=True)
    label(axC, pstain, "assess", "staining", C_ASSESS, 8, ha="right")
    point(axC, piop, "assess", C_ASSESS, approx=True)
    label(axC, piop, "assess", "tonometry", C_ASSESS, -9)
    point(axC, pphoto, "assess", C_ASSESS, approx=True)
    label(axC, pphoto, "assess", "photos", C_ASSESS, 8)
    axC.axvline(0, color=C_DEV, lw=0.8, ls="--", zorder=5, clip_on=False)
    axC.set_xticks([round(ja), round(rem)], [f"+{round(ja)}", f"+{round(rem)}"])

    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.14, top=0.97)
    pA, pB, pC = axA.get_position(), axB.get_position(), axC.get_position()
    fig.text((pA.x0 + pB.x1) / 2, 0.02, "Minutes from switch-on", ha="center", va="bottom")
    fig.text((pC.x0 + pC.x1) / 2, 0.02, "Minutes after switch-off", ha="center", va="bottom")
    fig.savefig(OUT, dpi=300)
    plt.close(fig)

    print(f"medians over {m['n_sessions']} sessions of animals {PRIMARY}: wait {wait:.1f}, "
          f"Jet before {jb:.1f}, Jet after {ja:.1f}, removal {rem:.1f}, RM after {rma:.1f} min")
    record(__file__, [OUT], [TIMES],
           origins={"figures/aim3/session_times.csv": "code/figures/aim3_session_times_table.py output"},
           notes=(f"Medians of animals {PRIMARY} from session_times.csv: wait {wait:.1f}, "
                  f"jet_before {jb:.1f}, jet_after {ja:.1f}, removal {rem:.1f}, rm_after "
                  f"{rma:.1f} min. Baseline assessment times approximate (Methods): RM "
                  f"{BASELINE_RM_BEFORE_PLACEMENT} min before placement, dark adaptation "
                  f"{DARK_ADAPT_MIN} min, staining {STAIN_BEFORE_ERG} min before the baseline ERG. "
                  "UPDATE-PIGS56."),
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
