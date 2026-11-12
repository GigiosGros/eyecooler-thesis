"""Chapter 3: bench performance of the integrated control board (board-bench.png).

Three panels, drawn from the bench logs of the firmware repository
(GitHub GigiosGros/SCEC-STM32-Firmware; set FIRMWARE_EVIDENCE to its docs/evidence folder), folder docs/evidence. Copies of the three
inputs are kept in figures/aim2/sources/bench/.

  (A) P8_soak_20260731.csv, run 1 -- closed-loop pull-down of the ring to the 4 degC set
      point and hold, 18 min at 5 Hz (firmware 0.9.47): ring temperature (column Tscr_C)
      and module current (I_A).
  (B) BENCH_20260824_ILIMC_STAIRCASE.md, table of section 1 -- module current reported
      by the board (column TECMON I) against a multimeter in series (column DMM), six
      settled points. The values were transcribed from the console during the session
      (the stream was not saved), so they are read here from the table of the note.
      Load: the module. Firmware 0.9.54, before the conversion constants were refitted.
      BENCH_20260831_0901_SAF14_AND_H20_ON_A_RESISTOR.md, cooling table of section 2
      (multimeter in series at each drive step) with bench_20260831.txt (UTF-16, the
      board's own reading at the same steps: median of the second half of the last
      complete stream at each step) -- thirteen points from 0.48 to 3.93 A into a
      0.24 ohm resistor, firmware 0.9.59, after the constants were refitted.
  (C) bench_20260827.txt (UTF-16), the 30-s TECMON stream that ends in FAULT TEC_OPEN --
      the open-circuit cutout with the module disconnected on purpose (firmware 0.9.59):
      module voltage applied by the loop (V) and measured current (I) until the cutout.

Plotting only; no value is computed except the summary printed for the text.

Run:  python code/figures/aim2_board_bench.py            (from the copies)
      python code/figures/aim2_board_bench.py --refresh  (re-copy the logs first)
"""
import os
import pathlib
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance import FIG, SOURCES, THESIS, record

EVIDENCE = pathlib.Path(os.environ.get("FIRMWARE_EVIDENCE",
                                    THESIS.parent / "SCEC-STM32-Firmware" / "docs" / "evidence"))
BENCH = SOURCES / "bench"
SOAK, STAIR, OPEN = "P8_soak_20260731.csv", "BENCH_20260824_ILIMC_STAIRCASE.md", "bench_20260827.txt"
RES_NOTE, RES_LOG = "BENCH_20260831_0901_SAF14_AND_H20_ON_A_RESISTOR.md", "bench_20260831.txt"
SET_POINT_C = 4.0
C_T, C_I, C_V = "#1f77b4", "#444444", "#d62728"      # ring (as in the other figures), current, voltage

# 2026-11-11: drawn at the printed width (6.0 in = textwidth, incl. the tight-bbox pad) so the smallest text prints at 7 pt
# (was 9.6 in wide at font 9/8, i.e. 4.9-6.2 pt in print).
plt.rcParams.update({"font.size": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def soak():
    d = pd.read_csv(BENCH / SOAK)
    return d[d["run"] == 1]


def staircase():
    """(multimeter A, board A) for the settled points of the staircase table."""
    rows = []
    for line in (BENCH / STAIR).read_text(encoding="utf-8").splitlines():
        # DAC | OUT2 | drive | DMM | TECMON I | ...   (units written in the first row only;
        # the last row has no multimeter reading and a range for the board, and is skipped)
        m = re.match(r"\|\s*\d{4}\s*\|[^|]*\|[^|]*\|\s*([\d.]+)(?: A)?\s*\|\s*([\d.]+)(?: A)?\s*\|", line)
        if m:
            rows.append((float(m.group(1)), float(m.group(2))))
    return np.array(rows)


def resistor_staircase():
    """(multimeter A, board A) for the cooling steps into the resistor on 31 August."""
    note = (BENCH / RES_NOTE).read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(note) if l.startswith("| DAC |"))
    cells = lambda l: [c.strip(" *") for c in l.strip().strip("|").split("|")][1:]
    meter = dict(zip(map(int, cells(note[i])), map(float, cells(note[i + 2]))))
    streams, cur = [], None
    for l in (BENCH / RES_LOG).read_bytes().decode("utf-16").splitlines():
        m = re.search(r"TEMP_SET = (\d+) counts", l)
        if m:
            cur = (int(m.group(1)), [])
            streams.append(cur)
            continue
        m = re.match(r"\s*(\d+)ms\s+(-?[\d.]+) A\s+(-?[\d.]+) V", l)
        if m and cur:
            cur[1].append(float(m.group(2)))
    board = {}
    for dac, amps in streams:
        if dac in meter and len(amps) >= 50:          # complete streams only; the last one at each step
            board[dac] = float(np.median(amps[len(amps) // 2:]))
    return np.array([(meter[d], board[d]) for d in sorted(meter, reverse=True)])


def open_circuit():
    """t (s), I (A), V (V) of the stream that ends in FAULT TEC_OPEN, and the time of the last sample."""
    lines = (BENCH / OPEN).read_bytes().decode("utf-16").splitlines()
    end = next(i for i, l in enumerate(lines) if "FAULT TEC_OPEN" in l)
    start = max(i for i, l in enumerate(lines[:end]) if "TEC ON (servo)" in l)
    rows = []
    for l in lines[start:end]:
        m = re.match(r"\s*(\d+)ms\s+(-?[\d.]+) A\s+(-?[\d.]+) V", l)
        if m:
            rows.append((int(m.group(1)) / 1000, float(m.group(2)), float(m.group(3))))
    return np.array(rows)


def main():
    BENCH.mkdir(parents=True, exist_ok=True)
    for name in (SOAK, STAIR, OPEN, RES_NOTE, RES_LOG):
        if "--refresh" in sys.argv or not (BENCH / name).exists():
            shutil.copy2(EVIDENCE / name, BENCH / name)
    s, st, oc, rs = soak(), staircase(), open_circuit(), resistor_staircase()

    fig = plt.figure(figsize=(5.85, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.0, 1.1], height_ratios=[1.35, 1.0])

    # (A) regulation and hold
    ax = fig.add_subplot(gs[0, 0])
    t = s["t_s"] / 60
    ax.axhline(SET_POINT_C, color="0.6", lw=0.8, ls="--")
    ax.plot(t, s["Tscr_C"], color=C_T, lw=1.2)
    ax.text(t.max(), SET_POINT_C + 1.2, "set point 4 °C", ha="right", va="bottom", fontsize=7, color="0.35")
    ax.set_ylabel("Ring temperature (°C)")
    ax.set_ylim(0, 33)
    ax.set_title("(A) Pull-down and hold", fontsize=8, loc="left")
    ax.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.plot(t, s["I_A"], color=C_I, lw=1.0)
    ax2.set_ylabel("Current (A)")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylim(-0.6, 4.4)
    ax2.set_xlim(0, t.max())

    # (B) current monitor against the multimeter
    ax = fig.add_subplot(gs[:, 1])
    lo, hi = 0.0, 4.2
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", label="identity")
    # 2026-10-07: only the final comparison is drawn (31 August, resistor, refitted constants);
    # the six points of 24 August with the module are still read and printed below for the record.
    ax.plot(rs[:, 0], rs[:, 1], "o", ms=3.5, mfc="white", mec=C_I, mew=1.0, label="settled point")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Multimeter in series (A)")
    ax.set_ylabel("Board current monitor (A)")
    ax.set_title("(B) Current monitor", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right", handlelength=1.6, borderaxespad=0.2)

    # (C) open-circuit cutout
    t_trip = oc[-1, 0]
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(oc[:, 0], oc[:, 2], color=C_V, lw=1.2)
    ax.axvline(t_trip, color="k", lw=0.9)
    ax.text(t_trip - 0.15, 0.03, "cutout, %.1f s" % t_trip, rotation=90, ha="right", va="bottom", fontsize=7,
            transform=ax.get_xaxis_transform())
    ax.set_ylabel("Module voltage (V)")
    ax.set_title("(C) Module disconnected", fontsize=8, loc="left")
    ax.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 2], sharex=ax)
    ax2.plot(oc[:, 0], oc[:, 1], color=C_I, lw=1.0)
    ax2.axvline(t_trip, color="k", lw=0.9)
    ax2.set_ylabel("Current (A)")
    ax2.set_xlabel("Time after the drive was applied (s)", loc="right")   # wider than the axis at 6 in
    ax2.set_ylim(-0.6, 4.4)
    ax2.set_xlim(0, 10)
    ax2.set_xticks(range(0, 11, 2))

    out = FIG / "board-bench.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    origin = "SCEC-STM32-Firmware/docs/evidence/"
    names = (SOAK, STAIR, OPEN, RES_NOTE, RES_LOG)
    record(__file__, [out], [BENCH / n for n in names],
           origins={n: origin + n for n in names},
           notes="(A) P8 soak run 1, fw 0.9.47; (B) staircase table of 2026-08-24 (module, fw 0.9.54), values "
                 "transcribed from the console in the session note, and the cooling steps of 2026-08-31 into a "
                 "0.24 ohm resistor (fw 0.9.59): multimeter from the session note, board reading from the log; "
                 "(C) TECMON stream ending in FAULT TEC_OPEN, fw 0.9.59.")
    print("written:", out.name)

    # numbers for the text
    first = s[s["Tscr_C"] <= SET_POINT_C + 0.3]["t_s"].min()
    low = s.loc[s["Tscr_C"].idxmin()]
    hold = s[s["t_s"] >= 700]
    print("(A) run of %.1f min; start %.1f degC; within 0.3 degC of the set point first at %.0f s; lowest %.2f degC "
          "at %.0f s; from 700 s to the end (%.1f min): mean %.2f, min %.2f, max %.2f degC, current %.2f A"
          % (s["t_s"].max() / 60, s["Tscr_C"].iloc[0], first, low["Tscr_C"], low["t_s"],
             (s["t_s"].max() - 700) / 60, hold["Tscr_C"].mean(), hold["Tscr_C"].min(), hold["Tscr_C"].max(),
             hold["I_A"].mean()))
    print("(B) %d points, multimeter %.2f-%.2f A; board minus multimeter: %s" % (
        len(st), st[:, 0].min(), st[:, 0].max(), np.round(st[:, 1] - st[:, 0], 2).tolist()))
    print("(B) resistor: %d points, multimeter %.2f-%.2f A; board minus multimeter: %s" % (
        len(rs), rs[:, 0].min(), rs[:, 0].max(), np.round(rs[:, 1] - rs[:, 0], 2).tolist()))
    print("(C) %d samples, last at %.1f s; voltage %.2f -> %.2f V; current %.2f to %.2f A" % (
        len(oc), t_trip, oc[0, 2], oc[-1, 2], oc[:, 1].min(), oc[:, 1].max()))


if __name__ == "__main__":
    main()
