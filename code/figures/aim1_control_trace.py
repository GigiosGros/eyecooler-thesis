"""Chapter 2 Methods: automatic feedback control of the SCR (control-trace.png).

Redrawn from the bench recording behind Methods-manuscript Fig. 6, in the style of the
other thesis plots (the image taken from the manuscript was a spreadsheet chart).

Source  03 - Experiments & Data/Bench Data & Working Notes/Automated supply/
        Automated supply.xlsx, sheet "Preserved_Air23.1°C_Ice4°C_Auto" (recorded
        2023-06-15, device on a preserved porcine eye; columns: time in s, thermistor
        voltage, SCR temperature in °C, 5 V rail, TEC current in A; 10 samples per s).
        The first 120 s are drawn, as in the manuscript figure. Nothing is filtered.
A snapshot of the workbook is kept in figures/aim1/methods/sources/.

Run:  python code/figures/aim1_control_trace.py
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from provenance import THESIS, record

BOX = THESIS.parents[2]
ORIGIN = (BOX / "03 - Experiments & Data" / "Bench Data & Working Notes" / "Automated supply"
          / "Automated supply.xlsx")
SHEET = "Preserved_Air23.1°C_Ice4°C_Auto"
OUT = THESIS / "figures" / "aim1" / "methods"
SNAP = OUT / "sources" / "automated-supply.xlsx"
DST = OUT / "control-trace.png"
T_END = 120.0          # s, as in the manuscript figure
SET_POINT = 4.0        # °C
C_TEMP, C_CURR = "#1f4e79", "#c0504d"   # as in the thesis-analysis package figures


def main():
    if not SNAP.exists():
        shutil.copyfile(ORIGIN, SNAP)
    d = pd.read_excel(SNAP, sheet_name=SHEET)
    d.columns = ["t", "v_therm", "T", "v_rail", "I"]
    d["t"] = pd.to_numeric(d["t"])
    d = d[d["t"] <= T_END]

    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ax2 = ax.twinx()
    h2, = ax2.plot(d["t"], d["I"], color=C_CURR, lw=0.8, zorder=1, label="TEC current")
    h1, = ax.plot(d["t"], d["T"], color=C_TEMP, lw=1.8, zorder=3, label="SCR temperature")
    ax.axhline(SET_POINT, color="0.35", lw=0.7, ls="--", zorder=2)
    ax.text(2, SET_POINT + 0.5, "set point, 4 °C", fontsize=8, color="0.25", va="bottom")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_xlim(0, T_END)
    ax.set_ylim(0, 25)
    ax2.set_ylim(0, 5)                  # 1 A per 5 °C grid line
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.82", lw=0.5)
    ax.grid(which="minor", color="0.93", lw=0.4)
    ax.tick_params(which="minor", length=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SCR temperature (°C)")
    ax2.set_ylabel("TEC current (A)")
    ax.legend(handles=[h1, h2], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(DST, dpi=300)
    plt.close(fig)

    record(__file__, [DST], [SNAP], manifest=OUT.parent / "MANIFEST.json",
           origins={"figures/aim1/methods/sources/automated-supply.xlsx":
                    str(ORIGIN.relative_to(BOX)).replace("\\", "/")},
           notes="Sheet '%s', first %d s, unfiltered. Replaces control-trace.jpg "
                 "(Methods manuscript Fig. 6, a spreadsheet chart of the same recording)."
                 % (SHEET, T_END))
    print("written:", DST)


if __name__ == "__main__":
    main()
