"""Chapter 4: Figure fig:aim3-systemic (systemic parameters during the cooling sessions).

Output (figures/aim3/):
  systemic.png     core temperature, heart rate, mean arterial pressure and end-tidal CO2
                   against time from switch-on, one line per session; one column per
                   cooling duration (30, 60, 120 min); switch-on (0 min) and switch-off
                   (end of the cooling, from session_times.csv) marked
  MANIFEST.json    provenance (provenance.record)

Inputs (figures/aim3/sources/, origins in sources/origins.json):
  systemic/PigN_Sk_monitoring.csv
                   the "Monitoring" sheet of each session's SystemicParameters workbook
                   (08 - Systemic Parameters; transcribed from the anesthesia record, one row
                   per reading, about every 10 min), copied by --refresh. Columns used:
                   Time, HR (bpm), SAP mean, EtCO2, Temp (F) (converted to degC here).
  ../session_times.csv
                   switch-on and switch-off clock times of each session
                   (aim3_session_times_table.py; effective events of the ring record, the
                   session log for animal 3 session 3 and the log placement of animal 1
                   session 1)

Animal 1 is exploratory: grey and dashed, labelled so in the legend.

Run:  python code/figures/aim3_systemic_figure.py            (from the copies)
      python code/figures/aim3_systemic_figure.py --refresh  (re-copy the inputs first)
"""
import csv
import datetime
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance import THESIS, record
from aim3_invivo_figures import STUDY, SESSIONS, DURATION, PIGS, animal_style, EXPLORATORY

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources"
SYS = SRC / "systemic"
TIMES = FIG3 / "session_times.csv"
OUT = FIG3 / "systemic.png"
COLS = {"HR (bpm)": "hr", "SAP mean": "map", "EtCO2": "etco2", "Temp (F)": "temp_F"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def _hms(v):
    if isinstance(v, datetime.datetime):
        return v.time()
    if isinstance(v, datetime.time):
        return v
    if isinstance(v, str) and ":" in v:
        parts = [int(x) for x in v.strip().split(":")]
        return datetime.time(*parts)
    return None


def refresh():
    import openpyxl
    SYS.mkdir(parents=True, exist_ok=True)
    origins = {}
    for pig in PIGS:
        for s, sname in SESSIONS.items():
            files = sorted((STUDY / f"Pig {pig}" / sname / "08 - Systemic Parameters")
                           .glob("*SystemicParameters*.xlsx"))
            if not files:
                continue
            src = files[0]
            wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
            rows = list(wb["Monitoring"].iter_rows(values_only=True))
            wb.close()
            head = [str(h) if h is not None else "" for h in rows[0]]
            out = SYS / f"Pig{pig}_S{s}_monitoring.csv"
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(head)
                for r in rows[1:]:
                    if not r or r[0] is None:
                        continue
                    t = _hms(r[0])
                    w.writerow([t.strftime("%H:%M:%S") if t else r[0]] + list(r[1:len(head)]))
            origins[str(out.relative_to(THESIS)).replace("\\", "/")] = (
                str(src) + " (sheet Monitoring, copied as CSV)")
    old = SRC / "origins.json"
    merged = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    merged.update(origins)
    old.write_text(json.dumps(merged, indent=1), encoding="utf-8")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _minutes(clock):
    h, m, s = (int(v) for v in clock.split(":"))
    return h * 60 + m + s / 60


def load():
    times = pd.read_csv(TIMES)
    out = []
    for _, r in times.iterrows():
        pig, s = int(r["pig"]), int(r["session"])
        if pig not in PIGS:
            continue
        f = SYS / f"Pig{pig}_S{s}_monitoring.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype=str)
        on, off = _minutes(r["switch_on"]), _minutes(r["switch_off"])
        rec = pd.DataFrame({"t_min": [_minutes(t) - on for t in d["Time"]]})
        for col, key in COLS.items():
            rec[key] = [_num(v) for v in d[col]]
        rec["temp_C"] = (rec["temp_F"] - 32) * 5 / 9
        out.append((pig, s, off - on, rec))
    return out


PANELS = [("temp_C", "Core temperature (°C)"), ("hr", "Heart rate (bpm)"),
          ("map", "Mean arterial\npressure (mmHg)"), ("etco2", "End-tidal CO$_2$ (mmHg)")]


def figure(data, out):
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.4), sharey="row", sharex="col",
                             gridspec_kw={"width_ratios": [1, 1.25, 1.75]})
    for c, s in enumerate((1, 2, 3)):
        for pig, ss, cool_min, rec in data:
            if ss != s:
                continue
            st = animal_style(pig)
            for r, (key, _) in enumerate(PANELS):
                m = rec[key].notna()
                axes[r, c].plot(rec["t_min"][m], rec[key][m], marker="o", ms=2.2, lw=0.8,
                                color=st["color"], ls=st["ls"], mfc=st["mfc"],
                                label=st["label"], zorder=1 if pig in EXPLORATORY else 2)
            # switch-off of this session: a short tick on the top panel and a faint line
            for r in range(4):
                axes[r, c].axvline(cool_min, lw=0.6, color=st["color"], alpha=0.45, ls=":")
        for r in range(4):
            axes[r, c].axvline(0, lw=0.9, color="0.3")
        axes[0, c].set_title(f"{DURATION[s]}-min sessions", fontsize=9, pad=12)
        # labels above the top panel: switch-on left of its line, switch-off right of it
        axes[0, c].annotate("on", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(-2, 1), textcoords="offset points", fontsize=6.5,
                            ha="right", va="bottom", color="0.3")
        axes[0, c].annotate("off", xy=(DURATION[s], 1.0),
                            xycoords=("data", "axes fraction"), xytext=(2, 1),
                            textcoords="offset points", fontsize=6.5, ha="left",
                            va="bottom", color="0.3")
        # from 2 h before switch-on (anesthesia records start 1.5 to 3 h before) to
        # 1 h after the planned switch-off
        axes[3, c].set_xlim(-120, DURATION[s] + 60)
        axes[3, c].set_xlabel("Time from switch-on (min)")
    for r, (_, lab) in enumerate(PANELS):
        axes[r, 0].set_ylabel(lab)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labs = sorted(handles)
    fig.legend([handles[k] for k in labs], labs, frameon=False, fontsize=7,
               loc="lower center", ncol=len(labs))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    if "--refresh" in sys.argv:
        refresh()
    data = load()
    figure(data, OUT)
    for pig, s, cool, rec in data:
        print(f"Pig {pig} S{s}: {len(rec)} readings, {rec['t_min'].min():.0f} to "
              f"{rec['t_min'].max():.0f} min, switch-off {cool:.1f} min; "
              f"T {rec['temp_C'].min():.2f}-{rec['temp_C'].max():.2f} C, "
              f"HR {rec['hr'].min():.0f}-{rec['hr'].max():.0f}, "
              f"MAP {rec['map'].min():.0f}-{rec['map'].max():.0f}, "
              f"EtCO2 {rec['etco2'].min():.0f}-{rec['etco2'].max():.0f}")
    inputs = sorted(SYS.glob("*.csv")) + [TIMES]
    origins_f = SRC / "origins.json"
    origins = json.loads(origins_f.read_text(encoding="utf-8")) if origins_f.exists() else {}
    origins = {k: v for k, v in origins.items() if k.startswith("figures/aim3/sources/systemic/")}
    origins["figures/aim3/session_times.csv"] = "code/figures/aim3_session_times_table.py output"
    record(__file__, [OUT], inputs, origins=origins,
           notes="Monitoring sheets of animals 1-4, sessions 1-3; animal 1 exploratory. "
                 "UPDATE-PIGS56: re-run with --refresh after animals 5 and 6.",
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
