"""Chapter 4: ring temperature and intraocular pressure of the in-vivo study.

Outputs (figures/aim3/):
  ring-temperature.png         ring temperature of the test device against time from
                               switch-on, one panel per cooling duration (30, 60, 120 min)
  iop-change.png               change in intraocular pressure over each session, test eye
                               and control eye, per animal
  ring_temperature_summary.csv per-session timing and ring-temperature figures quoted in
                               Section 4.3 (computed here, printed on every run)
  MANIFEST.json                provenance (provenance.record)

Inputs, copied by --refresh from the study folder (read only) into figures/aim3/sources/:
  ring/PigN_Sk.csv     1-s means of "Temperature Processed.csv" of each session
                       (_Tools/convert_temperature.py output, 10 Hz; since 2026-10-28 the
                       physical circuit model V=(aR+b)/(R+g) with the 10K3A thermistor law,
                       fitted per session; Device Off from the last current pulse)
  ring/events.json     effective event times of each session from "Temperature
                       Events.xlsx" (Manual Time overrides Auto Time, as in the tools; any
                       text in Manual Time, e.g. "none" or "not recorded" = no time, never
                       the auto time), the events tagged NOT RECORDED in its "Recorded?"
                       column (times from the session log, outside the recording), and the
                       restart list

Animal 1 is exploratory (EXPLORATORY): drawn grey and dashed/hollow, labelled so in the
legends, and flagged in the animal_set column of the summary CSV.
  master_iop.csv, master_scores.csv, master_cooling.csv, master_systemic.csv
                       Live Pigs Master Analysis (_Tools/master_analysis.py)

Definitions used for the summary (all times from the effective events):
  wait_min        switch-on minus placement on the eye
  cooling_min     switch-off minus switch-on
  removal_min     removal minus switch-off
  ring_before_C   mean ring temperature over the 60 s before switch-on
  reach_4p5_min   first time after switch-on with the ring at or below 4.5 C (blank if never)
  regulation_min  "Regulation Start" of the tools (relay/switch begins cycling) minus switch-on
  reach_hold_plus0p5_min  first time after switch-on within 0.5 C of the session's own
                  hold median
  hold_*          ring temperature from Regulation Start + 2 min to 1 min before switch-off
                  (the detected switch-off can trail the drive by up to a minute), excluding
                  each interruption of the drive (restart off time to back-on time + 3 min)
  min_C           lowest ring temperature between switch-on and switch-off
  removal_C       ring temperature at removal (mean of the 10 s before)

Session 3 of animal 3 has no ring record (recording lost; see its README) and is
listed with the bench-sheet times only.

Run:  python code/figures/aim3_invivo_figures.py            (from the copies)
      python code/figures/aim3_invivo_figures.py --refresh  (re-copy the inputs first)
"""
import json
import os
import pathlib
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance import THESIS, record

STUDY = pathlib.Path(os.environ.get("LIVE_PIG_STUDY", THESIS / "data" / "Live Pigs Safety Study"))
FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources"
RING = SRC / "ring"
MASTER = ["master_iop.csv", "master_scores.csv", "master_cooling.csv", "master_systemic.csv"]
SESSIONS = {1: "Session 1 - 30min", 2: "Session 2 - 1hr", 3: "Session 3 - 2hr"}
DURATION = {1: 30, 2: 60, 3: 120}
PIGS = [1, 2, 3, 4, 5, 6]    # sessions done so far; missing sessions are skipped
SET_POINT_C = 4.0
REACH_C = 4.5
COLORS = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c", 4: "#9467bd", 5: "#ff7f0e", 6: "#8c564b"}
# Animal 1 is EXPLORATORY (2026-10-28; thesis subsec:aim3-analysis "Animals analyzed"):
# reported per session, drawn grey and dashed/hollow, kept out of the summary statistics.
EXPLORATORY = {1}
EXPL_COLOR = "0.55"


def animal_style(pig):
    """Line/marker style and legend label of an animal in the Chapter 4 figures."""
    if pig in EXPLORATORY:
        return {"color": EXPL_COLOR, "ls": "--", "mfc": "white",
                "label": f"Animal {pig} (exploratory)"}
    return {"color": COLORS[pig], "ls": "-", "mfc": COLORS[pig], "label": f"Animal {pig}"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- refresh
def _events(xlsx):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["Events"]
    ev, restarts, in_restarts, not_recorded = {}, [], False, []
    for row in ws.iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        name = str(row[0])
        if name.startswith("#"):
            in_restarts = True
            continue
        if in_restarts:
            if isinstance(row[0], (int, float)) and row[1] is not None and row[3] is not None:
                restarts.append([float(row[1]), float(row[3])])
            continue
        if name in ("Device Connected", "Placed on Pig", "Cooling Start", "Regulation Start",
                    "Device Off", "Device Removed"):
            auto = row[1] if len(row) > 1 else None
            man = row[3] if len(row) > 3 else None
            status = row[5] if len(row) > 5 else None
            # Manual Time overrides Auto Time. Any text in Manual Time ("none", "not
            # recorded", ...) means the event has no time in this recording: None, never
            # the auto time. (2026-10-29)
            if isinstance(man, str) and man.strip():
                ev[name] = None
            elif isinstance(man, (int, float)) and not isinstance(man, bool):
                ev[name] = float(man)
            elif isinstance(auto, (int, float)) and not isinstance(auto, bool):
                ev[name] = float(auto)
            else:
                ev[name] = None
            if isinstance(status, str) and "NOT RECORDED" in status.upper():
                not_recorded.append(name)
    wb.close()
    return {"events": ev, "restarts": restarts, "not_recorded": not_recorded}


def refresh():
    RING.mkdir(parents=True, exist_ok=True)
    allev, origins = {}, {}
    for pig in PIGS:
        for s, sname in SESSIONS.items():
            cool = STUDY / f"Pig {pig}" / sname / "07 - Cooling"
            proc, evx = cool / "Temperature Processed.csv", cool / "Temperature Events.xlsx"
            if not proc.exists() or not evx.exists():
                continue
            key = f"Pig{pig}_S{s}"
            allev[key] = _events(evx)
            d = pd.read_csv(proc, encoding="latin-1")
            # first four columns: time, clock, current, temperature (blanked spans empty);
            # a recording with a Coverage sheet adds "Temp raw" and "Coverage", not used here.
            # Since 2026-11-04 LivePigs-Tools writes old-method sessions CORRECTED (+5.5 C) in
            # column 4 and keeps the reading as calibrated in "Temp uncorrected (degC)". This
            # script adds RING_OFFSET_C itself (ring_offset), so it takes the uncorrected one.
            unc = next((c for c in d.columns if str(c).startswith("Temp uncorrected")), None)
            if unc is not None:
                d.iloc[:, 3] = d[unc]
            d = d.iloc[:, :4]
            d.columns = ["t_s", "clock", "current_A", "temp_C"]
            d["sec"] = np.floor(d["t_s"]).astype(int)
            g = d.groupby("sec").agg(temp_C=("temp_C", "mean"), current_A=("current_A", "mean"))
            g.index.name = "t_s"
            out = RING / f"{key}.csv"
            g.round(3).to_csv(out)
            origins[str(out.relative_to(THESIS)).replace("\\", "/")] = (
                str(proc) + " (1-s means) ; events: " + str(evx))
    (RING / "events.json").write_text(json.dumps(allev, indent=1), encoding="utf-8")
    for f in MASTER:
        src = STUDY / "Live Pigs Master Analysis" / f
        shutil.copy2(src, SRC / f)
        origins[f"figures/aim3/sources/{f}"] = str(src)
    # merge with the entries written by other scripts (e.g. aim3_session_times_table.py)
    old = SRC / "origins.json"
    merged = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    merged.update(origins)
    old.write_text(json.dumps(merged, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- analysis
def load_ring():
    ev = json.loads((RING / "events.json").read_text(encoding="utf-8"))
    out = {}
    for key, e in ev.items():
        d = pd.read_csv(RING / f"{key}.csv")
        ev_k = dict(e["events"])
        ev_k["_not_recorded"] = list(e.get("not_recorded", []))
        out[key] = (d, ev_k, e["restarts"])
    return out


def cooling_recorded(ev):
    """False when switch-on or switch-off is tagged NOT RECORDED (times from the log only)."""
    nr = ev.get("_not_recorded", [])
    return not ("Cooling Start" in nr or "Device Off" in nr)


def summary(ring):
    rows = []
    for key, (d, ev, rs) in sorted(ring.items()):
        pig, s = int(key[3]), int(key[-1])
        on, off = ev.get("Cooling Start"), ev.get("Device Off")
        placed, removed = ev.get("Placed on Pig"), ev.get("Device Removed")
        if on is None or off is None:
            continue
        if not cooling_recorded(ev):
            # cooling not in the recording (e.g. animal 3 session 3): log times only
            rows.append({"pig": pig, "session": s, "duration_planned_min": DURATION[s],
                         "wait_min": (on - placed) / 60 if placed is not None else np.nan,
                         "cooling_min": (off - on) / 60,
                         "removal_min": (removed - off) / 60 if removed is not None else np.nan,
                         "restarts_n": len(rs),
                         "restart_offsets_min": "cooling NOT RECORDED (times from the session log)"})
            continue
        t, T = d["t_s"].to_numpy(float), d["temp_C"].to_numpy(float)
        row = {"pig": pig, "session": s, "duration_planned_min": DURATION[s],
               "wait_min": (on - placed) / 60 if placed is not None else np.nan,
               "cooling_min": (off - on) / 60,
               "removal_min": (removed - off) / 60 if removed is not None else np.nan,
               "restarts_n": len(rs)}
        pre = (t >= on - 60) & (t < on)
        row["ring_before_C"] = np.nanmean(T[pre]) if pre.any() else np.nan
        cool = (t >= on) & (t <= off)
        hit = np.where(cool & (T <= REACH_C))[0]
        row["reach_4p5_min"] = (t[hit[0]] - on) / 60 if len(hit) else np.nan
        reg = ev.get("Regulation Start")
        row["regulation_min"] = (reg - on) / 60 if reg is not None else np.nan
        hold = (t >= reg + 120) & (t <= off - 60)
        for a, b in rs:
            hold &= ~((t >= a) & (t <= b + 180))
        h = T[hold]
        if not len(h):
            raise SystemExit(f"{key}: no hold samples (reg={reg}, on={on}, off={off})")
        lvl = np.nanmedian(h)
        hit = np.where(cool & (T <= lvl + 0.5))[0]
        row["reach_hold_plus0p5_min"] = (t[hit[0]] - on) / 60 if len(hit) else np.nan
        reach = reg
        row.update({"hold_mean_C": np.nanmean(h), "hold_sd_C": np.nanstd(h, ddof=1),
                    "hold_p05_C": np.nanpercentile(h, 5), "hold_p95_C": np.nanpercentile(h, 95),
                    "hold_min_C": np.nanmin(h), "hold_max_C": np.nanmax(h),
                    "hold_frac_3p5_4p5": float(np.mean((h >= 3.5) & (h <= 4.5))),
                    "min_C": np.nanmin(T[cool]),
                    "max_during_cooling_after_reach_C": np.nanmax(T[(t >= reach) & (t <= off)])})
        if removed is not None:
            rm = (t >= removed - 10) & (t < removed)
            row["removal_C"] = np.nanmean(T[rm]) if rm.any() else np.nan
        else:
            row["removal_C"] = np.nan
        row["restart_offsets_min"] = "; ".join(f"{(a - on) / 60:.1f}-{(b - on) / 60:.1f}" for a, b in rs)
        rows.append(row)
    # animal 3 session 3: no ring record; bench-sheet times only (README - recording failure.txt).
    # Since 2026-10-28 its Temperature Events.xlsx carries the log times tagged NOT RECORDED and
    # the row above is built from them; this fallback is kept for an older copy of the events.
    if not any(r["pig"] == 3 and r["session"] == 3 for r in rows):
        rows.append({"pig": 3, "session": 3, "duration_planned_min": 120, "cooling_min": 120.0,
                     "restarts_n": 0,
                     "restart_offsets_min": "no ring record (bench sheet 10:42-12:42)"})
    df = pd.DataFrame(rows).sort_values(["session", "pig"])
    df.insert(2, "animal_set", np.where(df["pig"].isin(EXPLORATORY), "exploratory", "primary"))
    return df


# --------------------------------------------------------------------------- figures
RING_OFFSET_C = 5.5        # thesis code/checks/ring_calibration/REPORT.md; ring_temperature_summary.csv stays as read
CONTACT_CAL = {5: 2, 6: 1}  # pig -> first session calibrated in contact with the aluminum


def ring_offset(pig, session):
    return 0.0 if session >= CONTACT_CAL.get(pig, 99) else RING_OFFSET_C


def ring_figure(ring, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True,
                             gridspec_kw={"width_ratios": [60, 90, 150]})
    for ax, s in zip(axes, (1, 2, 3)):
        for key, (d, ev, rs) in sorted(ring.items()):
            if int(key[-1]) != s:
                continue
            pig = int(key[3])
            on, off = ev.get("Cooling Start"), ev.get("Device Off")
            removed = ev.get("Device Removed")
            if on is None or off is None or not cooling_recorded(ev):
                continue
            # from placement on the eye (at most 10 min before switch-on) to removal
            # (at most 20 min after switch-off)
            placed = ev.get("Placed on Pig")
            start = max(placed if placed is not None else on - 10 * 60, on - 10 * 60)
            end = min(removed if removed is not None else off + 20 * 60, off + 20 * 60)
            m = (d["t_s"] >= start) & (d["t_s"] <= end)
            st = animal_style(pig)
            # corrected for the calibration error (2026-11-01): animals 1-4 not in contact
            ax.plot((d["t_s"][m] - on) / 60, d["temp_C"][m] + ring_offset(pig, s), lw=0.8, color=st["color"],
                    ls=st["ls"], label=st["label"], zorder=1 if pig in EXPLORATORY else 2)
        for lev in (SET_POINT_C, SET_POINT_C + RING_OFFSET_C):
            ax.axhline(lev, ls="--", lw=0.7, color="0.4")
        ax.axvline(0, lw=0.5, color="0.7")
        ax.axvline(DURATION[s], lw=0.5, color="0.7")
        ax.set_title(f"{DURATION[s]}-min session", fontsize=9)
        ax.set_xlabel("Time from switch-on (min)")
        ax.set_xlim(-10, DURATION[s] + 22)
    axes[0].set_ylabel("Ring temperature, corrected (°C)")
    axes[0].set_ylim(0, 36)
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labs = sorted(handles)
    axes[2].legend([handles[k] for k in labs], labs, frameon=False, fontsize=7, loc="upper center")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def iop_figure(iop, out):
    iop = iop[(iop["Duration_min"] > 0) & iop["Pig"].isin(PIGS)]   # UPDATE-PIGS56: master has 5-6
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, dur in zip(axes, (30, 60, 120)):
        sub = iop[iop["Duration_min"] == dur]
        for _, r in sub.iterrows():
            pig = int(r["Pig"])
            st = animal_style(pig)
            ax.plot([0, 1], [r["dTest"], r["dControl"]], marker="o", ms=3.5, lw=0.8,
                    color=st["color"], ls=st["ls"], mfc=st["mfc"], label=st["label"],
                    zorder=1 if pig in EXPLORATORY else 2)
        ax.axhline(0, lw=0.5, color="0.6")
        ax.set_xticks([0, 1], ["Test", "Control"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{dur}-min session", fontsize=9)
    axes[0].set_ylabel("Change in tonometer reading\nover the session (mmHg)")
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labs = sorted(handles)
    # one legend below the panels, so it covers no data point
    fig.legend([handles[k] for k in labs], labs, frameon=False, fontsize=7,
               loc="lower center", ncol=len(labs))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    if "--refresh" in sys.argv:
        refresh()
    FIG3.mkdir(parents=True, exist_ok=True)
    ring = load_ring()
    summ = summary(ring)
    csv_out = FIG3 / "ring_temperature_summary.csv"
    summ.round(3).to_csv(csv_out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summ.round(2).to_string(index=False))
    iop = pd.read_csv(SRC / "master_iop.csv")
    f1, f2 = FIG3 / "ring-temperature.png", FIG3 / "iop-change.png"
    ring_figure(ring, f1)
    iop_figure(iop, f2)
    inputs = sorted(RING.glob("*.csv")) + [RING / "events.json"] + [SRC / f for f in MASTER]
    origins_f = SRC / "origins.json"
    origins = json.loads(origins_f.read_text(encoding="utf-8")) if origins_f.exists() else {}
    record(__file__, [f1, f2, csv_out], inputs, origins=origins,
           notes="Ring traces are 1-s means of the 10-Hz processed records; animals 1-4. "
                 "UPDATE-PIGS56: re-run with --refresh after animals 5 and 6.",
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
