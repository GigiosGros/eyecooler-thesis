"""Chapter 4: Table tab:aim3-session-times ("Timing of the cooling sessions as performed").

Output (figures/aim3/):
  session-times-table.tex   the tabular (same column spec as the hand-typed table; header,
                            one row per animal and session, footnote rows), read by thesis.tex
                            with \\input inside the table environment, which keeps the caption
                            and label. The \\begin{tabular}/\\end{tabular} pair is in this file
                            because an \\input between rows (before \\hline) is a "Misplaced
                            \\noalign" error in pdflatex.
  session_times.csv         the same quantities at full precision (minutes), for checks
  MANIFEST.json             provenance (provenance.record)

Inputs (all under figures/aim3/sources/, see origins.json):
  ring/events.json          effective event times of each session, in seconds of the ring
                            recording (aim3_invivo_figures.py --refresh, from the study's
                            "Temperature Events.xlsx": Manual Time overrides Auto Time; Device
                            Off from the last current pulse; events tagged NOT RECORDED carry
                            the session-log time in the recording's seconds)
  ring/events_xlsx/PigN_Sk_Temperature Events.xlsx
                            copies of the same workbooks, used only for the clock of the
                            recording: the UTC "Auto Clock" of the row whose "Auto Time (s)" is
                            0 gives the local time of second 0 (America/Chicago on the session
                            date), as in _Tools/cooling_live.py
  ring/PigN_Sk.csv          1-s means of the ring temperature (for the ring temperature at the
                            Jet recording after switch-off: mean of the 2 min around it, as in
                            _Tools/erg_cohort.py ring_at)
  ring/Pig1_S1_ActivityLog.csv
                            session log of animal 1 session 1, whose ring recording began with
                            the device already on the eye: placement = first "Device
                            Application" time (test eye)
  erg/erg_plan_PigN.csv     every ERG recording of each session with its instrument time
                            corrected to the wall clock (WallTime, to the minute), its
                            electrode, and the numbers of recordings before (NPre), during
                            (NIntra) and after (NPost) cooling, in recording order

Definitions (minutes, rounded half up to the whole minute for the table):
  Wait        switch-on - placement
  Cooling     switch-off - switch-on
  Removal     removal - switch-off
  Jet before  switch-on - the last Jet recording before cooling (the last of a repeat is
              the one used: of repeated ERG recordings, the last is kept)
  Jet during  each recording during cooling - switch-on
  Jet after   the Jet recording after switch-off - switch-off; ring temperature at that time
  RM after    the last RM recording after switch-off - switch-off
Switch-on/off and removal are the effective events of the ring recording (trace times; for
animal 3 session 3, whose cooling was not recorded, the session-log times). ERG times are
the instrument times of erg_plan (WallTime). Animal 1 is exploratory (footnote b, as in
Table tab:aim3-ring).

Run:  python code/figures/aim3_session_times_table.py
"""
import csv
import datetime
import json
import math
import pathlib
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from provenance import THESIS, record

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources"
RING = SRC / "ring"
ERG = SRC / "erg"
PIGS = [1, 2, 3, 4, 5, 6]    # sessions done so far; missing sessions are skipped
SESSIONS = [1, 2, 3]
EXPLORATORY = {1}            # 2026-10-28, as in aim3_invivo_figures.py
RING_OFFSET_C = 5.5          # as in aim3_invivo_figures.py
CONTACT_CAL = {5: 2, 6: 1}
SESSION_NAME = {1: "Session 1 - 30min", 2: "Session 2 - 1hr", 3: "Session 3 - 2hr"}
LOCAL_TZ = ZoneInfo("America/Chicago")
RING_WIN_S = 60              # ring temperature = mean of +-60 s around the recording
# Sessions whose recording began with the device already on the eye: placement from the log.
LOG_PLACEMENT = {(1, 1): RING / "Pig1_S1_ActivityLog.csv"}

# column widths of the hand-typed table (sum 12.9 cm + 16 x 4 pt gutters = 15.2 cm <= 15.24 cm)
COLSPEC = "L{1.6cm}R{1.0cm}R{1.6cm}R{1.7cm}R{1.6cm}R{1.7cm}R{2.3cm}R{1.4cm}"
OUT_TEX = FIG3 / "session-times-table.tex"
OUT_CSV = FIG3 / "session_times.csv"


def rnd(x):
    """Round half up to the whole minute (None stays None)."""
    return None if x is None else int(math.floor(x + 0.5))


def hhmm_s(s):
    h, m = s.strip().split(":")[:2]
    return int(h) * 3600 + int(m) * 60


def trace_t0_local(xlsx, date):
    """Local seconds-since-midnight of second 0 of the ring recording."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    t0 = None
    for r in wb["Events"].iter_rows(values_only=True):
        if (r and len(r) > 2 and isinstance(r[1], (int, float)) and not isinstance(r[1], bool)
                and isinstance(r[2], str) and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", r[2].strip())):
            h, m, s = (int(v) for v in r[2].strip().split(":"))
            t0 = h * 3600 + m * 60 + s - float(r[1])
            break
    wb.close()
    if t0 is None:
        raise SystemExit(f"{xlsx}: no Auto Clock")
    off_h = LOCAL_TZ.utcoffset(datetime.datetime(date.year, date.month, date.day, 12)).total_seconds() / 3600
    return t0 + off_h * 3600


def log_placement_s(logcsv):
    """First 'Device Application' time of the session log (test eye), local seconds."""
    with open(logcsv, encoding="latin-1", newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].strip().lower().startswith("device application"):
                return hhmm_s(row[1])
    raise SystemExit(f"{logcsv}: no Device Application row")


def erg_groups(plan, sname):
    """Recordings of one session split into pre / intra / post by NPre, NIntra, NPost."""
    rows = plan[plan["Session"] == sname].reset_index(drop=True)
    if rows.empty:
        return None
    npre, nintra, npost = (int(rows.loc[0, c]) for c in ("NPre", "NIntra", "NPost"))
    if npre + nintra + npost != len(rows):
        raise SystemExit(f"{sname}: NPre+NIntra+NPost != number of recordings")
    rec = [dict(wall=hhmm_s(r.WallTime), el=r.Electrode.strip(), label=r.Label,
                date=r.ErgDate) for r in rows.itertuples()]
    return dict(pre=rec[:npre], intra=rec[npre:npre + nintra], post=rec[npre + nintra:])


def ring_at(d, t0, wall_s):
    t = t0 + d["t_s"].to_numpy(float)
    m = np.abs(t - wall_s) <= RING_WIN_S
    T = d["temp_C"].to_numpy(float)[m]
    return float(np.nanmean(T)) if m.any() and np.isfinite(T).any() else None


def build():
    events = json.loads((RING / "events.json").read_text(encoding="utf-8"))
    rows, inputs = [], [RING / "events.json"]
    for pig in PIGS:
        plan_f = ERG / f"erg_plan_Pig{pig}.csv"
        plan = pd.read_csv(plan_f, dtype=str)
        inputs.append(plan_f)
        for s in SESSIONS:
            key = f"Pig{pig}_S{s}"
            if key not in events:
                continue                # session not done yet
            e = events[key]
            ev, nr = e["events"], set(e.get("not_recorded", []))
            g = erg_groups(plan, SESSION_NAME[s])
            date = datetime.date.fromisoformat(g["pre"][0]["date"])
            xlsx = RING / "events_xlsx" / f"{key}_Temperature Events.xlsx"
            t0 = trace_t0_local(xlsx, date)
            # the copied workbook must be the one events.json was read from
            from aim3_invivo_figures import _events
            ev_x = _events(xlsx)["events"]
            for k in ("Placed on Pig", "Cooling Start", "Device Off", "Device Removed"):
                a, b = ev_x.get(k), ev.get(k)
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 0.05):
                    raise SystemExit(f"{key} {k}: events.json {b} != {xlsx.name} {a}; "
                                     "re-run aim3_invivo_figures.py --refresh and re-copy")
            ring_csv = RING / f"{key}.csv"
            d = pd.read_csv(ring_csv)
            inputs += [xlsx, ring_csv]
            on, off, removed = (t0 + ev[k] for k in ("Cooling Start", "Device Off", "Device Removed"))
            placed_src = "events"
            if (pig, s) in LOG_PLACEMENT:
                placed = log_placement_s(LOG_PLACEMENT[(pig, s)])
                inputs.append(LOG_PLACEMENT[(pig, s)])
                placed_src = "log"
            else:
                placed = t0 + ev["Placed on Pig"]
            cooling_logged = "Cooling Start" in nr or "Device Off" in nr
            jet_pre = [r for r in g["pre"] if r["el"] == "Jet"]
            jet_post = [r for r in g["post"] if r["el"] == "Jet"]
            rm_post = [r for r in g["post"] if r["el"] == "RM"]
            during = [(r["wall"] - on) / 60 for r in g["intra"]]
            if any(x < 0 or x > (off - on) / 60 for x in during):
                print(f"  ! {key}: an intra recording falls outside switch-on..switch-off: {during}")
            after = (jet_post[-1]["wall"] - off) / 60 if jet_post else None
            after_T = (None if (not jet_post or cooling_logged)
                       else ring_at(d, t0, jet_post[-1]["wall"]))
            # control-eye lag, where the label gives both times "(T hh:mm / C hh:mm)"
            lags = []
            for r in jet_pre[-1:] + g["intra"] + jet_post[-1:] + rm_post[-1:]:
                m = re.search(r"T (\d{1,2}:\d{2}) / C (\d{1,2}:\d{2})", r["label"])
                if m:
                    lags.append((hhmm_s(m.group(2)) - hhmm_s(m.group(1))) / 60)
            rows.append(dict(
                pig=pig, session=s, exploratory=pig in EXPLORATORY,
                placement_source=placed_src, cooling_from_log=cooling_logged,
                switch_on=str(datetime.timedelta(seconds=round(on))),
                switch_off=str(datetime.timedelta(seconds=round(off))),
                wait_min=(on - placed) / 60, cooling_min=(off - on) / 60,
                removal_min=(removed - off) / 60,
                jet_before_min=(on - jet_pre[-1]["wall"]) / 60 if jet_pre else None,
                jet_during_min="; ".join(f"{x:.2f}" for x in during),
                jet_after_min=after, jet_after_ring_C=after_T,
                rm_after_min=(rm_post[-1]["wall"] - off) / 60 if rm_post else None,
                control_lag_min="; ".join(f"{x:.0f}" for x in lags)))
    return pd.DataFrame(rows), inputs


def tex(df):
    out = [r"\begin{tabular}{" + COLSPEC + "}",
           r"\hline",
           r"\textbf{Animal, session} & \textbf{Wait} & \textbf{Cooling} &",
           r"\textbf{Removal} & \textbf{Jet before} & \textbf{Jet during} &",
           r"\textbf{Jet after} & \textbf{RM after} \\",
           r"\hline"]
    for r in df.itertuples():
        lab = f"{r.pig}" + (r"\textsuperscript{b}" if r.exploratory else "") + f", {r.session}"
        if r.cooling_from_log:
            lab += r"\textsuperscript{a}"
        wait = f"{rnd(r.wait_min)}" + (r"\textsuperscript{c}" if r.placement_source == "log" else "")
        before = "None" if r.jet_before_min is None or pd.isna(r.jet_before_min) else f"{rnd(r.jet_before_min)}"
        during = "; ".join(str(rnd(float(x))) for x in r.jet_during_min.split("; ")) if r.jet_during_min else "None"
        if r.jet_after_min is None or pd.isna(r.jet_after_min):
            after = "None"
        elif r.jet_after_ring_C is None or pd.isna(r.jet_after_ring_C):
            after = f"{rnd(r.jet_after_min)}; --"
        else:
            # corrected for the calibration error, as in the rest of the chapter (2026-11-01)
            off = 0.0 if r.session >= CONTACT_CAL.get(r.pig, 99) else RING_OFFSET_C
            after = f"{rnd(r.jet_after_min)}; {r.jet_after_ring_C + off:.1f}~$^\\circ$C"
        rm = "None" if r.rm_after_min is None or pd.isna(r.rm_after_min) else f"{rnd(r.rm_after_min)}"
        out.append(f"{lab} & {wait} & {rnd(r.cooling_min)} & {rnd(r.removal_min)} & {before} & "
                   f"{during} & {after} & {rm} \\\\")
    out += [r"\hline",
            r"\multicolumn{8}{L{14.8cm}}{\footnotesize \textsuperscript{a}Switch-on, switch-off,"
            r" and removal from the session log; the ring record does not cover the cooling, so"
            r" the ring temperature after switch-off is not available.} \\",
            r"\multicolumn{8}{L{14.8cm}}{\footnotesize \textsuperscript{b}Exploratory animal"
            r" (Section~\ref{subsec:aim3-analysis}).} \\",
            r"\multicolumn{8}{L{14.8cm}}{\footnotesize \textsuperscript{c}Placement from the"
            r" session log; the ring record began with the device already on the eye.} \\",
            r"\end{tabular}"]
    head = ["% GENERATED by code/figures/aim3_session_times_table.py -- do not edit by hand.",
            "% Inputs and hashes: figures/aim3/MANIFEST.json (entry session-times-table.tex)."]
    return "\n".join(head + out) + "\n"


def main():
    df, inputs = build()
    df = df.sort_values(["pig", "session"]).reset_index(drop=True)
    df.round(3).to_csv(OUT_CSV, index=False)
    OUT_TEX.write_text(tex(df), encoding="utf-8", newline="\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.round(2).to_string(index=False))
    print(OUT_TEX.read_text(encoding="utf-8"))
    origins_f = SRC / "origins.json"
    origins_all = json.loads(origins_f.read_text(encoding="utf-8")) if origins_f.exists() else {}
    rel = {str(pathlib.Path(i).relative_to(THESIS)).replace("\\", "/") for i in inputs}
    origins = {k: v for k, v in origins_all.items() if k in rel}
    record(__file__, [OUT_TEX, OUT_CSV], sorted(set(inputs)), origins=origins,
           notes="Table tab:aim3-session-times; animals 1-4 (animal 1 exploratory). Trace times "
                 "for switch-on/off and removal (log times for animal 3 S3), instrument times "
                 "for the ERGs. UPDATE-PIGS56: copy erg_plan/events of animals 5-6 and re-run.",
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
