#!/usr/bin/env python3
"""
Live cooling parameters as EXCEL FORMULAS over the embedded trace.

For each session with a  07 - Cooling / Temperature Processed.csv  this creates a
"Cool <S#>" sheet holding:

  * the FULL-resolution trace (time / current / temp) as values - the temperature
    is a Steinhart-Hart DSP output, so it is embedded as values, but everything
    downstream of it is a live Excel formula;
  * event-boundary cells (Placed on Pig / Cooling Start / Device Off / Device
    Removed) that are EXTERNAL REFERENCES to  Temperature Events.xlsx, taking the
    Manual Time cell when the user filled it in and the Auto Time cell otherwise
    (decided at build time so an empty external ref can't read as 0);
  * the six cooling PARAMETERS as formulas that window the trace on those event
    cells: Min temp, Cooling depth, Total cooling time, Temp SD at target, Time to
    target, Max current. Editing an existing time in the events file moves the
    parameters (and the plot segmentation) on recalc; a Manual Time ADDED or
    CLEARED later needs a rebuild, because which cell is linked is decided at
    build time ("Pigs.ps1 analysis N -SkipTemp -SkipErg"). An event
    with no time at all reads MISSING and its parameters stay blank;
  * an editable Target temp cell (defaults to 4 C, the relay set point) that the
    time-to-target and SD-at-target formulas key off;
  * a decimated, live-segmented probe-temperature plot (device off / on / cooling /
    rewarming / restart), coloured by an Excel formula off the same event cells.

add_cooling_sheets(wb, folder) returns  ({short: {canon_metric: (sheet, cell)}},
{short: trace-geometry})  so the caller (the analysis workbook) can point its
Cooling parameter table at these live cells with same-workbook references, and
can look a wall-clock time up against the embedded trace (used by the ERG sheets
to report the cooled-eye temperature at each ERG recording).
"""
import csv
import re
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter as L
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.marker import Marker
import pig_sources as PS

HDR_FILL = PatternFill("solid", fgColor="1F4E79")
WB = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True)
ITAL = Font(italic=True, color="595959")

# decimated plot steps (must match the segment index the M-column formula emits)
STEP_COLORS = ["808080", "7030A0", "2E75B6", "ED7D31", "C00000"]  # 1..5
STEP_NAMES = ["Device off animal", "Device on animal", "Cooling", "Rewarming", "Restart (device off)"]

# Events-sheet rows are found by the event NAME in column A, not by a fixed row
# number (inserting "Regulation Start" once shifted every later row and broke the
# fixed map). EVENT_ROW is only the fallback for an old sheet.
EVENT_NAME = {"conn": "Device Connected", "placed": "Placed on Pig", "cs": "Cooling Start",
              "reg": "Regulation Start", "do": "Device Off", "rem": "Device Removed"}
EVENT_ROW = {"conn": 2, "placed": 3, "cs": 4, "reg": 5, "do": 6, "rem": 7}
# Written into an event cell that has neither a manual nor an automatic time. A
# blank cell read as 0 in every formula, so "Total cooling time" came out 0 and "Min
# temp" was the first sample; text makes those formulas return blank instead.
MISSING = "MISSING - enter a Manual Time in Temperature Events.xlsx"
# events whose 'Recorded?' cell says NOT RECORDED (taken from the log, outside the
# file), per events file - labelled as such on the Cool sheet
NOT_RECORDED_EVENTS = {}
TARGET_DEFAULT = 4.0   # relay set point (coldest calibration plateau); editable in-sheet

# The DAQ stamps its Clock column in UTC while every surgical log time is local
# (Chicago, CDT = UTC-5 for these July sessions). Checked on all three cooling
# sessions: the Cooling Start auto clock minus 5 h lands exactly on the log's
# "turn on test device + pump" minute (16:45:00->11:45 S1, 16:14:27->11:14 S2,
# 16:28:32->11:28 S3). Used to place a wall-clock ERG time on the trace.
# The offset is taken from the recording date (the _YYMMDD stamp on the raw DAQ
# files in the cooling folder), so sessions after the DST change (CST = UTC-6,
# from 2026-11-01) are placed right; DAQ_UTC_OFFSET_H is only the fallback when
# no dated raw file is found.
LOCAL_TZ = "America/Chicago"
DAQ_UTC_OFFSET_H = 5
_RAW_DATE_RE = re.compile(r"_(\d{6})(?:_\d+)?\.(?:WDH|xlsx)$", re.I)


def _session_utc_offset_h(cool: Path) -> float:
    """Hours to subtract from the DAQ's UTC clock to get Chicago local time on
    the day the session was recorded (5 in CDT, 6 in CST)."""
    from collections import Counter
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dates = Counter(m.group(1) for f in cool.iterdir()
                    if (m := _RAW_DATE_RE.search(f.name)))
    if not dates:
        print(f"  ! {cool}: no dated raw DAQ file - assuming UTC-{DAQ_UTC_OFFSET_H}")
        return DAQ_UTC_OFFSET_H
    d = datetime.strptime(dates.most_common(1)[0][0], "%y%m%d").replace(
        hour=12, tzinfo=ZoneInfo(LOCAL_TZ))
    return -d.utcoffset().total_seconds() / 3600

# canonical metric key -> (pretty label, unit)
METRICS = [
    ("min",      "Min temp (deg C)",             "0.00"),
    ("depth",    "Cooling depth (deg C)",        "0.00"),
    ("total",    "Total cooling time (min)",     "0.00"),
    ("efftotal", "Effective cooling time (min)", "0.00"),
    ("sd",       "Temp SD at target (deg C)",    "0.000"),
    ("ttt",      "Time to regulation (min)",     "0.00"),
    ("maxI",     "Max current (A)",              "0.00"),
]


def _short(name):
    m = re.search(r"session\s*(\d+)", name.lower())
    tag = f"S{m.group(1)}" if m else name
    if "end of life" in name.lower() or "eol" in name.lower():
        tag += " EoL"
    return tag


def _event_refs(evf: Path):
    """Return ({canon: ext-ref formula or None}, [(off,bon)...], {canon: value}).
    For each of the four boundary events (+ conn), link the Manual Time cell
    (col D) if the user filled it, else the Auto Time cell (col B); this bakes
    WHICH source cell at build time (so an empty external ref never reads as 0)
    but the VALUE stays live to edits of that cell. The third return is the
    value each ref resolves to right now - build-time-only, for sanity checks
    that need a number rather than a formula."""
    refs = {k: None for k in EVENT_ROW}
    vals = {k: None for k in EVENT_ROW}
    restarts = []
    if not evf.exists():
        return refs, restarts, vals
    try:
        wb = openpyxl.load_workbook(evf, data_only=True)
    except Exception as e:
        # don't swallow this: every parameter of the session would come out blank or
        # computed from zeros with no message
        raise RuntimeError(f"cannot read {evf} ({e}); close it in Excel / let Box finish "
                           f"syncing, then rebuild")
    if True:
        ws = wb["Events"] if "Events" in wb.sheetnames else wb.worksheets[0]

        def extref(col, row):
            return f"='{evf.parent}\\[{evf.name}]Events'!{col}{row}"

        by_name = {ws.cell(r, 1).value: r for r in range(2, min(ws.max_row, 20) + 1)
                   if isinstance(ws.cell(r, 1).value, str)}
        rows = {k: by_name.get(EVENT_NAME[k], EVENT_ROW[k]) for k in EVENT_ROW}
        for k, rw in rows.items():
            man = ws.cell(rw, 4).value
            auto = ws.cell(rw, 2).value
            st = ws.cell(rw, 6).value          # 'Recorded?' (convert_temperature.py)
            if isinstance(st, str) and st.upper().startswith("NOT RECORDED"):
                NOT_RECORDED_EVENTS.setdefault(str(evf), set()).add(k)
            if isinstance(man, (int, float)):
                refs[k] = extref("D", rw); vals[k] = float(man)
            elif isinstance(man, str) and man.strip():
                # text ("none", "not recorded") = no time. Falling back to the
                # AUTO time would put Pig 3 S3's false probe-handling guesses (Cooling
                # Start 3198.6 s, Device Off 3468 s) into its Cool sheet.
                pass
            elif isinstance(auto, (int, float)):
                refs[k] = extref("B", rw); vals[k] = float(auto)
        for r in range(1, ws.max_row + 1):
            n = ws.cell(r, 1).value
            if isinstance(n, (int, float)):          # a numbered restart row
                off = ws.cell(r, 2).value
                bon = ws.cell(r, 4).value
                if isinstance(off, (int, float)) and isinstance(bon, (int, float)):
                    restarts.append((float(off), float(bon)))
        wb.close()
    # Regulation Start too (2026-10-28): left empty it read as 0, so Pig 3 S3's Cool
    # sheet showed "Time to regulation" -53 min and an effective cooling time
    for k in ("placed", "cs", "do", "rem", "reg"):
        if refs[k] is None:
            refs[k] = MISSING
            print(f"  ! {evf.parent.parent.name}: no time for '{EVENT_NAME[k]}' - its "
                  f"parameters stay blank until a Manual Time is entered")
    return refs, restarts, vals


def _clock_s(txt):
    """'16:28:32' -> seconds since midnight (None if it isn't a clock string)."""
    if not isinstance(txt, str):
        return None
    p = txt.strip().split(":")
    if len(p) != 3:
        return None
    try:
        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    except ValueError:
        return None


def _read_trace(csvf: Path, utc_offset_h: float = DAQ_UTC_OFFSET_H, uncorrected: bool = False):
    """(times, currents, temps, t0_local_s) - t0_local_s is the LOCAL
    seconds-since-midnight of relative time 0, from the file's UTC Clock column
    shifted by utc_offset_h. None if the file carries no usable clock."""
    times, currents, temps = [], [], []
    t0 = None

    def pf(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    with open(csvf, encoding="latin-1", newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd, None) or []
        # uncorrected=True: the reading as calibrated, before the old-method offset
        # (convert_temperature RING_OFFSET_C); falls back to the main column
        ti = next((i for i, h in enumerate(hdr) if h.startswith("Temp uncorrected")), 3) if uncorrected else 3
        for row in rd:
            if not row:
                continue
            t = pf(row[0])
            if t is None:
                continue
            if t0 is None and len(row) > 1:
                utc = _clock_s(row[1])
                if utc is not None:
                    t0 = utc - utc_offset_h * 3600 - t   # local clock at rel t=0
            times.append(t)
            currents.append(pf(row[2]) if len(row) > 2 else None)
            temps.append(pf(row[ti]) if len(row) > ti else None)
    return times, currents, temps, t0


def add_cooling_sheets(wb, folder: Path, max_plot=4000):
    results, traces = {}, {}
    for sess in sorted(folder.glob("Session *")):
        cool = PS.cooling_dir(sess)
        csvf = cool / "Temperature Processed.csv"
        if not csvf.exists():
            continue
        times, currents, temps, t0_local = _read_trace(csvf, _session_utc_offset_h(cool))
        if len(times) < 5:
            continue
        refs, restarts, evals_ = _event_refs(cool / "Temperature Events.xlsx")
        nrst = len(restarts)
        short = _short(sess.name)
        ws = wb.create_sheet(f"Cool {short}"[:31])
        ws.sheet_view.showGridLines = False
        N = len(times)
        LAST = 1 + N                      # last data row (header in row 1)

        # -- full-resolution trace (plain values; temp is a DSP output) ----------
        ws.cell(1, 1, "time (s)").font = WB
        ws.cell(1, 2, "current (A)").font = WB
        ws.cell(1, 3, "temp (deg C)").font = WB
        for c in (1, 2, 3):
            ws.cell(1, c).fill = HDR_FILL
        for i in range(N):
            r = 2 + i
            ws.cell(r, 1, round(times[i], 1))
            if currents[i] is not None:
                ws.cell(r, 2, round(currents[i], 3))
            if temps[i] is not None:
                ws.cell(r, 3, round(temps[i], 3))

        Trng = f"$A$2:$A${LAST}"
        Irng = f"$B$2:$B${LAST}"
        Crng = f"$C$2:$C${LAST}"

        # -- parameters + metrics block, placed right after the plot / restart
        # columns so no empty column band is left to the side (previously fixed at
        # P/Q, which left cols J..O empty when a session had few/no restarts). The
        # block column tracks the restart count: L/M with no restarts, shifting one
        # column right per restart (so the 4-restart session lands back at P/Q). ---
        PC = 12 + nrst                    # param label column (was fixed 16 = P)
        VC = PC + 1                       # param value column (was fixed 17 = Q)
        PL, VL = L(PC), L(VC)             # their letters (VL replaces the old "Q")
        RSC, RBC = VC + 2, VC + 3         # restart off / back-on helper cols (was S/T)
        RSL, RBL = L(RSC), L(RBC)

        def put(row, label, value, fmt=None, font=BOLD):
            ws.cell(row, PC, label).font = font
            c = ws.cell(row, VC, value)
            if fmt:
                c.number_format = fmt
            return f"${VL}${row}"

        nr = NOT_RECORDED_EVENTS.get(str(cool / "Temperature Events.xlsx"), set())

        def lab(text, k):
            return text + ("  - NOT RECORDED (log time)" if k in nr else "")
        placed = put(2, lab("Placed on Pig (s)", "placed"), refs["placed"], "0.0")
        cs = put(3, lab("Cooling Start (s)", "cs"), refs["cs"], "0.0")
        do = put(4, lab("Device Off (s)", "do"), refs["do"], "0.0")
        rem = put(5, lab("Device Removed (s)", "rem"), refs["rem"], "0.0")
        put(6, lab("Device Connected (s)", "conn"), refs["conn"], "0.0")
        target = put(7, "Target temp (deg C, editable)", TARGET_DEFAULT, "0.0")
        # row 11 - rows 8/9/10 are the body-temp and SD-at-target helpers and the
        # metric block starts at 13. Do NOT reuse row 7: it is the target cell, and
        # putting Regulation Start there made every metric read the target instead.
        reg = put(11, "Regulation Start (s)", refs["reg"], "0.0")
        # helper windows
        body = f"${VL}$8"
        ws.cell(8, PC, "Body temp (mean, pre-cool) (deg C)").font = ITAL
        ws.cell(8, VC, f"=IFERROR(SUMPRODUCT(({Trng}>={cs}-90)*({Trng}<={cs})*{Crng})/"
                       f"SUMPRODUCT(({Trng}>={cs}-90)*({Trng}<={cs})*({Crng}<>\"\")),\"\")").number_format = "0.00"
        cond = f"({Trng}>={cs})*({Trng}<={do})*(ABS({Crng}-{target})<=2.5)*({Crng}<>\"\")"
        sd_n = f"${VL}$9"
        ws.cell(9, PC, "SD-at-target n").font = ITAL
        ws.cell(9, VC, f"=SUMPRODUCT({cond})")
        sd_mean = f"${VL}$10"
        ws.cell(10, PC, "SD-at-target mean").font = ITAL
        ws.cell(10, VC, f"=IFERROR(SUMPRODUCT({cond}*{Crng})/{sd_n},\"\")").number_format = "0.000"

        # -- the six live parameters ---------------------------------------------
        metric_cell = {}
        # Effective cooling time = the span the device actually spent REGULATING:
        # from Regulation Start (relay begins cycling) to Device Off, less any time
        # lost to restarts. It used to run from the last restart's back-on, which on
        # Pig 2 S3 collapsed a 2 h run to 1.1 min because a 110 s relay glitch
        # happened 68 s before shutdown.
        lost = (f"-SUMPRODUCT(${RBL}$2:${RBL}${1 + nrst}-${RSL}$2:${RSL}${1 + nrst})"
                if nrst else "")
        formulas = {
            # min/max are windowed on Cooling Start..Device Off like every other
            # parameter. Over the whole trace they pick up handling artifacts from
            # setup/teardown: Pig 2 S1 dipped to 1.14 deg C for 12 s at 17:23 while
            # the probe was being removed, which is not a cooling temperature and
            # dragged Cooling depth with it.
            "min":      f"=IFERROR(_xlfn.AGGREGATE(15,6,{Crng}/(({Trng}>={cs})*({Trng}<={do})*({Crng}<>\"\")),1),\"\")",
            "depth":    f"=IFERROR({body}-${VL}$13,\"\")",
            "total":    f"=IFERROR(({do}-{cs})/60,\"\")",
            "efftotal": f"=IFERROR(({do}-{reg}{lost})/60,\"\")",
            "sd":       f"=IFERROR(SQRT(SUMPRODUCT({cond}*({Crng}-{sd_mean})^2)/{sd_n}),\"\")",
            # cooling start -> the relay beginning to cycle, not a 4 degC crossing
            "ttt":      f"=IFERROR(({reg}-{cs})/60,\"\")",
            "maxI":     f"=IFERROR(_xlfn.AGGREGATE(14,6,{Irng}/(({Trng}>={cs})*({Trng}<={do})*({Irng}<>\"\")),1),\"\")",
        }
        r0 = 13
        for i, (canon, label, fmt) in enumerate(METRICS):
            r = r0 + i
            ws.cell(r, PC, label).font = BOLD
            ws.cell(r, VC, formulas[canon]).number_format = fmt
            metric_cell[canon] = (ws.title, f"{VL}{r}")

        # -- restart intervals (values) in the off/back-on helper cols (referenced
        #    by the restart shading columns and by eff_start). Header them so the
        #    times aren't two unlabelled columns; only present when there were
        #    supply restarts mid-run. --------------------------------------------
        if nrst:
            hs = ws.cell(1, RSC, "Restart: supply off (s)"); hs.font = WB; hs.fill = HDR_FILL
            hb = ws.cell(1, RBC, "Restart: supply back on (s)"); hb.font = WB; hb.fill = HDR_FILL
            ws.column_dimensions[RSL].width = 16
            ws.column_dimensions[RBL].width = 18
        for i, (off, bon) in enumerate(restarts):
            ws.cell(2 + i, RSC, round(off, 1))
            ws.cell(2 + i, RBC, round(bon, 1))

        # -- live-segmented plot columns, next to A/B/C, that REFERENCE column C --
        # (=$C$<row>, not a copy). Each phase is ONE CONTIGUOUS RUN so a scatter
        # chart has no interior gap to interpolate across (which is what drew the
        # stray lines); "Device off" is split into a before-run and an after-run
        # so it never spans the whole plot. Decimated rows (each still a reference
        # to a specific full-resolution A/C cell) keep the chart light.
        # cols: D=time  E=off(pre) F=on G=cooling H=rewarm I=off(post) J..=restart
        headers = [(4, "t (s)", None), (5, "Device off animal", "808080"),
                   (6, "Device on animal", "7030A0"), (7, "Cooling", "2E75B6"),
                   (8, "Rewarming", "ED7D31"), (9, "Device off animal", "808080")]
        for j in range(nrst):
            headers.append((10 + j, "Restart (device off)", "C00000"))
        for col, name, color in headers:
            c = ws.cell(1, col, name); c.font = WB
            c.fill = PatternFill("solid", fgColor=color) if color else HDR_FILL
        step = max(1, N // max_plot)
        dec = list(range(0, N, step))
        # bridge overlap: extend each phase's start back by ~1.5 decimation steps so
        # a phase's first plotted point coincides with the previous phase's last one
        # (adjacent colours touch instead of leaving a gap at the transition).
        diffs = [times[i + 1] - times[i] for i in range(min(len(times) - 1, 3000))]
        dt_med = sorted(diffs)[len(diffs) // 2] if diffs else 0.1
        br = round(1.5 * step * dt_med, 2)
        for row_i, src in enumerate(dec):
            r = 2 + row_i
            sc = 2 + src                      # the full-resolution row this refers to
            t = f"D{r}"
            ok = f"$C${sc}<>\"\""             # skip blank temp (recording gaps) - a
            v = f"$C${sc}"                    # reference to a blank cell returns 0
            ws.cell(r, 4, f"=$A${sc}")
            ws.cell(r, 5, f"=IF(AND({t}<{placed},{ok}),{v},NA())")                          # off pre
            ws.cell(r, 6, f"=IF(AND({t}>={placed}-{br},{t}<{cs},{ok}),{v},NA())")            # on
            ws.cell(r, 7, f"=IF(AND({t}>={cs}-{br},{t}<={do},{ok}),{v},NA())")               # cooling
            ws.cell(r, 8, f"=IF(AND({t}>={do}-{br},{t}<={rem},{ok}),{v},NA())")              # rewarm
            ws.cell(r, 9, f"=IF(AND({t}>={rem}-{br},{ok}),{v},NA())")                        # off post
            for j in range(nrst):                                                           # each restart = its own contiguous line
                ws.cell(r, 10 + j,
                        f"=IF(AND({t}>=${RSL}${2+j},{t}<${RBL}${2+j},{ok}),{v},NA())")
        ndec = len(dec)

        ch = ScatterChart()
        ch.title = f"{sess.name}: probe temperature  (deg C vs s)"
        ch.title.overlay = False
        ch.x_axis.delete = False; ch.y_axis.delete = False
        ch.x_axis.majorGridlines = None
        ch.legend.position = "r"; ch.legend.overlay = False
        ch.x_axis.crosses = "min"; ch.y_axis.crosses = "min"
        ch.y_axis.numFmt = "General"
        ch.width, ch.height = 15, 9
        xref = Reference(ws, min_col=4, min_row=2, max_row=1 + ndec)
        # all phases (and each restart interval) are contiguous line series, so
        # none interpolates across a gap; each restart is a separate contiguous
        # column overlaid in red on the (continuous) cooling line.
        series_cols = [(5, "808080"), (6, "7030A0"), (7, "2E75B6"),
                       (8, "ED7D31"), (9, "808080")]
        series_cols += [(10 + j, "C00000") for j in range(nrst)]
        for col, color in series_cols:
            yref = Reference(ws, min_col=col, min_row=1, max_row=1 + ndec)
            s = Series(yref, xref, title_from_data=True)
            s.marker = Marker(symbol="none")
            s.graphicalProperties.line.solidFill = color
            s.graphicalProperties.line.width = 18000
            s.smooth = False
            ch.series.append(s)
        ws.add_chart(ch, f"{L(VC + 5)}2")

        # explain the plot columns: they are a DECIMATED, phase-split copy of the
        # temperature (col C) that feeds the chart, so they don't line up row-for-
        # row with cols A-C (row r of col D+ points at trace row 2+step*(r-1)).
        last_plot_col = L(9 + nrst)
        note_row = r0 + len(METRICS) + 2      # a couple rows below the metric block
        nc = ws.cell(note_row, PC,
                     f"Cols D-{last_plot_col} are the chart's plot data only: a decimated "
                     f"(every {step}th sample) copy of the temperature in col C, split into "
                     f"one column per phase (blank outside that phase). They intentionally do "
                     f"NOT line up row-for-row with the full-resolution trace in cols A-C - each "
                     f"plotted point references its own source row in col C, and that row's time "
                     f"is shown in col D.")
        nc.font = ITAL
        nc.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="top")
        # merge a wide/tall block that stops just before the chart (at col VC+5)
        ws.merge_cells(start_row=note_row, start_column=PC, end_row=note_row + 5, end_column=VC + 4)
        if nr:
            gc = ws.cell(note_row + 7, PC,
                         "RECORDING GAP: the events marked NOT RECORDED happened outside this recording; "
                         "their times are the bench-sheet clocks. No parameter is computed over an "
                         "unrecorded span (those cells stay blank); Total cooling time is the log's. "
                         "Span by span: Temperature Events.xlsx, sheet Coverage.")
            gc.font = openpyxl.styles.Font(bold=True, color="C00000")
            gc.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="top")
            ws.merge_cells(start_row=note_row + 7, start_column=PC, end_row=note_row + 10, end_column=VC + 4)

        for col in ("A", "B", "C"):
            ws.column_dimensions[col].width = 11
        ws.column_dimensions["D"].width = 9
        ws.column_dimensions[PL].width = 30
        ws.column_dimensions[VL].width = 12
        results[short] = metric_cell
        # geometry another sheet needs to read this trace at a wall-clock time:
        # the time/temp ranges, the probe-on-eye window cells, and the local clock
        # that relative time 0 corresponds to.
        traces[short] = dict(
            sheet=ws.title, trange=f"'{ws.title}'!{Trng}", crange=f"'{ws.title}'!{Crng}",
            placed=f"'{ws.title}'!{placed}", removed=f"'{ws.title}'!{rem}",
            t0_local=t0_local, ev=evals_,
            t_end=times[-1] if times else None)
    return results, traces
