#!/usr/bin/env python3
"""
Extract ERG summary metrics + averaged waveforms from the split Pre/Intra/Post
ERG files in a session's  06 - ERG Recording  folder, and write

    06 - ERG Recording / ERG Summary.xlsx

with two sheets:
  * "Metrics"   - a-wave / b-wave amplitude & implicit time (and b/a ratio) for
    every eye (RE=Test, LE=Control) x flash (0.06/8/25) x phase (Pre/Intra/Post),
    from ONE recording per phase, the last taken (no averaging across recordings;
    erg_all.choose_by_phase). Laid out metric-rows x phase-cols
    so the master can link it. (These are the "extrapolated" parameters.)
  * "Waveforms" - the device-averaged reported waveform (segment 0, -20..+200 ms)
    per eye x flash x phase, time column + one uV column each, for plotting.

Each RETeval column holds segment 0 (the averaged reported waveform) followed by
the raw continuous trace; we use segment 0 (up to the point the time axis resets).

Re-run after (re-)splitting the ERG files (split_erg.py). Normally launched by
 Pigs.ps1 erg  - and automatically by  Pigs.ps1 analysis  unless -SkipErg.

Usage:
    python extract_erg.py [--root "<Live Pigs path>"] [pig numbers ...]
"""
import argparse, re, sys, warnings
from pathlib import Path
import numpy as np

# nanmean over an all-NaN phase/eye/flash slice is expected (missing recording);
# it yields NaN, which we handle. Silence the noisy RuntimeWarning so launchers
# that treat any stderr output as an error don't trip.
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import reteval_io as RIO

# the study root is the folder holding _Tools, so renaming the study folder
# (e.g. "Live Pigs" -> "Live Pigs Safety Study") cannot break the default
DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)
HDR = PatternFill("solid", fgColor="1F4E79")
WB = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True)
CEN = Alignment(horizontal="center")

PHASES = [("Pre", "01 - Pre Experiment"), ("Intra", "02 - Intra Experiment"),
          ("Post", "03 - Post Experiment")]
EYES = [("RightEye", "Test", "RE"), ("LeftEye", "Control", "LE")]
FLASHES = [0.06, 8, 25]
META_ROWS = 14
WF_KEEP = 460          # rows to scan for the averaged segment (segment 0 ~430 samples)

# ------------------------------------------------------- a-/b-wave detection
# Picking the plain extreme of a wide window (the original 3-45 / 12-150 ms
# global min / max) fails in two ways on these recordings:
#   * a slow post-response drift outgrows the real b-wave, so the pick lands at
#     60-150 ms (S2 intra #4 at 25 cd.s/m2 picked 118 ms; S1 pre RM picked 66.5);
#   * on a small, noisy response a later noise dip is deeper than the true
#     a-wave (S3 intra #4 control picked 34.5 ms instead of ~8).
# So the search is (a) restricted to a physiological window per flash level and
# (b) resolved with a separation rule: take the largest extreme UNLESS an earlier
# one is nearly as large AND far enough away to be a different component. The
# distance matters because the oscillatory potentials that ride on the b-wave sit
# only ~5-13 ms apart - an earlier peak that close is an OP and must not win -
# whereas a slow drift sits much further off. SEP_MS = 15 ms is the gap that
# keeps every clean trace on its original pick while rejecting both drift cases.
DT = 0.5                                    # ms between samples on the common grid
#            flash : (a window,      b window,      a smooth ms, b smooth ms)
AB_CFG = {"0.06": ((2.0, 35.0), (12.0, 110.0), 3.0, 7.0),      # rod-driven: slow
          "8":    ((2.0, 26.0), (8.0, 50.0),   2.5, 3.0),
          "25":   ((2.0, 26.0), (8.0, 50.0),   2.5, 3.0)}
AB_DEFAULT = AB_CFG["25"]
NEAR = 0.85            # "nearly as large" as the window extreme
SEP_MS = 15.0          # ... and this far earlier to count as a separate component
A_WIN, B_WIN = AB_DEFAULT[0], AB_DEFAULT[1]     # back-compat for older callers


def flash_key(f):
    try:
        return f"{float(f):g}"
    except (TypeError, ValueError):
        return str(f)


def _smooth(v, ms):
    """Light Savitzky-Golay pass used only to FIND the peak; every amplitude is
    then read back off the raw trace."""
    from scipy.signal import savgol_filter
    n = max(3, int(round(ms / DT)) | 1)
    if v.size < n + 2:
        return v
    good = np.isfinite(v)
    if not good.all():
        if good.sum() < 3:
            return v
        v = np.interp(np.arange(v.size), np.flatnonzero(good), v[good])
    return savgol_filter(v, n, 2)


def _pick(sig, lo, hi):
    """Index of the wave in sig[lo:hi]: the maximum, unless an earlier local
    maximum reaches NEAR x it and lies more than SEP_MS before it."""
    from scipy.signal import find_peaks
    seg = sig[lo:hi + 1]
    if seg.size < 3 or not np.isfinite(seg).any():
        return None
    gi = int(np.nanargmax(seg))
    top, floor_ = float(np.nanmax(seg)), float(np.nanmin(seg))
    rng = top - floor_
    if not np.isfinite(rng) or rng <= 0:
        return lo + gi
    idx, _ = find_peaks(seg, prominence=0.05 * rng)
    sep = int(round(SEP_MS / DT))
    for i in idx:
        if i < gi - sep and (seg[i] - floor_) >= NEAR * rng:
            return lo + int(i)
    return lo + gi


def _snap(v, i, sign, span):
    """Move a pick made on the smoothed trace onto the raw trace's own extremum
    nearby, so the reported time and amplitude are a real recorded sample."""
    k = int(round(span / DT))
    lo, hi = max(0, i - k), min(v.size - 1, i + k)
    seg = sign * v[lo:hi + 1]
    if not np.isfinite(seg).any():
        return i
    return lo + int(np.nanargmax(seg))


def seg0(times, volts):
    """Return (t, v) for segment 0 only (until the time axis first resets)."""
    t, v = [], []
    for tt, vv in zip(times, volts):
        if not isinstance(tt, (int, float)):
            continue
        if t and tt < t[-1] - 1:      # time went backwards => raw trace starts
            break
        t.append(tt); v.append(vv if isinstance(vv, (int, float)) else np.nan)
    return np.array(t, float), np.array(v, float)


LOWPASS_HZ = 75.0      # zero-phase low-pass applied before the waves are measured
LOWPASS_ORDER = 4


def lowpass(t, v, cutoff=LOWPASS_HZ, order=LOWPASS_ORDER):
    """Zero-phase Butterworth low-pass of an averaged waveform (default 75 Hz,
    fourth order, forward-backward). It removes the oscillatory potentials
    (about 75-300 Hz) that ride on the b-wave, so the b-wave peak is a single
    maximum. Samples outside the recorded span (NaN) stay NaN."""
    from scipy.signal import butter, filtfilt
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    good = np.flatnonzero(np.isfinite(v))
    if good.size < 30:
        return out
    lo, hi = int(good[0]), int(good[-1])
    seg = v[lo:hi + 1].copy()
    bad = ~np.isfinite(seg)
    if bad.any():                               # isolated gaps: bridge linearly
        idx = np.arange(seg.size)
        seg[bad] = np.interp(idx[bad], idx[~bad], seg[~bad])
    fs = 1000.0 / float(np.nanmedian(np.diff(t)))    # samples per second
    bb, aa = butter(order, cutoff, btype="low", fs=fs)
    out[lo:hi + 1] = filtfilt(bb, aa, seg)
    return out


def ab_metrics(t, v, flash="25"):
    """a-/b-wave amplitude + implicit time from an averaged waveform.

    METHOD (changed 2026-10-05): the waveform is low-pass filtered at 75 Hz
    (zero phase, see lowpass) and everything is read from the FILTERED waveform:
      a-wave = the minimum inside the a window, amplitude from the pre-stimulus
               baseline;
      b-wave = the maximum after the a-wave trough inside the b window,
               amplitude from the trough.
    The windows are those of AB_CFG (per flash strength). The earlier method
    (Savitzky-Golay smoothing to find the peak, value read from the raw trace,
    with the NEAR / SEP_MS separation rule) is kept in
    extract_erg.py.bak-savgol-20261005.
    Returns a dict (with 'a_i'/'b_i', the sample indices, and 'ok', False when
    the pick landed on the edge of its window and so is probably not a real
    wave) or None."""
    if v is None or t.size < 20 or np.all(np.isnan(v)):
        return None
    a_win, b_win, _sa, _sb = AB_CFG.get(flash_key(flash), AB_DEFAULT)
    f = lowpass(t, v)
    if not np.isfinite(f).any():
        return None
    pre = f[t < 0]
    base = float(np.nanmean(pre)) if np.isfinite(pre).any() else float(np.nanmean(f[np.isfinite(f)][:5]))
    ai0, ai1 = int(np.searchsorted(t, a_win[0])), int(np.searchsorted(t, a_win[1]))
    seg = f[ai0:ai1 + 1]
    if seg.size < 3 or not np.isfinite(seg).any():
        return None
    ia = ai0 + int(np.nanargmin(seg))
    bi0 = max(ia + 1, int(np.searchsorted(t, b_win[0])))
    bi1 = int(np.searchsorted(t, b_win[1]))
    if bi1 <= bi0:
        return None
    seg = f[bi0:bi1 + 1]
    if seg.size < 3 or not np.isfinite(seg).any():
        return None
    ib = bi0 + int(np.nanargmax(seg))
    a_amp, b_amp = float(f[ia] - base), float(f[ib] - f[ia])
    ba = (b_amp / abs(a_amp)) if abs(a_amp) > 1e-6 else np.nan
    # a pick sitting on a window edge means the real wave is probably outside it
    ok = bool(ai0 + 1 < ia < ai1 - 1 and bi0 + 1 < ib < bi1 - 1)
    return dict(base=base, a_amp=a_amp, a_t=float(t[ia]), b_amp=b_amp,
                b_t=float(t[ib]), ba=ba, a_i=int(ia), b_i=int(ib), ok=ok)


def read_phase(fp):
    """From a split phase file, return {(eye,flash): [(t,v averaged-waveform), ...per timepoint]}."""
    wb = openpyxl.load_workbook(fp, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[-1]]        # raw data sheet (last)
    rows = list(ws.iter_rows(min_row=1, max_row=200 + WF_KEEP, values_only=True))
    wb.close()
    import reteval_io as RIO
    s0 = RIO.waveform_start(rows)      # header moves down when extra fields were exported
    ncol = max(len(r) for r in rows[:META_ROWS])
    def meta(rn, c): return rows[rn - 1][c - 1] if c - 1 < len(rows[rn - 1]) else None
    out = {}
    for c in range(3, ncol + 1):
        eye = meta(4, c); flash = meta(7, c)
        if eye not in ("RightEye", "LeftEye") or not isinstance(flash, (int, float)):
            continue
        times = [rows[r][c - 1] if c - 1 < len(rows[r]) else None for r in range(s0, len(rows))]
        volts = [rows[r][c] if c < len(rows[r]) else None for r in range(s0, len(rows))]
        t, v = seg0(times, volts)
        out.setdefault((eye, round(float(flash), 2)), []).append((t, v))
    return out


def resample(t, v, grid):
    return np.interp(grid, t, v, left=np.nan, right=np.nan)


def build_summary(session: Path):
    """Write ERG Summary.xlsx for a session. NO AVERAGING: one recording is
    SELECTED per phase (erg_all.choose_by_phase), the last taken in every phase.
    The Metrics and Waveforms sheets show that single recording."""
    import erg_all as EA                       # lazy: erg_all imports this module
    recs = EA.extract(session)
    if not recs:
        return False
    chosen = EA.choose_by_phase(recs)          # {phase: representative rec}
    grid = EA.GRID
    metrics, wave = {}, {}
    for (eye, role, ab) in EYES:               # eye = 'RightEye'/'LeftEye'
        eshort = "RE" if eye == "RightEye" else "LE"
        for fl in FLASHES:
            key = (eshort, flash_key(fl))      # erg_all keys: ('RE','8') etc.
            for ph, _ in PHASES:
                rec = chosen.get(ph)
                if rec is None:
                    continue
                w = rec["waves"].get(key)
                if w is None:
                    continue
                wave[(eye, fl, ph)] = w
                m = rec["ab"].get(key)
                if m:
                    metrics[(eye, fl, ph)] = m

    erg_folder = RIO.erg_folder(session)
    if erg_folder is None:
        return False
    out = openpyxl.Workbook()
    _write_metrics(out.active, metrics, eol=RIO.is_eol(session))
    _write_waveforms(out.create_sheet("Waveforms"), grid, wave)
    groups = EA.choose_by_group(recs)
    _write_group_metrics(out.create_sheet("Metrics by electrode"), groups)
    _write_quality(out.create_sheet("Quality"), chosen, groups)
    path = erg_folder / "ERG Summary.xlsx"
    out.save(path)
    return path


METRIC_ROWS = [("a-wave amp (uV)", "a_amp"), ("a-wave time (ms)", "a_t"),
               ("b-wave amp (uV)", "b_amp"), ("b-wave time (ms)", "b_t"),
               ("b/a ratio", "ba")]


def _write_metrics(ws, metrics, eol=False):
    ws.title = "Metrics"
    ws.cell(1, 1, "ERG summary metrics").font = Font(bold=True, size=13)
    ws.cell(2, 1, "Eye = RightEye (TEST) / LeftEye (CONTROL). One recording per phase: the last taken (no averaging across recordings).").font = Font(italic=True, color="808080")
    hdr = ["Metric", "End of Life" if eol else "Pre", "Intra", "Post"]   # EoL: one time point
    for c, h in enumerate(hdr, 1):
        cell = ws.cell(4, c, h); cell.fill = HDR; cell.font = WB; cell.alignment = CEN
    r = 5
    for (eye, role, ab) in EYES:
        for fl in FLASHES:
            for label, key in METRIC_ROWS:
                ws.cell(r, 1, f"{ab} {fl} {label}").font = BOLD
                for c, ph in enumerate(["Pre", "Intra", "Post"], 2):
                    m = metrics.get((eye, fl, ph))
                    val = None if m is None else round(float(m[key]), 3)
                    ws.cell(r, c, val).alignment = CEN
                r += 1
    ws.column_dimensions["A"].width = 26
    for col in "BCD":
        ws.column_dimensions[col].width = 12
    ws.freeze_panes = "B5"


def _write_quality(ws, by_phase, by_group):
    """Which recording each value comes from (electrode, device/wall time, log
    label) and whether the detector trusted its picks. The Metrics sheets carry
    neither, so a value from an RM recording could not be told from a Jet one, and
    a pick on the edge of its search window (ok = False) was written like any other.
    The Metrics sheet itself is unchanged: the analysis
    workbook reads it at fixed rows."""
    hdr = ["Selection", "Column", "Electrode", "Device time", "Wall time", "Recording",
           "Eye", "Flash", "a-wave time (ms)", "b-wave time (ms)", "a-wave amp (uV)", "ok",
           "Check"]
    ws.cell(1, 1, "ERG summary: source recording and detector check of every value").font = Font(bold=True, size=13)
    ws.cell(2, 1, "ok = False: a pick sat on the edge of its search window. Check: a-wave amplitude "
                  "zero or positive (no a-wave found).").font = Font(italic=True, color="808080")
    for c, h in enumerate(hdr, 1):
        cell = ws.cell(4, c, h); cell.fill = HDR; cell.font = WB; cell.alignment = CEN
    r = 5
    for sel, table in (("Metrics (one per phase)", by_phase),
                       ("Metrics by electrode", by_group)):
        for colname, rec in table.items():
            if rec is None:
                continue
            for (eye, role, ab) in EYES:
                eshort = "RE" if eye == "RightEye" else "LE"
                for fl in FLASHES:
                    m = rec["ab"].get((eshort, flash_key(fl)))
                    if not m:
                        continue
                    check = "a-wave amp >= 0" if float(m["a_amp"]) >= 0 else ""
                    vals = [sel, colname, rec.get("electrode"), rec.get("devtime"),
                            rec.get("walltime"), rec.get("label"), ab, fl,
                            round(float(m["a_t"]), 1), round(float(m["b_t"]), 1),
                            round(float(m["a_amp"]), 2), bool(m.get("ok", True)), check or None]
                    for c, v in enumerate(vals, 1):
                        ws.cell(r, c, v)
                    if not m.get("ok", True) or check:
                        for c in range(1, len(vals) + 1):
                            ws.cell(r, c).font = Font(color="BF8F00", bold=True)
                    r += 1
    for col, wd in zip("ABCDEFGHIJKLM", (22, 12, 10, 11, 10, 40, 8, 7, 15, 15, 14, 7, 16)):
        ws.column_dimensions[col].width = wd
    ws.freeze_panes = "A5"


def _write_group_metrics(ws, chosen):
    """The thesis rule: one recording per electrode-and-phase group, the last
    taken (erg_all.choose_by_group). Same rows as Metrics, five group columns,
    plus which recording each column is and any repeated flash steps in it."""
    import erg_all as EA
    names = [EA.group_name(ph, el) for ph, el in EA.GROUPS]
    ws.cell(1, 1, "ERG metrics by electrode and phase").font = Font(bold=True, size=13)
    ws.cell(2, 1, "One recording per group, the last taken; no averaging. When a flash step "
                  "was repeated within a recording, the last waveform exported is used.").font = Font(italic=True, color="808080")
    for c, h in enumerate(["Metric"] + names, 1):
        cell = ws.cell(4, c, h); cell.fill = HDR; cell.font = WB; cell.alignment = CEN
    for c, n in enumerate(names, 2):
        rec = chosen.get(n)
        ws.cell(5, c, f"{rec['walltime']} {rec['label']}" if rec else "none").alignment = CEN
        rep = ", ".join(f"{e} {f}" for e, f in rec["repeats"]) if rec and rec.get("repeats") else ""
        ws.cell(6, c, rep or None).alignment = CEN
    ws.cell(5, 1, "Recording used").font = BOLD
    ws.cell(6, 1, "Repeated flash steps").font = BOLD
    r = 7
    for (eye, role, ab) in EYES:
        eshort = "RE" if eye == "RightEye" else "LE"
        for fl in FLASHES:
            for label, key in METRIC_ROWS:
                ws.cell(r, 1, f"{ab} {fl} {label}").font = BOLD
                for c, n in enumerate(names, 2):
                    rec = chosen.get(n)
                    m = rec["ab"].get((eshort, flash_key(fl))) if rec else None
                    ws.cell(r, c, None if not m else round(float(m[key]), 3)).alignment = CEN
                r += 1
    ws.column_dimensions["A"].width = 26
    for c in range(2, 2 + len(names)):
        ws.column_dimensions[openpyxl.utils.get_column_letter(c)].width = 22
    ws.freeze_panes = "B7"


def _write_waveforms(ws, grid, wave):
    ws.cell(1, 1, "Averaged reported waveforms (segment 0, -20..200 ms)").font = Font(bold=True, size=12)
    ws.cell(2, 1, "time (ms)").font = BOLD
    for i, t in enumerate(grid, 3):
        ws.cell(i, 1, round(float(t), 2))
    col = 2
    for (eye, role, ab) in EYES:
        for fl in FLASHES:
            for ph, _ in PHASES:
                key = (eye, fl, ph)
                if key not in wave:
                    continue
                ws.cell(2, col, f"{ab} {fl} {ph}").font = BOLD
                w = wave[key]
                for i, val in enumerate(w, 3):
                    ws.cell(i, col, None if np.isnan(val) else round(float(val), 3))
                col += 1
    ws.column_dimensions["A"].width = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("pigs", nargs="*", type=int)
    a = ap.parse_args()
    root = Path(a.root)
    errors = 0
    for p in (a.pigs or range(1, 7)):
        pig = root / f"Pig {p}"
        if not pig.exists():
            continue
        for sess in sorted(pig.glob("Session *")):
            if RIO.erg_folder(sess) is None:
                continue
            try:
                res = build_summary(sess)
            except Exception as e:
                # one bad session no longer aborts the rest, and the run reports
                # failure so Pigs.ps1 stops before building on stale summaries
                print(f"Pig {p} / {sess.name}: ERROR {e}"); errors += 1; continue
            if res:
                print(f"Pig {p} / {sess.name}: wrote {res.name}")
            else:
                print(f"Pig {p} / {sess.name}: no ERG data")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
