#!/usr/bin/env python3
"""
Extract EVERY individual ERG recording for a session - no averaging across
electrodes or timepoints. Reads the split Pre/Intra/Post files (which hold each
timepoint's per-eye / per-flash waveform) plus their Info sheet (which labels
each recording's electrode / device time / log label), and returns one entry per
timepoint with the segment-0 waveform and a-/b-wave metrics for every
(eye, flash) combination.

Eye map: RE = right = TEST (cooled), LE = left = CONTROL.
Electrode: 'RM' (corneal sensor, baseline/post) or 'Jet' (corneal, during cooling).

extract(session_folder) -> list of dicts sorted by device time:
  {phase, label, electrode, devtime, walltime, order,
   waves: {(eye, flash): np.ndarray on GRID},
   ab:    {(eye, flash): {a_amp,a_t,b_amp,b_t,ba,base} or None},
   ops:   {(eye, flash): {trace, amps, times, sum} or None}   (flashes OP_FLASHES)}
GRID is the common time axis (-20..200 ms @ 0.5 ms).
"""
import re
from array import array
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string as _ci
import extract_erg as E
import reteval_io as RIO

GRID = np.arange(-20, 200.01, 0.5)
FLASHES = ["0.06", "8", "25"]
PHASES = [("Pre", "01 - Pre Experiment"), ("Intra", "02 - Intra Experiment"),
          ("Post", "03 - Post Experiment")]
META_ROWS = 14                 # RETeval metadata rows before the waveform samples
MAX_SCAN_ROWS = 1500           # header (<= ~150 rows) + segment 0 (~430 samples), with margin
RAW_HEADER = "Raw Waveform"
# Since 2026-10-22 I measure the RAW waveform, not the device's Reported one. The raw
# block is the whole ~60 s record of one flash step (flashes at k / Stimulus
# Frequency, k = 0, 1, ...) riding on a large electrode drift (~1 mV/s). Per sweep:
# a straight line fitted to the DETREND_PRE_MS before the flash is subtracted, the
# sweep is cut to GRID, and the sweeps are AVERAGED (the
# "last recording, never average" rule is about repeated recordings, not sweeps).
# Pig 3 S1 check: matches the Reported trace within ~5 uV / 0.5 ms (r 0.98-0.99).
WAVE_SOURCE = "raw"            # "raw" | "reported"
# From 2026-10-22 the ISCEV 2022 band-pass lower corner (0.3 Hz high-pass) is the
# method; the pre-flash line fit agreed within a median 2 uV / 0 ms over 122
# recordings per flash and is kept only as the comparison.
DRIFT_METHOD = "hp03"          # "hp03" (ISCEV <=0.3 Hz high-pass) | "line" (pre-flash line fit)
ALT_DRIFT_METHOD = "line"      # computed alongside, for the method comparison only
HP_HZ = 0.3
REJECT_NOISE_X = 3.0           # artefact rejection: pre-flash noise > 3 x the recording's median
REJECT_B_FRAC = 0.5            # ... or b-wave > 50 % away from the recording's median
DETREND_PRE_MS = 1000.0
PRE_BASE_MS = 20.0             # ISCEV pre-stimulus baseline (>= 20 ms) each sweep is referred to
MIN_PRE_MS = 200.0             # a sweep needs at least this much pre-flash record
REPEAT_SPREAD_MIN = 5          # intra timepoints within this many minutes = "repeats"
# From 2026-10-23 oscillatory potentials are a PRIMARY ERG measure, handled as the
# ISCEV 2022 standard (Robson et al., Doc Ophthalmol 144:165) sets out:
#   stimulus   DA 3 or DA 10 flash, strength stated - ours 8 and 25 cd.s/m2; the 0.06
#              step is rod-driven and has no OPs
#   filter     band-pass 75 Hz to >= 300 Hz (roll-off not standardised; here
#              4th-order Butterworth, zero phase) on the same raw-averaged waveform as
#              the a-/b-wave, before the 75 Hz low-pass that would remove them
#   flashes    only the responses to the 2nd and later flashes are averaged (the
#              step's first flash is left out of the OP average)
#   where      OPs ride the RISING LIMB of the b-wave: wavelets searched from the
#              a-wave trough (not before 10 ms) to OP_AFTER_B_MS past the b-wave
#              peak; "typically three
#              main positive peaks, often followed by a smaller fourth"
#   measure    each wavelet from its preceding trough (ISCEV example method);
#              OP_sum = OP1 + ... + OP4 ("sum of amplitudes of specified peaks");
#              OP_rms = RMS of the filtered trace over the same window (ISCEV's
#              third example) is reported alongside
OP_BAND = (75.0, 300.0)
OP_WIN = (10.0, 70.0)          # fallback window when the a-/b-wave picks are missing
OP_AFTER_B_MS = 5.0
OP_N = 4
OP_FLASHES = ("8", "25")


def _g(row, i):
    return row[i] if i < len(row) else None


def _num(v):
    """float or None - split workbooks can hold CSV-origin numbers as text."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _epochs(t, v, freq, method):
    """[(epoch on GRID, pre-flash noise uV)] for every complete sweep.
    method "line": a line fitted to the pre-flash window is subtracted per sweep.
    method "hp03": the whole record is high-passed (zero-phase Butterworth, order 2,
    corner HP_HZ - the ISCEV 2022 band-pass lower corner, <= 0.3 Hz), then each sweep
    is referred to the mean of its pre-flash window."""
    from scipy.signal import butter, filtfilt
    period = 1000.0 / freq
    if method == "hp03":
        fs = 1000.0 / float(np.median(np.diff(t)))
        b, a = butter(2, HP_HZ, btype="highpass", fs=fs)
        # The record starts only ~1 s before the first flash: with filtfilt's default
        # 9-sample padding the filter rang there and sweep 1 was rejected as "noisy"
        # in 187 of 244 recordings (2026-10-22). 3 s of odd (mirrored) padding.
        v = filtfilt(b, a, v - v[0], padtype="odd", padlen=min(len(v) - 1, int(3.0 * fs)))
    out, k = [], 0
    while k * period + GRID[-1] <= t[-1]:
        tt = t - k * period
        pre_lo = max(-DETREND_PRE_MS, -period + GRID[-1])   # never reach into the last response
        pre = (tt >= pre_lo) & (tt < -2.0)
        if pre.sum() > 10 and (tt[pre][-1] - tt[pre][0]) >= MIN_PRE_MS - 5:
            if method == "line":
                slope, icpt = np.polyfit(tt[pre], v[pre], 1)
                base = slope * tt + icpt
            else:
                # ISCEV: referred to the average PRE-STIMULUS baseline, i.e. the
                # PRE_BASE_MS just before the flash. (Until 2026-10-22 the whole 1-s
                # pre-flash window was used; residual drift then left averages offset
                # by up to ~180 uV at the flash, e.g. Pig 4 S2 test eye.)
                b0 = (tt >= -PRE_BASE_MS) & (tt < 0)
                base = np.full_like(tt, float(np.mean(v[b0])) if b0.any() else float(np.mean(v[pre])))
            # noise = scatter about a line through the pre-flash window, so a slow
            # residual drift (or a filter transient) is not counted as noise
            sl, ic = np.polyfit(tt[pre], v[pre], 1)
            noise = float(np.std(v[pre] - (sl * tt[pre] + ic)))
            m = (tt >= GRID[0] - 1) & (tt <= GRID[-1] + 1)
            # drift = slope left in the pre-flash second after drift removal (uV/s)
            out.append((np.interp(GRID, tt[m], v[m] - base[m]), noise, float(sl) * 1000.0))
        k += 1
    return out


def _reject(eps, flash):
    """Artefact rejection within one recording x eye x flash (ISCEV 2022: 'Artefact
    rejection must be a part of any averaging system'). A sweep is rejected when its
    pre-flash noise exceeds REJECT_NOISE_X x the median of the recording's sweeps, or
    its b-wave amplitude lies more than REJECT_B_FRAC of the median from the median.
    If every sweep would be rejected, all are kept and flagged."""
    ab = [E.ab_metrics(GRID, e, flash=flash) for e, _n, _d in eps]
    noise = np.array([n for _e, n, _d in eps])
    bs = np.array([m["b_amp"] if m else np.nan for m in ab])
    med_n = float(np.median(noise))
    med_b = float(np.nanmedian(bs)) if np.isfinite(bs).any() else np.nan
    keep = []
    for i in range(len(eps)):
        why = []
        if med_n > 0 and noise[i] > REJECT_NOISE_X * med_n:
            why.append(f"noise {noise[i]:.0f} > {REJECT_NOISE_X:g}x median {med_n:.0f} uV")
        if not np.isfinite(bs[i]):
            why.append("no a/b pick")
        elif np.isfinite(med_b) and abs(bs[i] - med_b) > REJECT_B_FRAC * abs(med_b):
            why.append(f"b {bs[i]:.0f} vs median {med_b:.0f} uV")
        keep.append("; ".join(why))
    all_out = all(keep)
    return [dict(ab=ab[i], noise=round(float(noise[i]), 1), drift=round(eps[i][2], 1),
                 kept=(not keep[i]) or all_out,
                 reason=keep[i] + (" (kept: every sweep failed)" if all_out else ""))
            for i in range(len(eps))]


def raw_average(t, v, freq, flash, method=None, keep=None):
    """Mean of the accepted raw sweeps on GRID -> (wave, [per-sweep dicts], OP wave), or
    None when the record cannot be epoched (no frequency / no complete sweep). The OP
    wave is the same mean WITHOUT the step's first flash (ISCEV 2022: only the OPs to
    the second and later flashes are retained or averaged); None if nothing is left.
    keep = sweep numbers (1-based) chosen by hand (Reference/erg_sweep_overrides.csv); they replace the
    automatic rejection for this trace."""
    if not freq or freq <= 0 or len(t) < 100:
        return None
    t = np.frombuffer(t, dtype=float) if isinstance(t, array) else np.asarray(t, float)
    v = np.frombuffer(v, dtype=float) if isinstance(v, array) else np.asarray(v, float)
    eps = _epochs(t, v, freq, method or DRIFT_METHOD)
    if not eps:
        return None
    sw = _reject(eps, flash)
    if keep:
        for i, s in enumerate(sw):
            s["kept"] = (i + 1) in keep
            s["reason"] = "kept by hand (erg_sweep_overrides.csv)" if s["kept"] else "left out by hand (erg_sweep_overrides.csv)"
    kept = np.array([e for (e, _n, _d), s in zip(eps, sw) if s["kept"]])
    later = [e for i, ((e, _n, _d), s) in enumerate(zip(eps, sw)) if s["kept"] and i > 0]
    return kept.mean(axis=0), sw, (np.mean(later, axis=0) if later else None)


def op_metrics(t_ms, uv, per_s=None, ab=None):
    """Oscillatory potentials of one averaged waveform (see OP_* above) ->
    dict(trace, amps, times, sum, rms, window), or None when the trace is not usable.
    ab = the recording's a-/b-wave picks (a_t, b_t in ms) that bound the rising limb;
    without them the fixed OP_WIN is used. per_s defaults to the spacing of t_ms."""
    from scipy.signal import butter, filtfilt, find_peaks
    t_ms, uv = np.asarray(t_ms, float), np.asarray(uv, float)
    if len(uv) < 50 or not np.all(np.isfinite(uv)):
        return None
    if per_s is None:
        per_s = float(np.median(np.diff(t_ms))) / 1e3
    fs = 1.0 / per_s
    b, a = butter(4, [OP_BAND[0], min(OP_BAND[1], 0.45 * fs)], btype="bandpass", fs=fs)
    op = filtfilt(b, a, uv)
    lo_ms, hi_ms = OP_WIN
    if ab and ab.get("a_t") is not None and ab.get("b_t") is not None and ab["b_t"] > ab["a_t"]:
        # never before OP_WIN[0]: an a-wave pick at the window edge (e.g. 2 ms) would
        # otherwise admit the flash artefact as OP1
        lo_ms, hi_ms = max(float(ab["a_t"]), OP_WIN[0]), float(ab["b_t"]) + OP_AFTER_B_MS
    idx = np.where((t_ms >= lo_ms) & (t_ms <= hi_ms))[0]
    if len(idx) < 5:
        return None
    pk, _ = find_peaks(op[idx], height=0.0)          # positive wavelets only
    amps, times = [], []
    for p in pk[:OP_N]:
        j = idx[p]
        lo = idx[0] + (pk[pk < p][-1] if (pk < p).any() else 0)
        amps.append(float(op[j] - op[lo:j + 1].min()))
        times.append(float(t_ms[j]))
    return dict(trace=op, amps=amps, times=times, sum=(float(sum(amps)) if amps else None),
                rms=float(np.sqrt(np.mean(op[idx] ** 2))), window=(lo_ms, hi_ms))


def op_rewindow(op, a_t, b_t):
    """The OPs of an op_metrics() result measured again in the window bounded by other a-/b-wave picks
    (hand-corrected ones): same filtered trace, same peak rule. Returns a new dict."""
    from scipy.signal import find_peaks
    tr = np.asarray(op["trace"], float)
    lo_ms, hi_ms = max(float(a_t), OP_WIN[0]), float(b_t) + OP_AFTER_B_MS
    idx = np.where((GRID >= lo_ms) & (GRID <= hi_ms))[0]
    if len(idx) < 5:
        return op
    pk, _ = find_peaks(tr[idx], height=0.0)
    amps, times = [], []
    for p in pk[:OP_N]:
        j = idx[p]
        lo = idx[0] + (pk[pk < p][-1] if (pk < p).any() else 0)
        amps.append(float(tr[j] - tr[lo:j + 1].min()))
        times.append(float(GRID[j]))
    return dict(op, amps=amps, times=times, sum=(float(sum(amps)) if amps else None),
                rms=float(np.sqrt(np.mean(tr[idx] ** 2))), window=(lo_ms, hi_ms))


def _wallmin(s):
    """'HH:MM' -> minutes past midnight, else None."""
    m = re.match(r"\s*(\d{1,2}):(\d{2})", str(s or ""))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _cool_from_info(ws):
    """Read the 'Cooling window' row of a split-file Info sheet -> (on_min, off_min)
    or (None, None) if this session had no cooling. General: parses the two HH:MM
    times split_erg wrote ('device ON HH:MM -> OFF HH:MM ...')."""
    for r in ws.iter_rows(values_only=True):
        for cell in r:
            if isinstance(cell, str) and "device on" in cell.lower():
                ts = re.findall(r"(\d{1,2}:\d{2})", cell)
                if len(ts) >= 2:
                    return _wallmin(ts[0]), _wallmin(ts[1])
    return None, None


def choose_by_phase(recs):
    """Pick ONE recording per phase (NO averaging): the LAST one taken, by wall
    time, in every phase (Pre, Intra, Post).
    Changed 2026-10-05: the Intra phase used to take the recording nearest
    the cooling-window midpoint; it now takes the last, like Pre and Post.
    Returns {phase: rec}."""
    out = {}
    for ph in ("Pre", "Intra", "Post"):
        cand = [r for r in recs if r["phase"] == ph]
        if not cand:
            continue
        cand.sort(key=lambda r: (_wallmin(r["walltime"]) if _wallmin(r["walltime"]) is not None
                                 else r.get("order", 0)))
        out[ph] = cand[-1]
    return out


# The comparison groups of the thesis ERG rule (since 2026-10-15): recordings are
# grouped by electrode AND phase, and the
# last taken in each group is used. choose_by_phase above is electrode-blind - it
# keeps Jet as "Pre" and RM as "Post" and drops the RM baseline and the Jet post -
# and is kept only for the three-column Metrics sheet that the workbook links.
GROUPS = [("Pre", "RM"), ("Pre", "Jet"), ("Intra", "Jet"), ("Post", "Jet"), ("Post", "RM")]


def group_name(phase, electrode):
    return f"{phase} {electrode}"


def choose_by_group(recs):
    """ONE recording per (phase, electrode) group, the last taken by wall time;
    no averaging. Returns {"Pre RM": rec, ...}; a group with no recording (e.g.
    no Jet before switch-on in pig 1 session 3) is simply absent."""
    out = {}
    for ph, el in GROUPS:
        cand = [r for r in recs if r["phase"] == ph and r["electrode"] == el]
        if not cand:
            continue
        cand.sort(key=lambda r: (_wallmin(r["walltime"]) if _wallmin(r["walltime"]) is not None
                                 else r.get("order", 0)))
        out[group_name(ph, el)] = cand[-1]
    return out


def _flash_key(f):
    try:
        return f"{float(f):g}"
    except (TypeError, ValueError):
        return str(f)


SWEEP_FILE = "erg_sweep_overrides.csv"


def load_sweep_overrides(session_folder):
    """{(device time, eye, flash): set of sweep numbers to keep} for one session, from
    <Pig N>/Reference/erg_sweep_overrides.csv (columns Session, DeviceTime, Eye, Flash, Keep, Note;
    Keep = sweep numbers separated by ';'). Sweeps chosen by hand where the automatic rejection
    kept an artefact or dropped a clean sweep (2026-11-12)."""
    import csv
    from pathlib import Path
    p = Path(session_folder).parent / "Reference" / SWEEP_FILE
    out = {}
    if not p.exists():
        return out
    with open(p, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["Session"].strip() != Path(session_folder).name:
                continue
            keep = {int(x) for x in str(r["Keep"]).replace(",", ";").split(";") if x.strip()}
            out[(str(r["DeviceTime"]).strip(), r["Eye"].strip().upper(), _flash_key(r["Flash"]))] = keep
    return out


def extract(session_folder):
    erg = RIO.erg_folder(session_folder)      # '06 - ...' / '05 - ...' at End of Life
    if erg is None:
        return []
    sweep_ov = load_sweep_overrides(session_folder)
    recs = []
    for phase, sub in PHASES:
        files = RIO.phase_files(erg, phase, sub, session_folder)
        if not files:
            continue
        wb = load_workbook(files[0], read_only=True, data_only=True)
        if "Info" not in wb.sheetnames:
            wb.close(); continue
        # Info: out-col letters -> (eye, flash, electrode, devtime, walltime, label)
        cols = []
        for r in wb["Info"].iter_rows(values_only=True):
            eye, elec, flash = _g(r, 1), _g(r, 3), _g(r, 4)
            outc = _g(r, 0)
            if eye in ("RightEye", "LeftEye") and flash is not None and outc and "/" in str(outc):
                cols.append((str(outc), "RE" if eye == "RightEye" else "LE",
                             _flash_key(flash), str(_g(r, 3)), str(_g(r, 5)),
                             str(_g(r, 6)), str(_g(r, 7))))
        cool_on, cool_off = _cool_from_info(wb["Info"])
        data_sheet = next(s for s in wb.sheetnames if s != "Info")
        # Head + Reported Waveform kept as rows (<= MAX_SCAN_ROWS); the ~117k-row Raw
        # Waveform block is streamed into one float array per needed column (memory).
        used_ms = sorted({_ci(c[0].split("/")[0]) - 1 for c in cols})
        rows, raw = [], {mi: (array("d"), array("d")) for mi in used_ms}
        in_raw = False
        for r in wb[data_sheet].iter_rows(values_only=True):
            a0 = r[0] if r else None
            if isinstance(a0, str) and a0.strip():
                in_raw = a0.strip() == RAW_HEADER
                if not in_raw and raw[used_ms[0]][0]:
                    break                                   # past Raw (Pupil Waveform ...)
                if len(rows) < MAX_SCAN_ROWS:
                    rows.append(r)
                continue
            if in_raw:
                for mi in used_ms:
                    t, v = _num(_g(r, mi)), _num(_g(r, mi + 1))
                    if t is not None and v is not None:
                        raw[mi][0].append(t); raw[mi][1].append(v)
            elif len(rows) < MAX_SCAN_ROWS:
                rows.append(r)
        wb.close()
        freq_row = next((r for r in rows if r and r[0] == "Stimulus Frequency"), None)
        # samples start after the "Reported Waveform" header, which moves down when
        # the export carries extra metadata fields (2026-10-22, Pigs 3-4 / re-exports)
        if not any(r and r[0] == RIO.WAVEFORM_HEADER for r in rows):
            raise ValueError(f"{files[0].name}: no '{RIO.WAVEFORM_HEADER}' row in the "
                             f"first {MAX_SCAN_ROWS} rows")
        # The richer exports also carry the RETeval's own cursor values: not read (2026-11-09).
        rows = rows[RIO.waveform_start(rows):]

        # group Info columns by timepoint (label + device time + electrode).
        # RULE: when a flash step was repeated within a recording, the LAST waveform
        # exported is used (Info order = export order); `repeats` records which.
        groups, repeats = {}, {}
        for outc, eye, flash, elec, dev, wall, lab in cols:
            g = groups.setdefault((lab, dev, elec, wall), {})
            if (eye, flash) in g:
                repeats.setdefault((lab, dev, elec, wall), []).append((eye, flash))
            g[(eye, flash)] = outc

        for (lab, dev, elec, wall), efmap in groups.items():
            # 2026-11-09: only our own values are used. The RETeval's reported waveform and its cursor
            # values are not read for any purpose (no fallback, no reference, no comparison); a trace
            # without a raw record is left out.
            waves, sweeps, alt, op_waves = {}, {}, {}, {}
            for (eye, flash), outc in efmap.items():
                a, b = outc.split("/")
                mi, ui = _ci(a) - 1, _ci(b) - 1
                freq = _num(_g(freq_row, mi)) if freq_row else None
                rw = raw_average(raw[mi][0], raw[mi][1], freq, flash,
                                 keep=sweep_ov.get((str(dev).strip(), eye, flash)))
                if rw is None:
                    continue
                waves[(eye, flash)], sweeps[(eye, flash)], op_waves[(eye, flash)] = rw
                ra = raw_average(raw[mi][0], raw[mi][1], freq, flash, method=ALT_DRIFT_METHOD,
                                 keep=sweep_ov.get((str(dev).strip(), eye, flash)))
                if ra is not None:
                    alt[(eye, flash)] = dict(ab=E.ab_metrics(GRID, ra[0], flash=flash),
                                             n_kept=sum(s["kept"] for s in ra[1]))
            ab = {k: E.ab_metrics(GRID, w, flash=k[1]) for k, w in waves.items()}
            # the 75 Hz low-pass waveform that ab_metrics
            # measures, so the analysis workbook embeds and reads the same trace.
            waves_f = {k: E.lowpass(GRID, w) for k, w in waves.items()}
            # OPs from the average without the first flash (the full average when the
            # export has no raw record), searched on the b-wave's rising limb
            ops = {k: op_metrics(GRID, op_waves.get(k) if op_waves.get(k) is not None else w,
                                 ab=ab.get(k))
                   for k, w in waves.items() if k[1] in OP_FLASHES}
            recs.append(dict(phase=phase, label=lab, electrode=elec, devtime=dev,
                             walltime=wall, waves=waves, waves_f=waves_f, ab=ab, ops=ops,
                             sweeps=sweeps, source=WAVE_SOURCE,
                             drift=DRIFT_METHOD, alt=alt,
                             repeats=repeats.get((lab, dev, elec, wall), []),
                             cool_on=cool_on, cool_off=cool_off))
    recs.sort(key=lambda r: (("Pre", "Intra", "Post").index(r["phase"]), r["devtime"]))
    for i, r in enumerate(recs, 1):
        r["order"] = i
    return recs
