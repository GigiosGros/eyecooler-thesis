#!/usr/bin/env python3
"""
Cross-animal ERG analysis for the thesis (Ch. 4, Data Analysis).

Reads every pig's split ERG files through erg_all (one recording per phase x
electrode, the LAST one taken - never averaged), the hand-corrected latencies in
<Pig N>/Reference/erg_latency_overrides.csv, and the core temperature in each
session's Systemic 'Monitoring' sheet, and writes, to <study>/ERG Cohort Analysis/:

  ERG Cohort Analysis.xlsx   sheets Readme, Recordings, Effects, Cooling summary,
                             Restoration, Between sessions
  *.csv                      the same tables, one file each
  fig_*.png                  every individual waveform and phase overlays, flashes 25, 8
                             and 0.06; oscillatory-potential overlays (fig_overlay_ops_*),
                             flashes 8 and 25
  MANIFEST.json              inputs (size, mtime), code version, outputs

Definitions (thesis Ch. 4 Data Analysis). For an eye, d = value(later) - value(earlier).
  Cooling effect   (Intra Jet - Pre Jet)  test eye minus control eye  = dT - dC
  Jet restoration  (Post Jet  - Pre Jet)  dT - dC
  RM restoration   (Post RM   - Pre RM)   dT - dC
  Rewarming        (Post Jet  - Intra Jet) dT - dC
  Between sessions  Pre Jet of session k - Pre Jet of session 1 (PRIMARY) and
                    Pre RM of session k  - Pre RM of session 1 (CONFIRMATORY),
                    per eye and T - C (2026-11-04: Jet primary, RM confirmatory)
"Taken together": a session shows cooling when, for the b-wave, the implicit time
effect is > 0 AND the amplitude effect is < 0.
Per cooling duration: the mean of the animals' effects, its t-based 95 % CI, and two
paired t tests across animals - test vs control (dT against dC) and before vs after
(test eye, later against earlier).
2026-11-04 Holm and effect sizes, as in Chapter 2: p_holm_test_vs_control = Holm across the three
cooling durations within Set x Comparison x Flash x Measure (interpreted: primary set, flash 25,
b_t and b_amp, Cooling); g_paired = Hedges-corrected d_z with an approximate 95 % CI; CI90 of
the mean test-minus-control (largest plausible difference: Jet restoration primary, RM restoration
confirmatory).
Restored: |dT - dC| <= 2 x SD of the control eye's own change dC over the same
interval, the SD pooled over every pig and session (per electrode, flash, measure).
Waveform: the RAW record of each flash step (2026-10-22), drift removed by a
zero-phase 0.3 Hz high-pass (ISCEV 2022 lower corner), artefact-rejected sweeps averaged, 75 Hz low-pass; the
device's Reported trace and cursor values are not used (2026-11-09: all values are ours).
Oscillatory potentials (2026-10-23: primary, with a- and b-wave), as the ISCEV 2022
standard sets out: the raw average WITHOUT the first flash of the step, band-passed
75-300 Hz; wavelets on the b-wave's rising limb (a-wave trough, not before 10 ms, to
the b-wave peak + 5 ms), each from its preceding trough; measure = OP_sum (OP1-OP4),
with the RMS of the filtered window alongside. Flashes 8 and 25 only - the 0.06 step
is a rod response without OPs. Same comparisons, statistics and restoration rule as
the b-wave (measure "op_amp"). Every measure is analysed at all three flashes.
Amplitudes in uV (a-wave as a positive magnitude), implicit times in ms,
temperatures in degC. A session without a Jet baseline (Pig 1 S3) enters only the
RM comparisons.
Animal sets (2026-10-28): Pig 1 is EXPLORATORY - the method changed a lot after
it - and Pigs 2-6 are the PRIMARY set. Every row carries Animal_set; the Cooling
summary is given for the primary set and again with Pig 1 (Set column, primary rows
first); the restoration limit and the eye-check range are pooled over the primary
set, with the verdict including Pig 1 alongside. Figures draw Pig 1 in grey,
labelled exploratory, and leave it out of every mean.
Figures: every a-/b-wave waveform panel at flashes 25 and 8 has the OP trace (75-300
Hz) in a panel under it, same time axis, OP window shaded. fig_sessiontype_S<k>_*:
one figure per session type, test and control eyes overlaid at each instant, one
row per animal plus the primary-set mean.

    python erg_cohort.py              # all pigs with Reference/erg_plan.csv
    python erg_cohort.py --no-figs
"""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import re
import subprocess
import sys

import numpy as np
from scipy import stats
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

import erg_all as EA
import reteval_io as RIO
import extract_erg as E
import pig_sources as PS

TOOLS = Path(__file__).resolve().parent
STUDY = TOOLS.parent
OUTDIR_NAME = "ERG Cohort Analysis"

FLASHES = ["25", "8", "0.06"]                 # all three analysed alike
PRIMARY, SECONDARY = "25", "8"                 # headline tables / restoration context only
MEASURES = [("b_amp", "b-wave amplitude (uV)"), ("b_t", "b-wave implicit time (ms)"),
            ("a_amp", "a-wave amplitude (uV)"), ("a_t", "a-wave implicit time (ms)"),
            # primary measure, ISCEV 2022 handling (erg_all.OP_*); flashes 8 and 25
            ("op_amp", "oscillatory potentials, summed OP1-4 amplitude (uV)"),
            # width of the b-wave at 80 % of its amplitude, and number of peaks (added 2026-11-01)
            ("b_width", "b-wave width at 80 % of its amplitude (ms)"),
            ("b_peaks", "b-wave peaks above 80 % of its amplitude (n)")]
EYES = [("RE", "Test"), ("LE", "Control")]
COMPARISONS = [  # name, later group, earlier group, electrode
    ("Cooling (Intra Jet - Pre Jet)", "Intra Jet", "Pre Jet", "Jet"),
    ("Jet restoration (Post Jet - Pre Jet)", "Post Jet", "Pre Jet", "Jet"),
    ("RM restoration (Post RM - Pre RM)", "Post RM", "Pre RM", "RM"),
    # every pair: Pre->Intra, Pre->Post AND Intra->Post
    ("Rewarming (Post Jet - Intra Jet)", "Post Jet", "Intra Jet", "Jet"),
]
RESTORATION = [c for c in COMPARISONS if "restoration" in c[0]]   # judged against 2 x SD
OP_WAVES = {}                  # (pig, sess, group, eye, flash) -> OP trace (75-300 Hz), flashes 8/25
OP_WINDOWS = {}                # same key -> (lo_ms, hi_ms): the window the OPs were measured in
# Pig 1 is exploratory: the method changed a lot after it. Results start from Pig 2.
EXPLORATORY_PIGS = {1}
# Sessions left out of every summary (2026-11-08): (pig, session number) -> reason. Their rows are kept,
# with Animal_set "excluded", so they can still be read.
EXCLUDED_SESSIONS = {(3, 3): "systemic change in both eyes; the animal died in recovery from this session"}
# Single traces left out of every comparison (2026-11-09): (pig, session number, group, flash) -> reason, both eyes.
# The recording row is kept, with the reason in Trace_excluded.
EXCLUDED_TRACES = {(4, 1, "Pre Jet", "0.06"): "noise: bench sheet 'control noisy at first'; control trace swings "
                                              "+-600 uV (947 uV pick), test trace flat (14 uV)"}
SET_PRIMARY, SET_ALL = "primary", "all incl. exploratory"
# Animals reported in groups: Pig 1 exploratory, Pigs 2-3-4 together, Pigs 5-6 together.
# The groups follow the animal (Pig 5 S1 still used the old calibration).
ANIMAL_GROUPS = {"Pigs 2-4": {2, 3, 4}, "Pigs 5-6": {5, 6}, "Pig 1 (exploratory)": {1}}
# Ring temperatures corrected for the calibration error; the contact calibration is taken as truth.
# Well sensor not touching the aluminum until Pig 6 S1 and Pig 5 S2: those earlier sessions read low
# by 5.3 to 6.0 C (same pig, Pig 5 S1 vs S2; thesis code/checks/ring_calibration/REPORT.md).
RING_OFFSET_C = 5.5
CONTACT_CAL = {5: 2, 6: 1}      # pig -> first session calibrated in contact
# Exposure groups (2026-11-01): by the ring temperature delivered, so they follow the session.
GROUP_WARM, GROUP_COLD = "Ring about 9.5 C (Pigs 2-4, Pig 5 S1)", "Ring about 4 C (Pig 5 S2 on, Pig 6)"
WIDTH_FRAC = 0.80           # width of the b-wave at this fraction of its amplitude
PEAK_PROM = 0.05            # a separate peak stands out by this fraction of the amplitude
PEAK_AFTER_MS = 80.0        # peaks are looked for up to this long after the b-wave peak
TEMP_WINDOW_MIN = 10        # core temperature: nearest monitoring reading within +-10 min
TEMP_COL_NAME = "temp"      # Monitoring header containing this (e.g. 'Temp (F)')
RESTORE_K = 2.0             # restored within RESTORE_K x SD of the control eye's change
DRIFT_FLAG_X = 3.0          # flag a recording whose pre-flash drift > 3 x the cohort median
EYE_FLOOR = {"amp": 20.0, "time": 3.0}   # eye-check usual range never narrower than +-20 % / +-3 ms
RECOVERY_MIN_AFTER_OFF = 10 # a post Jet ERG counts as RECOVERY only >= 10 min after switch-off
                            # (log time); earlier ones are "end of cooling"
LAT_FILE = "erg_latency_overrides.csv"


# ------------------------------------------------------------------ helpers
def animal_set(pig, sess=None):
    if int(pig) in EXPLORATORY_PIGS:
        return "exploratory"
    if sess is not None and (int(pig), session_no(str(sess))) in EXCLUDED_SESSIONS:
        return "excluded"
    return "primary"


def excluded(r):
    return animal_set(r["Pig"], r.get("Session")) == "excluded"


def animal_group(pig):
    return next((g for g, ps in ANIMAL_GROUPS.items() if int(pig) in ps), "")


def contact_cal(pig, sess):
    first = CONTACT_CAL.get(int(pig))
    return first is not None and (session_no(sess) or 0) >= first


def ring_offset(pig, sess):
    """Add this to a reported ring temperature of the session (0 for contact calibration)."""
    return 0.0 if contact_cal(pig, sess) else RING_OFFSET_C


def ring_group(pig, sess):
    if int(pig) in EXPLORATORY_PIGS:
        return ""
    return GROUP_COLD if contact_cal(pig, sess) else GROUP_WARM


def analysis_sets():
    """(name, keep) for every across-animal summary: primary first, then each exposure group
    (ring delivered), then all animals.
    2026-11-11: the animal groups (Pigs 2-4, Pigs 5-6) are no longer summarized - they
    duplicated the exposure groups but put Pig 5 S1 (ring about 9.5 C) with the 4 C sessions.
    Animal_group stays on each row as a label only."""
    return ([(SET_PRIMARY, lambda r: animal_set(r["Pig"], r["Session"]) == "primary")]
            + [(g, lambda r, g=g: ring_group(r["Pig"], r["Session"]) == g and not excluded(r))
               for g in (GROUP_WARM, GROUP_COLD)]
            + [(SET_ALL, lambda r: not excluded(r))])


def primary_pigs_label(pigs):
    ps = sorted(int(p) for p in pigs if int(p) not in EXPLORATORY_PIGS)
    if not ps:
        return "none"
    return f"Pigs {ps[0]}-{ps[-1]}" if ps == list(range(ps[0], ps[-1] + 1)) and len(ps) > 1 \
        else "Pig" + ("s " if len(ps) > 1 else " ") + ", ".join(map(str, ps))


def session_minutes(name):
    """'Session 2 - 1hr' -> 60, '... 30min' -> 30, End of Life -> 0."""
    m = re.search(r"(\d+)\s*min", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*hr", name)
    return int(m.group(1)) * 60 if m else 0


def session_no(name):
    m = re.match(r"Session (\d+)", name)
    return int(m.group(1)) if m else None


def hhmm_to_min(s):
    if isinstance(s, datetime.time):
        return s.hour * 60 + s.minute + s.second / 60
    if isinstance(s, datetime.datetime):
        return s.hour * 60 + s.minute + s.second / 60
    m = re.match(r"\s*(\d{1,2}):(\d{2})", str(s or ""))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def load_overrides(pig_dir):
    """{(session, devtime, eye, flash, wave): latency_ms} - same file and keys as
    build_pig_TC_analysis.load_latency_overrides (snapped to the 0.5 ms grid)."""
    path = pig_dir / "Reference" / LAT_FILE
    out = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                v = round(float(r["Latency_ms"]) * 2) / 2
            except (TypeError, ValueError):
                continue
            out[(r["Session"].strip(), str(r["DeviceTime"]).strip(), r["Eye"].strip().upper(),
                 str(r["Flash"]).strip(), r["Wave"].strip().lower())] = v
    return out


def b_shape(f, ia, ib):
    """Width (ms) of the main b-wave peak at WIDTH_FRAC of its amplitude (a-wave trough to b-wave
    peak), and the number of peaks above that level. The width stops at the first trough after
    the peak: a later wave that keeps the trace above the level is not part of the peak.
    f = filtered trace on EA.GRID; ia, ib = picks."""
    from scipy.signal import find_peaks
    amp = float(f[ib] - f[ia])
    if amp <= 0 or ib <= ia:
        return None, None
    lev = float(f[ib]) - (1 - WIDTH_FRAC) * amp
    g = EA.GRID
    prom = PEAK_PROM * amp
    hi = min(len(f), ib + int(PEAK_AFTER_MS / (g[1] - g[0])) + 1)
    pk, _ = find_peaks(f[ia:hi], height=lev, prominence=prom)
    mins, _ = find_peaks(-f, prominence=prom)
    after = mins[mins > ib]
    before = mins[(mins < ib) & (mins > ia)]
    stop_r = int(after[0]) if len(after) else len(f) - 1
    stop_l = int(before[-1]) if len(before) else ia
    i = ib
    while i > stop_l and f[i - 1] >= lev:
        i -= 1
    j = ib
    while j < stop_r and f[j + 1] >= lev:
        j += 1
    left = g[i] if i <= stop_l else g[i - 1] + (lev - f[i - 1]) / (f[i] - f[i - 1]) * (g[i] - g[i - 1])
    right = g[j] if j >= stop_r else g[j] + (lev - f[j]) / (f[j + 1] - f[j]) * (g[j + 1] - g[j])
    return float(right - left), max(1, len(pk))


def metrics(rec, eye, flash, sess_name, overrides):
    """a/b amplitude + implicit time of one recording, with any hand-corrected
    latency applied (amplitude then re-read from the same filtered trace at that
    latency, as the workbook formula does). a_amp returned as a positive magnitude."""
    m = rec["ab"].get((eye, flash))
    if not m:
        return None
    f = rec["waves_f"][(eye, flash)]
    at, bt = m["a_t"], m["b_t"]
    hand = []
    for wave in ("a", "b"):
        v = overrides.get((sess_name, str(rec["devtime"]).strip(), eye, flash, wave))
        if v is not None:
            hand.append(wave)
            if wave == "a":
                at = v
            else:
                bt = v
    if hand:
        ia = int(np.argmin(np.abs(EA.GRID - at)))
        ib = int(np.argmin(np.abs(EA.GRID - bt)))
        a_amp = float(f[ia] - m["base"])
        b_amp = float(f[ib] - f[ia])
    else:
        a_amp, b_amp = m["a_amp"], m["b_amp"]
    op = (rec.get("ops") or {}).get((eye, flash))       # flashes 8 and 25 only (erg_all.OP_FLASHES)
    width, peaks = b_shape(f, int(np.argmin(np.abs(EA.GRID - at))), int(np.argmin(np.abs(EA.GRID - bt))))
    return dict(a_amp=-a_amp, a_t=at, b_amp=b_amp, b_t=bt, b_width=width, b_peaks=peaks, ok=bool(m.get("ok", True)),
                hand="+".join(hand), op_amp=(op or {}).get("sum"), op=op)


def read_core_temps(session_dir):
    """[(minute_of_day, degC)] from the Monitoring sheet, plus the minute from which
    the log says the probe was off (None if never). Readings above 50 are degF."""
    files = [f for f in session_dir.glob("*Systemic Parameters/*.xlsx")
             if not f.name.startswith("~$") and "BACKUP" not in f.name]
    if not files:
        return [], None, "no Systemic Parameters file"
    wb = openpyxl.load_workbook(files[0], read_only=True, data_only=True)
    if "Monitoring" not in wb.sheetnames:
        wb.close()
        return [], None, "no Monitoring sheet"
    rows = list(wb["Monitoring"].iter_rows(values_only=True))
    wb.close()
    hdr = [str(h or "").lower() for h in rows[0]]
    tc = next((i for i, h in enumerate(hdr) if TEMP_COL_NAME in h), None)
    if tc is None:
        return [], None, "no temperature column"
    out, probe_off = [], None
    for r in rows[1:]:
        t = hhmm_to_min(r[0]) if r and r[0] is not None else None
        v = r[tc] if tc < len(r) else None
        note = " ".join(str(x) for x in r if isinstance(x, str)).lower()
        if t is not None and "probe off" in note and probe_off is None:
            probe_off = t
        if t is None or not isinstance(v, (int, float)):
            continue
        c = (v - 32) * 5 / 9 if v > 50 else float(v)
        out.append((t, round(c, 2)))
    return out, probe_off, ""


def core_temp_at(temps, probe_off, why_none, wall_min):
    """(degC, offset_min, note) for the nearest reading within TEMP_WINDOW_MIN."""
    if wall_min is None:
        return None, None, "no wall time"
    if not temps:
        return None, None, why_none or "no readings"
    t, c = min(temps, key=lambda x: abs(x[0] - wall_min))
    off = round(t - wall_min, 1)
    if abs(off) <= TEMP_WINDOW_MIN:
        return c, off, ""
    if probe_off is not None and wall_min >= probe_off:
        return None, None, f"not measured (log: temperature probe off from {probe_off // 60:02.0f}:{probe_off % 60:02.0f})"
    return None, None, f"no reading within {TEMP_WINDOW_MIN} min (nearest {off:+.0f} min)"


RING_WIN_S = 60             # ring temperature = mean of the DAQ trace +-60 s around the ERG
RING_BASE_S = 120           # pre-cooling ring baseline = the 120 s before Cooling Start


def ring_trace(sess_dir):
    """Ring (SCR thermistor) trace on the local clock + its events, or None.
    -> dict(t=minutes-of-day array, T=degC array, start, off, removed, base, tmin, note)."""
    import cooling_live as CL
    cool = PS.cooling_dir(sess_dir)
    csvf = cool / "Temperature Processed.csv"
    if not csvf.exists():
        return None
    # uncorrected: this script adds ring_offset() itself (Ring_temp_C raw, Ring_temp_corrected_C)
    times, _cur, temps, t0 = CL._read_trace(csvf, CL._session_utc_offset_h(cool), uncorrected=True)
    if t0 is None:
        return None
    t = np.array(times, float)
    T = np.array([np.nan if x is None else x for x in temps], float)
    ev = {}
    evf = cool / "Temperature Events.xlsx"
    if evf.exists():
        wb = openpyxl.load_workbook(evf, read_only=True, data_only=True)
        for r in wb["Events"].iter_rows(values_only=True):
            if not r or not isinstance(r[0], str):
                continue
            man = r[3] if len(r) > 3 else None
            val = man if man not in (None, "") else (r[1] if len(r) > 1 else None)
            if isinstance(val, str) and val.strip().lower() in ("none", "n/a"):
                val = None
            ev[r[0].strip()] = float(val) if isinstance(val, (int, float)) else None
        wb.close()
    start, off = ev.get("Cooling Start"), ev.get("Device Off")
    base = tmin = None
    note = ""
    if start is not None:
        m = (t >= start - RING_BASE_S) & (t < start)
        base = float(np.nanmean(T[m])) if m.any() and np.isfinite(T[m]).any() else None
        if not (t >= start).any():
            note = "cooling NOT RECORDED (switch-on after the file ends; times from the log)"
        if off is not None:
            m = (t >= start) & (t <= off)
            tmin = float(np.nanmin(T[m])) if m.any() and np.isfinite(T[m]).any() else None
    else:
        note = "no cooling in this DAQ record"
    to_min = lambda s: None if s is None else (t0 + s) / 60.0
    return dict(t=(t0 + t) / 60.0, T=T, start=to_min(start), off=to_min(off),
                removed=to_min(ev.get("Device Removed")), base=base, tmin=tmin, note=note)


def ring_at(rt, wall_min):
    """(ring degC, deficit below baseline degC, fraction rewarmed) at a wall time."""
    if rt is None or wall_min is None:
        return None, None, None
    m = np.abs(rt["t"] - wall_min) <= RING_WIN_S / 60.0
    if not m.any() or not np.isfinite(rt["T"][m]).any():
        return None, None, None
    T = float(np.nanmean(rt["T"][m]))
    if rt["base"] is None:
        return round(T, 2), None, None
    deficit = rt["base"] - T
    frac = ((T - rt["tmin"]) / (rt["base"] - rt["tmin"])
            if rt["tmin"] is not None and rt["base"] - rt["tmin"] > 0.5 else None)
    return round(T, 2), round(deficit, 2), (round(frac, 3) if frac is not None else None)


def ci95(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    n = x.size
    if n == 0:
        return n, None, None, None, None
    m = float(x.mean())
    if n < 2:
        return n, m, None, None, None
    sd = float(x.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return n, m, sd, m - h, m + h


def paired_t(later, earlier):
    pairs = [(a, b) for a, b in zip(later, earlier) if a is not None and b is not None]
    if len(pairs) < 2:
        return None, None
    a, b = np.array(pairs).T
    if np.allclose(a - b, (a - b)[0]):
        return None, None
    r = stats.ttest_rel(a, b)
    return float(r.statistic), float(r.pvalue)


# Holm and effect sizes (2026-11-04): Chapter 4 statistics done the same way as Chapter 2 (Aim 1).
# The test stays the PAIRED t test (test eye's change against the control eye's change, same animal).
#  - Holm across the three cooling durations, within Set x Comparison x Flash x Measure, on
#    p_test_vs_control (only the primary set, flash 25, b_t and b_amp, Cooling, are interpreted;
#    the ring temperature, a device endpoint, is outside the family; within-eye tests stay descriptive)
#  - Hedges-corrected paired effect size g = d_z x J, d_z = mean / SD of the paired differences,
#    J = 1 - 3 / (4 (n - 1) - 1), with an approximate 95 % CI from the large-sample normal
#    approximation SE(d_z) = sqrt(1/n + d_z^2 / (2 n)), scaled by J (only approximate at n = 3-5)
#  - t-based 90 % CI of the mean test-minus-control difference: its limit farther from zero bounds
#    the largest difference compatible with the data (restoration, IOP)
def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def ci90(n, m, sd):
    n, m, sd = _f(n), _f(m), _f(sd)
    if n is None or m is None or sd is None or n < 2:
        return None, None
    h = float(stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n))
    return m - h, m + h


def hedges_gz(n, m, sd):
    """Hedges-corrected standardized mean of the paired differences, with approx. 95 % CI."""
    n, m, sd = _f(n), _f(m), _f(sd)
    if n is None or m is None or sd is None or n < 3 or sd <= 0:
        return None, None, None
    dz = m / sd
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    se = float(np.sqrt(1.0 / n + dz * dz / (2.0 * n)))
    z = float(stats.norm.ppf(0.975))
    return j * dz, j * (dz - z * se), j * (dz + z * se)


def holm(ps):
    """Holm step-down adjusted p values (None stays None)."""
    idx = [i for i, p in enumerate(ps) if _f(p) is not None]
    order = sorted(idx, key=lambda i: _f(ps[i]))
    k, out, run = len(order), [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (k - rank) * _f(ps[i])))
        out[i] = run
    return out


def add_ch2_stats(rows, mean="Mean_T_minus_C", sd="SD", p="p_test_vs_control",
                  family=("Set", "Comparison", "Flash", "Measure")):
    """Append g_paired (95 % CI), CI90 and p_holm to summary rows in place (the CSV keeps every
    earlier column as it was). Works on rows read back from a CSV too (strings)."""
    for r in rows:
        g, glo, ghi = hedges_gz(r.get("n_animals"), r.get(mean), r.get(sd))
        lo, hi = ci90(r.get("n_animals"), r.get(mean), r.get(sd))
        r.update(CI90_low=lo, CI90_high=hi, g_paired=g, g_CI95_low_approx=glo, g_CI95_high_approx=ghi)
    fams = {}
    for i, r in enumerate(rows):
        if _f(r.get(p)) is not None:
            fams.setdefault(tuple(r.get(k) for k in family), []).append(i)
    for r in rows:
        r["p_holm_test_vs_control"] = None
        r["Holm_family_size"] = None
    for ids in fams.values():
        for i, ph in zip(ids, holm([rows[i][p] for i in ids])):
            rows[i]["p_holm_test_vs_control"] = ph
            rows[i]["Holm_family_size"] = len(ids)
    return rows


# ------------------------------------------------------------------ collect
def collect(study):
    recs_out = []          # one row per pig/session/group/eye/flash
    by_key = {}            # (pig, sess, group, eye, flash) -> metrics dict
    sess_info = {}         # (pig, sess) -> dict
    waves = {}             # (pig, sess, group, eye, flash) -> filtered trace
    for pig_dir in sorted((d for d in study.iterdir()
                           if d.is_dir() and re.fullmatch(r"Pig \d+", d.name)),
                          key=lambda d: int(d.name.split()[1])):
        if not (pig_dir / "Reference" / "erg_plan.csv").exists():
            continue
        pig = int(pig_dir.name.split()[1])
        ov = load_overrides(pig_dir)
        for sess_dir in sorted(pig_dir.glob("Session *")):
            recs = EA.extract(sess_dir)
            if not recs:
                continue
            groups = EA.choose_by_group(recs)
            temps, probe_off, why = read_core_temps(sess_dir)
            rt = ring_trace(sess_dir)
            cool_on, cool_off = recs[0].get("cool_on"), recs[0].get("cool_off")
            sess_info[(pig, sess_dir.name)] = dict(
                minutes=session_minutes(sess_dir.name), no=session_no(sess_dir.name),
                cool_on=cool_on, cool_off=cool_off, groups=sorted(groups))
            for gname, rec in groups.items():
                wall = hhmm_to_min(rec["walltime"])
                tC, toff, tnote = core_temp_at(temps, probe_off, why, wall)
                ringT, ring_def, ring_frac = ring_at(rt, wall)
                daq_off = (round(wall - rt["off"], 1) if rt and rt["off"] is not None
                           and wall is not None and gname.startswith("Post") else None)
                after_off = (round(wall - cool_off, 1)
                             if wall is not None and cool_off is not None and gname.startswith("Post")
                             else None)
                sess_info[(pig, sess_dir.name)].setdefault("post_min", {})[gname] = after_off
                for eye, role in EYES:
                    for fl in FLASHES:
                        m = metrics(rec, eye, fl, sess_dir.name, ov)
                        if m is None:
                            continue
                        key = (pig, sess_dir.name, gname, eye, fl)
                        trace_out = EXCLUDED_TRACES.get((int(pig), session_no(sess_dir.name), gname, str(fl)), "")
                        if not trace_out:
                            by_key[key] = m
                            waves[key] = rec["waves_f"][(eye, fl)]
                        if m["op"] is not None:
                            OP_WAVES[key] = m["op"]["trace"]
                            OP_WINDOWS[key] = tuple(m["op"]["window"])
                        sw = rec.get("sweeps", {}).get((eye, fl), [])
                        sw_b = [s["ab"]["b_amp"] for s in sw if s["kept"] and s["ab"]]
                        n_kept = sum(s["kept"] for s in sw)
                        src = f"raw ({rec.get('drift')}), mean of {n_kept}/{len(sw)} sweeps"
                        alt = rec.get("alt", {}).get((eye, fl), {})
                        alt_ab = alt.get("ab") or {}
                        recs_out.append(dict(
                            Pig=pig, Animal_set=animal_set(pig, sess_dir.name), Animal_group=animal_group(pig),
                            Session=sess_dir.name, Duration_min=session_minutes(sess_dir.name),
                            Group=gname, Label=rec["label"], DeviceTime=rec["devtime"],
                            WallTime=rec["walltime"], Min_after_switch_off=after_off,
                            Core_temp_C=tC, Core_temp_offset_min=toff, Core_temp_note=tnote,
                            Min_after_DAQ_device_off=daq_off,
                            Ring_temp_C=ringT,
                            Ring_calibration=("contact" if contact_cal(pig, sess_dir.name) else "not in contact"),
                            Ring_temp_corrected_C=(round(ringT + ring_offset(pig, sess_dir.name), 2)
                                                   if ringT is not None else None),
                            Ring_group=ring_group(pig, sess_dir.name),
                            Ring_baseline_C=(round(rt["base"], 2) if rt and rt["base"] is not None else None),
                            Ring_min_C=(round(rt["tmin"], 2) if rt and rt["tmin"] is not None else None),
                            Ring_below_baseline_C=ring_def, Ring_rewarmed_fraction=ring_frac,
                            Ring_note=(rt["note"] if rt else "no cooling trace"),
                            Eye=role, Flash=fl,
                            b_amp_uV=round(m["b_amp"], 2), b_t_ms=m["b_t"],
                            b_width_ms=(round(m["b_width"], 2) if m["b_width"] is not None else None),
                            b_peaks=m["b_peaks"],
                            a_amp_uV=round(m["a_amp"], 2), a_t_ms=m["a_t"],
                            OP_sum_uV=(round(m["op_amp"], 1) if m["op_amp"] is not None else None),
                            OP_rms_uV=(round(m["op"]["rms"], 2) if m["op"] else None),
                            **{f"OP{k + 1}_{u}": (round(m["op"][key_][k], 1)
                                                  if m["op"] and k < len(m["op"][key_]) else None)
                               for k in range(EA.OP_N) for u, key_ in (("uV", "amps"), ("ms", "times"))},
                            OP_window_ms=(f"{m['op']['window'][0]:g}-{m['op']['window'][1]:g}" if m["op"] else ""),
                            Pick_ok=m["ok"], Hand_corrected=m["hand"] or "",
                            Waveform_source=src,
                            Sweeps_kept=n_kept, Sweeps_total=len(sw),
                            Sweep_b_amp_uV=", ".join(
                                (f"{s['ab']['b_amp']:.0f}" if s["ab"] else "-") + ("" if s["kept"] else "*")
                                for s in sw),
                            Sweep_b_amp_SD=(round(float(np.std(sw_b, ddof=1)), 1) if len(sw_b) > 1 else None),
                            Drift_uV_per_s=(round(float(np.median([abs(s["drift"]) for s in sw if s["kept"]])), 0)
                                            if sw and "drift" in sw[0] else None),
                            Sweep_rejected_why=" | ".join(f"#{i + 1}: {s['reason']}" for i, s in enumerate(sw)
                                                          if s["reason"]),
                            # same recording with the pre-flash line fit instead of the ISCEV 0.3 Hz
                            # high-pass (method comparison only)
                            LINEFIT_b_amp_uV=(round(alt_ab["b_amp"], 2) if alt_ab else None),
                            LINEFIT_b_t_ms=alt_ab.get("b_t"),
                            LINEFIT_a_amp_uV=(round(-alt_ab["a_amp"], 2) if alt_ab else None),
                            LINEFIT_a_t_ms=alt_ab.get("a_t"),
                            Repeated_flash_steps=", ".join(f"{e} {f}" for e, f in rec.get("repeats", [])),
                            Trace_excluded=trace_out))
    return recs_out, by_key, sess_info, waves


def effects(by_key, sess_info):
    rows = []
    for (pig, sess), info in sorted(sess_info.items(), key=lambda kv: (kv[0][0], kv[1]["no"] or 0)):
        if info["minutes"] == 0:
            continue                            # End of Life: no cooling
        for comp, later, earlier, el in COMPARISONS:
            for fl in FLASHES:
                for meas, _lab in MEASURES:
                    v = {}
                    for eye, _ in EYES:
                        a = by_key.get((pig, sess, later, eye, fl))
                        b = by_key.get((pig, sess, earlier, eye, fl))
                        v[eye] = (a.get(meas) if a else None, b.get(meas) if b else None)
                    if meas == "op_amp" and fl not in EA.OP_FLASHES:
                        continue                # no OPs at the dim (rod) flash
                    missing = [g for g in (later, earlier)
                               if not any((pig, sess, g, e, fl) in by_key for e, _ in EYES)]
                    dT = (v["RE"][0] - v["RE"][1]) if None not in v["RE"] else None
                    dC = (v["LE"][0] - v["LE"][1]) if None not in v["LE"] else None
                    dd = dT - dC if dT is not None and dC is not None else None
                    # recovery rule: a post Jet ERG made < RECOVERY_MIN_AFTER_OFF after
                    # switch-off is the END OF COOLING, not recovery - kept here, but
                    # left out of the restoration test and its pooled SD
                    pm = info.get("post_min", {}).get(later)
                    end_of_cooling = (later == "Post Jet" and pm is not None
                                      and pm < RECOVERY_MIN_AFTER_OFF)
                    if end_of_cooling:
                        missing = missing + [f"END OF COOLING: recorded {pm:g} min after switch-off "
                                             f"(< {RECOVERY_MIN_AFTER_OFF} min), not a recovery recording"]
                    rows.append(dict(Pig=pig, Animal_set=animal_set(pig, sess), Animal_group=animal_group(pig),
                                     Session=sess, Duration_min=info["minutes"],
                                     Comparison=comp, Electrode=el, Flash=fl, Measure=meas,
                                     Test_earlier=v["RE"][1], Test_later=v["RE"][0],
                                     Control_earlier=v["LE"][1], Control_later=v["LE"][0],
                                     dTest=dT, dControl=dC, Test_minus_Control=dd,
                                     Recovery_eligible=not end_of_cooling,
                                     Note="; ".join(m if m.startswith("END") else f"no {m} recording"
                                                    for m in missing)))
    return rows


def cooling_summary(eff_all):
    """Across-animal statistics, for the PRIMARY set (Pigs 2-6) first and then with the
    exploratory Pig 1 included; column Set says which."""
    out = []
    for set_name, keep in analysis_sets():
        out += _cooling_summary([r for r in eff_all if keep(r)], set_name)
    return add_ch2_stats(out)       # Holm and effect sizes, as Ch. 2


def _cooling_summary(eff, set_name):
    out = []
    for comp, *_ in COMPARISONS:
        for fl in FLASHES:
            for meas, lab in MEASURES:
                for dur in sorted({r["Duration_min"] for r in eff}):
                    rs = [r for r in eff if r["Comparison"] == comp and r["Flash"] == fl
                          and r["Measure"] == meas and r["Duration_min"] == dur
                          and r["Test_minus_Control"] is not None
                          and r.get("Recovery_eligible", True)]
                    if not rs:
                        continue
                    n, m, sd, lo, hi = ci95([r["Test_minus_Control"] for r in rs])
                    t_tc, p_tc = paired_t([r["dTest"] for r in rs], [r["dControl"] for r in rs])
                    nT, mT, sdT, loT, hiT = ci95([r["dTest"] for r in rs])
                    t_ba, p_ba = paired_t([r["Test_later"] for r in rs], [r["Test_earlier"] for r in rs])
                    # test vs test, control vs control, and the two compared
                    nC, mC, sdC, loC, hiC = ci95([r["dControl"] for r in rs])
                    t_cc, p_cc = paired_t([r["Control_later"] for r in rs], [r["Control_earlier"] for r in rs])
                    out.append(dict(Set=set_name, Comparison=comp, Flash=fl, Measure=meas, Duration_min=dur,
                                    n_animals=n, Pigs=", ".join(str(r["Pig"]) for r in rs),
                                    Per_animal=", ".join(f"{r['Test_minus_Control']:+.1f}" for r in rs),
                                    Mean_T_minus_C=m, SD=sd, CI95_low=lo, CI95_high=hi,
                                    t_test_vs_control=t_tc, p_test_vs_control=p_tc,
                                    Mean_dTest=mT, SD_dTest=sdT, dTest_CI95_low=loT, dTest_CI95_high=hiT,
                                    t_before_vs_after=t_ba, p_before_vs_after=p_ba,
                                    Mean_dControl=mC, SD_dControl=sdC, dControl_CI95_low=loC, dControl_CI95_high=hiC,
                                    t_control_before_vs_after=t_cc, p_control_before_vs_after=p_cc,
                                    Count=None))     # keeps Count where it was; new columns after it
    # "taken together" count, b-wave, cooling comparison
    for fl in FLASHES:
        for dur in sorted({r["Duration_min"] for r in eff}):
            def get(meas, pig):
                return next((r["Test_minus_Control"] for r in eff
                             if r["Comparison"] == COMPARISONS[0][0] and r["Flash"] == fl
                             and r["Measure"] == meas and r["Duration_min"] == dur
                             and r["Pig"] == pig), None)
            pigs = sorted({r["Pig"] for r in eff if r["Duration_min"] == dur})
            ok = [(p, get("b_t", p), get("b_amp", p)) for p in pigs]
            ok = [(p, t, a) for p, t, a in ok if t is not None and a is not None]
            yes = [p for p, t, a in ok if t > 0 and a < 0]
            out.append(dict(Set=set_name, Comparison="Taken together: b-wave slower AND smaller (cooling)",
                            Flash=fl, Measure="b_t>0 & b_amp<0", Duration_min=dur,
                            n_animals=len(ok), Pigs=", ".join(str(p) for p, *_ in ok),
                            Per_animal=", ".join(f"P{p}:{'yes' if p in yes else 'no'}" for p, *_ in ok),
                            Mean_T_minus_C=None, SD=None, CI95_low=None, CI95_high=None,
                            t_test_vs_control=None, p_test_vs_control=None, Mean_dTest=None,
                            dTest_CI95_low=None, dTest_CI95_high=None, t_before_vs_after=None,
                            p_before_vs_after=None, Count=f"{len(yes)} of {len(ok)}"))
    return out


def restoration(eff, recs_out=()):
    # the post recording's timing and ring temperature, beside each verdict (some
    # Pig 2 post Jet ERGs were taken before the ring had rewarmed)
    post = {(r["Pig"], r["Session"], r["Group"]): r for r in recs_out
            if r["Eye"] == "Test" and r["Flash"] == PRIMARY}
    out = []
    for comp, _l, _e, el in RESTORATION:
        for fl in FLASHES:
            for meas, _lab in MEASURES:
                rs = [r for r in eff if r["Comparison"] == comp and r["Flash"] == fl
                      and r["Measure"] == meas]
                # pooled over the PRIMARY set (Pig 1 is exploratory); the
                # pool with Pig 1 is kept alongside (Limit_incl_exploratory)
                pool_all = [r["dControl"] for r in rs if not excluded(r)
                            and r["dControl"] is not None and r.get("Recovery_eligible", True)]
                pool = [r["dControl"] for r in rs if animal_set(r["Pig"], r["Session"]) == "primary"
                        and r["dControl"] is not None and r.get("Recovery_eligible", True)]
                sd = float(np.std(pool, ddof=1)) if len(pool) >= 2 else None
                sd_all = float(np.std(pool_all, ddof=1)) if len(pool_all) >= 2 else None
                # Eye check. Change of each eye on its own scale: amplitudes
                # in % of that eye's earlier value (baselines differ between pigs), times in
                # ms. Usual range = control eyes' median +- max(2 x 1.4826 MAD, floor), so
                # a few large changes don't widen it (e.g. Pig 3 S3), and never narrower than the floor
                # (EYE_FLOOR) so changes at the 0.5-ms resolution are not flagged.
                amp = meas.endswith("amp")

                def own(later, earlier):
                    if later is None or earlier is None:
                        return None
                    if amp:
                        return 100.0 * (later - earlier) / abs(earlier) if abs(earlier) > 1e-9 else None
                    return later - earlier
                cpool = [own(r["Control_later"], r["Control_earlier"]) for r in rs
                         if r.get("Recovery_eligible", True) and animal_set(r["Pig"], r["Session"]) == "primary"]
                cpool = [x for x in cpool if x is not None]
                med = float(np.median(cpool)) if cpool else None
                half = (max(RESTORE_K * 1.4826 * float(np.median(np.abs(np.array(cpool) - med))),
                            EYE_FLOOR["amp" if amp else "time"]) if len(cpool) >= 3 else None)
                for r in rs:
                    dd = r["Test_minus_Control"]
                    # back to its own baseline, test eye vs test eye; the old
                    # test-minus-control verdict is kept in Verdict_T_minus_C
                    dt = r["dTest"]
                    verdict = ("" if dt is None or sd is None
                               else ("restored" if abs(dt) <= RESTORE_K * sd else "NOT restored"))
                    verdict_all = ("" if dt is None or sd_all is None
                                   else ("restored" if abs(dt) <= RESTORE_K * sd_all else "NOT restored"))
                    verdict_tc = ("" if dd is None or sd is None
                                  else ("restored" if abs(dd) <= RESTORE_K * sd else "NOT restored"))
                    if not r.get("Recovery_eligible", True):
                        verdict = verdict_all = verdict_tc = "excluded (end of cooling)"
                    p = post.get((r["Pig"], r["Session"], _l), {})
                    # when the TEST eye has not come back, look at the CONTROL
                    # eye over the same interval; if it moved the same way beyond its usual
                    # range, the cause is probably systemic (e.g. anaesthesia), not cooling.
                    flag, t_out, c_out = "", None, None
                    ot = own(r["Test_later"], r["Test_earlier"])
                    oc = own(r["Control_later"], r["Control_earlier"])
                    if half and ot is not None and oc is not None and r.get("Recovery_eligible", True):
                        zt, zc = ot - med, oc - med
                        t_out, c_out = abs(zt) > half, abs(zc) > half
                        if t_out and c_out and zt * zc > 0:
                            flag = "BOTH eyes changed the same way - systemic cause suspected (e.g. anaesthesia)"
                        elif t_out:
                            flag = "test eye not back to baseline (control eye within its usual range)"
                        elif c_out:
                            flag = "control eye changed beyond its usual range - check the recording"
                    out.append(dict(Comparison=comp, Electrode=el, Flash=fl, Measure=meas,
                                    Pig=r["Pig"], Animal_set=animal_set(r["Pig"], r["Session"]), Animal_group=animal_group(r["Pig"]),
                                    Session=r["Session"], Duration_min=r["Duration_min"],
                                    Test_minus_Control=dd, Control_change_SD_pooled=sd,
                                    n_pooled=len(pool), Limit=(RESTORE_K * sd if sd is not None else None),
                                    Verdict=verdict, Verdict_T_minus_C=verdict_tc,
                                    Limit_incl_exploratory=(RESTORE_K * sd_all if sd_all is not None else None),
                                    Verdict_incl_exploratory=verdict_all,
                                    dTest=r["dTest"], dControl=r["dControl"],
                                    Test_outside_usual=("yes" if t_out else ("no" if t_out is not None else "")),
                                    Control_outside_usual=("yes" if c_out else ("no" if c_out is not None else "")),
                                    Eye_check_unit=("% of own earlier value" if amp else "ms"),
                                    Test_own_change=ot, Control_own_change=oc,
                                    Usual_range_low=(med - half if half else None),
                                    Usual_range_high=(med + half if half else None),
                                    Eye_check=flag,
                                    Post_min_after_switch_off=p.get("Min_after_switch_off"),
                                    Post_ring_temp_C=p.get("Ring_temp_C"),
                                    Post_ring_below_baseline_C=p.get("Ring_below_baseline_C"),
                                    Post_ring_rewarmed_fraction=p.get("Ring_rewarmed_fraction"),
                                    Note=r["Note"]))
    return out


def between_sessions(by_key, sess_info, recs_out):
    """Baseline of session k minus baseline of session 1, per eye and test minus control.
    From 2026-11-04, for all measures: the Jet recording before
    cooling (Pre Jet) is the PRIMARY comparison, the RM recording before installation (Pre RM)
    the CONFIRMATORY one. The Pre RM rows come first, exactly as before; the Pre Jet rows follow,
    and the columns Baseline_recording / Electrode / Role say which is which."""
    out = []
    for grp, el, role in (("Pre RM", "RM", "confirmatory"), ("Pre Jet", "Jet", "primary")):
        out += _between_sessions(by_key, sess_info, recs_out, grp, el, role)
    return out


def _between_sessions(by_key, sess_info, recs_out, grp, el, role):
    temp = {(r["Pig"], r["Session"], r["Group"]): (r["Core_temp_C"], r["Core_temp_note"])
            for r in recs_out}
    out = []
    pigs = sorted({p for p, _ in sess_info})
    for pig in pigs:
        sess = sorted((s for p, s in sess_info if p == pig), key=lambda s: sess_info[(pig, s)]["no"] or 0)
        ref = next((s for s in sess if sess_info[(pig, s)]["no"] == 1), None)
        if ref is None:
            continue
        for s in sess:
            if s == ref:
                continue
            if grp != "Pre RM" and not any((pig, s, grp, e, f) in by_key for e, _ in EYES for f in FLASHES):
                continue            # e.g. End of Life: no Jet recording
            for fl in FLASHES:
                for meas, _lab in MEASURES:
                    if meas == "op_amp" and fl not in EA.OP_FLASHES:
                        continue
                    d = {}
                    for eye, _ in EYES:
                        a = by_key.get((pig, s, grp, eye, fl))
                        b = by_key.get((pig, ref, grp, eye, fl))
                        d[eye] = (a[meas] - b[meas]) if a and b and a[meas] is not None                             and b[meas] is not None else None
                    dd = d["RE"] - d["LE"] if None not in d.values() else None
                    tk, nk = temp.get((pig, s, grp), (None, ""))
                    t1, n1 = temp.get((pig, ref, grp), (None, ""))
                    out.append(dict(Pig=pig, Animal_set=animal_set(pig, s), Animal_group=animal_group(pig), Session=s, Reference=ref,
                                    Flash=fl, Measure=meas,
                                    dTest_vs_S1=d["RE"], dControl_vs_S1=d["LE"], Test_minus_Control=dd,
                                    Core_temp_C_session=tk, Core_temp_C_S1=t1,
                                    Core_temp_diff_C=(round(tk - t1, 2) if tk is not None and t1 is not None else None),
                                    Note="; ".join(x for x in (nk, n1) if x),
                                    Baseline_recording=grp, Electrode=el, Role=role))
    return out


def method_check(recs_out):
    """Line fit minus HP03 (the method), per flash x measure, over every recording; plus sweep counts."""
    out = []
    pairs = [("b_amp_uV", "LINEFIT_b_amp_uV"), ("b_t_ms", "LINEFIT_b_t_ms"),
             ("a_amp_uV", "LINEFIT_a_amp_uV"), ("a_t_ms", "LINEFIT_a_t_ms")]
    for fl in FLASHES:
        rs = [r for r in recs_out if r["Flash"] == fl]
        for base, hp in pairs:
            d = np.array([r[hp] - r[base] for r in rs if r.get(hp) is not None and r.get(base) is not None])
            if not d.size:
                continue
            big = sorted(((abs(r[hp] - r[base]), r) for r in rs
                          if r.get(hp) is not None and r.get(base) is not None),
                         key=lambda x: -x[0])[:3]
            out.append(dict(Flash=fl, Measure=base, n=int(d.size), Mean_linefit_minus_HP03=float(d.mean()),
                            SD=float(d.std(ddof=1)) if d.size > 1 else None,
                            Median_abs=float(np.median(np.abs(d))), Max_abs=float(np.abs(d).max()),
                            Largest=", ".join(f"P{r['Pig']} {r['Session'][:9]} {r['Group']} {r['Eye']} ({x:.1f})"
                                              for x, r in big)))
        tot = sum(r["Sweeps_total"] for r in rs)
        kept = sum(r["Sweeps_kept"] for r in rs)
        out.append(dict(Flash=fl, Measure="sweeps", n=tot, Mean_linefit_minus_HP03=None, SD=None,
                        Median_abs=None, Max_abs=None,
                        Largest=f"{tot - kept} of {tot} sweeps rejected; "
                                f"{sum(1 for r in rs if r['Sweeps_kept'] < r['Sweeps_total'])} recordings affected"))
    return out


# ------------------------------------------------------------------ figures
OP_PANEL_RATIO = 0.6            # height of an OP panel relative to its a-/b-wave panel
EXPL_FACE = "#f2f2f2"           # background of an exploratory (Pig 1) panel
EXPL_TEST, EXPL_CTRL = "0.30", "0.62"     # Pig 1: grey test / control lines
TEST_COL, CTRL_COL = "#d62728", "#1f77b4"


def _row_axes(fig, kinds, ncols, hspace=0.12, wspace=0.08):
    """Grid of axes, one row per entry of kinds ("erg" or "op"): OP rows are shorter;
    every panel shares the time axis, and the y-axis is shared within each kind."""
    gs = fig.add_gridspec(len(kinds), ncols,
                          height_ratios=[1.0 if k == "erg" else OP_PANEL_RATIO for k in kinds],
                          hspace=hspace, wspace=wspace)
    axes = np.empty((len(kinds), ncols), dtype=object)
    ref = {}
    for i, k in enumerate(kinds):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j], sharex=ref.get("x"), sharey=ref.get(k))
            ref.setdefault("x", ax)
            ref.setdefault(k, ax)
            axes[i, j] = ax
    return axes


def _finish_axes(axes, kinds):
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            ax.axvline(0, color="0.8", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.tick_params(labelsize=7)
            if i < axes.shape[0] - 1:
                ax.tick_params(labelbottom=False)
            if j > 0:
                ax.tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("Time from flash (ms)", fontsize=8)


def _draw_op(ax, op, win, color, ls="-", lw=0.8, shade=0.08):
    """One OP trace (75-300 Hz) with the window its OPs were measured in shaded."""
    ax.plot(EA.GRID, op, color=color, lw=lw, ls=ls)
    if win:
        ax.axvspan(win[0], win[1], color=color, alpha=shade, lw=0)


def _expl(ax):
    ax.set_facecolor(EXPL_FACE)


def figures(waves, sess_info, outdir, ops=None, op_windows=None):
    """Every recording used, one panel per pig x eye x session; at flashes 25 and 8 an
    OP panel (75-300 Hz, OP window shaded) sits under each a-/b-wave panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ops, op_windows = ops or {}, op_windows or {}
    made = []
    pigs = sorted({p for p, _ in sess_info})
    t = EA.GRID
    jet_style = {"Pre Jet": ("#1f77b4", "Pre (Jet)"), "Intra Jet": ("#d62728", "During cooling (Jet)"),
                 "Post Jet": ("#2ca02c", "After switch-off (Jet)")}
    rm_style = {"Pre RM": ("#1f77b4", "Before device (RM)"), "Post RM": ("#2ca02c", "After removal (RM)")}
    for fl in FLASHES:
        with_op = fl in EA.OP_FLASHES
        for kind, style, sess_filter in (("Jet", jet_style, lambda i: i["minutes"] > 0),
                                         ("RM", rm_style, lambda i: True)):
            cols = sorted({sess_info[k]["no"] for k in sess_info if sess_filter(sess_info[k])})
            rows = [(pig, eye, role, k) for pig in pigs for eye, role in EYES
                    for k in (("erg", "op") if with_op else ("erg",))]
            kinds = [r[3] for r in rows]
            h = sum(2.0 if k == "erg" else 2.0 * OP_PANEL_RATIO for k in kinds)
            H = h + 1.9
            fig = plt.figure(figsize=(max(3.1 * len(cols), 8.5), H))
            axes = _row_axes(fig, kinds, len(cols))
            for ri, (pig, eye, role, k) in enumerate(rows):
                for ci, sno in enumerate(cols):
                    ax = axes[ri][ci]
                    sess = next((s for p, s in sess_info if p == pig and sess_info[(p, s)]["no"] == sno), None)
                    if animal_set(pig) == "exploratory":
                        _expl(ax)
                    if ri == 0:
                        ax.set_title(f"Session {sno}" + (f" ({sess_info[(pig, sess)]['minutes']} min)"
                                                          if sess and sess_info[(pig, sess)]["minutes"] else
                                                          (" (end of life)" if sess else "")), fontsize=9)
                    if ci == 0:
                        tag = " (exploratory)" if animal_set(pig) == "exploratory" else ""
                        ax.set_ylabel(f"Pig {pig}{tag} {role}\n(µV)" if k == "erg" else "OPs\n(µV)",
                                      fontsize=7 if k == "erg" else 6.5)
                    if sess is None:
                        if k == "erg":
                            ax.text(0.5, 0.5, "no session", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=8, color="0.5")
                        continue
                    drawn = 0
                    for g, (col, lab) in style.items():
                        key = (pig, sess, g, eye, fl)
                        if k == "erg" and key in waves:
                            ax.plot(t, waves[key], color=col, lw=0.9, label=lab)
                            drawn += 1
                        elif k == "op" and key in ops:
                            _draw_op(ax, ops[key], op_windows.get(key), col)
                            drawn += 1
                    if not drawn and k == "erg":
                        ax.text(0.5, 0.5, f"no {kind} recordings", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8, color="0.5")
            _finish_axes(axes, kinds)
            hh, ll = [], []
            for ax in axes.flat:
                for a_, b_ in zip(*ax.get_legend_handles_labels()):
                    if b_ not in ll:
                        hh.append(a_); ll.append(b_)
            fig.suptitle(f"Every {kind} ERG used in the analysis, flash {fl} cd·s/m²\n"
                         f"raw record, 0.3 Hz high-pass, accepted sweeps averaged, 75 Hz low-pass; "
                         f"last recording per phase\n"
                         + ("under each: oscillatory potentials (75-300 Hz, shaded = OP window); "
                            if with_op else "")
                         + "grey panels = Pig 1, exploratory", fontsize=9, y=1 - 0.08 / H)
            fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.75 / H), ncol=len(ll),
                       fontsize=8, frameon=False)
            fig.subplots_adjust(top=1 - 1.45 / H, bottom=0.4 / H, left=0.1, right=0.99)
            name = f"fig_waveforms_{kind}_flash{fl.replace('.', 'p')}.png"
            fig.savefig(outdir / name, dpi=150)
            plt.close(fig)
            made.append(name)
    return made


# ------------------------------------------------------------------ overlay figures
DUR_COLOR = {30: "#1b9e77", 60: "#d95f02", 120: "#7570b3", 0: "#666666"}
OVERLAY_SETS = {   # Jet and RM in separate pictures
    "Jet": [("Pre Jet", "Before cooling"), ("Intra Jet", "During cooling"), ("Post Jet", "After switch-off")],
    "RM": [("Pre RM", "Before device"), ("Post RM", "After removal")],
}


def overlay_figures(waves, sess_info, outdir, flashes=FLASHES, what="ERG", tag="", xlim=None,
                    ops=None, op_windows=None):
    """All Pre / all Intra / all Post traces overlaid: one figure per pig (sessions
    coloured by cooling duration) and one for all pigs (thin lines, bold mean per
    duration). Every flash. Traces start at 0 (ISCEV pre-stimulus
    baseline). what/tag/xlim: the same pictures for the OP traces (tag "ops_").
    ops/op_windows: with them, each phase row at flashes 25 and 8 gets an OP panel
    under it (same time axis, OP windows shaded). Pig 1 (exploratory) is drawn grey
    and dotted and left out of the means."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    t = EA.GRID
    made = []
    pigs_ = sorted({p for p, _ in sess_info})
    ops, op_windows = ops or {}, op_windows or {}

    def colour(pig, sess):
        if animal_set(pig) == "exploratory":
            return "0.55"
        return DUR_COLOR.get(sess_info[(pig, sess)]["minutes"], "#666666")

    def draw(ax, items, mean=False, op=False):
        by = {}
        for sess, pig, w, win in items:
            d = sess_info[(pig, sess)]["minutes"]
            expl = animal_set(pig) == "exploratory"
            col = colour(pig, sess)
            if op:
                _draw_op(ax, w, win, col, ls=":" if expl else "-", lw=0.5 if mean else 0.8,
                         shade=0.04 if mean else 0.07)
            else:
                ax.plot(t, w, color=col, lw=0.6 if mean else 1.0, ls=":" if expl else "-",
                        alpha=0.45 if mean else 0.95)
            if not expl:
                by.setdefault(d, []).append(w)
        if mean:
            for d, ws_ in by.items():
                if len(ws_) > 1:
                    ax.plot(t, np.mean(ws_, axis=0), color=DUR_COLOR.get(d), lw=2.2 if not op else 1.6)

    for fl in flashes:
        with_op = bool(ops) and fl in EA.OP_FLASHES
        scope = [(f"Pig {p}", [p], False) for p in pigs_] + [("All pigs", pigs_, True)]
        for (label, ps, mean), (el, rows_) in [(s, e) for s in scope for e in OVERLAY_SETS.items()]:
            rows = [(g, gl, k) for g, gl in rows_ for k in (("erg", "op") if with_op else ("erg",))]
            kinds = [r[2] for r in rows]
            h = sum(2.3 if k == "erg" else 2.3 * OP_PANEL_RATIO for k in kinds)
            fig = plt.figure(figsize=(9, h + 1.0))
            axes = _row_axes(fig, kinds, 2)
            for i, (g, gl, k) in enumerate(rows):
                for j, (eye, role) in enumerate(EYES):
                    ax = axes[i][j]
                    src = waves if k == "erg" else ops
                    items = [(s, p, src[(p, s, g, eye, fl)], op_windows.get((p, s, g, eye, fl)))
                             for (p, s) in sess_info if p in ps and (p, s, g, eye, fl) in src]
                    if len(ps) == 1 and animal_set(ps[0]) == "exploratory":
                        _expl(ax)
                    draw(ax, items, mean, op=(k == "op"))
                    if i == 0:
                        ax.set_title(role + " eye", fontsize=9)
                    if j == 0:
                        ax.set_ylabel(gl + "\n(µV)" if k == "erg" else "OPs\n(µV)",
                                      fontsize=8 if k == "erg" else 7)
                    if not items and k == "erg":
                        ax.text(0.5, 0.5, "no recordings", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8, color="0.5")
            _finish_axes(axes, kinds)
            if xlim:
                axes[-1][0].set_xlim(*xlim)
            handles = [Line2D([], [], color=DUR_COLOR[d], lw=2, label=f"{d} min" if d else "end of life")
                       for d in sorted({sess_info[k]["minutes"] for k in sess_info
                                        if k[0] in ps and animal_set(k[0]) == "primary"})]
            if any(animal_set(p) == "exploratory" for p in ps):
                handles.append(Line2D([], [], color="0.55", lw=1.2, ls=":",
                                      label="Pig 1 (exploratory" + (", not in the mean)" if mean else ")")))
            if mean:
                handles.append(Line2D([], [], color="k", lw=2.2, label="bold = mean per duration (primary set)"))
            if with_op:
                handles.append(Patch(color="0.5", alpha=0.25, label="OP window"))
            fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), fontsize=8,
                       frameon=False, bbox_to_anchor=(0.5, 1 - 0.45 / (h + 1.0)))
            ttl = label + (" (exploratory)" if len(ps) == 1 and animal_set(ps[0]) == "exploratory" else "")
            fig.suptitle(f"{ttl}: every {el} {what} overlaid by phase, flash {fl} cd·s/m²"
                         + ("; OPs (75-300 Hz) under each" if with_op else ""), fontsize=10, y=0.997)
            fig.subplots_adjust(top=1 - 1.1 / (h + 1.0), bottom=0.45 / (h + 1.0), left=0.1, right=0.98)
            name = f"fig_overlay_{tag}{label.replace(' ', '').lower()}_{el.lower()}_flash{fl.replace('.', 'p')}.png"
            fig.savefig(outdir / name, dpi=130)
            plt.close(fig)
            made.append(name)
    return made


# ------------------------------------------------------------------ session-type figures
GROUP_ORDER = ["Pre RM", "Pre Jet", "Intra Jet", "Post Jet", "Post RM"]
GROUP_LABEL = {"Pre RM": "Before device (RM)", "Pre Jet": "Before cooling (Jet)",
               "Intra Jet": "During cooling (Jet)", "Post Jet": "After switch-off (Jet)",
               "Post RM": "After removal (RM)"}


def sessiontype_figures(waves, sess_info, outdir, ops=None, op_windows=None, flashes=(PRIMARY, SECONDARY)):
    """One figure per session type (S1 30 min, S2 1 h, S3 2 h, S4 end of
    life) and flash: test and control eye overlaid at each instant, in time order
    (before device, before cooling, during cooling, after switch-off, after removal;
    end of life: its recordings only), one row per animal plus the primary-set mean
    (Pigs 2-6, +-1 SD shaded). The recording of each instant is the LAST one taken
    (erg_all.choose_by_group), never the best or an average. OP panel under each row
    at flashes 25 and 8. Pig 1 is drawn grey on a grey panel, labelled exploratory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ops, op_windows = ops or {}, op_windows or {}
    t = EA.GRID
    made = []
    for sno in sorted({i["no"] for i in sess_info.values() if i["no"]}):
        sess_of = {p: s for (p, s), i in sess_info.items() if i["no"] == sno}
        pigs = sorted(sess_of)
        eol = all(sess_info[(p, s)]["minutes"] == 0 for p, s in sess_of.items())
        mins = sorted({sess_info[(p, s)]["minutes"] for p, s in sess_of.items()})
        for fl in flashes:
            groups = [g for g in GROUP_ORDER
                      if any((p, sess_of[p], g, e, fl) in waves for p in pigs for e, _ in EYES)]
            if not groups:
                continue
            with_op = fl in EA.OP_FLASHES
            prim = [p for p in pigs if animal_set(p) == "primary"]
            row_ids = [("pig", p) for p in pigs] + ([("mean", None)] if len(prim) >= 2 else [])
            rows = [(rid, k) for rid in row_ids for k in (("erg", "op") if with_op else ("erg",))]
            kinds = [k for _r, k in rows]
            h = sum(1.9 if k == "erg" else 1.9 * OP_PANEL_RATIO for k in kinds)
            fig = plt.figure(figsize=(2.9 * len(groups) + 0.8, h + 1.3))
            axes = _row_axes(fig, kinds, len(groups))
            for ri, ((what, pig), k) in enumerate(rows):
                for ci, g in enumerate(groups):
                    ax = axes[ri][ci]
                    if ri == 0:
                        ax.set_title(("End of life" + (" (RM)" if "RM" in g else " (Jet)")) if eol
                                     else GROUP_LABEL[g], fontsize=9)
                    if what == "pig":
                        expl = animal_set(pig) == "exploratory"
                        if expl:
                            _expl(ax)
                        if ci == 0:
                            ax.set_ylabel((f"Pig {pig}" + (" (exploratory)" if expl else "") + "\n(µV)")
                                          if k == "erg" else "OPs\n(µV)", fontsize=7.5 if k == "erg" else 6.5)
                        any_ = False
                        for eye, role in EYES:
                            key = (pig, sess_of[pig], g, eye, fl)
                            col = (EXPL_TEST if eye == "RE" else EXPL_CTRL) if expl else \
                                (TEST_COL if eye == "RE" else CTRL_COL)
                            ls = "-" if eye == "RE" else "--"
                            if k == "erg" and key in waves:
                                ax.plot(t, waves[key], color=col, lw=1.0, ls=ls); any_ = True
                            elif k == "op" and key in ops:
                                _draw_op(ax, ops[key], op_windows.get(key), col, ls=ls); any_ = True
                        if not any_ and k == "erg":
                            ax.text(0.5, 0.5, "no recording", ha="center", va="center",
                                    transform=ax.transAxes, fontsize=7.5, color="0.5")
                    else:
                        if ci == 0:
                            ax.set_ylabel(f"Mean, {primary_pigs_label(prim)}\n(µV)" if k == "erg" else "OPs\n(µV)",
                                          fontsize=7.5 if k == "erg" else 6.5)
                        src = waves if k == "erg" else ops
                        ns = []
                        for eye, role in EYES:
                            arr = [src[(p, sess_of[p], g, eye, fl)] for p in prim
                                   if (p, sess_of[p], g, eye, fl) in src]
                            if not arr:
                                continue
                            A = np.array(arr)
                            col = TEST_COL if eye == "RE" else CTRL_COL
                            mu = A.mean(axis=0)
                            ax.plot(t, mu, color=col, lw=1.4, ls="-" if eye == "RE" else "--")
                            if len(A) > 1:
                                sd = A.std(axis=0, ddof=1)
                                ax.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
                            ns.append(f"{role[0]} n={len(A)}")
                        if k == "erg" and ns:
                            ax.text(0.98, 0.04, ", ".join(ns), ha="right", va="bottom",
                                    transform=ax.transAxes, fontsize=6.5, color="0.3")
            _finish_axes(axes, kinds)
            handles = [Line2D([], [], color=TEST_COL, lw=1.4, label="Test eye (right, cooled)"),
                       Line2D([], [], color=CTRL_COL, lw=1.4, ls="--", label="Control eye (left)")]
            if any(animal_set(p) == "exploratory" for p in pigs):
                handles.append(Line2D([], [], color=EXPL_TEST, lw=1.2, label="Pig 1 (exploratory): grey"))
            if with_op:
                handles.append(Patch(color="0.5", alpha=0.25, label="OP window (OPs 75-300 Hz)"))
            if len(prim) >= 2:
                handles.append(Patch(color="0.5", alpha=0.15, label="mean ± 1 SD (primary set)"))
            fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7.5, frameon=False,
                       bbox_to_anchor=(0.5, 1 - 0.55 / (h + 1.3)))
            stitle = "end of life" if eol else ", ".join(f"{m} min" for m in mins) + " cooling"
            fig.suptitle(f"Session {sno} ({stitle}): test vs control eye at each instant, flash {fl} cd·s/m²\n"
                         f"last recording of each instant (never best or averaged); raw record, 0.3 Hz "
                         f"high-pass, accepted sweeps averaged, 75 Hz low-pass", fontsize=9.5, y=0.998)
            fig.subplots_adjust(top=1 - 1.05 / (h + 1.3), bottom=0.45 / (h + 1.3), left=0.09, right=0.99)
            name = f"fig_sessiontype_S{sno}_flash{fl.replace('.', 'p')}.png"
            fig.savefig(outdir / name, dpi=140)
            plt.close(fig)
            made.append(name)
    return made


# ------------------------------------------------------------------ output
HDR_FILL = PatternFill("solid", fgColor="1F4E79")
WHITE_BOLD = Font(color="FFFFFF", bold=True)


def write_sheet(wb, title, rows, note=None, flag_col=None):
    """Formatted table (xlsx_style): readable headers, rounded values, number formats."""
    import xlsx_style as XS
    return XS.write_table(wb, title, rows, note, flag_col=flag_col)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def _stat(p):
    s = p.stat()
    return dict(path=str(p.relative_to(STUDY)), bytes=s.st_size,
                mtime=datetime.datetime.fromtimestamp(s.st_mtime).isoformat(timespec="seconds"))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=str(STUDY))
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args(argv)
    study = Path(a.root)
    outdir = study / OUTDIR_NAME
    outdir.mkdir(exist_ok=True)

    recs_out, by_key, sess_info, waves = collect(study)
    # every trace (ours, 75 Hz low-pass) to one file, so QC and plots need not re-read the raw records
    keys = sorted(waves)
    np.savez_compressed(outdir / "erg_traces.npz", grid=EA.GRID,
                        keys=np.array(["|".join(map(str, k)) for k in keys]),
                        ours=np.array([waves[k] for k in keys]),
                        # 75-300 Hz OP trace (average without the first flash); NaN at 0.06
                        ops=np.array([OP_WAVES.get(k, np.full_like(EA.GRID, np.nan)) for k in keys]),
                        op_window=np.array([OP_WINDOWS.get(k, (np.nan, np.nan)) for k in keys], float))
    eff = effects(by_key, sess_info)
    summ = cooling_summary(eff)
    rest = restoration(eff, recs_out)
    betw = between_sessions(by_key, sess_info, recs_out)
    meth = method_check(recs_out)
    # drifting baseline: median |pre-flash drift| of the kept sweeps > 3 x the cohort median
    dr = [r["Drift_uV_per_s"] for r in recs_out if r.get("Drift_uV_per_s") is not None]
    dmed = float(np.median(dr)) if dr else None
    for r in recs_out:
        d = r.get("Drift_uV_per_s")
        r["Drift_flag"] = ("drifting baseline" if dmed and d is not None and d > DRIFT_FLAG_X * dmed else "")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Readme"
    for i, line in enumerate(__doc__.strip().splitlines(), 1):
        ws.cell(i, 1, line)
    ws.column_dimensions["A"].width = 110
    write_sheet(wb, "Recordings", recs_out, "One row per pig x session x phase-electrode group x eye x flash; "
                "the last recording of each group (never averaged). Test = right eye (cooled).")
    write_sheet(wb, "Effects", eff, "d = later - earlier for each eye; Test_minus_Control = dTest - dControl. "
                "Amplitudes uV (a-wave as positive magnitude), times ms.")
    note = ("Per cooling duration across animals. dTest = test eye vs itself, dControl = control eye vs itself, "
            "T_minus_C = the two changes compared; each with mean, t-based 95% CI and paired t test. ")
    write_sheet(wb, "Summary excl Pig 1", [r for r in summ if r["Set"] == SET_PRIMARY], note + "Pig 1 EXCLUDED (exploratory).")
    write_sheet(wb, "Summary incl Pig 1", [r for r in summ if r["Set"] == SET_ALL], note + "Pig 1 INCLUDED.")
    write_sheet(wb, "Summary by group", [r for r in summ if r["Set"] not in (SET_PRIMARY, SET_ALL)], note +
                "Groups by delivered ring temperature (Set): about 9.5 C = Pigs 2-4 and Pig 5 S1 (old calibration), "
                "about 4 C = Pig 5 from S2 and Pig 6 (contact calibration).")
    write_sheet(wb, "Restoration", rest, f"Restored when |test eye's own change| <= {RESTORE_K:g} x SD of the "
                "control eye's change (pooled); Verdict_T_minus_C keeps the earlier test-minus-control rule. Eye check: each eye's own change against the control eyes' usual "
                "range (amplitudes as % of the eye's own earlier value, times in ms; robust: median +- max(2 x 1.4826 MAD, "
                "20 % / 3 ms)); both eyes out the same way = systemic cause suspected.",
                flag_col="Eye_check")
    write_sheet(wb, "Drift method check", meth, "Same recordings measured with the "
                "pre-flash line fit instead of the ISCEV 0.3 Hz high-pass (the method): line minus HP03. "
                "Sweeps: artefact rejection counts.")
    write_sheet(wb, "Between sessions", betw, "Baseline of each session minus that of session 1, with the "
                "core temperature at both baselines (a confound of this comparison). Role: Pre Jet (Jet electrode, "
                "before cooling) = PRIMARY; Pre RM (RM electrode, before installation) = CONFIRMATORY "
                "(2026-11-04: Jet primary, RM confirmatory).")
    figs = [] if a.no_figs else (sessiontype_figures(waves, sess_info, outdir, OP_WAVES, OP_WINDOWS)
                                 + figures(waves, sess_info, outdir, OP_WAVES, OP_WINDOWS)
                                 + overlay_figures(waves, sess_info, outdir, ops=OP_WAVES, op_windows=OP_WINDOWS)
                                 + overlay_figures(OP_WAVES, sess_info, outdir, flashes=EA.OP_FLASHES[::-1],
                                                   what="oscillatory potentials (75-300 Hz)", tag="ops_",
                                                   xlim=(0, 60)))
    if figs:
        import xlsx_style as XS
        fws = wb.create_sheet("Figures", 1)
        fws.sheet_view.showGridLines = False
        r = 1
        for name in ([f for f in figs if f.startswith("fig_sessiontype")] +
                     [f for f in figs if f.startswith("fig_overlay_allpigs")] +
                     [f for f in figs if f.startswith("fig_overlay_ops_allpigs")] +
                     [f for f in figs if f.startswith("fig_waveforms")] +
                     [f for f in figs if f.startswith("fig_overlay_pig")] +
                     [f for f in figs if f.startswith("fig_overlay_ops_pig")]):
            fws.cell(r, 1, name).font = XS.TITLE
            r += 1 + XS.add_image(fws, outdir / name, f"A{r + 1}", width_px=820)
    xlsx = outdir / "ERG Cohort Analysis.xlsx"
    wb.save(xlsx)
    csvs = []
    for name, rows in (("recordings", recs_out), ("effects", eff), ("cooling_summary", summ),
                       ("restoration", rest), ("between_sessions", betw), ("drift_method_check", meth)):
        p = outdir / f"erg_cohort_{name}.csv"
        write_csv(p, rows)
        csvs.append(p.name)

    inputs = []
    for pig_dir in sorted(study.glob("Pig *")):
        for p in sorted(list(pig_dir.glob("Session */*ERG Recording/*/*.xlsx"))
                        + list(pig_dir.glob("Session */*ERG Recording/*" + RIO.EOL_SUFFIX))):
            if not p.name.startswith("~$"):
                inputs.append(_stat(p))
        for p in sorted(pig_dir.glob("Session */*Systemic Parameters/*.xlsx")):
            if not p.name.startswith("~$") and "BACKUP" not in p.name:
                inputs.append(_stat(p))
        for p in (pig_dir / "Reference" / "erg_plan.csv", pig_dir / "Reference" / LAT_FILE):
            if p.exists():
                inputs.append(_stat(p))
    try:
        rev = subprocess.run(["git", "-C", str(TOOLS), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(TOOLS), "status", "--porcelain", "erg_cohort.py",
                                     "erg_all.py", "extract_erg.py"],
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        rev, dirty = "", None
    code_sha = hashlib.sha256((TOOLS / "erg_cohort.py").read_bytes()).hexdigest()[:16]
    manifest = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"),
                    script="_Tools/erg_cohort.py", git_head=rev, uncommitted_changes=dirty,
                    script_sha256_16=code_sha, python=sys.version.split()[0],
                    settings=dict(flashes=FLASHES, primary=PRIMARY, secondary=SECONDARY,
                                  temp_window_min=TEMP_WINDOW_MIN, restore_k=RESTORE_K,
                                  lowpass="75 Hz zero-phase (extract_erg.lowpass)",
                                  waveform=(f"{EA.WAVE_SOURCE}, drift={EA.DRIFT_METHOD} (pre-flash window "
                                            f"{EA.DETREND_PRE_MS:g} ms; alt {EA.ALT_DRIFT_METHOD} = "
                                            f"{EA.HP_HZ:g} Hz high-pass), accepted sweeps averaged"),
                                  artefact_rejection=(f"pre-flash noise > {EA.REJECT_NOISE_X:g} x recording median, "
                                                      f"or b-wave > {EA.REJECT_B_FRAC:.0%} from recording median"),
                                  recovery_min_after_off=RECOVERY_MIN_AFTER_OFF,
                                  oscillatory_potentials=(f"ISCEV 2022: band-pass {EA.OP_BAND[0]:g}-{EA.OP_BAND[1]:g} Hz "
                                                          f"(4th-order Butterworth, zero phase) on the raw average "
                                                          f"without the first flash; wavelets from the a-wave trough "
                                                          f"(>= {EA.OP_WIN[0]:g} ms) to b-wave peak + {EA.OP_AFTER_B_MS:g} ms; "
                                                          f"each from its preceding trough; OP_sum = OP1-{EA.OP_N}; "
                                                          f"flashes {', '.join(EA.OP_FLASHES)}"),
                                  choice="last recording per phase x electrode (erg_all.choose_by_group)",
                                  exploratory_pigs=sorted(EXPLORATORY_PIGS),
                                  animal_sets="primary = every other pig; Cooling summary per Set "
                                              "(primary first, then incl. exploratory); restoration pooled "
                                              "over the primary set (2026-10-28)"),
                    inputs=inputs, outputs=[xlsx.name] + csvs + figs)
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(recs_out)} recording rows, {len(eff)} effect rows -> {outdir}")
    for f in [xlsx.name] + csvs + figs:
        print("  ", f)


if __name__ == "__main__":
    main()
