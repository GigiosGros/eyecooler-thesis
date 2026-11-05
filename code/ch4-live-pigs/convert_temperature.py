#!/usr/bin/env python3
"""
Convert the WinDaq cooling recordings to calibrated current + temperature, tag
the cooling events, and write a per-session temperature summary that the master
workbook pulls in.

Recordings: the WinDaq xlsx export (*_Experiment_*.xlsx / *_Calibration_*.xlsx:
col A Relative Time (s, 10 Hz), col C Time stamp, col D Ch1, col E Ch2) or, when
there is no export, the raw .WDH itself (read_wdh). Two channels:
  * CURRENT (TEC drive): ~5 V  = 0 A,  low plateau = 3.9 A  (linear, no cal).
  * TEMPERATURE: converted with a PHYSICAL circuit model fitted to the session's
    calibration file, which holds 5 plateaus at known dry-well temps (4, 12.3, 20.5,
    28.7, 37 C; highest voltage = 4 C, ~0 V = 20.5 C, lowest = 37 C):
        V = (a*R + b) / (R + g),   R = R(T) of the 10K3A thermistor (datasheet
        Steinhart-Hart: 1/T = A + B*ln R + C*(ln R)^3, A 1.129241e-3, B 2.341077e-4,
        C 8.775468e-8, T in K)
    where V is the amplifier output, R the thermistor resistance, a and b the
    gain/offset terms and g the resistance the thermistor sees (divider). a, b, g are
    fitted per session, so a divider re-trimmed between sessions is absorbed; only
    the thermistor law is fixed. Inversion is exact: R = (b - g*V)/(V - a).
    (Default since 2026-10-28. The earlier "SH" model applied Steinhart-Hart to ln(V+shift),
    i.e. to the amplifier voltage, not to R; the voltage is nearly linear in T, so it
    left an S-shaped error of +0.1 to +0.5 degC at 4 degC. Kept as MODEL="SH".)
  Pigs 1-2 have current on Ch1 and temperature on Ch2; from Pig 3 the leads are
  swapped. assign_channels() tells them apart by signal span in every file.

Per session this writes:
  * 01 - Calibration / Temperature Calibration.xlsx   (5 plateau voltages -
        auto-detected, EDITABLE; fitted SH coefficients; per-point residuals)
  * 07 - Cooling / Temperature Events.xlsx            (auto-detected event times,
        EDITABLE 'Manual Time (s)' column overrides)
  * 07 - Cooling / Temperature Processed.csv          (time, current A, temp C, phase)
  * 07 - Cooling / Temperature Summary.xlsx           (metrics the master reads)

Pig 1 Session 1 has NO 5-plateau calibration (exception): its calibration sheet
is written blank/flagged for manual entry (or it borrows another session's fit
if you set 'Borrow From' in the sheet).

Usage (normally via  Pigs.ps1 temperature ):
    python convert_temperature.py [--root "<Live Pigs>"] [pig numbers ...]
"""
import sys, csv, argparse, re, datetime, struct
from pathlib import Path
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# the study root is the folder holding _Tools, so renaming the study folder
# (e.g. "Live Pigs" -> "Live Pigs Safety Study") cannot break the default
DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

CAL_TEMPS   = [4.0, 12.3, 20.5, 28.7, 37.0]   # dry-well set points (paired highest V -> 4 C)
NOMINAL_V   = [8.7, 4.1, 0.05, -3.5, -6.7]     # typical plateau volts per temp; used only
                                               # to assign detected plateaus to the right temp
CURRENT_V0  = 5.0     # Ch1 volts at 0 A
CURRENT_MAX = 3.9     # amps at the low plateau
CURR_ON_A   = 0.5     # amps above which the device counts as "on"
EQUIL_DTDT  = 0.02    # |dT/dt| (C/s) below which temperature counts as equilibrated
MODEL       = "PHYS"  # "PHYS" (V=(aR+b)/(R+g), R from the 10K3A law; default since 2026-10-28),
                      # "SH" (1/T=A+B*ln(V+s)+C*ln(V+s)^3, s=|Vmin|+1; legacy) or
                      # "cubic" (1/T=a+b*V+c*V^2+d*V^3, a raw-voltage polynomial)
TH_SH = (1.129241e-3, 2.341077e-4, 8.775468e-8)   # 10K3A datasheet Steinhart-Hart (R in ohm)


def thermistor_R(T_C):
    """R (ohm) of the 10K3A thermistor at T (degC), exact inverse of its S-H law."""
    A, B, C = TH_SH
    y = (A - 1.0 / (np.asarray(T_C, float) + 273.15)) / C
    x = np.sqrt((B / (3 * C)) ** 3 + y ** 2 / 4)
    return np.exp(np.cbrt(x - y / 2) - np.cbrt(x + y / 2))


def thermistor_T(R):
    """T (degC) from R (ohm) by the 10K3A S-H law; NaN where R <= 0."""
    A, B, C = TH_SH
    R = np.asarray(R, float)
    lr = np.log(np.where(R > 0, R, np.nan))
    return 1.0 / (A + B * lr + C * lr ** 3) - 273.15


def _phys_V(T_C, a, b, g):
    R = thermistor_R(T_C)
    return (a * R + b) / (R + g)

HDR = PatternFill("solid", fgColor="1F4E79"); WB_ = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True); CEN = Alignment(horizontal="center")


# --------------------------------------------------------------- io
RAW_EXT = (".xlsx", ".wdh")
ADC_FULL_SCALE_V = 9.99   # WinDaq +/-10 V range; |V| at or above this is clipped


def read_wdh(path):
    """Read a WinDaq .WDH recording directly, with no xlsx export needed.

    Layout: header length = uint16 at byte 6, data length in bytes = uint32 at
    byte 8, start time (UTC, Unix seconds) = uint32 at 0x24, sample interval (s) =
    float64 at 0x1C; samples are int16 interleaved by channel, volts = n*10/32768.
    Checked against WinDaq's own xlsx export of Pig 2 S2 (calibration and cooling):
    both channels and the per-second UTC stamps are identical."""
    b = Path(path).read_bytes()
    nch = (b[0] & 0x1F) or 2
    hdr = struct.unpack_from("<H", b, 6)[0]
    nbytes = struct.unpack_from("<I", b, 8)[0]
    dt = struct.unpack_from("<d", b, 0x1C)[0]
    t0 = struct.unpack_from("<I", b, 0x24)[0]
    n = nbytes // (2 * nch)
    d = np.frombuffer(b, dtype="<i2", count=n * nch, offset=hdr).reshape(n, nch)
    v = d.astype(float) * 10.0 / 32768.0
    rt = np.round(np.arange(n) * dt, 6)
    start = datetime.datetime.fromtimestamp(t0, datetime.timezone.utc)
    clk = [(start + datetime.timedelta(seconds=float(t))).strftime("%H:%M:%S") for t in rt]
    return rt, clk, v[:, 0], v[:, 1]


def assign_channels(c1, c2, name=""):
    """Return (current, temperature), whichever WinDaq channel each was wired to.

    Pigs 1-2 recorded current on Ch1 and temperature on Ch2; from Pig 3 the leads
    were swapped. The thermistor channel always has the much larger span (about
    -7..+9 V across a calibration, several volts across any cooling run), while the
    current channel sits at ~5 V when off and only drops to ~1.5 V when on, or is
    flat when dead. So the wider 1-99 % span is temperature. If the two spans are
    too close to call, keep the original wiring and say so."""
    def span(x):
        x = x[np.isfinite(x)]
        return float(np.subtract(*np.percentile(x, [99, 1]))) if len(x) else 0.0
    s1, s2 = span(c1), span(c2)
    if s1 > 1.5 * s2 and s1 > 2.0:
        print(f"    channels: {name}  Ch1 = temperature, Ch2 = current (swapped wiring; spans {s1:.1f} / {s2:.1f} V)")
        return c2, c1
    if not (s2 > 1.5 * s1 and s2 > 2.0):
        print(f"    ! channels: {name}  spans {s1:.1f} / {s2:.1f} V too close to tell; "
              f"assuming Ch1 = current, Ch2 = temperature")
    return c1, c2


def read_windaq(path):
    """Return reltime[s], clock[str], current[V], temperature[V] as numpy arrays.

    Reads a WinDaq xlsx export or the raw .WDH, then puts the channels in the order
    (current, temperature) whatever the wiring was (see assign_channels)."""
    if Path(path).suffix.lower() == ".wdh":
        rt, clk, c1, c2 = read_wdh(path)
    else:
        rt, clk, c1, c2 = _read_windaq_xlsx(path)
    cur, temp = assign_channels(c1, c2, Path(path).name)
    # A thermistor reading pinned at the WinDaq +/-10 V input limit is not a
    # temperature (probe unplugged or out of range). Converted, it reads as a fake
    # ~2.6 C that became Pig 3 S2's "min temp" and moved its Device Off to the
    # end of the file, so it is blanked like a gap in the recording.
    clipped = np.isfinite(temp) & (np.abs(temp) >= ADC_FULL_SCALE_V)
    if clipped.any():
        temp = np.where(clipped, np.nan, temp)
        print(f"    {Path(path).name}: {int(clipped.sum())} temperature samples at the "
              f"{ADC_FULL_SCALE_V:g} V input limit left blank")
    return rt, clk, cur, temp


def raw_recordings(folder):
    """WinDaq recordings in a step folder: every xlsx export, plus every .WDH that
    has no export beside it, so an exported session is read exactly as before and a
    .WDH-only session needs no manual export. An export counts as the .WDH's when
    its stem is the .WDH stem, optionally plus the renamer's "_NN" counter (Pig 1 S1:
    "..._Experiment_LM_260706.WDH" + "..._Experiment_LM_260706_01.xlsx")."""
    if folder is None or not folder.exists():
        return []
    files = [f for f in sorted(folder.iterdir())
             if f.suffix.lower() in RAW_EXT and not f.name.startswith("~$")
             and not f.name.startswith("Temperature ")]
    xstems = [f.stem.lower() for f in files if f.suffix.lower() == ".xlsx"]

    def exported(wdh):
        s = wdh.stem.lower()
        return any(x == s or re.fullmatch(re.escape(s) + r"_\d{1,3}", x) for x in xstems)

    return [f for f in files if f.suffix.lower() == ".xlsx" or not exported(f)]


def _read_windaq_xlsx(path):
    """Return reltime[s], clock[str], ch1[V], ch2[V] from a WinDaq xlsx export."""
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.worksheets[0]
    rt, clk, c1, c2 = [], [], [], []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        try:
            _rt, _c1, _c2 = float(row[0]), float(row[3]), float(row[4])
        except (TypeError, ValueError):
            continue
        # append all four together so a row that only partially parses can never
        # desync the columns (a few malformed Ch2 cells otherwise shift rt/ch1
        # ahead of ch2/clk and crash np.gradient downstream)
        rt.append(_rt); c1.append(_c1); c2.append(_c2); clk.append(row[2])
    wb.close()
    return np.array(rt), clk, np.array(c1), np.array(c2)


def parse_clock(c):
    try:
        h, m, s = str(c).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return None


def sec_to_clock(s):
    s = int(round(s)) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def read_experiment_concat(cool_folder):
    """Stitch every WinDaq recording in a Cooling folder into one continuous,
    CLOCK-ACCURATE trace ordered by wall-clock start. When there is a real time
    gap between files (e.g. Pig 1 S1, saved as 'Experiment' + 'Experiment-
    ReWarming' before the file structure was finalized), the missing interval is
    filled with blank (NaN) Ch1/Ch2 samples so the time axis stays true instead
    of being spliced contiguous. Returns ((rt, clk, ch1, ch2), [files]) or (None, [])."""
    files = [f for f in raw_recordings(cool_folder) if "experiment" in f.name.lower()]
    if not files:
        return None, []
    parts = []
    for f in files:
        rt, clk, c1, c2 = read_windaq(f)
        if len(rt) == 0:
            continue
        pc = parse_clock(clk[0]) if len(clk) else None
        parts.append((pc, "rewarm" in f.name.lower(), f.name, rt, clk, c1, c2))
    if not parts:
        return None, []
    parts.sort(key=lambda p: (p[0] if p[0] is not None else 1e18, p[1], p[2]))
    base = parts[0][0]
    have_clock = base is not None
    dt = float(np.median(np.diff(parts[0][3]))) if len(parts[0][3]) > 1 else 0.1

    RT, CLK, C1, C2, used = [], [], [], [], []
    running_end = 0.0
    for k, (pc, _, name, rt, clk, c1, c2) in enumerate(parts):
        if k == 0:
            seg_off = 0.0
        elif have_clock and pc is not None and (pc - base) > running_end:
            seg_off = pc - base                       # true clock position
        else:
            seg_off = running_end + dt                # no/overlapping clock -> contiguous
        # fill the gap before this segment with blank samples
        gap_start = running_end + dt
        if seg_off > gap_start + dt / 2:
            fill_t = np.arange(gap_start, seg_off, dt)
            if len(fill_t):
                RT.append(fill_t)
                C1.append(np.full(len(fill_t), np.nan)); C2.append(np.full(len(fill_t), np.nan))
                CLK += [sec_to_clock(base + t) if have_clock else "" for t in fill_t]
        gt = (rt - rt[0]) + seg_off
        RT.append(gt); C1.append(c1); C2.append(c2); CLK += list(clk)
        running_end = gt[-1]; used.append(name)
    return (np.concatenate(RT), CLK, np.concatenate(C1), np.concatenate(C2)), used


def find_file(folder, keyword):
    """First WinDaq recording (xlsx export, else .WDH) whose name contains keyword
    (skip lock/rewarming files)."""
    if folder is None or not folder.exists():
        return None
    raw = raw_recordings(folder)
    cands = [f for f in raw if keyword.lower() in f.name.lower()]
    # prefer the plain experiment over the ReWarming file, and an export over a .WDH
    cands.sort(key=lambda f: ("rewarm" in f.name.lower(), f.suffix.lower() != ".xlsx", f.name))
    if cands:
        return cands[0]
    return raw[0] if raw else None


def find_sub(sess_path, suffix):
    pat = re.compile(r"^\d+\s*-\s*" + re.escape(suffix) + r"$", re.I)
    for d in sorted(sess_path.iterdir()):
        if d.is_dir() and pat.match(d.name):
            return d
    return None


# --------------------------------------------------------------- calibration
def detect_plateaus(vals, n=5, minsep=1.0):
    """Voltages of the n densest, well-separated plateaus, sorted descending.
    Pads the histogram so plateaus at the extreme temps (histogram edges) are
    still found."""
    lo, hi = np.percentile(vals, 0.1), np.percentile(vals, 99.9)
    if hi - lo < 1e-6:
        return []
    hist, edges = np.histogram(vals, bins=400, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    sm = gaussian_filter1d(hist.astype(float), 1.5)
    binw = centers[1] - centers[0]
    dist = max(1, int(minsep / binw))
    smp = np.concatenate([[0.0], sm, [0.0]])          # pad so edge peaks register
    pk, props = find_peaks(smp, distance=dist, prominence=len(vals) * 0.001)
    pk = pk - 1
    keep = (pk >= 0) & (pk < len(centers))
    pk = pk[keep]; prom = props["prominences"][keep]
    order = np.argsort(prom)[::-1]
    chosen = []
    for idx in pk[order]:
        c = centers[idx]
        if all(abs(c - x) >= minsep for x in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return sorted(chosen, reverse=True)


def assign_to_temps(detected):
    """Map detected plateau voltages to the 5 known temps by nearest NOMINAL_V
    (greedy, one-to-one). Returns list len 5 (voltage or None per temp), so it
    works when only 3-4 plateaus were recorded (e.g. Pig 1 S1 lacks 12.3 & 28.7)."""
    slots = [None] * len(CAL_TEMPS)
    used = set()
    for v in sorted(detected, key=lambda x: -x):          # highest V first
        order = sorted(range(len(NOMINAL_V)), key=lambda i: abs(NOMINAL_V[i] - v))
        for i in order:
            if i not in used:
                slots[i] = float(v); used.add(i); break
    return slots


SETTLE_WINDOW_S = 180.0     # my protocol: ~7 min to settle, the last 3 min are the value
SETTLE_DROP_S = 10.0        # ignore the last 10 s before the ramp (departure)
SETTLE_DRIFT_V_MIN = 0.005  # last 3 min still moving faster than this -> stable-band fallback
STABLE_BAND_V = 0.01        # stable band: 10-s medians all within +-0.01 V


def settle_plateaus(t, dt, approx):
    """Settled voltage of each plateau, located by the histogram value in `approx`.

    The histogram peak (detect_plateaus) only LOCATES a plateau: it pools the whole
    file, so at the 4 C edge it is pulled toward the well's overshoot (up to ~0.05 V,
    ~0.1 C; 2026-11-04 comparison). The value used is the median of the last 3 min
    before the ramp to the next temperature (7 + 3 min protocol). If those
    3 min still drift by more than 5 mV/min, the plateau had not settled and the
    median of the longest stretch whose 10-s medians all stay within +-0.01 V is
    used instead. The plateau is the longest run of samples within 0.3 V of the
    histogram value (gaps < 5 s bridged). Returns (values, methods)."""
    vals, methods = [], []
    w10 = max(1, int(round(10.0 / dt)))
    for v0 in approx:
        if v0 is None:
            vals.append(None); methods.append(None); continue
        idx = np.where(np.isfinite(t) & (np.abs(t - v0) < 0.3))[0]
        if len(idx) < w10 * 6:
            vals.append(v0); methods.append("histogram (plateau < 1 min)"); continue
        runs = np.split(idx, np.where(np.diff(idx) > int(5.0 / dt))[0] + 1)
        run = max(runs, key=len)
        seg = t[run[0]:run[-1] + 1]
        end = len(seg) - int(SETTLE_DROP_S / dt)
        last = seg[max(0, end - int(SETTLE_WINDOW_S / dt)):end]
        last = last[np.isfinite(last)]
        dur_min = len(seg) * dt / 60.0
        drift = (float(np.polyfit(np.arange(len(last)) * dt / 60.0, last, 1)[0])
                 if len(last) > 10 else float("inf"))
        if abs(drift) <= SETTLE_DRIFT_V_MIN:
            vals.append(float(np.median(last)))
            methods.append(f"last 3 min (drift {drift*1000:+.1f} mV/min, plateau {dur_min:.1f} min)")
            continue
        nb = len(seg) // w10
        m = np.array([np.nanmedian(seg[k * w10:(k + 1) * w10]) for k in range(nb)])
        best = (0, 0, 0)
        for s0 in range(nb):
            lo = hi = m[s0]; e = s0
            while e + 1 < nb and max(hi, m[e + 1]) - min(lo, m[e + 1]) <= 2 * STABLE_BAND_V:
                e += 1; lo = min(lo, m[e]); hi = max(hi, m[e])
            if e - s0 + 1 > best[0]:
                best = (e - s0 + 1, s0, e)
        n, s0, e = best
        vals.append(float(np.nanmedian(seg[s0 * w10:(e + 1) * w10])))
        methods.append(f"stable band +-{STABLE_BAND_V:g} V, {n*10/60:.1f} min "
                       f"(last 3 min drifted {drift*1000:+.1f} mV/min)")
    return vals, methods


def sh_shift(V):
    """Steinhart-Hart voltage shift: s = |min(V)| + 1, so the coldest (most
    negative-voltage) plateau maps to ln(V+s)=0 and the cubic term never flips
    sign inside the calibrated range."""
    return float(abs(np.min(V)) + 1.0)


def fit_calibration(volts, temps, model=MODEL):
    """Least-squares fit of 1/T(K) over the PRESENT (volt, temp) pairs.
    SH fits in ln(V+shift) (a proxy for ln R — the true Steinhart-Hart variable);
    cubic fits a raw-voltage polynomial. SH needs >=3 points; cubic needs >=5.
    Returns (coef, pred_C, rmse, shift) or None (shift is None for cubic)."""
    pairs = [(v, t) for v, t in zip(volts, temps) if v is not None]
    if (model == "cubic" and len(pairs) < 5) or len(pairs) < 3:
        return None
    V = np.array([p[0] for p in pairs]); T = np.array([p[1] for p in pairs]); TK = T + 273.15
    if model == "PHYS":
        from scipy.optimize import least_squares
        R = thermistor_R(T)
        # start from the algebraic solution of V*R = a*R + b - g*V (linear in a, b, g),
        # then minimise the TEMPERATURE residual
        x0, *_ = np.linalg.lstsq(np.column_stack([R, np.ones_like(R), -V]), V * R, rcond=None)
        res = least_squares(lambda c: volts_to_C(V, c, "PHYS") - T, x0, x_scale="jac",
                            max_nfev=20000)
        coef = res.x
        pred = volts_to_C(V, coef, "PHYS")
        if not np.all(np.isfinite(pred)):
            return None
        rmse = float(np.sqrt(np.mean((pred - T) ** 2)))
        return coef, {t: p for (v, t), p in zip(pairs, pred)}, rmse, None
    if model == "cubic":
        shift = None
        X = np.column_stack([np.ones_like(V), V, V**2, V**3])
    else:  # SH: 1/T = A + B*ln(V+s) + C*ln(V+s)^3
        shift = sh_shift(V)
        u = np.log(V + shift)
        X = np.column_stack([np.ones_like(u), u, u**3])
    coef, *_ = np.linalg.lstsq(X, 1.0 / TK, rcond=None)
    pred = volts_to_C(V, coef, model, shift)
    rmse = float(np.sqrt(np.nanmean((pred - T) ** 2)))
    pred_by_temp = {t: p for (v, t), p in zip(pairs, pred)}
    return coef, pred_by_temp, rmse, shift


def volts_to_C(volts, coef, model=MODEL, shift=None):
    v = np.asarray(volts, float)
    if model == "PHYS":
        a, b, g = coef
        with np.errstate(divide="ignore", invalid="ignore"):
            return thermistor_T((b - g * v) / (v - a))
    if model == "cubic":
        return 1.0 / (np.column_stack([np.ones_like(v), v, v**2, v**3]) @ coef) - 273.15
    # SH: fit lives in ln(V+shift) as a stand-in for ln(R). Clip V+shift > 0 so
    # samples hotter than the calibrated range yield NaN instead of a garbage °C.
    u = np.log(np.clip(v + shift, 1e-12, None))
    inv_TK = coef[0] + coef[1] * u + coef[2] * u**3
    inv_TK = np.where(inv_TK > 0, inv_TK, np.nan)
    return 1.0 / inv_TK - 273.15


def model_dTdV(V, coef, model=MODEL, shift=None):
    """|dT/dV| (°C per volt) of the fitted curve at voltage V, for turning a
    voltage SD into a temperature error."""
    V = float(V)
    if model == "PHYS":
        h = 1e-3
        d = volts_to_C([V - h, V + h], coef, "PHYS")
        return float(abs(d[1] - d[0]) / (2 * h)) if np.all(np.isfinite(d)) else None
    if model == "cubic":
        f = coef[0] + coef[1]*V + coef[2]*V**2 + coef[3]*V**3
        fp = coef[1] + 2*coef[2]*V + 3*coef[3]*V**2
    else:
        w = max(V + shift, 1e-12); u = np.log(w)
        f = coef[0] + coef[1]*u + coef[2]*u**3
        fp = (coef[1] + 3*coef[2]*u**2) / w        # chain rule: du/dV = 1/(V+s)
    if abs(f) < 1e-12:
        return None
    return abs(fp) / (f * f)          # dT = -f'/f^2, magnitude same in K and °C


def plateau_sd(ch2, V, win=0.25):
    """SD (volts) of the calibration samples sitting on the plateau at voltage V."""
    seg = ch2[np.abs(ch2 - V) < win]
    return float(np.std(seg)) if len(seg) > 5 else None


def write_calibration_sheet(path, autoslots, temps, coef, pred_by_temp, rmse, model,
                            plateaus=None, sds=None, note="", shift=None, manual=None, methods=None):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Calibration"
    ws["A1"] = "Temperature Calibration"; ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = f"Model: {model} (" + {
        "PHYS": "V = (a*R + b)/(R + g), R = 10K3A thermistor law (physical circuit model)",
        "SH": "1/T = A+B*ln(V+s)+C*ln(V+s)^3, Steinhart-Hart in volts (legacy)",
        "cubic": "1/T = a+b*V+c*V^2+d*V^3"}[model] + ")"
    ws["A3"] = f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}"
    npts = sum(1 for p in autoslots if p is not None)
    ws["A4"] = f"Plateaus used: {npts}   (>=3 fits; {max(0, npts-3)} spare point(s) for validation)"
    ws["A5"] = ("'Fitted Temp' = what the curve predicts at that plateau voltage; it differs "
                "from the known temp by the least-squares 'Residual' (0 when points <= 3 coeffs). "
                "'Plateau SD' is the sensor noise on that plateau; '± Noise' converts it to °C "
                "via the local curve slope.")
    ws["A5"].font = Font(italic=True, color="808080")
    if note:
        ws["A6"] = note; ws["A6"].font = Font(italic=True, color="C00000")
    hdr = ["Known Temp (°C)", "Plateau Voltage (auto)", "Plateau Voltage (manual)",
           "Plateau SD (V)", "± Noise (°C)", "Fitted Temp (°C)", "Residual (°C)", "Auto method"]
    r0 = 8
    for c, h in enumerate(hdr, 1):
        cell = ws.cell(r0, c, h); cell.fill = HDR; cell.font = WB_; cell.alignment = CEN
    for i, t in enumerate(temps):
        r = r0 + 1 + i
        ws.cell(r, 1, t)
        ws.cell(r, 2, round(autoslots[i], 4) if autoslots[i] is not None else None)
        if manual and manual[i] is not None:
            ws.cell(r, 3, manual[i])        # the user's override survives the rewrite
        sd = sds[i] if sds else None
        ws.cell(r, 4, round(sd, 5) if sd is not None else None)
        if sd is not None and coef is not None and plateaus and plateaus[i] is not None:
            slope = model_dTdV(plateaus[i], coef, model, shift)
            ws.cell(r, 5, round(sd * slope, 4) if slope is not None else None)
        pv = pred_by_temp.get(t) if pred_by_temp else None
        ws.cell(r, 6, round(float(pv), 3) if pv is not None else None)
        ws.cell(r, 7, round(float(pv - t), 3) if pv is not None else None)
        if methods and methods[i]:
            ws.cell(r, 8, methods[i])
    rr = r0 + 2 + len(temps)
    ws.cell(rr, 1, "RMSE (°C)").font = BOLD; ws.cell(rr, 2, round(rmse, 4) if rmse is not None else None)
    if coef is not None:
        names = {"SH": ["A", "B", "C"], "PHYS": ["a (V)", "b (V*ohm)", "g (ohm)"]}.get(model, ["a", "b", "c", "d"])
        for i, nm in enumerate(names):
            ws.cell(rr + 1 + i, 1, f"coef {nm}").font = BOLD
            ws.cell(rr + 1 + i, 2, float(coef[i]))
        if model == "SH" and shift is not None:
            ws.cell(rr + 1 + len(names), 1, "shift s (V)").font = BOLD
            ws.cell(rr + 1 + len(names), 2, float(shift))
    ws.cell(rr + 7, 1, "Leave a temp's voltage blank if that plateau was not recorded; "
                       "edit 'manual' to override auto-detection, then re-run.").font = Font(italic=True, color="808080")
    for col, w in zip("ABCDEFGH", (18, 22, 22, 14, 14, 16, 14, 60)):
        ws.column_dimensions[col].width = w
    wb.save(path)


def read_manual_overrides(path):
    """Return list of manual plateau voltages (len 5, None where blank) if the
    sheet exists and the user filled the manual column."""
    if not path.exists():
        return None
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb["Calibration"] if "Calibration" in wb.sheetnames else wb.active
    except Exception as e:
        # an unreadable sheet would be rewritten without the user's manual voltages
        raise RuntimeError(f"cannot read {path.name} ({e}); close it in Excel / let Box "
                           f"finish syncing, then re-run. Nothing was rewritten.")
    out = []
    for i in range(len(CAL_TEMPS)):
        v = ws.cell(9 + i, 3).value  # manual column C, data rows 9..13
        out.append(float(v) if isinstance(v, (int, float)) else None)
    return out if any(x is not None for x in out) else None


# Sessions calibrated with the OLD method (thermistor loose in the well, not touching the
# aluminum). Contact calibration began with Pig 6 S1 (2026-10-27) and Pig 5 S2.
NO_CONTACT_SESSIONS = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3),
                       (4, 1), (4, 2), (4, 3), (5, 1)}
RING_OFFSET_NOTE = (
    "OLD-METHOD CALIBRATION: the calibration thermistor sat loose in the dry well, not in contact "
    "with the aluminum, so self-heating and poor contact made it read warmer than the well. Ring "
    "temperatures from this calibration read about 5.5 C LOW: true ring ~ reading + 5.5 C "
    "(RING_OFFSET_C; same-day double calibration 2026-11-02: 5.5-5.6 C low from 2 to 20 C, 5.2 C "
    "at 35 C). The session's temperatures ARE corrected (+5.5 C) and are the values to use; the "
    "uncorrected reading is kept for the record only ('Temp uncorrected' column of Temperature "
    "Processed.csv, '[uncorrected]' rows of Temperature Summary.xlsx).")
RING_OFFSET_C = 5.5   # added to every temperature of a NO_CONTACT session (from 2026-11-04)


def session_offset(sess_path):
    """Calibration offset (degC) to add to this session's ring temperatures."""
    m = re.search(r"Pig\s*(\d+).*?Session\s*(\d)", str(Path(sess_path).resolve()).replace("\\", "/"))
    if not m:
        return 0.0
    return RING_OFFSET_C if (int(m.group(1)), int(m.group(2))) in NO_CONTACT_SESSIONS else 0.0
SESSION_CAL_NOTES = {
    (5, 3): ("The upper leg of the comparator divider was moved by accident during this "
             "calibration (4 C plateau, 15:00-15:06 UTC on the WinDaq clock) and put back; the "
             "calibration was repeated upward (12.3 to 37 C) for accuracy. The manual column holds "
             "the rerun plateaus (2026-11-04). First pass and rerun differ by 0.03 V or less "
             "(0.02 C or less on the cooling temperatures). The calibration preceded the cooling run."),
}


def session_cal_note(cal_folder):
    """Fixed explanatory note for a session's calibration sheet ('' if none)."""
    m = re.search(r"Pig\s*(\d+).*?Session\s*(\d)", str(Path(cal_folder).resolve()).replace("\\", "/"))
    if not m:
        return ""
    key = (int(m.group(1)), int(m.group(2)))
    parts = [SESSION_CAL_NOTES.get(key, "")]
    if key in NO_CONTACT_SESSIONS:
        parts.append(RING_OFFSET_NOTE)
    return "  ".join(x for x in parts if x)


def calibrate_session(cal_folder):
    """Build/refresh the calibration sheet and return (coef, rmse, model) or None."""
    cal_file = find_file(cal_folder, "Calibration")
    if not cal_file:
        return None
    rt, _, _, ch2_full = read_windaq(cal_file)
    ch2 = ch2_full[np.isfinite(ch2_full)]   # histogram: plateau voltages only; drop clipped samples
    located = assign_to_temps(detect_plateaus(ch2, n=len(CAL_TEMPS)))
    dt = float(rt[1] - rt[0]) if len(rt) > 1 else 0.1
    autoslots, methods = settle_plateaus(ch2_full, dt, located)
    sheet = cal_folder / "Temperature Calibration.xlsx"
    manual = read_manual_overrides(sheet)
    # Auto stays auto, manual stays: writing the merged voltages into the AUTO column
    # and blanking the manual one would make the next run go back to the automatic value.

    # per-temp voltage: manual override wins, else auto-detected
    plateaus = list(autoslots)
    if manual:
        plateaus = [manual[i] if manual[i] is not None else plateaus[i] for i in range(len(CAL_TEMPS))]
    sds = [plateau_sd(ch2, v) if v is not None else None for v in plateaus]

    fit = fit_calibration(plateaus, CAL_TEMPS)
    if fit is not None:
        coef, pred_by_temp, rmse, shift = fit
        write_calibration_sheet(sheet, autoslots, CAL_TEMPS, coef, pred_by_temp, rmse, MODEL,
                                plateaus=plateaus, sds=sds, shift=shift, manual=manual, methods=methods,
                                note=session_cal_note(cal_folder))
        # target = what the sensor reads at the 4°C calibration plateau (the relay switch point)
        v4 = plateaus[CAL_TEMPS.index(4.0)] if 4.0 in CAL_TEMPS else None
        target_temp = float(volts_to_C([v4], coef, MODEL, shift)[0]) if v4 is not None else None
        return coef, rmse, MODEL, target_temp, shift
    else:
        note = ("Fewer than 3 plateaus found. Enter at least 3 plateau voltages "
                "(known temp <-> voltage) in the manual column and re-run.")
        write_calibration_sheet(sheet, autoslots, CAL_TEMPS, None, None, None, MODEL,
                                plateaus=plateaus, sds=sds, note=note, manual=manual)
        return None


# --------------------------------------------------------------- conversion + events
CURRENT_DEAD_PTP_V = 0.2   # a working channel swings ~5 V between 0 A and full current


def ch1_to_amps(ch1):
    finite = ch1[np.isfinite(ch1)]
    if len(finite) == 0:
        return np.full_like(ch1, np.nan)
    # A channel that does not move recorded nothing: Pig 2's channel 1 is exactly 0 V
    # in every sample of all three sessions, which the scaling below turned into a
    # constant 3.900 A presented as a measurement. No current.
    # The span is taken between the 0.5 and 99.5 % points, not peak to peak: Pig 3
    # S3's channel sits at 5.06 V throughout (device never switched on) apart from a
    # few switching spikes, and one spike was enough to report "Max current 3.9 A".
    if float(np.subtract(*np.percentile(finite, [99.5, 0.5]))) < CURRENT_DEAD_PTP_V:
        print(f"    ! current channel is flat ({finite.min():.3f} to {finite.max():.3f} V): "
              f"not recorded, current left blank")
        return np.full_like(ch1, np.nan)
    med = np.median(finite)
    hi_vals = finite[finite > med]
    hi = np.median(hi_vals) if len(hi_vals) else CURRENT_V0
    v0 = max(hi, CURRENT_V0 * 0.9)
    mid = (finite.max() + finite.min()) / 2
    low = finite[finite < mid]
    vfull = np.median(low) if len(low) else finite.min()
    if abs(v0 - vfull) < 1e-6:
        return np.where(np.isfinite(ch1), 0.0, np.nan)
    amps = CURRENT_MAX * (v0 - ch1) / (v0 - vfull)   # NaN stays NaN in gaps
    return np.clip(amps, 0, CURRENT_MAX)


def smooth(y, dt_s, seconds=20.0):
    """Centred moving average over `seconds`. Each output is the mean of the finite
    samples inside its window, so the first and last half-window are averaged over
    the samples that exist instead of being pulled toward zero (np.convolve's zero
    padding halved the last 10 s of a trace: Pig 1 S3 read 4.41 for 8.74 degC, and
    Pig 2 S3's automatic Device Off survived by 0.055 degC)."""
    y = np.asarray(y, float)
    w = max(1, int(seconds / dt_s))
    if w <= 1:
        return y
    ok = np.isfinite(y)
    k = np.ones(w)
    num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    out = np.full_like(y, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    if not ok.any():
        return np.zeros_like(y)
    # a gap longer than the window keeps the old behaviour (filled with the mean)
    out[den == 0] = float(np.nanmean(y))
    return out


def _first_sustained(rt, slope, i0, i1, thr, dt_s, hold_s=20.0):
    """First time in [i0,i1) where slope stays past thr (sign of thr) for hold_s."""
    i1 = min(i1, len(slope))
    if i1 - i0 < 2:
        return None
    hold = max(1, int(hold_s / dt_s))
    cond = slope > thr if thr > 0 else slope < thr
    run = 0; start = i0
    for k in range(i0, i1):
        if cond[k]:
            if run == 0:
                start = k
            run += 1
            if run >= hold:
                return float(rt[start])
        else:
            run = 0
    return None


# Temperature-only restart detection (used when the current channel is unusable).
# The probe sits ON the device, so an interruption shows as the probe leaving the
# cold plateau and warming, then falling back once power returns.
RESTART_RISE_C = 1.0      # how far above the cold plateau counts as "not cooling"
RESTART_MIN_S = 8.0       # ignore single-sample spikes / probe handling blips
RESTART_MERGE_S = 120.0   # bumps closer than this are ONE interruption, not two


def regulation_start(rt, amps, temp, tsm, cs, do, dt_s):
    """When the relay starts cycling, i.e. cooling switches from pull-down to hold.

    Preferred evidence is the CURRENT: REG_SWITCHES on/off transitions inside
    REG_WINDOW_S. Where the current carries no timing (Pig 2's clamp logs a constant
    3.900 A), fall back to the temperature and take the end of the pull-down - the
    first moment the smoothed slope stops falling faster than REG_FALL_C_PER_MIN and
    stays that way for REG_HOLD_S."""
    if cs is None or do is None or not len(rt):
        return None
    i_cs, i_do = int(np.searchsorted(rt, cs)), int(np.searchsorted(rt, do))
    if i_do - i_cs < 2:
        return None

    on = np.isfinite(amps) & (amps > CURR_ON_A)
    if on.all():                      # never-off channel carries no timing
        on = np.zeros_like(on)
    if on.any():
        edges = np.where(np.diff(on[i_cs:i_do].astype(int)) != 0)[0] + i_cs
        need = REG_SWITCHES
        for j in range(len(edges) - need + 1):
            if rt[edges[j + need - 1]] - rt[edges[j]] <= REG_WINDOW_S:
                return float(rt[edges[j]])
        return None

    if not np.isfinite(temp).any():
        return None
    heavy = smooth(temp, dt_s, REG_SLOPE_SMOOTH_S)
    settled = (rt >= cs + REG_LEVEL_FROM * (do - cs)) & (rt <= do) & np.isfinite(heavy)
    if not settled.any():
        return None
    level = float(np.median(heavy[settled]))
    hit = np.where((rt >= cs) & (rt <= do) & (heavy <= level + REG_LEVEL_TOL_C))[0]
    return float(rt[hit[0]]) if len(hit) else None


def temp_restarts(rt, amps, temp, cs, do):
    """Restart (off, back-on) pairs read from the TEMPERATURE, for recordings whose
    current channel carries no timing (Pig 2's clamp logged a constant 3.900 A in
    every session, so `on.any()` is False and the current-based detector in
    detect_events can never fire). A restart = a sustained warm excursion above the
    cold plateau, inside the cooling window and after the initial pull-down.

    A failing relay chatters rather than switching once, so excursions less than
    RESTART_MERGE_S apart are merged: Pig 2 S3's relay breakdown produced an 11 s
    bump, a cold overshoot, then a 36 s bump - one interruption, not two.

    Returns [] when the current channel IS usable: there its off-gaps are the better
    evidence and this must not second-guess them (Pig 1 is unaffected).

    Call this with the EFFECTIVE (manual-override-resolved) Cooling Start / Device
    Off. Against the AUTO times it reads the post-cooling rewarm of a session whose
    Device Off was corrected by hand as one long restart (Pig 2 S1 and S2)."""
    if cs is None or do is None or not len(rt):
        return []
    on = np.isfinite(amps) & (amps > CURR_ON_A)
    if on.all():                    # never-off channel carries no timing (see above)
        on = np.zeros_like(on)
    if on.any() or not np.isfinite(temp).any():
        return []
    dt_s = float(np.median(np.diff(rt))) or 0.1
    tsm = smooth(temp, dt_s, 20)
    i_cs = int(np.searchsorted(rt, cs))
    i_do = int(np.searchsorted(rt, do))
    if i_do - i_cs < 2:
        return []
    seg = tsm[i_cs:i_do + 1]
    finite = seg[np.isfinite(seg)]
    if not len(finite):
        return []
    # Plateau reference = the cold part of the window, so the pull-down and any
    # excursions don't drag it up.
    cold = finite[finite < np.nanmin(finite) + 2.5]
    if not len(cold):
        return []
    ref = float(np.median(cold)) + RESTART_RISE_C

    warm = np.isfinite(seg) & (seg > ref)
    # Skip the initial pull-down: start only once the probe first reaches plateau.
    settled = np.where(~warm)[0]
    if not len(settled):
        return []
    warm[:settled[0]] = False

    idx = np.where(warm)[0]
    if not len(idx):
        return []
    merge_n = max(1, int(RESTART_MERGE_S / dt_s))
    groups = np.split(idx, np.where(np.diff(idx) > merge_n)[0] + 1)
    out = []
    for g in groups:
        if (g[-1] - g[0]) * dt_s < RESTART_MIN_S:
            continue
        out.append((float(rt[i_cs + g[0]]), float(rt[i_cs + g[-1]])))
    return out


# The full run passes through these phases (S1 skips the first two — recording
# started right before cooling):
#   Device Connected (room temp) -> Placed on Pig (warms to body) -> Cooling Start
#   (device on) -> [restarts] -> Device Off (rewarms to body) -> Device Removed
#   (returns to room temp).
EVENT_ORDER = ["Device Connected", "Placed on Pig", "Cooling Start", "Regulation Start",
               "Device Off", "Device Removed"]

# "Regulation Start" = the moment the controller stops pulling the eye down and
# starts holding it, i.e. when the relay begins switching on and off. This replaces
# an absolute 4.0 degC crossing as the basis for "time to target", because the set
# point is not always crossed: Pig 2 S3 settled at 4.21 degC and only dipped under
# 4.0 once, 52.7 min in, which made "time to target" meaningless for that run.
#
# The relay itself cycles FAST - Pig 1's live current channel switches with a
# median interval of 1.3 s - and the probe's thermal mass filters that out almost
# entirely (Pig 2 S3's smoothed ripple is 0.03-0.05 degC peak-to-peak, well under
# the plateau noise). So the switching is only directly detectable in the CURRENT.
# Where the current is unusable, what is still plainly visible is the end of the
# pull-down: the trace stops descending and starts wandering both ways.
REG_SWITCHES = 4        # transitions that count as "cycling has begun"
REG_WINDOW_S = 60.0     # ...within this many seconds
# Temperature fallback = "the pull-down has finished", measured RELATIVE to the run's
# own regulated level rather than against an absolute set point.
#
# Where the current is dead the relay cycling is NOT
# recoverable from the probe at all, so this is a proxy, not a detection. Two rules
# were tried and rejected first - "slope stops falling" (probe noise swings the
# apparent slope past any threshold, putting Pig 2 S3 at 53 min) and "first rise
# above the running minimum" (the probe keeps creeping down for tens of minutes, so
# the first real rise is also ~53 min).
# The proxy is validated: on Pig 1, where the current gives the true answer, it lands
# within 0.1 min of it on all three sessions (1.69/1.66/1.56 vs 1.77/1.62/1.54).
REG_SLOPE_SMOOTH_S = 30.0
REG_LEVEL_FROM = 0.2    # regulated level = median of the window after this fraction
REG_LEVEL_TOL_C = 0.25  # within this of that level counts as pulled down


def detect_events(rt, amps, temp, gap_s=20.0):
    """Auto event times (s) for the phase model above. Cooling Start / Device Off
    and the restart off/on times come from the CURRENT (reliable); the placement
    and removal boundaries are best-effort from the smoothed TEMPERATURE and are
    meant to be reviewed/corrected in the events sheet. The TEC current
    duty-cycles, so a 'restart' is a SUSTAINED off-gap (>= gap_s), not toggling."""
    ev = {k: None for k in EVENT_ORDER}
    ev["Restarts (n)"] = 0; ev["Restart Times"] = []
    n = len(rt)
    if n == 0:
        return ev
    dt_s = float(np.median(np.diff(rt))) or 0.1
    on = np.isfinite(amps) & (amps > CURR_ON_A)
    # A current channel that is never off carries no timing information. Pig 2 S1
    # recorded a constant 3.900 A for all 86361 samples (clamp not logging), which
    # otherwise puts Cooling Start at t=0 and stretches the window over the whole
    # recording. Treat a never-off channel as unusable and fall back to the
    # temperature, same as a missing one.
    if on.all():
        on = np.zeros_like(on)
    ev["Device Connected"] = float(rt[0])
    has_temp = np.isfinite(temp).any()
    tsm = smooth(temp, dt_s, 20) if has_temp else None

    # Cooling Start + restarts come from the CURRENT (its onset is reliable).
    if on.any():
        idx = np.where(on)[0]; first, last = idx[0], idx[-1]
        ev["Cooling Start"] = float(rt[first])
        gap_n = max(1, int(gap_s / dt_s))
        off = ~on[first:last + 1]
        run = 0; restarts = []
        for j, v in enumerate(off):
            if v:
                if run == 0:
                    gstart = first + j
                run += 1
            else:
                if run >= gap_n:
                    restarts.append((float(rt[gstart]), float(rt[first + j])))
                run = 0
        ev["Restart Times"] = restarts
        ev["Restarts (n)"] = len(restarts)
    elif has_temp:                                        # no current: fall back to the temp drop
        ev["Cooling Start"] = _first_sustained(rt, np.gradient(tsm, rt), 0, n, -0.03, dt_s)

    # Device Off = the last drive pulse where the current works (since 2026-10-28; the
    # temperature rule below lands 13-60 s after it in every MOSFET session, because the
    # smoothed ring stays within 2.5 degC of its minimum for a while after switch-off).
    # Without a usable current: end of the cold plateau (temperature leaves target and
    # rewarms). A manual hold with no current at the very end would make the pulse rule
    # early, so a temperature end > 5 min later is reported for a hand check.
    cs = ev["Cooling Start"]
    t_plateau = None
    if has_temp and cs is not None:
        i_cs = int(np.searchsorted(rt, cs))
        tmin = float(np.nanmin(tsm[i_cs:])) if i_cs < n else float(np.nanmin(tsm))
        # only RECORDED samples can be the last one on the plateau: where two files are
        # joined (Pig 1 S1, 16-s gap) the smoothed trace runs on through the gap
        cold = (tsm < (tmin + 2.5)) & np.isfinite(temp)
        w = np.where(cold[i_cs:])[0]
        if len(w):
            t_plateau = float(rt[i_cs + w[-1]])          # last moment still on the cold plateau
    if on.any():
        ev["Device Off"] = float(rt[last])
        if t_plateau is not None and t_plateau - ev["Device Off"] > 300:
            print(f"    ! last current pulse at {ev['Device Off'] / 60:.1f} min but the probe stays cold "
                  f"until {t_plateau / 60:.1f} min (manual hold?) - check Device Off by hand")
    else:
        ev["Device Off"] = t_plateau

    # Placed on Pig (warming before cooling) and Device Removed (falls after the
    # rewarm peak) are best-effort from the smoothed temperature slope.
    if has_temp:
        slope = np.gradient(tsm, rt)
        if cs is not None:
            i_cs = int(np.searchsorted(rt, cs))
            ev["Placed on Pig"] = _first_sustained(rt, slope, 0, i_cs, +0.03, dt_s)
        do = ev["Device Off"]
        if do is not None:
            i_do = int(np.searchsorted(rt, do))
            if i_do < n - 1:
                peak = i_do + int(np.nanargmax(tsm[i_do:n]))
                ev["Device Removed"] = _first_sustained(rt, slope, peak, n, -0.03, dt_s)

    return ev


def _clk(base, t):
    return sec_to_clock(base + t) if (base is not None and t is not None) else None


# ------------------------------------------------------------ recording coverage
# A recording can miss part of the session (Pig 3 S3, 2026-10-19: the laptop went into
# standby and WinDaq closed the file 10 min BEFORE switch-on, so the whole 2-h cooling
# is absent). Two hand-kept parts of the events workbook make every gap explicit:
#   * Events col G "Log time (local)": the bench-sheet clock of an event that happened
#     but is not in the file. Its Manual Time (s) then holds that clock converted to the
#     recording's relative seconds (it may lie after the file end). Col F "Recorded?"
#     is written by this script and reads "NOT RECORDED" for any event outside the
#     recorded samples. Durations between such times (Total cooling time) are kept;
#     statistics over them are not.
#   * Sheet "Coverage": one row per span of the session - Start/End (s), local clocks,
#     what the probe was doing, and its Use:
#       summary       recorded ring/device temperature, used for the summary;
#       trace only    recorded and shown, but kept out of every summary statistic
#                     (e.g. a cold device off the eye that is not room temperature);
#       blank         recorded, but NOT a ring temperature (probe in the dry well, out
#                     of the ring): blanked in Temperature Processed.csv (the raw value
#                     stays in its 'Temp raw' column) and kept out of the summary;
#       not recorded  no samples (dropout, standby, after the file stopped).
# With no Coverage sheet the whole file counts as "summary" (every older session).
# A summary statistic is computed only when its whole window was recorded; a window
# that reaches into an unrecorded span, or past either end of the file, reads N/A.
COVERAGE_SHEET = "Coverage"
COVERAGE_HDR = ["Start (s)", "End (s)", "Start (local)", "End (local)", "Recorded",
                "Use", "Probe / device", "What happened (source)"]
USE_VALUES = ("summary", "trace only", "blank", "not recorded")


def _coverage_rows(path):
    """(rows above the header, header index, all rows) of the Coverage sheet, or None."""
    if not path.exists():
        return None
    wb = openpyxl.load_workbook(path, data_only=True)
    if COVERAGE_SHEET not in wb.sheetnames:
        return None
    rows = [list(r) for r in wb[COVERAGE_SHEET].iter_rows(values_only=True)]
    hdr = next((i for i, r in enumerate(rows) if r and r[0] == COVERAGE_HDR[0]), None)
    return None if hdr is None else (rows[:hdr], hdr, rows)


def read_coverage(path):
    """[(start_s, end_s, use, row_values)] from the events workbook's Coverage sheet,
    or [] when it has none. Raises on an unknown Use value (a typo must not silently
    put a span into the statistics)."""
    got = _coverage_rows(path)
    if got is None:
        return []
    _pre, hdr, rows = got
    out = []
    for r in rows[hdr + 1:]:
        if not r or r[0] is None:
            continue
        r = (r + [None] * len(COVERAGE_HDR))[:len(COVERAGE_HDR)]
        use = str(r[5] or "").strip().lower()
        if use not in USE_VALUES:
            raise RuntimeError(f"{path.name} / {COVERAGE_SHEET}: Use '{r[5]}' is not one of "
                               f"{', '.join(USE_VALUES)}. Nothing was rewritten.")
        out.append((float(r[0]), float(r[1]), use, r))
    return out


class Coverage:
    """What part of a session the samples cover: the file span [rt0, rt_end] minus the
    Coverage sheet's 'not recorded' spans."""

    def __init__(self, rt, spans):
        self.rt0 = float(rt[0]) if len(rt) else None
        self.end = float(rt[-1]) if len(rt) else None
        self.tol = 1.0                        # s: clock rounding at the span edges
        self.gaps = [(a, b) for a, b, use, _r in spans if use == "not recorded"]
        self.declared = bool(spans)

    def covered(self, t0, t1):
        """True when the whole window [t0, t1] was recorded."""
        if t0 is None or t1 is None or self.rt0 is None:
            return False
        if t0 < self.rt0 - self.tol or t1 > self.end + self.tol:
            return False
        return not any(min(t1, b) - max(t0, a) > self.tol for a, b in self.gaps)

    def fraction(self, t0, t1):
        """Share of [t0, t1] that was recorded (0..1); None without a window."""
        if t0 is None or t1 is None or t1 <= t0 or self.rt0 is None:
            return None
        inside = max(0.0, min(t1, self.end) - max(t0, self.rt0))
        for a, b in self.gaps:
            inside -= max(0.0, min(t1, b, self.end) - max(t0, a, self.rt0))
        return max(0.0, min(1.0, inside / (t1 - t0)))

    def recorded(self, t):
        return t is not None and self.covered(t, t)


def event_status(t, manual, cov):
    """Text for the events sheet's 'Recorded?' column."""
    if isinstance(manual, str) and manual.strip().lower() in NO_EVENT:
        return "did not happen"
    if isinstance(manual, str) and manual.strip():
        return "NOT RECORDED - no time known"
    if t is None:
        return "no time"
    if cov is not None and not cov.recorded(t):
        return "NOT RECORDED - time from the log"
    return "recorded (manual)" if isinstance(manual, (int, float)) else "recorded (auto)"


def _read_event_table(path):
    """{event: (manual time, note, log time)} from an existing events sheet, matched by
    the event NAME in column A (not by row). Raises if the file exists but cannot be
    read: rewriting it then would destroy the hand-entered times."""
    out = {}
    if not path.exists():
        return out
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb["Events"] if "Events" in wb.sheetnames else wb.active
    except Exception as e:
        raise RuntimeError(f"cannot read {path.name} ({e}); close it in Excel / let Box "
                           f"finish syncing, then re-run. Nothing was rewritten.")
    for r in range(2, ws.max_row + 1):
        name = ws.cell(r, 1).value
        if name in EVENT_ORDER:
            out[name] = (ws.cell(r, 4).value, ws.cell(r, 5).value, ws.cell(r, 7).value)
    return out


def write_events_sheet(path, ev, base_clock, cov=None):
    old = _read_event_table(path)           # keep manual times, notes AND log times
    manual = {k: v[0] for k, v in old.items()}
    notes = {k: v[1] for k, v in old.items()}
    logt = {k: v[2] for k, v in old.items()}
    cov_rows = _coverage_rows(path)         # hand-kept coverage table, rewritten verbatim
    read_coverage(path)                     # validate its Use column before rewriting
    eff = read_event_overrides(path, dict(ev))
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Events"
    for c, h in enumerate(["Event", "Auto Time (s)", "Auto Clock", "Manual Time (s)", "Notes",
                           "Recorded?", "Log time (local)"], 1):
        cell = ws.cell(1, c, h); cell.fill = HDR; cell.font = WB_; cell.alignment = CEN
    for i, name in enumerate(EVENT_ORDER):
        r = i + 2
        ws.cell(r, 1, name).font = BOLD
        t = ev.get(name)
        ws.cell(r, 2, round(t, 1) if t is not None else None)
        ws.cell(r, 3, _clk(base_clock, t))
        if manual.get(name) is not None:
            ws.cell(r, 4, manual[name])
        if notes.get(name):
            ws.cell(r, 5, notes[name])      # keep the Notes column
        st = event_status(eff.get(name), manual.get(name), cov)
        c6 = ws.cell(r, 6, st)
        if st.startswith("NOT RECORDED"):
            c6.font = Font(bold=True, color="C00000")
        if logt.get(name) is not None:
            ws.cell(r, 7, logt[name])
    # restart off/on times
    rr = len(EVENT_ORDER) + 3
    ws.cell(rr, 1, f"Restarts: {ev.get('Restarts (n)', 0)}").font = BOLD
    for c, h in enumerate(["#", "Off Time (s)", "Off Clock", "Back-On Time (s)", "Back-On Clock"], 1):
        cell = ws.cell(rr + 1, c, h); cell.fill = HDR; cell.font = WB_; cell.alignment = CEN
    restarts = ev.get("Restart Times", [])
    for i, (off_t, on_t) in enumerate(restarts):
        r = rr + 2 + i
        ws.cell(r, 1, i + 1)
        ws.cell(r, 2, round(off_t, 1)); ws.cell(r, 3, _clk(base_clock, off_t))
        ws.cell(r, 4, round(on_t, 1)); ws.cell(r, 5, _clk(base_clock, on_t))
    ws.cell(rr + 3 + len(restarts), 1,
            "Manual Time (s) overrides Auto Time (s). Re-run after editing.").font = Font(italic=True, color="808080")
    ws.cell(rr + 4 + len(restarts), 1,
            "'none' = did not happen. An event that happened but is not in the file: Manual Time = its "
            "log clock in this recording's seconds (may be after the file end), the clock in col G. "
            "'Recorded?' is written by convert_temperature.py. See the Coverage sheet if there is one."
            ).font = Font(italic=True, color="808080")
    for col, w in zip("ABCDEFG", (18, 14, 12, 16, 14, 30, 16)):
        ws.column_dimensions[col].width = w
    if cov_rows is not None:
        pre, hdr, rows = cov_rows
        cs_ = wb.create_sheet(COVERAGE_SHEET)
        for r, row in enumerate(rows, 1):
            for c, v in enumerate(row, 1):
                if v is None:
                    continue
                cell = cs_.cell(r, c, v)
                if r == hdr + 1:
                    cell.fill = HDR; cell.font = WB_; cell.alignment = CEN
                elif r <= hdr:
                    cell.font = BOLD if r == 1 else Font(italic=True, color="595959")
                elif str(row[5] if len(row) > 5 else "").strip().lower() == "not recorded" and c in (5, 6):
                    cell.font = Font(bold=True, color="C00000")
        for col, w in zip("ABCDEFGH", (10, 10, 12, 12, 11, 13, 46, 80)):
            cs_.column_dimensions[col].width = w
    wb.active = 0
    wb.save(path)


def read_event_overrides(path, ev):
    """Apply the Manual Time column. A read error stops the session rather than
    falling back to the automatic times (for Pig 2 S1/S2 that would move Device Off
    by 32 and 42 min).
    Text in Manual Time clears the event: "none" = it did not happen, anything else
    (e.g. "not recorded") = no usable time, never the AUTO time (for Pig 3 S3 the
    auto times were guesses from the probe being handled). A number is used even when it lies outside the file
    (an event taken from the log; see the Coverage notes above)."""
    for name, (man, _note, _log) in _read_event_table(path).items():
        if name in ev and isinstance(man, (int, float)):
            ev[name] = float(man)
        elif name in ev and isinstance(man, str) and man.strip():
            ev[name] = None
    return ev


NO_EVENT = ("none", "n/a", "na", "-")


def _r(x, n=2):
    return round(float(x), n) if x is not None and np.isfinite(x) else None


TARGET_C = 4.0   # nominal cooling target (the coldest calibration set point)
# Safeguard band. Both limits are tracked over the whole time the device is ON
# THE ANIMAL (Placed on Pig .. Device Removed), not just while it is actively
# cooling: the eye is exposed for that entire period, and the upper limit exists
# to catch a device that heats (e.g. TEC driven the wrong way) whether or not a
# cooling cycle is running.
CRIT_C    = 2.0    # too cold: track cumulative time colder than this (was 3.5 until 2026-10-30)
CRIT_HI_C = 39.0   # too warm: track cumulative time hotter than this


def rewarm_equilibrium(rt, temp, t0, t1):
    """Extrapolated asymptote (°C) of the rewarming curve, fit as T(t)=A-B*exp(-t/tau)
    over [t0,t1] from the minimum onward. Because the device is removed slightly
    before full equilibrium, A (the asymptote) is the temperature it *would* have
    reached and normally exceeds the observed rewarm max."""
    m = np.isfinite(temp) & (rt >= t0) & (rt <= t1)
    if m.sum() < 20:
        return None
    x = rt[m]; y = temp[m]
    imin = int(np.argmin(y))
    x = x[imin:] - x[imin]; y = y[imin:]
    if len(x) < 20 or x[-1] <= 0:
        return None
    A0 = float(y.max() + 0.5 * (y.max() - y.min())); B0 = float(A0 - y[0])
    tau0 = max(x[-1] / 3.0, 1.0)

    def f(t, A, B, tau):
        return A - B * np.exp(-t / tau)
    try:
        popt, _ = curve_fit(f, x, y, p0=[A0, B0, tau0], maxfev=10000,
                            bounds=([y.min(), 0, 1.0], [y.max() + 40, 500, 1e5]))
        return float(popt[0]) if np.isfinite(popt[0]) else None
    except Exception:
        return None


TEMP_SUMMARY_ROWS = [
    "Room temp – initial (°C)", "Time to heat to body (min)", "Body equilibrium temp (°C)",
    "Time to regulation (min)", "Min temp (°C)", "Temp SD at target (°C)", "Cooling depth (°C)",
    "Time below 2°C on animal (min)", "Time above 39°C on animal (min)",
    "Total cooling time (min)", "Rewarm time to body (min)",
    "Rewarm max temp (°C)", "Rewarm equilibrium temp (extrap. °C)", "Room temp – final (°C)",
    "Max current (A)", "Restarts (n)", "Calibration RMSE (°C)",
    "Cooling window recorded (%)", "Total cooling time source", "Recording coverage",
]


def summarize(rt, amps, temp, ev, target_temp=None, over_range=None, cov=None):
    """over_range: optional bool mask of samples whose probe voltage was valid but
    converted to NaN (hotter than the calibrated curve, see volts_to_C). They are
    counted as hotter than CRIT_HI_C, so a hot ring cannot read as zero time.
    cov: Coverage of the recording. A statistic whose window is not wholly recorded
    reads N/A (None); `temp` must already have its 'blank' / 'trace only' spans set
    to NaN by the caller."""
    if cov is None:
        cov = Coverage(rt, [])
    dt_s = float(np.median(np.diff(rt))) or 0.1
    fin = float(rt[-1])
    finite = np.isfinite(temp)
    target = target_temp if target_temp is not None else TARGET_C   # relay switches at the 4°C calib point
    conn = ev.get("Device Connected"); place = ev.get("Placed on Pig")
    cs = ev.get("Cooling Start"); do = ev.get("Device Off"); rem = ev.get("Device Removed")
    reg = ev.get("Regulation Start")

    def mean_in(t0, t1, arr=temp):
        if t0 is None:
            return None
        m = np.isfinite(arr) & (rt >= t0) & (rt <= (t1 if t1 is not None else fin))
        return float(np.mean(arr[m])) if m.any() else None

    def first_after(mask, after):
        m = mask & (rt >= (after if after is not None else rt[0]))
        w = np.where(m)[0]
        return float(rt[w[0]]) if len(w) else None

    def mins(x):
        return x / 60.0 if x is not None else None

    # every window below must be wholly RECORDED (cov), else the statistic is N/A
    room_i = mean_in(conn, place) if cov.covered(conn, place) else None
    body = mean_in(cs - 90, cs) if (cs is not None and cov.covered(cs - 90, cs)) else None
    # A recording that holds only the placement, with the device never switched on
    # (Pig 3 S3, Cooling Start marked "none"), still measures the eye-surface
    # equilibrium: take it from the 90 s before the device came off the eye.
    if body is None and cs is None and rem is not None and place is not None and rem > place             and cov.covered(max(place, rem - 90), rem):
        body = mean_in(max(place, rem - 90), rem)
    t_heat = None
    if place is not None and body is not None:
        tr = first_after(finite & (temp >= body - 0.5), place)
        t_heat = (tr - place) if tr is not None else None
    # Min temp describes the COOLED EYE, so it is restricted to the active
    # cooling window (Cooling Start .. Device Off). Over the whole trace it
    # picks up bench handling and the post-removal probe, which sits on the
    # still-running device and therefore keeps falling once it is off the eye:
    # Pig 2 S2 read 1.58 °C at t=10961 s, ~41 min after removal, against a true
    # in-window minimum of 3.69 °C. Falls back to the whole trace if the run has
    # no usable Cooling Start / Device Off. Cooling depth inherits this.
    cool_m = finite.copy()
    if cs is not None:
        cool_m &= (rt >= cs)
    if do is not None:
        cool_m &= (rt <= do)
    cool_frac = cov.fraction(cs, do)
    if not cool_m.any() and not (cs is not None and do is not None):
        cool_m = finite
    tmin = float(np.min(temp[cool_m])) if cool_m.any() else None
    if cool_frac is not None and cool_frac == 0.0:
        tmin = None      # the cooling window holds no recorded sample (Pig 3 S3)
    if cs is None and do is None and "Cooling Start" in ev:
        # no cooling in this recording (both marked "none"): there is no cooled-eye
        # minimum, and the whole-trace fallback would report bench handling
        tmin = None
    depth = (body - tmin) if (body is not None and tmin is not None) else None
    # Time to regulation: cooling start -> the relay starting to cycle, NOT the first
    # crossing of the 4 degC calibration point. The set point is not reliably crossed
    # - Pig 2 S3 held 4.21 degC and dipped under 4.0 only once, 52.7 min in, which is
    # not "time to target" in any useful sense.
    t_tt = (reg - cs) if (cs is not None and reg is not None) else None
    # SD of temperature while regulated at target (near target; excludes warm failure blips)
    sd_target = None
    if cs is not None and do is not None:
        m = finite & (rt >= cs) & (rt <= do) & (np.abs(temp - target) <= 2.5)
        if m.sum() > 5:
            sd_target = float(np.std(temp[m]))
    # Safeguard exposure, over the whole time the device sat ON THE ANIMAL
    # (Placed on Pig .. Device Removed) — wider than the cooling window, because
    # the eye is exposed for the whole placement, including the settle before
    # Cooling Start and the rewarm after Device Off. Whole-trace it instead
    # counted bench handling and the post-removal probe: Pig 2 S1/S2 each read
    # ~0.2 min below 3.5 °C while never going under 3.69 °C on the eye.
    # Falls back to the cooling window, then the whole trace, if placement or
    # removal was never identified.
    on_w = np.ones_like(finite)
    if place is not None:
        on_w &= (rt >= place)
    if rem is not None:
        on_w &= (rt <= rem)
    on_m = finite & on_w
    if not on_m.any():
        on_m = cool_m
        on_w = cool_m
    crit_min = float(np.sum(on_m & (temp < CRIT_C))) * dt_s / 60.0
    hot = on_m & (temp > CRIT_HI_C)
    if over_range is not None:
        hot_nan = on_w & over_range
        if hot_nan.any():
            print(f"    ! {hot_nan.sum() * dt_s / 60.0:.2f} min on the animal hotter than the "
                  f"calibrated range (NaN) - counted as above {CRIT_HI_C:g} degC")
        hot |= hot_nan
    crit_hi_min = float(np.sum(hot)) * dt_s / 60.0
    if not cov.covered(place if place is not None else float(rt[0]), rem if rem is not None else fin):
        # time on the animal was partly NOT recorded: a count over the recorded part
        # would read as a measured (low) exposure - N/A instead
        crit_min = crit_hi_min = None
    if cs is None and do is None and "Cooling Start" in ev:
        # no cooling in this recording (both marked "none"): "0 min below 3.5 degC"
        # would read as a measured safe exposure (Pig 3 S3, whose cooling was not
        # recorded) - report N/A instead
        crit_min = crit_hi_min = None
    cool_dur = (do - cs) if (cs is not None and do is not None) else None
    rw_end = rem if rem is not None else fin
    rw_max = rw_eq = t_reheat = None
    if do is not None and cov.covered(do, rw_end):
        m = finite & (rt >= do) & (rt <= rw_end)
        rw_max = float(np.max(temp[m])) if m.any() else None
        rw_eq = rewarm_equilibrium(rt, temp, do, rw_end)
        # an asymptote far above anything the eye reached is a failed fit, not a
        # temperature (Pig 4 S3 extrapolated 62.7 degC): leave it blank
        if rw_eq is not None and rw_max is not None and (rw_eq > 40.0 or rw_eq - rw_max > 8.0):
            print(f"    ! rewarm extrapolation {rw_eq:.1f} degC rejected (max reached {rw_max:.1f} degC)")
            rw_eq = None
        if body is not None:
            tr = first_after(finite & (temp >= body - 0.5), do)
            if tr is not None and (rem is None or tr <= rem):
                t_reheat = tr - do
    room_f = mean_in(rem, fin) if rem is not None else None
    return {
        "Room temp – initial (°C)":              _r(room_i),
        "Time to heat to body (min)":            _r(mins(t_heat), 2),
        "Body equilibrium temp (°C)":            _r(body),
        "Time to regulation (min)":              _r(mins(t_tt), 2),
        "Min temp (°C)":                         _r(tmin),
        "Temp SD at target (°C)":                _r(sd_target, 3),
        "Cooling depth (°C)":                    _r(depth),
        "Time below 2°C on animal (min)":      _r(crit_min, 2),
        "Time above 39°C on animal (min)":       _r(crit_hi_min, 2),
        "Total cooling time (min)":              _r(mins(cool_dur), 2),
        "Rewarm time to body (min)":             _r(mins(t_reheat), 2),
        "Rewarm max temp (°C)":                  _r(rw_max),
        "Rewarm equilibrium temp (extrap. °C)":  _r(rw_eq),
        "Room temp – final (°C)":                _r(room_f),
        "Max current (A)":                       _r(np.nanmax(amps)) if np.isfinite(amps).any() else None,
        # restarts can only be counted over a wholly recorded cooling window
        "Restarts (n)":                          (int(ev.get("Restarts (n)", 0))
                                                  if cool_frac is None or cool_frac >= 0.999 else None),
        # coverage: where the numbers come from
        "Cooling window recorded (%)":           (_r(100.0 * cool_frac, 1) if cool_frac is not None else None),
        "Total cooling time source":             (None if cool_dur is None else
                                                  "recording" if cov.covered(cs, do) else
                                                  "log (event times outside the recording)"),
    }


def write_summary_sheet(path, summary, rmse):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Summary"
    ws["A1"] = "Temperature Summary"; ws["A1"].font = Font(bold=True, size=13)
    for c, h in enumerate(["Metric", "Value"], 1):
        cell = ws.cell(3, c, h); cell.fill = HDR; cell.font = WB_; cell.alignment = CEN
    rows = dict(summary); rows["Calibration RMSE (°C)"] = round(rmse, 3) if rmse is not None else None
    for i, k in enumerate(TEMP_SUMMARY_ROWS):
        ws.cell(4 + i, 1, k).font = BOLD
        ws.cell(4 + i, 2, rows.get(k))
    # old-method sessions: the offset and the uncorrected values, for the record only
    extra = [k for k in rows if k == "Calibration offset added (°C)" or k.endswith("[uncorrected]")]
    r = 4 + len(TEMP_SUMMARY_ROWS) + 1
    for k in extra:
        ws.cell(r, 1, k).font = Font(italic=True, color="808080")
        ws.cell(r, 2, rows.get(k)).font = Font(italic=True, color="808080")
        r += 1
    ws.column_dimensions["A"].width = 24; ws.column_dimensions["B"].width = 14
    wb.save(path)


def convert_session(sess_path):
    """Full pipeline for one session. Returns (summary_dict, rmse) or (None, None)."""
    cal_folder  = find_sub(sess_path, "Calibration")
    cool_folder = find_sub(sess_path, "Cooling")
    if cool_folder is None:
        return None, None, []
    cal = calibrate_session(cal_folder) if cal_folder else None
    concat, used_files = read_experiment_concat(cool_folder)
    if concat is None:
        return None, (cal[1] if cal else None), []
    rt, clk, ch1, ch2 = concat
    amps = ch1_to_amps(ch1)
    if cal is None:
        # no usable calibration -> current + events only, temp left blank
        temp = np.full_like(ch2, np.nan)
        rmse = None; target_temp = None
    else:
        coef, rmse, model, target_temp, shift = cal
        temp = volts_to_C(ch2, coef, model, shift)

    ev = detect_events(rt, amps, temp)
    ev_path = cool_folder / "Temperature Events.xlsx"
    base_clock = parse_clock(clk[0]) if clk else None
    # Both of these need the window the operator actually meant, so resolve the
    # sheet's manual overrides BEFORE deriving anything from it. Against the auto
    # times, Pig 2 S2 put Regulation Start at 2316 s - BEFORE its manual Cooling
    # Start of 4869 - and S1 put it 15 s before the manual Device Off.
    eff = read_event_overrides(ev_path, dict(ev))
    if not ev["Restart Times"]:
        ev["Restart Times"] = temp_restarts(rt, amps, temp,
                                            eff.get("Cooling Start"), eff.get("Device Off"))
        ev["Restarts (n)"] = len(ev["Restart Times"])
    dt_s = float(np.median(np.diff(rt))) or 0.1
    ev["Regulation Start"] = regulation_start(
        rt, amps, temp, None, eff.get("Cooling Start"), eff.get("Device Off"), dt_s)
    spans = read_coverage(ev_path)          # hand-kept recording coverage (may be [])
    cov = Coverage(rt, spans)
    write_events_sheet(ev_path, ev, base_clock, cov)
    ev = read_event_overrides(ev_path, ev)

    # valid probe voltage that converted to NaN = hotter than the calibrated curve
    # (from the UNMASKED conversion, so a blanked span never counts as hot)
    over_range = (np.isfinite(ch2) & ~np.isfinite(temp)) if cal is not None else None
    # Old-method calibration (2026-11-04): the session's temperatures are corrected by
    # RING_OFFSET_C from here on; events above were detected on the reading as calibrated
    # (an offset changes no slope or current), and the uncorrected trace and summary are
    # kept for the record. Downstream (erg_cohort, master_analysis) read the uncorrected
    # values and add the offset themselves, so their outputs are unchanged.
    offset = session_offset(sess_path) if cal is not None else 0.0
    temp_unc_raw = temp
    temp = temp + offset
    # Coverage: 'blank' spans are not ring temperature -> blanked in the trace;
    # 'trace only' spans stay in the trace but out of every statistic
    temp_raw = temp
    seg = np.full(len(rt), "", dtype=object)
    stats_mask = np.zeros(len(rt), bool)
    blank_mask = np.zeros(len(rt), bool)
    for k, (a, b, use, row) in enumerate(spans, 1):
        m = (rt >= a) & (rt <= b)
        label = f"{use} (Coverage span {k})"
        seg[m & (seg == "")] = label
        if use in ("blank", "trace only"):
            stats_mask |= m
        if use == "blank":
            blank_mask |= m
    temp = np.where(blank_mask, np.nan, temp_raw)
    temp_unc = np.where(blank_mask, np.nan, temp_unc_raw)
    if over_range is not None:
        over_range = over_range & ~stats_mask

    # processed CSV (full rate). Sessions with a Coverage sheet get two extra columns:
    # the unblanked temperature and the span label (readers use columns 1-4 only).
    with open(cool_folder / "Temperature Processed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = ["Relative Time (s)", "Clock", "Current (A)", "Temp (°C)"]
        if spans:
            hdr += ["Temp raw (°C, incl. blanked spans)", "Coverage"]
        if offset:
            hdr += ["Temp uncorrected (°C)"]
        w.writerow(hdr)
        for i in range(len(rt)):
            row = [f"{rt[i]:.1f}", clk[i] if i < len(clk) else "",
                   "" if not np.isfinite(amps[i]) else f"{amps[i]:.3f}",
                   "" if not np.isfinite(temp[i]) else f"{temp[i]:.3f}"]
            if spans:
                row += ["" if not np.isfinite(temp_raw[i]) else f"{temp_raw[i]:.3f}", seg[i]]
            if offset:
                row += ["" if not np.isfinite(temp_unc[i]) else f"{temp_unc[i]:.3f}"]
            w.writerow(row)

    temp_stats = np.where(stats_mask, np.nan, temp)
    summary = summarize(rt, amps, temp_stats, ev,
                        (target_temp + offset) if target_temp is not None else None, over_range, cov)
    if offset:
        unc = summarize(rt, amps, np.where(stats_mask, np.nan, temp_unc), ev, target_temp, over_range, cov)
        summary["Calibration offset added (°C)"] = offset
        for k, v in unc.items():
            if "°C" in k:
                summary[f"{k} [uncorrected]"] = v
    if spans:
        nr = [r for _a, _b, use, r in spans if use == "not recorded"]
        summary["Recording coverage"] = (
            f"file {_clk(base_clock, float(rt[0]))}-{_clk(base_clock, float(rt[-1]))} (DAQ clock, UTC); "
            f"NOT RECORDED: " + "; ".join(f"{r[2]}-{r[3]} local {r[6] or ''}".strip() for r in nr)
            + " - see Temperature Events.xlsx / Coverage")
    write_summary_sheet(cool_folder / "Temperature Summary.xlsx", summary, rmse)
    return summary, rmse, used_files


# --------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("pigs", nargs="*")
    args = ap.parse_args()
    root = Path(args.root)
    pig_dirs = [d for d in sorted(root.iterdir())
                if d.is_dir() and re.match(r"^Pig\s*\d+$", d.name)]
    if args.pigs:
        want = set(args.pigs)
        pig_dirs = [d for d in pig_dirs if re.match(r"^Pig\s*(\d+)$", d.name).group(1) in want]

    errors = 0
    for pig in pig_dirs:
        print(pig.name)
        for sess in sorted(pig.iterdir()):
            if not (sess.is_dir() and re.match(r"^Session\s*\d", sess.name)):
                continue
            try:
                summary, rmse, used = convert_session(sess)
            except Exception as e:
                print(f"    {sess.name}: ERROR {e}"); errors += 1; continue
            if summary is None:
                print(f"    {sess.name}: no cooling recording")
            else:
                tmin = summary.get("Min temp (°C)")
                fnote = f"  [{len(used)} files]" if len(used) > 1 else ""
                print(f"    {sess.name}: min={tmin}°C  RMSE={rmse}  "
                      f"restarts={summary.get('Restarts (n)')}{fnote}")
    print("\nDone." if not errors else f"\nDone, with {errors} session(s) FAILED (see ERROR above).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
