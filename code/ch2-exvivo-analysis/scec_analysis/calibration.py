"""Voltage -> temperature (and current) conversion.

Human eyes: one curve per calibration session and channel, fitted to the dry-block
plateaus (37, 20.5, 4 degC). calibration.human_curve selects it:
  "linear"          least-squares straight line through the three plateaus. The
                    channels are linearised and trimmed like the porcine ones (0 V
                    at 20.5 degC, -8 V at 37 degC), so the transfer is close to a
                    line; on the 11-temperature bench table of 2022-05-26 a line
                    stays within 0.34 degC between 2 and 39 degC.
  "steinhart_hart"  notebook v4.28: Steinhart-Hart form in ln(V + shift), exact at
                    the three plateaus but up to 2 degC high between 37 and 20.5 degC
                    on the same bench table (kept for the reproduction test).
The TEC current uses one linear fit pooled over all sessions' current plateaus.
Porcine eyes: the signal-conditioning gain and offset were trimmed so that every
channel reads 0 V at 20.5 degC and -8.00 V at 37.0 degC; one fixed linear transfer
function therefore converts all channels. The current was stored in A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def plateau_levels(v: pd.Series, n: int = 3, seed: int = 42) -> list[float]:
    """Plateau voltages: KMeans partition, then the densest 0.1 V bin of each
    cluster (refined over +/-0.2 V), so ramp samples do not bias the level."""
    v = pd.Series(v).dropna()
    if len(v) < n * 10:
        return []
    x = v.to_numpy()
    km = KMeans(n_clusters=n, random_state=seed, n_init=10).fit(x.reshape(-1, 1))
    levels = []
    for i in np.argsort(km.cluster_centers_.flatten()):
        c = pd.Series(x[km.labels_ == i])
        mode = float(c.round(1).value_counts().index[0])
        near = (c >= mode - 0.2) & (c <= mode + 0.2)
        levels.append(float(c[near].mean()) if near.any() else mode)
    return sorted(levels)


# Settled plateau rule (2026-11-04): the same rule as the in-vivo ring calibration
# (in-vivo temperature conversion), so both cohorts are
# calibrated alike. The KMeans level only LOCATES a plateau; its value is the median of the
# last 3 min before the ramp to the next temperature (the last 10 s dropped), or, if those
# 3 min still drift by more than 5 mV/min, the median of the longest stretch whose 10-s
# medians all stay within +/-0.01 V.
SETTLE_WINDOW_S = 180.0
SETTLE_DROP_S = 10.0
SETTLE_DRIFT_V_MIN = 0.005
STABLE_BAND_V = 0.01
LOCATE_BAND_V = 0.3
BRIDGE_S = 5.0


def _runs(mask: np.ndarray, t: np.ndarray, bridge_s: float = BRIDGE_S):
    """Contiguous runs of True (gaps shorter than bridge_s bridged), as (i0, i1) inclusive."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[p] >= bridge_s:
            out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return out


def _stable_band(t: np.ndarray, v: np.ndarray, half: float = STABLE_BAND_V, block_s: float = 10.0):
    """Median and duration of the longest stretch of 10-s block medians within +/-half."""
    b = np.floor((t - t[0]) / block_s).astype(int)
    ids = np.unique(b)
    med = np.array([np.median(v[b == k]) for k in ids])
    best, lo = (0, 0), 0
    for hi in range(len(med)):
        while med[lo:hi + 1].max() - med[lo:hi + 1].min() > 2 * half:
            lo += 1
        if hi - lo > best[1] - best[0]:
            best = (lo, hi)
    sel = (b >= ids[best[0]]) & (b <= ids[best[1]])
    return float(np.median(v[sel])), float(t[sel][-1] - t[sel][0])


def settled_plateau_levels(v: pd.Series, n: int = 3, seed: int = 42, details: list | None = None) -> list[float]:
    """Plateau voltages by the settled rule (see SETTLE_* above), ascending like plateau_levels.

    v must carry the time in seconds as its index (the calibration DataFrames do).
    A plateau shorter than 3 min + 10 s is the human 37 degC plateau of eyes HE1-HE4
    (138-178 s recorded): the block had settled for more than 7 min BEFORE the recording
    started (2026-11-04), so its value is the median of the whole recorded plateau with
    the first and last 10 s dropped, and no drift fallback is applied to so short a window.
    `details`, if given, receives one dict per plateau (method, window, drift)."""
    s = pd.to_numeric(v, errors="coerce").dropna()
    located = plateau_levels(s, n, seed)
    if len(located) != n:
        return located
    t = s.index.to_numpy(float)
    x = s.to_numpy(float)
    out = []
    for a in located:
        rr = _runs(np.abs(x - a) <= LOCATE_BAND_V, t)
        if not rr:
            out.append(a)
            if details is not None:
                details.append(dict(level=a, method="kmeans (no plateau run)"))
            continue
        i0, i1 = max(rr, key=lambda r: t[r[1]] - t[r[0]])
        t0, t1 = t[i0], t[i1]
        dur = t1 - t0
        wend = t1 - SETTLE_DROP_S
        if dur < SETTLE_WINDOW_S + SETTLE_DROP_S:
            sel = (t >= t0 + SETTLE_DROP_S) & (t <= wend)
            val = float(np.median(x[sel])) if sel.sum() > 10 else a
            out.append(val)
            if details is not None:
                details.append(dict(level=val, method="whole short plateau (settled before recording)",
                                    plateau_s=round(dur, 1)))
            continue
        sel = (t >= wend - SETTLE_WINDOW_S) & (t <= wend)
        tw, vw = t[sel], x[sel]
        drift = float(np.polyfit(tw / 60.0, vw, 1)[0])
        if abs(drift) <= SETTLE_DRIFT_V_MIN:
            val, meth = float(np.median(vw)), "last 3 min"
        else:
            seg = (t >= t0) & (t <= t1)
            val, sdur = _stable_band(t[seg], x[seg])
            meth = f"stable band ({sdur:.0f} s)"
        out.append(val)
        if details is not None:
            details.append(dict(level=val, method=meth, plateau_s=round(dur, 1),
                                drift_mV_per_min=round(1e3 * drift, 2)))
    return sorted(out)


def human_plateau_levels(v: pd.Series, cfg: dict, details: list | None = None) -> list[float]:
    """The human calibration plateaus with the configured rule
    ([calibration] plateau_method: "settled", the default since 2026-11-04, or "kmeans",
    notebook v4.28 and the thesis numbers before 2026-11-04)."""
    seed = cfg["calibration"]["kmeans_seed"]
    if cfg["calibration"].get("plateau_method", "settled") == "kmeans":
        return plateau_levels(v, 3, seed)
    return settled_plateau_levels(v, 3, seed, details)


def kmeans_levels(v: pd.Series, k: int = 2, seed: int = 42) -> list[float]:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(v.to_numpy().reshape(-1, 1))
    return sorted(km.cluster_centers_.flatten())


def fit_steinhart_hart(v_levels, t_ref_C):
    """Exact 3-point fit of 1/T(K) = A + B ln(V+s) + C ln(V+s)^3, with the shift s
    placing the lowest plateau at ln = 0."""
    v = np.asarray(v_levels, float)
    shift = abs(v.min()) + 1.0
    ln_v = np.log(v + shift)
    X = np.column_stack([np.ones_like(ln_v), ln_v, ln_v ** 3])
    A, B, C = np.linalg.solve(X, 1.0 / (np.asarray(t_ref_C, float) + 273.15))
    return float(A), float(B), float(C), float(shift)


def steinhart_hart(v, A, B, C, shift):
    ln = np.log(np.clip(np.asarray(v, float) + shift, 1e-12, None))
    inv_T = A + B * ln + C * ln ** 3
    return 1.0 / np.where(inv_T > 0, inv_T, np.nan) - 273.15


def fit_linear(v_levels, t_ref_C):
    """Least-squares line T = m V + c through the calibration plateaus."""
    m, c = np.polyfit(np.asarray(v_levels, float), np.asarray(t_ref_C, float), 1)
    return float(m), float(c)


def human_temperature(v, coeffs):
    """V -> degC with a channel's curve: (m, c) linear, (A, B, C, shift) Steinhart-Hart."""
    if len(coeffs) == 2:
        return coeffs[0] * np.asarray(v, float) + coeffs[1]
    return steinhart_hart(v, *coeffs)


def dT_dV(v, coeffs, dv=1e-4):
    return (human_temperature([v + dv], coeffs)[0] - human_temperature([v - dv], coeffs)[0]) / (2 * dv)


def fit_human_calibration(calibration: dict, cfg: dict) -> dict:
    """{eye_id: {channel: curve coefficients}} (see human_temperature); a channel
    without three clean plateaus gets no curve (and becomes NaN)."""
    fit = fit_linear if cfg["calibration"].get("human_curve", "steinhart_hart") == "linear" else fit_steinhart_hart
    t_ref = cfg["calibration"]["reference_temps_C"]
    seed = cfg["calibration"]["kmeans_seed"]
    coeffs = {}
    for eye, df in calibration.items():
        coeffs[eye] = {}
        for ch in cfg["channels"]["temperature"]:
            if ch in df.columns:
                levels = human_plateau_levels(df[ch], cfg)
                if len(levels) == 3:
                    coeffs[eye][ch] = fit(levels, t_ref)
    return coeffs


def fit_pooled_current(calibration: dict, cfg: dict):
    """One linear V -> A fit over every eye's two current plateaus; eyes whose own
    slope departs from the median by > max(3 IQR, 0.5) are dropped.
    The current sensor's output FALLS as current rises (the porcine files, which
    store both, give A = -0.98 V + 4.9), so the LOW-voltage plateau is the 3.9 A
    (drive on) reference and the HIGH-voltage plateau is 0 A. Notebook v4.28 maps
    them the other way round; this affects only the descriptive current summary."""
    current = cfg["channels"]["current"]
    i_on, i_off = cfg["calibration"]["current_reference_A"]
    seed = cfg["calibration"]["kmeans_seed"]
    per_eye = {}
    for eye, df in calibration.items():
        if current not in df.columns:
            continue
        lo, hi = kmeans_levels(df[current].dropna(), 2, seed)
        per_eye[eye] = (lo, hi)
    slopes = np.array([(i_off - i_on) / (hi - lo) for lo, hi in per_eye.values()])
    med, iqr = np.median(slopes), np.subtract(*np.percentile(slopes, [75, 25]))
    pairs = [(lo, i_on, hi, i_off) for (eye, (lo, hi)), s in zip(per_eye.items(), slopes)
             if abs(s - med) <= max(3 * iqr, 0.5)]
    V = np.array([p for lo, _, hi, _ in pairs for p in (lo, hi)])
    I = np.array([p for _, a, _, b in pairs for p in (a, b)])
    return tuple(float(x) for x in np.polyfit(V, I, 1))


def calibrate(runs: dict, calibration: dict, cfg: dict) -> dict:
    """Fill run.temp (degC, plus current in A). Returns the human curve coefficients."""
    temp_ch, current = cfg["channels"]["temperature"], cfg["channels"]["current"]
    lo, hi = cfg["calibration"]["clip_C"]
    sh = fit_human_calibration(calibration, cfg)
    m_i, c_i = fit_pooled_current(calibration, cfg)
    for r in runs.values():
        df = r.volts.copy()
        for ch in temp_ch:
            if ch not in df.columns:
                continue
            v = pd.to_numeric(df[ch], errors="coerce").to_numpy()
            if r.species == "Human":
                # Workbook columns exchanged by hand after acquisition: the column already carries
                # the name of the sensor it is, so it is converted with the curve of the amplifier
                # channel it was acquired on, which is the other column's curve.
                src = ch
                for ex in cfg["acquisition"].get("column_exchanges", []):
                    if r.species == ex["species"] and r.eye == ex["eye"] and ch in ex["channels"]:
                        src = [c_ for c_ in ex["channels"] if c_ != ch][0]
                c = sh.get(r.eye_id, {}).get(src)
                t = human_temperature(v, c) if c is not None else np.full_like(v, np.nan)
            else:
                t = cfg["calibration"]["porcine_slope"] * v + cfg["calibration"]["porcine_intercept"]
            t = t.copy()
            t[(t < lo) | (t > hi)] = np.nan
            df[ch] = t
        if r.species == "Human" and current in df.columns:
            df[current] = m_i * pd.to_numeric(df[current], errors="coerce").to_numpy() + c_i
        r.temp = df
        # Sensors plugged into each other's amplifier channel: each recording is converted with
        # the curve of the channel it was acquired on (above), then given the name of the
        # sensor it really is. The raw voltages are renamed too (hardware gate).
        for swap in cfg["acquisition"].get("channel_swaps", []):
            a, b = swap["channels"]
            if r.species == swap["species"] and r.eye == swap["eye"] and {a, b} <= set(df.columns):
                r.temp = df.rename(columns={a: b, b: a})[list(df.columns)]
                r.volts = r.volts.rename(columns={a: b, b: a})[list(r.volts.columns)]
    return sh
