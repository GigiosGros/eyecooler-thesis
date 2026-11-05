"""Denoising and the single-exponential fit.

Each temperature trace is (1) median filtered, which removes the TEC relay
spikes without smearing the cooling onset, and (2) low-pass filtered with a
zero-phase Butterworth. Before (2), the start of the trace is extended by point
reflection, x(-t) = 2 x0 - x(t): a zero-phase filter applied to a signal that is
point-symmetric about its first sample returns that sample unchanged, so the
onset is not spread backward to t = 0 and the starting temperature is preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


def sampling_rate(index) -> float:
    t = pd.to_numeric(pd.Series(index), errors="coerce").dropna().to_numpy()
    dt = np.median(np.diff(t))
    return 1.0 / dt if dt > 0 else 1.0


def median_only(x, kernel: int):
    """Median filter; edges padded with the edge value (steady pre-cooling plateau)."""
    k = kernel if kernel % 2 == 1 else kernel - 1
    return median_filter(np.asarray(x, float), size=k, mode="nearest")


def lowpass_onset_preserving(x, cutoff_Hz: float, fs: float, order: int, reflect_s: float):
    b, a = butter(order, cutoff_Hz / (0.5 * fs), btype="low")
    x = np.asarray(x, float)
    n = int(min(len(x) - 1, round(reflect_s * fs)))
    if n < 3:
        return filtfilt(b, a, x)
    ext = 2.0 * x[0] - x[n:0:-1]
    return filtfilt(b, a, np.r_[ext, x])[n:]


def denoise(x, fs: float, f: dict):
    """(median-only trace, median + low-pass trace) for one channel's finite samples."""
    x = np.asarray(x, float)
    k = f["median_kernel"] if f["median_kernel"] % 2 == 1 else f["median_kernel"] - 1
    med = median_only(x, f["median_kernel"])
    if np.isfinite(x).sum() < k:
        return med, x
    if np.isfinite(med).sum() <= 3 * (f["lowpass_order"] + 1):
        return med, med
    return med, lowpass_onset_preserving(med, f["lowpass_cutoff_Hz"], fs,
                                         f["lowpass_order"], f["onset_reflection_s"])


def process(runs: dict, cfg: dict):
    """Fill run.median and run.filtered (same index as run.temp)."""
    f = cfg["filter"]
    for r in runs.values():
        df = r.temp
        t = pd.to_numeric(pd.Series(df.index), errors="coerce").to_numpy()
        fs = sampling_rate(df.index)
        med_df, filt_df = df.copy(), df.copy()
        for ch in cfg["channels"]["temperature"]:
            if ch not in df.columns:
                continue
            y = pd.to_numeric(df[ch], errors="coerce").to_numpy().copy()
            m = np.isfinite(t) & np.isfinite(y)
            if m.sum() > f["lowpass_order"] * 4:
                med, filt = denoise(y[m], fs, f)
                ym, yf = y.copy(), y.copy()
                ym[m], yf[m] = med, filt
                med_df[ch], filt_df[ch] = ym, yf
        r.median, r.filtered = med_df, filt_df


def exponential(t, A, k, C):
    return A * np.exp(-k * t) + C


def fit_exponential(t, y, min_samples: int = 20):
    """Fit T(t) = A exp(-k t) + C to t >= 0. Returns {A, k, tau, C, R2} or None.
    A single exponential is an empirical descriptor here: cooling through the
    sclera, vitreous and nerve is physically multi-exponential."""
    m = np.isfinite(t) & np.isfinite(y) & (t >= 0)
    if m.sum() < min_samples:
        return None
    x, v = t[m], y[m]
    try:
        p, _ = curve_fit(exponential, x, v, p0=[float(np.ptp(v)), 0.005, float(v.min())],
                         bounds=([0, 0, 0], [50, 1, 40]), maxfev=20000)
    except Exception:
        return None
    resid = v - exponential(x, *p)
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((v - v.mean()) ** 2)
    A, k, C = (float(q) for q in p)
    return {"A": A, "k": k, "tau": 1.0 / k if k > 0 else np.inf, "C": C, "R2": float(r2)}
