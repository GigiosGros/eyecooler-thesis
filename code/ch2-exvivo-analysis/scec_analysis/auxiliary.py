"""Auxiliary recordings: intraocular pressure (human runs), TEC drive current and
room temperature. Descriptive only; they do not enter the thermal statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .processing import median_only, sampling_rate


def _smooth(y, fs, kernel, cutoff):
    y = np.asarray(y, float)
    if fs <= 0 or len(y) < kernel * 4:
        return y
    b, a = butter(2, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, median_only(y, kernel))


def pressure_summary(runs: dict, cfg: dict) -> pd.DataFrame:
    a = cfg["auxiliary"]
    rows = []
    for rid, r in runs.items():
        if r.pressure is None:
            continue
        df = r.pressure
        fs = 1.0 / np.median(np.diff(df.index)) if len(df) > 1 else 1.0
        row = {"run_id": rid, "group": " · ".join(r.group), "n_samples": len(df),
               "offset_A_V": r.pressure_offsets.get("A"), "offset_B_V": r.pressure_offsets.get("B")}
        for ch in df.columns:
            yf = _smooth(pd.to_numeric(df[ch], errors="coerce").to_numpy(), fs,
                         a["pressure_median_kernel"], a["pressure_cutoff_Hz"])
            row[f"mean_{ch}_PSI"] = float(np.nanmean(yf))
        if {"Inflow", "Outflow"} <= set(df.columns):
            row["mean_dP_PSI"] = row["mean_Inflow_PSI"] - row["mean_Outflow_PSI"]
        rows.append(row)
    return pd.DataFrame(rows)


def current_summary(runs: dict, cfg: dict) -> pd.DataFrame:
    """TEC drive current (A): median-filtered mean over the run, and the mean over
    the first 60 s (maximum drive) and the last 60 s (holding current)."""
    a, cur = cfg["auxiliary"], cfg["channels"]["current"]
    rows = []
    for rid, r in runs.items():
        if cur not in r.temp.columns:
            continue
        t = pd.to_numeric(pd.Series(r.temp.index), errors="coerce").to_numpy()
        y = pd.to_numeric(r.temp[cur], errors="coerce").to_numpy()
        m = np.isfinite(t) & np.isfinite(y)
        if m.sum() < 100:
            continue
        t, y = t[m], median_only(y[m], a["current_median_kernel"])
        rows.append({"run_id": rid, "group": " · ".join(r.group),
                     "mean_A": float(y.mean()),
                     "first_60s_A": float(y[t <= 60].mean()),
                     "last_60s_A": float(y[t >= t.max() - 60].mean())})
    return pd.DataFrame(rows)


def ambient_summary(runs: dict) -> pd.DataFrame:
    rows = [{"run_id": rid, "group": " · ".join(r.group), "room_temp_C": r.room_temp_C}
            for rid, r in runs.items() if r.room_temp_C is not None]
    return pd.DataFrame(rows)


def group_means(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    g = df.groupby("group")[value_cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    out.insert(0, "n_runs", df.groupby("group").size())
    return out.reset_index()
