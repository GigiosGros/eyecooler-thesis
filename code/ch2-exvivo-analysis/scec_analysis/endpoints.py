"""Per-trace endpoints and group summaries.

For each included trace (filtered):
  T_initial   first sample (t = 0; exact under the onset-preserving filter)
  T_final     mean over the final window (qc.final_window_s; 240 s = minutes
              16-20, after every trace's t90; notebook v4.28 used 15 s)
  dT          T_initial - T_final (primary endpoint; + = cooling)
  t50, t90    first time the trace falls by 50 % / 90 % of dT
  rate_5min   mean cooling rate over the first 300 s
  plateau_SD  SD of the median-only trace (fluctuation on the cooled plateau; the
              low-pass trace would hide it) over qc.plateau_window: "final" = the
              same final window as T_final (default), "t90" = from t90 to the end
              (notebook v4.28; includes the last 10 % of the cooling slope)
  plateau_band  1st-to-99th percentile range of the median-only trace over the final
              window (always the T_final window): for the ring under comparator
              control this is the switching band of the controller (added 2026-10-08)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _xy(df: pd.DataFrame, ch: str):
    t = pd.to_numeric(pd.Series(df.index), errors="coerce").to_numpy()
    y = pd.to_numeric(df[ch], errors="coerce").to_numpy()
    m = np.isfinite(t) & np.isfinite(y)
    return t[m], y[m]


def trace_kinetics(run, ch: str, window_s: float, plateau_window: str = "t90"):
    if ch not in run.filtered.columns:
        return None
    t, v = _xy(run.filtered, ch)
    if len(t) < 10:
        return None
    t0 = float(v[0])
    w = t >= t.max() - window_s
    tf = float(v[w].mean()) if w.sum() >= 3 else float(v[-1])
    dT = t0 - tf

    def crossing(frac):
        ix = np.where(v <= t0 - frac * dT)[0]
        return float(t[ix[0]]) if len(ix) else np.nan

    t50, t90 = crossing(0.5), crossing(0.9)
    m300 = t <= 300
    rate = (t0 - v[m300][-1]) / t[m300][-1] if m300.sum() > 5 else np.nan
    plat_sd = band = np.nan
    if np.isfinite(t90) and ch in run.median.columns:
        tm, vm = _xy(run.median, ch)
        start = (tm.max() - window_s) if plateau_window == "final" else t90
        plat = vm[tm >= start]
        plat_sd = float(np.std(plat, ddof=1)) if len(plat) > 10 else np.nan
        last = vm[tm >= tm.max() - window_s]
        if len(last) > 10:
            band = float(np.percentile(last, 99) - np.percentile(last, 1))
    return {"T_initial": t0, "T_final": tf, "dT": dT, "t50": t50, "t90": t90,
            "rate_5min": rate, "plateau_SD": plat_sd, "plateau_band": band}


def included(run, ch: str) -> bool:
    return not run.exclusions.get(ch)


def kinetics_table(runs: dict, cfg: dict) -> pd.DataFrame:
    """One row per included (run, channel)."""
    rows = []
    for rid, r in runs.items():
        for ch in cfg["channels"]["temperature"]:
            if not included(r, ch):
                continue
            k = trace_kinetics(r, ch, cfg["qc"]["final_window_s"], cfg["qc"].get("plateau_window", "t90"))
            if k:
                rows.append({"run_id": rid, "species": r.species, "status": r.status,
                             "ring_material": r.ring_material, "ring": r.ring,
                             "eye": r.eye_id, "channel": ch, **k})
    return pd.DataFrame(rows)


def group_summary(kin: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mean (SD) across runs per group and channel, with CV of dT."""
    out = []
    keys = ["species", "status", "ring_material", "ring"]
    for key, g in kin.groupby(keys, sort=False):
        for ch in cfg["channels"]["temperature"]:
            s = g[g["channel"] == ch]
            if s.empty:
                continue
            row = dict(zip(keys, key), channel=ch, n=len(s))
            for col in ["T_initial", "T_final", "dT", "t50", "t90", "rate_5min", "plateau_SD", "plateau_band"]:
                v = s[col].dropna()
                row[f"{col}_mean"] = float(v.mean()) if len(v) else np.nan
                row[f"{col}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            row["CV_pct"] = abs(row["dT_sd"] / row["dT_mean"]) * 100 if row["dT_mean"] else 0.0
            out.append(row)
    return pd.DataFrame(out)


def endpoint_per_run(runs: dict, ids: list[str], ch: str, window_s: float) -> np.ndarray:
    """dT of every included trace of `ch` in the runs `ids` (Welch comparisons)."""
    out = []
    for rid in ids:
        r = runs[rid]
        if not included(r, ch) or ch not in r.filtered.columns:
            continue
        t, v = _xy(r.filtered, ch)
        if len(t) < 10:
            continue
        w = t >= t.max() - window_s
        if w.sum() >= 3:
            out.append(float(v[0]) - float(v[w].mean()))
    return np.asarray(out, float)


def endpoint_by_eye(runs: dict, ids: list[str], ch: str, window_s: float) -> dict:
    """{(species, status, eye): mean dT} — repeats of one eye averaged, so each
    specimen contributes once (and within-eye comparisons can be paired)."""
    by = {}
    for rid in ids:
        r = runs[rid]
        if not included(r, ch) or ch not in r.filtered.columns:
            continue
        t, v = _xy(r.filtered, ch)
        if len(t) < 10:
            continue
        w = t >= t.max() - window_s
        if w.sum() >= 3:
            by.setdefault((r.species, r.status, r.eye_id), []).append(float(v[0]) - float(v[w].mean()))
    return {k: float(np.mean(v)) for k, v in by.items()}
