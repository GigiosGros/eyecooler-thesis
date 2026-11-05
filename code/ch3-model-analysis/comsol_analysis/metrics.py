"""Per-trace metrics (notebook Step 4d, Step 12 master table, Step 13 summaries)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.optimize as spo

# Which manifest blocks each study covers (notebook Step 3e, STUDY_MAP).
STUDY_MAP = {
    "SCR": {"Parametric Sweep": ["scr_parametric"], "Material Sweep": ["scr_material"],
            "Silicone Sweep": ["scr_silicone_internal"], "ReHeat": ["scr_reheat"]},
    "Waterblock": {"Auxiliary Sweep": ["wb_aux"], "Turbulent": ["wb_sweep"]},
}


def exp_model(t, T_inf, T_0, tau):
    return T_inf + (T_0 - T_inf) * np.exp(-t / tau)


def fit_exp_decay(t, y, cfg):
    """T(t) = T_inf + (T_0 - T_inf) exp(-t/tau); T bounds = data range ± half-span, clamped."""
    mc = cfg["metrics"]
    t, y = np.asarray(t, float), np.asarray(y, float)
    msk = (t >= mc["fit_skip_first_min"]) & np.isfinite(y)
    nan = dict(T_inf=np.nan, T_0=np.nan, tau=np.nan, rmse=np.nan, r2=np.nan, n_pts=int(msk.sum()), bound_hit=False)
    if msk.sum() < 5:
        return nan
    ymn, ymx = float(np.nanmin(y[msk])), float(np.nanmax(y[msk]))
    span = max(ymx - ymn, 1.0)
    Tlo = max(mc["fit_T_bounds_C"][0], ymn - 0.5 * span)
    Thi = min(mc["fit_T_bounds_C"][1], ymx + 0.5 * span)
    lo = [Tlo, Tlo, mc["fit_tau_bounds_min"][0]]
    hi = [Thi, Thi, mc["fit_tau_bounds_min"][1]]
    try:
        p0 = [float(np.clip(ymn, Tlo, Thi)), float(np.clip(y[msk][0], Tlo, Thi)), 10.0]
        popt, _ = spo.curve_fit(exp_model, t[msk], y[msk], p0=p0, bounds=(lo, hi), maxfev=4000)
        yh = exp_model(t[msk], *popt)
        ss_res = float(np.sum((y[msk] - yh) ** 2))
        ss_tot = float(np.sum((y[msk] - y[msk].mean()) ** 2))
        tol = 1e-3 * max(Thi - Tlo, 1.0)
        return dict(T_inf=popt[0], T_0=popt[1], tau=popt[2], rmse=float(np.sqrt(ss_res / msk.sum())),
                    r2=(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan, n_pts=int(msk.sum()),
                    bound_hit=bool(min(abs(popt[0] - Tlo), abs(popt[0] - Thi)) < tol))
    except Exception:  # noqa: BLE001 — a failed fit is reported as NaN, as in the notebook
        return nan


def time_to_value(t, y, target, decreasing=True):
    """First t at which y reaches target (<= if decreasing, >= otherwise). NaN if never."""
    t, y = np.asarray(t), np.asarray(y)
    idx = np.where(y <= target)[0] if decreasing else np.where(y >= target)[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def time_to_therapeutic(t, y, target, decreasing, cfg):
    """(time, starts_beyond). The notebook returns t[0] (= 0) for a trace that is already past
    the threshold at t = 0; with the correction on that is NaN and flagged instead."""
    t, y = np.asarray(t), np.asarray(y)
    beyond = bool(y[0] <= target) if decreasing else bool(y[0] >= target)
    if beyond and cfg["corrections"]["therapeutic_nan_if_start_beyond"]:
        return np.nan, True
    return time_to_value(t, y, target, decreasing), beyond


def time_to_equilibrium(t, y, cfg):
    """First t after which |dy/dt| stays below the rate for a whole window.
    Notebook: the window may run past the end of the trace (then fewer samples are checked);
    corrected: the window must end inside the trace."""
    rate = cfg["metrics"]["equilibrium_rate_K_per_min"]
    win = cfg["metrics"]["equilibrium_window_min"]
    full = cfg["corrections"]["equilibrium_full_window"]
    t, y = np.asarray(t), np.asarray(y)
    if len(t) < 4:
        return np.nan
    abs_d = np.abs(np.gradient(y, t))
    # A site that lags (the cold has not arrived yet) is flat at the start: that is not
    # equilibrium. Corrected: a flat window counts only once most of the change has happened.
    total = y[-1] - y[0]
    frac = cfg["metrics"].get("equilibrium_min_change_fraction", 0.9)
    lagged = cfg["corrections"].get("equilibrium_after_change", False) and abs(total) > 1e-6
    for i in range(len(t)):
        if lagged and abs(y[i] - y[0]) < frac * abs(total):
            continue
        end_t = t[i] + win
        if full and end_t > t[-1] + 1e-9:
            break
        msk = (t >= t[i]) & (t <= end_t)
        if not msk.any():
            break
        if abs_d[msk].max() < rate:
            return float(t[i])
    return np.nan


def cooling_rate(t, y):
    """Mean dT/dt over the 10–90 % band of the total change (K/min)."""
    t, y = np.asarray(t), np.asarray(y)
    y0, yf = y[0], y[-1]
    if not np.isfinite(y0) or not np.isfinite(yf) or y0 == yf:
        return np.nan
    lo, hi = y0 + 0.1 * (yf - y0), y0 + 0.9 * (yf - y0)
    msk = ((y <= lo) & (y >= hi)) if y0 > yf else ((y >= lo) & (y <= hi))
    if msk.sum() < 2:
        return np.nan
    return float((y[msk][-1] - y[msk][0]) / (t[msk][-1] - t[msk][0]))


def metrics_for_series(t_min, y_vals, signal_kind, cfg):
    """Step 12 row for one (run, probe)."""
    th = cfg["thresholds"]
    y = pd.to_numeric(pd.Series(y_vals), errors="coerce").dropna()
    if len(y) < 2:
        return None
    x, yv = np.asarray(t_min)[y.index], y.values
    T0_run, Tf = float(yv[0]), float(yv[-1])
    dT = T0_run - Tf
    base = {"Tf_simulated (C)": Tf, "dT (K)": dT,
            "safety_hot?": bool(np.any(yv > th["safety_hot_C"])),
            "safety_cold?": bool(np.any(yv < th["safety_cold_C"]))}
    nan_rest = {"t50 (min)": np.nan, "t90 (min)": np.nan, "t->therapeutic (min)": np.nan,
                "starts_beyond_therapeutic?": np.nan, "cooling rate (K/min)": np.nan,
                "T_inf (C)": np.nan, "t_eq (min)": np.nan}
    if signal_kind == "steady_state":
        return {**base, **nan_rest}
    if len(y) < 4:
        return None
    t_th, beyond = time_to_therapeutic(x, yv, th["therapeutic_C"], dT > 0, cfg)
    return {**base,
            "t50 (min)": time_to_value(x, yv, T0_run - 0.5 * dT, decreasing=(dT > 0)),
            "t90 (min)": time_to_value(x, yv, T0_run - 0.9 * dT, decreasing=(dT > 0)),
            "t->therapeutic (min)": t_th,
            "starts_beyond_therapeutic?": beyond,
            "cooling rate (K/min)": cooling_rate(x, yv),
            "T_inf (C)": fit_exp_decay(x, yv, cfg)["T_inf"],
            "t_eq (min)": time_to_equilibrium(x, yv, cfg)}


_SCR_PARAMS = {"T0_K": "T0_input (C)", "Tb_K": "Tb_input (C)", "Tc_K": "Tc_input (C)",
               "material": "material", "silicone": "silicone"}
_WB_PARAMS = {"Q0": "Q0 (m^3/s)", "Tin_K": "Tin_input (C)", "T0_K": "T0_input (C)", "Qh": "Qh"}


def master_table(reg, family, study, cfg, variants=None):
    """Step 12 table for one (family, study): one row per (run, probe)."""
    k = cfg["units"]["k_to_c"]
    M = reg.manifest
    sub = M[M["block"].isin(STUDY_MAP[family][study])]
    sub = sub[sub["variant"].isin(variants or cfg["vocabulary"]["variants"])]
    if study == "Auxiliary Sweep":
        sub = sub[sub["mesh"] == "Fine"]
    params = _SCR_PARAMS if family == "SCR" else _WB_PARAMS
    chans = (cfg["probes"]["scr_point"] + cfg["probes"]["retina_agg"] + cfg["probes"]["eye_agg"]
             if family == "SCR" else cfg["probes"]["waterblock"])
    rows = []
    for _, r in sub.iterrows():
        sig = reg.signals[r["run_id"]]
        meta = {"run_id": r["run_id"], "variant": r["variant"], "signal_kind": r["signal_kind"]}
        for k_in, k_out in params.items():
            v = r[k_in] if k_in in r else np.nan
            meta[k_out] = v - k if (k_in.endswith("_K") and pd.notna(v)) else v
        for ch in chans:
            if ch not in sig.columns:
                continue
            m = metrics_for_series(sig.index.values, sig[ch].values, r["signal_kind"], cfg)
            if m is not None:
                rows.append({**meta, "probe": ch, **m})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ids = ["run_id", "variant", "signal_kind", "probe"]
    ps = [c for c in params.values() if c in df.columns]
    return df[ids + ps + [c for c in df.columns if c not in ids + ps]]


def master_metrics(reg, cfg):
    """Step 14 'Master_Metrics': every (family, study)."""
    chunks = []
    for family, studies in STUDY_MAP.items():
        for study in studies:
            df = master_table(reg, family, study, cfg)
            if not df.empty:
                df.insert(0, "study", study)
                df.insert(0, "family", family)
                chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def thermal_summary_run(sig, cfg):
    """Step 13 per-group table for ONE run: T0, Tf, ΔT, t->35 °C, cooling rate per channel.
    Notebook: the 35 °C crossing is always searched downward; corrected: in the trace's direction."""
    th = cfg["thresholds"]["therapeutic_C"]
    t = sig.index.values
    rows = []
    for ch in sig.columns:
        y = pd.to_numeric(sig[ch], errors="coerce").dropna().values
        if len(y) < 4:
            continue
        T0, Tf = float(y[0]), float(y[-1])
        dec = (T0 - Tf > 0) if cfg["corrections"]["therapeutic_direction_step13"] else True
        t_th, beyond = time_to_therapeutic(t[:len(y)], y, th, dec, cfg)
        rows.append({"probe": ch, "T0 (C)": T0, "Tf (C)": Tf, "dT (K)": T0 - Tf,
                     "t->therapeutic (min)": t_th, "starts_beyond_therapeutic?": beyond,
                     "cooling rate (K/min)": cooling_rate(t[:len(y)], y)})
    return pd.DataFrame(rows)
