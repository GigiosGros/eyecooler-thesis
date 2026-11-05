"""Thermal resistance (notebook Step 11).

SCR:  R_th(t) = (T_hot - T_cold) / |q_TEC|, q_TEC being the half-model TEC power; the
      full-device value is R_half / 2 (the notebook REPORTS the full-device value but its
      Step 14 export wrote R_half under a bare 'R_steady' — both are labelled here).
WB:   R_WB = (T_TEC - (T_INw + T_OUTw)/2) / (ρ Q0 cp ΔT_coolant) at the last sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .energy import water_props


def scr_thermal_resistance(reg, variant, cfg):
    M = reg.manifest
    sub = M[(M.block == "scr_tr") & (M.variant == variant)]
    if sub.empty:
        return None
    ch = {}
    for _, r in sub.iterrows():
        sig = reg.signals[r.run_id]
        for c in r.channels:
            if c in sig.columns:
                ch[c] = sig[c].dropna()
    if any(k not in ch for k in ("Thot", "Tcold", "HeatFlux_TEC")):
        return None
    Tc = ch["Tcold"]
    t = Tc.index.values
    Th = np.interp(t, ch["Thot"].index.values, ch["Thot"].values)
    Q = np.interp(t, ch["HeatFlux_TEC"].index.values, ch["HeatFlux_TEC"].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (Th - Tc.values) / np.where(np.abs(Q) < 1e-12, np.nan, np.abs(Q))
    n = len(R)
    n_ss = max(1, int(n * cfg["resistance"]["steady_fraction"]))
    Rf = R[np.isfinite(R)]
    return {"df": pd.DataFrame({"Thot": Th, "Tcold": Tc.values, "HeatFlux_TEC": Q, "R_thermal_half": R},
                               index=pd.Index(t, name="Time (min)")),
            "R_steady": float(np.nanmean(R[-n_ss:])) if np.isfinite(R[-n_ss:]).any() else np.nan,
            "t_ss_start": t[-n_ss] if n_ss <= n else t[0],
            "R_overall_mean": float(np.nanmean(Rf)) if Rf.size else np.nan,
            "R_min": float(np.nanmin(Rf)) if Rf.size else np.nan,
            "R_max": float(np.nanmax(Rf)) if Rf.size else np.nan}


def scr_rth_table(reg, cfg):
    f2 = cfg["geometry"]["half_model_factor"]
    # 2026-10-07: the exported q_TEC is half the heat through the half model's TEC face (interior
    # boundary, mean of two sides), so the half-model resistance is the raw ratio / 2 and the
    # full-device value the raw ratio / 4.  (The notebook reported raw / 2 as "full device".)
    tf = 2.0 if cfg["corrections"].get("tec_flux_interior_average", False) else 1.0
    rows, series = [], {}
    for v in cfg["vocabulary"]["variants"]:
        res = scr_thermal_resistance(reg, v, cfg)
        if res is None:
            continue
        series[v] = res["df"]
        rows.append({"Variant": v, "R_steady_full_device (K/W)": res["R_steady"] / (f2 * tf),
                     "R_steady_half_model (K/W)": res["R_steady"] / tf,
                     "R_steady_raw_export (K/W)": res["R_steady"],
                     "R_overall_mean_half (K/W)": res["R_overall_mean"],
                     "R_min_half (K/W)": res["R_min"], "R_max_half (K/W)": res["R_max"],
                     "steady_window_start (min)": res["t_ss_start"]})
    return pd.DataFrame(rows), series


def wb_thermal_resistance(reg, cfg):
    K = cfg["units"]["k_to_c"]
    rows = []
    for _, r in reg.manifest[reg.manifest.block == "wb_sweep"].iterrows():
        sig = reg.signals[r.run_id]
        if not {"INw", "OUTw", "TEC"} <= set(sig.columns):
            continue
        try:
            T_in, T_out, T_tec = (float(pd.to_numeric(sig[c], errors="coerce").dropna().iloc[-1])
                                  for c in ("INw", "OUTw", "TEC"))
        except Exception:  # noqa: BLE001
            continue
        Q0 = float(r.Q0) if pd.notna(r.Q0) else None
        if not Q0 or abs(T_out - T_in) < 1e-6:
            continue
        wp = water_props((r.Tin_K - K) if pd.notna(r.Tin_K) else cfg["water"]["fallback_T_C"], cfg)
        P = wp["rho"] * Q0 * wp["cp"] * (T_out - T_in)
        if abs(P) < 1e-9:
            continue
        Tw = (T_in + T_out) / 2
        rows.append({"Variant": r.variant, "Q0 (m^3/s)": Q0,
                     "Tin (C)": r.Tin_K - K if pd.notna(r.Tin_K) else np.nan,
                     "T0 (C)": r.T0_K - K if pd.notna(r.T0_K) else np.nan, "Qh": r.Qh,
                     "T_w_mean (C)": Tw, "T_TEC (C)": T_tec, "P_absorbed (W)": P, "R_WB (K/W)": (T_tec - Tw) / P})
    return pd.DataFrame(rows)


def wb_study_consistency(reg, cfg):
    """DIAGNOSTIC (not in the notebook). At every operating point present in BOTH waterblock
    studies — the turbulent transient sweep (last sample; the runs are flat after ~1 min) and the
    stationary Fine auxiliary sweep (alpha = 1) — compare coolant ΔT, TEC temperature and R_WB."""
    K = cfg["units"]["k_to_c"]
    M, S = reg.manifest, reg.signals
    fine = M[(M.block == "wb_aux") & (M.mesh == "Fine")]
    rows = []
    for _, r in M[M.block == "wb_sweep"].iterrows():
        f = fine[(fine.variant == r.variant) & np.isclose(fine.Q0, r.Q0) & np.isclose(fine.Tin_K, r.Tin_K)
                 & np.isclose(fine.T0_K, r.T0_K) & np.isclose(fine.Qh, r.Qh)]
        if f.empty:
            continue
        s, a = S[r.run_id].iloc[-1], S[f.iloc[0].run_id].loc[1.0]
        wp = water_props(r.Tin_K - K, cfg)
        row = {"Variant": r.variant, "Q0 (m^3/s)": r.Q0, "Tin (C)": r.Tin_K - K, "T0 (C)": r.T0_K - K, "Qh": r.Qh}
        for lab, x in (("sweep", s), ("stationary", a)):
            P = wp["rho"] * r.Q0 * wp["cp"] * (x.OUTw - x.INw)
            Tw = (x.OUTw + x.INw) / 2
            row[f"dT_coolant_{lab} (K)"] = x.OUTw - x.INw
            row[f"T_TEC_{lab} (C)"] = x.get("TEC", np.nan)
            row[f"R_WB_{lab} (K/W)"] = (x.get("TEC", np.nan) - Tw) / P if P else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def wb_rth_summary(df):
    if df.empty:
        return df
    return df.groupby("Variant").agg(n_runs=("R_WB (K/W)", "count"), R_mean=("R_WB (K/W)", "mean"),
                                     R_median=("R_WB (K/W)", "median"), R_std=("R_WB (K/W)", "std"),
                                     R_min=("R_WB (K/W)", "min"), R_max=("R_WB (K/W)", "max"),
                                     P_mean_W=("P_absorbed (W)", "mean")).reset_index()
