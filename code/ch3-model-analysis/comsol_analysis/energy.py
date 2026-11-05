"""Energy delivered, cooling efficiency and energy closure (notebook Step 8), plus an SCR
TEC-vs-interface energy balance the notebook does not compute (diagnostic only)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .io import read_comsol_table


def water_props(T_C, cfg):
    w = cfg["water"]
    return {p: float(np.interp(T_C, w["T_C"], w[p])) for p in ("nu", "rho", "k", "cp")}


def _integral_abs(t_min, q):
    t = pd.to_numeric(pd.Series(t_min), errors="coerce").values * 60.0
    q = pd.to_numeric(pd.Series(q), errors="coerce").values
    m = np.isfinite(t) & np.isfinite(q)
    return float(np.trapezoid(np.abs(q[m]), t[m])) if m.sum() >= 2 else np.nan


def scr_energies(reg, cfg):
    """Per (variant, silicone, interface): ∫|q| dt in J/m^2 (from W/m^2) and, full device,
    2 x ∫|q_W| dt in J (COMSOL's own area-integrated half-model power)."""
    M, S = reg.manifest, reg.signals
    f2 = cfg["geometry"]["half_model_factor"]
    # 2026-10-07: the exported interface heat rates are single-sided half-model values (checked against
    # COMSOL's stored boundary fluxes), so no halving; the flag is kept, switched off, for the record.
    fi = 0.5 if cfg["corrections"]["interface_flux_interior_double"] else 1.0
    areas = cfg["geometry"]["interface_areas_m2"]
    rows = []
    for sil in cfg["vocabulary"]["silicones"]:
        block = "scr_flux" if sil == "Baseline" else "scr_silicone_flux"
        for v in cfg["vocabulary"]["variants"]:
            for iface in cfg["vocabulary"]["interfaces"]:
                q = M[(M.block == block) & (M.variant == v) & (M.interface == iface)]
                if sil != "Baseline":
                    q = q[q.silicone == sil]
                if q.empty:
                    continue
                sig = S[q.iloc[0].run_id]
                e_d = fi * _integral_abs(sig.index, sig["q_total"])
                a = areas.get(iface, {}).get(v)
                if "q_total_W" in sig.columns:
                    e_J = fi * f2 * _integral_abs(sig.index, sig["q_total_W"])
                else:
                    e_J = e_d * a if (a and np.isfinite(e_d)) else np.nan
                rows.append({"variant": v, "silicone": sil, "interface": iface,
                             "energy_J_per_m2": e_d, "area_m2": a, "energy_J": e_J})
    return pd.DataFrame(rows)


def efficiencies(E):
    """η_eye = E_eye/E_total, η_body = (E_eye+E_eyelids)/E_total, Eye/Body, in both the J and the
    J/m^2 framings (the J/m^2 sums add densities over different areas — see README)."""
    out = []
    have_J = E["energy_J"].notna().all()
    for (v, sil), sub in E.groupby(["variant", "silicone"]):
        row = {"Variant": v, "Silicone": sil}
        for col, unit in (("energy_J_per_m2", "J/m^2"), ("energy_J", "J")):
            if unit == "J" and not have_J:
                continue
            e = {i: float(sub.loc[sub.interface == i, col].sum()) for i in ("Eye", "Eyelids", "Air")}
            tot, body = sum(e.values()), e["Eye"] + e["Eyelids"]
            row.update({f"E_eye ({unit})": e["Eye"], f"E_eyelids ({unit})": e["Eyelids"], f"E_air ({unit})": e["Air"],
                        f"eta_eye (%) [{unit}]": 100 * e["Eye"] / tot if tot > 0 else np.nan,
                        f"eta_body (%) [{unit}]": 100 * body / tot if tot > 0 else np.nan,
                        f"Eye/Body (%) [{unit}]": 100 * e["Eye"] / body if body > 0 else np.nan})
        out.append(row)
    return pd.DataFrame(out)


def area_check(E, tol=0.10):
    """Baseline only: area implied by energy_J / energy density vs the input full area."""
    rows = []
    for _, r in E[E.silicone == "Baseline"].iterrows():
        if not (np.isfinite(r.energy_J) and np.isfinite(r.energy_J_per_m2) and r.energy_J_per_m2 > 0 and r.area_m2):
            continue
        a_imp = r.energy_J / r.energy_J_per_m2
        rel = abs(a_imp / r.area_m2 - 1.0)
        rows.append({"variant": r.variant, "interface": r.interface, "input_area_m2": r.area_m2,
                     "implied_area_m2": a_imp, "rel_diff": rel, "ok": rel <= tol})
    return pd.DataFrame(rows)


def scr_energy_balance(reg, cfg):
    """DIAGNOSTIC (not in the notebook). The heat drawn through the eye, eyelid and air interfaces
    must leave through the TEC cold face (plus the ring's own sensible heat during the transient),
    so at steady state Σ|interface power| / |q_TEC| should be ~1. Both are scaled x2 by the
    notebook; this reports the ratio of the RAW (half-model) exports, which the x2 does not change.

    2026-10-07: the RAW exports give 2 because the TEC cold face is an interior boundary and the
    exported q_TEC is the mean of its two sides, one of them inert, i.e. half the heat through the
    face (correction tec_flux_interior_average).  With q_TEC doubled the ratio is 1 where every face
    of the ring is in the exported selections."""
    M, S = reg.manifest, reg.signals
    tf = 2.0 if cfg["corrections"].get("tec_flux_interior_average", False) else 1.0
    f2 = cfg["geometry"]["half_model_factor"]
    rows = []
    for tr_block, fx_block in (("scr_tr", "scr_flux"), ("scr_silicone_tr", "scr_silicone_flux")):
        for _, tr in M[M.block == tr_block].iterrows():
            s = S[tr.run_id]
            fx = M[(M.block == fx_block) & (M.variant == tr.variant) & (M.silicone == tr.silicone)]
            if fx.empty or "HeatFlux_TEC" not in s.columns:
                continue
            P = pd.DataFrame({r.interface: S[r.run_id]["q_total_W"] for _, r in fx.iterrows()}).abs().sum(axis=1)
            q = s["HeatFlux_TEC"].abs()
            t = s.index.values * 60
            steady = s.index >= s.index.max() - 5.0          # last 5 min
            ratio_ss = float((P[steady] / q[steady]).mean())
            rows.append({"variant": tr.variant, "silicone": tr.silicone,
                         "E_interfaces_raw_J": float(np.trapezoid(P.values, t)),
                         "E_TEC_raw_J": float(np.trapezoid(q.values, t)),
                         "energy_ratio": float(np.trapezoid(P.values, t) / np.trapezoid(q.values, t)),
                         "steady_power_ratio_last5min": ratio_ss,
                         "closure": ratio_ss / tf,
                         "closure_if_interfaces_halved": ratio_ss / tf,     # old column name, same number
                         "heat_not_in_interface_selections (%)": 100 * (1 - ratio_ss / tf),
                         "interface_power_end_W_raw": float(P.iloc[-1]), "q_TEC_end_W_raw": float(q.iloc[-1]),
                         "q_TEC_end_W_half_model": tf * float(q.iloc[-1]),
                         "q_TEC_end_W_full_device": f2 * tf * float(q.iloc[-1]),
                         "interface_power_end_W_full_device": f2 * float(P.iloc[-1])})
    return pd.DataFrame(rows)


def wb_dissipation(reg, cfg):
    """Per variant, over the turbulent sweep: coolant ΔT and P_absorbed = ρ Q0 cp ΔT at the last
    sample, water properties at each run's inlet temperature."""
    K = cfg["units"]["k_to_c"]
    rows = []
    for v in cfg["vocabulary"]["variants"]:
        per = []
        for _, r in reg.manifest[(reg.manifest.block == "wb_sweep") & (reg.manifest.variant == v)].iterrows():
            sig = reg.signals[r.run_id]
            if not {"INw", "OUTw"} <= set(sig.columns):
                continue
            tin, tout = sig["INw"].dropna(), sig["OUTw"].dropna()
            if tin.empty or tout.empty:
                continue
            dT = float(tout.iloc[-1] - tin.iloc[-1])
            wp = water_props((float(r.Tin_K) - K) if pd.notna(r.Tin_K) else cfg["water"]["fallback_T_C"], cfg)
            per.append((dT, wp["rho"] * float(r.Q0) * wp["cp"] * dT))
        if per:
            dT, P = np.array(per).T
            rows.append({"Variant": v, "Mean dT_coolant (K)": dT.mean(), "Max dT_coolant (K)": dT.max(),
                         "Mean P_absorbed (W)": P.mean(), "Max P_absorbed (W)": P.max(), "n_sweep_points": len(per)})
    return pd.DataFrame(rows)


WB_KEYS = ["Q0 (m^3/s)", "Tin (K)", "T0 (K)", "Qh (W/m^2)"]


def wb_closure(reg, cfg, per_point=False):
    """Hot-side heat in (Fine Energies, |alpha=1 hot-side W|) vs coolant enthalpy out
    (ρ Q0 cp ΔT from the Fine Auxiliary sweep at alpha = 1), matched on operating point.
    Ratio = mean(coolant) / mean(hot side), as plotted."""
    K = cfg["units"]["k_to_c"]
    fe = {r.variant: reg.signals[r.run_id] for _, r in reg.manifest[reg.manifest.block == "wb_fine_energies"].iterrows()}
    rows, points = [], []
    for v in cfg["vocabulary"]["variants"]:
        path = os.path.join(reg.data_dir, f"{v}_Auxiliary_Fine_Sweep.csv")
        if v not in fe or not os.path.exists(path):
            continue
        A = read_comsol_table(path)
        inw = next((c for c in A.columns if "INw" in c), None)
        outw = next((c for c in A.columns if "OUTw" in c), None)
        if inw is None or outw is None:
            continue
        ac = [c for c in A.columns if c.strip().lower().startswith("alpha")]
        A1 = (A[A[ac[0]].astype(float) == 1.0] if ac else A)[WB_KEYS + [inw, outw]].copy()
        Tin_C = A1["Tin (K)"].astype(float) - K
        rho = np.interp(Tin_C, cfg["water"]["T_C"], cfg["water"]["rho"])
        cp = np.interp(Tin_C, cfg["water"]["T_C"], cfg["water"]["cp"])
        A1["P_coolant_W"] = rho * A1["Q0 (m^3/s)"].astype(float) * cp * (A1[outw].astype(float) - A1[inw].astype(float))
        m = fe[v].merge(A1[WB_KEYS + ["P_coolant_W"]], on=WB_KEYS, how="inner")
        if m.empty:
            continue
        rows.append({"Variant": v, "n": len(m), "Q_h_W": m.HotSide_W.mean(), "P_coolant_W": m.P_coolant_W.mean(),
                     "ratio": m.P_coolant_W.mean() / m.HotSide_W.mean()})
        points.append(m.assign(Variant=v, ratio=m.P_coolant_W / m.HotSide_W))
    out = pd.DataFrame(rows)
    return (out, pd.concat(points, ignore_index=True) if points else pd.DataFrame()) if per_point else out
