"""Waterblock sizing metrics (added).

1. Flow-model / mesh check (Model_data/flow_model_study/: V{n} {Coarse,Normal,Fine} {Laminar,Turbulent}.csv,
   one operating point, 3.5 min transient): final TEC / waterblock temperatures per flow model and mesh,
   and three-grid GCI per flow model. Shows which flow model is mesh-converged.
2. Hot-side resistance from the mesh-converged LAMINAR stationary Fine sweep (alpha = 1):
     R_TEC = (T_TEC − T_water,mean) / Q_h        (V1, V2 — V3 has no TEC probe)
     R_WB  = (T_WB,avg − T_water,mean) / Q_h     (all variants)
     rise  = (T_TEC − T_in) / Q_h                 K per W above the coolant inlet
   Q_h = integrated hot-side heat (Fine Energies), matched on operating point.
Not available: pressure drop / pumping power (not exported).
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .energy import WB_KEYS
from .io import read_comsol_table
from .mesh import calculate_gci, mesh_band


def flow_model_study(data_dir, cfg):
    folder = os.path.join(data_dir, "flow_model_study")
    if not os.path.isdir(folder):
        return pd.DataFrame(), pd.DataFrame()
    K = cfg["units"]["k_to_c"]
    rows = []
    for f in sorted(os.listdir(folder)):
        m = re.match(r"(V\d) (Coarse|Normal|Fine) (Laminar|Turbulent)\.csv$", f)
        if not m:
            continue
        lines = open(os.path.join(folder, f)).read().splitlines()
        hi = max(i for i, l in enumerate(lines[:12]) if l.startswith("%"))
        df = pd.read_csv(os.path.join(folder, f), skiprows=hi)
        df.columns = [c.lstrip("% ").strip() for c in df.columns]
        last = df.iloc[-1]
        get = lambda k: next((float(last[c]) - K for c in df.columns if k in c), np.nan)  # noqa: E731
        vals = {p: get(p) for p in ("TEC", "INw", "OUTw", "Waterblock")}
        issue = ""
        if float(last[df.columns[0]]) < 3.5 - 1e-6:
            issue = f"run stops at {float(last[df.columns[0]]):g} min"
        elif np.nanstd([vals["INw"], vals["OUTw"], vals["Waterblock"]]) < 1e-9:
            issue = "all probes identical (export error)"
        rows.append({"variant": m.group(1), "mesh": m.group(2), "flow": m.group(3), "t_end (min)": float(last[df.columns[0]]),
                     **{f"{p} (C)": v for p, v in vals.items()}, "issue": issue})
    runs = pd.DataFrame(rows)
    gci = []
    order, dofs = cfg["mesh"]["wb_order"], cfg["mesh"]["dof_wb"]
    for (v, flow), g in runs.groupby(["variant", "flow"]):
        g = g.set_index("mesh")
        if not set(order) <= set(g.index) or (g.issue != "").any():
            gci.append({"variant": v, "flow": flow, "probe": "-", "note": "not computable: " + "; ".join(i for i in g.issue if i)})
            continue
        for p in ("TEC", "Waterblock"):
            f = [g.loc[m, f"{p} (C)"] for m in order]
            if any(not np.isfinite(x) for x in f):
                continue
            b = mesh_band(f[2], f[1], f[0], dofs[v][order[2]], dofs[v][order[1]], dofs[v][order[0]], cfg)
            gci.append({"variant": v, "flow": flow, "probe": p, **{f"T_{m} (C)": f[i] for i, m in enumerate(order)},
                        "Normal-Fine (K)": f[1] - f[2], "p": b["p"], "p_used": b["p_used"], "GCI (%)": b["pct"],
                        "U_mesh (K)": b["U"], "band_method": b["method"],
                        "note": b["asymp"] if isinstance(b["asymp"], str) else ""})
    return runs, pd.DataFrame(gci)


def _difference_row(variant, study, name, f, dofs, order, cfg):
    """Mesh band of a surface-minus-coolant temperature difference (f in mesh `order`)."""
    b = mesh_band(f[2], f[1], f[0], dofs[order[2]], dofs[order[1]], dofs[order[0]], cfg)
    return {"variant": variant, "study": study, "difference": name,
            **{f"dT_{m} (C)": f[i] for i, m in enumerate(order)},
            "change Normal to Fine (%)": 100 * (f[2] - f[1]) / f[1] if abs(f[1]) > 1e-12 else np.nan,
            "p": b["p"], "p_used": b["p_used"], "band (% of dT)": b["pct"], "U_mesh (C)": b["U"],
            "band_method": b["method"],
            "status": "DIVERGING" if b["div"] else ("NON-MONOTONIC" if b["osc"] else
                                                    "CONVERGED" if b["pct"] < 10 else "NOT CONVERGED")}


def difference_mesh_band(reg, flow_runs, cfg):
    """Mesh band of the quantities the hot-side resistances are built from: the difference
    between a surface temperature and the mean coolant temperature, (T_surface - T_water).
    The GCI of gci_table is taken on the absolute temperatures, where a change of a few
    hundredths of a degree is a fraction of a percent; on a difference of 0.04-0.7 degC the
    same change is large. A resistance R = dT / Q_h inherits the band of its dT (Q_h is
    imposed). Stationary auxiliary study at the mesh reference point, and the laminar and
    turbulent transient flow-model study."""
    from .mesh import final_value
    order, rows = cfg["mesh"]["wb_order"], []
    for v in cfg["vocabulary"]["variants"]:
        dofs = cfg["mesh"]["dof_wb"].get(v)
        if not dofs:
            continue
        T = {ch: [final_value(reg, "Waterblock", v, m, ch, cfg) for m in order]
             for ch in ("TEC", "Waterblock", "INw", "OUTw")}
        ok = lambda ch: all(x is not None and np.isfinite(x) for x in T[ch])  # noqa: E731
        if ok("INw") and ok("OUTw"):
            Tw = [(a + b) / 2 for a, b in zip(T["INw"], T["OUTw"])]
            for ch, name in (("TEC", "TEC face - water"), ("Waterblock", "block surface - water")):
                if ok(ch):
                    rows.append(_difference_row(v, "stationary auxiliary", name,
                                                [t - w for t, w in zip(T[ch], Tw)], dofs, order, cfg))
        if flow_runs is None or flow_runs.empty:
            continue
        for flow, g in flow_runs[flow_runs.variant == v].groupby("flow"):
            g = g.set_index("mesh")
            if not set(order) <= set(g.index) or (g.issue != "").any():
                continue
            Tw = [(g.loc[m, "INw (C)"] + g.loc[m, "OUTw (C)"]) / 2 for m in order]
            for ch, name in (("TEC", "TEC face - water"), ("Waterblock", "block surface - water")):
                f = [g.loc[m, f"{ch} (C)"] - w for m, w in zip(order, Tw)]
                if all(np.isfinite(x) for x in f):
                    rows.append(_difference_row(v, f"flow model, {flow.lower()}", name, f, dofs, order, cfg))
    return pd.DataFrame(rows)


def hot_side_resistance(reg, cfg):
    """Per operating point of the laminar stationary Fine sweep."""
    K = cfg["units"]["k_to_c"]
    fe = {r.variant: reg.signals[r.run_id] for _, r in reg.manifest[reg.manifest.block == "wb_fine_energies"].iterrows()}
    rows = []
    for v in cfg["vocabulary"]["variants"]:
        path = os.path.join(reg.data_dir, f"{v}_Auxiliary_Fine_Sweep.csv")
        if v not in fe or not os.path.exists(path):
            continue
        A = read_comsol_table(path)
        A = A[A[[c for c in A.columns if c.lower().startswith("alpha")][0]].astype(float) == 1.0]
        col = lambda k: next((c for c in A.columns if k in c and "Boundary Probe" in c), None)  # noqa: E731
        m = fe[v].merge(A, on=WB_KEYS)
        Tin, Tout, Twb = m[col("INw")] - K, m[col("OUTw")] - K, m[col("Waterblock")] - K
        Ttec = m[col("TEC")] - K if col("TEC") else np.nan
        Tw = (Tin + Tout) / 2
        Q = m.HotSide_W
        rows.append(pd.DataFrame({"Variant": v, "Q0 (m^3/s)": m["Q0 (m^3/s)"], "Q0 (L/min)": m["Q0 (m^3/s)"] * 6e4,
                                  "Tin (C)": m["Tin (K)"] - K, "T0 (C)": m["T0 (K)"] - K, "Qh (W/m^2)": m["Qh (W/m^2)"],
                                  "Q_h (W)": Q, "dT_coolant (K)": Tout - Tin, "T_TEC (C)": Ttec, "T_WB_avg (C)": Twb,
                                  "R_TEC (K/W)": (Ttec - Tw) / Q, "R_WB (K/W)": (Twb - Tw) / Q,
                                  "TEC_rise_above_inlet (K/W)": (Ttec - Tin) / Q,
                                  "WB_rise_above_inlet (K/W)": (Twb - Tin) / Q}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def resistance_vs_flow(df):
    """R vs flow rate (mean and range over Tin, T0, Qh at each Q0)."""
    if df.empty:
        return df
    cols = ["R_TEC (K/W)", "R_WB (K/W)", "TEC_rise_above_inlet (K/W)", "WB_rise_above_inlet (K/W)"]
    g = df.groupby(["Variant", "Q0 (L/min)"])
    out = g[cols].mean().add_suffix(" mean")
    out = out.join(g[cols].min().add_suffix(" min")).join(g[cols].max().add_suffix(" max"))
    out["n_points"] = g.size()
    return out.reset_index()


def max_tec_temperature(df, cfg):
    """Hottest TEC/waterblock surface over the sweep per variant vs the 43 °C applied-part limit."""
    lim = cfg["thresholds"]["safety_hot_C"]
    rows = []
    for v, g in df.groupby("Variant"):
        for p in ("T_TEC (C)", "T_WB_avg (C)"):
            if g[p].notna().any():
                i = g[p].idxmax()
                rows.append({"Variant": v, "probe": p, "max (C)": g.loc[i, p], "at Q0 (L/min)": g.loc[i, "Q0 (L/min)"],
                             "at Tin (C)": g.loc[i, "Tin (C)"], "at Q_h (W)": g.loc[i, "Q_h (W)"], "margin_to_43C (K)": lim - g.loc[i, p]})
    return pd.DataFrame(rows)
