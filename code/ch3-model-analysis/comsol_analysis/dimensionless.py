"""Dimensionless numbers for the waterblock coolant (notebook Step 10).

Re from Q0 and the variant geometry (inlet pipe; V1 tip-gap / open / throat paths, V2/V3
open region / throat); Pr, Pe, St, Gz at the characteristic region (V1 tip gap, V2/V3 throat).
Not reproduced from the notebook: its 'measured Nu' = alpha·D_h/k and the Biot marker at the
'median measured alpha' treat the COMSOL continuation parameter alpha (0 -> 1) as a heat-transfer
coefficient h, which it is not.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .energy import water_props

RE_BANDS = [(0, 2000, "Laminar"), (2000, 2300, "Laminar (flag)"), (2300, 4000, "Transitional"), (4000, 1e9, "Turbulent")]
NU_LAMINAR_CONST_T, NU_LAMINAR_CONST_Q = 3.66, 4.36


def re_regime(re):
    if not np.isfinite(re):
        return "-"
    return next((lab for lo, hi, lab in RE_BANDS if lo <= re < hi), "-")


def _Dh_rect(w, h):
    return 2.0 * w * h / (w + h)


def regions_for(variant, cfg):
    g = cfg["geometry"]["waterblock"][variant]
    split = cfg["geometry"]["v1_path_split"]
    out = [{"name": "Inlet pipe", "A_mm2": math.pi / 4 * g["D_in"] ** 2, "Dh_mm": g["D_in"], "Q_frac": 1.0}]
    if g["tip_gap"]:
        h_top = g["H_ch"] - g["H_fin"]
        out += [{"name": "Path A - tip gap", "A_mm2": g["w"] * h_top, "Dh_mm": _Dh_rect(g["w"], h_top), "Q_frac": split["Q_A_frac"]},
                {"name": "Path B - open", "A_mm2": g["w"] * g["H_fin"], "Dh_mm": _Dh_rect(g["w"], g["H_fin"]), "Q_frac": split["Q_B_frac"]},
                {"name": "Path B - throat", "A_mm2": (g["w"] - g["L_fin"]) * g["H_fin"],
                 "Dh_mm": _Dh_rect(g["w"] - g["L_fin"], g["H_fin"]), "Q_frac": split["Q_B_frac"]}]
    else:
        out += [{"name": "Open region", "A_mm2": g["w"] * g["H_ch"], "Dh_mm": _Dh_rect(g["w"], g["H_ch"]), "Q_frac": 1.0},
                {"name": "Throat", "A_mm2": (g["w"] - g["L_fin"]) * g["H_ch"],
                 "Dh_mm": _Dh_rect(g["w"] - g["L_fin"], g["H_ch"]), "Q_frac": 1.0}]
    return out


def reynolds_for(variant, Q, Tin_C, cfg):
    nu = water_props(Tin_C, cfg)["nu"]
    rows = []
    for r in regions_for(variant, cfg):
        v = r["Q_frac"] * Q / (r["A_mm2"] * 1e-6)
        Re = v * r["Dh_mm"] * 1e-3 / nu
        rows.append({"Variant": variant, "Region": r["name"], "A (mm^2)": r["A_mm2"], "Dh (mm)": r["Dh_mm"],
                     "v (m/s)": v, "Re": Re, "Regime": re_regime(Re)})
    return rows


def entrance_length(Re, D_mm):
    if not np.isfinite(Re) or Re <= 0:
        return np.nan
    return 0.06 * Re * D_mm if Re < 2300 else 4.4 * Re ** (1 / 6) * D_mm


def prandtl(T_C, cfg):
    w = water_props(T_C, cfg)
    return w["nu"] * w["rho"] * w["cp"] / w["k"]


def nusselt_dittus_boelter(Re, Pr, heating):
    """0.023 Re^0.8 Pr^n, n = 0.4 when the fluid is heated, 0.3 when cooled. (Nominally valid for
    Re > 10 000; the throat Re here is 1 000–5 000, so treat it as an order-of-magnitude reference.)"""
    return 0.023 * Re ** 0.8 * Pr ** (0.4 if heating else 0.3) if (Re > 0 and Pr > 0) else np.nan


def char_region(v):
    return "Path A - tip gap" if v == "V1" else "Throat"


def dimensionless_table(reg, cfg):
    """Every unique (Q0, Tin) in the manifest x variant (the notebook's Step 14 'Dimensionless' sheet)."""
    K = cfg["units"]["k_to_c"]
    heating = cfg["corrections"]["dittus_boelter_heating"]
    pts = reg.manifest.dropna(subset=["Q0", "Tin_K"])[["Q0", "Tin_K"]].drop_duplicates()
    rows = []
    for _, p in pts.iterrows():
        Q, Tin = float(p.Q0), float(p.Tin_K) - K
        Pr = prandtl(Tin, cfg)
        for v in cfg["vocabulary"]["variants"]:
            rr = next(r for r in reynolds_for(v, Q, Tin, cfg) if r["Region"] == char_region(v))
            Re, Dh = rr["Re"], rr["Dh (mm)"] * 1e-3
            Nu = nusselt_dittus_boelter(Re, Pr, heating)
            g = cfg["geometry"]["waterblock"][v]
            L_path = 2 * math.pi * (g["R_i"] + g["R_o"]) / 2 * 1e-3
            rows.append({"Variant": v, "Q0 (m^3/s)": Q, "Tin (C)": Tin, "Char. region": char_region(v), "Re": Re, "Pr": Pr,
                         "Nu (Dittus-B.)": Nu, "Pe": Re * Pr, "St": Nu / (Re * Pr) if (Re > 0 and Pr > 0) else np.nan,
                         "Gz": (Dh / L_path) * Re * Pr, "Regime": rr["Regime"],
                         "Dittus-Boelter valid (Re>1e4)": bool(Re > 1e4)})
    return pd.DataFrame(rows)


def reynolds_nominal(cfg):
    """All regions at the Reynolds-report reference point (Q = 8.0e-6 m^3/s, 25 °C)."""
    f = cfg["flow"]
    rows = [r for v in cfg["vocabulary"]["variants"] for r in reynolds_for(v, f["reynolds_nominal_Q_m3s"], f["reynolds_nominal_Tin_C"], cfg)]
    df = pd.DataFrame(rows)
    ent = []
    for v in cfg["vocabulary"]["variants"]:
        Re = df[(df.Variant == v) & (df.Region == "Inlet pipe")].Re.iloc[0]
        g = cfg["geometry"]["waterblock"][v]
        Le = entrance_length(Re, g["D_in"])
        ent.append({"Variant": v, "Re_pipe": Re, "L_pipe (mm)": g["L_pipe"], "L_e required (mm)": Le,
                    "L / L_e": g["L_pipe"] / Le if Le else np.nan, "Developed?": "No" if (Le and g["L_pipe"] < Le) else "Yes"})
    return df, pd.DataFrame(ent)
