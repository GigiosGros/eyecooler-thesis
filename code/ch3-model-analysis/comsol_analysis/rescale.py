"""Rescaling of the SCR runs to another starting condition, by superposition.

The SCR model runs start from a stationary solution in which the ring cold face is
held at T0 (a slip in the model set-up: the T = T0 condition on the TEC should not
be in the stationary study). Every run therefore starts from an eye already
pre-cooled by a ring at T0, and the ring then steps only from T0 to Tc.

The model is linear (constant properties, linear perfusion term): at every site and
time the cooling is a fixed fraction of the ring step,

    s(t) = (T(0) - T(t)) / (T0 - Tc)          unit step response of the site,

checked on the parametric runs to 0.004 K (effectiveness.superposition_fit). A run
can therefore be rescaled exactly to any other ring start:

  * eye at body temperature everywhere, ring stepped to Tc (what the corrected model
    run will give):                    T(t) = Tb - (Tb - Tc) s(t)
  * the ex-vivo experiment (ring starts at its measured temperature, steady state):
        T(0) = Tb - s(end) (Tb - T_ring,start)
        dT   = (T_ring,start - T_ring,final) s(16-20 min)

OFF by default ([superposition] enabled = false): these tables are
derived, not simulated, and are to be confirmed by the corrected COMSOL run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exvivo import SITE_MAP, _model_run


def step_response(y: pd.Series, T0_C: float, Tc_C: float):
    """(t, s): unit step response of one site from a run that starts at ring = T0."""
    y = y.dropna()
    t, v = y.index.values.astype(float), y.values.astype(float)
    return t, (v[0] - v) / (T0_C - Tc_C)


def _first_crossing(t, T, level):
    below = np.where(T <= level)[0]
    if not len(below):
        return np.nan
    i = below[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - T[i - 1]) * (t[i] - t[i - 1]) / (T[i] - T[i - 1]))


def from_body_start(reg, cfg) -> pd.DataFrame:
    """Every SCR parametric run rescaled to an eye that starts at Tb everywhere, ring
    stepped to Tc: temperature at 20 min and at the end, and time to the therapeutic edge."""
    sup = cfg["superposition"]
    edge = cfg["thresholds"]["therapeutic_C"]
    M = reg.manifest
    rows = []
    for r in M[M.block == "scr_parametric"].itertuples():
        T0, Tb, Tc = r.T0_K - 273.15, r.Tb_K - 273.15, r.Tc_K - 273.15
        if abs(T0 - Tc) < sup["min_ring_step_K"]:      # step too small to define s(t) reliably
            continue
        sig = reg.signals[r.run_id]
        for site in cfg["probes"]["main"]:
            if site not in sig.columns or site == "SCR":
                continue
            t, s = step_response(sig[site], T0, Tc)
            T = Tb - (Tb - Tc) * s
            rows.append({"run_id": r.run_id, "variant": r.variant, "T0 as run (C)": T0, "Tb (C)": Tb, "Tc (C)": Tc,
                         "site": site, "step response (end)": float(s[-1]),
                         "T at 20 min (C)": float(np.interp(20.0, t, T)), "T at end (C)": float(T[-1]),
                         "t to therapeutic (min)": _first_crossing(t, T, edge),
                         "t to therapeutic as run (min)": _first_crossing(t, sig[site].dropna().values, edge)
                         if sig[site].dropna().values[0] > edge else np.nan})
    return pd.DataFrame(rows)


def exvivo_like_for_like(reg, exp: pd.DataFrame, cfg) -> pd.DataFrame:
    """Model rescaled to each experimental group's own ring start and end, read over the
    experiment's final window; one row per (group, site)."""
    if exp.empty:
        return pd.DataFrame()
    sup, b = cfg["superposition"], cfg["geometry"]["scr_baseline_point"]
    T0, Tb, Tc = b["T0_K"] - 273.15, cfg["exvivo"]["reference_C"], b["Tc_K"] - 273.15
    lo, hi = sup["exvivo_window_min"]
    exp = exp[exp.species.isin(cfg["exvivo"].get("species", exp.species.unique()))]
    rows = []
    for key, g in exp.groupby(["species", "status", "ring_material", "ring"], sort=False):
        variant = cfg["exvivo"]["ring_to_variant"].get(key[3])
        rid = _model_run(reg, cfg, variant, key[2]) if variant else None
        if rid is None or g[g.channel == "SCR"].empty:
            continue
        sig, ring = reg.signals[rid], g[g.channel == "SCR"].iloc[0]
        for _, e in g.iterrows():
            site = SITE_MAP.get(e.channel)
            if site is None or site == "SCR" or site not in sig.columns:
                continue
            t, s = step_response(sig[site], T0, Tc)
            s_win = float(s[(t >= lo) & (t <= hi)].mean())
            T_start = Tb - float(s[-1]) * (Tb - ring.T_initial_mean)
            dT = (ring.T_initial_mean - ring.T_final_mean) * s_win
            rows.append({"group": " ".join(key), "model_run": rid, "site": site, "n_eyes": int(e.n),
                         "ring start exp (C)": ring.T_initial_mean, "ring final exp (C)": ring.T_final_mean,
                         "step response model (window)": s_win,
                         "T_initial exp (C)": e.T_initial_mean, "T_initial model (C)": T_start,
                         "dT exp (C)": e.dT_mean, "dT exp SD": e.dT_sd, "dT model (C)": dT,
                         "dT ratio exp/model": e.dT_mean / dT,
                         "T_final exp (C)": e.T_final_mean, "T_final model (C)": T_start - dT})
    return pd.DataFrame(rows)
