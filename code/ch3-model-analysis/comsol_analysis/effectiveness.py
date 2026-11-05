"""Physically meaningful SCR outcome metrics (added; the notebook has no equivalent).

Steady state is linear in the boundary temperatures: over the 27-run parametric sweep
    T_final(site) = w_Tb·Tb + w_Tc·Tc (+ w_T0·T0 ≈ 0), with w_Tb + w_Tc = 1 (residual ≤ 0.004 K),
so every site is summarised by one number, the cooling EFFECTIVENESS
    ε = (Tb − T_final) / (Tb − T_ring),
the fraction of the ring's cold setpoint that reaches the site. From ε follow the ring
temperature needed for a target, and design comparisons valid at every operating point.
Kinetics: t50 / t90 / τ per site; persistence: reheat time until a site re-warms past 35 °C.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .metrics import fit_exp_decay, time_to_value


def _sites(cfg):
    return cfg["probes"]["scr_point"][:-1] + cfg["probes"]["retina_agg"] + cfg["probes"]["eye_agg"]  # all but SCR


def superposition_fit(reg, cfg):
    """Per (variant, site): least-squares T_final = a·T0 + b·Tb + c·Tc + d over the parametric sweep."""
    K = cfg["units"]["k_to_c"]
    M = reg.manifest
    rows = []
    for v in cfg["vocabulary"]["variants"]:
        P = M[(M.block == "scr_parametric") & (M.variant == v)]
        X = np.c_[P.T0_K - K, P.Tb_K - K, P.Tc_K - K, np.ones(len(P))]
        for site in _sites(cfg):
            y = np.array([reg.signals[r][site].iloc[-1] for r in P.run_id])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ b
            rows.append({"variant": v, "site": site, "n_runs": len(P), "w_T0": b[0], "w_Tb": b[1],
                         "epsilon (w_Tc)": b[2], "intercept (C)": b[3], "sum_w": b[:3].sum(),
                         "max_abs_residual (K)": np.abs(res).max()})
    return pd.DataFrame(rows)


def required_ring_temperature(eps_df, cfg):
    """Ring (cold-face) temperature needed for each site to reach each target, at Tb = 35/37/39 °C.
    Flagged unreachable when it would have to be below the 4 °C safety floor."""
    th = cfg["thresholds"]
    rows = []
    for _, r in eps_df.iterrows():
        e = r["epsilon (w_Tc)"]
        for Tb in (35.0, 37.0, 39.0):
            for target in (th["therapeutic_C"], th["therapeutic_moderate_C"]):
                need = Tb - (Tb - target) / e if e > 0 else np.nan
                rows.append({"variant": r.variant, "site": r.site, "epsilon": e, "Tb (C)": Tb, "target (C)": target,
                             "required_ring_T (C)": need,
                             "reachable_above_safety_floor": bool(need >= th["safety_cold_C"]) if np.isfinite(need) else False,
                             "T_site_at_4C_ring (C)": Tb - e * (Tb - th["safety_cold_C"])})
    return pd.DataFrame(rows)


def run_epsilon(reg, cfg):
    """ε per run and site for every baseline-point run (mesh, material, silicone, parametric baseline),
    using the run's own final ring (SCR-probe) temperature as the cold reference."""
    b = cfg["geometry"]["scr_baseline_point"]
    Tb = b["Tb_K"] - cfg["units"]["k_to_c"]
    M = reg.manifest
    base = M[(M.block == "scr_parametric") & (abs(M.T0_K - b["T0_K"]) < .5) & (abs(M.Tb_K - b["Tb_K"]) < .5) & (abs(M.Tc_K - b["Tc_K"]) < .5)]
    runs = pd.concat([base.assign(study="Parametric baseline"),
                      M[M.block == "scr_material"].assign(study="Material"),
                      M[M.block == "scr_silicone_internal"].assign(study="Silicone"),
                      M[(M.block == "scr_mesh_convergence") & (M.mesh == cfg["mesh"]["scr_order"][-1])].assign(study="Mesh (finest)")])
    rows = []
    for _, r in runs.iterrows():
        s = reg.signals[r.run_id]
        Tring = float(s["SCR"].iloc[-1])
        for site in _sites(cfg):
            if site not in s.columns:
                continue
            y = s[site].dropna()
            rows.append({"study": r.study, "variant": r.variant, "material": r.material or "Aluminum",
                         "silicone": r.silicone or "Baseline", "site": site, "t_end (min)": float(y.index[-1]),
                         "T_ring_final (C)": Tring, "T_final (C)": float(y.iloc[-1]),
                         "epsilon": (Tb - y.iloc[-1]) / (Tb - Tring)})
    return pd.DataFrame(rows)


def kinetics(reg, cfg):
    """Per parametric run and site: t50, t90 (from T(0) to T(35 min)) and the exponential τ."""
    K = cfg["units"]["k_to_c"]
    M = reg.manifest
    rows = []
    for _, r in M[M.block == "scr_parametric"].iterrows():
        s = reg.signals[r.run_id]
        for site in _sites(cfg):
            y = s[site].dropna()
            t, v = y.index.values, y.values
            d = v[0] - v[-1]
            fit = fit_exp_decay(t, v, cfg)
            rows.append({"variant": r.variant, "T0 (C)": r.T0_K - K, "Tb (C)": r.Tb_K - K, "Tc (C)": r.Tc_K - K, "site": site,
                         "t50 (min)": time_to_value(t, v, v[0] - 0.5 * d, d > 0),
                         "t90 (min)": time_to_value(t, v, v[0] - 0.9 * d, d > 0),
                         "tau_fit (min)": fit["tau"], "fit_r2": fit["r2"]})
    return pd.DataFrame(rows)


def kinetics_summary(kin):
    g = kin.groupby(["variant", "site"])
    return g.agg(t50_median=("t50 (min)", "median"), t90_median=("t90 (min)", "median"), t90_min=("t90 (min)", "min"),
                 t90_max=("t90 (min)", "max"), tau_median=("tau_fit (min)", "median")).reset_index()


def persistence(reg, cfg):
    """Reheat runs (device off): minutes until each site re-warms past the therapeutic edge (35 °C),
    NaN = still below at 35 min; 'never_cooled' = already above 35 °C when the device switched off."""
    K, th = cfg["units"]["k_to_c"], cfg["thresholds"]["therapeutic_C"]
    rows = []
    for _, r in reg.manifest[reg.manifest.block == "scr_reheat"].iterrows():
        s = reg.signals[r.run_id]
        for site in [c for c in _sites(cfg) if c in s.columns]:
            y = s[site].dropna()
            start_above = bool(y.iloc[0] >= th)
            rows.append({"variant": r.variant, "T0 (C)": r.T0_K - K, "Tb (C)": r.Tb_K - K, "Tc (C)": r.Tc_K - K, "site": site,
                         "T_at_switch_off (C)": float(y.iloc[0]), "T_at_35min (C)": float(y.iloc[-1]),
                         "never_cooled_below_35": start_above,
                         "minutes_below_35_after_off": np.nan if start_above else time_to_value(y.index.values, y.values, th, False)})
    return pd.DataFrame(rows)


def design_across_conditions(reg, gci, cfg):
    """Every variant pair, every site: T_final difference at each of the 27 matched parametric points
    (mean / min / max), ε difference, and the finest-mesh uncertainty band for comparison."""
    M = reg.manifest
    P = M[M.block == "scr_parametric"].copy()
    P["key"] = list(zip(P.T0_K.round(2), P.Tb_K.round(2), P.Tc_K.round(2)))
    U = gci[gci.family == "SCR"].set_index(["variant", "channel"])["U_mesh (K)"]
    rows = []
    for a, b in itertools.combinations(cfg["vocabulary"]["variants"], 2):
        A = dict(zip(P[P.variant == a].key, P[P.variant == a].run_id))
        B = dict(zip(P[P.variant == b].key, P[P.variant == b].run_id))
        keys = sorted(set(A) & set(B))
        for site in _sites(cfg):
            d = np.array([reg.signals[A[k]][site].iloc[-1] - reg.signals[B[k]][site].iloc[-1] for k in keys])
            ua, ub = U.get((a, site), np.nan), U.get((b, site), np.nan)
            u = np.sqrt(ua ** 2 + ub ** 2) if np.isfinite(ua) and np.isfinite(ub) else np.nan
            rows.append({"A": a, "B": b, "site": site, "n_points": len(keys), "dT_final mean (K)": d.mean(),
                         "dT_final min (K)": d.min(), "dT_final max (K)": d.max(),
                         "same_sign_everywhere": bool((d > 0).all() or (d < 0).all()),
                         "U_mesh combined (K)": u,
                         "exceeds_mesh_band": bool(np.isfinite(u) and np.abs(d).min() > u)})
    return pd.DataFrame(rows)


def timeseries(reg, cfg):
    """Long table of the site temperature time courses used for the thesis figures, read
    unchanged from the registry: baseline-point cooling runs of each design (parametric),
    the V1 material runs, and the V1 re-heat runs at the baseline body temperature
    (time 0 = switch-off). Columns: study, variant, material, Tb/Tc (C), run_id,
    time (min), channel, T (C)."""
    K = cfg["units"]["k_to_c"]
    b = cfg["geometry"]["scr_baseline_point"]
    M = reg.manifest
    near = lambda col, v: (M[col] - v).abs() < .5  # noqa: E731
    sel = pd.concat([
        M[(M.block == "scr_parametric") & near("T0_K", b["T0_K"]) & near("Tb_K", b["Tb_K"])
          & near("Tc_K", b["Tc_K"])].assign(study="Cooling, baseline point"),
        M[(M.block == "scr_material") & (M.variant == "V1")].assign(study="Material, V1"),
        M[(M.block == "scr_reheat") & (M.variant == "V1") & near("Tb_K", b["Tb_K"])].assign(study="Re-heat, V1"),
    ])
    chans = ["SCR"] + _sites(cfg)
    rows = []
    for _, r in sel.iterrows():
        s = reg.signals[r.run_id]
        for ch in [c for c in chans if c in s.columns]:
            y = s[ch].dropna()
            rows.append(pd.DataFrame({"study": r.study, "variant": r.variant,
                                      "material": r.material if isinstance(r.material, str) and r.material else "Aluminum",
                                      "Tb (C)": r.Tb_K - K if pd.notna(r.Tb_K) else np.nan,
                                      "Tc (C)": r.Tc_K - K if pd.notna(r.Tc_K) else np.nan,
                                      "run_id": r.run_id, "time (min)": y.index.values.astype(float),
                                      "channel": ch, "T (C)": y.values.astype(float)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
