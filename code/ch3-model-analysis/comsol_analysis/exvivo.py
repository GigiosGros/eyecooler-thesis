"""Model vs ex-vivo experiment — a COMPARISON UNDER DIFFERENT CONDITIONS, not a validation.

The model was not re-run with the ex-vivo boundary conditions (licence expired), so the two are
compared through quantities that are defined identically on both sides and are as independent
of the absolute temperatures as possible:

  ε_bath    = (T_ref − T_final,site) / (T_ref − T_final,ring)   T_ref = 37 °C (body / bath)
  ε_start   = ΔT_site / (T_initial,site − T_final,ring)          uses each side's own start
              (= ε_Δ in the thesis; the primary measure: it counts only the
              cooling the device produced, from the site's own temperature at switch-on, so the
              ex-vivo eyes are not credited with having started below 37 °C)
  t50, t90  of the site (minutes)

Known differences (each pushes the model toward LESS / SLOWER cooling than ex vivo):
  • perfusion + metabolism (Pennes) in the model; none in the enucleated eye
  • eyelids and orbital tissue in the model deliver heat to the ring; none ex vivo
  • body boundary at Tb vs a 37 °C oil bath with convection
  • model probe points vs micro-CT thermistor positions; model eye size vs donor eyes
  • ring cold face held at 4 °C (model) vs the measured ring thermistor (≈ 4.1 °C)
  • 35 min (model) vs a 16–20 min final window (experiment)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

SITE_MAP = {"SCR": "SCR", "Anterior Sclera": "Anterior Sclera", "Equatorial Sclera": "Equatorial Sclera",
            "Vitreous": "Vitreous Humor", "Posterior Sclera": "Posterior Sclera", "Optic Nerve": "Optic Nerve Head"}


def load_experiment(path):
    return pd.read_csv(path) if path and os.path.exists(path) else pd.DataFrame()


def _model_run(reg, cfg, variant, material):
    b = cfg["geometry"]["scr_baseline_point"]
    M = reg.manifest
    if material == "Aluminum":
        R = M[(M.block == "scr_parametric") & (M.variant == variant) & (abs(M.T0_K - b["T0_K"]) < .5)
              & (abs(M.Tb_K - b["Tb_K"]) < .5) & (abs(M.Tc_K - b["Tc_K"]) < .5)]
    else:
        R = M[(M.block == "scr_material") & (M.variant == variant) & (M.material == material)]
    return None if R.empty else R.iloc[0].run_id


def compare(reg, exp, cfg):
    """One row per (experimental group, site) with the matching model run."""
    if exp.empty:
        return pd.DataFrame()
    Tref = cfg["exvivo"]["reference_C"]
    exp = exp[exp.species.isin(cfg["exvivo"].get("species", exp.species.unique()))]
    rows = []
    for key, g in exp.groupby(["species", "status", "ring_material", "ring"], sort=False):
        variant = cfg["exvivo"]["ring_to_variant"].get(key[3])
        rid = _model_run(reg, cfg, variant, key[2]) if variant else None
        if rid is None:
            continue
        s = reg.signals[rid]
        ring_m = float(s["SCR"].iloc[-1])
        ring_e = float(g[g.channel == "SCR"].T_final_mean.iloc[0])
        for _, e in g.iterrows():
            site = SITE_MAP.get(e.channel)
            if site is None or site == "SCR" or site not in s.columns:
                continue
            y = s[site].dropna()
            t, v = y.index.values, y.values
            d = v[0] - v[-1]
            t50 = float(t[np.argmax(v <= v[0] - 0.5 * d)])
            t90 = float(t[np.argmax(v <= v[0] - 0.9 * d)])
            eps_m = (Tref - v[-1]) / (Tref - ring_m)
            eps_e = (Tref - e.T_final_mean) / (Tref - ring_e)
            eps_e_sd = e.T_final_sd / (Tref - ring_e)
            se = eps_e_sd / np.sqrt(e.n)
            # 95% CI of the mean over n eyes: Student t with n-1 df (the normal quantile 1.96
            # gives an interval that is too narrow for n = 4-9 eyes)
            q = float(stats.t.ppf(0.975, int(e.n) - 1)) if e.n > 1 else np.nan
            # ε_Δ: same approximation as ε_bath (denominator taken as fixed; spread from ΔT)
            eps_sm = d / (v[0] - ring_m)
            eps_se = e.dT_mean / (e.T_initial_mean - ring_e)
            eps_se_sd = e.dT_sd / (e.T_initial_mean - ring_e)
            se_s = eps_se_sd / np.sqrt(e.n)
            rows.append({"group": " ".join(key), "model_run": rid, "site": site, "n_eyes": int(e.n),
                         "T_final model (C)": v[-1], "T_final exp (C)": e.T_final_mean, "T_final exp SD": e.T_final_sd,
                         "eps_bath model": eps_m, "eps_bath exp": eps_e, "eps_bath exp SD": eps_e_sd,
                         "eps_bath exp 95%CI low": eps_e - q * se, "eps_bath exp 95%CI high": eps_e + q * se,
                         "eps ratio exp/model": eps_e / eps_m,
                         "eps_start model": eps_sm,
                         "eps_start exp": eps_se,
                         "eps_start exp SD": eps_se_sd,
                         "eps_start exp 95%CI low": eps_se - q * se_s,
                         "eps_start exp 95%CI high": eps_se + q * se_s,
                         "t quantile (n-1 df)": q,
                         "eps_start ratio exp/model": eps_se / eps_sm,
                         "T_initial model (C)": v[0], "T_initial exp (C)": e.T_initial_mean,
                         "dT model (C)": d, "dT exp (C)": e.dT_mean,
                         "t50 model (min)": t50, "t50 exp (min)": e.t50_mean / 60,
                         "t90 model (min)": t90, "t90 exp (min)": e.t90_mean / 60,
                         "model within exp 95%CI": bool(eps_e - q * se <= eps_m <= eps_e + q * se),
                         "model within exp 95%CI (eps_start)": bool(eps_se - q * se_s <= eps_sm <= eps_se + q * se_s)})
    return pd.DataFrame(rows)


def site_ordering(cmp):
    """Does the model rank the sites (by ε) the same way as each experimental group? Spearman ρ."""
    rows = []
    for g, d in cmp.groupby("group", sort=False):
        rho = d["eps_bath model"].rank().corr(d["eps_bath exp"].rank())
        rho_s = d["eps_start model"].rank().corr(d["eps_start exp"].rank())
        rows.append({"group": g, "n_sites": len(d), "spearman_rho_eps": rho,
                     "median_eps_ratio_exp_over_model": d["eps ratio exp/model"].median(),
                     "spearman_rho_eps_start": rho_s,
                     "median_eps_start_ratio_exp_over_model": d["eps_start ratio exp/model"].median()})
    return pd.DataFrame(rows)


def material_comparison(reg, cfg):
    """Aluminum − ceramic (original ring, V1): model material runs vs the paired ex-vivo comparison
    (same human eyes, both rings; thesis-analysis 03_comparisons). Quantity: ΔT = T_initial − T_final
    at each site, plus T_final. The experimental CI is the 90% CI of the paired mean difference."""
    path = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         cfg["exvivo"]["material_comparison_csv"]))
    if not os.path.exists(path):
        return pd.DataFrame()
    exp = pd.read_csv(path)
    M = reg.manifest
    run = lambda m: M[(M.block == "scr_material") & (M.variant == "V1") & (M.material == m)].run_id.iloc[0]  # noqa: E731
    A, C = reg.signals[run("Aluminum")], reg.signals[run("Ceramic")]
    rows = []
    for _, e in exp.iterrows():
        site = SITE_MAP.get(e.channel)
        if site is None or site not in A.columns:
            continue
        a, c = A[site].dropna(), C[site].dropna()
        d_dT = (a.iloc[0] - a.iloc[-1]) - (c.iloc[0] - c.iloc[-1])
        rows.append({"site": site, "model dT Al (K)": a.iloc[0] - a.iloc[-1], "model dT ceramic (K)": c.iloc[0] - c.iloc[-1],
                     "model d(dT) Al-ceramic (K)": d_dT, "model d(T_final) Al-ceramic (K)": a.iloc[-1] - c.iloc[-1],
                     "exp d(dT) Al-ceramic (K)": e.diff_C, "exp 90%CI low": e.ci90_low, "exp 90%CI high": e.ci90_high,
                     "exp design": e.design, "exp p (Holm)": e.p_holm,
                     "model inside exp 90%CI": bool(e.ci90_low <= d_dT <= e.ci90_high),
                     "same direction": bool(np.sign(d_dT) == np.sign(e.diff_C))})
    return pd.DataFrame(rows)
