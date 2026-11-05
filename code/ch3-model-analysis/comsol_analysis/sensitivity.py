"""Parameter sensitivity over the sweeps (notebook Step 9): Pearson ρ, factorial ANOVA
(main effects + all 2-way interactions; η² / ω² variance attribution) and PCA of the probe
responses. The FEM runs are deterministic, so F / p are not inferential — read η²."""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .metrics import time_to_therapeutic

MODES = {("SCR", "Time-dependent"): ("scr_parametric", None, ["T0_K", "Tb_K", "Tc_K"]),
         ("Waterblock", "Time-dependent"): ("wb_sweep", None, ["Q0", "Tin_K", "T0_K", "Qh"]),
         ("Waterblock", "Stationary"): ("wb_aux", "Fine", ["Q0", "Tin_K", "T0_K", "Qh"])}
METRICS = {"dT": "ΔT = T(0) − T_final (K)", "Tfinal": "T_final (°C)", "t_th": "time to 35 °C (min)"}


def io_table(reg, family, mode, variant, metric, cfg):
    """X = run inputs (°C for temperatures), Y = per-probe output metric. variant: V1/V2/V3,
    'All' (pooled; variant not a factor) or 'Avg' (mean over variants at each input point)."""
    K = cfg["units"]["k_to_c"]
    block, mesh, pcols = MODES[(family, mode)]
    sub = reg.manifest[reg.manifest.block == block]
    if mesh is not None:
        sub = sub[sub.mesh == mesh]
    if variant in cfg["vocabulary"]["variants"]:
        sub = sub[sub.variant == variant]
    if sub.empty:
        return None, None
    xin, yout = [], []
    for _, r in sub.iterrows():
        sig = reg.signals[r.run_id]
        xin.append({**{p: float(r[p]) for p in pcols if pd.notna(r[p])}, "variant": r.variant})
        out = {}
        for ch in sig.columns:
            y = pd.to_numeric(sig[ch], errors="coerce").dropna()
            if len(y) < 2:
                continue
            if metric == "t_th":   # first time at/below 35 °C (notebook: t = 0 if it starts there)
                out[ch] = time_to_therapeutic(y.index.values, y.values, cfg["thresholds"]["therapeutic_C"], True, cfg)[0]
            elif metric == "dT":
                out[ch] = float(y.iloc[0] - y.iloc[-1])
            else:
                out[ch] = float(y.iloc[-1])
        yout.append(out)
    X, Y = pd.DataFrame(xin), pd.DataFrame(yout)
    if variant == "Avg" and X.variant.nunique() > 1:
        m = pd.concat([X.drop(columns="variant"), Y], axis=1).groupby(pcols).mean().reset_index()
        X, Y = m[pcols].copy(), m.drop(columns=pcols)
    else:
        X = X.drop(columns="variant", errors="ignore")
    for c in [c for c in X.columns if c.endswith("_K")]:
        X[c] = X[c] - K
    return X.rename(columns={c: c.replace("_K", "_C") for c in X.columns if c.endswith("_K")}), Y


def pearson(X, Y):
    return pd.concat([X, Y], axis=1).corr().loc[X.columns, Y.columns]


def anova(X, Y, probe):
    """Type-II factorial ANOVA, main effects + 2-way interactions; 3-way+ left in the residual."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if probe not in Y.columns:
        return None
    d = pd.concat([X, Y[probe].rename("_y")], axis=1).dropna()
    if len(d) < 4 or float(((d._y - d._y.mean()) ** 2).sum()) < 1e-12:
        return None
    pcols = [c for c in X.columns if d[c].nunique() > 1]
    if not pcols:
        return None
    safe = {c: f"f{i}" for i, c in enumerate(pcols)}
    d = d.rename(columns=safe)
    terms = [f"C({safe[c]})" for c in pcols] + [f"C({safe[a]}):C({safe[b]})" for a, b in combinations(pcols, 2)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aov = sm.stats.anova_lm(smf.ols("_y ~ " + " + ".join(terms), data=d).fit(), typ=2)
    except Exception:  # noqa: BLE001 — e.g. an empty cell after dropping NaN outputs (as in the notebook)
        return None
    ss_tot = float(aov.sum_sq.sum())
    ms_err = (aov.loc["Residual", "sum_sq"] / aov.loc["Residual", "df"]
              if ("Residual" in aov.index and aov.loc["Residual", "df"] > 0) else np.nan)
    inv = {v: k for k, v in safe.items()}
    rows = []
    for term, r in aov.iterrows():
        if term == "Residual":
            continue
        lab = term
        for s, orig in inv.items():
            lab = lab.replace(f"C({s})", orig)
        lab = lab.replace(":", " x ")
        ss, df_ = float(r.sum_sq), float(r.df)
        rows.append({"Factor": lab, "Type": "Inter" if " x " in lab else "Main", "df": int(df_),
                     "F": float(r.get("F", np.nan)), "p": float(r.get("PR(>F)", np.nan)),
                     "eta^2 (%)": 100 * ss / ss_tot if ss_tot > 0 else np.nan,
                     "omega^2 (%)": 100 * max((ss - df_ * ms_err) / (ss_tot + ms_err), 0.0) if np.isfinite(ms_err) else np.nan})
    return pd.DataFrame(rows).sort_values("eta^2 (%)", ascending=False)


def pca(X, Y):
    Yc = Y.dropna(axis=1, how="any")
    if Yc.shape[0] < 3 or Yc.shape[1] < 2:
        return None
    Z = StandardScaler().fit_transform(Yc)
    n = min(4, Z.shape[1])
    p = PCA(n_components=n).fit(Z)
    pcs = [f"PC{i + 1}" for i in range(n)]
    load = pd.DataFrame(p.components_.T, index=Yc.columns, columns=pcs)
    scores = pd.DataFrame(p.transform(Z), index=Yc.index, columns=pcs)
    Xc = X.loc[Yc.index] if Yc.index.isin(X.index).all() else X.iloc[:len(scores)]
    Xz = StandardScaler().fit_transform(Xc.values)
    reg = {pc: np.linalg.lstsq(Xz, scores[pc].values, rcond=None)[0] for pc in pcs}
    return {"evr": p.explained_variance_ratio_, "loadings": load, "scores": scores,
            "reg": pd.DataFrame(reg, index=Xc.columns), "X": Xc}


def run_all(reg, cfg):
    """Every (family, mode) x variant (V1–V3, All, Avg) x metric."""
    out = []
    for (fam, mode) in MODES:
        for v in cfg["vocabulary"]["variants"] + ["All", "Avg"]:
            for metric in METRICS:
                X, Y = io_table(reg, fam, mode, v, metric, cfg)
                if X is None or X.empty:
                    continue
                out.append(dict(family=fam, mode=mode, variant=v, metric=metric, X=X, Y=Y, pearson=pearson(X, Y),
                                anova={p: a for p in Y.columns if (a := anova(X, Y, p)) is not None and not a.empty},
                                pca=pca(X, Y)))
    return out


def dominant_factors(results, metric="dT", variant="All"):
    """Compact table: for each (family, mode, probe), the two largest η² terms."""
    rows = []
    for e in results:
        if e["metric"] != metric or e["variant"] != variant:
            continue
        for probe, a in e["anova"].items():
            top = a.nlargest(2, "eta^2 (%)")
            rows.append({"family": e["family"], "mode": e["mode"], "probe": probe,
                         "factor_1": top.Factor.iloc[0], "eta2_1 (%)": top["eta^2 (%)"].iloc[0],
                         "factor_2": top.Factor.iloc[1] if len(top) > 1 else "",
                         "eta2_2 (%)": top["eta^2 (%)"].iloc[1] if len(top) > 1 else np.nan})
    return pd.DataFrame(rows)
