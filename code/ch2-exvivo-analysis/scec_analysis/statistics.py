"""Group comparisons of the cooling excursion dT.

For groups A and B (contrast A - B, + = A cooled more):
  PRIMARY    per channel: paired t-test when the groups share >= 2 eyes (within-eye
             ring comparisons), otherwise Welch t-test on per-trace values; Holm
             correction across the five tissue sites (SCR is its own family);
             Hedges' g with 95 % CI; equivalence by two one-sided tests: the 90 %
             CI of the difference must lie within +/- margin.
  SENSITIVITY  donor level: fellow human eyes are averaged to one value per donor
             (each porcine eye is its own donor) and the same test is repeated, so
             that eyes of one donor are not counted as independent.
             Tukey-trimmed difference (whole pairs dropped in paired designs;
             undefined, and so reported as such, below n = 4).
  LEGACY     statistics.mixed_model = true adds the notebook v4.28 linear mixed
             model  dT ~ condition * channel  (REML, donor intercept, eye nested).
             Off by default (2026-10-03): it pools one residual variance over
             six sites whose between-eye SD differs up to ten-fold (about 1 degC at
             the optic nerve, 5-7 degC at the vitreous), so its interval at the
             optic nerve is 2-5 times wider than the per-site one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .endpoints import _xy, endpoint_by_eye, endpoint_per_run, included


def ci_paired(a, b, alpha):
    d = np.asarray(a, float) - np.asarray(b, float)
    n, md = d.size, d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(1 - alpha, n - 1)
    return md, (md - q * se, md + q * se)


def ci_welch(a, b, alpha):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    dfree = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    diff, se = a.mean() - b.mean(), np.sqrt(va + vb)
    q = stats.t.ppf(1 - alpha, dfree)
    return diff, (diff - q * se, diff + q * se)


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(p.size), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min((p.size - rank) * p[i], 1.0))
        adj[i] = run
    return adj


def hedges_g(a, b, paired, alpha):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        d = a - b
        n, sd = d.size, d.std(ddof=1)
        if n < 2 or not np.isfinite(sd) or sd == 0:
            return np.nan, (np.nan, np.nan)
        dz, dfree = d.mean() / sd, n - 1
        se = np.sqrt(1.0 / n + dz ** 2 / (2 * n))
    else:
        na, nb = a.size, b.size
        if na < 2 or nb < 2:
            return np.nan, (np.nan, np.nan)
        sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        if not np.isfinite(sp) or sp == 0:
            return np.nan, (np.nan, np.nan)
        dz, dfree = (a.mean() - b.mean()) / sp, na + nb - 2
        se = np.sqrt((na + nb) / (na * nb) + dz ** 2 / (2 * (na + nb)))
    J = 1.0 - 3.0 / (4.0 * dfree - 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    g = J * dz
    return float(g), (float(g - z * J * se), float(g + z * J * se))


def tukey_keep(x):
    x = np.asarray(x, float)
    keep = np.isfinite(x)
    if keep.sum() < 4:
        return keep
    q1, q3 = np.percentile(x[keep], [25, 75])
    return keep & (x >= q1 - 1.5 * (q3 - q1)) & (x <= q3 + 1.5 * (q3 - q1))


def _donor(species, status, eye, cfg):
    sid = f"{species[:1]}_{status[:2]}_{eye}"
    if species == "Human":
        return cfg["statistics"]["donor_of_human_eye"].get(eye.lstrip("E"), sid)
    return sid


def donor_means(by_eye: dict, cfg) -> dict:
    """{donor: mean of that donor's eyes} from {(species, status, eye): value}."""
    d = {}
    for (sp, st, eye), v in by_eye.items():
        d.setdefault(_donor(sp, st, eye, cfg), []).append(v)
    return {k: float(np.mean(v)) for k, v in d.items()}


def long_frame(runs, A, B, cfg):
    rows = []
    w = cfg["qc"]["final_window_s"]
    for name, ids in (A, B):
        for ch in cfg["channels"]["temperature"]:
            for (sp, st, eye), val in endpoint_by_eye(runs, ids, ch, w).items():
                rows.append({"eye": f"{sp[:1]}_{st[:2]}_{eye}", "donor": _donor(sp, st, eye, cfg),
                             "channel": ch, "condition": name, "value": val})
    return pd.DataFrame(rows)


def mixed_model(long_df, formula, ref_condition, ref_channel, channels, cfg):
    import statsmodels.formula.api as smf
    d = long_df.dropna(subset=["value"])
    d = d[d["channel"].isin(channels)].copy()
    if d.empty or d["eye"].nunique() < 3 or d["condition"].nunique() < 2:
        return None
    conds = list(pd.unique(d["condition"]))
    if ref_condition in conds:     # reference = group B, so coefficients read A - B
        conds = [ref_condition] + [c for c in conds if c != ref_condition]
    chans = [c for c in cfg["channels"]["temperature"] if c in set(d["channel"])]
    if ref_channel in chans:
        chans = [ref_channel] + [c for c in chans if c != ref_channel]
    d["condition"] = pd.Categorical(d["condition"], categories=conds)
    d["channel"] = pd.Categorical(d["channel"], categories=chans)
    vc = {"eye": "0 + C(eye)"} if d.groupby("donor")["eye"].nunique().max() > 1 else None

    def attempt(vc):
        # re_formula="1" keeps the donor intercept when a variance component is added
        # (statsmodels otherwise drops it silently).
        md = (smf.mixedlm(formula, d, groups=d["donor"], re_formula="1", vc_formula=vc) if vc
              else smf.mixedlm(formula, d, groups=d["donor"]))
        for method in ("lbfgs", "bfgs", "cg", "powell"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = md.fit(reml=True, method=method)
                if np.all(np.isfinite(res.params.values)) and np.all(np.isfinite(res.conf_int().values)):
                    return res
            except Exception:
                continue
        return None

    res = attempt(vc) if vc else None
    return res if res is not None else attempt(None)


def _condition_effect(res):
    if res is None:
        return {"effect": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    term = next(p for p in res.params.index if p.startswith("C(condition)") and ":" not in p)
    ci = res.conf_int()
    return {"effect": float(res.params[term]), "ci_low": float(ci.loc[term, 0]),
            "ci_high": float(ci.loc[term, 1]), "p": float(res.pvalues[term])}


def compare(runs, A, B, cfg):
    """A, B = (name, run_ids). Returns (model_table, channel_table)."""
    s, ch_cfg = cfg["statistics"], cfg["channels"]
    alpha, margin, w = s["alpha"], s["equivalence_margin_C"], cfg["qc"]["final_window_s"]
    temp, device = ch_cfg["temperature"], ch_cfg["device"]
    long_df = long_frame(runs, A, B, cfg)
    present = [c for c in temp if c in set(long_df["channel"])]
    ref = ch_cfg["primary_endpoint"] if ch_cfg["primary_endpoint"] in present else present[0]
    tissue = [c for c in present if c not in device]

    models = []
    if s.get("mixed_model", True):
        full = mixed_model(long_df, "value ~ C(condition)*C(channel)", B[0], ref, present, cfg)
        inter = [] if full is None else [p for p in full.params.index
                                         if p.startswith("C(condition)") and ":" in p]
        models.append({"model": f"interaction, effect at {ref}", **_condition_effect(full),
                       "n_interaction_p<alpha": sum(full.pvalues[p] < alpha for p in inter) if full else np.nan})
        models.append({"model": "additive, tissue sites", **_condition_effect(
            mixed_model(long_df, "value ~ C(condition) + C(channel)", B[0], ref, tissue, cfg)
            if len(tissue) >= 2 else None)})
        models.append({"model": "SCR only", **_condition_effect(
            mixed_model(long_df, "value ~ C(condition)", B[0], None, device, cfg))})

    rows = []
    for ch in temp:
        ea, eb = endpoint_by_eye(runs, A[1], ch, w), endpoint_by_eye(runs, B[1], ch, w)
        shared = sorted(set(ea) & set(eb))
        paired = len(shared) >= 2
        if paired:
            a, b = np.array([ea[k] for k in shared]), np.array([eb[k] for k in shared])
        else:
            a, b = endpoint_per_run(runs, A[1], ch, w), endpoint_per_run(runs, B[1], ch, w)
        row = {"channel": ch, "role": "device" if ch in device else "tissue",
               "design": f"paired k={a.size}" if paired else f"Welch {a.size},{b.size}"}
        if a.size < 2 or b.size < 2:
            rows.append(row)
            continue
        p = (stats.ttest_rel(a, b) if paired else stats.ttest_ind(a, b, equal_var=False)).pvalue
        diff, ci90 = (ci_paired if paired else ci_welch)(a, b, alpha)
        g, gci = hedges_g(a, b, paired, alpha)
        if paired:
            keep = tukey_keep(a - b)
            trimmed, n_rm, n_eff = (a - b)[keep].mean(), int((~keep).sum()), a.size
        else:
            ka, kb = tukey_keep(a), tukey_keep(b)
            trimmed, n_rm, n_eff = a[ka].mean() - b[kb].mean(), int((~ka).sum() + (~kb).sum()), min(a.size, b.size)
        row.update({"diff_C": float(diff), "p_raw": float(p), "hedges_g": g,
                    "g_ci_low": gci[0], "g_ci_high": gci[1],
                    "ci90_low": float(ci90[0]), "ci90_high": float(ci90[1]),
                    "equivalent": bool(ci90[0] > -margin and ci90[1] < margin),
                    "diff_trimmed_C": float(trimmed), "n_trimmed": n_rm,
                    "trim_status": "no fence (n<4)" if n_eff < 4 else
                                   ("nothing removed" if n_rm == 0 else f"{n_rm} removed")})
        # donor-level sensitivity: one value per donor
        if paired:
            dd = np.array(list(donor_means({k: ea[k] - eb[k] for k in shared}, cfg).values()))
            ok, n_don = dd.size >= 2, f"{dd.size}"
            if ok:
                d_diff, d_ci = ci_paired(dd, np.zeros_like(dd), alpha)
        else:
            da, db = (np.array(list(donor_means(e, cfg).values())) for e in (ea, eb))
            ok, n_don = da.size >= 2 and db.size >= 2, f"{da.size},{db.size}"
            if ok:
                d_diff, d_ci = ci_welch(da, db, alpha)
        if ok:
            row.update({"n_donors": n_don, "diff_donor_C": float(d_diff),
                        "ci90_donor_low": float(d_ci[0]), "ci90_donor_high": float(d_ci[1]),
                        "equivalent_donor": bool(d_ci[0] > -margin and d_ci[1] < margin)})
        rows.append(row)
    table = pd.DataFrame(rows)
    fam = table["role"].eq("tissue") & table.get("p_raw", pd.Series(np.nan, index=table.index)).notna()
    table["p_holm"] = np.nan
    if fam.any():
        table.loc[fam, "p_holm"] = holm(table.loc[fam, "p_raw"].to_numpy())
    return pd.DataFrame(models), table


def species_sensitivity(runs, groups, cfg):
    """POST HOC sensitivity of one unpaired comparison at the primary endpoint
    (config [sensitivity]; added 2026-10-05). Same statistic as `compare`: Welch
    difference of dT with its (1 - 2 alpha) confidence interval against the
    equivalence margin. Rows:
      as specified          reference group against the comparator group;
      leave one out         the same with each included trace of either group
                            removed in turn;
      pooled comparator     reference group against all the groups listed in
                            pooled_comparators taken together.
    Returns an empty table when [sensitivity] is absent or a group is missing."""
    sc = cfg.get("sensitivity")
    cols = ["analysis", "removed", "n_reference", "n_comparator", "diff_C",
            "ci90_low", "ci90_high", "equivalent"]
    if not sc or sc["reference"] not in groups or sc["comparator"] not in groups:
        return pd.DataFrame(columns=cols)
    s = cfg["statistics"]
    alpha, margin, w = s["alpha"], s["equivalence_margin_C"], cfg["qc"]["final_window_s"]
    ch = cfg["channels"]["primary_endpoint"]

    def traces(ids):
        out = []
        for rid in ids:
            v = endpoint_per_run(runs, [rid], ch, w)
            if v.size:
                out.append((rid, float(v[0])))
        return out

    def row(label, removed, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        diff, ci = ci_welch(a, b, alpha)
        return {"analysis": label, "removed": removed, "n_reference": a.size, "n_comparator": b.size,
                "diff_C": float(diff), "ci90_low": float(ci[0]), "ci90_high": float(ci[1]),
                "equivalent": bool(ci[0] > -margin and ci[1] < margin)}

    ref, comp = traces(groups[sc["reference"]]), traces(groups[sc["comparator"]])
    va, vb = [v for _, v in ref], [v for _, v in comp]
    rows = [row("as specified", "", va, vb)]
    for i, (rid, _) in enumerate(ref):
        if len(va) > 2:
            rows.append(row("leave one out", rid, va[:i] + va[i + 1:], vb))
    for i, (rid, _) in enumerate(comp):
        if len(vb) > 2:
            rows.append(row("leave one out", rid, va, vb[:i] + vb[i + 1:]))
    pooled = [g for g in sc.get("pooled_comparators", []) if g in groups]
    if pooled:
        vp = [v for g in pooled for _, v in traces(groups[g])]
        rows.append(row("pooled comparator", " + ".join(pooled), va, vp))
    return pd.DataFrame(rows, columns=cols)


def _values_by_eye(runs, ids, ch, window_s):
    """{(species, status, eye): (T_initial, T_final)} of the included traces of `ch`,
    repeats of one eye averaged (as endpoint_by_eye does for dT)."""
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
            by.setdefault((r.species, r.status, r.eye_id), []).append((float(v[0]), float(v[w].mean())))
    return {k: tuple(np.mean(np.asarray(v), axis=0)) for k, v in by.items()}


def geometry_contrast(runs, groups, cfg):
    """POST HOC contrast of one ring against the mean of several others in the same
    eye (config [geometry_contrast]; added 2026-10-06). For every site, the eyes with
    an included trace under the target ring and under every pooled ring: the paired
    difference  mean(pooled) - target  of dT (+ = the pooled rings cooled more), of the
    starting temperature and of the final temperature, each with its (1 - 2 alpha)
    confidence interval; equivalence of dT against the margin; Holm correction of the
    dT tests across the tissue sites. Returns an empty table when the section is
    absent or a group is missing."""
    gc = cfg.get("geometry_contrast")
    cols = ["channel", "role", "n_eyes", "dT_pooled_C", "dT_target_C", "diff_dT_C", "ci90_low",
            "ci90_high", "p_raw", "p_holm", "equivalent", "diff_T_initial_C",
            "ci90_T_initial_low", "ci90_T_initial_high", "diff_T_final_C", "ci90_T_final_low",
            "ci90_T_final_high"]
    if not gc or gc["target"] not in groups or any(g not in groups for g in gc["pooled"]):
        return pd.DataFrame(columns=cols)
    s, ch_cfg = cfg["statistics"], cfg["channels"]
    alpha, margin, w = s["alpha"], s["equivalence_margin_C"], cfg["qc"]["final_window_s"]
    rows = []
    for ch in ch_cfg["temperature"]:
        tgt = _values_by_eye(runs, groups[gc["target"]], ch, w)
        pool = [_values_by_eye(runs, groups[g], ch, w) for g in gc["pooled"]]
        shared = sorted(set(tgt).intersection(*[set(p) for p in pool]))
        row = {"channel": ch, "role": "device" if ch in ch_cfg["device"] else "tissue",
               "n_eyes": len(shared)}
        if len(shared) >= 2:
            t = np.array([tgt[k] for k in shared])                       # columns: T_initial, T_final
            p = np.mean([[q[k] for k in shared] for q in pool], axis=0)
            dt_p, dt_t = p[:, 0] - p[:, 1], t[:, 0] - t[:, 1]
            d, ci = ci_paired(dt_p, dt_t, alpha)
            d0, ci0 = ci_paired(p[:, 0], t[:, 0], alpha)
            d1, ci1 = ci_paired(p[:, 1], t[:, 1], alpha)
            row.update({"dT_pooled_C": float(dt_p.mean()), "dT_target_C": float(dt_t.mean()),
                        "diff_dT_C": float(d), "ci90_low": float(ci[0]), "ci90_high": float(ci[1]),
                        "p_raw": float(stats.ttest_rel(dt_p, dt_t).pvalue),
                        "equivalent": bool(ci[0] > -margin and ci[1] < margin),
                        "diff_T_initial_C": float(d0), "ci90_T_initial_low": float(ci0[0]),
                        "ci90_T_initial_high": float(ci0[1]), "diff_T_final_C": float(d1),
                        "ci90_T_final_low": float(ci1[0]), "ci90_T_final_high": float(ci1[1])})
        rows.append(row)
    table = pd.DataFrame(rows, columns=[c for c in cols if c != "p_holm"])
    table["p_holm"] = np.nan
    fam = table["role"].eq("tissue") & table["p_raw"].notna()
    if fam.any():
        table.loc[fam, "p_holm"] = holm(table.loc[fam, "p_raw"].to_numpy())
    return table[cols]
