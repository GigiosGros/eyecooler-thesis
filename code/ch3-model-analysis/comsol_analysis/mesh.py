"""Mesh convergence: three-grid GCI (Celik et al. 2008), final value and time-resolved
(notebook Steps 6b–6d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import filter_manifest


def calculate_gci(f_fine, f_med, f_coarse, n_fine, n_med, n_coarse, Fs, p_cap=None):
    """Celik iterative three-grid GCI. r from the DOF cube root per grid pair.
    Returns (GCI_fine %, asymptotic ratio or 'Oscillatory', observed order p).
    p_cap: if given, the GCI is computed with min(p, p_cap) (the observed p and the
    asymptotic ratio are still reported with the observed p). None = notebook behaviour."""
    try:
        r21 = (n_fine / n_med) ** (1 / 3)
        r32 = (n_med / n_coarse) ** (1 / 3)
        eps21, eps32 = f_med - f_fine, f_coarse - f_med
        if abs(eps21) < 1e-15:
            return 0.0, 1.0, float("nan")
        s = np.sign(eps32 / eps21)
        if s == -1:
            return float("nan"), "Oscillatory", float("nan")
        p = 2.0
        for _ in range(100):
            num, den = r21 ** p - s, r32 ** p - s
            if abs(den) < 1e-15:
                break
            p_new = (1 / np.log(r21)) * abs(np.log(abs(eps32 / eps21)) + np.log(abs(num / den)))
            if abs(p_new - p) < 1e-4:
                p = p_new
                break
            p = p_new
        ea21 = abs((f_fine - f_med) / f_fine) if abs(f_fine) > 1e-15 else 0
        gci21 = (Fs * ea21) / (r21 ** p - 1)
        ea32 = abs((f_med - f_coarse) / f_med) if abs(f_med) > 1e-15 else 0
        gci32 = (Fs * ea32) / (r32 ** p - 1)
        asymp = gci32 / (r21 ** p * gci21) if abs(gci21) > 1e-15 else float("nan")
        if p_cap is not None and p > p_cap:
            gci21 = (Fs * ea21) / (r21 ** p_cap - 1)
        return round(gci21 * 100, 4), round(asymp, 4), round(p, 3)
    except Exception:  # noqa: BLE001
        return float("nan"), "Error", float("nan")


def mesh_band(f_fine, f_med, f_coarse, n_fine, n_med, n_coarse, cfg):
    """GCI with the configured corrections. Returns a dict with the observed and used
    order, GCI %, the absolute band U (K), the asymptotic ratio and the band method."""
    mc, cor = cfg["mesh"], cfg["corrections"]
    cap = mc.get("p_formal", 2.0) if cor.get("gci_cap_p_formal", False) else None
    pct, asymp, p = calculate_gci(f_fine, f_med, f_coarse, n_fine, n_med, n_coarse,
                                  mc["gci_safety_factor"], p_cap=cap)
    osc = isinstance(asymp, str)
    spread = max(f_fine, f_med, f_coarse) - min(f_fine, f_med, f_coarse)
    # diverging: the change between the two finest meshes is larger than between the two
    # coarsest, so the sequence is not converging and Richardson extrapolation has no meaning
    # (changes below mesh.divergent_min_change_C are numerical noise, not divergence)
    div = bool(cor.get("gci_flag_divergent", False) and not osc
               and abs(f_med - f_fine) > abs(f_coarse - f_med)
               and abs(f_med - f_fine) > mc.get("divergent_min_change_C", 0.0))
    if div:
        U_gci = pct / 100 * abs(f_fine) if np.isfinite(pct) else 0.0
        U, method, p_used = max(spread, U_gci), "larger of 3-mesh spread and GCI (diverging)", np.nan
        pct = 100 * U / abs(f_fine) if abs(f_fine) > 1e-15 else np.nan
    elif osc and cor.get("gci_oscillatory_spread", False):
        U, method, p_used = spread, "spread of 3 meshes (oscillatory)", np.nan
        pct = 100 * U / abs(f_fine) if abs(f_fine) > 1e-15 else np.nan
    else:
        U = pct / 100 * abs(f_fine) if np.isfinite(pct) else np.nan
        p_used = min(p, cap) if (cap is not None and np.isfinite(p)) else p
        method = ("GCI, p capped at formal order" if cap is not None and np.isfinite(p) and p > cap
                  else "GCI" if np.isfinite(pct) else "undefined")
    return {"pct": pct, "asymp": asymp, "p": p, "p_used": p_used, "U": U, "osc": osc,
            "spread": spread, "method": method, "div": div}


def gci_status(val):
    if pd.isna(val) or isinstance(val, str):
        return "NON-MONOTONIC"
    if val < 0.1:
        return "EXCELLENT"
    if val < 1.0:
        return "CONVERGED"
    if val < 5.0:
        return "ACCEPTABLE"
    return "REVISE"


def _wb_fine_runs(reg, variant, cfg):
    """Fine-mesh sweep runs at the reference Q0 / Tin / Qh (all T0 levels)."""
    ref = cfg["mesh"]["wb_ref_point"]
    F = filter_manifest(reg.manifest, family="Waterblock", block="wb_aux", variant=variant, mesh="Fine")
    return F, ref


def wb_ref_run(reg, variant, cfg):
    """Notebook rule: the Fine run nearest the reference point (relative distance)."""
    F, ref = _wb_fine_runs(reg, variant, cfg)
    if F.empty:
        return None
    d = sum(((F[k] - ref[k]) / max(abs(ref[k]), 1e-9)) ** 2 for k in ("Q0", "Tin_K", "T0_K", "Qh"))
    return F.loc[d.idxmin(), "run_id"]


def wb_fine_value(reg, variant, channel, cfg):
    """Fine-mesh value (alpha = 1) at the Coarse/Normal operating point.
    Notebook: nearest run (T0 = 308.15 K; the reference T0 = 298.15 K is not on the Fine grid).
    Corrected: linear interpolation in T0 between the two Fine runs bracketing it at the same
    Q0 / Tin / Qh."""
    if not cfg["corrections"]["wb_fine_interpolate_T0"]:
        rid = wb_ref_run(reg, variant, cfg)
        return _last(reg.signals[rid], channel) if rid else None
    F, ref = _wb_fine_runs(reg, variant, cfg)
    same = F[np.isclose(F.Q0, ref["Q0"]) & np.isclose(F.Tin_K, ref["Tin_K"]) & np.isclose(F.Qh, ref["Qh"])]
    pts = sorted((float(r.T0_K), _last(reg.signals[r.run_id], channel)) for _, r in same.iterrows())
    pts = [(t, v) for t, v in pts if v is not None and np.isfinite(v)]
    if len(pts) < 2:
        return None
    T, Y = zip(*pts)
    if not (T[0] <= ref["T0_K"] <= T[-1]):
        return None                      # would be extrapolation
    return float(np.interp(ref["T0_K"], T, Y))


def _last(sig, channel):
    if channel not in sig.columns:
        return None
    col = sig[channel].dropna()
    return float(col.iloc[-1]) if len(col) else None


def final_value(reg, family, variant, mesh, channel, cfg):
    """Converged value for (family, variant, mesh, channel): last time sample (SCR) / alpha = 1 (WB)."""
    if family == "SCR":
        sub = filter_manifest(reg.manifest, family="SCR", block="scr_mesh_convergence", variant=variant, mesh=mesh)
        return None if sub.empty else _last(reg.signals[sub.iloc[0].run_id], channel)
    if mesh == "Fine":
        return wb_fine_value(reg, variant, channel, cfg)
    sub = filter_manifest(reg.manifest, family="Waterblock", block="wb_aux", variant=variant, mesh=mesh)
    return None if sub.empty else _last(reg.signals[sub.iloc[0].run_id], channel)


def gci_table(reg, cfg):
    """Step 6d table for every (family, variant, channel), plus two scale-free columns:
    U_mesh_K = GCI_fine x |phi_fine| = Fs |phi_1 - phi_2| / (r21^p - 1)  (absolute band, K), and
    GCI_kelvin_pct = the same band as a % of the fine value in kelvin. The notebook's GCI % is
    relative to the value in °C, so it depends on where 0 °C falls (see README)."""
    mc, V = cfg["mesh"], cfg["vocabulary"]["variants"]
    K = cfg["units"]["k_to_c"]
    fams = [("SCR", cfg["probes"]["scr_point"] + cfg["probes"]["retina_agg"] + cfg["probes"]["eye_agg"],
             mc["scr_order"], mc["dof_scr"]),
            ("Waterblock", cfg["probes"]["waterblock"], mc["wb_order"], mc["dof_wb"])]
    rows = []
    for fam, chans, order, dofs in fams:
        for v in V:
            for ch in chans:
                vals = [final_value(reg, fam, v, m, ch, cfg) for m in order]
                if any(x is None or not np.isfinite(x) for x in vals):
                    continue
                d = dofs[v]
                b = mesh_band(vals[2], vals[1], vals[0], d[order[2]], d[order[1]], d[order[0]], cfg)
                pct, asymp, p, osc, U = b["pct"], b["asymp"], b["p"], b["osc"], b["U"]
                rows.append({"family": fam, "variant": v, "channel": ch,
                             f"T_{order[0]} (C)": vals[0], f"T_{order[1]} (C)": vals[1], f"T_{order[2]} (C)": vals[2],
                             "mesh_spread (K)": max(vals) - min(vals), "p": p, "p_used": b["p_used"],
                             "band_method": b["method"], "GCI (%)": pct,
                             "asymptotic_ratio": np.nan if osc else asymp,
                             "status": "DIVERGING" if b["div"] else gci_status(asymp if osc else pct),
                             "U_mesh (K)": U, "GCI_kelvin (%)": 100 * U / (vals[2] + K) if np.isfinite(U) else np.nan})
    return pd.DataFrame(rows)


def gci_timeseries(reg, variant, channel, cfg):
    """GCI at every sample of the COARSER-mesh time grid (other meshes interpolated onto it)."""
    mc = cfg["mesh"]
    order, dofs = mc["scr_order"], mc["dof_scr"].get(variant, {})
    sigs = {}
    for m in order:
        sub = filter_manifest(reg.manifest, family="SCR", block="scr_mesh_convergence", variant=variant, mesh=m)
        if sub.empty:
            return None
        s = reg.signals[sub.iloc[0].run_id]
        if channel not in s.columns:
            return None
        sigs[m] = s[channel].dropna()
    if any(len(s) < 4 for s in sigs.values()) or any(dofs.get(m) is None for m in order):
        return None
    t = sigs[order[0]].index.values
    f = {m: np.interp(t, sigs[m].index.values, sigs[m].values) for m in order}
    gci, pp, mono = [], [], []
    for i in range(len(t)):
        cap = mc.get("p_formal", 2.0) if cfg["corrections"].get("gci_cap_p_formal", False) else None
        pct, asymp, p = calculate_gci(f[order[2]][i], f[order[1]][i], f[order[0]][i],
                                      dofs[order[2]], dofs[order[1]], dofs[order[0]], mc["gci_safety_factor"],
                                      p_cap=cap)
        gci.append(pct if np.isfinite(pct) else np.nan)
        pp.append(p if np.isfinite(p) else np.nan)
        mono.append(not isinstance(asymp, str))
    return t, np.asarray(gci, float), np.asarray(pp, float), np.asarray(mono)


def gci_time_summary(reg, cfg):
    """Step 6c table: mean GCI over the early / mid / late thirds of the transient."""
    rows, series = [], {}
    chans = cfg["probes"]["scr_point"] + cfg["probes"]["retina_agg"] + cfg["probes"]["eye_agg"]
    for v in cfg["vocabulary"]["variants"]:
        for ch in chans:
            out = gci_timeseries(reg, v, ch, cfg)
            if out is None:
                continue
            series[(v, ch)] = out
            t, g, p, mono = out
            n = len(g)
            third = max(1, n // 3)
            m = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else np.nan  # noqa: E731
            early, mid, late = m(g[:third]), m(g[third:2 * third]), m(g[2 * third:])
            drift = late - early
            note = ("drifts up" if late > 1.5 * max(early, 1e-9) and drift > 0.5 else "stable" if abs(drift) < 0.5
                    else "trending down" if drift < -0.5 else "small drift") if np.isfinite(drift) else "undefined (oscillatory)"
            rows.append({"variant": v, "channel": ch, "n_samples": n, "n_monotonic": int(mono.sum()),
                         "early mean (%)": early, "mid mean (%)": mid, "late mean (%)": late,
                         "drift early->late (pp)": drift, "mean p": m(p), "note": note})
    return pd.DataFrame(rows), series
