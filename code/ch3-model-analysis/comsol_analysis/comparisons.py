"""Design comparisons (notebook Step 13, made batch): each group is ONE deterministic FEM run,
so the per-channel difference is the effect (no p-values). Fixed comparison sets:

  variants   V1 vs V2 vs V3 at the baseline operating point (parametric T0 25, Tb 37, Tc 4 °C)
  materials  Aluminum vs Ceramic vs Glass, per variant
  silicone   Baseline (parametric baseline run) vs 0.5 mm vs 1.0 mm, per variant
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .metrics import thermal_summary_run


def _baseline_run(reg, variant, cfg):
    b = cfg["geometry"]["scr_baseline_point"]
    P = reg.manifest[(reg.manifest.block == "scr_parametric") & (reg.manifest.variant == variant)]
    P = P[(abs(P.T0_K - b["T0_K"]) < 0.5) & (abs(P.Tb_K - b["Tb_K"]) < 0.5) & (abs(P.Tc_K - b["Tc_K"]) < 0.5)]
    return None if P.empty else P.iloc[0].run_id


def comparison_sets(reg, cfg):
    M, V = reg.manifest, cfg["vocabulary"]["variants"]
    sets = {"variants_baseline": {v: _baseline_run(reg, v, cfg) for v in V}}
    for v in V:
        sets[f"materials_{v}"] = {m: M[(M.block == "scr_material") & (M.variant == v) & (M.material == m)].run_id.iloc[0]
                                  for m in ("Aluminum", "Ceramic", "Glass")
                                  if not M[(M.block == "scr_material") & (M.variant == v) & (M.material == m)].empty}
        sil = {"Baseline": _baseline_run(reg, v, cfg)}
        for s in ("0.5 mm", "1.0 mm"):
            r = M[(M.block == "scr_silicone_internal") & (M.variant == v) & (M.silicone == s)]
            if not r.empty:
                sil[s] = r.run_id.iloc[0]
        sets[f"silicone_{v}"] = sil
    return {k: {g: rid for g, rid in s.items() if rid} for k, s in sets.items()}


def _dT(sig, ch):
    y = pd.to_numeric(sig[ch], errors="coerce").dropna().values
    return float(y[0] - y[-1]) if len(y) >= 4 else np.nan


def effect_table(reg, groups, cfg):
    """Every pair: per channel ΔT_A, ΔT_B, Δ(ΔT), Δ(T_final), plus trace-difference mean / max / RMS."""
    rows = []
    for (na, ra), (nb, rb) in itertools.combinations(groups.items(), 2):
        A, B = reg.signals[ra], reg.signals[rb]
        for ch in [c for c in A.columns if c in B.columns]:
            dA, dB = _dT(A, ch), _dT(B, ch)
            ya, yb = A[ch].dropna(), B[ch].dropna()
            lo, hi = max(A.index.min(), B.index.min()), min(A.index.max(), B.index.max())
            tg = A.index[(A.index >= lo) & (A.index <= hi)]
            diff = np.interp(tg, A.index, A[ch].values) - np.interp(tg, B.index, B[ch].values)
            rows.append({"A": na, "B": nb, "channel": ch, "dT_A (K)": dA, "dT_B (K)": dB, "d(dT) (K)": dA - dB,
                         "d(T_final) (K)": float(ya.iloc[-1] - yb.iloc[-1]),
                         "mean_trace_diff (K)": float(np.nanmean(diff)), "max_abs_trace_diff (K)": float(np.nanmax(np.abs(diff))),
                         "RMS_trace_diff (K)": float(np.sqrt(np.nanmean(diff ** 2)))})
    df = pd.DataFrame(rows)
    return df.assign(_e=df["d(dT) (K)"].abs()).sort_values(["A", "B", "_e"], ascending=[True, True, False]).drop(columns="_e") if not df.empty else df


def group_summaries(reg, groups, cfg):
    return pd.concat([thermal_summary_run(reg.signals[rid], cfg).assign(group=g, run_id=rid)
                      for g, rid in groups.items()], ignore_index=True)
