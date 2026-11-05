"""Figures. Probe colours match the ex-vivo experiment analysis; variant = colour/linestyle
where probes are not the grouping."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .safe_write import save_figure  # noqa: E402

PROBE_COLORS = {"SCR": "#1f77b4", "Equatorial Sclera": "#ff7f0e", "Vitreous Humor": "#2ca02c",
                "Optic Nerve Head": "#d62728", "Anterior Sclera": "#9467bd", "Posterior Sclera": "#8c564b",
                "Optic Nerve": "#8B0000", "Retina Avg": "#7E8AA2", "Retina Max": "#D95F02", "Retina Min": "#17becf",
                "Eye Avg": "#5B6B8C", "Eye Max": "#B0541F", "Eye Min": "#128B9E",
                "TEC": "#4285F4", "INw": "#0F9D58", "OUTw": "#F4B400", "Waterblock": "#DB4437"}
VARIANT_COLORS = {"V1": "#1f77b4", "V2": "#ff7f0e", "V3": "#2ca02c"}
VARIANT_STYLES = {"V1": "-", "V2": "--", "V3": ":"}
GROUP_STYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1)))

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "legend.fontsize": 8})


def _save(fig, path):
    fig.tight_layout()
    save_figure(fig, path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def mesh_convergence(gci, family, order, path):
    d = gci[gci.family == family]
    chans = list(dict.fromkeys(d.channel))
    n = len(chans)
    ncol = 4 if n > 4 else n
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, ch in zip(axes.flat, chans):
        for _, r in d[d.channel == ch].iterrows():
            vals = [r[f"T_{m} (C)"] for m in order]
            ax.plot(order, vals, marker="o", color=VARIANT_COLORS[r.variant], ls=VARIANT_STYLES[r.variant],
                    label=f"{r.variant}: GCI {r['GCI (%)']:.2g}% · ±{r['U_mesh (K)']:.3f} K" if np.isfinite(r["GCI (%)"])
                    else f"{r.variant}: oscillatory")
        ax.set_title(ch, color=PROBE_COLORS.get(ch, "#333"), fontweight="bold")
        ax.set_ylabel("final T (°C)")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Mesh convergence — {family} (final value per mesh; U_mesh = absolute GCI band)", fontweight="bold")
    _save(fig, path)


def gci_time(series, channels, path):
    chans = [c for c in channels if any(k[1] == c for k in series)]
    n = len(chans)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False, sharex=True)
    for ax, ch in zip(axes.flat, chans):
        for (v, c), (t, g, p, mono) in series.items():
            if c == ch:
                ax.plot(t, g, color=VARIANT_COLORS[v], ls=VARIANT_STYLES[v], label=v)
        ax.axhline(1, color="grey", ls=":", lw=1)
        ax.axhline(5, color="#b00", ls=":", lw=1)
        ax.set_title(ch, color=PROBE_COLORS.get(ch, "#333"), fontweight="bold")
        ax.set_ylabel("GCI_fine (%)")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (min)")
    fig.suptitle("Time-resolved GCI — SCR transient (gaps = oscillatory convergence)", fontweight="bold")
    _save(fig, path)


def scr_energy(E, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    sils = list(dict.fromkeys(E.silicone))
    hatches = {"Baseline": "", "0.5 mm": "//", "1.0 mm": "\\\\"}
    for row, (col, lab) in enumerate((("energy_J_per_m2", "energy density (MJ/m²)"), ("energy_J", "energy, full device (J)"))):
        scale = 1e6 if col == "energy_J_per_m2" else 1.0
        for ax, iface in zip(axes[row], ("Eye", "Eyelids", "Air")):
            sub = E[E.interface == iface]
            vs = list(dict.fromkeys(sub.variant))
            x = np.arange(len(vs))
            w = 0.8 / len(sils)
            for i, s in enumerate(sils):
                vals = [sub[(sub.variant == v) & (sub.silicone == s)][col].sum() / scale for v in vs]
                b = ax.bar(x + (i - (len(sils) - 1) / 2) * w, vals, w, color=[VARIANT_COLORS[v] for v in vs],
                           hatch=hatches.get(s, ""), edgecolor="black", lw=0.5, label=s)
                ax.bar_label(b, fmt="%.2f" if scale > 1 else "%.0f", fontsize=7)
            ax.set_xticks(x, vs)
            ax.set_title(iface if row == 0 else "", fontweight="bold")
            ax.set_ylabel(lab)
            if iface == "Eye":
                ax.legend(fontsize=7, title="silicone")
    fig.suptitle("SCR energy through each interface over 35 min (|q| integrated)", fontweight="bold")
    _save(fig, path)


def wb_closure(cl, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(cl))
    ax.bar(x - 0.19, cl.Q_h_W, 0.38, color="#C0504D", label="hot-side heat in (Fine Energies)")
    ax.bar(x + 0.19, cl.P_coolant_W, 0.38, color="#4E79A7", label="coolant ρ·Q0·cp·ΔT out")
    for i, r in cl.reset_index(drop=True).iterrows():
        ax.text(i, max(r.Q_h_W, r.P_coolant_W) * 1.02, f"{100 * r.ratio:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, cl.Variant)
    ax.set_ylabel("mean power over matched operating points (W)")
    ax.set_title("Waterblock energy closure", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def energy_balance(reg, path):
    """Σ|interface power| / |q_TEC| over time — should approach 1 at steady state."""
    M, S = reg.manifest, reg.signals
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for sil, ls in (("Baseline", "-"), ("0.5 mm", "--"), ("1.0 mm", ":")):
        trb, fxb = ("scr_tr", "scr_flux") if sil == "Baseline" else ("scr_silicone_tr", "scr_silicone_flux")
        for v in ("V1", "V2", "V3"):
            tr = M[(M.block == trb) & (M.variant == v) & (M.silicone == sil)]
            fx = M[(M.block == fxb) & (M.variant == v) & (M.silicone == sil)]
            if tr.empty or fx.empty:
                continue
            P = pd.DataFrame({r.interface: S[r.run_id]["q_total_W"] for _, r in fx.iterrows()}).abs().sum(axis=1)
            q = S[tr.iloc[0].run_id]["HeatFlux_TEC"].abs()
            ax.plot(P.index, P / q, color=VARIANT_COLORS[v], ls=ls, label=f"{v} {sil}")
    ax.axhline(1, color="black", lw=1)
    ax.axhline(2, color="grey", lw=1, ls=":")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("Σ|interface power| / |q_TEC|")
    ax.set_title("SCR energy balance (expected ≈ 1 at steady state)", fontweight="bold")
    ax.legend(ncol=3, fontsize=7)
    _save(fig, path)


def scr_rth(series, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for v, df in series.items():
        ax.plot(df.index, df.R_thermal_half / 2, color=VARIANT_COLORS[v], ls=VARIANT_STYLES[v], label=v)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("R_th, full device (K/W)")
    ax.set_title("SCR thermal resistance (T_hot − T_cold) / |q_TEC|", fontweight="bold")
    ax.legend()
    _save(fig, path)


def wb_rth(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for v, g in df.groupby("Variant"):
        axes[0].scatter(g["Q0 (m^3/s)"], g["R_WB (K/W)"], color=VARIANT_COLORS[v], s=18, label=v)
        axes[1].scatter(g["Tin (C)"], g["R_WB (K/W)"], color=VARIANT_COLORS[v], s=18, label=v)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Q0 (m³/s)")
    axes[1].set_xlabel("Tin (°C)")
    for ax in axes:
        ax.set_ylabel("R_WB (K/W)")
        ax.legend()
    fig.suptitle("Waterblock thermal resistance (V3 has no TEC probe in the sweep export)", fontweight="bold")
    _save(fig, path)


def pearson_heatmap(rho, title, path):
    fig, ax = plt.subplots(figsize=(max(0.75 * rho.shape[1] + 3, 7), 0.7 * rho.shape[0] + 1.8))
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(rho.shape[1]), rho.columns, rotation=45, ha="right")
    ax.set_yticks(range(rho.shape[0]), rho.index)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if np.isfinite(rho.values[i, j]):
                ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Pearson ρ")
    ax.set_title(title, fontweight="bold")
    _save(fig, path)


def anova_bars(a, title, path):
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(a) + 1.5))
    ax.barh(a.Factor, a["eta^2 (%)"], color=["#3B7DD8" if t == "Main" else "#9E9E9E" for t in a.Type])
    for y, v in enumerate(a["eta^2 (%)"]):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("η² (% of variance)")
    ax.set_title(title, fontweight="bold")
    _save(fig, path)


def contour(X, Y, probe, title, path):
    d = pd.concat([X, Y[probe].rename("val")], axis=1).dropna()
    tcs = [tc for tc in sorted(d.Tc_C.unique()) if d[d.Tc_C == tc].T0_C.nunique() >= 2 and d[d.Tc_C == tc].Tb_C.nunique() >= 2]
    if not tcs:
        return
    fig, axes = plt.subplots(1, len(tcs), figsize=(4.6 * len(tcs), 3.9), squeeze=False, sharey=True)
    for ax, tc in zip(axes[0], tcs):
        piv = d[d.Tc_C == tc].pivot_table(index="Tb_C", columns="T0_C", values="val")
        cs = ax.contourf(piv.columns, piv.index, piv.values, levels=10, cmap="RdBu_r")
        fig.colorbar(cs, ax=ax)
        ax.set_title(f"Tc = {tc:.0f} °C")
        ax.set_xlabel("T0 (°C)")
    axes[0][0].set_ylabel("Tb (°C)")
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def group_overlay(reg, groups, channels, title, path):
    chans = [c for c in channels if any(c in reg.signals[r].columns for r in groups.values())][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, squeeze=False)
    for ax, ch in zip(axes.flat, chans):
        for k, (g, rid) in enumerate(groups.items()):
            s = reg.signals[rid]
            if ch in s.columns:
                ax.plot(s.index, s[ch], color=PROBE_COLORS.get(ch, "#333"), ls=GROUP_STYLES[k % 6], lw=2, label=g)
        ax.set_title(ch, color=PROBE_COLORS.get(ch, "#333"), fontweight="bold")
        ax.set_ylabel("°C")
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(chans):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (min)")
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def epsilon_bars(eps, run_eps, path):
    """ε per site: variants from the parametric superposition fit; materials / silicone at baseline."""
    sites = ["Anterior Sclera", "Equatorial Sclera", "Vitreous Humor", "Retina Avg", "Posterior Sclera",
             "Optic Nerve Head", "Optic Nerve"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    x = np.arange(len(sites))
    for i, v in enumerate(("V1", "V2", "V3")):
        d = eps[eps.variant == v].set_index("site").reindex(sites)
        axes[0].bar(x + (i - 1) * 0.27, d["epsilon (w_Tc)"], 0.27, color=VARIANT_COLORS[v], label=v)
    axes[0].set_title("Ring design (27-run sweep fit)", fontweight="bold")
    for ax, study, key, levels in ((axes[1], "Material", "material", ["Aluminum", "Ceramic", "Glass"]),
                                   (axes[2], "Silicone", "silicone", ["0.5 mm", "1.0 mm"])):
        r = run_eps[(run_eps.study == study) & (run_eps.variant == "V1")]
        cols = ["#4E79A7", "#F28E2B", "#9C755F"]
        if study == "Silicone":
            base = run_eps[(run_eps.study == "Parametric baseline") & (run_eps.variant == "V1")].set_index("site").reindex(sites)
            ax.bar(x - 0.27, base.epsilon, 0.27, color="#999", label="Baseline")
        for i, lv in enumerate(levels):
            d = r[r[key] == lv].set_index("site").reindex(sites)
            ax.bar(x + (i if study == "Silicone" else i - 1) * 0.27, d.epsilon, 0.27, color=cols[i], label=lv)
        ax.set_title(f"{study} (V1, baseline point)", fontweight="bold")
    for ax in axes:
        ax.set_xticks(x, [s.replace(" ", "\n") for s in sites], fontsize=8)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("ε = (Tb − T_site) / (Tb − T_ring)")
    fig.suptitle("Cooling effectiveness — fraction of the ring's cold setpoint reaching each site", fontweight="bold")
    _save(fig, path)


def exvivo(cmp, path):
    groups = list(dict.fromkeys(cmp.group))
    sites = list(dict.fromkeys(cmp.site))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.3), sharey=True, squeeze=False)
    x = np.arange(len(sites))
    for ax, g in zip(axes[0], groups):
        d = cmp[cmp.group == g].set_index("site").reindex(sites)
        ax.bar(x - 0.2, d["eps_bath model"], 0.4, color="#4E79A7", label=f"model ({d.model_run.dropna().iloc[0].split('_')[2]})")
        ax.bar(x + 0.2, d["eps_bath exp"], 0.4, color="#E15759", label="ex vivo (mean)")
        ax.errorbar(x + 0.2, d["eps_bath exp"], yerr=[d["eps_bath exp"] - d["eps_bath exp 95%CI low"],
                    d["eps_bath exp 95%CI high"] - d["eps_bath exp"]], fmt="none", color="black", capsize=3)
        ax.set_xticks(x, [s.replace(" ", "\n") for s in sites], fontsize=7)
        ax.set_title(g, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("ε_bath = (37 − T_final) / (37 − T_ring)")
    fig.suptitle("Model vs ex vivo (different conditions: no perfusion / eyelids ex vivo) — 95% CI of the mean",
                 fontweight="bold")
    _save(fig, path)


def flow_model(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = ["Coarse", "Normal", "Fine"]
    for ax, p in zip(axes, ("TEC (C)", "Waterblock (C)")):
        for (v, flow), g in runs[runs.issue == ""].groupby(["variant", "flow"]):
            g = g.set_index("mesh").reindex(order)
            if g[p].notna().all():
                ax.plot(order, g[p], marker="o", color=VARIANT_COLORS[v], ls="-" if flow == "Laminar" else "--",
                        label=f"{v} {flow}")
        ax.set_title(p.replace(" (C)", "") + " temperature, same operating point", fontweight="bold")
        ax.set_ylabel("°C at 3.5 min")
        ax.legend(fontsize=7)
    fig.suptitle("Waterblock: laminar (solid) is mesh-converged, turbulent SST (dashed) is not", fontweight="bold")
    _save(fig, path)


def wb_resistance(hot, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for v, g in hot.groupby("Variant"):
        m = g.groupby("Q0 (L/min)")[["R_TEC (K/W)", "R_WB (K/W)"]].mean()
        if m["R_TEC (K/W)"].notna().any():
            axes[0].plot(m.index, m["R_TEC (K/W)"], marker="o", color=VARIANT_COLORS[v], label=v)
        axes[1].plot(m.index, m["R_WB (K/W)"], marker="o", color=VARIANT_COLORS[v], label=v)
    axes[0].set_title("TEC hot face → coolant (V3: no TEC probe)", fontweight="bold")
    axes[1].set_title("Waterblock surface (avg) → coolant", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("flow rate (L/min)")
        ax.set_ylabel("R (K/W), laminar stationary Fine mesh")
        ax.legend()
    _save(fig, path)
