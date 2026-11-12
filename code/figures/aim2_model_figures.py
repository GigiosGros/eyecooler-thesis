"""Chapter 3 result figures, drawn from the analysis packages' result tables.

Figures used in the thesis: paired-differences (Fig 3.7). The script also draws the COMSOL
versions of mesh-convergence, material-effectiveness, energy-split, required-ring-temperature,
model-vs-exvivo, model-time-course, model-vs-exvivo-time and heat-exchanger-performance; the
thesis takes those from aim2_python_figures.py and aim2_python_hx.py. Since 2026-11-11 the
script does NOT write them (it skips every name in THESIS_FROM_OTHER, and any figure that
MANIFEST.json credits to another script) unless run with --force, so the run order no longer
matters. Not in the thesis, still written: energy-split-all, heat-exchanger-mesh-convergence.

Inputs are SNAPSHOTS of the packages' result CSVs, kept in figures/aim2/sources/data/:
  model/   <- 03 - Experiments & Data/Analysis Code/thesis-model-analysis/results/
             (package = GitHub GigiosGros/EyeCooler-AnalysisCode, branch
              thesis-model-analysis, release thesis-model-analysis-v1.1.2)
  exvivo/  <- 03 - Experiments & Data/Analysis Code/thesis-analysis/results/03_comparisons/
             (thesis-analysis package, notebook-v4.28 settings of 2026-09-30; not under git)
Plotting only: no value is computed here except the per-bar share of heat drawn from the
eye (eye / total) and the selection of which energy balances closed (closure < 0.99).

Run:  python code/figures/aim2_model_figures.py            (from the snapshots)
      python code/figures/aim2_model_figures.py --refresh  (re-copy the snapshots first)
Every run updates figures/aim2/MANIFEST.json (script, inputs, SHA-256, origins).
Conventions: temperatures in degC; "heat exchanger", not "waterblock"; "ring", not "SCR";
site names as Chapter 2 (model optic nerve head = proximal optic nerve; probe 1 cm
behind the globe = distal optic nerve; retina domain mean = average retina).
"""
import shutil
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from provenance import FIG, SOURCES, record, written_by_other  # noqa: E402

BOX = FIG.parents[4]  # .../Eye Cooler
PKG_MODEL = BOX / "03 - Experiments & Data" / "Analysis Code" / "thesis-model-analysis" / "results"
PKG_EXVIVO = BOX / "03 - Experiments & Data" / "Analysis Code" / "thesis-analysis" / "results" / "03_comparisons"
DATA = SOURCES / "data"
MODEL_FILES = ["02_mesh_convergence/gci.csv", "03_energy/scr_energy.csv",
               "03_energy/scr_energy_balance.csv", "09_effectiveness/run_epsilon.csv",
               "09_effectiveness/required_ring_temperature.csv",
               "10_exvivo_comparison/model_vs_exvivo.csv",
               "11_waterblock/resistance_vs_flow.csv", "11_waterblock/hot_side_resistance.csv",
               "09_effectiveness/timeseries_thesis.csv"]
PKG_GROUPS = PKG_EXVIVO.parent / "02_groups"
GROUP_CURVES = ["Human__aluminum_original", "Human__aluminum_reduced_bulk", "Human__ceramic_original"]
EXVIVO_DIRS = ["Human__aluminum_original__vs__Human__aluminum_reduced_bulk",
               "Human__aluminum_original__vs__Human__ceramic_original"]
USED, ORIGINS = [], {}


def snap_model(rel):
    return DATA / "model" / rel.replace("/", "__")


def snap_exvivo(d):
    return DATA / "exvivo" / (d + "__channels.csv")


PKG_MODEL_CFG = PKG_MODEL.parent / "config" / "config.toml"


def refresh():
    (DATA / "model").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PKG_MODEL_CFG, DATA / "model" / "config.toml")
    for rel in MODEL_FILES:
        dst = snap_model(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PKG_MODEL / rel, dst)
    for g in GROUP_CURVES:
        dst = DATA / "exvivo" / (g + "__mean_curves.csv")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PKG_GROUPS / g / "mean_curves.csv", dst)
    for d in EXVIVO_DIRS:
        dst = snap_exvivo(d)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PKG_EXVIVO / d / "channels.csv", dst)
    print("snapshots refreshed")


def mdata(rel):
    p = snap_model(rel)
    USED.append(p)
    ORIGINS[p.name] = "thesis-model-analysis/results/" + rel + " (release thesis-model-analysis-v1.1.2)"
    return p


def mconfig():
    p = DATA / "model" / "config.toml"
    USED.append(p)
    ORIGINS[p.name] = "thesis-model-analysis/config/config.toml (release thesis-model-analysis-v1.1.2)"
    import tomllib
    return tomllib.loads(p.read_text(encoding="utf-8"))


def gcurves(g):
    p = DATA / "exvivo" / (g + "__mean_curves.csv")
    USED.append(p)
    ORIGINS[p.name] = "thesis-analysis/results/02_groups/" + g + "/mean_curves.csv"
    return p


def xdata(d):
    p = snap_exvivo(d)
    USED.append(p)
    ORIGINS[p.name] = "thesis-analysis/results/03_comparisons/" + d + "/channels.csv"
    return p


if "--refresh" in sys.argv:
    refresh()

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
SITE_C = {"Ring": "#1f77b4", "Anterior Sclera": "#9467bd", "Equatorial Sclera": "#ff7f0e",
          "Vitreous Humor": "#2ca02c", "Posterior Sclera": "#8c564b",
          "Optic Nerve Head": "#d62728", "Optic Nerve": "#7f1d1d", "Retina Avg": "#555555"}
NAME = {"Anterior Sclera": "Anterior sclera", "Equatorial Sclera": "Equatorial sclera",
        "Vitreous Humor": "Vitreous", "Posterior Sclera": "Posterior sclera",
        "Optic Nerve Head": "Proximal optic nerve", "Optic Nerve": "Distal optic nerve",
        "Retina Avg": "Average retina", "Ring": "SCR", "SCR": "SCR"}
ORDER = ["Anterior Sclera", "Equatorial Sclera", "Vitreous Humor", "Posterior Sclera",
         "Optic Nerve Head", "Optic Nerve", "Retina Avg"]
DES_C = {"V1": "#2b4c7e", "V2": "#d9822b", "V3": "#5a9e6f"}
DES_M = {"V1": ("o", "-"), "V2": ("s", "--"), "V3": ("^", ":")}

WRITTEN = []
# The thesis prints these from aim2_python_figures.py / aim2_python_hx.py (corrected Python runs);
# the versions drawn here from the COMSOL tables are superseded and must not overwrite them.
THESIS_FROM_OTHER = {"mesh-convergence.png", "material-effectiveness.png", "energy-split.png",
                     "required-ring-temperature.png", "model-vs-exvivo.png", "model-time-course.png",
                     "model-vs-exvivo-time.png", "heat-exchanger-performance.png"}
FORCE = "--force" in sys.argv


def save(fig, name):
    other = written_by_other(name, __file__)
    if not FORCE and (name in THESIS_FROM_OTHER or other):
        plt.close(fig)
        print(f"skipped {name}: the thesis uses the version drawn by "
              f"{other or 'aim2_python_figures.py / aim2_python_hx.py'} (use --force to overwrite)")
        return
    dst = FIG / name
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    WRITTEN.append(dst)
    print("wrote", name)

# 1 ---- mesh convergence
g = pd.read_csv(mdata("02_mesh_convergence/gci.csv"))
g = g[g.family == "SCR"]
sites = ORDER + ["SCR"]
cfg = mconfig()
order = cfg["mesh"]["scr_order"]          # coarse -> fine, as the gci.csv columns
nodes = cfg["mesh"]["dof_scr"]             # vertex counts per ring and mesh
fig, axs = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)
MESH_XLIM = (1.2e4, 2.5e5)   # mesh vertices: the same axis in the eye-model and the heat-exchanger mesh figures
MESH_XTICKS = [2e4, 5e4, 1e5, 2e5]


def mesh_xaxis(ax):
    ax.set_xscale("log")
    ax.set_xlim(*MESH_XLIM)
    ax.set_xticks(MESH_XTICKS)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, pos: f"{val / 1000:.0f}k"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


Y_HALF_SPAN = 0.5   # degC: every panel shows the middle of its own values plus and minus this (2026-10-12), so the
                    # panels have one vertical scale and the size of a change can be compared from site to site
for ax, site in zip(axs.flat, sites):
    seen = []
    for v in ["V1", "V2", "V3"]:
        r = g[(g.variant == v) & (g.channel == site)].iloc[0]
        y = [r[f"T_{m} (C)"] for m in order]
        x = [nodes[v][m] for m in order]
        mk, ls = DES_M[v]
        ax.plot(x, y, marker=mk, ls=ls, color=DES_C[v], ms=4, lw=1.2, label=v.lower())
        # 2026-10-12: the band of the grid convergence index is no longer drawn.  The bar on the finest mesh
        # is the size of the last step: plus and minus the change from the intermediate to the finest mesh
        # (Table tab:aim2-umesh, columns (b)).
        ax.errorbar(x[-1], y[-1], yerr=abs(y[-1] - y[-2]), color=DES_C[v], capsize=2.5, lw=1, fmt="none")
        seen += list(y) + [y[-1] - abs(y[-1] - y[-2]), y[-1] + abs(y[-1] - y[-2])]
    ax.set_title(NAME.get(site, site).replace("\n", " "), color=SITE_C.get(site, SITE_C["Ring"]),
                 fontsize=9.5, fontweight="bold")
    mesh_xaxis(ax)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    mid = 0.5 * (min(seen) + max(seen))
    assert max(seen) - min(seen) <= 2 * Y_HALF_SPAN, (site, min(seen), max(seen))
    ax.set_ylim(mid - Y_HALF_SPAN, mid + Y_HALF_SPAN)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.ticklabel_format(axis="y", useOffset=False)
for ax in axs[:, 0]: ax.set_ylabel("Final temperature (°C)", fontsize=9)
for ax in axs[1]: ax.set_xlabel("Mesh vertices (thousands)", fontsize=9)
axs[0, 0].legend(fontsize=8, frameon=False)
save(fig, "mesh-convergence.png")

# 1b ---- the same for the heat exchangers (2026-10-12): COMSOL's three meshes at the operating point of the
# mesh study, one panel per read-out, one line per design, the same 1 degC span and the same bar on the finest mesh
w = pd.read_csv(mdata("02_mesh_convergence/gci.csv"))
w = w[w.family == "Waterblock"]
worder = cfg["mesh"]["wb_order"]
wnodes = cfg["mesh"]["dof_wb"]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)
for ax, (ch, title) in zip(axs, [("TEC", "Module face"), ("Waterblock", "Block surface, average"), ("OUTw", "Coolant at the outlet")]):
    seen = []
    for v in ["V1", "V2", "V3"]:
        rows = w[(w.variant == v) & (w.channel == ch)]
        if rows.empty:                      # the module face of v3 was not exported from the model
            continue
        r = rows.iloc[0]
        y = [r[f"T_{m} (C)"] for m in worder]
        x = [wnodes[v][m] for m in worder]
        mk, ls = DES_M[v]
        ax.plot(x, y, marker=mk, ls=ls, color=DES_C[v], ms=4, lw=1.2, label=v.lower())
        ax.errorbar(x[-1], y[-1], yerr=abs(y[-1] - y[-2]), color=DES_C[v], capsize=2.5, lw=1, fmt="none")
        seen += list(y) + [y[-1] - abs(y[-1] - y[-2]), y[-1] + abs(y[-1] - y[-2])]
    mid = 0.5 * (min(seen) + max(seen))
    assert max(seen) - min(seen) <= 2 * Y_HALF_SPAN, (ch, min(seen), max(seen))
    ax.set_ylim(mid - Y_HALF_SPAN, mid + Y_HALF_SPAN)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    mesh_xaxis(ax)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Mesh vertices (thousands)", fontsize=9)
axs[0].set_ylabel("Temperature (°C)", fontsize=9)
axs[1].legend(fontsize=8, frameon=False)
save(fig, "heat-exchanger-mesh-convergence.png")

# 2 ---- paired differences (forest)
CH = [("SCR", "SCR"), ("Anterior Sclera", "Anterior sclera"), ("Equatorial Sclera", "Equatorial sclera"),
      ("Vitreous", "Vitreous"), ("Posterior Sclera", "Posterior sclera"),
      ("Optic Nerve", "Proximal optic nerve")]
CCOL = {"SCR": SITE_C["Ring"], "Anterior Sclera": SITE_C["Anterior Sclera"],
        "Equatorial Sclera": SITE_C["Equatorial Sclera"], "Vitreous": SITE_C["Vitreous Humor"],
        "Posterior Sclera": SITE_C["Posterior Sclera"], "Optic Nerve": SITE_C["Optic Nerve Head"]}
comps = [("Human__aluminum_original__vs__Human__ceramic_original", "(A) Aluminum − ceramic ring"),
         ("Human__aluminum_original__vs__Human__aluminum_reduced_bulk", "(B) Original − reduced-bulk ring")]
# x range from the data: every interval drawn in full, plus room for the n labels at right
_ci = pd.concat([pd.read_csv(xdata(d)) for d, _ in comps])
X_LO = np.floor(min(_ci.ci90_low.min(), -2)) - 0.5
X_HI = np.ceil(max(_ci.ci90_high.max(), 2)) + 2.0
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True)
for ax, (d, title) in zip(axs, comps):
    c = pd.read_csv(xdata(d)).set_index("channel")
    # 2026-10-12: no fixed +-2 degC band; each interval is read by its own limits
    ax.axvline(0, color="k", lw=0.8)
    for i, (ch, lab) in enumerate(CH):
        y = len(CH) - 1 - i
        r = c.loc[ch]
        col = CCOL[ch]
        ax.plot([r.ci90_low, r.ci90_high], [y, y], color=col, lw=2)
        ax.plot(r.diff_C, y, "o", ms=7, color=col, mfc=col, mew=1.5)
        for end in (r.ci90_low, r.ci90_high):                                       # a cap on both ends of the interval
            ax.plot([end, end], [y - 0.18, y + 0.18], color=col, lw=2)
        n = r.design.split("k=")[-1]
        ax.text(X_HI - 0.2, y, f"n = {n}", va="center", ha="right", fontsize=8, color="#444")
    ax.set_yticks(range(len(CH)), [l for _, l in CH][::-1])
    ax.set_xlim(X_LO, X_HI)
    ax.set_xlabel("Difference in cooling excursion ΔT (°C)")
    ax.set_title(title, fontsize=10, loc="left")
fig.legend(handles=[plt.Line2D([], [], marker="o", color="k", ls="-", lw=2, label="mean difference and 90 % confidence interval")],
           loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
save(fig, "paired-differences.png")

# 3 ---- material
e = pd.read_csv(mdata("09_effectiveness/run_epsilon.csv"))
mat = e[(e.study == "Material") & (e.variant == "V1")]
MC = {"Aluminum": "#264653", "Ceramic": "#e9c46a", "Glass": "#8ecae6"}   # 2026-10-28: clearer palette
ML = {"Aluminum": "Aluminum", "Ceramic": "Aluminum oxide", "Glass": "Fused silica"}
# two panels (2026-10-12): (A) the three materials on ring v1, (B) the three geometries in aluminum
geo = e[e.study == "Parametric baseline"]
fig, axs = plt.subplots(2, 1, figsize=(8, 6.4), constrained_layout=True, sharey=True)
x = np.arange(len(ORDER)); w = 0.26
ax = axs[0]
for k, m in enumerate(["Aluminum", "Ceramic", "Glass"]):
    vals = [mat[(mat.material == m) & (mat.site == s)].epsilon.iloc[0] for s in ORDER]
    ax.bar(x + (k - 1) * w, vals, w, color=MC[m], edgecolor="k", lw=0.5, label=ML[m])
ax.set_title("(A) Ring material (v1)", fontsize=10, loc="left")
ax = axs[1]
for k, v in enumerate(["V1", "V2", "V3"]):
    vals = [geo[(geo.variant == v) & (geo.site == s)].epsilon.iloc[0] for s in ORDER]
    ax.bar(x + (k - 1) * w, vals, w, color=DES_C[v], edgecolor="k", lw=0.5, label=v.lower())
ax.set_title("(B) Ring geometry (aluminum)", fontsize=10, loc="left")
for ax in axs:
    ax.set_xticks(x, [NAME[s].replace(" sclera", "\nsclera").replace(" optic", "\noptic")
                      .replace("Average ", "Average\n") for s in ORDER], fontsize=8.5)
    # site names in black (2026-10-28)
    ax.set_ylabel("Cooling effectiveness ε")
    ax.legend(frameon=False, fontsize=9)
save(fig, "material-effectiveness.png")

# 4 ---- energy split (full device, J over 35 min) from the COMSOL tables; the thesis uses the version
#   drawn by aim2_python_figures.py, so energy-split.png is skipped here (see save()):
#   energy-split.png      A: configurations whose energy balance did not close are hatched,
#                            eye share printed only where it closed
#   energy-split-all.png  B: all nine drawn alike, eye share printed on every bar
en = pd.read_csv(mdata("03_energy/scr_energy.csv"))
bal = pd.read_csv(mdata("03_energy/scr_energy_balance.csv"))
openset = {(r.variant, r.silicone) for r in bal.itertuples() if r.closure_if_interfaces_halved < 0.99}
IC = {"Eye": "#c0392b", "Eyelids": "#e6b0aa", "Air": "#d5dbdb"}


def energy_split(mark_open, name):
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    labels, pos = [], []
    p = 0
    for v in ["V1", "V2", "V3"]:
        for sil in ["Baseline", "0.5 mm", "1.0 mm"]:
            bottom = 0
            inc = mark_open and (v, sil) in openset
            for itf in ["Eye", "Eyelids", "Air"]:
                val = en[(en.variant == v) & (en.silicone == sil) & (en.interface == itf)].energy_J.iloc[0]
                ax.bar(p, val, 0.8, bottom=bottom, color=IC[itf], edgecolor="k", lw=0.5,
                       hatch="///" if inc else None, alpha=0.55 if inc else 1)
                bottom += val
            eye = en[(en.variant == v) & (en.silicone == sil) & (en.interface == "Eye")].energy_J.iloc[0]
            if not inc:
                ax.text(p, bottom + 80, f"{100*eye/bottom:.0f}%", ha="center", fontsize=7.5)
            labels.append({"Baseline": "none", "0.5 mm": "0.5", "1.0 mm": "1.0"}[sil])
            pos.append(p)
            p += 1
        p += 0.6
    ax.set_xticks(pos, labels, fontsize=8)
    for i, v in enumerate(["v1", "v2", "v3"]):
        ax.text(np.mean(pos[3*i:3*i+3]), -1120, v, ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Heat drawn over 35 min, whole device (J)")
    ax.set_xlabel("\nInsulating layer (mm), by ring", fontsize=9)
    handles = [Patch(fc=IC[k], ec="k", label=k) for k in IC]
    if mark_open:
        handles.append(Patch(fc="white", ec="k", hatch="///", label="incomplete (mesh not stored)"))
    ax.legend(handles=handles, frameon=False, fontsize=8.5, ncol=4, loc="upper right")
    ax.set_ylim(0, 9200)
    save(fig, name)


energy_split(True, "energy-split.png")
energy_split(False, "energy-split-all.png")

# 5 ---- required ring temperature
q = pd.read_csv(mdata("09_effectiveness/required_ring_temperature.csv"))
q = q[q.variant == "V1"]
# one panel (2026-10-12): the two targets give the same lines shifted, so they are drawn together,
# solid for 35 degC and dashed for 32 degC
fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
ax.axhspan(-40, 4, color="#f2d7d5", zorder=0)
ax.text(35.05, -37, "below the 4 °C floor of the device", fontsize=7.5, color="#922b21", va="bottom")
TARGET_STYLE = {35.0: ("-", "o", True), 32.0: ("--", "o", False)}
for tgt, (ls, mk, filled) in TARGET_STYLE.items():
    for s in ["Vitreous Humor", "Retina Avg", "Posterior Sclera", "Optic Nerve Head"]:
        d = q[(q.site == s) & (q["target (C)"] == tgt)].sort_values("Tb (C)")
        ax.plot(d["Tb (C)"], d["required_ring_T (C)"], marker=mk, ls=ls, color=SITE_C[s],
                mfc=SITE_C[s] if filled else "white")
ax.set_xlabel("Body temperature (°C)")
ax.set_xticks([35, 37, 39])
ax.set_ylabel("Ring temperature needed (°C)")
ax.set_ylim(-40, 36)
hs = [plt.Line2D([], [], color=SITE_C[s], lw=2, label=NAME[s].replace("\n", " "))
      for s in ["Vitreous Humor", "Retina Avg", "Posterior Sclera", "Optic Nerve Head"]]
ht = [plt.Line2D([], [], color="#444", ls="-", marker="o", label="site at 35 °C"),
      plt.Line2D([], [], color="#444", ls="--", marker="o", mfc="white", label="site at 32 °C")]
fig.legend(handles=hs + ht, frameon=False, fontsize=8.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.14))
save(fig, "required-ring-temperature.png")

# 6 ---- model vs ex vivo
mv = pd.read_csv(mdata("10_exvivo_comparison/model_vs_exvivo.csv"))
GM = {"Human Fresh Aluminum Original": ("o", "Original ring / v1"),
      "Human Fresh Aluminum Reduced Bulk": ("s", "Reduced-bulk ring / v2"),
      "Human Fresh Ceramic Original": ("^", "Ceramic ring / v1, aluminum oxide")}
fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
for _, r in mv.iterrows():
    mk, _ = GM[r.group]
    col = SITE_C[r.site]
    xm = r["eps_start model"]
    ym = r["eps_start exp"]
    lo = r["eps_start exp 95%CI low"]
    hi = r["eps_start exp 95%CI high"]
    ax.errorbar(xm, ym, yerr=[[ym - lo], [hi - ym]], fmt=mk, color=col, mfc=col, ms=6, capsize=2, lw=1)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Model effectiveness ε$_{Δ}$")
ax.set_ylabel("Ex-vivo effectiveness ε$_{Δ}$ (mean, 95% CI)")
h1 = [plt.Line2D([], [], marker=GM[k][0], color="#444", ls="", label=GM[k][1]) for k in GM]
h2 = [plt.Line2D([], [], marker="o", color=SITE_C[s], ls="", label=NAME[s]) for s in
      ["Anterior Sclera", "Equatorial Sclera", "Vitreous Humor", "Posterior Sclera", "Optic Nerve Head"]]
leg = ax.legend(handles=h1, frameon=False, fontsize=7.5, loc="upper left")
ax.add_artist(leg)
ax.legend(handles=h2, frameon=False, fontsize=7.5, loc="lower right")
save(fig, "model-vs-exvivo.png")

# 7 ---- heat exchanger
# 2026-10-05: the thermal-resistance panel was removed; the figure shows only
# whether each design keeps the hot side near the coolant and below the 43 °C limit.
hs = pd.read_csv(mdata("11_waterblock/hot_side_resistance.csv"))
fig = plt.figure(figsize=(7.5, 3.4), constrained_layout=True)
sub = fig.add_gridspec(1, 3, wspace=0.05)
for k, v in enumerate(["V1", "V2", "V3"]):
    ax = fig.add_subplot(sub[k], sharey=None if k == 0 else fig.axes[-1])
    d = hs[hs.Variant == v]
    col = "T_TEC (C)" if d["T_TEC (C)"].notna().any() else "T_WB_avg (C)"
    mk, _ = DES_M[v]
    ax.scatter(d["Tin (C)"], d[col], s=14, marker=mk, color=DES_C[v], alpha=0.7)
    ax.axhline(43, color="#922b21", lw=1.2)
    ax.set_ylim(0, 46)
    ax.set_xticks(sorted(d["Tin (C)"].unique()))
    ax.set_title(f"{v.lower()} ({'module face' if col == 'T_TEC (C)' else 'block'})", fontsize=8.5)
    ax.set_xlabel("Coolant inlet temperature (°C)" if k == 1 else "")
    if k == 0:
        ax.set_ylabel("Face-average temperature (°C)")
    else:
        ax.tick_params(labelleft=False)
    if k == 2:
        ax.text(0.97, 0.90, "43 °C limit", transform=ax.transAxes, ha="right", color="#922b21", fontsize=8)
save(fig, "heat-exchanger-performance.png")

# 8 ---- model time courses: cooling (baseline point, v1) then re-heat after switch-off
ts = pd.read_csv(mdata("09_effectiveness/timeseries_thesis.csv"))
cool = ts[(ts.study == "Cooling, baseline point") & (ts.variant == "V1")]
heat = ts[(ts.study == "Re-heat, V1") & ((ts["Tc (C)"] - 4).abs() < 0.5)]
t_off = cool["time (min)"].max()
fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
for site in ["SCR"] + ORDER:
    c = cool[cool.channel == site].sort_values("time (min)")
    h = heat[heat.channel == site].sort_values("time (min)")
    col = SITE_C.get(site, SITE_C["Ring"])
    lab = NAME.get(site, site).replace("\n", " ")
    ax.plot(c["time (min)"], c["T (C)"], color=col, lw=1.6, label=lab)
    if not h.empty:
        ax.plot(h["time (min)"] + t_off, h["T (C)"], color=col, lw=1.6, ls="--")
ax.axvline(t_off, color="0.4", lw=0.8)
ax.axhline(35, color="0.6", lw=0.8, ls=":")
ax.text(t_off + 0.5, 1.5, "device off", fontsize=8, color="0.3")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Temperature (°C)")
ax.set_ylim(0, 40)
ax.legend(frameon=False, fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "model-time-course.png")

# 9 ---- model vs ex vivo time courses, the three tested rings, absolute temperature, 0-20 min
RINGS = [("Human__aluminum_original", "Original ring", "model v1",
          (ts.study == "Cooling, baseline point") & (ts.variant == "V1")),
         ("Human__aluminum_reduced_bulk", "Reduced-bulk ring", "model v2",
          (ts.study == "Cooling, baseline point") & (ts.variant == "V2")),
         ("Human__ceramic_original", "Ceramic ring", "model v1, aluminum oxide",
          (ts.study == "Material, V1") & (ts.material == "Ceramic"))]
PAIRS = [("SCR", "SCR", "SCR"), ("Anterior Sclera", "Anterior Sclera", "Anterior sclera"),
         ("Equatorial Sclera", "Equatorial Sclera", "Equatorial sclera"),
         ("Vitreous", "Vitreous Humor", "Vitreous"), ("Posterior Sclera", "Posterior Sclera", "Posterior sclera"),
         ("Optic Nerve", "Optic Nerve Head", "Proximal optic nerve")]
fig, axs = plt.subplots(len(PAIRS), 3, figsize=(8.2, 11.5), sharex=True, sharey=True, constrained_layout=True)
for j, (grp, title, mlab, msel) in enumerate(RINGS):
    ev = pd.read_csv(gcurves(grp))
    mod = ts[msel]
    axs[0, j].set_title(title, fontsize=10, fontweight="bold")
    for i, (ech, mch, lab) in enumerate(PAIRS):
        ax = axs[i, j]
        col = SITE_C["Ring"] if mch == "SCR" else SITE_C[mch]
        e = ev[ev.channel == ech].sort_values("time (s)")
        e = e[e["time (s)"] <= 20 * 60]
        t = e["time (s)"].to_numpy() / 60
        ax.fill_between(t, e["mean (C)"] - e["sd (C)"], e["mean (C)"] + e["sd (C)"], color=col, alpha=0.18, lw=0)
        ax.plot(t, e["mean (C)"], color=col, lw=1.5, label=f"ex vivo (n = {int(e['n'].iloc[0])})")
        m = mod[(mod.channel == mch) & (mod["time (min)"] <= 20)].sort_values("time (min)")
        ax.plot(m["time (min)"], m["T (C)"], color="k", lw=1.1, ls="--", label=mlab)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right" if mch == "SCR" else "lower left")
        if j == 0:
            ax.set_ylabel(f"{lab}\n(°C)", color=col, fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel("Time (min)")
axs[0, 0].set_ylim(0, 40)
axs[0, 0].set_xlim(0, 20)
save(fig, "model-vs-exvivo-time.png")

record(__file__, WRITTEN, sorted(set(USED)), origins=ORIGINS,
       notes="Chapter 3 result figures; see script docstring.")
