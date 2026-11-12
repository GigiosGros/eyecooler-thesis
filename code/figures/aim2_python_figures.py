"""Chapter 3 figures of the eye model, drawn from the corrected runs of the Python program.

Same drawing as aim2_model_figures.py, aim2_temperature_field.py and aim2_retina_profile.py
(blocks copied, styling unchanged), with the input tables replaced by those of
aim2_python_corrected.py (figures/aim2/sources/data/python/corrected/). Plotting only.
Since 2026-11-11 the mesh figure, the temperature field and the retina profile come from the corrected-set-up runs
of 2026-11-11 (aim2_python_reruns.py: mesh_steps.csv, mesh_nodes.csv, plane_fields_python_35min.npz,
retina_profile_python_v*.csv), on the program's own mesh: no COMSOL solution, node set or plane is read. They are
drawn at the printed width (6.0 in; retina profile 5.4 in) so that the smallest text prints at 7 pt or more.

Figures: mesh-convergence, material-effectiveness, required-ring-temperature, model-vs-exvivo,
model-time-course, model-vs-exvivo-time, temperature-field, retina-temperature-profile,
energy-split. heat-exchanger-performance is drawn by aim2_python_hx.py. Not here, because it stays
on COMSOL's data: heat-exchanger-mesh-convergence. paired-differences has no model data.

Run:  python code/figures/aim2_python_results.py
      python code/figures/aim2_python_corrected.py
      python code/figures/aim2_python_reruns.py
      python code/figures/aim2_python_figures.py
Every run updates figures/aim2/MANIFEST.json.
"""
import tomllib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from provenance import FIG, SOURCES, record  # noqa: E402

PY = SOURCES / "data" / "python" / "corrected"
EXV = SOURCES / "data" / "exvivo"
USED, WRITTEN = [], []

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
SITE_C = {"Ring": "#1f77b4", "Anterior Sclera": "#9467bd", "Equatorial Sclera": "#ff7f0e",
          "Vitreous Humor": "#2ca02c", "Posterior Sclera": "#8c564b",
          "Optic Nerve Head": "#d62728", "Optic Nerve": "#7f1d1d", "Retina Avg": "#555555"}
NAME = {"Anterior Sclera": "Anterior sclera", "Equatorial Sclera": "Equatorial sclera",
        "Vitreous Humor": "Vitreous", "Posterior Sclera": "Posterior sclera",
        "Optic Nerve Head": "Proximal optic nerve", "Optic Nerve": "Distal optic nerve",
        "Retina Avg": "Average retina", "SCR": "SCR"}
ORDER = ["Anterior Sclera", "Equatorial Sclera", "Vitreous Humor", "Posterior Sclera",
         "Optic Nerve Head", "Optic Nerve", "Retina Avg"]
DES_C = {"V1": "#2b4c7e", "V2": "#d9822b", "V3": "#5a9e6f"}
DES_M = {"V1": ("o", "-"), "V2": ("s", "--"), "V3": ("^", ":")}


def tab(name):
    p = PY / name
    USED.append(p)
    return pd.read_csv(p)


def save(fig, name):
    dst = FIG / name
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    WRITTEN.append(dst)
    print("wrote", name)


# 1 ---- mesh convergence: the program in the corrected set-up on COMSOL's three meshes (runs of 2026-11-11;
# the COMSOL-set-up runs, data/python/mesh_steps.csv, are no longer drawn)
USED.append(PY / "mesh_steps.csv")
g = pd.read_csv(PY / "mesh_steps.csv")
g = g[g.source == "python"]
# vertex counts of the meshes as the program solves them (air and module layer removed)
nodp = PY / "mesh_nodes.csv"
USED.append(nodp)
nodes = {}
for _, r in pd.read_csv(nodp).iterrows():
    nodes.setdefault(r.variant, {})[r.mesh] = r.vertices
sites = ORDER + ["SCR"]
fig, axs = plt.subplots(2, 4, figsize=(6.0, 3.5), constrained_layout=True)
Y_HALF_SPAN = 0.5
for ax, site in zip(axs.flat, sites):
    seen = []
    for v in ["V1", "V2", "V3"]:
        r = g[(g.variant == v) & (g.site == site)].iloc[0]
        y = [r.T_coarser, r.T_coarse, r.T_finest]
        x = [nodes[v][m] for m in ("Coarser", "Coarse", "Normal")]
        mk, ls = DES_M[v]
        ax.plot(x, y, marker=mk, ls=ls, color=DES_C[v], ms=3, lw=1.0, label=v.lower())
        ax.errorbar(x[-1], y[-1], yerr=abs(y[-1] - y[-2]), color=DES_C[v], capsize=2, lw=0.9, fmt="none")
        seen += list(y) + [y[-1] - abs(y[-1] - y[-2]), y[-1] + abs(y[-1] - y[-2])]
    ax.set_title(NAME.get(site, site), color=SITE_C.get(site, SITE_C["Ring"]), fontsize=7.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1.2e4, 2.5e5)
    ax.set_xticks([2e4, 5e4, 2e5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, pos: f"{val / 1000:.0f}k"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="both", labelsize=7.5)
    mid = 0.5 * (min(seen) + max(seen))
    assert max(seen) - min(seen) <= 2 * Y_HALF_SPAN, (site, min(seen), max(seen))
    ax.set_ylim(mid - Y_HALF_SPAN, mid + Y_HALF_SPAN)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.ticklabel_format(axis="y", useOffset=False)
for ax in axs[:, 0]:
    ax.set_ylabel("Final temperature (°C)", fontsize=7.5)
fig.supxlabel("Mesh vertices", fontsize=7.5)   # 2026-11-12: ticks already read 20k, 50k, 200k
axs[0, 0].legend(fontsize=7.5, frameon=False, handlelength=2.2)
save(fig, "mesh-convergence.png")

# 3 ---- material
e = tab("run_epsilon.csv")
e = e[e.source == "python"]
mat = e[(e.study == "Material") & (e.variant == "V1")]
MC = {"Aluminum": "#264653", "Ceramic": "#e9c46a", "Glass": "#8ecae6"}
ML = {"Aluminum": "Aluminum", "Ceramic": "Aluminum oxide", "Glass": "Fused silica"}
geo = e[e.study == "Parametric baseline"]
fig, axs = plt.subplots(2, 1, figsize=(8, 6.4), constrained_layout=True, sharey=True)
x = np.arange(len(ORDER)); w = 0.26
ax = axs[0]
for k, m in enumerate(["Aluminum", "Ceramic", "Glass"]):
    vals = [100 * mat[(mat.material == m) & (mat.site == s)].epsilon.iloc[0] for s in ORDER]
    ax.bar(x + (k - 1) * w, vals, w, color=MC[m], edgecolor="k", lw=0.5, label=ML[m])
ax.set_title("(A) Ring material (v1)", fontsize=10, loc="left")
ax = axs[1]
for k, v in enumerate(["V1", "V2", "V3"]):
    vals = [100 * geo[(geo.variant == v) & (geo.site == s)].epsilon.iloc[0] for s in ORDER]
    ax.bar(x + (k - 1) * w, vals, w, color=DES_C[v], edgecolor="k", lw=0.5, label=v.lower())
ax.set_title("(B) Ring geometry (aluminum)", fontsize=10, loc="left")
for ax in axs:
    ax.set_xticks(x, [NAME[s].replace(" sclera", "\nsclera").replace(" optic", "\noptic")
                      .replace("Average ", "Average\n") for s in ORDER], fontsize=8.5)
    ax.set_ylabel("Cooling effectiveness ε (%)")
    ax.legend(frameon=False, fontsize=9)
save(fig, "material-effectiveness.png")

# 5 ---- required ring temperature
q = tab("required_ring_temperature.csv")
q = q[(q.source == "python") & (q.variant == "V1")]
# one small panel per site, both targets in each, same axes so the panels compare directly
SITES4 = ["Vitreous Humor", "Retina Avg", "Posterior Sclera", "Optic Nerve Head"]
fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0), constrained_layout=True, sharex=True, sharey=True)
TARGET_STYLE = {35.0: ("-", True), 32.0: ("--", False)}
for k, (ax, s) in enumerate(zip(axs.flat, SITES4)):
    ax.axhspan(-40, 4, color="#f2d7d5", zorder=0)
    for tgt, (ls, filled) in TARGET_STYLE.items():
        d = q[(q.site == s) & (q["target (C)"] == tgt)].sort_values("Tb (C)")
        ax.plot(d["Tb (C)"], d["required_ring_T (C)"], marker="o", ls=ls, color=SITE_C[s],
                mfc=SITE_C[s] if filled else "white")
    ax.set_title(f"({'ABCD'[k]}) {NAME[s]}", fontsize=9.5, loc="left", pad=8)
    ax.set_xticks([35, 37, 39])
    ax.set_ylim(-40, 40)                 # headroom: the 35 C points sit at the top
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.grid(axis="y", color="0.9", lw=0.6)
axs[1, 0].text(34.9, -38, "below the 4 °C floor\nof the device", fontsize=7, color="#922b21", va="bottom")
for ax in axs[1]:
    ax.set_xlabel("Body temperature (°C)")
for ax in axs[:, 0]:
    ax.set_ylabel("Ring temperature\nneeded (°C)")
ht = [plt.Line2D([], [], color="#444", ls="-", marker="o", label="site at 35 °C"),
      plt.Line2D([], [], color="#444", ls="--", marker="o", mfc="white", label="site at 32 °C")]
fig.legend(handles=ht, frameon=False, fontsize=8.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.06))
save(fig, "required-ring-temperature.png")

# 6 ---- model vs ex vivo
mv = tab("model_vs_exvivo.csv")
GM = {"Human Fresh Aluminum Original": ("o", "Original ring / v1"),
      "Human Fresh Aluminum Reduced Bulk": ("s", "Reduced-bulk ring / v2"),
      "Human Fresh Ceramic Original": ("^", "Ceramic ring / v1, aluminum oxide")}
fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
ax.plot([0, 100], [0, 100], color="k", lw=0.8, ls="--")
for _, r in mv.iterrows():
    mk, _ = GM[r.group]
    col = SITE_C[r.site]
    xm, ym = 100 * r["eps_start model"], 100 * r["eps_start exp"]
    lo, hi = 100 * r["eps_start exp 95%CI low"], 100 * r["eps_start exp 95%CI high"]
    ax.errorbar(xm, ym, yerr=[[ym - lo], [hi - ym]], fmt=mk, color=col, mfc=col, ms=6, capsize=2, lw=1)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xlabel("Model effectiveness ε$_{Δ}$ (%)")
ax.set_ylabel("Ex-vivo effectiveness ε$_{Δ}$ (%, mean, 95% CI)")
h1 = [plt.Line2D([], [], marker=GM[k][0], color="#444", ls="", label=GM[k][1]) for k in GM]
h2 = [plt.Line2D([], [], marker="o", color=SITE_C[s], ls="", label=NAME[s]) for s in
      ["Anterior Sclera", "Equatorial Sclera", "Vitreous Humor", "Posterior Sclera", "Optic Nerve Head"]]
leg = ax.legend(handles=h1, frameon=False, fontsize=7.5, loc="upper left")
ax.add_artist(leg)
ax.legend(handles=h2, frameon=False, fontsize=7.5, loc="lower right")
save(fig, "model-vs-exvivo.png")

# 8 ---- time courses
ts = tab("timeseries_thesis.csv")
cool = ts[(ts.study == "Cooling, baseline point") & (ts.variant == "V1")]
heat = ts[(ts.study == "Re-heat, V1") & ((ts["Tc (C)"] - 4).abs() < 0.5)]
t_off = cool["time (min)"].max()
fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
for site in ["SCR"] + ORDER:
    c = cool[cool.channel == site].sort_values("time (min)")
    h = heat[heat.channel == site].sort_values("time (min)")
    col = SITE_C.get(site, SITE_C["Ring"])
    ax.plot(c["time (min)"], c["T (C)"], color=col, lw=1.6, label=NAME.get(site, site))
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

# 9 ---- model vs ex vivo time courses
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
    p = EXV / (grp + "__mean_curves.csv")
    USED.append(p)
    ev = pd.read_csv(p)
    mod = ts[msel]
    axs[0, j].set_title(title, fontsize=10, fontweight="bold")
    for i, (ech, mch, lab) in enumerate(PAIRS):
        ax = axs[i, j]
        col = SITE_C["Ring"] if mch == "SCR" else SITE_C[mch]
        ee = ev[ev.channel == ech].sort_values("time (s)")
        ee = ee[ee["time (s)"] <= 20 * 60]
        t = ee["time (s)"].to_numpy() / 60
        ax.fill_between(t, ee["mean (C)"] - ee["sd (C)"], ee["mean (C)"] + ee["sd (C)"], color=col, alpha=0.18, lw=0)
        ax.plot(t, ee["mean (C)"], color=col, lw=1.5, label=f"ex vivo (n = {int(ee['n'].iloc[0])})")
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

# temperature field: plane of symmetry of the program's own mesh, corrected set-up (no air, no module layer), 35 min
d = np.load(PY / "plane_fields_python_35min.npz")
USED.append(PY / "plane_fields_python_35min.npz")
T_LO, T_HI, STEP = 4.0, 37.0, 2.0
fig, axs = plt.subplots(1, 3, figsize=(6.0, 4.3), sharey=True, constrained_layout=True)
levels = np.arange(T_LO + STEP, T_HI, STEP)
for ax, letter, v in zip(axs, "ABC", ("v1", "v2", "v3")):
    x, z, tris, T, dom = (d[f"{v}_{k}"] for k in ("x", "z", "tris", "T", "dom"))
    tri = mtri.Triangulation(x, z, tris)
    im = ax.tripcolor(tri, T, shading="gouraud", cmap="viridis", vmin=T_LO, vmax=T_HI, rasterized=True)
    ax.tricontour(tri, T, levels=levels, colors="white", linewidths=0.35, alpha=0.75)
    _, pid = np.unique(np.round(np.stack([x, z], 1), 6), axis=0, return_inverse=True)
    pid = pid.ravel()
    first = np.zeros(pid.max() + 1, int)
    first[pid] = np.arange(len(pid))
    pt = pid[tris]
    ed_ = np.sort(np.concatenate([pt[:, [0, 1]], pt[:, [1, 2]], pt[:, [0, 2]]]), axis=1)
    ed = np.tile(dom, 3)
    o = np.lexsort((ed_[:, 1], ed_[:, 0]))
    ed_, ed = ed_[o], ed[o]
    inner = np.where((ed_[1:] == ed_[:-1]).all(1) & (ed[1:] != ed[:-1]))[0]
    i0, i1 = first[ed_[inner, 0]], first[ed_[inner, 1]]
    seg = np.stack([np.stack([x[i0], z[i0]], 1), np.stack([x[i1], z[i1]], 1)], 1)
    ax.add_collection(LineCollection(seg, colors="black", linewidths=0.45, alpha=0.8))
    ax.set_aspect("equal")
    ax.set_title(f"({letter}) {v}", loc="left", fontsize=8.5)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.tick_params(labelsize=7.5)
axs[0].set_ylabel("z (mm)", fontsize=8)
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02, ticks=np.arange(5, 38, 5))
cb.set_label("Temperature (°C)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
save(fig, "temperature-field.png")


# retina profile
def load(v):
    USED.append(PY / f"retina_profile_python_{v}.csv")
    r = pd.read_csv(PY / f"retina_profile_python_{v}.csv")
    r["s"] = r.arc_mm - r.arc_mm[int(r.z_mm.values.argmin())]
    return r


R = {v: load(v) for v in ("v1", "v2", "v3")}
v1 = R["v1"]
TIMES = {0: ("Before cooling", "#8a8a8a", "-"), 35: ("35 min", "#08519c", "-"), 15: ("15 min", "#6baed6", (0, (4, 3)))}
RC = {"v2": ("#D55E00", "-"), "v3": ("#009E73", (0, (5, 2)))}
fig, (a, b) = plt.subplots(2, 1, figsize=(5.4, 4.9), sharex=True, constrained_layout=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1]))
for t, (label, color, ls) in TIMES.items():
    a.plot(v1.s, v1[f"T_python_{t}min_C"], color=color, ls=ls, lw=2.0, label=label)
a.axhline(35, color="0.35", lw=0.8, ls=":")
a.text(v1.s.min(), 35.25, "35 °C", fontsize=8, color="0.35", va="bottom")
a.set_ylabel("Temperature (°C)", fontsize=9)
a.set_ylim(5, 38)
a.set_title("(A) Original ring (v1)", loc="left", fontsize=10)
h, l = a.get_legend_handles_labels()
o = [l.index(TIMES[t][0]) for t in sorted(TIMES)]
a.legend([h[i] for i in o], [l[i] for i in o], frameon=False, fontsize=9, loc="lower center")
a.grid(axis="y", color="0.9", lw=0.6)
a.annotate("anterior edge\nof the retina", (v1.s[0], v1.T_python_35min_C[0]), xytext=(v1.s[0] + 3.2, 9.5),
           fontsize=8, color="0.3", va="center", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
pole = int(np.argmin(np.abs(v1.s.values)))
a.annotate("posterior pole", (0, v1.T_python_35min_C[pole]), xytext=(0, 29.5), ha="center",
           fontsize=8, color="0.3", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
notes = {}
for v, (color, ls) in RC.items():
    diff = np.interp(v1.s, R[v].s, R[v].T_python_35min_C) - v1.T_python_35min_C
    b.plot(v1.s, diff, color=color, ls=ls, lw=1.8, label=f"{v} − v1")
    notes[v] = "%+.2f to %+.2f, pole %+.2f" % (diff.min(), diff.max(), diff[pole])
b.axhline(0, color="0.35", lw=0.8)
b.set_ylabel("Difference from v1\nafter 35 min (°C)", fontsize=9)
b.set_xlabel("Distance along the retina from the posterior pole (mm)", fontsize=9)
b.set_title("(B) Other rings against v1", loc="left", fontsize=10)
b.legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
b.grid(axis="y", color="0.9", lw=0.6)
save(fig, "retina-temperature-profile.png")
for t in TIMES:
    T = v1[f"T_python_{t}min_C"]
    print(t, "min: ends %.2f %.2f, max %.2f, pole %.2f" % (T.iloc[0], T.iloc[-1], T.max(), T[pole]))
print(notes)

# energy split (whole device, J over 35 min)
en = tab("energy_summary.csv")
IC = {"Eye": "#c0392b", "Eyelids": "#e6b0aa", "Air": "#d5dbdb"}
COL = {"Eye": "eye_J", "Eyelids": "eyelids_J", "Air": "air_J"}
fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
labels, pos = [], []
p = 0
for v in ["V1", "V2", "V3"]:
    for sil in ["Baseline", "0.5 mm", "1.0 mm"]:
        r = en[(en.variant == v) & (en.silicone == sil)].iloc[0]
        bottom = 0
        for itf in COL:
            ax.bar(p, r[COL[itf]], 0.8, bottom=bottom, color=IC[itf], edgecolor="k", lw=0.5)
            bottom += r[COL[itf]]
        # the eye share sits inside the eye segment, the part it describes
        ax.text(p, r.eye_J / 2, f"{r.eye_share_pct:.0f}%", ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")
        labels.append({"Baseline": "none", "0.5 mm": "0.5", "1.0 mm": "1.0"}[sil])
        pos.append(p)
        p += 1
    p += 0.6
ax.set_xticks(pos, labels, fontsize=8)
for i, v in enumerate(["v1", "v2", "v3"]):
    ax.text(np.mean(pos[3*i:3*i+3]), -1120, v, ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Heat drawn over 35 min, whole device (J)")
ax.set_xlabel("Insulating layer (mm), by ring", fontsize=9, labelpad=14)
ax.legend(handles=[Patch(fc=IC[k], ec="k", label=k) for k in IC], frameon=False, fontsize=8.5, ncol=3, loc="upper right")
ax.set_ylim(0, 9200)
save(fig, "energy-split.png")

record(__file__, WRITTEN, sorted(set(USED)),
       origins={"python": "aim2_python_corrected.py from the tables of 2026-10-13 (original/new_runs); see sources/data/python/corrected/inputs_manifest.json"},
       notes="Eye-model figures from the corrected runs of the Python program, tag scr-heat-replica-v1.1. Since 2026-11-11 "
             "mesh-convergence, temperature-field and retina-temperature-profile come from the corrected-set-up runs of "
             "2026-11-11 (aim2_python_reruns.py: corrected/mesh_steps.csv, mesh_nodes.csv, plane_fields_python_35min.npz, "
             "retina_profile_python_v*.csv; program's own mesh, no COMSOL solution or node set read) and are drawn at the "
             "printed width (smallest text >= 7 pt). Superseded, kept unused: data/python/mesh_steps.csv and "
             "mesh_nodes_without_air.csv (COMSOL set-up), corrected/plane_fields_35min.npz and corrected/retina_profile_v*.csv "
             "(COMSOL plane and nodes; equal to the new ones within 2e-6 C); old PNGs in figures/aim2/_superseded_2026-11-11/.")
