"""Heat-exchanger results of Chapter 3 from the Python water-block runs, same heat on the three designs.

Sweep W13 of scr-heat-replica (results/waterblock_research_2026-10-27): heat on the module footprint,
annulus 7-13 mm, module plane cooled by the air, temperature marched with the flow where it is unsteady.
Each case gives R and c_air of  T_face - T_inlet = R P + c_air (T_ambient - T_inlet).  The 36 operating
points are the grid of the COMSOL sweep (flow, inlet, ambient 10 and 35 C), with the same 10 and 15.5 W on
all three designs in place of the two fluxes.  W3 (same set-up, mean flow) gives what W13 does not store:
hottest aluminium, peak of the heated face, heat balance.

Run:  python code/figures/aim2_python_hx.py [--store]    (--store re-copies the tables from the replica)
Writes tables to figures/aim2/sources/data/python/hx/ and heat-exchanger-performance.png.
Do not run aim2_model_figures.py afterwards: it would overwrite the figure with the COMSOL one.
"""
import os
import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from provenance import FIG, SOURCES, record  # noqa: E402

HX = SOURCES / "data" / "python" / "hx"
RES = Path(os.environ.get("HX_SWEEP_RESULTS", FIG.parents[1] / "code" / "ch3-heat-replica" / "results"
                           / "waterblock_research_2026-10-27"))
FILES = {"sweep_W13.csv": "tables/sweep_W13.csv", "cases_W9_W15.csv": "tables/cases.csv",
         "w13_equal_pumping.csv": "summary/w13_equal_pumping.csv", "corrected_W3.csv": "w1_w8/corrected_W3.csv",
         "cases_W1_W8.csv": "w1_w8/cases.csv"}
if "--store" in sys.argv:
    HX.mkdir(parents=True, exist_ok=True)
    for dst, src in FILES.items():
        shutil.copy2(RES / src, HX / dst)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
DES_C = {"v1": "#2b4c7e", "v2": "#d9822b", "v3": "#5a9e6f"}
DES_M = {"v1": "o", "v2": "s", "v3": "^"}
FLOWS, INLETS, AMB, LOADS = (0.10, 0.25, 0.50), (4.0, 10.0, 20.0), (10.0, 35.0), (10.0, 15.5)

w = pd.read_csv(HX / "sweep_W13.csv")
w = w[(w.coolant == "water") & w.flow_L_min.isin(FLOWS) & w.inlet_C.isin(INLETS)]
assert len(w) == 27 and w.valid.all()
rows = []
for r in w.itertuples():
    for a in AMB:
        for p in LOADS:
            rise = r.R_C_per_W * p + r.c_air * (a - r.inlet_C)
            rows.append(dict(design=r.design, flow_L_min=r.flow_L_min, inlet_C=r.inlet_C, ambient_C=a, power_W=p,
                             state=r.state, rise_from=r.rise_from, R_C_per_W=r.R_C_per_W, rise_C=rise,
                             face_C=r.inlet_C + rise, pressure_drop_kPa=r.pressure_drop_kPa))
d = pd.DataFrame(rows)
d.to_csv(HX / "hx_36points.csv", index=False)

out = []
for g, s in d.groupby("design"):
    y, tot = s.face_C, ((s.face_C - s.face_C.mean()) ** 2).sum()
    share = {k: 100 * sum(len(q) * (q.face_C.mean() - y.mean()) ** 2 for _, q in s.groupby(k)) / tot
             for k in ("inlet_C", "flow_L_min", "power_W", "ambient_C")}
    out.append(dict(design=g, rise_min=s.rise_C.min(), rise_max=s.rise_C.max(), rise_10W_min=s[s.power_W == 10].rise_C.min(),
                    rise_10W_max=s[s.power_W == 10].rise_C.max(), face_max=s.face_C.max(), margin_43=43 - s.face_C.max(),
                    **{"R_%.2f" % q: s[s.flow_L_min == q].R_C_per_W.mean() for q in FLOWS},
                    **{"Rmin_%.2f" % q: s[s.flow_L_min == q].R_C_per_W.min() for q in FLOWS},
                    **{"Rmax_%.2f" % q: s[s.flow_L_min == q].R_C_per_W.max() for q in FLOWS},
                    **{"dp_min_%.2f" % q: s[s.flow_L_min == q].pressure_drop_kPa.min() for q in FLOWS},
                    **{"dp_max_%.2f" % q: s[s.flow_L_min == q].pressure_drop_kPa.max() for q in FLOWS},
                    **{"share_" + k: v for k, v in share.items()}))
sm = pd.DataFrame(out)
sm.to_csv(HX / "hx_summary.csv", index=False)
print(sm.T.round(3).to_string())
print(w.groupby(["design", "state"]).size())
print("ratio to v1, same flow and inlet")
p = w.pivot_table(index=["flow_L_min", "inlet_C"], columns="design", values="R_C_per_W")
print((p.div(p.v1, axis=0)).agg(["min", "max"]).round(2))
q = w.pivot_table(index=["flow_L_min", "inlet_C"], columns="design", values="pressure_drop_kPa")
print((q.div(q.v1, axis=0)).agg(["min", "max"]).round(1))
print("superposition", w[["superposition_check_22C", "superposition_check_35C"]].abs().max().to_dict())
c = pd.read_csv(HX / "cases_W9_W15.csv")
c = c[c.case.isin(w.case)]
print("fluctuation", c.groupby("design").fluctuation_of_pressure_drop.agg(["min", "max"]))

t = pd.read_csv(HX / "corrected_W3.csv")
print(t.groupby("design")[["hottest_aluminium_C"]].max())
t15 = t[t.power_W == 15.5]
print("peak of the face above inlet, 15.5 W, by flow\n", t15.groupby(["design", "flow_L_min"]).heated_face_max_above_inlet.max().unstack())
w3 = pd.read_csv(HX / "cases_W1_W8.csv")
w3 = w3[w3.block == "W3"]
print("W3 heat balance, percent", (100 * w3.heat_balance_error_W.abs() / w3.heat_in_W).agg(["mean", "max"]).to_dict())
print(pd.read_csv(HX / "w13_equal_pumping.csv").query("coolant=='water' and inlet_C==4"))

# the 43 C limit is checked against the peak of the whole heat exchanger: the larger of the peak of the heated
# face and the hottest aluminium (in v3 the face is partly coolant, so the two differ); only W3 stored them
pk = t[t.power_W.isin(LOADS) & t.ambient_C.isin(AMB) & t.flow_L_min.isin(FLOWS) & t.inlet_C.isin(INLETS)].copy()
pk["face_peak_C"] = pk.inlet_C + pk.heated_face_max_above_inlet
pk["peak_C"] = pk[["face_peak_C", "hottest_aluminium_C"]].max(axis=1)
print("points where the hottest aluminium exceeds the face peak:", int((pk.hottest_aluminium_C > pk.face_peak_C + 1e-3).sum()))
pk.to_csv(HX / "hx_36points_peak.csv", index=False)
print(pk.groupby("design").peak_C.max())

fig = plt.figure(figsize=(7.5, 3.4), constrained_layout=True)
sub = fig.add_gridspec(1, 3, wspace=0.05)
for k, v in enumerate(("v1", "v2", "v3")):
    ax = fig.add_subplot(sub[k], sharey=None if k == 0 else fig.axes[-1])
    s = pk[pk.design == v]
    ax.scatter(s.inlet_C, s.peak_C, s=14, marker=DES_M[v], color=DES_C[v], alpha=0.7)
    ax.axhline(43, color="#922b21", lw=1.2)
    ax.set_ylim(0, 46)
    ax.set_xticks(sorted(s.inlet_C.unique()))
    ax.set_title(v, fontsize=9)
    ax.set_xlabel("Coolant inlet temperature (°C)" if k == 1 else "")
    if k == 0:
        ax.set_ylabel("Peak temperature of the\nheat exchanger (°C)")
    else:
        ax.tick_params(labelleft=False)
    if k == 2:
        ax.text(0.97, 0.90, "43 °C limit", transform=ax.transAxes, ha="right", color="#922b21", fontsize=8)
dst = FIG / "heat-exchanger-performance.png"
fig.savefig(dst, bbox_inches="tight", facecolor="white")
plt.close(fig)
record(__file__, [dst], sorted(HX / f for f in FILES) + [HX / "hx_36points.csv", HX / "hx_summary.csv"],
       origins={"python": "scr-heat-replica results/waterblock_research_2026-10-27, code 735b4f3, results ef68008"},
       notes="Same heat on the three heat exchangers: sweep W13 (water) with the superposition of each case; W3 for hottest aluminium and heat balance.")
