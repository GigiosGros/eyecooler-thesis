"""Temperatures of the CORRECTED model before cooling at the sites where Shinoda et al. 2021 measured (thesis Ch. 3
Discussion), read from the start state of a corrected run of dev/new_runs.py (nothing is solved).

    python -X utf8 dev/start_state_corrected.py [run folder of B_v1_Al]
    default: results/corrected_rerun_2026-11-11/B_v1_Al

The start state is fields.npz["T_start"] of the run (stationary, device in place and switched off, no air domain,
room 25 C, body 37 C): the start of every corrected run of the thesis. Sites as dev/start_state_vs_literature.places
(on the axis through the centres of the cornea and the lens; before the disc = 1 mm from the proximal-optic-nerve
probe towards the centre of the vitreous); the probe position is read from the COMSOL model file, as the run does.
Writes start_state_vs_shinoda.csv beside the run folder.  Degrees C.
"""
import csv
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICA = os.path.dirname(HERE)
sys.path.insert(0, REPLICA)
sys.path.insert(0, HERE)
K0 = 273.15
SHINODA = {"anterior chamber": (30.0, 1.7), "behind the lens": (32.8, 1.2), "mid-vitreous": (34.0, 0.9),
           "before the optic disc": (35.0, 0.9)}


def main(run):
    import run_all
    from replica import fem, model
    from replica.mphread import Mph
    from start_state_vs_literature import places
    st = json.load(open(os.path.join(run, "settings.json"), encoding="utf-8"))
    ring = st["settings"]["ring"]
    mph = Mph(run_all.mph_path(r"Ring Comparison\Ring Comparison.mph"))
    comp = next(c for c, v in mph.components().items() if v["label"].endswith(ring))
    info = mph.components()[comp]
    base = model.build(mph, component=comp, heat=info["heat"], geom=info["geom"], insulation_slit=False)
    m = np.load(os.path.join(run, "mesh.npz"), allow_pickle=True)
    T = np.load(os.path.join(run, "fields.npz"))["T_start"].astype(float)
    prob = types.SimpleNamespace(points=m["points"], tets=m["tets"], dom=m["dom"],
                                 domain_names=dict(zip(m["domain_ids"].tolist(), m["domain_names"].tolist())),
                                 probes=base.probes)
    pts, sets, geo = places(prob)
    rows = []
    for k, x in pts.items():
        e, lam, w = fem.locate(prob.points, prob.tets, np.asarray(x)[None])[0]
        rows.append(dict(site=k, T_C=float(T[prob.tets[e]] @ lam - K0), tissue=prob.domain_names[int(prob.dom[e])]))
    for k, n in sets.items():
        rows.append(dict(site=k + ", mean of nodes", T_C=float(T[n].mean() - K0), tissue=k))
    for r in rows:
        key = {"anterior chamber, on the axis": "anterior chamber", "end of the vitreous on the axis, 1 mm before the retina": None}.get(r["site"], r["site"])
        if key in SHINODA:
            r["shinoda_mean_C"], r["shinoda_sd_C"] = SHINODA[key]
    out = os.path.join(os.path.dirname(run), "start_state_vs_shinoda.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["site", "T_C", "tissue", "shinoda_mean_C", "shinoda_sd_C"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['site']:58s} {r['T_C']:7.3f}  {r['tissue']}")
    print("geometry", geo, "->", out)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0] if a else os.path.join(REPLICA, "results", "corrected_rerun_2026-11-11", "B_v1_Al"))
