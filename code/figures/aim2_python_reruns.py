"""Chapter 3 / Appendix D: tables from the corrected-set-up runs of the Python program made on 2026-11-11.

The runs (scr-heat-replica, dev/new_runs.py, corrected set-up: no air, no module layer, workbook properties, start at
rest with the device off) were made on the laboratory PC with

    python dev/new_runs.py --id H1_v1_coarser H1_v1_coarse H1_v2_coarser H1_v2_coarse H1_v3_coarser H1_v3_coarse \
        --out results/corrected_rerun_2026-11-11
    python dev/new_runs.py --id B_v1_Al B_v2_Al B_v3_Al P1_v1_all_x0.5 P1_v1_all_x2 --out results/corrected_rerun_2026-11-11

B_v{1,2,3}_Al reproduce the stored corrected runs of the thesis (tables of 2026-10-13, run_store.npz) to 1e-11 C.
Case definitions are in dev/new_runs.py: H1_<ring>_<coarse|coarser> = baseline on COMSOL's two coarser mesh-study
meshes; P1_v1_all_x<f> = ring v1, perfusion of every tissue times f. No wall-of-the-orbit (OB_*) case is used.

Two steps, so that the tables can be rebuilt without the 1.3 GB of fields:
  python code/figures/aim2_python_reruns.py --store   (needs the run folder; SCR_RESULTS = scr-heat-replica/results)
      copies probes, settings and energy balance of every case into data/python/corrected/rerun_2026-11-11/, and
      extracts from the fields of B_v{1,2,3}_Al the plane of symmetry at 35 min and the retina line at 0, 15, 35 min
      (both on the program's own mesh: no COMSOL file or solution is read)
  python code/figures/aim2_python_reruns.py
      writes, under figures/aim2/sources/data/python/corrected/:
        mesh_steps.csv          final temperature (35 min) at each site on the three meshes, steps (a) and (b)
        table_aim2_umesh_rows.tex   the rows of Table D.2 (tab:aim2-umesh) as printed
        mesh_nodes.csv          vertices of each mesh as solved (no air, no module layer)
        extension_cases.csv     perfusion x0.5, nominal, x2 (ring v1): proximal optic nerve before and at 35 min
        heat_sources.csv        where the heat drawn through the cooling face came from (J over 35 min, %)
        start_state_vs_shinoda.csv   (copied by --store) model before cooling at the sites of Shinoda et al. 2021,
                                written by scr-heat-replica dev/start_state_corrected.py from B_v1_Al
      and records every output in figures/aim2/MANIFEST.json.
"""
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

from provenance import SOURCES, THESIS, record, sha256

DEFAULT = THESIS.parents[2] / "03 - Experiments & Data" / "Analysis Code" / "scr-heat-replica" / "results"
RES = pathlib.Path(os.environ.get("SCR_RESULTS", DEFAULT))
RUN_NAME = "corrected_rerun_2026-11-11"
OUT = SOURCES / "data" / "python" / "corrected"
STORE = OUT / "rerun_2026-11-11"
K = 273.15
RINGS = ("v1", "v2", "v3")
MESH_CASES = {"Coarser": "H1_{v}_coarser", "Coarse": "H1_{v}_coarse", "Normal": "B_{v}_Al"}
EXT_CASES = {0.5: "P1_v1_all_x0.5", 1.0: "B_v1_Al", 2.0: "P1_v1_all_x2"}
CASES = sorted({c.format(v=v) for c in MESH_CASES.values() for v in RINGS} | set(EXT_CASES.values()))
SITES = {"Anterior Sclera": "Anteriror Sclera Probe 1 (K)", "Equatorial Sclera": "Equatorial Sclera Probe 1 (K)",
         "Vitreous Humor": "Vitreous Humor Probe 1 (K)", "Posterior Sclera": "Posterior Sclera Probe 1 (K)",
         "Optic Nerve Head": "Optic Nerve Head Probe 1 (K)", "Optic Nerve": "Optic Nerve Probe 1 (K)",
         "Retina Avg": "Domain Probe Retina Avg 1 (K)", "SCR": "SCR 1 (K)"}


def store():
    runs = RES / RUN_NAME
    used = {}
    for c in CASES:
        dst = STORE / c
        dst.mkdir(parents=True, exist_ok=True)
        st = json.loads((runs / c / "STATUS.json").read_text(encoding="utf-8"))
        assert st.get("state") == "ok", (c, st)
        for n in ("probes_cooling.csv", "settings.json", "energy_balance.json", "run.log"):
            shutil.copy2(runs / c / n, dst / n)
            used[f"{RUN_NAME}/{c}/{n}"] = sha256(runs / c / n)
    # start state of B_v1_Al at the sites of Shinoda et al. 2021 (scr-heat-replica dev/start_state_corrected.py)
    shutil.copy2(runs / "start_state_vs_shinoda.csv", OUT / "start_state_vs_shinoda.csv")
    used[f"{RUN_NAME}/start_state_vs_shinoda.csv"] = sha256(runs / "start_state_vs_shinoda.csv")
    fld = {}
    for v in RINGS:
        c = f"B_{v}_Al"
        m = np.load(runs / c / "mesh.npz", allow_pickle=True)
        f = np.load(runs / c / "fields.npz")
        used[f"{RUN_NAME}/{c}/mesh.npz"] = sha256(runs / c / "mesh.npz")
        used[f"{RUN_NAME}/{c}/fields.npz"] = sha256(runs / c / "fields.npz")
        P = m["points"] * 1e3                                       # mm
        names = dict(zip(m["domain_ids"].tolist(), m["domain_names"].tolist()))
        tc = f["time_cool_s"] / 60.0
        it = {tm: int(np.argmin(np.abs(tc - tm))) for tm in (15, 35)}
        assert all(abs(tc[i] - tm) < 1e-6 for tm, i in it.items())
        Tf = {0: f["T_start"] - K, 15: f["T_cool"][it[15]] - K, 35: f["T_cool"][it[35]] - K}
        # plane of symmetry y = 0: boundary triangles of the half model lying in it, with the domain they bound
        on = np.abs(P[:, 1]) < 1e-6
        sel = np.where(on[m["tris"]].all(1))[0]
        tris_g = m["tris"][sel]
        dom = np.maximum(m["tri_up"][sel], m["tri_down"][sel])
        nodes, tris = np.unique(tris_g, return_inverse=True)
        tris = tris.reshape(-1, 3)
        fld.update({f"{v}_x": P[nodes, 0], f"{v}_z": P[nodes, 2], f"{v}_tris": tris, f"{v}_T": Tf[35][nodes].astype(float),
                    f"{v}_dom": dom, f"{v}_node": nodes})
        # retina line: the edges in the plane shared by a retina triangle and a vitreous triangle (the model's
        # "Retina Edge" selection: the inner contour of the retina, from one anterior edge through the posterior
        # pole to the other), ordered along the line, arc length from the end at negative x
        dn = np.array([names.get(int(d), "") for d in dom])

        def edges(mask):
            t = tris_g[mask]
            e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [0, 2]]]), axis=1)
            return {tuple(x) for x in e.tolist()}
        line = edges(dn == "Retina") & edges(dn == "Vitreous Humor")
        nb = {}
        for a, b in line:
            nb.setdefault(a, []).append(b)
            nb.setdefault(b, []).append(a)
        ends = [n for n, k in nb.items() if len(k) == 1]
        assert len(ends) == 2 and all(len(k) <= 2 for k in nb.values()), "retina line is not one open path"
        start = min(ends, key=lambda n: P[n, 0])
        order, prev = [start], None
        while len(order) < len(nb):
            nxt = [n for n in nb[order[-1]] if n != prev][0]
            prev = order[-1]
            order.append(nxt)
        order = np.array(order)
        xz = P[order][:, [0, 2]]
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xz, axis=0).T))])
        rp = pd.DataFrame(dict(node=order, x_mm=xz[:, 0], z_mm=xz[:, 1], arc_mm=arc))
        for tm in (0, 15, 35):
            rp[f"T_python_{tm}min_C"] = Tf[tm][order].astype(float)
        rp.to_csv(OUT / f"retina_profile_python_{v}.csv", index=False)
        print(v, "plane triangles", len(tris), "retina line nodes", len(order), "arc", round(arc[-1], 2), "mm")
    np.savez_compressed(OUT / "plane_fields_python_35min.npz", **fld)
    (STORE / "inputs_manifest.json").write_text(json.dumps(
        {"script": "code/figures/aim2_python_reruns.py --store", "runs": str(RES / RUN_NAME),
         "made_by": "scr-heat-replica dev/new_runs.py (see the run.log of each case for the code stamp)",
         "inputs_sha256": used}, indent=1), encoding="utf-8")
    print("store written:", len(used), "inputs")


def probes(c):
    d = pd.read_csv(STORE / c / "probes_cooling.csv")
    d.columns = [" ".join(("1" if w.isdigit() else w) for w in x.split(" ")) if " Probe " in x or x.startswith("SCR ") else x
                 for x in d.columns]
    return d


def at35(d, col):
    return float(np.interp(35.0, d.iloc[:, 0].to_numpy(float), d[col].to_numpy(float))) - K


def tables():
    used = [STORE / "inputs_manifest.json"]
    # mesh steps
    rows, nrows = [], []
    for v in RINGS:
        T = {}
        for lvl, pat in MESH_CASES.items():
            c = pat.format(v=v)
            d = probes(c)
            used += [STORE / c / "probes_cooling.csv", STORE / c / "settings.json"]
            T[lvl] = {s: at35(d, col) for s, col in SITES.items()}
            st = json.loads((STORE / c / "settings.json").read_text(encoding="utf-8"))
            nrows.append(dict(variant=v.upper(), mesh=lvl, case=c, vertices=st["nodes"], elements=st["elements"],
                              model_file=st["model_file"]))
        for s in SITES:
            a0, a1, a2 = T["Coarser"][s], T["Coarse"][s], T["Normal"][s]
            rows.append(dict(source="python", setup="corrected", variant=v.upper(), site=s, T_coarser=a0, T_coarse=a1,
                             T_finest=a2, step_a=a1 - a0, step_b=a2 - a1,
                             trend=("shrinks" if abs(a2 - a1) < abs(a1 - a0) else "grows")
                             if np.sign(a2 - a1) == np.sign(a1 - a0) else "opposite signs"))
    pd.DataFrame(rows).to_csv(OUT / "mesh_steps.csv", index=False)
    # rows of Table tab:aim2-umesh (Appendix D), as printed: two decimals, sign always shown
    lab = {"Anterior Sclera": "Anterior sclera", "Equatorial Sclera": "Equatorial sclera", "Posterior Sclera": "Posterior sclera",
           "Vitreous Humor": "Vitreous", "Optic Nerve Head": "Proximal optic nerve", "Optic Nerve": "Distal optic nerve",
           "Retina Avg": "Average retina", "SCR": "SCR"}
    fmt = lambda x: "0.00" if round(x, 2) == 0 else (f"+{x:.2f}" if x > 0 else f"$-${abs(x):.2f}")
    ms = pd.DataFrame(rows)
    lines = ["% Table tab:aim2-umesh rows, written by code/figures/aim2_python_reruns.py from corrected/mesh_steps.csv"]
    for s_, name in lab.items():
        cells = []
        for v in RINGS:
            r = ms[(ms.variant == v.upper()) & (ms.site == s_)].iloc[0]
            cells += [fmt(r.step_a), fmt(r.step_b)]
        lines.append(f"{name:26s} & " + " & ".join(cells) + r" \\")
    (OUT / "table_aim2_umesh_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(nrows).to_csv(OUT / "mesh_nodes.csv", index=False)
    # extension runs: perfusion
    erows = []
    for f, c in EXT_CASES.items():
        d = probes(c)
        used.append(STORE / c / "probes_cooling.csv")
        col = SITES["Optic Nerve Head"]
        erows.append(dict(case=c, ring="v1", perfusion_factor=f, wall="body temperature (fixed)",
                          proximal_optic_nerve_start_C=float(d[col].iloc[0]) - K, proximal_optic_nerve_35min_C=at35(d, col)))
    pd.DataFrame(erows).to_csv(OUT / "extension_cases.csv", index=False)
    # heat sources: grouping of dev/draft_numbers.py split()
    hrows = []
    for v in RINGS:
        c = f"B_{v}_Al"
        e = json.loads((STORE / c / "energy_balance.json").read_text(encoding="utf-8"))
        used.append(STORE / c / "energy_balance.json")
        j = e["J"]
        g = dict(wall=j["fixed: Temperature Body"], blood=j["perfusion"] + j["metabolic"],
                 stored=-j["storage rate"],
                 room=j.get("air convection, ring", 0.0) + j["air convection, tissue"] + j.get("radiation, ring", 0.0)
                 + j["radiation, tissue"] + j.get("evaporation", 0.0))
        face = j["cooling face (out)"]
        hrows.append(dict(case=c, ring=v, cooling_face_J=face,
                          **{f"{k}_J": x for k, x in g.items()}, **{f"{k}_pct": 100 * x / face for k, x in g.items()},
                          closure_J_pct=e["closure_J_pct"]))
    pd.DataFrame(hrows).to_csv(OUT / "heat_sources.csv", index=False)
    outs = [OUT / n for n in ("mesh_steps.csv", "table_aim2_umesh_rows.tex", "mesh_nodes.csv", "extension_cases.csv", "heat_sources.csv",
                              "plane_fields_python_35min.npz", "retina_profile_python_v1.csv",
                              "retina_profile_python_v2.csv", "retina_profile_python_v3.csv", "start_state_vs_shinoda.csv")]
    record(__file__, outs, sorted(set(used)), origins={"rerun_2026-11-11": f"scr-heat-replica/results/{RUN_NAME} (copied by --store)"},
           notes="corrected set-up, Python program, runs of 2026-11-11 (B_v*_Al equal the stored corrected runs to 1e-11 C); "
                 "mesh_steps/mesh_nodes replace data/python/mesh_steps.csv (COMSOL set-up, kept as the comparison with COMSOL) "
                 "for Table D.2 and Figure D.4")
    print(pd.DataFrame(rows).round(3).to_string())
    print(pd.DataFrame(erows).round(3).to_string())
    print(pd.DataFrame(hrows).round(2).to_string())


if __name__ == "__main__":
    if "--store" in sys.argv:
        store()
    tables()
