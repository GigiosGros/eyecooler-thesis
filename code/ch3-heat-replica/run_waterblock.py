"""Water-block test: re-solve the COMSOL water-block models (laminar, stationary) with the replica.

    python run_waterblock.py                       # v1-v3, coarse and normal meshes, default parameters
    python run_waterblock.py --versions 1 --meshes "Mesh Coarse"

The model is conjugate heat transfer: an aluminium block heated on the TEC
face (Qh), cooled by water forced through its channel (flow rate Q0, inlet
temperature Tin), in a room at T0.  As in the COMSOL study, the load is
ramped with the parameter alpha (0 -> 1), each solution starting the next.
Results are compared with the "Auxiliary Stationary Solution Laminar" tables.

Limits: laminar flow only (the turbulence model in the files is not
reproduced), and the flow stabilisation differs from COMSOL's, which matters
here because the flow is fast compared with the element size.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

import run_all
from replica import fem, model, solve
from replica.mphread import Mph

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = r"D:\COMSOL\Projects\Waterblock\Auxiliary Step"
BOX_PROJ = os.path.join(run_all.BOX, "Projects", "Waterblock", "Auxiliary Step")
REF = os.path.join(run_all.BOX, "Reports", "Waterblock", "Auxiliary Stationary Solution Laminar")
REF_NAME = {"Mesh Coarse": "Coarse", "Mesh Normal": "Normal"}


def boundary_probes(mph, prob):
    """{probe name: (triangles, areas)} for the Boundary probes of the model (average temperature)."""
    out = {}
    for p in mph.root.iter("ProbeFeature"):
        if p.get("op") == "Boundary":
            _, ents, _ = mph.selection_of(p)
            t = prob.tris[np.isin(prob.tri_bnd, ents)]
            out[p.get("name")] = (t, fem.tri_area(prob.points, t))
    return out


def heat_carried(prob, h, T):
    """Net heat leaving with the coolant [W]: rho*cp * integral of T (u.n) over inlets and outlets."""
    from replica.flow import outward_normals
    total = 0.0
    for r in h.regions:
        rc = prob.rho_cp[r.spec["domains"][0]]
        surf = (np.isin(prob.tri_up, r.spec["domains"]).astype(int) + np.isin(prob.tri_down, r.spec["domains"]).astype(int)) == 1
        open_b = list(r.spec["outlets"]) + [b for bl, _ in r.spec["inlets"] for b in bl]
        ft = prob.tris[surf & np.isin(prob.tri_bnd, open_b)]
        nn = outward_normals(prob.points, prob.tets[r.el], ft)
        loc = {int(a): i for i, a in enumerate(r.nodes)}
        for t, n_ in zip(ft, nn):
            v = [loc[int(a)] for a in t]
            Tm = T[t]
            c = rc(Tm.mean()) if callable(rc) else rc
            if hasattr(r, "edge_of"):             # quadratic velocity: vertex and mid-side values
                m = [r.edge_of[tuple(sorted((v[i], v[j])))] for i, j in ((0, 1), (1, 2), (0, 2))]
                total += c * _tri_p1p2(Tm, r.u[v] @ n_, r.u[m] @ n_)
            else:
                un = r.u[v] @ n_
                total += c * (Tm.sum() * un.sum() + (Tm * un).sum()) / 12.0
    return float(total)


def _tri_p1p2(T, uv, um):
    """Integral over a triangle (area folded into u) of a linear T times a quadratic u (vertex, mid-side values)."""
    a, b, w1, w2 = 0.445948490915965, 0.091576213509771, 0.223381589678011, 0.109951743655322     # exact to degree 4
    pts = [(a, a, 1 - 2 * a), (a, 1 - 2 * a, a), (1 - 2 * a, a, a), (b, b, 1 - 2 * b), (b, 1 - 2 * b, b), (1 - 2 * b, b, b)]
    s = 0.0
    for l, w in zip(pts, [w1] * 3 + [w2] * 3):
        u = sum(l[i] * (2 * l[i] - 1) * uv[i] for i in range(3)) + 4 * (l[0] * l[1] * um[0] + l[1] * l[2] * um[1] + l[0] * l[2] * um[2])
        s += w * u * (l[0] * T[0] + l[1] * T[1] + l[2] * T[2])
    return s


def solve_case(mph, mesh_tag, alphas, log, element="p2"):
    info = next(iter(mph.components().values()))
    comp = next(iter(mph.components()))
    prev, out = None, {}
    for a in alphas:
        prob = model.build(mph, component=comp, heat=info["heat"], geom=info["geom"], mesh_tag=mesh_tag,
                           overrides=dict(alpha=a))
        h = solve.Heat(prob, log=lambda s: None, flow=True, supg_heat=True, flow_element=element)
        if prev is not None:                       # start from the previous load level
            for r, r0 in zip(h.regions, prev):
                r.u[~r.fixed_u] = r0.u[~r0.fixed_u]
                r.p[:] = r0.p
        step = next(k for k, v in mph.study_steps().items() if v["op"] == "Stationary")
        t0, n_it = time.time(), 0
        for r in h.regions:                        # the flow does not depend on temperature much: converge it first
            n_it = r.solve(prob.t_init, r.spec["t_ref"], prob.flow["gravity"], maxit=80, tol=1e-5)
            if not getattr(r, "converged", True):          # (2026-10-28) no longer silent
                print(f"  WARNING alpha {a}: flow pre-solve stopped at maxit 80 "
                      f"(relative change {r.last_change / max(r.last_scale, 1e-300):.2e})")
        T = h.stationary(step=step, flow_passes=2, tol=1e-6)
        probes = boundary_probes(mph, prob)
        out[a] = {n: float((T[t].mean(1) * ar).sum() / ar.sum()) for n, (t, ar) in probes.items()}
        umax = max((np.linalg.norm(r.u, axis=1).max() for r in h.regions), default=0.0)
        # heat balance: heat put in on the TEC face against the heat carried out by the water
        q_in = sum(float((s["q"] * fem.tri_area(prob.points, s["tris"])).sum()) for s in prob.surface_source)
        q_out = heat_carried(prob, h, T)
        log(f"    alpha {a:4.2f}: flow iterations {n_it}, max speed {umax:.3f} m/s, heat in {q_in:.2f} W, carried out by the water {q_out:.2f} W, "
            f"{time.time() - t0:.0f} s; " + ", ".join(f"{k.replace('Boundary Probe ', '')} {v - 273.15:.3f}" for k, v in out[a].items()))
        prev = h.regions or prev
    return out, prob


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--meshes", nargs="*", default=["Mesh Coarse", "Mesh Normal"])
    ap.add_argument("--element", default="p2", choices=["p1", "p2"], help="velocity element: p2 quadratic (default), p1 linear")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "waterblock"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, logs = [], []

    def log(s):
        print(s, flush=True)
        logs.append(s)

    for v in args.versions:
        name = f"Waterblock_v{v}_auxiliary_step.mph"
        path = os.path.join(PROJ, name)
        if not os.path.exists(path):
            path = os.path.join(BOX_PROJ, name)
        mph = Mph(path)
        seqs = {d["name"]: tag for tag, d in mph.mesh_sequences().items()}
        for mesh in args.meshes:
            if mesh not in seqs:
                log(f"v{v} {mesh}: no stored mesh")
                continue
            log(f"== water block v{v}, {mesh}")
            try:
                res, prob = solve_case(mph, seqs[mesh], [0.0, 0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], log, args.element)
            except Exception as e:
                log(f"   FAILED: {e!r}")
                continue
            log(f"   {prob.points.shape[0]} nodes, {len(prob.tets)} elements; notes: " + "; ".join(prob.notes))
            ref_path = os.path.join(REF, f"V{v}_Auxiliary_{REF_NAME.get(mesh, mesh)}.csv")
            if not os.path.exists(ref_path):
                continue
            head, data = run_all._table(ref_path)
            for row in data:
                a = float(row[0])
                if a not in res:
                    continue
                for j, hname in enumerate(head[1:], 1):
                    key = next((k for k in res[a] if k.replace("Boundary Probe ", "").lower() in hname.lower()), None)
                    if key:
                        rows.append(dict(version=f"v{v}", mesh=mesh, alpha=a, probe=key, comsol_K=row[j],
                                         replica_K=res[a][key], diff_K=res[a][key] - row[j]))
    if rows:
        with open(os.path.join(args.out, "comparison.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        log(f"{'case':22s} {'probe':28s} {'COMSOL':>9s} {'replica':>9s} {'diff':>8s}")
        for r in rows:
            if r["alpha"] == 1.0:
                log(f"{r['version'] + ' ' + r['mesh']:22s} {r['probe']:28s} {r['comsol_K'] - 273.15:9.3f} "
                    f"{r['replica_K'] - 273.15:9.3f} {r['diff_K']:+8.3f}")
    with open(os.path.join(args.out, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs) + "\n")
    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(dict(code="run_waterblock.py + replica/", inputs=[PROJ, REF], versions=args.versions,
                       meshes=args.meshes, outputs=["comparison.csv", "run.log"]), f, indent=1)


if __name__ == "__main__":
    main()
