"""A water-block mesh made from the STEP files (dev/wb_cad_mesh.py), dressed as the mesh that the solver reads
from a COMSOL file, so that dev/wb_common.build_problem and everything after it work on it unchanged.

    import wb_cad_problem as cp
    mesh, rep = cp.load_cad_mesh(1, 2)                   # design 1, level 2 (MESH_DIR/v1_m2.npz)
    prob = cp.build_problem(1, 2, wb.BASE)               # the same kind of problem as wb.build_problem(mph, mesh, op)

    python -X utf8 dev/wb_cad_problem.py --design 1 --level 1 --dry-run     # build, sizes, PARDISO analysis; no solve
    python -X utf8 dev/wb_cad_problem.py --all --dry-run                    # every mesh found (analysis up to --analyse-max)

What is added to the CAD mesh
  * COMSOL's boundary numbers.  The physics of the model files is attached to boundary numbers (inlet, outlet,
    module face, faces cooled by air, insulated faces).  Every boundary triangle of the CAD mesh takes the number
    of the nearest triangle of COMSOL's fine mesh that separates the same two domains and faces the same way
    (distance from the centre of the CAD triangle to the COMSOL triangle).  The CAD meshes are written in COMSOL's
    frame with COMSOL's domain numbers, so nothing else has to be translated.
  * Checks, returned in `rep` and raised as errors when they fail: domain volumes and boxes against the COMSOL
    mesh (1 %, 0.05 mm), the area of every pair of domains (2 %), the distance of the CAD triangles from the
    COMSOL surface (less than 0.5 mm: the two meshes describe the same surface), the area of every COMSOL boundary
    number on the two meshes.

The dry run builds the heat and flow problem of one case (corrected set-up: 13 W on the footprint), reports the
sizes and runs PARDISO's analysis phase (no factorisation, no solve) for the memory the flow matrix would need.
Results: results/waterblock_research_2026-10-13/meshes/dry_run.json (one entry per mesh; entries are replaced
only by a new run of the same mesh).  Nothing in wb_common.py, wb_runs.py or replica/ is changed.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICA = os.path.dirname(HERE)
for _p in (HERE, REPLICA):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import wb_common as wb  # noqa: E402

MESH_DIR = os.environ.get("WB_CAD_MESHES", os.path.join(REPLICA, "meshes", "wb_cad"))
OUT = os.path.join(REPLICA, "results", "waterblock_research_2026-10-13", "meshes")
_REF = {}


def reference(v):
    """COMSOL's fine mesh of design v (cached): the source of the boundary numbers."""
    if v not in _REF:
        mph = wb.open_model(v)
        _REF[v] = (mph, wb.load_mesh(mph, "fine"))
    return _REF[v]


def _normals(P, tris):
    X = P[tris]
    n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-300)[:, None], 0.5 * a


def _point_tri_distance(p, tri):
    """Distance from points p (m, 3) to triangles tri (m, 3, 3)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    dz = np.einsum("ij,ij->i", p - a, n)
    q = p - dz[:, None] * n                                           # foot on the plane of the triangle
    inside = np.ones(len(p), bool)
    best = np.full(len(p), np.inf)
    for u, w in ((a, b), (b, c), (c, a)):
        e = w - u
        t = np.clip(np.einsum("ij,ij->i", q - u, e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-300), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(q - (u + t[:, None] * e), axis=1))
        inside &= np.einsum("ij,ij->i", np.cross(e, q - u), n) >= 0.0
    return np.hypot(dz, np.where(inside, 0.0, best))


def load_cad_mesh(v, level, folder=MESH_DIR, suffix="", k=12):
    """(mesh dict for wb.build_problem, report dict) of the CAD mesh v{v}_m{level}{suffix}.npz."""
    from scipy.spatial import cKDTree
    from replica import fem
    path = os.path.join(folder, f"v{v}_m{level}{suffix}.npz")
    z = np.load(path)
    mph, ref = reference(v)
    P, T, D = z["points"], z["tets"].astype(np.int32), z["tet_dom"].astype(np.int32)
    tris, up, down = z["tris"].astype(np.int32), z["tri_up"].astype(np.int32), z["tri_down"].astype(np.int32)
    rep = dict(file=os.path.basename(path), nodes=int(len(P)), elements=int(len(T)))
    # ---- domains
    vol, _ = fem.tet_geometry(P, T)
    rvol, _ = fem.tet_geometry(ref["points"], ref["tets"])
    rep["domains"] = {}
    for d in np.unique(ref["tet_dom"]):
        a, b = float(vol[D == d].sum()) * 1e9, float(rvol[ref["tet_dom"] == d].sum()) * 1e9
        if not (D == d).any():
            raise RuntimeError(f"{path}: domain {d} of the COMSOL model is not in the CAD mesh")
        ba = np.r_[P[np.unique(T[D == d])].min(0), P[np.unique(T[D == d])].max(0)] * 1e3
        bb = np.r_[ref["points"][np.unique(ref["tets"][ref["tet_dom"] == d])].min(0), ref["points"][np.unique(ref["tets"][ref["tet_dom"] == d])].max(0)] * 1e3
        rep["domains"][int(d)] = dict(volume_mm3_cad_mesh=round(a, 3), volume_mm3_comsol_fine=round(b, 3), box_difference_mm=round(float(np.abs(ba - bb).max()), 4))
        if abs(a - b) > 0.01 * b + 0.3 or np.abs(ba - bb).max() > 0.05:
            raise RuntimeError(f"{path}: domain {d} differs from the COMSOL mesh: {rep['domains'][int(d)]}")
    # ---- boundary numbers
    nc, ac = _normals(P, tris)
    nr, ar = _normals(ref["points"], ref["tris"])
    cc, cr = P[tris].mean(1), ref["points"][ref["tris"]].mean(1)
    bnd = np.zeros(len(tris), np.int32)
    dist = np.zeros(len(tris))
    rep["pairs"] = {}
    for pr in sorted(set(zip(up.tolist(), down.tolist()))):
        s = np.where((up == pr[0]) & (down == pr[1]))[0]
        r = np.where((ref["tri_up"] == pr[0]) & (ref["tri_down"] == pr[1]))[0]
        if not len(r):
            raise RuntimeError(f"{path}: the CAD mesh has faces between domains {pr}, the COMSOL mesh has none")
        kk = min(k, len(r))
        _, j = cKDTree(cr[r]).query(cc[s], k=kk)
        j = j.reshape(len(s), kk)
        best = np.full(len(s), np.inf)
        pick = np.zeros(len(s), np.int64)
        for c in range(kk):
            rj = r[j[:, c]]
            dd = _point_tri_distance(cc[s], ref["points"][ref["tris"][rj]])
            dd = dd + 1e-3 * (np.abs(np.einsum("ij,ij->i", nc[s], nr[rj])) < 0.7)      # a face turned away: only if nothing else is near
            better = dd < best
            best[better], pick[better] = dd[better], rj[better]
        bnd[s] = ref["tri_bnd"][pick]
        dist[s] = np.minimum(best, _point_tri_distance(cc[s], ref["points"][ref["tris"][pick]]))
        a, b = float(ac[s].sum()) * 1e6, float(ar[r].sum()) * 1e6
        rep["pairs"][f"{pr[0]}|{pr[1]}"] = dict(area_mm2_cad_mesh=round(a, 3), area_mm2_comsol_fine=round(b, 3),
                                                distance_mm_median=round(float(np.median(dist[s])) * 1e3, 4), distance_mm_max=round(float(dist[s].max()) * 1e3, 4))
        if abs(a - b) > 0.02 * b + 0.5:
            raise RuntimeError(f"{path}: area between domains {pr}: {a:.2f} mm2 on the CAD mesh, {b:.2f} on the COMSOL mesh")
    rep["distance_from_comsol_surface_mm"] = dict(median=round(float(np.median(dist)) * 1e3, 4), p99=round(float(np.percentile(dist, 99)) * 1e3, 4),
                                                  max=round(float(dist.max()) * 1e3, 4))
    if dist.max() > 0.5e-3:
        raise RuntimeError(f"{path}: a CAD triangle lies {dist.max() * 1e3:.3f} mm from the COMSOL surface")
    worst, rows = 0.0, {}
    for b in np.unique(ref["tri_bnd"]):
        a1, a2 = float(ac[bnd == b].sum()) * 1e6, float(ar[ref["tri_bnd"] == b].sum()) * 1e6
        rows[int(b)] = (round(a1, 3), round(a2, 3))
        worst = max(worst, abs(a1 - a2))
    sel = wb.selections(mph)
    rep["boundary_numbers"] = dict(count=len(rows), largest_area_difference_mm2=round(worst, 3),
                                   by_selection={n: dict(area_mm2_cad_mesh=round(float(ac[np.isin(bnd, e)].sum()) * 1e6, 3),
                                                         area_mm2_comsol_fine=round(float(ar[np.isin(ref["tri_bnd"], e)].sum()) * 1e6, 3))
                                                 for n, e in sel.items() if n.endswith("Bnd") or n in ("TEC Interface", "HotSide")})
    groups = [str(x) for x in z["group_names"]]
    rep["groups_mm2"] = {g: round(float(ac[z["tri_group"] == i].sum()) * 1e6, 3) for i, g in enumerate(groups) if (z["tri_group"] == i).any()}
    mesh = dict(points=P, tets=T, tet_dom=D, tris=tris, tri_bnd=bnd, tri_up=up, tri_down=down,
                dropped=dict(boundaries={}, nodes=0), tri_group=z["tri_group"], group_names=groups)
    return mesh, rep


def build_problem(v, level, op, alpha=1.0, folder=MESH_DIR, suffix=""):
    """The heat + flow problem (replica.model.build) of design v on a CAD mesh; returns (prob, mesh, report)."""
    mesh, rep = load_cad_mesh(v, level, folder, suffix)
    mph, _ = reference(v)
    return wb.build_problem(mph, mesh, op, alpha), mesh, rep


# --------------------------------------------------------------------------------------------- dry run
def dry_run(v, level, folder=MESH_DIR, suffix="", analyse=True, log=print):
    import psutil
    import scipy.sparse as sp
    from replica import fem, solve
    import wb_flow
    import wb_runs
    t0 = time.time()
    tag = f"v{v}_m{level}{suffix}"
    prob, mesh, rep = build_problem(v, level, wb.BASE, folder=folder, suffix=suffix)
    mph, _ = reference(v)
    log(f"== {tag}: problem built in {time.time() - t0:.1f} s; {rep['nodes']} nodes, {rep['elements']} elements; "
        f"distance from the COMSOL surface: median {rep['distance_from_comsol_surface_mm']['median']} mm, largest {rep['distance_from_comsol_surface_mm']['max']} mm")
    vol, g = fem.tet_geometry(prob.points, prob.tets)
    spec = dict(prob.flow["regions"][0])
    r = wb_flow.WBFlow(prob, spec, vol, g)
    r.tasm = fem.Assembler(prob.tets[r.el], prob.points.shape[0])
    r.set_temperature(np.full(prob.points.shape[0], wb.BASE["Tin"]))
    unknowns = int(sum(len(x) for x in r.free_c) + r.n1)
    d_in = 2.0 * np.sqrt(r.inlet_faces["area"].sum() / np.pi)
    out = dict(case=tag, design=v, level=level, check=rep, water_elements=int(len(r.el)), velocity_nodes=int(r.n2), pressure_unknowns=int(r.n1),
               flow_unknowns=unknowns, inlet_area_mm2=round(float(r.inlet_faces["area"].sum()) * 1e6, 4),
               outlet_area_mm2=round(float(r.outlet_faces["area"].sum()) * 1e6, 4), inlet_diameter_mm=round(float(d_in) * 1e3, 4),
               transit_time_s_at_0p25_L_min=round(float(r.vol.sum() / r.inflow), 4))
    # the corrected set-up: 13 W on the footprint (as block W3 of dev/wb_runs.py)
    s = dict(annulus_mm=(7.0, 13.0), alpha=1.0)
    src, heated, hinfo = wb_runs.correct_heating(prob, s, dict(P_W=13.0, T0=wb.BASE["T0"]), lambda m: None)
    prob.surface_source = src
    h = solve.Heat(prob, log=lambda m: None, flow=False, supg_heat=True)
    h.regions = [r]
    q_in = float(sum((np.asarray(x["q"]) * fem.tri_area(prob.points, x["tris"])).sum() for x in prob.surface_source))
    faces = wb.face_sets(mph, prob)
    out["heating"] = dict(hinfo, heat_in_W=round(q_in, 6))
    out["read_out_faces_mm2"] = {k_: round(float(a.sum()) * 1e6, 3) for k_, (_, a) in faces.items()}
    fgb, mem, sec = wb_runs.estimate(unknowns, v)
    out["estimate_from_the_fit_of_the_comsol_meshes"] = dict(factor_GB=round(fgb, 1), memory_to_request_GB=round(mem, 1), seconds_per_flow_step_cluster=round(sec, 1))
    log(f"   flow: {out['water_elements']} water elements, {unknowns} unknowns; heat in {q_in:.4f} W on {hinfo['heated_area_mm2']} mm2 "
        f"({hinfo['triangles_cut_by_the_edge']} triangles cut by the edge of the footprint); fit of the COMSOL meshes: factor {fgb:.1f} GB, request {mem:.0f} GB")
    if analyse:
        import pypardiso
        rng = np.random.default_rng(0)
        u = r.u.copy()
        free = ~r.fixed_u
        u[free] = 0.1 * rng.standard_normal(int(free.sum()))
        t1 = time.time()
        J = sp.csr_matrix(r.jacobian(u, a=0.0, newton=True))
        J.sort_indices()
        t_asm = time.time() - t1
        ps = pypardiso.PyPardisoSolver()
        ps.set_iparm(18, -1)
        ps.set_iparm(19, -1)
        ps.set_phase(11)
        t1 = time.time()
        ps._call_pardiso(J, np.zeros((J.shape[0], 1)))
        out["pardiso_analysis"] = dict(matrix="newton", matrix_nonzeros=int(J.nnz), factor_nonzeros=int(ps.get_iparm(18)), factor_MFlop=int(ps.get_iparm(19)),
                                       pardiso_peak_symbolic_GB=round(ps.get_iparm(15) / 1e6, 3),
                                       pardiso_total_GB=round(max(ps.get_iparm(15), ps.get_iparm(16) + ps.get_iparm(17)) / 1e6, 3),
                                       assemble_seconds=round(t_asm, 1), analysis_seconds=round(time.time() - t1, 1), threads=os.environ.get("MKL_NUM_THREADS"))
        ps.free_memory(everything=True)
        mi = psutil.Process().memory_info()
        out["pardiso_analysis"]["process_peak_GB"] = round(getattr(mi, "peak_wset", mi.rss) / 1e9, 2)
        a = out["pardiso_analysis"]
        # memory of a run: the factors, twice the matrix, and the Python side as in dev/wb_runs.estimate
        a["memory_to_request_GB"] = round(1.15 * a["pardiso_total_GB"] + 6300.0 * unknowns / 1e9 + 2.0, 1)
        log(f"   PARDISO analysis: factor {a['factor_nonzeros'] / 1e6:.0f} M non-zeros, {a['pardiso_total_GB']} GB, {a['factor_MFlop'] / 1e3:.0f} GFlop per factorisation; "
            f"memory to request {a['memory_to_request_GB']} GB; peak of this process {a['process_peak_GB']} GB")
    out["seconds"] = round(time.time() - t0, 1)
    out["made"] = time.strftime("%Y-%m-%d %H:%M")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=int, nargs="*", default=[1])
    ap.add_argument("--level", type=int, nargs="*", default=[1])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--all", action="store_true", help="every v*_m*.npz in the folder")
    ap.add_argument("--folder", default=MESH_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse-max", type=int, default=450000, help="largest number of water elements for PARDISO's analysis phase")
    a = ap.parse_args()
    if not a.dry_run:
        ap.error("only --dry-run is implemented here: solves go through the run driver")
    if a.all:
        todo = []
        for p in sorted(glob.glob(os.path.join(a.folder, "v*_m*.npz"))):
            n = os.path.basename(p)[:-4]
            todo.append((int(n[1]), int(n.split("_m")[1][0]), n.split("_m")[1][1:]))
    else:
        todo = [(v, L, a.suffix) for v in a.design for L in a.level]
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "dry_run.json")
    res = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    for v, L, sfx in todo:
        with open(os.path.join(a.folder, f"v{v}_m{L}{sfx}.json"), encoding="utf-8") as fh:
            nw = json.load(fh)["mesh"]["water_elements"]
        d = dry_run(v, L, a.folder, sfx, analyse=nw <= a.analyse_max)
        res[d["case"]] = d
        wb.write_json(f, res)
    print("written:", f)


if __name__ == "__main__":
    main()
