"""Adapted meshes for the new eye runs (dev/new_runs.py set-up), with a solve behind every claim.

Written 2026-10-08 (mesh work).  Nothing in replica/, run_all.py or dev/new_runs.py is edited:
the problem is built exactly as new_runs.run_one does (model.build on the COMSOL file with mesh=<candidate>,
then new_runs.transform), and solved with the same calls (stationary start, cooling transient).

Two ways of making a mesh from a starting mesh read out of the COMSOL file
  refine_local   exact, no mesher: marked elements are split into eight ("red"), their neighbours are closed
                 with one or four children ("green") so that no node hangs.  Geometry (the flat facets of the
                 starting mesh), domain and boundary numbers are unchanged.  Needs numpy only, so it runs on
                 the cluster.  Marking is by domain name (two elements across the thin layers), by distance,
                 by an error indicator (replica.mesher.error_indicator), or by a level map taken from another
                 adapted mesh (transfer of a study on the coarser mesh to the production mesh).
  layers2_mmg    the layered recipe: every element split into eight, the thin domains frozen, the
                 rest brought back to the starting size with MMG (metric_tools.adapt_frozen), optional
                 smooth_slivers.  Needs mmgpy (this PC only).

Commands (run from the repository root, python -X utf8)
  python dev/mesh_adapt.py selftest
        red/green refinement on COMSOL's coarser mesh: volumes, conformity, boundary numbers.
  python dev/mesh_adapt.py study --ring v1 --config invivo --out D:\\COMSOL\\overnight_scratch\\mesher\\study_v1_invivo
        the convergence study on COMSOL's coarser mesh: uniform splitting, layer-only splitting, MMG recipe
        with and without sliver smoothing, indicator-driven passes.  One solve at a time; a candidate that
        is already there is not solved again.  Writes candidates.csv, readouts.csv, indicator_by_domain.csv,
        verdict.csv, levels_<pass>.npz (the level map) and the meshes.
  python dev/mesh_adapt.py loop --ring v1 --config invivo --start production --max-elements 3e6 --out <dir>
        the indicator-driven loop alone, from any starting mesh (for the cluster: needs no mmgpy).
  python dev/mesh_adapt.py transfer --ring v1 --config invivo --levels <levels.npz> --layers-all 0|1 --out mesh.npz
        apply a level map of the coarser study to the production mesh (no solve).
  python dev/mesh_adapt.py solve --ring v1 --config invivo --mesh-file mesh.npz --out <dir>
        one solve on a stored mesh (what the driver would do with a mesh_file setting).

Acceptance: deep read-outs within 0.05 C, read-outs next to the ring within 0.10 C.
"""
import argparse
import csv
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICA = os.path.dirname(HERE)
sys.path.insert(0, REPLICA)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import metric_tools as mt  # noqa: E402
import new_runs as nr  # noqa: E402

K0 = 273.15
EDGES = mt.EDGES                                  # (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
# faces of a tetrahedron: (corner indices, indices of its three edges in EDGES order ab ac bc, opposite corner)
FACES = (((0, 1, 2), (0, 1, 3), 3), ((0, 1, 3), (0, 2, 4), 2), ((0, 2, 3), (1, 2, 5), 1), ((1, 2, 3), (3, 4, 5), 0))
LAYERS = ("Sclera", "Choroid", "Retina")
SMALL = ("TEC Layer", "Ciliary Body", "Cornea", "Tear film")       # frozen besides the layers in the MMG recipe
TOL_DEEP, TOL_RING = 0.05, 0.10


# =========================================================================== exact local refinement
def refine_local(mesh, marked, log=None):
    """Split the ``marked`` elements (boolean per tetrahedron) into eight, conformingly.  Returns a new mesh dict.

    Closure rule: after marking all six edges of every marked element, an element may have 0, 1, 3 (the three
    edges of one face) or 6 split edges.  Two split edges on one face -> the third is split too; any other
    combination -> the element is split into eight.  Repeated until nothing changes.  Elements with one split
    edge are cut in two, with one split face in four ("green"); these have worse angles than their parent and
    are flagged in ``tet_green``.  ``tet_level`` counts how many times an element's ancestors were split in
    eight.  Boundary triangles follow their edges; their boundary number and the domains on their two sides
    are inherited, so materials and conditions apply unchanged.
    """
    P, T = mesh["points"], mesh["tets"].astype(np.int64)
    nt, n = len(T), len(P)
    level = np.asarray(mesh.get("tet_level", np.zeros(nt, np.int8)), np.int8)
    green = np.asarray(mesh.get("tet_green", np.zeros(nt, bool)), bool)
    e = np.concatenate([T[:, [a, b]] for a, b in EDGES])
    e.sort(axis=1)
    uk, inv = np.unique(e[:, 0] * n + e[:, 1], return_inverse=True)
    te = inv.reshape(6, -1).T                                        # (nt, 6) edge number of each element edge
    em = np.zeros(len(uk), bool)
    em[te[np.asarray(marked, bool)].ravel()] = True
    for it in range(200):
        m = em[te]
        c = m.sum(1)
        before = int(em.sum())
        on_face = np.zeros(nt, bool)
        for _, ed, _ in FACES:
            cnt = m[:, ed].sum(1)
            full = (c == 3) & (cnt == 3)
            on_face |= full
            two = (c == 2) & (cnt == 2)
            if two.any():
                em[te[two][:, ed].ravel()] = True
                on_face |= two
        red = ((c >= 2) & (c < 6)) & ~on_face
        if red.any():
            em[te[red].ravel()] = True
        if int(em.sum()) == before:
            break
    else:
        raise RuntimeError("closure did not settle")
    m = em[te]
    c = m.sum(1)
    mid = np.full(len(uk), -1, np.int64)
    mid[em] = n + np.arange(int(em.sum()))
    Pn = np.vstack([P, 0.5 * (P[uk[em] // n] + P[uk[em] % n])])
    tm = mid[te]                                                     # midpoint node per element edge (-1: none)

    out_t, out_src, out_lvl, out_green = [], [], [], []

    def add(tets, src, dl, g):
        out_t.append(tets)
        out_src.append(src)
        out_lvl.append(level[src] + dl)
        out_green.append(green[src] | g)

    s = np.where(c == 0)[0]
    add(T[s], s, 0, False)
    # ---- red (as metric_tools.refine_uniform: inner octahedron cut along its shortest diagonal)
    s = np.where(c == 6)[0]
    if len(s):
        a, b, cc, d = T[s].T
        ab, ac, ad, bc, bd, cd = tm[s].T
        for q in ((a, ab, ac, ad), (ab, b, bc, bd), (ac, bc, cc, cd), (ad, bd, cd, d)):
            add(np.stack(q, 1), s, 1, False)
        L = np.stack([np.linalg.norm(Pn[ab] - Pn[cd], axis=1), np.linalg.norm(Pn[ac] - Pn[bd], axis=1),
                      np.linalg.norm(Pn[ad] - Pn[bc], axis=1)], 1)
        pick = L.argmin(1)
        opts = (((ab, cd, ac, ad), (ab, cd, ad, bd), (ab, cd, bd, bc), (ab, cd, bc, ac)),
                ((ac, bd, ab, ad), (ac, bd, ad, cd), (ac, bd, cd, bc), (ac, bd, bc, ab)),
                ((ad, bc, ab, ac), (ad, bc, ac, cd), (ad, bc, cd, bd), (ad, bc, bd, ab)))
        for k, four in enumerate(opts):
            mk = pick == k
            for q in four:
                add(np.stack(q, 1)[mk], s[mk], 1, False)
    # ---- one split edge: two children
    for j, (ia, ib) in enumerate(EDGES):
        s = np.where((c == 1) & m[:, j])[0]
        if not len(s):
            continue
        ic, idd = [x for x in range(4) if x not in (ia, ib)]
        a, b, cc, d, mm = T[s, ia], T[s, ib], T[s, ic], T[s, idd], tm[s, j]
        add(np.stack((a, mm, cc, d), 1), s, 0, True)
        add(np.stack((mm, b, cc, d), 1), s, 0, True)
    # ---- one split face: four children
    for (ia, ib, ic), ed, idd in FACES:
        s = np.where((c == 3) & m[:, ed].all(1))[0]
        if not len(s):
            continue
        a, b, cc, d = T[s, ia], T[s, ib], T[s, ic], T[s, idd]
        ab, ac, bc = tm[s, ed[0]], tm[s, ed[1]], tm[s, ed[2]]
        for q in ((a, ab, ac, d), (ab, b, bc, d), (ac, bc, cc, d), (ab, bc, ac, d)):
            add(np.stack(q, 1), s, 0, True)
    tets = np.concatenate(out_t)
    src = np.concatenate(out_src)
    x = Pn[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]

    # ---- boundary triangles
    tr = mesh["tris"].astype(np.int64)
    p, q, r = tr.T

    def edge_mid(u, v):
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        return mid[np.searchsorted(uk, lo * n + hi)]
    pq, qr, pr = edge_mid(p, q), edge_mid(q, r), edge_mid(p, r)
    cnt = (pq >= 0).astype(int) + (qr >= 0) + (pr >= 0)
    if (cnt == 2).any():
        raise RuntimeError("a boundary triangle has two split edges: closure failed")
    tl, ts = [], []
    s = cnt == 0
    tl.append(tr[s]); ts.append(np.where(s)[0])
    s = cnt == 3
    for qd in ((p, pq, pr), (pq, q, qr), (pr, qr, r), (pq, qr, pr)):
        tl.append(np.stack(qd, 1)[s]); ts.append(np.where(s)[0])
    for mnode, (u, v, w) in ((pq, (p, q, r)), (qr, (q, r, p)), (pr, (r, p, q))):
        s = (cnt == 1) & (mnode >= 0)
        tl.append(np.stack((u, mnode, w), 1)[s]); ts.append(np.where(s)[0])
        tl.append(np.stack((mnode, v, w), 1)[s]); ts.append(np.where(s)[0])
    tris, tsrc = np.concatenate(tl), np.concatenate(ts)
    out = dict(points=Pn, tets=tets.astype(np.int32), tet_dom=mesh["tet_dom"][src].astype(np.int32),
               tris=tris.astype(np.int32), tri_bnd=mesh["tri_bnd"][tsrc].astype(np.int32),
               tri_up=mesh["tri_up"][tsrc].astype(np.int32), tri_down=mesh["tri_down"][tsrc].astype(np.int32),
               tet_level=np.concatenate(out_lvl).astype(np.int8), tet_green=np.concatenate(out_green), version=0)
    if log:
        log(f"    refine_local: {int(np.asarray(marked).sum())} marked, {int((c == 6).sum())} split in eight, "
            f"{int((c == 1).sum())} in two, {int((c == 3).sum())} in four; {nt} -> {len(tets)} elements "
            f"({it + 1} closure rounds)")
    return out


def volumes(mesh):
    x = mesh["points"][mesh["tets"]] * 1e3
    v = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0]) / 6
    return v


def conformity(mesh):
    """(number of faces shared by more than two elements, outer area in mm2, least element volume in mm3)."""
    T = mesh["tets"].astype(np.int64)
    n = int(T.max()) + 1
    f = np.concatenate([np.sort(T[:, c], axis=1) for c in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])])
    key = (f[:, 0] * n + f[:, 1]) * n + f[:, 2]
    uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
    outer = f[first[cnt == 1]]
    x = mesh["points"][outer] * 1e3
    area = 0.5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1).sum()
    return int((cnt > 2).sum()), float(area), float(volumes(mesh).min())


def angle_stats(mesh, kept=None, layer_ids=()):
    """Largest dihedral angle per element: summary over the elements in ``kept`` (boolean) or all."""
    a = mt.max_dihedral(mesh["points"], mesh["tets"])
    k = np.ones(len(a), bool) if kept is None else kept
    lay = np.isin(mesh["tet_dom"], list(layer_ids))
    g = np.asarray(mesh.get("tet_green", np.zeros(len(a), bool)), bool)
    return dict(angle_median=round(float(np.median(a[k])), 1), angle_p99=round(float(np.percentile(a[k], 99)), 1),
                angle_max=round(float(a[k].max()), 1), n_above_150=int((a[k] > 150).sum()),
                n_above_160=int((a[k] > 160).sum()), n_above_170=int((a[k] > 170).sum()),
                n_above_160_layers=int((a[k & lay] > 160).sum()), n_above_160_green=int((a[k & g] > 160).sum()),
                n_green=int((k & g).sum()))


# =========================================================================== the problem, as the driver builds it
class Setup:
    """One ring and configuration: reads the COMSOL file once, builds and solves candidates."""

    def __init__(self, ring="v1", ex_vivo=False, start="coarser", log=print, **over):
        import run_all
        import sensor_probes
        from replica.mphread import Mph
        self.run_all, self.sensor_probes, self.log = run_all, sensor_probes, log
        s = dict(nr.DEFAULTS)
        s.update(ring=ring, ex_vivo=ex_vivo, mesh=start)
        s.update(over)
        self.s = s
        rel = nr.MESH_FILE[start]
        path = run_all.mph_path(rel)
        if not os.path.exists(path):
            path = os.path.join(run_all.BOX, "Projects", "SCR", *rel.split("\\"))
        self.path = path
        self.mph = Mph(path)
        self.comp = next(c for c, v in self.mph.components().items() if v["label"].endswith(ring))
        self.info = self.mph.components()[self.comp]
        self.start_mesh = self.mph.mesh(self.mph.mesh_resources()[self.info["geom"]]["resource"])
        self.fixed = None                       # scleral sensor points snapped once, on the starting mesh
        base = self._base(self.start_mesh)
        self.names = dict(base.domain_names)
        ex = self.sensor_probes.sensor_probes()
        self.fixed = {k.replace("(on sclera)", "(on sclera, fixed)"): v
                      for k, v in self.sensor_probes.snap_to_sclera(base, ex).items()}
        self.dropped = set(nr._doms(base, "Air") + nr._doms(base, "TEC Layer")
                           + (nr._doms(base, *nr.ORBIT) if ex_vivo else []))

    def ids(self, *names):
        return sorted(d for d, n in self.names.items() if n in names or any(k in n for k in names if k == "Cloquet"))

    def kept(self, mesh):
        return ~np.isin(mesh["tet_dom"], list(self.dropped))

    def _base(self, mesh):
        from replica import model
        s = self.s
        extra = self.sensor_probes.sensor_probes()
        base = model.build(self.mph, component=self.comp, heat=self.info["heat"], geom=self.info["geom"],
                           insulation_slit=False, mesh=mesh,
                           overrides=dict(T0=s["T0"] + K0, Tb=s["Tb"] + K0, Tc=s["Tc"] + K0),
                           switch_index=nr.MATERIAL_INDEX[s["material"]], extra_probes=extra)
        extra.update(self.sensor_probes.snap_to_sclera(base, extra))
        base.probes.update(extra)
        if self.fixed:
            base.probes.update(self.fixed)
        return base

    def build(self, mesh, log=None):
        base = self._base(mesh)
        prob, cool, tinfo = nr.transform(base, self.s, log or self.log)
        return base, prob, cool, tinfo

    def solve(self, mesh, snap_min=(2.0, 10.0), log=None, dual=False):
        """Start state and cooling transient, as new_runs.run_one.  Returns a dict of read-outs and fields."""
        log = log or self.log
        t0 = time.time()
        base, prob, cool, tinfo = self.build(mesh, log)
        out = self.march(base, prob, cool, snap_min, log, dual)
        out.update(seconds=time.time() - t0, tinfo=tinfo)
        return out

    @staticmethod
    def _dual(h, prob, T, tau):
        """Influence field of the judged read-outs, for goal-oriented marking.

        z solves (K + radiation and surface coefficients at the end state + M / tau) z = sum over the judged
        read-outs of (read-out functional) / tolerance, z = 0 on the cooling face and the body shell.  K is
        symmetric, so this is the adjoint of the linearised end state; M / tau (tau = the cooling time)
        stands for the finite duration.  Where z changes fast, an error in the heat flux reaches a read-out.
        """
        from replica import solve
        rhs = np.zeros(h.n)
        for name, pr in h._pp.items():
            tol = probe_class(name)
            if tol is None:
                continue
            if pr[0] == "point":
                np.add.at(rhs, pr[1], np.asarray(pr[2]) / tol)
            elif pr[0] == "average":
                np.add.at(rhs, pr[1].ravel(), np.repeat(pr[2] / 4.0 / pr[2].sum() / tol, 4))
        Kv, Mv = h.variable_matrices(T)
        _, Br = h.radiation(T)
        A = (h.K0 + Kv + Br + (h.M0 + Mv) / tau).tocsr()
        fixed, _ = h.constraints("cool", tau)
        free = np.setdiff1d(np.arange(h.n), fixed)
        lu = solve.factorize(A[free][:, free])
        z = np.zeros(h.n)
        z[free] = lu(rhs[free])
        if hasattr(lu, "free"):
            lu.free()
        return z

    def march(self, base, prob, cool, snap_min=(2.0, 10.0), log=None, dual=False):
        """The solve itself, for a problem already built (``base`` is only read for the model's ramp shape)."""
        from replica import solve
        log = log or self.log
        s = self.s
        t0 = time.time()
        h = solve.Heat(prob, log=log)
        h.probe_setup()
        names = list(prob.probes)
        T = h.stationary(step="start")
        face0 = float(T[cool["nodes"]].mean())
        Tc = s["Tc"] + K0
        old = next(d for d in base.dirichlet if "step" in d.get("expr", ""))
        T0k, t_ramp = s["T0"] + K0, float(base.params.get("t_ramp", 90.0))
        span = (T0k - Tc) if abs(T0k - Tc) > 1e-9 else 1.0

        def ramp(t, f=t_ramp / s["ramp_s"]):
            return min(1.0, max(0.0, (T0k - old["value"](t * f)) / span)) if abs(T0k - Tc) > 1e-9 else float(t > 0)

        cool["value"] = lambda t, a=face0, b=Tc: float(a + (b - a) * ramp(t))
        start = h.probes(T)
        fields = {"start": T.copy()}
        rows, last, want = [], {}, list(snap_min)

        def cb(t, Tn):
            p = h.probes(Tn)
            q = float("nan")
            if "T" in last and t > last["t"]:
                Kv, Mv = h.variable_matrices(Tn)
                fr, _ = h.radiation(Tn)
                r = (h.K0 + Kv) @ Tn + (h.M0 + Mv) @ ((Tn - last["T"]) / (t - last["t"])) - h.F0 - fr
                q = -2.0 * float(r[cool["nodes"]].sum())
            last["T"], last["t"] = Tn.copy(), t
            rows.append([t / 60.0] + [p[nm] for nm in names] + [q])
            if want and t / 60.0 >= want[0] - 1e-9:
                fields[f"{want.pop(0):g}min"] = Tn.copy()

        stages = [(tu, dt * s["dt_scale"]) for tu, dt in self.run_all.stages(s["t_cool_min"] * 60.0)]
        T = h.transient(T, s["t_cool_min"] * 60.0, stages, step="cool", callback=cb, tol=1e-5)
        fields["end"] = T.copy()
        peak = float("nan")
        try:
            import psutil
            mi = psutil.Process().memory_info()
            peak = getattr(mi, "peak_wset", mi.rss) / 2 ** 30
        except Exception:
            try:                                           # Linux without psutil: peak resident size in kB
                import resource
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
            except Exception:
                pass
        if dual:
            fields_dual = self._dual(h, prob, T, s["t_cool_min"] * 60.0)
        out = dict(prob=prob, names=names + ["Cold-side heat, whole device (W)"], start=start,
                   dual=fields_dual if dual else None,
                   series=np.array(rows, float), fields=fields, seconds=time.time() - t0, peak_gb=peak,
                   face0_C=face0 - K0)
        del h
        gc.collect()
        return out


def probe_class(name):
    """Tolerance of a read-out [C], or None if it is reported but not judged."""
    p = name.lower()
    if "(w)" in p or " max" in p or " min" in p or "minimum" in p or "maximum" in p:
        return None                               # extremes sit at single nodes; heat is reported separately
    if "(on sclera)" in p:
        return None                               # the snapped point moves with the mesh: use the fixed copy
    if any(k in p for k in ("anter", "equatorial", "scr ", "ring average")):
        return TOL_RING
    return TOL_DEEP


def short(name):
    import re
    name = re.sub(r" Probe [123]$", "", name.replace(" (K)", ""))
    if name.startswith("Domain Probe ") or name.startswith("SCR "):
        name = re.sub(r" [123]$", "", name.replace("Domain Probe ", ""))
    return name.replace("Anteriror", "Anterior")


# =========================================================================== candidates
def mark_names(setup, mesh, names):
    return np.isin(mesh["tet_dom"], setup.ids(*names))


def layers2_mmg(setup, base, factor=2.0, gradation=3.0, smooth=False, log=None, extra_frozen=SMALL):
    """Layered recipe: split everything in eight, freeze the thin domains, bring the rest back by ``factor`` in size."""
    fine = mt.refine_uniform(base)
    frozen = setup.ids(*LAYERS, *extra_frozen, "Cloquet")
    M = mt.implied_metric(fine["points"] * 1e3, fine["tets"])
    new = mt.adapt_frozen(fine, mt.scale(M, factor), frozen, gradation=gradation, surface_tol_mm=0.2)
    if smooth:
        new, hist = mt.smooth_slivers(new, frozen)
        if log:
            log(f"    smooth_slivers outside the frozen domains: {hist[0][0]} -> {hist[-1][0]} above 160 deg")
    return new


def indicator(setup, mesh, res, goal=False):
    """Squared flux-recovery indicator per element of the FULL mesh (zero in removed domains), summed over the
    stored states of the solve ``res``.  ``goal``: weighted by the influence field of the read-outs instead
    (Setup._dual), so that elements whose flux error does not reach a read-out are not refined."""
    from replica import mesher
    import copy
    prob = res["prob"]
    e2 = np.zeros(len(prob.tets))
    for T in res["fields"].values():
        eta, _ = mesher.error_indicator(prob, T)
        e2 += eta ** 2
    if goal:
        # goal-oriented: element indicator of the temperature times that of the influence field of the read-outs
        p2 = copy.copy(prob)
        p2.k = {d: (float(np.atleast_1d(k(np.array([293.15])))[0]) if callable(k) else k) for d, k in prob.k.items()}
        ez, _ = mesher.error_indicator(p2, res["dual"])
        e2 = np.sqrt(e2) * ez
    full = np.zeros(len(mesh["tets"]))
    full[np.where(setup.kept(mesh))[0]] = e2
    return full


def mark_by_indicator(e2, theta, max_marked):
    order = np.argsort(-e2)
    cum = np.cumsum(e2[order])
    n = int(np.searchsorted(cum, theta * cum[-1]) + 1)
    n = max(1, min(n, int(max_marked)))
    m = np.zeros(len(e2), bool)
    m[order[:n]] = True
    return m


def level_map(mesh, kept):
    """What to store to carry an adapted mesh's refinement pattern to another mesh of the same geometry."""
    c = mesh["points"][mesh["tets"]].mean(1)
    lvl = np.asarray(mesh.get("tet_level", np.zeros(len(c), np.int8)))
    return dict(centroids=c[kept].astype(np.float32), level=lvl[kept], dom=mesh["tet_dom"][kept])


def transfer_levels(setup, mesh, lm, layers_all=False, bulk_offset=1, log=None, layer_offset=0):
    """Refine ``mesh`` (e.g. production) to the levels of a level map made on a coarser mesh of the same geometry.

    In the thin layers the level is applied as it is (what counts there is the number of elements across the
    layer, and COMSOL's meshes all have about one).  Elsewhere the level is lowered by ``bulk_offset``
    (production elements are about half the size of the coarser mesh's: 1.04 million against 90 thousand).
    ``layers_all``: at least one split of every layer element (two across everywhere).  ``layer_offset``: levels taken
    off in the layers too, but never below one split where the map has one or more.
    """
    from scipy.spatial import cKDTree
    lay = setup.ids(*LAYERS)
    for rnd in range(4):
        kept = setup.kept(mesh)
        c = mesh["points"][mesh["tets"]].mean(1)
        cur = np.asarray(mesh.get("tet_level", np.zeros(len(c), np.int8))).astype(int)
        want = np.zeros(len(c), int)
        for d in np.unique(mesh["tet_dom"][kept]):
            src = lm["dom"] == d
            if not src.any():
                continue
            sel = np.where(kept & (mesh["tet_dom"] == d))[0]
            _, j = cKDTree(lm["centroids"][src]).query(c[sel], k=4)
            w = lm["level"][src][j].max(1).astype(int)             # the finest of the four nearest
            want[sel] = np.where(w > 0, np.maximum(w - layer_offset, 1), 0) if d in lay else np.maximum(w - bulk_offset, 0)
        if layers_all:
            want[np.isin(mesh["tet_dom"], lay)] = np.maximum(want[np.isin(mesh["tet_dom"], lay)], 1)
        green = np.asarray(mesh.get("tet_green", np.zeros(len(c), bool)))
        mark = (want > cur) & ~green
        if not mark.any():
            break
        mesh = refine_local(mesh, mark, log)
    return mesh


def save_mesh(path, mesh):
    keys = ("points", "tets", "tet_dom", "tris", "tri_bnd", "tri_up", "tri_down", "tet_level", "tet_green")
    np.savez_compressed(path, **{k: mesh[k] for k in keys if k in mesh and mesh[k] is not None})


def load_mesh(path):
    return dict(np.load(path))


# =========================================================================== the study
class Study:
    def __init__(self, setup, out, log):
        self.su, self.out, self.log = setup, out, log
        os.makedirs(out, exist_ok=True)
        self.cand = {}                      # name -> row of candidates.csv
        self.series = {}                    # name -> (names, start dict, series array)
        self.lay = setup.ids(*LAYERS)
        import glob
        found = sorted(glob.glob(os.path.join(out, "result_*.json")), key=os.path.getmtime)
        for jp in found:                    # results of earlier calls
            name = os.path.basename(jp)[len("result_"):-len(".json")]
            rp = jp[:-5] + ".npz"
            if os.path.exists(rp):
                with open(jp, encoding="utf-8") as f:
                    self.cand[name] = json.load(f)
                z = np.load(rp, allow_pickle=True)
                self.series[name] = (list(z["names"]), dict(zip(z["start_names"], z["start_vals"])), z["series"])

    def run(self, name, make, note="", keep_fields=False, dual=False):
        """Make (or load) the mesh ``name``, solve on it once, store everything.  Returns (mesh, result or None)."""
        su = self.su
        mpath = os.path.join(self.out, f"mesh_{name}.npz")
        rpath = os.path.join(self.out, f"result_{name}.npz")
        jpath = os.path.join(self.out, f"result_{name}.json")
        t0 = time.time()
        if os.path.exists(mpath):
            mesh = load_mesh(mpath)
            made_s = None
        else:
            self.log(f"[{name}] making the mesh")
            mesh = make()
            made_s = time.time() - t0
            save_mesh(mpath, mesh)
        res = None
        if os.path.exists(rpath) and os.path.exists(jpath) and not keep_fields:
            with open(jpath, encoding="utf-8") as f:
                row = json.load(f)
            z = np.load(rpath, allow_pickle=True)
            self.series[name] = (list(z["names"]), dict(zip(z["start_names"], z["start_vals"])), z["series"])
        else:
            kept = su.kept(mesh)
            nbad, area, vmin = conformity(dict(points=mesh["points"], tets=mesh["tets"][kept]))
            self.log(f"[{name}] {int(kept.sum())} elements to solve ({len(mesh['tets'])} with the removed domains)")
            res = su.solve(mesh, log=lambda m: self.log("    " + m.strip()), dual=dual)
            row = dict(candidate=name, note=note, elements_total=len(mesh["tets"]), elements_solved=int(kept.sum()),
                       nodes_solved=int(res["prob"].points.shape[0]),
                       elements_in_layers=int(np.isin(mesh["tet_dom"], self.lay).sum()),
                       mesh_seconds=None if made_s is None else round(made_s, 1),
                       solve_seconds=round(res["seconds"], 1), peak_memory_GB=round(res["peak_gb"], 2),
                       faces_shared_by_more_than_two=nbad, least_volume_mm3=vmin,
                       start_face_C=round(res["face0_C"], 3))
            row.update(angle_stats(mesh, kept, self.lay))
            vol = np.abs(volumes(mesh))
            for d in sorted(set(mesh["tet_dom"][kept].tolist())):
                row[f"vol_mm3_{su.names[d]}_{d}"] = round(float(vol[mesh["tet_dom"] == d].sum()), 3)
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=1)
            np.savez_compressed(rpath, names=np.array(res["names"]), series=res["series"],
                                start_names=np.array(list(res["start"])), start_vals=np.array(list(res["start"].values())))
            self.series[name] = (res["names"], res["start"], res["series"])
            self.log(f"[{name}] solved in {res['seconds']:.0f} s, peak memory {res['peak_gb']:.1f} GB; "
                     f"angles: max {row['angle_max']}, above 160 deg {row['n_above_160']}")
        self.cand[name] = row
        return mesh, res

    def readout(self, name):
        """{(state, read-out): C} with states start, 5 min, 15 min, end."""
        names, start, ser = self.series[name]
        out = {}
        for n in names[:-1]:
            out[("start", n)] = float(start[n]) - K0
        for label, tmin in (("5 min", 5.0), ("15 min", 15.0), ("end", ser[-1, 0])):
            i = int(np.abs(ser[:, 0] - tmin).argmin())
            for j, n in enumerate(names):
                out[(label, n)] = ser[i, j + 1] - (0.0 if "(W)" in n else K0)
        return out

    def distance(self, a, b):
        """Largest |a - b| / tolerance over the judged read-outs (start state and end of cooling), the read-out,
        the difference; and the largest ratio over the whole time course."""
        ra, rb = self.readout(a), self.readout(b)
        best = (0.0, "", 0.0)
        for k in ra:
            tol = probe_class(k[1])
            if tol is None or k not in rb or k[0] not in ("start", "end"):
                continue
            d = abs(ra[k] - rb[k])
            if d / tol > best[0]:
                best = (d / tol, f"{short(k[1])} ({k[0]})", d)
        na, _, sa = self.series[a]
        nb, _, sb = self.series[b]
        worst_t = (0.0, "", 0.0)
        if sa.shape == sb.shape:
            for j, n in enumerate(na):
                tol = probe_class(n)
                if tol is None:
                    continue
                d = np.abs(sa[:, j + 1] - sb[:, j + 1])
                i = int(d.argmax())
                if d[i] / tol > worst_t[0]:
                    worst_t = (d[i] / tol, f"{short(n)} at {sa[i, 0]:.1f} min", float(d[i]))
        return best, worst_t

    def write(self, reference=None):
        rows = list(self.cand.values())
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(self.out, "candidates.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(self.out, "readouts.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["candidate", "elements_solved", "state", "read_out", "tolerance_C", "value"])
            for name in self.cand:
                for (st, n), v in self.readout(name).items():
                    w.writerow([name, self.cand[name]["elements_solved"], st, short(n), probe_class(n) or "", round(v, 4)])
        if reference and reference in self.series:
            with open(os.path.join(self.out, "verdict.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["candidate", "elements_solved", "solve_seconds", "reference", "worst_difference_C", "at",
                            "times_tolerance", "within_tolerance", "worst_over_time_C", "at_time", "times_tolerance_over_time"])
                for name in self.cand:
                    b, wt = self.distance(name, reference)
                    w.writerow([name, self.cand[name]["elements_solved"], self.cand[name]["solve_seconds"], reference,
                                round(b[2], 4), b[1], round(b[0], 2), b[0] < 1, round(wt[2], 4), wt[1], round(wt[0], 2)])


def indicator_table(setup, mesh, e2, name, path):
    kept = setup.kept(mesh)
    tot = e2.sum()
    rows = []
    for d in sorted(set(mesh["tet_dom"][kept].tolist())):
        m = mesh["tet_dom"] == d
        rows.append(dict(candidate=name, domain=setup.names[d], id=d, elements=int(m.sum()),
                         share_of_squared_error=round(float(e2[m].sum() / tot), 4)))
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    return rows


def adaptive_loop(st, mesh0, prefix, theta, growth, max_elements, passes, reference=None, start_name=None, goal=False):
    """Indicator-driven red/green passes from ``mesh0``.  Stops when the read-outs have changed by less than their
    tolerance over two successive passes, at the element limit, at the pass limit, or when a pass fails.

    Resumable: every pass leaves mesh_<name>.npz, result_<name>.json/.npz and, once the next mesh is marked,
    mesh_<next>.npz.  Called again on the same folder, a pass whose result AND whose next mesh exist is not solved
    again.  A pass that fails (out of memory, solver error) is written to FAILED_<name>.json and ends the loop
    without an error exit; what was done before stays valid.  loop_status.json holds the state after every pass.
    """
    import traceback
    su, log = st.su, st.log
    mesh, names, calm = mesh0, [], 0
    ipath = os.path.join(st.out, "indicator_by_domain.csv")
    state = dict(prefix=prefix, goal=bool(goal), theta=theta, growth=growth, max_elements=max_elements, passes=[],
                 stopped="running", chosen=None)

    def save_state(stopped=None):
        if stopped:
            state["stopped"] = stopped
            log(f"[{prefix}] stopped: {stopped}")
        if calm >= 2 and len(names) >= 3:
            state["chosen"] = names[-3]              # the mesh the last two passes confirm
        with open(os.path.join(st.out, "loop_status.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)

    for k in range(passes + 1):
        name = start_name if (k == 0 and start_name) else f"{prefix}{k}"
        nxt = os.path.join(st.out, f"mesh_{prefix}{k + 1}.npz")
        have = all(os.path.exists(os.path.join(st.out, f"{x}_{name}.{e}")) for x, e in (("result", "json"), ("result", "npz")))
        resumed = have and os.path.exists(nxt) and name in st.series
        res = None
        if not resumed:
            m_in = mesh
            try:
                mesh, res = st.run(name, lambda: m_in, note=("goal-oriented" if goal else "indicator") + f" pass {k}",
                                   keep_fields=True, dual=goal)
            except BaseException as e:                       # includes MemoryError
                with open(os.path.join(st.out, f"FAILED_{name}.json"), "w", encoding="utf-8") as f:
                    json.dump(dict(mesh=name, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(),
                                   time=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)
                save_state(f"pass {name} failed: {type(e).__name__}: {e}")
                break
        else:
            log(f"[{name}] already solved and the next mesh is there: not solved again")
        names.append(name)
        st.write(reference)
        entry = dict(mesh=name, elements_solved=st.cand[name]["elements_solved"], solve_seconds=st.cand[name]["solve_seconds"],
                     peak_memory_GB=st.cand[name].get("peak_memory_GB"))
        if k:
            b, wt = st.distance(name, names[-2])
            log(f"[{name}] change from {names[-2]}: {b[2]:.3f} C at {b[1]} = {b[0]:.2f} x tolerance "
                f"(over time: {wt[2]:.3f} C, {wt[1]}, {wt[0]:.2f} x)")
            entry.update(change_C=round(b[2], 4), at=b[1], times_tolerance=round(b[0], 2),
                         change_over_time_C=round(wt[2], 4))
            calm = calm + 1 if b[0] < 1 else 0
        state["passes"].append(entry)
        if calm >= 2:
            save_state("every judged read-out within tolerance over two successive passes")
            break
        if k == passes:
            save_state("pass limit")
            break
        if resumed:
            mesh = load_mesh(nxt)
            save_state()
            continue
        try:
            e2 = indicator(su, mesh, res, goal)
            rows = indicator_table(su, mesh, e2, name + (" (goal)" if goal else ""), ipath)
            top = sorted(rows, key=lambda r: -r["share_of_squared_error"])[:5]
            log(f"[{name}] indicator: " + ", ".join(f"{r['domain']} {100 * r['share_of_squared_error']:.0f} %" for r in top))
            kept_n = int(su.kept(mesh).sum())
            mark = mark_by_indicator(e2, theta, (growth - 1.0) * kept_n / 9.0)
            new = refine_local(mesh, mark, log)
        except BaseException as e:
            with open(os.path.join(st.out, f"FAILED_marking_after_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(dict(mesh=name, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()), f, indent=1)
            save_state(f"marking after {name} failed: {type(e).__name__}: {e}")
            break
        if int(su.kept(new).sum()) > max_elements:
            save_state(f"the next mesh would have {int(su.kept(new).sum())} elements to solve, above the limit")
            break
        del res
        gc.collect()
        np.savez_compressed(os.path.join(st.out, f"levels_{prefix}{k + 1}.npz"), **level_map(new, su.kept(new)))
        save_mesh(nxt + ".tmp.npz", new)
        os.replace(nxt + ".tmp.npz", nxt)                    # the next mesh appears only when it is complete
        mesh = new
        save_state()
    return names


def study(a, log):
    su = Setup(a.ring, a.config == "exvivo", a.start, log=lambda m: None)
    st = Study(su, a.out, log)
    base = su.start_mesh
    log(f"ring {a.ring}, {a.config}, start {a.start}: {len(base['tets'])} elements, "
        f"{int(su.kept(base).sum())} after removing {sorted(su.names[d] for d in su.dropped)}")
    want = set(a.only) if a.only else None

    def on(n):
        return want is None or n in want

    if on("U0"):
        st.run("U0", lambda: base, "starting mesh")
    if on("E_layers"):
        st.run("E_layers", lambda: refine_local(base, mark_names(su, base, LAYERS), log),
               "sclera, choroid, retina split in eight (two across), closure in the neighbours, rest as U0")
    if on("E_sclera"):
        st.run("E_sclera", lambda: refine_local(base, mark_names(su, base, ("Sclera",)), log), "sclera only split")
    if on("E_chorret"):
        st.run("E_chorret", lambda: refine_local(base, mark_names(su, base, ("Choroid", "Retina")), log),
               "choroid and retina only split")
    if on("E_rest"):
        st.run("E_rest", lambda: refine_local(base, su.kept(base) & ~mark_names(su, base, LAYERS), log),
               "everything except the three layers split (layers mostly one across; closure splits some)")
    if on("M_layers2"):
        st.run("M_layers2", lambda: layers2_mmg(su, base, smooth=False, log=log),
               "MMG recipe: all split, layers + small domains frozen, rest back to starting size; no smoothing")
    if on("M_layers2_smooth"):
        def mk():
            p = os.path.join(a.out, "mesh_M_layers2.npz")
            m = load_mesh(p) if os.path.exists(p) else layers2_mmg(su, base, smooth=False, log=log)
            frozen = su.ids(*LAYERS, *SMALL, "Cloquet")
            m2, hist = mt.smooth_slivers(m, frozen)
            log(f"    smooth_slivers: {hist[0][0]} -> {hist[-1][0]} elements above 160 deg outside the frozen domains")
            return m2
        st.run("M_layers2_smooth", mk, "as M_layers2, then smooth_slivers (interior nodes moved)")
    if on("U1"):
        st.run("U1", lambda: mt.refine_uniform(base), "every element split in eight (the driver's refine=1)")
    st.write("U1" if "U1" in st.cand else None)
    if on("A"):
        if "U0" not in st.series:
            st.run("U0", lambda: base, "starting mesh")
        adaptive_loop(st, base, "A", a.theta, a.growth, a.max_elements, a.passes, "U1" if "U1" in st.cand else None,
                      start_name="U0")
    if on("G"):
        adaptive_loop(st, base, "G", a.theta, a.growth, a.max_elements, a.passes, "U1" if "U1" in st.cand else None,
                      start_name="U0", goal=True)
    if on("AL"):
        mesh0 = refine_local(base, mark_names(su, base, LAYERS), log)
        adaptive_loop(st, mesh0, "AL", a.theta, a.growth, a.max_elements, a.passes, "U1" if "U1" in st.cand else None,
                      start_name="E_layers")
    st.write("U1" if "U1" in st.cand else None)


def selftest(log):
    su = Setup("v1", False, "coarser", log=lambda m: None)
    base = su.start_mesh
    v0 = np.abs(volumes(base))
    for label, mark in (("layers", mark_names(su, base, LAYERS)),
                        ("random 5 %", np.random.default_rng(0).random(len(base["tets"])) < 0.05),
                        ("all", np.ones(len(base["tets"]), bool))):
        m = refine_local(base, mark, log)
        nbad, area, vmin = conformity(m)
        _, area0, _ = conformity(base)
        v = volumes(m)
        dv = max(abs(np.abs(v)[m["tet_dom"] == d].sum() - v0[base["tet_dom"] == d].sum()) / v0[base["tet_dom"] == d].sum()
                 for d in np.unique(base["tet_dom"]))
        from replica.mphread import _neighbours
        up, down = _neighbours(m["tets"], m["tet_dom"], m["tris"])
        x0, x1 = base["points"][base["tris"]], m["points"][m["tris"]]
        ar0 = 0.5 * np.linalg.norm(np.cross(x0[:, 1] - x0[:, 0], x0[:, 2] - x0[:, 0]), axis=1)
        ar1 = 0.5 * np.linalg.norm(np.cross(x1[:, 1] - x1[:, 0], x1[:, 2] - x1[:, 0]), axis=1)
        db = max(abs(ar1[m["tri_bnd"] == b].sum() - ar0[base["tri_bnd"] == b].sum()) / ar0[base["tri_bnd"] == b].sum()
                 for b in np.unique(base["tri_bnd"]))
        m2 = refine_local(m, np.random.default_rng(1).random(len(m["tets"])) < 0.03)
        nbad2, area2, vmin2 = conformity(m2)
        log(f"  {label}: faces shared by more than two elements {nbad} (second pass {nbad2}); outer area "
            f"{area0:.4f} -> {area:.4f} -> {area2:.4f} mm2; least volume {vmin:.2e} / {vmin2:.2e} mm3 (positive = "
            f"not inverted); largest domain volume change {dv:.1e}; largest boundary area change {db:.1e}; "
            f"neighbours inherited == recomputed: {bool((up == m['tri_up']).all() and (down == m['tri_down']).all())}; "
            f"angles {angle_stats(m)['angle_max']} / {angle_stats(m2)['angle_max']} (start {angle_stats(base)['angle_max']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("selftest", "study", "loop", "transfer", "solve", "summary"))
    ap.add_argument("--ring", default="v1")
    ap.add_argument("--config", default="invivo", choices=("invivo", "exvivo"))
    ap.add_argument("--start", default="coarser", choices=list(nr.MESH_FILE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", help="study: candidates to run (U0 U1 E_layers E_sclera E_chorret E_rest "
                                              "M_layers2 M_layers2_smooth A G AL)")
    ap.add_argument("--theta", type=float, default=0.5, help="share of the squared indicator whose elements are split")
    ap.add_argument("--growth", type=float, default=2.0, help="largest growth of the element count per pass")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--max-elements", type=float, default=8e5)
    ap.add_argument("--levels", help="transfer: level map (levels_*.npz of a study)")
    ap.add_argument("--layers-all", type=int, default=0, help="transfer: 1 = every layer element split at least once")
    ap.add_argument("--bulk-offset", type=int, default=1, help="transfer: levels taken off outside the thin layers")
    ap.add_argument("--layer-offset", type=int, default=0,
                    help="transfer: levels taken off in the thin layers (never below one where the map has one or more)")
    ap.add_argument("--mesh-file", help="solve: mesh .npz")
    ap.add_argument("--reference", default="U1", help="summary: candidate the others are judged against")
    ap.add_argument("--from-mesh", help="loop: start from this mesh .npz instead of the COMSOL mesh")
    ap.add_argument("--prefix", default=None, help="loop: name of the passes")
    ap.add_argument("--goal", type=int, default=1, help="loop: 1 = goal-oriented marking (default), 0 = plain indicator")
    a = ap.parse_args()
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    if a.cmd == "selftest":
        selftest(log)
        return
    if a.cmd == "summary":
        st = Study(Setup(a.ring, a.config == "exvivo", a.start, log=lambda m: None), a.out, log)
        st.write(a.reference)
        return
    if a.cmd == "study":
        study(a, log)
    elif a.cmd == "loop":
        su = Setup(a.ring, a.config == "exvivo", a.start, log=lambda m: None)
        st = Study(su, a.out, log)
        adaptive_loop(st, load_mesh(a.from_mesh) if a.from_mesh else su.start_mesh,
                      a.prefix or ("G" if a.goal else "A"), a.theta, a.growth, a.max_elements, a.passes, goal=bool(a.goal))
        st.write()
    elif a.cmd == "transfer":
        su = Setup(a.ring, a.config == "exvivo", a.start, log=lambda m: None)
        lm = dict(np.load(a.levels))
        t0 = time.time()
        m = transfer_levels(su, su.start_mesh, lm, bool(a.layers_all), a.bulk_offset, log, a.layer_offset)
        kept = su.kept(m)
        save_mesh(a.out, m)
        st = angle_stats(m, kept, su.ids(*LAYERS))
        log(f"{a.out}: {len(m['tets'])} elements, {int(kept.sum())} to solve, "
            f"{int(np.isin(m['tet_dom'], su.ids(*LAYERS)).sum())} in the layers; {st}; {time.time() - t0:.0f} s; "
            f"{os.path.getsize(a.out) / 2 ** 20:.1f} MB")
        return
    elif a.cmd == "solve":
        su = Setup(a.ring, a.config == "exvivo", a.start, log=lambda m: None)
        st = Study(su, a.out, log)
        m = load_mesh(a.mesh_file)
        st.run(os.path.splitext(os.path.basename(a.mesh_file))[0], lambda: m)
        st.write()
    if a.out:
        with open(os.path.join(a.out, f"log_{a.cmd}_{time.strftime('%H%M%S')}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
