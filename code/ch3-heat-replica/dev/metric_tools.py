"""Anisotropic size maps for the mesher: keep the shape of the elements COMSOL made, change only their size.

Why: COMSOL's meshes are flat in the thin layers (choroid, retina, sclera): elements there are short across
the layer and long along it.  replica.mesher describes a mesh by one length per node, so asking MMG for "the
same size" turns those flat elements into regular ones of the short length: 90 thousand elements became 595
thousand in a test on the coarser mesh (2026-10-08), and the self-refining loop of mesh_study.py asked for 11
million on its second pass.  Here the size map is a metric tensor per node, taken from the mesh itself
(``implied_metric``), so "the same size" really is the same mesh density and the same flatness, and refining
or coarsening is a scaling of that tensor.

A metric M at a node is a symmetric positive matrix; an edge e has the wanted length when e.M.e = 1.  MMG's
order of the six numbers is (m11, m12, m13, m22, m23, m33).  Lengths here are in mm (as replica.mesher.adapt).

Kept in dev/ while the validation re-run is going; to be merged into replica/mesher.py afterwards.
"""
import numpy as np

EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
IDX = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _to6(M):
    return np.stack([M[:, i, j] for i, j in IDX], axis=1)


def _to33(m6):
    M = np.empty((len(m6), 3, 3))
    for k, (i, j) in enumerate(IDX):
        M[:, i, j] = M[:, j, i] = m6[:, k]
    return M


def element_metric(points_mm, tets):
    """The metric in which each element is regular with unit edges: (n_elements, 3, 3)."""
    x = points_mm[tets]
    A = np.empty((len(tets), 6, 6))
    for r, (a, b) in enumerate(EDGES):
        e = x[:, b] - x[:, a]
        A[:, r] = np.stack([e[:, 0] ** 2, 2 * e[:, 0] * e[:, 1], 2 * e[:, 0] * e[:, 2],
                            e[:, 1] ** 2, 2 * e[:, 1] * e[:, 2], e[:, 2] ** 2], axis=1)
    m6 = np.linalg.solve(A, np.ones((len(tets), 6, 1)))[:, :, 0]
    return _to33(m6)


def implied_metric(points_mm, tets, h_limits=(1e-3, 50.0)):
    """Nodal metric of a mesh: volume-weighted log-average of the metrics of the elements at each node.

    Returns (n_nodes, 6) in MMG's order.  Element metrics are made positive first (eigenvalues clipped so
    that sizes stay within ``h_limits`` mm).
    """
    M = element_metric(points_mm, tets)
    w, V = np.linalg.eigh(M)
    lo, hi = 1 / h_limits[1] ** 2, 1 / h_limits[0] ** 2
    w = np.clip(w, lo, hi)
    logM = np.einsum("eij,ej,ekj->eik", V, np.log(w), V)
    x = points_mm[tets]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6
    n = len(points_mm)
    acc = np.zeros((n, 3, 3))
    wsum = np.bincount(tets.ravel(), np.repeat(vol, 4), n)
    for i in range(3):
        for j in range(i, 3):
            a = np.bincount(tets.ravel(), np.repeat(vol * logM[:, i, j], 4), n)
            acc[:, i, j] = acc[:, j, i] = a
    acc /= np.maximum(wsum, 1e-300)[:, None, None]
    w, V = np.linalg.eigh(acc)
    return _to6(np.einsum("eij,ej,ekj->eik", V, np.exp(w), V))


def scale(metric6, size_factor):
    """Multiply every length asked by the metric by ``size_factor`` (per node or one number); < 1 refines."""
    f = np.asarray(size_factor, float)
    return metric6 / (f ** 2)[..., None] if f.ndim else metric6 / f ** 2


def sizes(metric6):
    """Smallest and largest length asked at each node [mm]."""
    w = np.linalg.eigvalsh(_to33(metric6))
    return 1 / np.sqrt(w[:, 2]), 1 / np.sqrt(w[:, 0])


def adapt(mesh, metric6, gradation=1.3, surface_tol_mm=0.02, verbose=-1, **kw):
    """Adapt ``mesh`` (dict as from the .mph reader, lengths in m) to a nodal metric tensor; returns a new mesh dict."""
    import mmgpy
    from replica.mphread import _neighbours
    P = np.ascontiguousarray(mesh["points"] * 1e3)
    mm = mmgpy.MmgMesh3D()
    mm.set_mesh_size(vertices=len(P), tetrahedra=len(mesh["tets"]), triangles=len(mesh["tris"]))
    mm.set_vertices(P)
    mm.set_tetrahedra(np.ascontiguousarray(mesh["tets"].astype(np.int32)), mesh["tet_dom"].astype(np.int64))
    mm.set_triangles(np.ascontiguousarray(mesh["tris"].astype(np.int32)), mesh["tri_bnd"].astype(np.int64))
    mm.set_field("tensor", np.ascontiguousarray(metric6))
    mm.remesh(hgrad=gradation, hausd=surface_tol_mm, verbose=verbose, **kw)
    v = mm.get_vertices()
    tets, dom = mm.get_elements_with_refs()
    tris, bnd = mm.get_triangles_with_refs()
    keep = np.isin(bnd, np.unique(mesh["tri_bnd"]))
    tris, bnd = tris[keep], bnd[keep]
    x = v[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets = tets.copy()
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    dom = dom.astype(np.int32)
    up, down = _neighbours(tets, dom, tris)
    return dict(points=v * 1e-3, tets=tets.astype(np.int32), tet_dom=dom, tris=tris.astype(np.int32),
                tri_bnd=bnd.astype(np.int32), tri_up=up, tri_down=down, version=0)


def metric_quality(points_mm, tets, metric6):
    """Quality of each element measured in the metric it was asked to follow (1 = exactly as asked, 0 = flat).

    A flat element in a thin layer scores high here if the metric asked for that flatness; the ordinary
    quality (replica.mesher.quality) would mark it down.
    """
    M = _to33(metric6)[tets].mean(1)
    x = points_mm[tets]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6
    vol_m = vol * np.sqrt(np.maximum(np.linalg.det(M), 0))
    L2 = 0.0
    for a, b in EDGES:
        e = x[:, b] - x[:, a]
        L2 = L2 + np.einsum("ei,eij,ej->e", e, M, e)
    return 12 * (3 * vol_m) ** (2 / 3) / np.maximum(L2, 1e-300)        # 1 for the regular element


def drift(old, new):
    """How much a remesh changed the geometry: per-domain volume and per-boundary area, old against new."""
    out = {"domains": {}, "boundaries": {}}
    for m, key in ((old, 0), (new, 1)):
        x = m["points"][m["tets"]] * 1e3
        vol = np.abs(np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])) / 6
        for d in np.unique(m["tet_dom"]):
            out["domains"].setdefault(int(d), [0.0, 0.0])[key] = float(vol[m["tet_dom"] == d].sum())
        t = m["points"][m["tris"]] * 1e3
        area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        for b in np.unique(m["tri_bnd"]):
            out["boundaries"].setdefault(int(b), [0.0, 0.0])[key] = float(area[m["tri_bnd"] == b].sum())
    return out


# ----------------------------------------------------------------------------- exact refinement and freezing
def refine_uniform(mesh):
    """Split every tetrahedron into eight (and every boundary triangle into four); returns a new mesh dict.

    New nodes are the edge midpoints, so the geometry is exactly that of the starting mesh (its flat facets),
    and each child has the shape of its parent or close to it (the inner octahedron is cut along its
    shortest diagonal).  Domain and boundary numbers are inherited.  This is the systematic refinement
    (ratio 2) that a grid-convergence index assumes, and the way to put two elements across a layer that
    has one, without the mesher touching the layer.
    """
    from replica.mphread import _neighbours
    P, T = mesh["points"], mesh["tets"].astype(np.int64)
    n = len(P)
    e = np.concatenate([T[:, [a, b]] for a, b in EDGES])
    e.sort(axis=1)
    key = e[:, 0] * n + e[:, 1]
    uk, inv = np.unique(key, return_inverse=True)
    mid = n + inv.reshape(6, -1).T                                  # (n_tets, 6): midpoint node of each edge
    Pn = np.vstack([P, 0.5 * (P[uk // n] + P[uk % n])])
    a, b, c, d = T.T
    ab, ac, ad, bc, bd, cd = mid.T
    corner = [np.stack(x, 1) for x in ((a, ab, ac, ad), (ab, b, bc, bd), (ac, bc, c, cd), (ad, bd, cd, d))]
    # inner octahedron: three possible diagonals (ab-cd, ac-bd, ad-bc); use the shortest
    L = np.stack([np.linalg.norm(Pn[ab] - Pn[cd], axis=1), np.linalg.norm(Pn[ac] - Pn[bd], axis=1),
                  np.linalg.norm(Pn[ad] - Pn[bc], axis=1)], 1)
    pick = L.argmin(1)
    inner = np.empty((len(T), 4, 4), np.int64)
    opts = (((ab, cd, ac, ad), (ab, cd, ad, bd), (ab, cd, bd, bc), (ab, cd, bc, ac)),
            ((ac, bd, ab, ad), (ac, bd, ad, cd), (ac, bd, cd, bc), (ac, bd, bc, ab)),
            ((ad, bc, ab, ac), (ad, bc, ac, cd), (ad, bc, cd, bd), (ad, bc, bd, ab)))
    for k, four in enumerate(opts):
        m = pick == k
        for j, q in enumerate(four):
            inner[m, j] = np.stack(q, 1)[m]
    tets = np.concatenate(corner + [inner[:, j] for j in range(4)])
    dom = np.tile(mesh["tet_dom"], 8)
    x = Pn[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    tr = mesh["tris"].astype(np.int64)

    def midnode(u, v):
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        return n + np.searchsorted(uk, lo * n + hi)
    p, q, r = tr.T
    pq, qr, pr = midnode(p, q), midnode(q, r), midnode(p, r)
    tris = np.concatenate([np.stack(x, 1) for x in ((p, pq, pr), (pq, q, qr), (pr, qr, r), (pq, qr, pr))])
    bnd = np.tile(mesh["tri_bnd"], 4)
    tets, tris = tets.astype(np.int32), tris.astype(np.int32)
    up, down = _neighbours(tets, dom.astype(np.int32), tris)
    return dict(points=Pn, tets=tets, tet_dom=dom.astype(np.int32), tris=tris, tri_bnd=bnd.astype(np.int32),
                tri_up=up, tri_down=down, version=0)


def adapt_frozen(mesh, metric6, frozen_domains, gradation=-1, surface_tol_mm=0.2, verbose=-1, **kw):
    """As ``adapt``, but the elements of ``frozen_domains`` and every face that bounds them are left untouched.

    Used to keep the thin layers exactly as they are while the rest of the mesh is refined or coarsened.
    """
    import mmgpy
    from replica.mphread import _neighbours
    P = np.ascontiguousarray(mesh["points"] * 1e3)
    mm = mmgpy.MmgMesh3D()
    mm.set_mesh_size(vertices=len(P), tetrahedra=len(mesh["tets"]), triangles=len(mesh["tris"]))
    mm.set_vertices(P)
    mm.set_tetrahedra(np.ascontiguousarray(mesh["tets"].astype(np.int32)), mesh["tet_dom"].astype(np.int64))
    mm.set_triangles(np.ascontiguousarray(mesh["tris"].astype(np.int32)), mesh["tri_bnd"].astype(np.int64))
    fz = np.isin(mesh["tet_dom"], list(frozen_domains))
    ft = np.isin(mesh["tri_up"], list(frozen_domains)) | np.isin(mesh["tri_down"], list(frozen_domains))
    mm.set_required_tetrahedra(np.where(fz)[0].astype(np.int32))
    mm.set_required_triangles(np.where(ft)[0].astype(np.int32))
    mm.set_required_vertices(np.unique(mesh["tets"][fz]).astype(np.int32))
    mm.set_field("tensor", np.ascontiguousarray(metric6))
    mm.remesh(hgrad=gradation, hausd=surface_tol_mm, verbose=verbose, **kw)
    v = mm.get_vertices()
    tets, dom = mm.get_elements_with_refs()
    tris, bnd = mm.get_triangles_with_refs()
    keep = np.isin(bnd, np.unique(mesh["tri_bnd"]))
    tris, bnd = tris[keep], bnd[keep]
    x = v[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets = tets.copy()
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    dom = dom.astype(np.int32)
    up, down = _neighbours(tets, dom, tris)
    return dict(points=v * 1e-3, tets=tets.astype(np.int32), tet_dom=dom, tris=tris.astype(np.int32),
                tri_bnd=bnd.astype(np.int32), tri_up=up, tri_down=down, version=0)


def max_dihedral(points, tets):
    """Largest of the six dihedral angles of each tetrahedron [degrees] (near 180 = sliver)."""
    x = points[tets]
    faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    n = []
    for k, (a, b, c) in enumerate(faces):
        v = np.cross(x[:, b] - x[:, a], x[:, c] - x[:, a])
        v *= -np.sign(np.einsum("ij,ij->i", v, x[:, k] - x[:, a]))[:, None]
        n.append(v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None])
    best = np.zeros(len(tets))
    for i in range(4):
        for j in range(i + 1, 4):
            best = np.maximum(best, np.degrees(np.arccos(np.clip(-np.einsum("ij,ij->i", n[i], n[j]), -1, 1))))
    return best


def repair(mesh, frozen_domains, limit=160.0, shrink=0.6, passes=4, log=None, **kw):
    """Remove slivers outside the frozen domains: remesh with a smaller size asked at the nodes of every
    element whose largest dihedral angle exceeds ``limit``; repeat until none is left or ``passes`` is reached.
    A pass that makes things worse is discarded.  Returns (mesh, history of (elements, slivers, worst angle))."""
    fz = list(frozen_domains)

    def state(m):
        a = max_dihedral(m["points"], m["tets"])
        free = ~np.isin(m["tet_dom"], fz)
        bad = free & (a > limit)
        return a, bad, (len(m["tets"]), int(bad.sum()), float(a[free].max()))
    ang, bad, s = state(mesh)
    hist = [s]
    for _ in range(passes):
        if not bad.any():
            break
        f = np.ones(len(mesh["points"]))
        f[np.unique(mesh["tets"][bad])] = shrink
        M = implied_metric(mesh["points"] * 1e3, mesh["tets"])
        try:
            new = adapt_frozen(mesh, scale(M, f), fz, **kw)
        except Exception as e:                        # keep what we have
            if log:
                log(f"repair pass failed: {e!r}"[:160])
            break
        ang2, bad2, s2 = state(new)
        if log:
            log(f"repair: {s[0]} -> {s2[0]} elements, slivers above {limit:g} deg {s[1]} -> {s2[1]}, worst angle {s[2]:.1f} -> {s2[2]:.1f}")
        if (s2[1], s2[2]) >= (s[1], s[2]):
            break
        mesh, ang, bad, s = new, ang2, bad2, s2
        hist.append(s)
    return mesh, hist


def smooth_slivers(mesh, frozen_domains, limit=160.0, rounds=8, log=None):
    """Remove slivers by moving interior nodes: no element is added, removed or reconnected.

    For every element outside the frozen domains whose largest dihedral angle exceeds ``limit``, each of its
    nodes that lies on no interface or boundary face (so moving it changes no geometry) is tried at a few
    new positions: towards the centre of its neighbours, and away from the plane of the sliver.  A move is
    kept only if it lowers the largest dihedral angle among all the elements around that node and leaves
    every one of them with positive volume.  Returns (mesh with new points, history).
    """
    P = mesh["points"].copy()
    T = mesh["tets"]
    fz = np.isin(mesh["tet_dom"], list(frozen_domains))
    fixed = np.zeros(len(P), bool)
    fixed[np.unique(mesh["tris"])] = True                          # nodes on any interface or outer face
    fixed[np.unique(T[fz])] = True
    order = np.argsort(T.ravel(), kind="stable")
    owner = order // 4
    start = np.searchsorted(T.ravel()[order], np.arange(len(P) + 1))
    faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))

    def star_score(elems, pts):
        t = T[elems]
        x = pts[t]
        vol = np.einsum("ij,ij->i", np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0])
        return max_dihedral(pts, t).max(), vol.min()

    hist = []
    for r in range(rounds):
        ang = max_dihedral(P, T)
        bad = np.where(~fz & (ang > limit))[0]
        hist.append((int(len(bad)), float(ang[~fz].max())))
        if log:
            log(f"sliver smoothing round {r}: {len(bad)} elements above {limit:g} deg, worst {ang[~fz].max():.1f}")
        if not len(bad):
            break
        moved = 0
        for e in bad[np.argsort(-ang[bad])]:
            for k in range(4):
                nd = T[e, k]
                if fixed[nd]:
                    continue
                elems = owner[start[nd]:start[nd + 1]]
                s0, v0 = star_score(elems, P)
                ring = np.unique(T[elems])
                ring = ring[ring != nd]
                centre = P[ring].mean(0)
                h = np.linalg.norm(P[ring] - P[nd], axis=1).mean()
                a, b, c = (T[e, j] for j in faces[k])
                nrm = np.cross(P[b] - P[a], P[c] - P[a])
                nrm /= max(np.linalg.norm(nrm), 1e-300)
                if np.dot(P[nd] - P[a], nrm) < 0:
                    nrm = -nrm
                p0, best, best_p = P[nd].copy(), s0, None
                trials = [p0 + al * (centre - p0) for al in (0.3, 0.6, 1.0)] + [p0 + be * h * nrm for be in (0.15, 0.3, 0.5)]
                trials += [p0 + 0.5 * (centre - p0) + be * h * nrm for be in (0.15, 0.3)]
                for q in trials:
                    P[nd] = q
                    s, v = star_score(elems, P)
                    if v > 0.05 * max(v0, 1e-30) and v > 0 and s < best - 1e-9:
                        best, best_p = s, q.copy()
                P[nd] = p0 if best_p is None else best_p
                moved += best_p is not None
        if log:
            log(f"   moved {moved} nodes")
        if not moved:
            break
    out = dict(mesh)
    out["points"] = P
    return out, hist
