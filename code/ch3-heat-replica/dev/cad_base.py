"""A mesh made from the STEP files (dev/cad_mesh.py), dressed as the problem that the driver reads from a COMSOL file.

    base = cad_base.base_from_cad(z, ref, log)        # z: dict of the CAD mesh .npz, ref: Problem of model.build

new_runs.transform() finds its faces from the domains on either side of each triangle: ring | module layer = cooling
face, tissue | air = surfaces in air, and so on.  The CAD files hold neither air nor module layer, so here the outer
faces are classified from the geometry (the rules of cad_mesh.problem, which were checked against the COMSOL-mesh
runs there) and labelled as if those domains were present:

    cooling face   ring faces, flat and normal to z, at the level with the largest such area in the upper half of
                   the ring                                                     -> other side "TEC Layer" (no elements)
    in air         in vivo: every other outer face that is not body shell or symmetry plane;
                   ex vivo: the rest of the ring, and globe faces above the lowest point of the ring-globe contact -> other side "Air"
    body shell     in vivo: outer faces of fat and muscles, and the eyelid's faces on the side wall of the orbit
                                                                               -> outside, boundary number SHELL
    oil            ex vivo: globe faces below that contact level                -> outside (transform's rule)
    symmetry       outer faces with all nodes at y = 0                          -> outside, nothing applied
    evaporation    cornea faces in air (in vivo)

Materials, perfusion, metabolic heat and probes are taken from ``ref`` by tissue name, so every setting of the
driver (ring material, property changes, perfusion factors, soft shell, oil and air coefficients ...) acts on a CAD
mesh exactly as on a COMSOL mesh.  Nothing in replica/ is used beyond the Problem class.
"""
import numpy as np

AIR, TEC, SHELL = 9001, 9002, 9003          # domain numbers of the two absent domains; boundary number of the shell
ORBIT = ("Fat", "Eyelids", "Extraocular muscles")
MODEL_NAME = {"Acqueous": "Aqueous Humor", "Vitreous": "Vitreous Humor", "Eye Lid": "Eyelids",
              "Extraocular Muscle": "Extraocular muscles", "Extraocular Muscles": "Extraocular muscles",
              "Thin Layer": "Tear film"}
FALLBACK = {"Tear film": "Aqueous Humor"}    # a tissue the reference model file does not have: properties of this one


def model_name(n):
    n = str(n)
    if n.startswith("Ring"):
        return "Ring"
    if n.startswith("Fat"):
        return "Fat"
    return MODEL_NAME.get(n, n)


def base_from_cad(z, ref, log=print):
    from replica.model import Problem
    P, T, dom = z["points"], z["tets"], z["tet_dom"]
    names = dict(zip(np.asarray(z["domain_ids"]).tolist(), np.asarray(z["domain_names"]).tolist()))
    model_of = {int(d): model_name(n) for d, n in names.items()}
    in_vivo = any(n in ORBIT for n in model_of.values())
    ref_id = {}
    for d, n in ref.domain_names.items():
        ref_id.setdefault(n, d)

    def rid(n):
        if n in ref_id:
            return ref_id[n]
        if n in FALLBACK and FALLBACK[n] in ref_id:
            log(f"  CAD tissue '{n}' is not in the reference model file: properties of '{FALLBACK[n]}'")
            return ref_id[FALLBACK[n]]
        raise RuntimeError(f"CAD tissue '{n}' has no counterpart in the reference model file ({sorted(ref_id)})")

    src = {d: rid(n) for d, n in model_of.items()}
    tris, up, dn = z["tris"], np.asarray(z["tri_up"]).astype(np.int64), np.asarray(z["tri_down"]).astype(np.int64)
    outer = (up == 0) != (dn == 0)
    od = np.where(up == 0, dn, up)                       # the domain of an outer face
    # every element face that belongs to one element only must be among the stored outer triangles
    n = len(P)
    fl = np.concatenate([T[:, c] for c in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])]).astype(np.int64)
    fs = np.sort(fl, axis=1)
    _, cnt = np.unique((fs[:, 0] * n + fs[:, 1]) * n + fs[:, 2], return_counts=True)
    if int((cnt == 1).sum()) != int(outer.sum()):
        raise RuntimeError(f"CAD mesh: {int((cnt == 1).sum())} outer element faces but {int(outer.sum())} outer triangles stored")

    X = P[tris]
    on_plane = outer & (np.abs(X[:, :, 1]) < 1e-9).all(1)
    ring = [d for d, m in model_of.items() if m == "Ring"]
    rn = np.unique(T[np.isin(dom, ring)])
    zbot = P[rn, 2].min()
    is_ring = outer & np.isin(od, ring)
    nrm = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    ar = 0.5 * np.linalg.norm(nrm, axis=1)
    flat = is_ring & ~on_plane & (np.abs(nrm[:, 2]) > 0.9999 * 2 * ar) & (np.ptp(X[:, :, 2], axis=1) < 1e-9)
    zl = np.round(X[flat][:, 0, 2] * 1e6).astype(np.int64)
    lv, inv = np.unique(zl, return_inverse=True)
    med = np.median(P[rn, 2])
    tot = np.array([ar[flat][inv == i].sum() if (zbot < l * 1e-6 - 1e-5 and l * 1e-6 >= med) else 0.0 for i, l in enumerate(lv)])
    if not len(tot) or tot.max() <= 0:
        raise RuntimeError("CAD mesh: the ring has no flat face normal to z in its upper half: cooling face not found")
    ztop = lv[int(tot.argmax())] * 1e-6
    f_cool = flat & (np.abs(X[:, 0, 2] - ztop) < 1e-6)
    cen = X.mean(1)
    rad = np.hypot(cen[:, 0], cen[:, 1])
    orbit_ids = [d for d, m in model_of.items() if m in ORBIT]
    lid = outer & np.isin(od, [d for d, m in model_of.items() if m == "Eyelids"])
    cornea = outer & np.isin(od, [d for d, m in model_of.items() if m == "Cornea"])
    f_body = np.zeros(len(tris), bool)
    if in_vivo:
        on = np.unique(T[np.isin(dom, orbit_ids)])
        r_wall = np.hypot(P[on, 0], P[on, 1]).max()
        f_body = outer & ~on_plane & np.isin(od, orbit_ids) & (~lid | (rad > r_wall - 5e-5))
        f_air = outer & ~on_plane & ~f_cool & ~f_body
    else:
        # (2026-10-19) the level is the lowest node the ring shares with the globe, not the ring's lowest node: the
        # rounded rims of v2 and v3 reach 0.114 mm below their contact, which put 4.37 mm2 of sclera in air instead of
        # oil (found in the 10-09 anterior-sclera check; v1 unchanged: its lowest ring node is on the contact)
        contact = np.intersect1d(rn, np.unique(T[~np.isin(dom, ring)]))
        z_oil = P[contact, 2].min() if len(contact) else zbot
        f_air = outer & ~on_plane & ((is_ring & ~f_cool) | (~is_ring & (cen[:, 2] > z_oil)))

    area = lambda m: float(ar[m].sum() * 1e6)
    log(f"  CAD mesh: {len(T)} elements, {len(P)} nodes, {'in vivo' if in_vivo else 'ex vivo'}; faces (mm2, half model): "
        f"cooling {area(f_cool):.1f}, air on ring {area(f_air & is_ring):.1f}, air on tissue {area(f_air & ~is_ring):.1f}, "
        f"body shell {area(f_body):.1f}, symmetry {area(on_plane):.1f}, "
        f"other outer (oil if ex vivo) {area(outer & ~on_plane & ~f_cool & ~f_air & ~f_body):.1f}")

    new_up = np.where(outer, 0, up)
    new_dn = np.where(outer, od, dn)
    new_up[f_air] = AIR
    new_up[f_cool] = TEC
    bnd = np.asarray(z["tri_bnd"]).astype(np.int64).copy()
    bnd[bnd >= 9000] += 100000                           # keep the special number free (not expected to occur)
    bnd[f_body] = SHELL

    dnames = dict(model_of)
    dnames[AIR], dnames[TEC] = "Air", "TEC Layer"
    k = {d: ref.k[src[d]] for d in model_of}
    rho_cp = {d: ref.rho_cp[src[d]] for d in model_of}
    perfusion = {d: ref.perfusion[src[d]] for d in model_of if src[d] in ref.perfusion}
    t_blood = {d: ref.t_blood[src[d]] for d in perfusion}
    q_met = {d: ref.q_met[src[d]] for d in perfusion}
    dirichlet, source = [], []
    if in_vivo:
        dirichlet.append(dict(name="Temperature Body", nodes=np.unique(tris[f_body]), boundaries=[SHELL],
                              value=(lambda t: np.nan), steps=["start", "cool", "rewarm"]))
        if (f_air & cornea).any():
            source.append(dict(name="Tear Evaporation Cooling", tris=tris[f_air & cornea], q=0.0))
    probes = {}
    for nm, p in ref.probes.items():
        if p["kind"] == "point":
            probes[nm] = p
        else:
            want = {ref.domain_names[d] for d in p["domains"]}
            ds = sorted(d for d, m in model_of.items() if m in want)
            if ds:
                probes[nm] = dict(kind=p["kind"], domains=ds)
    return Problem(points=P, tets=T, dom=dom, n_original=len(P), k=k, rho_cp=rho_cp,
                   variable={d for d in model_of if src[d] in ref.variable}, perfusion=perfusion, t_blood=t_blood,
                   q_met=q_met, t_init=np.full(len(P), np.nan), dirichlet=dirichlet, radiation=[], surface_source=source,
                   probes=probes, domain_names=dnames, tris=tris, tri_bnd=bnd, tri_up=new_up, tri_down=new_dn,
                   flow=None, convective=[], notes=list(ref.notes) + ["mesh from the STEP file (dev/cad_base.py)"],
                   params=dict(ref.params), selections={})
