"""Translate one component of the COMSOL model into a heat-transfer problem.

Everything is taken from the .mph archive: mesh, domain and boundary
selections, material properties, physics features and their settings.  Nothing
about the eye model is hard-coded here.

What is reproduced (Heat Transfer in Solids and Fluids interface):
  * conduction with the domain materials, including temperature-dependent
    air and ring properties;
  * Pennes perfusion and metabolic heat in the Biological Tissue domains;
  * fixed-temperature boundaries, with the TEC face following the same
    smoothed ramp, and the study-step switch between the two TEC conditions;
  * surface-to-ambient radiation and the tear-evaporation boundary heat source;
  * thermal insulation on interior boundaries as a temperature slit;
  * symmetry plane (natural condition).

What is not reproduced: the Laminar Flow interface.  Aqueous humour, vitreous
and air are treated as stagnant conductors (see README for the consequence).
"""
from dataclasses import dataclass, field

import numpy as np

from .expr import Scope, smooth_step

DOMAIN_MODELS = ("SolidHeatTransferModel", "FluidHeatTransferModel", "BiologicalTissue")


@dataclass
class Problem:
    points: np.ndarray
    tets: np.ndarray
    dom: np.ndarray
    n_original: int                    # nodes before slit duplication
    k: dict                            # domain -> callable(T) or float
    rho_cp: dict                       # domain -> callable(T) or float
    variable: set                      # domains whose k or rho*cp depend on T
    perfusion: dict                    # domain -> rho_b*c_b*omega_b  [W/(m^3 K)]
    t_blood: dict                      # domain -> arterial temperature [K]
    q_met: dict                        # domain -> metabolic heat [W/m^3]
    t_init: np.ndarray
    dirichlet: list                    # dicts: name, nodes, value(t), steps
    radiation: list                    # dicts: name, tris, eps, t_amb
    surface_source: list               # dicts: name, tris, q [W/m^2]
    probes: dict
    domain_names: dict
    tris: np.ndarray = None
    tri_bnd: np.ndarray = None
    tri_up: np.ndarray = None
    tri_down: np.ndarray = None
    flow: dict = None                  # fluid regions and gravity (None if the model has no flow coupling)
    convective: list = field(default_factory=list)   # dicts: name, tris, h(T_surface) [W/(m^2 K)], t_ext
    notes: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    selections: dict = field(default_factory=dict)


def _slit(mesh, tets, dom, boundaries):
    """Duplicate nodes so the temperature is discontinuous across interior boundaries.

    Elements of the "up" domain of each slit boundary get their own copy of the
    nodes on it.  Nodes on the rim of the slit stay shared unless the rim lies
    on the outer surface of the model.
    """
    tris, bnd, up, down = mesh["tris"], mesh["tri_bnd"], mesh["tri_up"], mesh["tri_down"]
    sel = np.isin(bnd, boundaries) & (up > 0) & (down > 0)
    if not sel.any():
        return tets, np.zeros(0, int)
    st = tris[sel]
    e = np.sort(np.concatenate([st[:, [0, 1]], st[:, [1, 2]], st[:, [0, 2]]]), axis=1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    rim = ue[cnt == 1]
    ext = tris[(up == 0) | (down == 0)]
    ee = np.unique(np.sort(np.concatenate([ext[:, [0, 1]], ext[:, [1, 2]], ext[:, [0, 2]]]), axis=1), axis=0)
    n = mesh["points"].shape[0]
    on_ext = np.isin(rim[:, 0].astype(np.int64) * n + rim[:, 1], ee[:, 0].astype(np.int64) * n + ee[:, 1])
    keep_shared = np.unique(rim[~on_ext])
    dup = np.setdiff1d(np.unique(st), keep_shared)
    new_id = np.arange(n)
    new_id[dup] = n + np.arange(len(dup))
    tets = tets.copy()
    side = np.isin(dom, np.unique(up[sel]))
    tets[side] = new_id[tets[side]]
    return tets, dup


def _h_horizontal_plate_up(L, t_ext):
    """External natural convection on the upper side of a horizontal plate, in air.

    Nu = 0.54 Ra^(1/4) (Ra <= 1e7) or 0.15 Ra^(1/3) when the surface is warmer
    than the air, 0.27 Ra^(1/4) when it is colder; air properties at the film
    temperature.  These are the textbook correlations COMSOL's library uses.
    """
    def h(Ts):
        Tf = 0.5 * (Ts + t_ext)
        k = -0.00227583562 + 1.15480022e-4 * Tf - 7.90252856e-8 * Tf ** 2 + 4.11702505e-11 * Tf ** 3 - 7.43864331e-15 * Tf ** 4
        mu = -8.38278e-7 + 8.35717342e-8 * Tf - 7.69429583e-11 * Tf ** 2 + 4.6437266e-14 * Tf ** 3 - 1.06585607e-17 * Tf ** 4
        rho = 101325.0 * 0.02897 / 8.31446261815324 / Tf
        cp = 1047.63657 - 0.372589265 * Tf + 9.45304214e-4 * Tf ** 2 - 6.02409443e-7 * Tf ** 3 + 1.2858961e-10 * Tf ** 4
        Ra = 9.80665 * (1.0 / Tf) * np.abs(Ts - t_ext) * L ** 3 * rho ** 2 * cp / (mu * k)
        Nu = np.where(Ts > t_ext, np.where(Ra <= 1e7, 0.54 * Ra ** 0.25, 0.15 * Ra ** (1 / 3)), 0.27 * Ra ** 0.25)
        return k / L * Nu
    return h


def _flow(mph, heat, scope, mats, ref_temp, p_abs, notes):
    """Read the Laminar Flow interface coupled to the heat interface: one region per fluid domain."""
    cpl = mph.flow_coupling(heat)
    if cpl is None:
        return None
    spf = cpl["fluid"]
    _, fl_dom, _ = mph.selection_of(mph.physics(spf))
    gvec = (0.0, 0.0, 0.0)
    inlets, outlets, sym = [], [], []
    for f in mph.features(spf):
        if f["op"] == "Gravity":
            gvec = tuple(float(scope.eval(e)) for e in f["par"]["g"])
        elif f["op"] == "InletBoundary":
            if f["par"].get("BoundaryCondition", ["Velocity"])[0] == "MassFlow":
                inlets.append((f["entities"], ("mass_flow", float(scope.eval(f["par"]["mfr"][0])))))
            else:
                inlets.append((f["entities"], float(scope.eval(f["par"]["U0in"][0]))))
        elif f["op"] == "OutletBoundary":
            outlets += f["entities"]
        elif f["op"] == "Symmetry":
            sym += f["entities"]
    regions = []
    for m in mats:
        if m["dim"] != 3:
            continue
        ms = scope.child(m["funcs"])
        for d in m["entities"]:
            if d not in (fl_dom or []):
                continue
            Tr = ref_temp[d]
            rho = lambda T, ms=ms, e=m["props"]["density"]: float(ms.eval(e, T=T, pA=p_abs))
            rho_ref = rho(Tr)
            beta = -(rho(Tr + 0.5) - rho(Tr - 0.5)) / rho_ref if cpl["boussinesq"] else 0.0
            if abs(beta) < 1e-12:
                beta = 0.0
            e_mu = m["props"]["dynamicviscosity"]
            mu = (lambda T, ms=ms, e=e_mu: ms.eval(e, T=T, pA=p_abs)) if ms.uses(e_mu, "T") else float(ms.eval(e_mu))
            regions.append(dict(domains=[d], name=m["name"], rho_ref=rho_ref, beta=beta, t_ref=Tr, mu=mu,
                                inlets=inlets, outlets=outlets, symmetry=sym))
    return dict(physics=spf, gravity=gvec, regions=regions)


def build(mph, component="comp1", heat="ht", geom="geom1", insulation_slit=True,
          overrides=None, switch_index=0, study=None, extra_probes=None, mesh_tag=None, mesh=None):
    """overrides: parameter values replacing the stored ones (a parametric-sweep case);
    switch_index: member of a Material Switch to use (a material-sweep case)."""
    par = mph.parameters()
    par.update(overrides or {})
    scope = Scope(par)
    if mesh is None:                   # default: the mesh stored in the .mph; else one of our own (replica.mesher)
        mesh = mph.mesh(mph.mesh_sequences()[mesh_tag]["resource"] if mesh_tag
                        else mph.mesh_resources()[geom]["resource"])
    pts, tets, dom = mesh["points"], mesh["tets"], mesh["tet_dom"]
    domains = sorted(set(dom.tolist()))
    notes = []

    # ---- ramp function used by the time-dependent TEC temperature
    funcs = {}
    for f in mph.root.iter("FunctionFeature"):
        if f.get("op") == "Step" and (f.findtext("storedParent") or "").endswith(component):
            pv = {p.get("name"): p.get("value") for p in f.findall("propertyValue")}
            funcs[pv.get("p:funcname", f.get("tag"))] = smooth_step(float(pv.get("p:location", 0.5)),
                                                                   float(pv.get("p:smooth", 1)))
    scope = Scope(par, funcs)

    # ---- physics features, in tree order (later ones override)
    feats = mph.features(heat)
    model_of, ref_temp, bio = {}, {}, {}
    t_init = np.full(pts.shape[0], np.nan)
    node_dom = {d: np.unique(tets[dom == d]) for d in domains}
    for f in feats:
        ents = domains if f["entities"] is None else f["entities"]
        if f["op"] in DOMAIN_MODELS:
            for d in ents:
                model_of[d] = f["op"]
                ref_temp[d] = scope.eval(f["par"].get("minput_strainreferencetemperature", ["293.15[K]"])[0])
                bio.pop(d, None)
                if f["op"] == "BiologicalTissue":
                    bh = next(s for s in f["sub"] if s["op"] == "Bioheat")["par"]
                    bio[d] = dict(Tb=scope.eval(bh["Tb"][0]), cb=scope.eval(bh["Cp_b"][0]),
                                  w=scope.eval(bh["omegab"][0]), rhob=scope.eval(bh["rhobl"][0]),
                                  q=scope.eval(bh["Qmet"][0]))
        elif f["op"] == "init":
            for d in ents:
                t_init[node_dom[d]] = scope.eval(f["par"]["Tinit"][0])

    # ---- materials
    k, rho_cp, variable, names = {}, {}, set(), {}
    p_abs = scope.eval(mph.features(heat)[0]["par"].get("pref", ["1[atm]"])[0])
    mats = mph.materials(component, switch_index)
    for m in mats:
        if m["dim"] != 3:
            continue
        ms = scope.child(m["funcs"])
        pr = m["props"]
        for d in m["entities"]:
            names[d] = m["name"]
            ek, er, ec = pr["thermalconductivity"], pr["density"], pr["heatcapacity"]
            # Density: solids keep the mass of the reference state; fluids use the
            # reference density because the flow coupling is Boussinesq.
            rho0 = float(ms.eval(er, T=ref_temp[d], pA=p_abs))
            if any(ms.uses(e, "T") for e in (ek, ec)):
                variable.add(d)
                k[d] = (lambda T, ms=ms, e=ek: ms.eval(e, T=T, pA=p_abs))
                rho_cp[d] = (lambda T, ms=ms, e=ec, r=rho0: r * ms.eval(e, T=T, pA=p_abs))
            else:
                k[d] = float(ms.eval(ek))
                rho_cp[d] = rho0 * float(ms.eval(ec))
            if m["switch_members"]:
                notes.append(f"{m['name']}: material switch -> member {switch_index + 1} '{m['source']}'")
    missing = [d for d in domains if d not in k]
    if missing:
        raise ValueError(f"domains without a material: {missing}")

    # ---- boundaries
    tris, bnd, up, down = mesh["tris"], mesh["tri_bnd"], mesh["tri_up"], mesh["tri_down"]
    steps = mph.study_steps(study)
    off = {s: {x.split("/")[1] for x in d.get("disabledphysics", []) if x.split("/")[0] == heat}
           for s, d in steps.items()}
    dirichlet, radiation, source, slit_bnd, convective = [], [], [], [], []
    for f in feats:
        if f["dim"] != 2 or f["entities"] is None:
            continue
        tr = tris[np.isin(bnd, f["entities"])]
        active = [s for s in steps if f["tag"] not in off[s]]
        if f["op"] == "TemperatureBoundary":
            expr = f["par"]["T0"][0]
            dirichlet.append(dict(name=f["name"], tag=f["tag"], boundaries=f["entities"], nodes=np.unique(tr),
                                  expr=expr, steps=active,
                                  value=(lambda t, e=expr: float(scope.eval(e, t=t)))))
        elif f["op"] == "SurfaceToAmbientRadiation":
            if f["par"].get("epsilon_rad_mat", ["userdef"])[0] == "from_mat":
                bm = next((m for m in mats if m["dim"] == 2 and set(f["entities"]) <= set(m["entities"] or [])), None)
                if bm is None or "emissivity" not in bm["props"]:
                    notes.append(f"{f['name']}: emissivity is taken from a boundary material that defines none; not applied")
                    continue
                eps = float(scope.child(bm["funcs"]).eval(bm["props"]["emissivity"], T=293.15))
            else:
                eps = scope.eval(f["par"]["epsilon_rad"][0])
            radiation.append(dict(name=f["name"], boundaries=f["entities"], tris=tr,
                                  eps=eps, t_amb=scope.eval(f["par"]["Tamb"][0])))
        elif f["op"] == "HeatFluxBoundary":
            kind = f["par"].get("HeatFluxType", ["GeneralInwardHeatFlux"])[0]
            if kind == "GeneralInwardHeatFlux":
                source.append(dict(name=f["name"], boundaries=f["entities"], tris=tr,
                                   q=float(scope.eval(f["par"]["q0_input"][0]))))
            elif kind == "ConvectiveHeatFlux":
                t_ext = float(scope.eval(f["par"]["Text"][0]))
                how = f["par"].get("HeatTransferCoefficientType", ["UserDef"])[0]
                if how == "UserDef":
                    hval = float(scope.eval(f["par"]["h"][0]))
                    hfun = lambda Ts, hval=hval: np.full_like(Ts, hval)
                elif how == "ExtNaturalConvection" and f["par"]["ExtNaturalConvectionType"][0] == "HorizontalPlateUp":
                    hfun = _h_horizontal_plate_up(float(scope.eval(f["par"]["Ldia"][0])), t_ext)
                else:
                    notes.append(f"{f['name']}: heat transfer coefficient type {how} not translated; not applied")
                    continue
                convective.append(dict(name=f["name"], boundaries=f["entities"], tris=tr, h=hfun, t_ext=t_ext))
            else:
                notes.append(f"{f['name']}: heat flux type {kind} not translated")
        elif f["op"] == "Inflow":
            expr = f["par"]["Tustr"][0]
            dirichlet.append(dict(name=f["name"], tag=f["tag"], boundaries=f["entities"], nodes=np.unique(tr),
                                  expr=expr, steps=active, value=(lambda t, e=expr: float(scope.eval(e, t=t)))))
            notes.append(f"{f['name']}: inflow condition imposed as the upstream temperature on the inlet face")
        elif f["op"] == "ConvectiveOutflow":
            pass
        elif f["op"] == "BoundaryHeatSource":
            source.append(dict(name=f["name"], boundaries=f["entities"], tris=tr,
                               q=scope.eval(f["par"]["Qb_input"][0])))
        elif f["op"] == "ThermalInsulation":
            interior = [b for b in f["entities"] if ((bnd == b) & (up > 0) & (down > 0)).any()]
            slit_bnd += interior
        elif f["op"] == "PairThermalContact":
            notes.append(f"{f['name']}: pairs {f['par'].get('pairs')} have the same boundary as source and "
                         "destination in a Form Union geometry, so the temperature is continuous there "
                         "and the contact layer carries no temperature jump; treated as continuity")
        elif f["op"] in ("Symmetry", "Continuity"):
            pass
        else:
            notes.append(f"boundary feature not translated: {f['name']} ({f['op']})")

    n0 = pts.shape[0]
    if insulation_slit and slit_bnd:
        tets, dup = _slit(mesh, tets, dom, slit_bnd)
        pts = np.vstack([pts, pts[dup]])
        t_new = np.empty(len(dup))
        # a duplicated node belongs to the up-side domain; take that domain's initial value
        for d in domains:
            nd = np.unique(tets[dom == d])
            nd = nd[nd >= n0]
            for f in feats:
                if f["op"] == "init" and (f["entities"] is None or d in f["entities"]):
                    t_new[nd - n0] = scope.eval(f["par"]["Tinit"][0])
        t_init = np.concatenate([t_init, t_new])
        notes.append(f"thermal insulation on interior boundaries {sorted(slit_bnd)}: temperature slit, "
                     f"{len(dup)} nodes duplicated")

    # ---- probes
    probes = {}
    for p in mph.root.iter("ProbeFeature"):
        if not (p.findtext("storedParent") or "").endswith(component):
            continue
        pv = {q.get("name"): q for q in p.findall("propertyValue")}
        if p.get("op") == "DomainPoint":
            from .mphread import values
            xyz = [float(scope.eval(v)) for v in values(pv["p:coords3"].get("valueMatrix"))]
            probes[p.get("name")] = dict(kind="point", xyz=xyz)
        elif p.get("op") == "Domain":
            _, ents, _ = mph.selection_of(p)
            kind = pv["p:type"].get("value") if "p:type" in pv else "average"
            probes[p.get("name")] = dict(kind=kind, domains=ents)

    flow = _flow(mph, heat, scope, mats, ref_temp, p_abs, notes)

    probes.update(extra_probes or {})

    return Problem(flow=flow, points=pts, tets=tets, dom=dom, n_original=n0, k=k, rho_cp=rho_cp, variable=variable,
                   perfusion={d: b["rhob"] * b["cb"] * b["w"] for d, b in bio.items()},
                   t_blood={d: b["Tb"] for d, b in bio.items()}, q_met={d: b["q"] for d, b in bio.items()},
                   t_init=t_init, dirichlet=dirichlet, radiation=radiation, surface_source=source,
                   convective=convective,
                   probes=probes, domain_names=names, tris=tris, tri_bnd=bnd, tri_up=up, tri_down=down,
                   notes=notes, params=par,
                   selections={v["name"]: v for v in mph.selections().values() if v["geom"] == geom})
