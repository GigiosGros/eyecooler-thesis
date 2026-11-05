"""Energy balance (gate V4) and division of the ring's heat by neighbour, from the saved fields of dev/new_runs.py.

    python -X utf8 dev/post_energy.py --root "D:\\COMSOL\\lakeshore_results\\new_runs"      # every finished case
    python -X utf8 dev/post_energy.py --root <folder> --case B_v1_Al C_                     # substring match
    python -X utf8 dev/post_energy.py --case-dir <folder of one case>
    python -X utf8 dev/post_energy.py --tables-only                                         # rebuild the two csv files
    options: --out <folder> (default results/overnight_2026-10-08/energy_balance), --redo

A case already done (cases/<case>.json with state ok) is skipped unless --redo.  A failing case writes its
error into cases/<case>.json and the next one starts.  Nothing is solved or factorised: the problem is rebuilt
exactly as the driver does (same calls as new_runs.run_one up to transform), checked against mesh.npz, and the
terms are evaluated from the saved temperature fields.

What is computed (all per time step n of the saved cooling history, half model, then doubled):

  residual      r = K(Tx) T_n + M(Tx) dT/dt - F0 - f_boundary(T_n), with replica.solve.Heat's own matrices and
                with the SAME time derivative the solver used (second-order BDF with its step ratio, backward
                Euler on the first step, properties at the extrapolated temperature Tx).  r is zero at every
                free node of the solved system and equals the heat supplied by the constraint at a
                fixed-temperature node.  Sum over the cooling-face nodes = -(heat leaving through the face);
                sum over the body-temperature nodes = heat entering there.
  terms         evaluated separately from the matrices, element by element and face by face:
                perfusion  sum_e w_e V_e (T_b - mean T_e), metabolic  sum_e q_e V_e,
                convection sum_f h A_f (T_ext - mean T_f), radiation (3-point rule, as the solver),
                evaporation sum_f q A_f, storage rate sum_e (rho c)_e V_e mean(dT/dt)_e.
  closure       (sum of entering terms - storage rate - heat through the cooling face) / heat through the face.
                Because the basis functions sum to one, the conduction matrix has zero column sums, so the
                discrete equations conserve heat exactly: the closure error is the sum of r over the free nodes
                (solver tolerance, float32 storage of the fields) plus any error in the separately evaluated
                terms.  Over the 35 min: flows integrated by the trapezoidal rule, against the change of stored
                heat taken directly from the first and last field; that one also contains the time error.
  split         heat into a set of domains S (the ring; the ring plus silicone) through each node of its
                surface, by where it comes from.  For a node i of S the nodal equation is
                    sum over all elements e at i of c_e,i = load_i + r_i
                (c_e,i: conduction + storage + perfusion contribution of element e to node i).  The elements
                of S hold the left-over; every other element delivers -c_e,i to the node, and that is booked to
                the element's domain ("consistent flux" of that side, no temperature is differentiated).
                Boundary loads on faces owned by S are booked to air convection / radiation, loads on a face of
                a neighbour that ends at the node to that neighbour.  The pieces sum to the storage rate of S
                plus the heat through the cooling face, exactly up to r at the free nodes.
"""
import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICA = os.path.dirname(HERE)
sys.path.insert(0, REPLICA)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

K0 = 273.15
OUT_DEFAULT = os.path.join(REPLICA, "results", "overnight_2026-10-08", "energy_balance")
N3 = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5]])
_cache = {}


# ------------------------------------------------------------------ problem
def build_problem(case_dir, log=print):
    """The Problem of a finished case, built with the driver's own steps. Returns (prob, cool, settings, cid)."""
    import new_runs
    import run_all
    from replica import model
    from replica.mphread import Mph
    import sensor_probes

    with open(os.path.join(case_dir, "settings.json"), encoding="utf-8") as f:
        js = json.load(f)
    cid = js["case"]
    s = dict(new_runs.DEFAULTS)
    s.update(js["settings"])                       # the settings the run was made with (older runs lack newer keys)
    rel = new_runs.SILICONE_FILE[s["silicone"]] if s["silicone"] else new_runs.MESH_FILE[s["mesh"]]
    path = run_all.mph_path(rel)
    if not os.path.exists(path):
        path = os.path.join(run_all.BOX, "Projects", "SCR", *rel.split("\\"))
    if os.path.basename(path) != js.get("model_file", os.path.basename(path)):
        raise RuntimeError(f"model file {os.path.basename(path)} is not the one of the run ({js['model_file']})")
    key = (path, s["ring"], s["material"], s["silicone"], s["T0"], s["Tb"], s["Tc"], s["refine"])
    if _cache.get("key") != key:
        _cache.clear()
        gc.collect()
        mph = Mph(path)
        comp = next(c for c, v in mph.components().items() if v["label"].endswith(s["ring"]))
        info = mph.components()[comp]
        extra = sensor_probes.sensor_probes()
        donor = None
        if info["geom"] not in mph.mesh_resources():
            dp = os.path.join(HERE, "inputs", f"mesh_silicone{s['silicone']}_{s['ring']}.npz")
            if not os.path.exists(dp):
                raise RuntimeError(f"{os.path.basename(path)} stores no mesh for {s['ring']} and {dp} does not exist")
            donor = dict(np.load(dp))
        if s["refine"]:
            import metric_tools
            donor = donor or mph.mesh(mph.mesh_resources()[info["geom"]]["resource"])
            for _ in range(int(s["refine"])):
                donor = metric_tools.refine_uniform(donor)
        base = model.build(mph, component=comp, heat=info["heat"], geom=info["geom"], insulation_slit=False,
                           mesh=donor, overrides=dict(T0=s["T0"] + K0, Tb=s["Tb"] + K0, Tc=s["Tc"] + K0),
                           switch_index=new_runs.MATERIAL_INDEX[s["material"]], extra_probes=extra)
        extra.update(sensor_probes.snap_to_sclera(base, extra))
        base.probes.update(extra)
        del mph
        _cache.update(key=key, base=base)
    base = _cache["base"]
    prob, cool, _ = new_runs.transform(base, s, lambda m: None)
    assert prob.dirichlet[-1]["name"] == "Cooling face"
    # the set-point ramp, exactly as new_runs.run_one defines it (face0 = mean start temperature of the face)
    old = next(d for d in base.dirichlet if "step" in d.get("expr", ""))
    Tc, T0k, t_ramp = s["Tc"] + K0, s["T0"] + K0, float(base.params.get("t_ramp", 90.0))
    span = (T0k - Tc) if abs(T0k - Tc) > 1e-9 else 1.0

    def ramp(t, f=t_ramp / s["ramp_s"]):
        return min(1.0, max(0.0, (T0k - old["value"](t * f)) / span)) if abs(T0k - Tc) > 1e-9 else float(t > 0)

    cool["make_value"] = lambda face0: (lambda t, a=face0, b=Tc: float(a + (b - a) * ramp(t)))
    return prob, cool, s, cid


def check_mesh(prob, cool, case_dir):
    m = np.load(os.path.join(case_dir, "mesh.npz"))
    if m["points"].shape != prob.points.shape or m["tets"].shape != prob.tets.shape:
        raise RuntimeError(f"mesh differs from mesh.npz: {prob.points.shape} / {m['points'].shape} nodes, "
                           f"{prob.tets.shape} / {m['tets'].shape} elements")
    d = float(np.abs(m["points"] - prob.points).max())
    if d > 1e-12 or not np.array_equal(m["tets"], prob.tets) or not np.array_equal(m["dom"], prob.dom) \
            or not np.array_equal(m["cooling_face_nodes"], cool["nodes"]):
        raise RuntimeError(f"mesh differs from mesh.npz (largest coordinate difference {d:.3e} m)")
    return d


# -------------------------------------------------------------- face loads
def face_loads(kind, spec, area, T):
    """Nodal loads (faces, 3) of one boundary condition at temperature T: the solver's own formulas."""
    from replica import fem
    tr = spec["tris"]
    if kind == "conv":
        Tt = T[tr]
        hA = spec["h"](Tt.mean(1)) * area
        return (hA * spec["t_ext"] / 3)[:, None] - hA[:, None] * (Tt @ fem.TRI_MASS_REF)
    if kind == "rad":
        Tq = T[tr] @ N3.T
        w = (area / 3 * spec["eps"] * fem.SIGMA)[:, None]
        return (w * (spec["t_amb"] ** 4 - Tq ** 4)) @ N3
    return np.repeat((spec["q"] * area / 3)[:, None], 3, 1)


class Split:
    """Heat into the domains S through the nodes of S, by origin (see the module docstring)."""

    def __init__(self, prob, vol, grad, S, bcs, label):
        from replica import fem
        p = self.p = prob
        self.label = label
        inS_el = np.isin(p.dom, S)
        self.inS = np.zeros(p.points.shape[0], bool)
        self.inS[np.unique(p.tets[inS_el])] = True
        self.el_S = np.where(inS_el)[0]
        self.el_N = np.where(~inS_el & self.inS[p.tets].any(1))[0]
        self.sets = {}
        for nm, el in (("S", self.el_S), ("N", self.el_N)):
            g, v, d = grad[el], vol[el], p.dom[el]
            wb = np.array([p.perfusion.get(x, 0.0) for x in d])
            src = np.array([p.q_met.get(x, 0.0) + p.perfusion.get(x, 0.0) * p.t_blood.get(x, 0.0) for x in d])
            self.sets[nm] = dict(t=p.tets[el], d=d, v=v, wb=wb, src=src,
                                 stiff=np.einsum("eik,ejk->eij", g, g) * v[:, None, None],
                                 mass=fem.MASS_REF[None] * v[:, None, None])
        self.names = sorted({p.domain_names[d] for d in np.unique(self.sets["N"]["d"])})
        self.name_idx = np.array([self.names.index(p.domain_names[d]) for d in self.sets["N"]["d"]])
        # owner of every boundary face that touches S: S itself if all its nodes are nodes of S, else the neighbour
        face_dom = {}
        tn = self.sets["N"]["t"]
        for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            f = np.sort(tn[:, [a, b, c]], axis=1)
            for row, d in zip(map(tuple, f.tolist()), self.name_idx.tolist()):
                face_dom[row] = d
        self.bc = []
        for kind, spec, area in bcs:
            tr = spec["tris"]
            touch = self.inS[tr].any(1)
            allS = self.inS[tr].all(1)
            owner = np.full(len(tr), -2)                       # -2: does not touch S
            owner[touch & allS] = -1                           # -1: a face of S
            for i in np.where(touch & ~allS)[0]:
                owner[i] = face_dom[tuple(sorted(tr[i].tolist()))]
            self.bc.append(owner)
        self.area_S = {spec["name"]: float(area[o == -1].sum()) for (kind, spec, area), o in zip(bcs, self.bc)}

    def _vec(self, st, kc, cc, T, Td):
        Te, Tde = T[st["t"]], Td[st["t"]]
        return (kc[:, None] * np.einsum("eij,ej->ei", st["stiff"], Te)
                + st["wb"][:, None] * np.einsum("eij,ej->ei", st["mass"], Te) - (st["src"] * st["v"] / 4)[:, None]
                + cc[:, None] * np.einsum("eij,ej->ei", st["mass"], Tde))

    def coefficients(self, st, Tx):
        p = self.p
        Te = Tx[st["t"]].mean(1)
        kc, cc = np.empty(len(Te)), np.empty(len(Te))
        for d in np.unique(st["d"]):
            m = st["d"] == d
            kc[m] = p.k[d](Te[m]) if d in p.variable else p.k[d]
            cc[m] = p.rho_cp[d](Te[m]) if d in p.variable else p.rho_cp[d]
        return kc, cc

    def step(self, T, Td, Tx, r, loads, groups):
        """loads: list of (faces, 3) arrays in the order of bcs; groups: {name: nodes} of the fixed nodes."""
        out = {}
        sS, sN = self.sets["S"], self.sets["N"]
        kc, cc = self.coefficients(sS, Tx)
        out["held by S (storage rate)"] = float(self._vec(sS, kc, cc, T, Td).sum())
        kc, cc = self.coefficients(sN, Tx)
        v = self._vec(sN, kc, cc, T, Td) * self.inS[sN["t"]]
        by = -np.bincount(self.name_idx, weights=v.sum(1), minlength=len(self.names))
        for (kind, spec, area), owner, ld in zip(self._bcs, self.bc, loads):
            m = self.inS[spec["tris"]]
            if (owner == -1).any():
                key = ("radiation" if kind == "rad" else spec["name"] if kind == "conv" else "surface source") + " on S"
                out[key] = out.get(key, 0.0) + float(ld[owner == -1].sum())
            sel = owner >= 0
            if sel.any():
                by += np.bincount(owner[sel], weights=(ld[sel] * m[sel]).sum(1), minlength=len(self.names))
        for n, x in zip(self.names, by):
            out["from " + n] = float(x)
        used = np.zeros(len(T), bool)
        for n, nodes in groups.items():
            nd = nodes[self.inS[nodes]]
            if len(nd):
                out["constraint: " + n] = float(r[nd].sum())
                used[nd] = True
        out["free-node residual"] = float(r[self.inS & ~used].sum())
        return out


# ---------------------------------------------------------------- one case
def analyse(case_dir, out_dir, log=print, built=None):
    """built: (prob, cool, settings, case id, Heat) of a run that has just finished in this process (the driver
    calls it that way, see REPORT.md); otherwise the problem is rebuilt from settings.json and the model file."""
    from replica import fem, solve
    t0 = time.time()
    h = None
    if built:
        prob, cool, s, cid, h = built
        cool = dict(cool)
    else:
        prob, cool, s, cid = build_problem(case_dir, log)
    dxyz = check_mesh(prob, cool, case_dir)
    p = prob
    n = p.points.shape[0]
    f = np.load(os.path.join(case_dir, "fields.npz"))
    times = f["time_cool_s"]
    Tc_all = f["T_cool"]
    if Tc_all.shape != (len(times), n):
        raise RuntimeError(f"fields {Tc_all.shape} do not fit {len(times)} times and {n} nodes")
    h = h or solve.Heat(p, log=lambda m: None)
    vol, grad = fem.tet_geometry(p.points, p.tets)
    names = p.domain_names
    ring = sorted(d for d, nm in names.items() if nm == "Ring")
    sil = sorted(d for d, nm in names.items() if nm == "Silicone")

    # element coefficients for the separately evaluated terms
    wb = np.zeros(len(vol)); qm = np.zeros(len(vol)); tb = np.zeros(len(vol)); cc0 = np.zeros(len(vol))
    for d in np.unique(p.dom):
        m = p.dom == d
        if d in p.perfusion:
            wb[m], qm[m], tb[m] = p.perfusion[d], p.q_met[d], p.t_blood[d]
        if d not in p.variable:
            cc0[m] = p.rho_cp[d]
    var_el = np.where(np.isin(p.dom, list(p.variable)))[0]
    dom_names = sorted(set(names.values()))
    el_name = np.array([dom_names.index(names[d]) for d in p.dom])
    q_met_W = float((qm * vol).sum())

    bcs = [("conv", c, fem.tri_area(p.points, c["tris"])) for c in p.convective] \
        + [("rad", c, fem.tri_area(p.points, c["tris"])) for c in p.radiation] \
        + [("src", c, fem.tri_area(p.points, c["tris"])) for c in p.surface_source]
    splits = [Split(p, vol, grad, ring, bcs, "ring")]
    if sil:
        splits.append(Split(p, vol, grad, ring + sil, bcs, "ring+silicone"))
    for sp_ in splits:
        sp_._bcs = bcs
    ring_nodes = splits[0].inS

    def rho_c(Tx):
        c = cc0.copy()
        if len(var_el):
            Te = Tx[p.tets[var_el]].mean(1)
            dv = p.dom[var_el]
            for d in p.variable:
                m = dv == d
                c[var_el[m]] = p.rho_cp[d](Te[m])
        return c

    def fixed_groups(step, t):
        idx, _ = h.constraints(step, t)
        g, taken = {}, np.zeros(n, bool)
        order = sorted((d for d in p.dirichlet if step in d["steps"]), key=lambda d: d["name"] != "Cooling face")
        for d in order:
            nd = np.asarray(d["nodes"])
            nd = nd[~taken[nd]]
            taken[nd] = True
            g[d["name"]] = nd
        return idx, g

    # The fields are stored in single precision.  At the fixed-temperature nodes the solution is known exactly
    # (set point, body temperature, bath), and its rounding is the same at every node of a face (4 C = 277.15 K
    # is stored as 277.149994 K), which is a systematic error of the conduction out of the face.  Those nodes
    # get their exact values back; everywhere else the rounding is random and stays.
    face0 = float(f["T_start"][cool["nodes"]].astype(np.float64).mean())
    if "make_value" in cool:                       # (the driver passes its own, already set)
        cool["value"] = cool["make_value"](face0)
    raw_end = None
    rows, srows = [], [[] for _ in splits]
    q_first = []                                   # the driver's formula (first-order difference, properties at T_n)
    stor_dom = []
    T1 = T2 = None
    t1 = t2 = None
    for i, t in enumerate(times):
        T = Tc_all[i].astype(np.float64)
        fix_idx, fix_val = h.constraints("start" if i == 0 else "cool", float(t))
        if i == len(times) - 1:
            raw_end = (T.copy(), float(np.abs(T[fix_idx] - fix_val).max()))
        T[fix_idx] = fix_val
        if i == 0:
            Td, Tx, step = np.zeros(n), T, "start"
        elif i == 1:
            dt = t - t1
            Td, Tx, step = (T - T1) / dt, T1, "cool"
        else:
            dt, dto = t - t1, t1 - t2
            w = dt / dto
            Td = ((1 + 2 * w) / (1 + w) * T - (1 + w) * T1 + w * w / (1 + w) * T2) / dt
            Tx, step = T1 + w * (T1 - T2), "cool"
        Kv, Mv = h.variable_matrices(Tx)
        fr, _ = h.radiation(T)
        r = h.K0 @ T + Kv @ T + h.M0 @ Td + Mv @ Td - h.F0 - fr
        idx, groups = fixed_groups(step, t)
        free = np.ones(n, bool)
        free[idx] = False
        if raw_end is not None:                  # the last step again with the fields exactly as stored
            Tr = raw_end[0]
            rr = h.K0 @ Tr + Kv @ Tr + h.M0 @ Td + Mv @ Td - h.F0 - h.radiation(Tr)[0]
            raw_end = dict(free_residual_sum_W=2 * float(rr[free].sum()),
                           cooling_face_out_W=-2 * float(rr[cool["nodes"]].sum()),
                           largest_rounding_at_fixed_nodes_K=raw_end[1])
        if i > 0:
            Kn, Mn = h.variable_matrices(T)
            Td1 = (T - T1) / (t - t1)
            r1 = h.K0 @ T + Kn @ T + h.M0 @ Td1 + Mn @ Td1 - h.F0 - fr
            q_first.append(-2.0 * float(r1[cool["nodes"]].sum()))
        else:
            q_first.append(float("nan"))

        row = {"t_s": float(t)}
        row["cooling face (out)"] = -float(r[groups["Cooling face"]].sum()) if "Cooling face" in groups else 0.0
        for nm, nd in groups.items():
            if nm != "Cooling face":
                row["fixed: " + nm] = float(r[nd].sum())
        row["perfusion"] = float((wb * vol * (tb - T[p.tets].mean(1))).sum())
        row["metabolic"] = q_met_W
        loads = []
        for kind, spec, area in bcs:
            ld = face_loads(kind, spec, area, T)
            loads.append(ld)
            on_ring = ring_nodes[spec["tris"]].all(1)
            if kind == "conv" and spec["name"] == "Air":
                row["air convection, ring"] = float(ld[on_ring].sum())
                row["air convection, tissue"] = float(ld[~on_ring].sum())
            elif kind == "conv":
                row["convection: " + spec["name"]] = float(ld.sum())
            elif kind == "rad":
                row["radiation, " + ("ring" if "ring" in spec["name"].lower() else "tissue")] = float(ld.sum())
            else:
                row["evaporation"] = row.get("evaporation", 0.0) + float(ld.sum())
        c = rho_c(Tx)
        st = c * vol * Td[p.tets].mean(1)
        row["storage rate"] = float(st.sum())
        stor_dom.append(np.bincount(el_name, weights=st, minlength=len(dom_names)))
        row["free-node residual sum"] = float(r[free].sum())
        row["free-node residual max"] = float(np.abs(r[free]).max())
        rows.append(row)
        for k, sp_ in enumerate(splits):
            srows[k].append(sp_.step(T, Td, Tx, r, loads, groups))
        T2, t2, T1, t1 = T1, t1, T, t

    # ---- stored heat directly from the first and last field (Simpson in temperature where rho*c varies)
    Ta, Tz = Tc_all[0].astype(np.float64), Tc_all[-1].astype(np.float64)
    c_mean = (rho_c(Ta) + 4 * rho_c(0.5 * (Ta + Tz)) + rho_c(Tz)) / 6
    dE_el = c_mean * vol * (Tz - Ta)[p.tets].mean(1)
    dE = float(dE_el.sum())
    dE_dom = np.bincount(el_name, weights=dE_el, minlength=len(dom_names))

    keys = []
    for rw in rows:
        keys += [k for k in rw if k not in keys]
    A = {k: np.array([rw.get(k, 0.0) for rw in rows]) for k in keys}
    inputs = [k for k in keys if k not in ("t_s", "cooling face (out)", "storage rate", "free-node residual sum",
                                           "free-node residual max")]
    trap = lambda y: float(np.trapezoid(y, times))
    W = {k: 2 * float(A[k][-1]) for k in keys if k != "t_s"}
    J = {k: 2 * trap(A[k]) for k in keys if k not in ("t_s", "free-node residual max")}
    W["free-node residual max"] = float(A["free-node residual max"][1:].max())      # largest of all steps, half model
    sumW = sum(W[k] for k in inputs)
    sumJ = sum(J[k] for k in inputs)
    res = dict(case=cid, state="ok", case_dir=case_dir, settings=s, nodes=int(n), elements=int(len(vol)),
               t_end_min=float(times[-1] / 60), steps=int(len(times) - 1), mesh_check_max_m=dxyz,
               variable_domains=sorted(names[d] for d in p.variable), input_terms=inputs,
               W=W, J=J, stored_change_J=2 * dE,
               stored_change_by_domain_J={nm: 2 * float(x) for nm, x in zip(dom_names, dE_dom)},
               storage_rate_by_domain_W={nm: 2 * float(x) for nm, x in zip(dom_names, stor_dom[-1])},
               sum_in_W=sumW, sum_in_J=sumJ, as_stored_float32=raw_end, face_start_C=face0 - K0)
    qc, Ec = W["cooling face (out)"], J["cooling face (out)"]
    res["closure_W_pct"] = 100 * (sumW - W["storage rate"] - qc) / qc if qc else float("nan")
    res["closure_J_pct"] = 100 * (sumJ - 2 * dE - Ec) / Ec if Ec else float("nan")
    res["closure_J_bdf_storage_pct"] = 100 * (sumJ - J["storage rate"] - Ec) / Ec if Ec else float("nan")
    res["closure_W_worst_step_pct"] = float(np.nanmax(np.abs(
        [100 * (sum(rw[k] for k in inputs) - rw["storage rate"] - rw["cooling face (out)"]) / rw["cooling face (out)"]
         for rw in rows[1:] if abs(rw["cooling face (out)"]) > 1e-9] or [np.nan])))
    # the driver's own value
    drv = np.full(len(times), np.nan)
    try:
        with open(os.path.join(case_dir, "probes_cooling.csv"), encoding="utf-8") as fh:
            rd = list(csv.reader(fh))[1:]
        drv = np.array([float(x[-1]) for x in rd])
    except Exception as e:                                    # the comparison is extra; the balance does not need it
        log(f"  probes_cooling.csv not read: {e}")
    q1 = np.array(q_first)
    res["driver"] = dict(
        csv_W_end=float(drv[-1]), first_order_recomputed_W_end=float(q1[-1]), bdf2_W_end=qc,
        csv_minus_recomputed_max_W=float(np.nanmax(np.abs(drv - q1))) if len(drv) == len(q1) else None,
        csv_minus_bdf2_end_pct=100 * (float(drv[-1]) - qc) / qc if qc else None,
        csv_minus_bdf2_max_W=float(np.nanmax(np.abs(drv - 2 * A["cooling face (out)"]))) if len(drv) == len(q1) else None,
        csv_J=float(np.trapezoid(np.nan_to_num(drv), times)) if len(drv) == len(times) else None, bdf2_J=Ec)
    res["split"] = {}
    for sp_, sr in zip(splits, srows):
        ks = []
        for rw in sr:
            ks += [k for k in rw if k not in ks]
        B = {k: np.array([rw.get(k, 0.0) for rw in sr]) for k in ks}
        res["split"][sp_.label] = dict(W={k: 2 * float(B[k][-1]) for k in ks}, J={k: 2 * trap(B[k]) for k in ks},
                                       exposed_area_mm2={k: 2e6 * v for k, v in sp_.area_S.items()})
        B["t_s"] = times
        _series(os.path.join(out_dir, "series", f"{cid}__split_{sp_.label}.csv"), B)
    A["driver cold-side heat, whole device (W)"] = drv / 2 if len(drv) == len(times) else np.full(len(times), np.nan)
    _series(os.path.join(out_dir, "series", f"{cid}__balance.csv"), A)
    res["seconds"] = round(time.time() - t0, 1)
    del h, splits, Tc_all, f
    gc.collect()
    return res


def _series(path, cols):
    """Time series, whole device (every column but the time and the largest residual is doubled)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ks = ["t_s"] + [k for k in cols if k != "t_s"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time (s)"] + [k + " (W)" for k in ks[1:]])
        for i in range(len(cols["t_s"])):
            w.writerow([f"{cols['t_s'][i]:.1f}"] + [f"{(1 if 'max' in k else 2) * cols[k][i]:.6g}" for k in ks[1:]])


# ------------------------------------------------------------------ tables
EYE = ("Sclera", "Cornea", "Tear", "Conjunctiva", "Ciliary", "Choroid", "Retina", "Aqueous", "Vitreous", "Lens", "Iris")
LIDS = ("Eyelids", "Fat", "Extraocular")


def groups_of(d):
    """Grouping of one split (W or J dict) into the thesis's headings."""
    g = dict(eye=0.0, tear_cornea=0.0, eyelids_orbit=0.0, silicone=0.0, air_convection=0.0, radiation=0.0, other=0.0,
             cooling_face_out=0.0, ring_storage=0.0, residual=0.0)
    for k, v in d.items():
        if k.startswith("from "):
            nm = k[5:]
            if nm == "Silicone":
                g["silicone"] += v
            elif nm.startswith(("Tear", "Cornea")):
                g["tear_cornea"] += v
            elif nm.startswith(LIDS):
                g["eyelids_orbit"] += v
            elif nm.startswith(EYE):
                g["eye"] += v
            else:
                g["other"] += v
        elif k == "radiation on S":
            g["radiation"] += v
        elif k == "constraint: Cooling face":
            g["cooling_face_out"] -= v
        elif k.startswith("held by S"):
            g["ring_storage"] += v
        elif k == "free-node residual":
            g["residual"] += v
        elif k.endswith(" on S") and k.startswith("Air"):
            g["air_convection"] += v
        else:
            g["other"] += v                                   # unpowered-module coupling, other constraints, sources
    return g


def write_tables(out_dir):
    cdir = os.path.join(out_dir, "cases")
    res = []
    for fn in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
        if fn.endswith(".json"):
            with open(os.path.join(cdir, fn), encoding="utf-8") as fh:
                res.append(json.load(fh))
    ok = [r for r in res if r.get("state") == "ok"]
    terms = []
    for r in ok:
        terms += [k for k in r["input_terms"] if k not in terms]
    with open(os.path.join(out_dir, "energy_balance.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "state", "nodes", "elements", "minutes", "cooling face out (W)"]
                   + [f"{k} (W)" for k in terms] + ["sum entering (W)", "storage rate (W)", "closure at end (%)",
                                                    "worst closure of any step (%)", "largest free-node residual (W)",
                                                    "cooling face out (J)"]
                   + [f"{k} (J)" for k in terms] + ["sum entering (J)", "change of stored heat (J)",
                                                    "closure over the run (%)", "same with integrated BDF storage rate (%)",
                                                    "driver cold-side heat at end (W)", "driver minus this at end (%)",
                                                    "driver integral (J)", "error"])
        for r in res:
            if r.get("state") != "ok":
                w.writerow([r.get("case"), r.get("state")] + [""] * (2 * len(terms) + 17) + [r.get("error", "")])
                continue
            W, J, d = r["W"], r["J"], r["driver"]
            f4 = lambda x: "" if x is None else f"{x:.5g}"
            w.writerow([r["case"], "ok", r["nodes"], r["elements"], f"{r['t_end_min']:g}", f4(W["cooling face (out)"])]
                       + [f4(W.get(k, 0.0)) for k in terms]
                       + [f4(r["sum_in_W"]), f4(W["storage rate"]), f"{r['closure_W_pct']:.4f}",
                          f"{r['closure_W_worst_step_pct']:.4f}", f"{W['free-node residual max']:.3g}",
                          f4(J["cooling face (out)"])]
                       + [f4(J.get(k, 0.0)) for k in terms]
                       + [f4(r["sum_in_J"]), f4(r["stored_change_J"]), f"{r['closure_J_pct']:.4f}",
                          f"{r['closure_J_bdf_storage_pct']:.4f}", f4(d["csv_W_end"]), f4(d["csv_minus_bdf2_end_pct"]),
                          f4(d["csv_J"]), ""])
    gk = ["eye", "tear_cornea", "eyelids_orbit", "silicone", "air_convection", "radiation", "other"]
    with open(os.path.join(out_dir, "heat_split.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "set", "cooling face out (J)", "release of heat stored in the set (J)"]
                   + [f"{k} (J)" for k in gk] + ["sum drawn from neighbours (J)"] + [f"{k} share (%)" for k in gk]
                   + ["unbalanced (J)", "cooling face out (W)", "storage rate of the set (W)"] + [f"{k} (W)" for k in gk]
                   + ["sum drawn from neighbours (W)"] + [f"{k} share at end (%)" for k in gk] + ["detail (J)"])
        for r in ok:
            for label, sp_ in r["split"].items():
                gJ, gW = groups_of(sp_["J"]), groups_of(sp_["W"])
                tJ, tW = sum(gJ[k] for k in gk), sum(gW[k] for k in gk)
                unb = tJ - gJ["cooling_face_out"] - gJ["ring_storage"]
                det = "; ".join(f"{k}: {v:.1f}" for k, v in sp_["J"].items())
                w.writerow([r["case"], label, f"{gJ['cooling_face_out']:.1f}", f"{-gJ['ring_storage']:.1f}"]
                           + [f"{gJ[k]:.1f}" for k in gk] + [f"{tJ:.1f}"] + [f"{100 * gJ[k] / tJ:.2f}" for k in gk]
                           + [f"{unb:.2f}", f"{gW['cooling_face_out']:.4f}", f"{gW['ring_storage']:.4f}"]
                           + [f"{gW[k]:.4f}" for k in gk] + [f"{tW:.4f}"] + [f"{100 * gW[k] / tW:.2f}" for k in gk] + [det])
    return res


# -------------------------------------------------------------------- main
def run_case(case_dir, out_dir, redo=False):
    name = os.path.basename(os.path.normpath(case_dir))
    jp = os.path.join(out_dir, "cases", name + ".json")
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    if os.path.exists(jp) and not redo:
        try:
            with open(jp, encoding="utf-8") as fh:
                if json.load(fh).get("state") == "ok":
                    return "skipped"
        except Exception:
            pass
    t0 = time.time()
    try:
        res = analyse(case_dir, out_dir)
    except BaseException as e:                                   # includes MemoryError; the next case still runs
        if isinstance(e, KeyboardInterrupt):
            raise
        res = dict(case=name, state="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        _cache.clear()
        gc.collect()
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    if res["state"] == "ok":
        return (f"ok  out {res['W']['cooling face (out)']:.3f} W, closure {res['closure_W_pct']:+.4f} % at end, "
                f"{res['closure_J_pct']:+.4f} % over the run, {time.time() - t0:.0f} s")
    return "FAILED " + res["error"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="folder with one sub-folder per case")
    ap.add_argument("--case", nargs="*", help="only cases whose name contains one of these")
    ap.add_argument("--case-dir", nargs="*", help="folder(s) of single cases")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = list(a.case_dir or [])
    if a.root and not a.tables_only:
        for nm in sorted(os.listdir(a.root)):
            d = os.path.join(a.root, nm)
            if os.path.exists(os.path.join(d, "fields.npz")) and os.path.exists(os.path.join(d, "settings.json")) \
                    and (not a.case or any(x in nm for x in a.case)):
                dirs.append(d)

    def order(d):                 # cases that share a model file, ring and material follow each other (one read each)
        try:
            with open(os.path.join(d, "settings.json"), encoding="utf-8") as fh:
                js = json.load(fh)
            s = js["settings"]
            return (js.get("model_file", ""), s["ring"], s["material"], s["T0"], s["Tb"], s["Tc"], js["case"])
        except Exception:
            return ("~",)
    dirs.sort(key=order)
    for i, d in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] {os.path.basename(os.path.normpath(d))}: {run_case(d, a.out, a.redo)}", flush=True)
        if i % 5 == 0:
            write_tables(a.out)
    res = write_tables(a.out)
    ok = [r for r in res if r.get("state") == "ok"]
    print(f"{len(ok)} of {len(res)} cases in the tables are ok; tables in {a.out}")


if __name__ == "__main__":
    main()
