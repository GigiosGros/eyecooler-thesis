"""Shared pieces of the water-block scripts (dev/wb_*.py).  Imported, not run.

What is here
  * where the water-block inputs are (PC and cluster: same folder arrangement, forward or back slashes);
  * `open_model(v)`: the "Auxiliary Step" model of design v (1, 2, 3).  The reduced copy in
    dev/wb_inputs/ (made by dev/wb_make_inputs.py: dmodel.xml, smodel.json and the three meshes,
    member for member identical to the 635-783 MB originals) is used when present;
  * `load_mesh(mph, level)`: the stored mesh "coarse" / "normal" / "fine".  The reader in replica/ refuses the
    v3 meshes, because v3 holds two meshed faces that belong to no volume (the central disc, boundary 29, and a
    1 mm strip, boundary 76).  Here those triangles and their nodes are dropped and reported;
  * `build_problem(...)`: the heat + flow problem of replica.model.build at one operating point;
  * `reference_rows(v)`: COMSOL's laminar stationary tables as rows with their operating point;
  * read-outs: face averages and maxima, heat balance, pressure drop;
  * water properties and Reynolds-number helpers.

replica/ is used as it is; nothing in it is changed (its `_neighbours` is replaced only inside this process
while a mesh is being read).
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICA = os.path.dirname(HERE)
for _p in (HERE, REPLICA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BOX = os.path.normpath(os.path.join(REPLICA, "..", "..", "Computational model"))
PROJ = os.path.join(BOX, "Projects", "Waterblock")
AUX = os.path.join(PROJ, "Auxiliary Step")
REF = os.path.join(BOX, "Reports", "Waterblock", "Auxiliary Stationary Solution Laminar")
LITE = os.path.join(HERE, "wb_inputs")
LOCAL = os.environ.get("WB_COMSOL_LOCAL", os.path.join(REPLICA, "comsol", "Waterblock", "Auxiliary Step"))  # local copy, same meshes (sha-1 checked)

K0 = 273.15
MESH_NAME = {"coarse": "Mesh Coarse", "normal": "Mesh Normal", "fine": "Mesh Fine"}
REF_NAME = {"coarse": "Coarse", "normal": "Normal", "fine": "Fine_Sweep"}
# COMSOL's sweep (Study Fine): 3 x 3 x 2 x 2 = 36 points; flow in m3/s, temperatures in K, flux in W/m2
SWEEP = dict(Q0=(1.7e-6, 4.2e-6, 8.3e-6), Tin=(277.15, 283.15, 293.15), T0=(283.15, 308.15), Qh=(22000.0, 34000.0))
# the single operating point of the coarse and normal tables (parameters of the files as saved in May 2026)
BASE = dict(Q0=4.2e-6, Tin=293.15, T0=298.15, Qh=22000.0)
LPM = 6.0e4                                           # m3/s -> L/min


# ------------------------------------------------------------------ files
def model_file(v):
    name = f"Waterblock_v{v}_auxiliary_step.mph"
    for p in (os.path.join(LITE, name.replace(".mph", ".lite.mph")), os.path.join(AUX, name), os.path.join(LOCAL, name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}: not in {LITE}, {AUX} or {LOCAL}")


def open_model(v):
    from replica.mphread import Mph
    return Mph(model_file(v))


def code_stamp():
    """Hash of the solver files (same as run_all.code_version) and of the water-block dev files."""
    import glob
    out = {}
    h = hashlib.sha256()
    for path in [os.path.join(REPLICA, "run_all.py")] + sorted(glob.glob(os.path.join(REPLICA, "replica", "*.py"))):
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(os.path.basename(path).encode() + f.read())
    out["replica"] = h.hexdigest()[:12]
    h = hashlib.sha256()
    for name in ("wb_common.py", "wb_flow.py", "wb_runs.py", "flow2_td.py"):
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(name.encode() + f.read().replace(b"\r\n", b"\n"))
    out["waterblock"] = h.hexdigest()[:12]
    return out


# ------------------------------------------------------------------- mesh
def _neighbours_tolerant(tets, tet_dom, tris):
    """As replica.mphread._neighbours, but a triangle that is a face of no tetrahedron gets up = down = -1."""
    n = int(max(tets.max(), tris.max())) + 1

    def key(f):
        f = np.sort(f.astype(np.int64), axis=1)
        return (f[:, 0] * n + f[:, 1]) * n + f[:, 2]
    fk = np.concatenate([key(tets[:, c]) for c in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])])
    fd = np.tile(tet_dom, 4)
    order = np.argsort(fk, kind="stable")
    fk, fd = fk[order], fd[order]
    tk = key(tris)
    lo = np.searchsorted(fk, tk, "left")
    cnt = np.searchsorted(fk, tk, "right") - lo
    if (cnt > 2).any():
        raise ValueError("a triangle is shared by more than two tetrahedra")
    a = np.where(cnt >= 1, fd[np.minimum(lo, len(fd) - 1)], -1)
    b = np.where(cnt == 2, fd[np.minimum(lo + 1, len(fd) - 1)], 0)
    up = np.where(cnt == 0, -1, np.where(b == 0, 0, np.maximum(a, b)))
    down = np.where(cnt == 0, -1, np.where(b == 0, a, np.minimum(a, b)))
    return up.astype(np.int32), down.astype(np.int32)


def load_mesh(mph, level, refine=0):
    """Mesh dict (points m, tets, tet_dom, tris, tri_bnd, tri_up, tri_down) of a stored mesh level.

    Triangles that bound no volume, and nodes used by no tetrahedron, are removed; `mesh["dropped"]` says what.
    refine: number of times every element is split into eight (dev/metric_tools.refine_uniform)."""
    from replica import mphread
    seqs = {d["name"]: d for d in mph.mesh_sequences().values()}
    keep = mphread._neighbours
    mphread._neighbours = _neighbours_tolerant
    try:
        m = mph.mesh(seqs[MESH_NAME[level]]["resource"])
    finally:
        mphread._neighbours = keep
    orphan = m["tri_up"] < 0
    dropped = {}
    if orphan.any():
        ar = 0.5 * np.linalg.norm(np.cross(m["points"][m["tris"][orphan]][:, 1] - m["points"][m["tris"][orphan]][:, 0],
                                           m["points"][m["tris"][orphan]][:, 2] - m["points"][m["tris"][orphan]][:, 0]), axis=1)
        for b in np.unique(m["tri_bnd"][orphan]):
            s = m["tri_bnd"][orphan] == b
            dropped[int(b)] = dict(triangles=int(s.sum()), area_mm2=round(float(ar[s].sum()) * 1e6, 3))
    used = np.unique(m["tets"])
    new = np.full(len(m["points"]), -1, np.int64)
    new[used] = np.arange(len(used))
    out = dict(points=m["points"][used], tets=new[m["tets"]].astype(np.int32), tet_dom=m["tet_dom"].copy(),
               tris=new[m["tris"][~orphan]].astype(np.int32), tri_bnd=m["tri_bnd"][~orphan],
               tri_up=m["tri_up"][~orphan], tri_down=m["tri_down"][~orphan])
    out["dropped"] = dict(boundaries=dropped, nodes=int(len(m["points"]) - len(used)))
    for _ in range(int(refine)):
        import metric_tools
        d = out["dropped"]
        out = metric_tools.refine_uniform(out)
        out["tets"] = out["tets"].astype(np.int32)
        out["tris"] = out["tris"].astype(np.int32)
        out["dropped"] = d
    return out


def selections(mph):
    return {s["name"]: s["entities"] for s in mph.selections().values()}


# ---------------------------------------------------------------- problem
def build_problem(mph, mesh, op, alpha=1.0):
    """replica.model.build at the operating point op = dict(Q0, Tin, T0, Qh) and load level alpha."""
    from replica import model
    comp, info = next(iter(mph.components().items()))
    over = dict(Q0=float(op["Q0"]), Tin=float(op["Tin"]), T0=float(op["T0"]), Qh=float(op["Qh"]), alpha=float(alpha))
    prob = model.build(mph, component=comp, heat=info["heat"], geom=info["geom"], mesh=mesh, overrides=over,
                       insulation_slit=False)
    return prob


def face_sets(mph, prob):
    """{name: (triangles, areas)} of the faces read out: COMSOL's four boundary probes, the 'HotSide' and
    'Water Bnd' selections (the latter is the "Wet" of the Fine Energies tables), and the heated face
    split into its part on aluminium and its part on water (v3)."""
    from replica import fem
    sel = selections(mph)
    out = {}

    def add(name, mask):
        t = prob.tris[mask]
        if len(t):
            out[name] = (t, fem.tri_area(prob.points, t))
    for p in mph.root.iter("ProbeFeature"):
        if p.get("op") == "Boundary":
            _, ents, _ = mph.selection_of(p)
            add(p.get("name").replace("Boundary Probe ", ""), np.isin(prob.tri_bnd, ents))
    hot = np.isin(prob.tri_bnd, sel.get("HotSide", sel.get("TEC Interface", [])))
    water = fluid_domains(prob)
    on_water = np.isin(prob.tri_up, water) | np.isin(prob.tri_down, water)
    add("HotSide", hot)
    add("HotSide on aluminium", hot & ~on_water)
    add("HotSide on water", hot & on_water)
    add("Wet", np.isin(prob.tri_bnd, sel.get("Water Bnd", [])))
    add("Inlet", np.isin(prob.tri_bnd, sel["Inlet Bnd"]))
    add("Outlet", np.isin(prob.tri_bnd, sel["Outlet Bnd"]))
    solid_ext = (prob.tri_up == 0) & ~np.isin(prob.tri_down, water)
    add("Solid outer surface", solid_ext)
    return out


def fluid_domains(prob):
    return sorted({d for r in (prob.flow or {}).get("regions", []) for d in r["domains"]})


def face_average(T, face):
    t, a = face
    return float((T[t].mean(1) * a).sum() / a.sum())


def face_max(T, face):
    return float(T[np.unique(face[0])].max())


# ------------------------------------------------------------- references
def _table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = list([r for r in rows if r and r[0].startswith("%")][-1])
    head[0] = head[0].lstrip("% ")
    data = [[float(x) if x not in ("", "NaN") else float("nan") for x in r] for r in rows if r and not r[0].startswith("%")]
    return [h.strip() for h in head], data


def _probe_key(h):
    for k in ("TEC", "INw", "OUTw", "Waterblock"):
        if h.startswith(k):
            return k
    return None


def reference_rows(v, ref_dir=None):
    """COMSOL's laminar stationary results for design v: a list of dicts
    (mesh, Q0, Tin, T0, Qh, alpha, table, values {read-out: K or W}).

    coarse, normal: one operating point (BASE), alpha 0 / 0.5 / 1;
    fine: the 36-point sweep, alpha 0 / 0.5 / 1, plus the 'Fine Energies' export (HotSide and Wet
    average temperature, heat through the hot side in W) at the same points."""
    ref_dir = ref_dir or REF
    rows = []
    for level in ("coarse", "normal"):
        p = os.path.join(ref_dir, f"V{v}_Auxiliary_{REF_NAME[level]}.csv")
        if not os.path.exists(p):
            continue
        head, data = _table(p)
        for r in data:
            vals = {_probe_key(h): x for h, x in zip(head[1:], r[1:]) if _probe_key(h) and x == x}
            rows.append(dict(mesh=level, alpha=r[0], table=os.path.basename(p), values=vals, **BASE))
    fine = {}
    p = os.path.join(ref_dir, f"V{v}_Auxiliary_Fine_Sweep.csv")
    if os.path.exists(p):
        head, data = _table(p)
        for r in data:
            vals = {_probe_key(h): x for h, x in zip(head[5:], r[5:]) if _probe_key(h) and x == x}
            key = (r[0], r[1], r[2], r[3], r[4])
            fine[key] = dict(mesh="fine", Q0=r[0], Tin=r[1], T0=r[2], Qh=r[3], alpha=r[4], table=os.path.basename(p), values=vals)
    p = os.path.join(ref_dir, f"Fine Energies v{v}.csv")
    if os.path.exists(p):
        head, data = _table(p)
        for r in data:
            for h, x in zip(head[4:], r[4:]):
                if not h.startswith("alpha="):
                    continue
                a, what = h.split(",", 1)
                a, what = float(a.split("=")[1]), " ".join(what.split())
                name = None
                if what.startswith("Temperature HotSide"):
                    name = "HotSide"
                elif what.startswith("Temperature Wet"):
                    name = "Wet"
                elif what.startswith("Normal total heat flux HotSide") and what.endswith("(W)"):
                    name, x = "Heat in (W)", -x
                key = (r[0], r[1], r[2], r[3], a)
                if name and key in fine and x == x:
                    fine[key]["values"][name] = x
    return rows + list(fine.values())


def op_label(op):
    return f"Q{op['Q0'] * LPM:.2f}_Tin{op['Tin'] - K0:02.0f}_T0{op['T0'] - K0:02.0f}_q{op['Qh'] / 1000:.0f}k"


# ------------------------------------------------------------------ water
def water_nu(T):
    """Kinematic viscosity of water [m2/s] at T [K] (Vogel-type fit, within 1 % of tables from 0 to 40 C)."""
    mu = 2.414e-5 * 10.0 ** (247.8 / (np.asarray(T, float) - 140.0))
    return mu / 999.0


def sha1_of(path, block=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=conv)
