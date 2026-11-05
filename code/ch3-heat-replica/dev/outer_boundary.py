"""Outer boundary of the in-vivo eye model (gate V10): what the fixed 37 C shell hides.

    python -X utf8 dev/outer_boundary.py layers      # 1-D layered model: equivalent coefficient of what lies beyond
    python -X utf8 dev/outer_boundary.py uniform     # (a) uniform shell coefficient 5 ... 5000 W/m2K
    python -X utf8 dev/outer_boundary.py regions     # (b) one region of the shell soft at a time
    python -X utf8 dev/outer_boundary.py extend      # (c) tissue added beyond the shell (extruded prisms -> tetrahedra)
    python -X utf8 dev/outer_boundary.py conseq      # best estimate and range x rings x perfusion
    python -X utf8 dev/outer_boundary.py tables      # summary_tables.md, csv, figures, MANIFEST.json from the cached runs
    python -X utf8 dev/outer_boundary.py all
    python -X utf8 dev/outer_boundary.py export production v1 v2 v3 [10]   # extended meshes in the driver's mesh_file format

Everything is solved on COMSOL's "coarser" mesh (73 thousand elements in the new set-up, about 15 s a run),
one run at a time.  The problem is built by dev/new_runs.py (transform) through dev/mesh_adapt.py (Setup);
neither is edited.  Each finished run is cached as a small json in
D:/COMSOL/overnight_scratch/outer_boundary/runs (delete a file to run it again).

Axes of the model (mm): z along the eye, cornea at 0, orbit apex at -45.6; x < 0 is nasal (the optic nerve
leaves towards -x); the plane y = 0 is the plane of symmetry, so "superior" and "inferior" are one region.
The shell is a cylinder of radius 17.5 mm from z = -5 to about -33 that closes in a dome at the apex.
"""
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

import new_runs as nr  # noqa: E402
import mesh_adapt as ma  # noqa: E402

K0 = 273.15
OUT = os.path.join(REPLICA, "results", "overnight_2026-10-08", "outer_boundary")
SCRATCH = r"D:\COMSOL\overnight_scratch\outer_boundary"
RUNS = os.path.join(SCRATCH, "runs")
SITES = [("Anteriror Sclera Probe", "anterior sclera"), ("Equatorial Sclera Probe", "equatorial sclera"),
         ("Vitreous Humor Probe", "vitreous"), ("Posterior Sclera Probe", "posterior sclera"),
         ("Optic Nerve Head Probe", "proximal optic nerve"), ("Optic Nerve Probe", "distal optic nerve"),
         ("Domain Probe Retina Avg", "average retina")]
DEEP = ("proximal optic nerve", "posterior sclera", "average retina", "vitreous")
PRODUCTION = {"proximal optic nerve": 33.77, "posterior sclera": 32.78, "average retina": 30.47, "vitreous": 24.64}
RHOC_BLOOD = 1068.0 * 3682.0            # the model's blood: W/(m3 K) per (1/s) of perfusion

# ---------------------------------------------------------------- materials beyond the shell
# k W/mK, rc = rho*cp J/m3K, w = perfusion term W/m3K (blood at body temperature), q = metabolic heat W/m3.
# Sources and status (read in the source / from memory) are in summary_tables.md, table "Properties".
MAT = {
    "fat":    dict(k=0.2245, rc=920.0 * 2500.0, w=0.0004 * RHOC_BLOOD, q=199.0),      # the model's orbital fat
    "bone":   dict(k=0.32, rc=1908.0 * 1313.0, w=10e-6 / 60 * 1908.0 * RHOC_BLOOD, q=0.0),    # cortical, 10 mL/min/kg
    "bone_hi": dict(k=0.68, rc=1908.0 * 1313.0, w=10e-6 / 60 * 1908.0 * RHOC_BLOOD, q=0.0),   # measured human cortical
    "brain":  dict(k=0.51, rc=1045.0 * 3650.0, w=500e-6 / 60 * 1045.0 * RHOC_BLOOD, q=0.0),   # grey/white mix
    "muscle": dict(k=0.49, rc=1090.0 * 3421.0, w=37e-6 / 60 * 1090.0 * RHOC_BLOOD, q=0.0),    # resting skeletal
    "mucosa": dict(k=0.34, rc=1102.0 * 3150.0, w=400e-6 / 60 * 1102.0 * RHOC_BLOOD, q=0.0),   # sinus lining
    # air-filled sinus as a solid: radiation (about 6) + conduction and weak convection (about 2) = 8 W/m2K across
    # the cavity; the conductivity is set per scenario so that 8 W/m2K results over the layer's thickness
    "cavity": dict(k=None, rc=1.15 * 1005.0, w=0.0, q=0.0),
}
H_CAVITY = 8.0
DEPTHS = {5: [0.5, 1, 2, 3.5, 5], 10: [0.5, 1, 2, 3, 4.5, 6, 8, 10], 20: [0.5, 1, 2, 3, 4.5, 6, 8, 10, 13, 16, 20]}
SECTORS = ("lateral", "sup-inf", "medial")
BANDS = (("rim", -14.3, 1e9), ("mid", -27.0, -14.3), ("deep", -38.0, -27.0))
REGIONS = [f"{b[0]} {s}" for b in BANDS for s in SECTORS] + ["apex"]


def region_of(cen_mm):
    """Region index (into REGIONS) of shell faces from their centroids (mm)."""
    x, y, z = cen_mm[:, 0], cen_mm[:, 1], cen_mm[:, 2]
    phi = np.degrees(np.arctan2(y, x))                       # 0 = temporal, 180 = nasal
    sec = np.where(phi < 60, 0, np.where(phi < 120, 1, 2))
    band = np.where(z > BANDS[0][1], 0, np.where(z > BANDS[1][1], 1, 2))
    return np.where(z <= BANDS[2][1], 9, band * 3 + sec)


# ---------------------------------------------------------------- stacks: best estimate of what lies beyond
# per region: list of (outer depth mm, material) and the condition at the outer surface of the added tissue
# ("fixed" 37 C, or a coefficient to 37 C).  Thicknesses use the layer depths of DEPTHS[10].
def anatomical(si="roof", bone="bone", sinus_T=37.0):
    sinus = [(0.5, bone), (1.0, "mucosa"), (10.0, "cavity")]
    roof = [(2.0, bone), (10.0, "brain")]
    st = {}
    for band in ("rim", "mid", "deep"):
        st[f"{band} medial"] = sinus                                        # lamina papyracea, ethmoid cells
        st[f"{band} sup-inf"] = roof if si == "roof" else sinus             # frontal lobe / maxillary sinus
    st["rim lateral"] = [(6.0, bone), (10.0, "muscle")]                     # zygomatic and frontal process, temporalis
    st["mid lateral"] = [(4.5, bone), (10.0, "muscle")]
    st["deep lateral"] = [(3.0, bone), (10.0, "brain")]                     # greater wing of sphenoid, temporal pole
    for s in ("medial", "sup-inf"):
        st[f"rim {s}"] = [(6.0, bone), (10.0, "muscle")]                    # orbital margin: thick bone, soft tissue
    st["apex"] = [(2.0, bone), (10.0, "brain")]                             # sphenoid, cavernous sinus, brain
    return st


# ---------------------------------------------------------------- 1-D layered model
def layers_1d(stack, outer="fixed", t_end=2100.0, n_per_mm=40, dt=2.0, times=(300.0, 900.0, 2100.0)):
    """Equivalent coefficient q / (T_body - T_surface) of a layered wall whose inner surface is stepped down by
    1 K at t = 0, everything starting at body temperature.  stack: [(thickness mm, material name or dict)];
    outer: "fixed" (body temperature), "insulated" or a coefficient W/m2K to body temperature.
    Returns ({time s: h}, steady h).  Finite volumes, backward Euler."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    dx, k, rc, w = [], [], [], []
    for th, m in stack:
        m = MAT[m] if isinstance(m, str) else m
        n = max(4, int(round(th * n_per_mm)))
        dx += [th * 1e-3 / n] * n; k += [m["k"]] * n; rc += [m["rc"]] * n; w += [m["w"]] * n
    dx, k, rc, w = map(np.array, (dx, k, rc, w))
    n = len(dx)
    g = 1.0 / (dx[:-1] / (2 * k[:-1]) + dx[1:] / (2 * k[1:]))            # between cells
    g0 = 2 * k[0] / dx[0]                                                # to the stepped surface
    gN = 0.0 if outer == "insulated" else 2 * k[-1] / dx[-1] if outer == "fixed" else \
        1.0 / (dx[-1] / (2 * k[-1]) + 1.0 / float(outer))
    diag = w * dx
    diag[0] += g0; diag[-1] += gN
    diag[:-1] += g; diag[1:] += g
    A = sp.diags([diag, -g, -g], [0, 1, -1]).tocsc()
    b = np.zeros(n); b[0] = g0 * (-1.0)                                  # theta = T - T_body; surface at -1
    steady = splu(A).solve(b)
    h_steady = g0 * (steady[0] + 1.0)
    M = sp.diags(rc * dx / dt).tocsc()
    lu = splu((A + M).tocsc()).solve
    th_, out, t = np.zeros(n), {}, 0.0
    for tt in sorted(times):
        while t < tt - 1e-9:
            th_ = lu(b + M @ th_); t += dt
        out[tt] = g0 * (th_[0] + 1.0)
    return out, h_steady


def h_semi(m, t=None):
    """Semi-infinite perfused tissue: steady sqrt(k w); after a surface step, at time t,
    sqrt(k w) erf(sqrt(t/tau)) + sqrt(k rho c / (pi t)) exp(-t/tau), tau = rho c / w."""
    from math import erf, exp, pi, sqrt
    m = MAT[m] if isinstance(m, str) else m
    if t is None:
        return sqrt(m["k"] * m["w"])
    if m["w"] == 0:
        return sqrt(m["k"] * m["rc"] / (pi * t))
    tau = m["rc"] / m["w"]
    return sqrt(m["k"] * m["w"]) * erf(sqrt(t / tau)) + sqrt(m["k"] * m["rc"] / (pi * t)) * exp(-t / tau)


def stack_with_cavity(stack):
    """Resolve the 'cavity' material: conductivity giving H_CAVITY over its own thickness."""
    out, prev = [], 0.0
    for depth, m in stack:
        mm = dict(MAT[m]) if isinstance(m, str) else dict(m)
        if m == "cavity":
            mm["k"] = H_CAVITY * (depth - prev) * 1e-3
        out.append((depth - prev, mm)); prev = depth
    return out


# ---------------------------------------------------------------- extruded mesh
def extend_mesh(mesh, shell_bnd, depths_mm, fat_domain=1, log=print):
    """Add layers of tetrahedra outside the faces whose boundary number is in shell_bnd.

    Each shell triangle is extruded along the node normals (area-weighted, kept in the plane y = 0 for nodes
    on it, lengthened so that the layer keeps its thickness on the coarse facets of the apex dome) into one
    prism per layer; each prism is split into three tetrahedra with the diagonal of every side face starting
    at its lowest-numbered corner, which makes neighbours agree.  The new elements carry the fat's domain
    number (so the driver gives them a material); `ext_layer` (0 = original, 1.. = layer) and `ext_region`
    (index into REGIONS, -1 = original) say what they are.  The new outer faces inherit the shell's boundary
    numbers, the old shell faces that are now inside fat are dropped, the other old shell faces and the new
    side faces get new boundary numbers that no condition uses."""
    P = np.asarray(mesh["points"], float)
    tris, bnd = np.asarray(mesh["tris"]), np.asarray(mesh["tri_bnd"])
    up, dn = np.asarray(mesh["tri_up"]), np.asarray(mesh["tri_down"])
    on = np.isin(bnd, list(shell_bnd))
    assert (np.minimum(up[on], dn[on]) == 0).all(), "a shell face is not on the outside"
    inner = np.maximum(up, dn)                               # domain on the inside (the outside is 0 in up or in down)
    up_is_out = bool((up[on] == 0).all())                    # which of the two arrays holds the outside in this mesh
    st = tris[on].astype(np.int64)
    a, b, c = P[st[:, 0]], P[st[:, 1]], P[st[:, 2]]
    nrm = np.cross(b - a, c - a)
    cen = (a + b + c) / 3.0
    flip = np.einsum("ij,ij->i", nrm, cen - np.array([0.0, 0.0, -0.025])) < 0
    st[flip] = st[flip][:, [0, 2, 1]]
    nrm[flip] *= -1
    unit = nrm / np.linalg.norm(nrm, axis=1)[:, None]
    nodes = np.unique(st)
    rank = np.full(len(P), -1, np.int64); rank[nodes] = np.arange(len(nodes))
    nn = np.zeros((len(P), 3))
    for j in range(3):
        np.add.at(nn, st[:, j], nrm)                         # area-weighted
    on_plane = np.abs(P[:, 1]) < 1e-9
    nn[on_plane, 1] = 0.0
    nn[nodes] /= np.linalg.norm(nn[nodes], axis=1)[:, None]
    cosmin = np.ones(len(P))
    for j in range(3):
        np.minimum.at(cosmin, st[:, j], np.einsum("ij,ij->i", nn[st[:, j]], unit))
    step = nn[nodes] / np.clip(cosmin[nodes], 0.5, 1.0)[:, None]
    n0, m = len(P), len(nodes)
    depths = np.asarray(depths_mm, float) * 1e-3
    newP = [P] + [P[nodes] + d * step for d in depths]
    P2 = np.vstack(newP)

    def idx(layer, node):                                    # layer 0 = the shell itself
        return np.where(layer == 0, node, n0 + (layer - 1) * m + rank[node])

    # rotate every triangle so that its lowest shell rank comes first (keeps the orientation)
    r = rank[st]
    first = r.argmin(1)
    st = np.take_along_axis(st, (first[:, None] + np.arange(3)[None]) % 3, 1)
    r = rank[st]
    low = r[:, 1] < r[:, 2]
    tets, lay, reg = [], [], []
    region = region_of(cen * 1e3)
    for L in range(1, len(depths) + 1):
        v1, v2, v3 = (idx(L - 1, st[:, j]) for j in range(3))
        v4, v5, v6 = (idx(L, st[:, j]) for j in range(3))
        A = np.where(low[:, None], np.stack([v1, v2, v3, v6], 1), np.stack([v1, v2, v3, v5], 1))
        B = np.where(low[:, None], np.stack([v1, v2, v6, v5], 1), np.stack([v1, v5, v3, v6], 1))
        C = np.stack([v1, v5, v6, v4], 1)
        for T in (A, B, C):
            tets.append(T); lay.append(np.full(len(T), L)); reg.append(region)
    tets = np.vstack(tets); lay = np.concatenate(lay); reg = np.concatenate(reg)
    q = P2[tets]
    vol = np.einsum("ij,ij->i", np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), q[:, 3] - q[:, 0]) / 6.0
    neg = vol < 0
    tets[neg] = tets[neg][:, [1, 0, 2, 3]]
    assert np.abs(vol).min() > 0, "degenerate element in the extension"

    # faces
    nb_int, nb_side = int(bnd.max()) + 1, int(bnd.max()) + 2
    keep = ~(on & (inner == fat_domain))                        # old shell faces inside fat disappear
    bnd2 = bnd.copy(); up2 = up.copy(); dn2 = dn.copy()
    bnd2[on] = nb_int
    up2[on], dn2[on] = np.maximum(inner[on], fat_domain), np.minimum(inner[on], fat_domain)
    Lz = len(depths)
    top = np.stack([idx(Lz, st[:, j]) for j in range(3)], 1)
    top_bnd = bnd[on]
    # side faces: edges of the shell surface that belong to one triangle only
    e = np.vstack([st[:, [0, 1]], st[:, [1, 2]], st[:, [2, 0]]])
    key = np.sort(e, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    free = e[cnt[inv.ravel()] == 1]
    side = []
    for L in range(1, Lz + 1):
        i, j = free[:, 0], free[:, 1]
        sw = rank[i] > rank[j]
        i, j = np.where(sw, j, i), np.where(sw, i, j)         # i has the lower rank: the diagonal goes i(bottom)-j(top)
        bi, bj, ti, tj = idx(L - 1, i), idx(L - 1, j), idx(L, i), idx(L, j)
        side += [np.stack([bi, bj, tj], 1), np.stack([bi, tj, ti], 1)]
    side = np.vstack(side)
    out = dict(points=P2, tets=np.vstack([mesh["tets"], tets.astype(mesh["tets"].dtype)]),
               tet_dom=np.concatenate([mesh["tet_dom"], np.full(len(tets), fat_domain, mesh["tet_dom"].dtype)]),
               tris=np.vstack([tris[keep], top.astype(tris.dtype), side.astype(tris.dtype)]),
               tri_bnd=np.concatenate([bnd2[keep], top_bnd, np.full(len(side), nb_side, bnd.dtype)]),
               tri_up=np.concatenate([up2[keep], np.full(len(top) + len(side), 0 if up_is_out else fat_domain, up.dtype)]),
               tri_down=np.concatenate([dn2[keep], np.full(len(top) + len(side), fat_domain if up_is_out else 0, dn.dtype)]))
    extra = dict(ext_layer=np.concatenate([np.zeros(len(mesh["tets"]), np.int16), lay.astype(np.int16)]),
                 ext_region=np.concatenate([np.full(len(mesh["tets"]), -1, np.int16), reg.astype(np.int16)]),
                 ext_depths_mm=np.asarray(depths_mm, float), top_cen_mm=P2[top].mean(1) * 1e3, top_region=region)
    # conformity: every face of the new elements is shared by two elements or is a listed boundary face
    allt = out["tets"][len(mesh["tets"]):].astype(np.int64)
    f = np.sort(np.vstack([allt[:, [0, 1, 2]], allt[:, [0, 1, 3]], allt[:, [0, 2, 3]], allt[:, [1, 2, 3]]]), 1)
    _, cnt = np.unique(f, axis=0, return_counts=True)
    n_open = int((cnt == 1).sum())
    expect = 2 * len(st) + len(side)                          # bottom + top + sides
    assert cnt.max() == 2 and n_open == expect, f"extension not conforming: {n_open} open faces, expected {expect}"
    log(f"  extension {depths_mm[-1]:g} mm in {Lz} layers: {len(tets)} new elements, {Lz * m} new nodes, "
        f"volume {np.abs(vol).sum() * 1e6:.2f} mL (half model); conforming (open faces {n_open} = expected)")
    return out, extra


def apply_stacks(prob, ext_layer, ext_region, depths_mm, stacks, fat_domain, Tb_C):
    """Give the added elements (the last ones of prob, all labelled fat) their materials.

    stacks: {region name: [(outer depth mm, material name or dict)]} or the name of a set in STACK_SETS.
    New domain numbers start at 101.  Tissues beyond the orbit are NOT scaled by the perf_* settings."""
    if isinstance(stacks, str):
        stacks = STACK_SETS[stacks]()
    n_new = int((ext_layer > 0).sum())
    lay, reg = ext_layer[-n_new:], ext_region[-n_new:]
    assert (prob.dom[-n_new:] == fat_domain).all(), "the added elements are not the last ones"
    depth = np.concatenate([[0.0], np.asarray(depths_mm, float)])
    mid = 0.5 * (depth[lay - 1] + depth[lay])
    dom = prob.dom.copy()
    code = 100
    for r, name in enumerate(REGIONS):
        prev = 0.0
        for (d_out, _), (th, m) in zip(stacks[name], stack_with_cavity(stacks[name])):
            sel = (reg == r) & (mid > prev) & (mid <= d_out)
            prev = d_out
            if not sel.any():
                continue
            code += 1
            dom[len(dom) - n_new + np.where(sel)[0]] = code
            prob.k[code], prob.rho_cp[code] = float(m["k"]), float(m["rc"])
            prob.domain_names[code] = f"beyond: {name}"
            if m["w"] > 0:
                prob.perfusion[code] = m["w"]
                prob.t_blood[code] = Tb_C + K0
                prob.q_met[code] = m["q"]
    prob.dom = dom
    return prob


STACK_SETS = {"anat_roof": lambda: anatomical("roof"), "anat_floor": lambda: anatomical("floor"),
              "anat_roof_bonehi": lambda: anatomical("roof", bone="bone_hi"),
              "anat_floor_bonehi": lambda: anatomical("floor", bone="bone_hi")}


# ---------------------------------------------------------------- one run
class Model:
    """One ring on the coarser mesh; builds the driver's problem and lets the boundary be replaced."""

    def __init__(self, ring="v1", **over):
        self.setup = ma.Setup(ring=ring, start="coarser", log=lambda m: None, **over)
        self.s = self.setup.s
        base = self.setup._base(self.setup.start_mesh)
        self.shell_bnd = next(d["boundaries"] for d in base.dirichlet if d["name"] == "Temperature Body")
        self.fat = nr._doms(base, "Fat")[0]
        self._ext = {}

    def extended(self, T):
        if T not in self._ext:
            self._ext[T] = extend_mesh(self.setup.start_mesh, self.shell_bnd, DEPTHS[T], self.fat,
                                       log=lambda m: print(m, flush=True))
        return self._ext[T]

    def problem(self, shell, ext=None, stacks=None):
        """shell: None = fixed 37 C; a number = uniform coefficient; {region name: coefficient or None} with
        optional key "T" {region: outside temperature C}.  ext: thickness of added tissue (mm) or None;
        stacks: {region: [(outer depth mm, material)]} for the added tissue (default: all fat)."""
        s = dict(self.s)
        regional = isinstance(shell, dict)
        s["shell_h"] = None if shell is None else (1.0 if regional else float(shell))
        self.setup.s = s
        mesh, extra = (self.setup.start_mesh, None) if ext is None else self.extended(ext)
        base, prob, cool, _ = self.setup.build(mesh, log=lambda m: None)
        if regional:                                          # replace the driver's single soft-shell entry
            Tb = s["Tb"] + K0
            i = next(j for j, c in enumerate(prob.convective) if c["name"] == "Body shell (soft)")
            tr = prob.convective.pop(i)["tris"]
            reg = self.face_region(prob, tr, ext)
            T_out = shell.get("T", {})
            fixed_nodes = []
            for r, name in enumerate(REGIONS):
                sel = tr[reg == r]
                if not len(sel):
                    continue
                hh = shell.get(name, shell.get("default"))
                if hh is None:
                    fixed_nodes.append(np.unique(sel))
                else:
                    prob.convective.append(dict(name=f"Shell {name}", tris=sel,
                                                h=(lambda Ts, h=float(hh): np.full_like(Ts, h)),
                                                t_ext=T_out.get(name, s["Tb"]) + K0))
            if fixed_nodes:
                prob.dirichlet.insert(0, dict(name="Temperature Body (part)", nodes=np.unique(np.concatenate(fixed_nodes)),
                                              value=(lambda t, v=Tb: v), steps=["start", "cool", "rewarm"]))
        if ext is not None and stacks is not None:
            apply_stacks(prob, extra["ext_layer"], extra["ext_region"], DEPTHS[ext], stacks, self.fat, s["Tb"])
        return base, prob, cool

    def face_region(self, prob, tr, ext):
        cen = prob.points[tr].mean(1) * 1e3
        if ext is None:
            return region_of(cen)
        from scipy.spatial import cKDTree                     # moved-out faces: region of the shell face they came from
        extra = self.extended(ext)[1]
        d, i = cKDTree(extra["top_cen_mm"]).query(cen)
        assert d.max() < 1e-6, "outer face not found"
        return extra["top_region"][i]

    def solve(self, shell=None, ext=None, stacks=None):
        from replica import solve
        base, prob, cool = self.problem(shell, ext, stacks)
        s = self.setup.s
        h = solve.Heat(prob, log=lambda m: None)
        h.probe_setup()
        T = h.stationary(step="start")
        face0, Tc = float(T[cool["nodes"]].mean()), s["Tc"] + K0
        old = next(d for d in base.dirichlet if "step" in d.get("expr", ""))
        T0k, t_ramp = s["T0"] + K0, float(base.params.get("t_ramp", 90.0))
        span = T0k - Tc

        def ramp(t, f=t_ramp / s["ramp_s"]):
            return min(1.0, max(0.0, (T0k - old["value"](t * f)) / span))

        cool["value"] = lambda t, a=face0, b=Tc: float(a + (b - a) * ramp(t))
        start = h.probes(T)
        rows = []
        stages = self.setup.run_all.stages(s["t_cool_min"] * 60.0)
        T = h.transient(T, s["t_cool_min"] * 60.0, stages, step="cool", tol=1e-5,
                        callback=lambda t, Tn: rows.append((t / 60.0, h.probes(Tn))))
        end = h.probes(T)
        steady = h.probes(h.stationary(step="cool", t=1.0e7))
        out = dict(elements=int(len(prob.tets)), nodes=int(len(prob.points)))
        for key, name in SITES:
            k = next(n for n in start if n.startswith(key))
            out[name] = dict(start=start[k] - K0, end=end[k] - K0, steady=steady[k] - K0,
                             t=[r[0] for r in rows[::4]], series=[r[1][k] - K0 for r in rows[::4]])
        del h
        gc.collect()
        return out


_models = {}


def run(tag, ring="v1", shell=None, ext=None, stacks=None, **over):
    """Cached run. tag names the json file."""
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, tag + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    key = (ring, json.dumps(over, sort_keys=True))
    if key not in _models:
        _models.clear()
        _models[key] = Model(ring, **over)
    t0 = time.time()
    res = _models[key].solve(shell, ext, stacks)
    res.update(tag=tag, ring=ring, settings=over, seconds=round(time.time() - t0, 1),
               shell=shell if not isinstance(shell, dict) else {k: v for k, v in shell.items()}, ext=ext,
               stacks=stacks)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f)
    print(f"{tag:46s} {res['seconds']:5.1f} s  " + "  ".join(f"{n[:9]} {res[n]['end']:.2f}" for n in DEEP), flush=True)
    return res


# ---------------------------------------------------------------- the studies
H_UNIFORM = (5, 10, 20, 35, 50, 75, 100, 150, 300, 1000, 5000)
# best estimate of the equivalent coefficient per region and its range (W/m2K to 37 C): see summary_tables.md
BEST = {"rim lateral": 20, "rim sup-inf": 20, "rim medial": 20, "mid lateral": 28, "mid sup-inf": 45,
        "mid medial": 22, "deep lateral": 60, "deep sup-inf": 45, "deep medial": 22, "apex": 75}
LOW = {"rim lateral": 10, "rim sup-inf": 10, "rim medial": 10, "mid lateral": 15, "mid sup-inf": 15,
       "mid medial": 9, "deep lateral": 30, "deep sup-inf": 15, "deep medial": 9, "apex": 30}
HIGH = {"rim lateral": 40, "rim sup-inf": 40, "rim medial": 40, "mid lateral": 50, "mid sup-inf": 90,
        "mid medial": 35, "deep lateral": 90, "deep sup-inf": 90, "deep medial": 35, "apex": 300}


def study_uniform():
    run("fixed")
    for h in H_UNIFORM:
        run(f"uniform_h{h}", shell=float(h))


def study_regions():
    run("uniform_h5000", shell=5000.0)
    run("uniform_h20", shell=20.0)
    for name in REGIONS:
        run(f"region_soft_{name.replace(' ', '_')}", shell={"default": 5000.0, name: 20.0})
    for band in ("rim", "mid", "deep"):                        # whole bands, and each sector over the whole length
        run(f"band_soft_{band}", shell=dict({"default": 5000.0}, **{f"{band} {s}": 20.0 for s in SECTORS}))
    for sct in SECTORS:
        run(f"sector_soft_{sct}", shell=dict({"default": 5000.0}, **{f"{b[0]} {sct}": 20.0 for b in BANDS}))
    # the other way round (the effects do not add up): everything at 20, one part held at 37 C
    for name in REGIONS:
        run(f"region_stiff_{name.replace(' ', '_')}", shell={"default": 20.0, name: 5000.0})
    for band in ("rim", "mid", "deep"):
        run(f"band_stiff_{band}", shell=dict({"default": 20.0}, **{f"{band} {s}": 5000.0 for s in SECTORS}))
    for sct in SECTORS:
        run(f"sector_stiff_{sct}", shell=dict({"default": 20.0}, **{f"{b[0]} {sct}": 5000.0 for b in BANDS}))


def study_extend():
    run("fixed")
    for T in (5, 10, 20):
        run(f"ext{T}_fat_fixed", ext=T)                                        # more orbital fat, 37 C further out
        run(f"ext{T}_fat_semi", ext=T, shell=h_semi("fat"))                    # ... ending in perfused fat for ever
    run("ext10_fat_insulated", ext=10, shell=1e-6)
    # check of the extension itself: added tissue that conducts perfectly must give back the fixed shell
    run("ext5_check_perfect_conductor", ext=5, stacks={n: [(5.0, dict(k=1.0e5, rc=1.0, w=0.0, q=0.0))] for n in REGIONS})
    for T in (5, 10, 20):                                                      # bone all round, then 37 C
        run(f"ext{T}_bone_fixed", ext=T, stacks={n: [(float(T), "bone")] for n in REGIONS})
    for si in ("roof", "floor"):                                               # anatomical stacks (10 mm)
        run(f"ext10_anat_{si}", ext=10, stacks=anatomical(si))
        run(f"ext10_anat_{si}_bonehi", ext=10, stacks=anatomical(si, bone="bone_hi"))
    # 2 mm of bone then brain everywhere (stiffest plausible) and bone + sinus everywhere (softest plausible)
    run("ext10_all_brain", ext=10, stacks={n: [(2.0, "bone"), (10.0, "brain")] for n in REGIONS})
    run("ext10_all_sinus", ext=10, stacks={n: [(0.5, "bone"), (1.0, "mucosa"), (10.0, "cavity")] for n in REGIONS})
    run("ext10_all_bone_muscle", ext=10, stacks={n: [(4.5, "bone"), (10.0, "muscle")] for n in REGIONS})


def study_conseq():
    for ring in ("v1", "v2", "v3"):
        for f in (1.0, 0.5, 2.0):
            if ring != "v1" and f != 1.0:
                continue
            kw = {} if f == 1.0 else dict(perf_all=f)
            sfx = f"{ring}" + ("" if f == 1.0 else f"_perf{f:g}")
            run(f"cq_{sfx}_fixed", ring=ring, **kw)
            run(f"cq_{sfx}_best", ring=ring, shell=dict(BEST), **kw)
            run(f"cq_{sfx}_low", ring=ring, shell=dict(LOW), **kw)
            run(f"cq_{sfx}_high", ring=ring, shell=dict(HIGH), **kw)
            run(f"cq_{sfx}_anat_roof", ring=ring, ext=10, stacks=anatomical("roof"), **kw)
            run(f"cq_{sfx}_anat_floor", ring=ring, ext=10, stacks=anatomical("floor"), **kw)
            for h in (20.0, 35.0, 50.0, 100.0):
                run(f"cq_{sfx}_h{h:g}", ring=ring, shell=h, **kw)
    run("cq_v1_perf0.25_fixed", perf_all=0.25)
    run("cq_v1_perf0.25_best", shell=dict(BEST), perf_all=0.25)
    run("cq_v1_perf0.25_low", shell=dict(LOW), perf_all=0.25)
    run("cq_v1_noperf_fixed", perf_all=0.0)
    run("cq_v1_noperf_best", shell=dict(BEST), perf_all=0.0)


def export(level="coarser", rings=("v1",), thicknesses=(5, 10, 20), folder=SCRATCH):
    """Extended meshes in the driver's mesh_file format (all added elements = fat, so the present driver runs
    them as "more orbital fat, 37 C further out"); ext_layer / ext_region / ext_depths_mm are stored as well,
    for the materials patch.  Reads the mesh only (no problem is built, nothing solved).  Copy the files to
    dev/inputs/ to use them."""
    import run_all
    from replica.mphread import Mph
    path = run_all.mph_path(nr.MESH_FILE[level])
    mph = Mph(path)
    os.makedirs(folder, exist_ok=True)
    for ring in rings:
        ref = Model(ring)                                     # boundary and domain numbers of this ring (coarser mesh)
        comp = next(c for c, v in mph.components().items() if v["label"].endswith(ring))
        info = mph.components()[comp]
        mesh = mph.mesh(mph.mesh_resources()[info["geom"]]["resource"])
        on = np.isin(mesh["tri_bnd"], ref.shell_bnd)
        cen = mesh["points"][mesh["tris"][on]].mean(1) * 1e3
        r = np.hypot(cen[:, 0], cen[:, 1])
        assert (cen[:, 2] < -4.9).all() and (r < 17.53).all() and ((r > 16.5) | (cen[:, 2] < -30)).all(),             "the boundary numbers of the shell differ in this mesh"
        for T in thicknesses:
            out, extra = extend_mesh(mesh, ref.shell_bnd, DEPTHS[T], ref.fat, log=lambda m: print(m, flush=True))
            p = os.path.join(folder, f"mesh_ext{T}mm_{ring}_{level}.npz")
            np.savez_compressed(p, **out, **extra)
            print("wrote", p, len(out["tets"]), "elements", flush=True)


if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    os.chdir(REPLICA)
    if "uniform" in what or "all" in what:
        study_uniform()
    if "regions" in what or "all" in what:
        study_regions()
    if "extend" in what or "all" in what:
        study_extend()
    if "conseq" in what or "all" in what:
        study_conseq()
    if "export" in what:                                      # export [level] [ring ...]
        rest = what[what.index("export") + 1:]
        export(rest[0] if rest else "coarser", tuple(r for r in rest[1:] if r.startswith("v")) or ("v1",),
               tuple(int(r) for r in rest[1:] if r.isdigit()) or (5, 10, 20))
    if "layers" in what or "tables" in what or "all" in what:
        import outer_boundary_report as rep
        rep.main()
