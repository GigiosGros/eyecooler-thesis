"""Shorten the optic-nerve stump of an ex-vivo problem (simulation plan, open item 7).

The models carry a straight nerve stump, 2.06 mm across and about 17.8 mm long (sheath = Sclera body, core =
part of the Retina body; the 2026-10-08 cad_geometry check section 4); the donor eyes had
10-13 mm. Ex vivo (dev/new_runs.py transform) the orbit tissue is removed and every outer face of the globe and
the stump - its side and its end - is in the oil bath: h_oil (W/m2K) to T_bath. This module cuts the stump on
the mesh in use, so the rest of the mesh is untouched and the difference is the stump alone.

Definitions (all from the mesh, nothing typed in):
  axis    principal direction of the nodes of the Sclera / Retina elements lying more than 1 mm behind the
          posterior pole (z < -26.9 mm); points away from the globe;
  exit    where the axis leaves the outer scleral surface: intersection of the axis line with a sphere fitted
          to the outer scleral boundary nodes of the posterior globe (away from the stump);
  length  L = (largest axial position of the stump nodes) - (axial position of the exit);
  cut     the plane normal to the axis at exit + L_new.

Rules for the elements at the cut (the mesh is not re-made, so the cut face follows element faces):
  "centroid"  remove the elements whose centroid lies beyond the plane (cut face within half an element of
              the plane on both sides; mean position reported);
  "inside"    remove every element with a node beyond the plane (cut face entirely on the globe side).
The difference between the two is the sensitivity to the jagged face.

The new face gets the condition of the original stump end: it is appended to the "Oil bath" convective
entry (same h, same T_bath). Domain read-outs, point probes, the cooling face and every other condition are
carried over with renumbered nodes. Ex vivo only (in vivo the nerve ends in orbital fat 2 mm before the 37 C
shell, a truncation of the orbit, not a stump: see results/scr_closeout_2026-10-16/REPORT.md section 2).
"""
import numpy as np

Z_POLE_MM = -25.936                 # posterior pole of the outer scleral sphere (CAD report)


def _names(prob, *keys):
    return [d for d, n in prob.domain_names.items() if n in keys]


def geometry(prob):
    """Axis, exit point and length of the stump (mm). Returns dict."""
    P = prob.points * 1e3
    tissue = _names(prob, "Sclera", "Retina")
    el = np.isin(prob.dom, tissue)
    nodes = np.unique(prob.tets[el])
    nodes = nodes[P[nodes, 2] < Z_POLE_MM - 1.0]
    X = P[nodes]
    c = X.mean(0)
    _, _, vt = np.linalg.svd(X - c, full_matrices=False)
    a = vt[0] / np.linalg.norm(vt[0])
    if a[2] > 0:
        a = -a
    # half model: the axis lies in the plane of symmetry y = 0 (the centroid of a half cylinder does not)
    c[1], a[1] = 0.0, 0.0
    a = a / np.linalg.norm(a)
    s = (X - c) @ a
    t_end = float(s.max())
    sc_nodes = np.unique(prob.tets[np.isin(prob.dom, _names(prob, "Sclera"))])
    sc_nodes = sc_nodes[P[sc_nodes, 2] < Z_POLE_MM - 1.0]
    t_end_sheath = float(((P[sc_nodes] - c) @ a).max())
    # radius: distance of the stump's nodes from the axis line
    r = np.linalg.norm((X - c) - np.outer(s, a), axis=1)
    # exit: the axis line meets the outer scleral sphere of the globe, fitted (least squares) to the outer
    # boundary nodes of the sclera in the posterior half, away from the stump and off the plane of symmetry
    sc = _names(prob, "Sclera")
    outer = ((prob.tri_up == 0) & np.isin(prob.tri_down, sc)) | ((prob.tri_down == 0) & np.isin(prob.tri_up, sc))
    tr = prob.tris[outer]
    tr = tr[~(np.abs(P[tr][:, :, 1]) < 1e-4).all(1)]
    bn = np.unique(tr)
    B = P[bn]
    sb = (B - c) @ a
    rb = np.linalg.norm((B - c) - np.outer(sb, a), axis=1)
    R = float(np.percentile(r, 99.5))
    sel = (B[:, 2] < -13.4) & (rb > R + 0.3)
    Bs = B[sel]
    M = np.column_stack([2 * Bs, np.ones(len(Bs))])
    sol = np.linalg.lstsq(M, (Bs ** 2).sum(1), rcond=None)[0]
    C, Rs = sol[:3], float(np.sqrt(sol[3] + sol[:3] @ sol[:3]))
    resid = float(np.abs(np.linalg.norm(Bs - C, axis=1) - Rs).max())
    # |c + t a - C| = Rs, the root on the stump side
    d = c - C
    bq = float(d @ a)
    t_exit = -bq + float(np.sqrt(bq * bq - (d @ d - Rs * Rs)))
    return dict(axis=a, centre=c, t_end=t_end, t_exit=t_exit, radius_mm=R, length_mm=t_end - t_exit,
                length_sheath_mm=t_end_sheath - t_exit,
                tilt_deg=float(np.degrees(np.arccos(abs(a[2])))), sphere_centre_mm=C.tolist(), sphere_radius_mm=Rs,
                sphere_fit_max_residual_mm=resid, sphere_nodes=int(len(Bs)))


def shorten(prob, cool, length_mm, rule="centroid", log=print):
    """Cut the stump to length_mm (from the exit, along the axis). Returns (prob, cool, info); prob is modified."""
    if "Oil bath" not in [c["name"] for c in prob.convective]:
        raise RuntimeError("nerve_stump.shorten is for ex-vivo problems (no oil-bath condition found)")
    g = geometry(prob)
    if not np.isfinite(g["t_exit"]):
        raise RuntimeError("exit point of the nerve axis not found")
    t_cut = g["t_exit"] + float(length_mm)
    P = prob.points * 1e3
    tissue = _names(prob, "Sclera", "Retina")
    sax = (P - g["centre"]) @ g["axis"]
    if rule == "centroid":
        beyond = sax[prob.tets].mean(1) > t_cut
    elif rule == "inside":
        beyond = (sax[prob.tets] > t_cut + 1e-9).any(1)
    else:
        raise ValueError(rule)
    beyond &= (P[prob.tets][:, :, 2] < Z_POLE_MM - 1.0).all(1)    # never the globe
    if not np.isin(prob.dom[beyond], tissue).all():
        raise RuntimeError("the cut would remove elements other than the stump's sclera and retina")
    removed_vol = 0.0
    if beyond.any():
        e = P[prob.tets[beyond]]
        removed_vol = float(np.abs(np.einsum("ij,ij->i", e[:, 1] - e[:, 0],
                                             np.cross(e[:, 2] - e[:, 0], e[:, 3] - e[:, 0]))).sum() / 6)
    keep = ~beyond
    # new boundary: faces shared between a kept and a removed element
    loc = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    fk = np.sort(prob.tets[keep][:, loc].reshape(-1, 3), 1)
    dk = np.repeat(prob.dom[keep], 4)
    fr = np.sort(prob.tets[beyond][:, loc].reshape(-1, 3), 1)
    if len(fr):
        fr_set = {tuple(f) for f in fr.tolist()}
        is_cut = np.array([tuple(f) in fr_set for f in fk.tolist()], bool)
    else:
        is_cut = np.zeros(len(fk), bool)
    cut_tris, cut_dom = fk[is_cut], dk[is_cut]

    tets = prob.tets[keep]
    used = np.unique(tets)
    new = np.full(len(prob.points), -1, np.int64)
    new[used] = np.arange(len(used))

    def tmap(t):
        t = new[np.asarray(t)]
        return t[(t >= 0).all(1)]

    def nmap(n):
        n = new[np.asarray(n)]
        return n[n >= 0]

    old_cool = len(cool["nodes"])
    for d in prob.dirichlet:
        d["nodes"] = nmap(d["nodes"])
    if len(cool["nodes"]) != old_cool:
        raise RuntimeError("the cut touched the cooling face")
    for lst in (prob.radiation, prob.surface_source, prob.convective):
        for c in lst:
            c["tris"] = tmap(c["tris"])
    oil = next(c for c in prob.convective if c["name"] == "Oil bath")
    oil["tris"] = np.vstack([oil["tris"], new[cut_tris]]) if len(cut_tris) else oil["tris"]

    keep_tri = (new[prob.tris] >= 0).all(1)
    nb = int(prob.tri_bnd.max()) + 1 if len(prob.tri_bnd) else 1
    prob.tris = np.vstack([new[prob.tris[keep_tri]], new[cut_tris]]).astype(prob.tris.dtype)
    prob.tri_bnd = np.concatenate([prob.tri_bnd[keep_tri], np.full(len(cut_tris), nb, prob.tri_bnd.dtype)])
    prob.tri_up = np.concatenate([prob.tri_up[keep_tri], np.zeros(len(cut_tris), prob.tri_up.dtype)])
    prob.tri_down = np.concatenate([prob.tri_down[keep_tri], cut_dom.astype(prob.tri_down.dtype)])
    prob.points = prob.points[used]
    prob.tets = new[tets].astype(prob.tets.dtype)
    prob.dom = prob.dom[keep]
    prob.t_init = prob.t_init[used]
    prob.n_original = len(prob.points)

    # face position and area against the plane
    Pn = prob.points * 1e3
    cut_new = new[cut_tris]
    if len(cut_new):
        snodes = np.unique(cut_new)
        sn = (Pn[snodes] - g["centre"]) @ g["axis"]
        v = Pn[cut_new]
        area = float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2)
        face = dict(mean_mm=float(sn.mean() - g["t_exit"]), min_mm=float(sn.min() - g["t_exit"]),
                    max_mm=float(sn.max() - g["t_exit"]), area_mm2=area)
    else:
        face = dict(mean_mm=float("nan"), min_mm=float("nan"), max_mm=float("nan"), area_mm2=0.0)
    s_all = (Pn - g["centre"]) @ g["axis"]
    info = dict(rule=rule, length_requested_mm=float(length_mm), length_model_mm=g["length_mm"],
                length_model_sheath_mm=g["length_sheath_mm"], sphere_radius_mm=g["sphere_radius_mm"],
                sphere_centre_mm=g["sphere_centre_mm"], sphere_fit_max_residual_mm=g["sphere_fit_max_residual_mm"],
                length_after_mm=float(s_all.max() - g["t_exit"]), axis=g["axis"].tolist(),
                tilt_deg=g["tilt_deg"], radius_mm=g["radius_mm"], exit_axial_mm=g["t_exit"],
                elements_removed=int(beyond.sum()), volume_removed_mm3=removed_vol,
                cut_face_triangles=int(len(cut_tris)), cut_face_axial_mm=face,
                half_disc_area_mm2=float(np.pi * g["radius_mm"] ** 2 / 2))
    log(f"  nerve stump: model length {g['length_mm']:.2f} mm (sheath {g['length_sheath_mm']:.2f}; outer sclera "
        f"sphere {g['sphere_radius_mm']:.3f} mm, fit residual {g['sphere_fit_max_residual_mm']:.3f}; axis tilt {g['tilt_deg']:.2f} deg, radius "
        f"{g['radius_mm']:.2f} mm) cut to {length_mm:g} mm ({rule}): {info['elements_removed']} elements, "
        f"{removed_vol:.2f} mm3 removed; new end face {len(cut_tris)} triangles, {face['area_mm2']:.2f} mm2 "
        f"(half disc {info['half_disc_area_mm2']:.2f}), at {face['min_mm']:.2f}-{face['max_mm']:.2f} mm "
        f"(mean {face['mean_mm']:.2f}); in the oil bath like the old end")
    return prob, cool, info
