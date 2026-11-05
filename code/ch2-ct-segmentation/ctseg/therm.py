"""Thermistor beads at full resolution, eye-frame coordinates, and validation against Slicer picks."""
import json, os
import numpy as np
import h5py
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from . import io


def metal_threshold(cal, metal_n, raw_max):
    thr = cal["B"] + metal_n * (cal["T"] - cal["B"])
    saturated = thr > 0.97 * raw_max
    return (raw_max - 1 if saturated else thr), bool(saturated)


def find_beads(scan, xdir, f, s0, s, gl, cal, P, raw_max):
    """Search the globe bounding box (+margin) at full resolution for metal; one bead per EDT peak."""
    thr, sat = metal_threshold(cal, P["metal_n"], raw_max)
    idx = np.argwhere(gl)
    m = int(np.ceil(P["thermistor_search_margin_mm"] / s))
    lo = (idx.min(0) - m) * f; hi = (idx.max(0) + m + 1) * f
    pts = []
    step = 64
    for k0 in range(lo[0], hi[0], step):
        crop, (ka, ja, ia) = io.read_crop(xdir, k0, min(k0 + step, hi[0]), lo[1], hi[1], lo[2], hi[2])
        w = np.argwhere(crop >= thr)
        if len(w):
            pts.append(w + np.array([ka, ja, ia]))
    if not pts:
        return [], dict(threshold_raw=float(thr), saturated_scan=sat, n_metal_voxels=0)
    pts = np.concatenate(pts)
    # cluster on a half-resolution grid
    h = pts // 2; hmin = h.min(0); grid = np.zeros(tuple(h.max(0) - hmin + 1), bool)
    grid[tuple((h - hmin).T)] = True
    grid = ndi.binary_dilation(grid, iterations=1)
    L, nl = ndi.label(grid)
    cid = L[tuple((h - hmin).T)]
    edt_gl_in = ndi.distance_transform_edt(gl) * s
    edt_gl_out = ndi.distance_transform_edt(~gl) * s
    beads = []
    for c in range(1, nl + 1):
        q = pts[cid == c]
        if len(q) * s0 ** 3 < 0.003:
            continue
        qmin = q.min(0) - 2
        box = np.zeros(tuple(q.max(0) - qmin + 3), bool); box[tuple((q - qmin).T)] = True
        box = ndi.binary_fill_holes(ndi.binary_closing(box, iterations=1))
        D = ndi.distance_transform_edt(box)
        dmax = D.max()
        pk = peak_local_max(D, min_distance=max(1, int(0.8 / s0)), threshold_abs=max(0.6 * dmax, 0.12 / s0),
                            exclude_border=False)
        if len(pk) == 0:
            pk = np.array([np.unravel_index(D.argmax(), D.shape)])
        # bead = thickest part of a metal cluster; lead wires / saturated streaks are thinner. A second bead in
        # the same cluster must be nearly as thick (>= 92 %) and >= 1.5 mm from an already accepted one.
        pk = sorted([p for p in pk if D[tuple(p)] >= 0.92 * dmax], key=lambda p: -D[tuple(p)])
        acc = []
        for p in pk:
            if all(np.linalg.norm((p - a) * s0) >= 1.5 for a in acc):
                acc.append(p)
        pk = acc
        for p in pk:
            rw = int(0.6 / s0) + 1                            # EDT-weighted centroid of the bead core
            o = np.maximum(p - rw, 0); W = D[o[0]:p[0] + rw + 1, o[1]:p[1] + rw + 1, o[2]:p[2] + rw + 1]
            sel = W >= 0.8 * D[tuple(p)]
            ctr = (np.argwhere(sel) * W[sel][:, None]).sum(0) / W[sel].sum() + o
            full_mm = (ctr + qmin) * s0                       # (k, j, i) mm, full-res grid
            down_vox = (full_mm - (f - 1) / 2 * s0) / s
            inside = ndi.map_coordinates(gl.astype(np.uint8), down_vox[:, None], order=0)[0] > 0
            depth = (ndi.map_coordinates(edt_gl_in, down_vox[:, None], order=1)[0] if inside
                     else -ndi.map_coordinates(edt_gl_out, down_vox[:, None], order=1)[0])
            if depth < -P["thermistor_search_margin_mm"]:
                continue
            beads.append(dict(full_mm_kji=full_mm.round(3).tolist(),
                              lps_mm=full_mm[::-1].round(3).tolist(),
                              bead_diam_mm=round(float(2 * D[tuple(p)] * s0), 3),
                              cluster_metal_mm3=round(float(len(q) * s0 ** 3), 4),
                              depth_below_surface_mm=round(float(depth), 2)))
    # drop wire-only clusters: bead diameter well below the typical bead
    if beads:
        typical = np.median(sorted([b["bead_diam_mm"] for b in beads])[-max(1, len(beads) // 2):])
        beads = [b for b in beads if b["bead_diam_mm"] >= max(0.75 * typical, 0.3)]
    # merge duplicates closer than 0.8 mm
    keep = []
    for b in sorted(beads, key=lambda b: -b["bead_diam_mm"]):
        if all(np.linalg.norm(np.subtract(b["lps_mm"], k["lps_mm"])) > 0.8 for k in keep):
            keep.append(b)
    return keep, dict(threshold_raw=float(thr), saturated_scan=sat, n_metal_voxels=int(len(pts)))


def eye_coords(beads, anat, f, s0):
    o = np.array(anat["frame"]["origin_mm"]); F = np.array(anat["frame"]["axes"])
    apex = np.array(anat["anterior_pole_mm"])
    for b in beads:
        p = np.array(b["full_mm_kji"]) - (f - 1) / 2 * s0
        r = F @ (p - o)
        b["eye_xyz_mm"] = r.round(3).tolist()
        b["polar_from_anterior_pole_deg"] = round(float(np.degrees(np.arccos(np.clip(r[2] / np.linalg.norm(r), -1, 1)))), 1)
        b["azimuth_deg"] = round(float(np.degrees(np.arctan2(r[1], r[0]))), 1)
        b["axial_dist_from_anterior_pole_mm"] = round(float(F[2] @ (apex - p)), 2)
    return beads


# ---------------- Slicer picks (made in the H9 reference frame) -----------------------------
def read_itk_affine(fn):
    with h5py.File(fn) as h:
        g = h["TransformGroup"]; t = g[sorted(g.keys())[-1]]
        p = t["TransformParameters"][:]; c = t["TransformFixedParameters"][:3]
    return p[:9].reshape(3, 3), p[9:12], c


def load_picks(picks_dir, sid):
    mf = os.path.join(picks_dir, f"{sid}.mrk.json")
    if not os.path.exists(mf):
        return None
    cps = json.load(open(mf))["markups"][0]["controlPoints"]
    Pp = np.array([c["position"] for c in cps]); labels = [c["label"] for c in cps]
    cands = {"as_saved": Pp}
    tf = os.path.join(picks_dir, f"{sid} centering transform.h5")
    if os.path.exists(tf):
        A, t, c = read_itk_affine(tf)
        cands["tf_forward"] = (Pp - c) @ A.T + c + t
        cands["tf_inverse"] = (Pp - c - t) @ np.linalg.inv(A).T + c
    return dict(file=mf, labels=labels, candidates=cands)


def match_picks(beads, picks, max_mm=3.0):
    """Choose the pick-coordinate convention that best fits the beads, then label beads 1:1."""
    if picks is None or not beads:
        return None
    B = np.array([b["lps_mm"] for b in beads])
    best = None
    for name, Pp in picks["candidates"].items():
        D = np.linalg.norm(Pp[:, None] - B[None], axis=-1)
        r, c = linear_sum_assignment(D)
        med = float(np.median(D[r, c]))
        if best is None or med < best[1]:
            best = (name, med, r, c, D)
    name, med, r, c, D = best
    for i, j in zip(r, c):
        if D[i, j] <= max_mm:
            beads[j]["pick_label"] = picks["labels"][i]
            beads[j]["pick_dist_mm"] = round(float(D[i, j]), 3)
    return dict(convention=name, median_dist_mm=round(med, 3), n_picks=len(picks["labels"]),
                n_matched=int(sum(D[i, j] <= max_mm for i, j in zip(r, c))),
                dists_mm=[round(float(D[i, j]), 3) for i, j in zip(r, c)])


def picks_in_eye_frame(picks, convention, names, gl, s, f, s0, anat):
    """Every Slicer pick (sensor) in the anatomical frame: depth below the globe surface, polar angle, radius."""
    if picks is None:
        return []
    conv = convention if convention in picks["candidates"] else ("tf_forward" if "tf_forward" in picks["candidates"] else "as_saved")
    din = ndi.distance_transform_edt(gl) * s; dout = ndi.distance_transform_edt(~gl) * s
    o = np.array(anat["frame"]["origin_mm"]); F = np.array(anat["frame"]["axes"]); apex = np.array(anat["anterior_pole_mm"])
    R = anat["axial_length_mm"] / 2
    out = []
    for i, (lbl, p) in enumerate(zip(picks["labels"], picks["candidates"][conv])):
        w = p[::-1] - (f - 1) / 2 * s0                        # LPS full-res -> (k,j,i) mm on the working grid
        v = (w / s)[:, None]
        inside = ndi.map_coordinates(gl.astype(np.uint8), v, order=0)[0] > 0
        depth = ndi.map_coordinates(din, v, order=1)[0] if inside else -ndi.map_coordinates(dout, v, order=1)[0]
        x = F @ (w - o)
        # the sensor is named by the number in the pick's label (H8-3 -> 3rd name), as the
        # thesis-analysis package does (markers.key_by = "label"); list position is only the
        # fallback. They differ in H8.mrk.json alone (points 2 and 3 carry labels H8-3, H8-2).
        num = lbl.rsplit("-", 1)[-1] if "-" in str(lbl) else ""
        k = int(num) - 1 if num.isdigit() and 1 <= int(num) <= len(names) else i
        out.append(dict(pick=lbl, sensor=names[k] if k < len(names) else f"pt{i + 1}", convention=conv,
                        work_mm_kji=w.round(3).tolist(), eye_xyz_mm=x.round(3).tolist(),
                        depth_below_surface_mm=round(float(depth), 2),
                        polar_from_anterior_pole_deg=round(float(np.degrees(np.arccos(np.clip(x[2] / np.linalg.norm(x), -1, 1)))), 1),
                        azimuth_deg=round(float(np.degrees(np.arctan2(x[1], x[0]))), 1),
                        radius_over_half_axial=round(float(np.linalg.norm(x) / R), 3),
                        axial_dist_from_anterior_pole_mm=round(float(F[2] @ (apex - w)), 2)))
    return out
