"""Lens, optic-nerve stump, optical axis, axial length, equatorial diameters and cornea.

Eye frame: origin = globe centroid, e_z = optical axis pointing ANTERIOR, (e_x, e_y) arbitrary
orthonormal completion. All lengths in mm on the downsampled grid (voxel s).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball

# plausible lens volume (mm^3) by group; outside this range the lens is not trusted for the axis
LENS_VOL = {"human": (100, 450), "fresh": (300, 900), "preserved": (100, 650)}


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def frame(ez):
    ez = unit(ez)
    t = np.array([1.0, 0, 0]) if abs(ez[0]) < 0.9 else np.array([0, 1.0, 0])
    ex = unit(np.cross(ez, t)); ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez])          # rows: e_x, e_y, e_z


def find_lens(n, lab, gl, s, group, sig=(2.0, 5.0)):
    inner = ndi.distance_transform_edt(gl) * s
    t = n[(lab == 2) & (inner > 3)]
    med = np.median(t); mad = np.median(np.abs(t - med)) + 1e-6
    z = np.clip((n - med) / mad, -6, 6); z[~gl] = 0
    bad = ndi.binary_dilation(lab == 5, iterations=max(1, int(round(1 / s)))) | (lab == 4)
    z[bad] = 0
    dog = ndi.gaussian_filter(z, sig[0] / s) - ndi.gaussian_filter(z, sig[1] / s)
    dog[(inner < 1.0) | (inner > 9)] = -1e9
    p = np.unravel_index(dog.argmax(), dog.shape)
    sm = ndi.gaussian_filter(z, 0.4 / s)
    zz, yy, xx = np.ogrid[:n.shape[0], :n.shape[1], :n.shape[2]]
    R2 = ((zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2) * s * s
    core = np.median(sm[(R2 < 1.5 ** 2) & gl & ~bad])
    ring = np.median(sm[(R2 > 8 ** 2) & (R2 < 11 ** 2) & gl & ~bad & (inner > 2)])
    m = (sm > (core + ring) / 2) & (R2 < 7.5 ** 2) & gl & ~bad & (inner > 1.5)
    m = ndi.binary_opening(m, structure=ball(max(1, int(round(1.0 / s)))))
    L, nl = ndi.label(m)
    if nl == 0:
        return None, dict(found=False)
    lp = L[p] if L[p] else np.bincount(L.ravel())[1:].argmax() + 1
    m = ndi.binary_fill_holes(L == lp)
    vol = float(m.sum() * s ** 3)
    lo, hi = LENS_VOL[group]
    info = dict(found=True, contrast_mad=round(float(core - ring), 2), vol_mm3=round(vol, 1),
                centroid_mm=(np.argwhere(m).mean(0) * s).round(3).tolist())
    info["trusted"] = bool(lo <= vol <= hi and info["contrast_mad"] > 1.5)
    holder = np.isin(lab, [6, 7])
    if holder.any():
        # a bright crescent just inside the wall where the globe presses on holder plastic is a contact
        # artefact, not a lens (human eyes 2, 5-9): real lenses sit > 3.5 mm (median) from any holder part
        dh = float(np.median((ndi.distance_transform_edt(~holder) * s)[m]))
        info["holder_dist_median_mm"] = round(dh, 2)
        if dh < 3.5:
            info["trusted_but_for_holder_rule"] = info["trusted"]
            info["trusted"] = False; info["rejected"] = "holder-contact artefact"
    return m, info


def nerve_attachment(parts, gl, s):
    cand = [p for p in parts if p["cls"] == "attached_tissue" and p["touches_globe"] and p["vol_mm3"] > 30]
    if not cand:
        return None
    p = max(cand, key=lambda q: q["vol_mm3"])
    c = np.array(p["centroid_mm"]) / s
    surf = np.argwhere(gl & ~ndi.binary_erosion(gl))
    a = surf[np.argmin(((surf - c) ** 2).sum(1))] * s
    return dict(vol_mm3=p["vol_mm3"], centroid_mm=p["centroid_mm"], attach_mm=a.round(3).tolist())


def ring_axis(lab, gl, s, A):
    """Axis from the holder ring. The eye always sits on the ring the same way: nerve stump down through
    the hole, cornea up, so the normal of the ring plate is the eye axis and the plate side is posterior.
    The plate is the plane slab (+/- tol_mm) holding most of the holder plastic within near_mm of the globe
    (RANSAC, fixed seed, refined by PCA). It is accepted only if it sits like a ring around the posterior
    half: plate_to_centre_mm from the globe centroid and centred within max_off_centre_mm of it."""
    gc = np.argwhere(gl).mean(0) * s
    P = np.argwhere(np.isin(lab, [6, 7]) & (ndi.distance_transform_edt(~gl) * s < A["near_mm"])) * s
    if len(P) < 1000:
        return None
    rng = np.random.default_rng(0)
    P = P[rng.choice(len(P), min(len(P), 20000), replace=False)]
    best, plane = 0, None
    for _ in range(A["trials"]):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        nrm = np.cross(b - a, c - a); L = np.linalg.norm(nrm)
        if L < 1e-6:
            continue
        k = int((np.abs((P - a) @ (nrm / L)) < A["tol_mm"]).sum())
        if k > best:
            best, plane = k, (a, nrm / L)
    a, nrm = plane
    for _ in range(3):
        Q = P[np.abs((P - a) @ nrm) < A["tol_mm"]]; a = Q.mean(0)
        nrm = np.linalg.eigh(np.cov((Q - a).T))[1][:, 0]
    if nrm @ (gc - a) < 0:
        nrm = -nrm                                   # anterior = from the plate towards the globe centre
    out = dict(normal=nrm.round(4).tolist(), plate_centre_mm=a.round(3).tolist(), inlier_fraction=round(len(Q) / len(P), 2),
               plate_to_centre_mm=round(float((gc - a) @ nrm), 2), off_centre_mm=round(float(np.linalg.norm(np.cross(gc - a, nrm))), 2))
    lo, hi = A["plate_to_centre_mm"]
    out["accepted"] = bool(lo <= out["plate_to_centre_mm"] <= hi and out["off_centre_mm"] <= A["max_off_centre_mm"])
    return out


def ray_exit(gl, s, origin_mm, d, step=0.05, rmax=40):
    """Distance from origin to where the ray leaves the globe mask for good (gaps < 0.5 mm ignored)."""
    t = np.arange(0, rmax, step)
    P = (np.asarray(origin_mm)[None] + t[:, None] * d[None]) / s
    inside = ndi.map_coordinates(gl.astype(np.uint8), P.T, order=0, mode="constant") > 0
    last, gap = 0.0, 0
    for ti, ins in zip(t, inside):
        if ins:
            last, gap = ti, 0
        else:
            gap += 1
            if gap * step > 0.5:
                break
    return last


def calipers(Q2):
    """Max / min caliper width of a 2-D point set (mm)."""
    ang = np.deg2rad(np.arange(0, 180, 1.0))
    W = np.array([np.ptp(Q2 @ np.array([np.cos(a), np.sin(a)])) for a in ang])
    return float(W.max()), float(W.min())


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]; b = (P ** 2).sum(1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]; R = np.sqrt(x[3] + (c ** 2).sum())
    return c, R


def analyse(n, lab, gl, s, group, parts, sig, lens_given=None, axis_override=None, ring=None, opposite_mm=None,
            nerve_over_lens=True):
    """lens_given=(mask, info) skips CT lens detection (used for MRI, where the lens is segmented directly).
    ring = result of ring_axis; when accepted it sets the axis, and the lens / nerve cues are only reported.
    opposite_mm = a point known to be posterior (the optic-nerve sensor): a lens candidate rejected only by the
    holder-distance rule is accepted when it lies more than 110 deg away from that point, seen from the centroid
    (the contact artefacts of the ring-mounted eyes sit at the posterior plate; the rule itself shifts with the outline).
    nerve_over_lens=False keeps a trusted lens when it disagrees with the nerve-stump cue (used after the stump cue has
    failed the optic-sensor check)."""
    out = {}
    P = np.argwhere(gl) * s
    gc = P.mean(0)
    lens_m, lens = lens_given if lens_given is not None else find_lens(n, lab, gl, s, group, sig)
    nerve = nerve_attachment(parts, gl, s)
    if opposite_mm is not None and lens.get("trusted_but_for_holder_rule"):
        ang = float(np.degrees(np.arccos(np.clip(unit(np.array(lens["centroid_mm"]) - gc) @ unit(np.asarray(opposite_mm) - gc), -1, 1))))
        lens["angle_to_optic_sensor_deg"] = round(ang, 1)
        if ang > 110:
            lens["trusted"] = True; lens["rejected"] = None; lens["holder_rule_overruled"] = "optic-nerve sensor on the opposite side"
    out["lens"] = lens; out["nerve"] = nerve

    # ---- optical axis -------------------------------------------------------------------
    cues = {}
    if lens.get("trusted"):
        lc = np.array(lens["centroid_mm"])
        if 3.0 < np.linalg.norm(lc - gc) < 12:
            cues["lens"] = unit(lc - gc)
    if nerve:
        cues["nerve"] = unit(gc - np.array(nerve["attach_mm"]))
    ang = None
    if "lens" in cues and "nerve" in cues:
        ang = float(np.degrees(np.arccos(np.clip(cues["lens"] @ cues["nerve"], -1, 1))))
    if ring is not None and ring["accepted"]:
        cues["ring"] = np.array(ring["normal"])
    if axis_override is not None:
        ez, src = unit(axis_override[0]), axis_override[1]
    elif "ring" in cues:
        ez, src = cues["ring"], "holder_ring"
    elif nerve_over_lens and "nerve" in cues and ang is not None and ang > 60 and group == "human":   # pig lenses are validated; keep them
        ez, src = cues["nerve"], "optic_nerve(lens disagreed)"      # the stump is the harder landmark
    elif "lens" in cues:
        ez, src = cues["lens"], "lens"
    elif "nerve" in cues:
        ez, src = cues["nerve"], "optic_nerve"
    else:
        w, V = np.linalg.eigh(np.cov((P - gc).T))
        ez, src = V[:, 0], "pca_shortest_axis(sign_arbitrary)"
    out["axis"] = dict(source=src, ez=ez.round(4).tolist())
    if ring is not None:
        out["axis"]["ring"] = ring
    for k in ("lens", "nerve"):                  # the other cues, as a cross-check of the axis used
        if k in cues and src != ("optic_nerve" if k == "nerve" else k):
            out["axis"][f"{k}_cue_vs_axis_deg"] = round(float(np.degrees(np.arccos(np.clip(cues[k] @ ez, -1, 1)))), 1)
    if "lens" in cues and "nerve" in cues:
        out["axis"]["lens_vs_nerve_deg"] = round(float(np.degrees(np.arccos(np.clip(cues["lens"] @ cues["nerve"], -1, 1)))), 1)
    F = frame(ez)
    Q = (P - gc) @ F.T                           # globe voxels in eye frame

    # ---- axial length along the axis through the lens centre (else the centroid) ---------
    out["axis"]["independent_of_sensors"] = not src.startswith("optic_sensor")
    o = np.array(lens["centroid_mm"]) if src == "lens" else gc
    a_ant = ray_exit(gl, s, o, ez); a_post = ray_exit(gl, s, o, -ez)
    apex = o + a_ant * ez; ppole = o - a_post * ez
    out["axial_length_mm"] = round(a_ant + a_post, 2)
    out["anterior_pole_mm"] = apex.round(3).tolist(); out["posterior_pole_mm"] = ppole.round(3).tolist()

    # ---- equatorial diameters: widest slab perpendicular to the axis -------------------
    zb = np.round(Q[:, 2] / 0.5).astype(int)
    vals, cnt = np.unique(zb, return_counts=True)
    zmax = vals[cnt.argmax()]
    sl = Q[np.abs(Q[:, 2] - zmax * 0.5) <= 0.25][:, :2]
    wmax, wmin = calipers(sl)
    area = cnt.max() * s ** 3 / 0.5
    out["equatorial"] = dict(max_mm=round(wmax + s, 2), min_mm=round(wmin + s, 2),
                             equiv_circle_mm=round(2 * np.sqrt(area / np.pi), 2),
                             plane_z_from_centroid_mm=round(zmax * 0.5, 2))

    # ---- cornea: anterior cap protruding from a sphere fitted to the scleral surface --------
    surf = np.argwhere(gl & ~ndi.binary_erosion(gl)) * s
    S = (surf - gc) @ F.T
    th = np.degrees(np.arccos(np.clip(S[:, 2] / np.linalg.norm(S, axis=1), -1, 1)))
    c0, R0 = fit_sphere(S[th > 60])
    res = np.linalg.norm(S - c0, axis=1) - R0
    bins = np.arange(0, 62, 2)
    medres = np.array([np.median(res[(th >= b) & (th < b + 2)]) if ((th >= b) & (th < b + 2)).sum() > 5 else np.nan
                       for b in bins])
    cornea = dict(resolved=False, apex_protrusion_mm=round(float(np.nanmax(medres[:3])), 2))
    corn_m = None
    if np.nanmax(medres[:3]) > 0.4:
        k = 0
        while k < len(bins) and (np.isnan(medres[k]) or medres[k] > 0.25):
            k += 1
        lim = bins[min(k, len(bins) - 1)]
        ring = S[(th >= lim - 2) & (th < lim + 2)]
        cap = S[th < lim]
        if len(ring) > 10 and len(cap) > 20 and lim < 60:
            rr = np.linalg.norm(ring[:, :2], axis=1)
            zl = float(np.median(ring[:, 2])); zap = float((apex - gc) @ F[2])
            cc, Rc = fit_sphere(cap)
            cornea = dict(resolved=True, limbus_polar_deg=float(lim), diameter_mm=round(2 * float(np.median(rr)), 2),
                          sagittal_height_mm=round(zap - zl, 2), radius_of_curvature_mm=round(float(Rc), 2),
                          apex_protrusion_mm=cornea["apex_protrusion_mm"])
            # label: shell within 1 mm of the outer surface, anterior to the limbus plane
            zv = (np.argwhere(gl) * s - gc) @ F[2]
            depth = ndi.distance_transform_edt(gl) * s
            idx = np.argwhere(gl)
            sel = (zv > zl) & (depth[tuple(idx.T)] <= 1.0)
            corn_m = np.zeros_like(gl); corn_m[tuple(idx[sel].T)] = True
    out["cornea"] = cornea
    out["scleral_sphere_radius_mm"] = round(float(R0), 2)

    # ---- lens geometry in the eye frame + anterior chamber depth ---------------------
    if lens.get("found"):
        L = (np.argwhere(lens_m) * s - gc) @ F.T
        lw, _ = calipers(L[:, :2])
        lens.update(thickness_along_axis_mm=round(float(np.ptp(L[:, 2])) + s, 2), diameter_mm=round(lw + s, 2),
                    apex_to_lens_front_mm=round(float((apex - gc) @ F[2] - L[:, 2].max()), 2))
    out["frame"] = dict(origin_mm=gc.round(3).tolist(), axes=F.round(4).tolist())
    return out, lens_m, corn_m
