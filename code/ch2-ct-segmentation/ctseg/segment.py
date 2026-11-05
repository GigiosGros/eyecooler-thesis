"""Intensity calibration and multi-label segmentation of one downsampled scan.

Labels (uint8):
  0 background/air   1 medium (liquid bath)   2 globe tissue   3 globe fluid (sub-threshold, liquid-filled)
  4 globe air void   5 thermistor metal       6 tube wall       7 plastic holder
  8 attached tissue (optic nerve / muscle / fat stump)   9 other small solid
 10 lens   11 cornea   (only written when params.label_lens_cornea is true; off by default.
                         The MRI lens, which is segmented directly, is always label 10.)

Globe outline (params.globe_surface.method):
  "smooth_star"  one smooth closed surface r(direction) around the globe centre, fitted to the tissue
                 edges found along radial rays (see star_surface). Used for fresh and human eyes.
  "opening"      legacy: 3 mm morphological opening of the solid mask. Always used for the collapsed
                 preserved eyes (shell: true), whose real dents a smooth surface would bridge.
Corrections drawn by hand on the outline-evidence pictures (<ID>_outline_marks_LM.csv, see outline_marks.py) are
applied on top of either outline (apply_marks): the surface is moved onto the marked points and left as it is
away from them. For an opening-based outline the mask is first written as r(direction) (mask_to_star).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y
from skimage.filters import threshold_multiotsu

NAMES = {0: "background", 1: "medium", 2: "globe_tissue", 3: "globe_fluid", 4: "globe_air_void",
         5: "thermistor_metal", 6: "tube_wall", 7: "plastic_holder", 8: "attached_tissue",
         9: "other_solid", 10: "lens", 11: "cornea", 12: "optic_nerve_stump", 13: "aqueous", 14: "iris_ciliary_body",
         15: "lens_nucleus", 16: "intraocular_membranes", 17: "extraocular_tissue", 18: "retina_choroid"}
COLORS = {0: (0, 0, 0), 1: (.15, .3, .35), 2: (.75, .75, .75), 3: (.4, .6, .9), 4: (1, 0, 1),
          5: (1, 0, 0), 6: (.1, .5, .9), 7: (1, .75, .1), 8: (.3, .8, .3), 9: (.6, .3, .1),
          10: (0, 1, 1), 11: (1, .5, .5), 12: (.1, .55, .1), 13: (.75, .9, 1), 14: (.55, .3, .7),
          15: (0, .6, .7), 16: (.9, .6, .8), 17: (.45, .65, .35), 18: (1, .3, .1)}


def edt_open(m, r):
    """Morphological opening with a ball of radius r voxels, via distance transforms (fast for large r)."""
    return ndi.distance_transform_edt(~(ndi.distance_transform_edt(m) > r)) <= r


def edt_close(m, r):
    return ~edt_open(~m, r)


def fill3(m):
    """3-D hole fill followed by 2-D fills along each axis (closes voids that touch one slice plane)."""
    m = ndi.binary_fill_holes(m)
    for a in range(3):
        m = np.moveaxis(np.stack([ndi.binary_fill_holes(q) for q in np.moveaxis(m, a, 0)]), 0, a)
    return m


def _sh_basis(th, ph, L):
    """Real spherical harmonics up to degree L at polar angle th, azimuth ph; also the degree of each column."""
    cols, deg = [], []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), th, ph)
            cols.append(Y.real if m == 0 else np.sqrt(2) * (Y.real if m > 0 else Y.imag)); deg.append(l)
    return np.stack(cols, 1), np.array(deg)


def _fit_sh(B, deg, r, w, lam):
    """Weighted ridge fit of r on the basis B; the penalty lam * (l(l+1))^2 damps high degrees."""
    ok = np.isfinite(r) & (w > 0)
    A = B[ok] * w[ok, None]
    pen = lam * (deg * (deg + 1.0)) ** 2 * ok.sum() / len(r) + 1e-9
    return np.linalg.solve(A.T @ B[ok] + np.diag(pen), A.T @ r[ok])


def star_surface(n, s, thr, c0_vox, G):
    """Globe outline as one smooth closed surface r(direction) around the globe centre.

    Along radial rays every tissue -> non-tissue crossing is a candidate edge. A stiff, robust baseline
    (spherical harmonics of low degree, outward residuals cut harder) says which candidate is the globe
    wall; a smoother of higher degree then follows the wall where an edge lies within tol_mm, and bridges
    the directions with no usable edge: metal streaks and insertion tracks (edge too far in), holder
    plastic, ring and nerve stump of tissue density (edge too far out or absent). Bumps and notches
    narrower than 2 * morph_deg are finally removed by a grey opening and closing on the sphere.
    Returns the centre (mm), r on a (polar, azimuth) grid and which directions carry a measured edge.
    """
    nth, nph = int(round(180 / G["ray_step_deg"])), int(round(360 / G["ray_step_deg"]))
    th = (np.arange(nth) + 0.5) * np.pi / nth; ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
    TH, PH = (q.ravel() for q in np.meshgrid(th, ph, indexing="ij"))
    d = np.stack([np.cos(TH), np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH)], 1)     # (slice, row, col)
    aw = np.sin(TH)                                                                    # solid angle per ray
    step = 0.5 * s; rr = np.arange(0, G["ray_max_mm"], step); idx = np.arange(len(rr))
    B, deg = _sh_basis(TH, PH, G["degree"])
    tol, band = G["tol_mm"], G["band_mm"]
    c = np.asarray(c0_vox, float) * s
    for outer in range(2):                               # second pass from the centroid of the first surface
        pts = (c[None, None] + rr[None, :, None] * d[:, None]) / s
        ins = ndi.map_coordinates(n, pts.reshape(-1, 3).T, order=1, mode="constant", cval=0).reshape(len(d), -1) > thr
        nxt = np.empty(ins.shape, int); last = np.full(len(d), len(rr))
        for k in range(len(rr) - 1, -1, -1):             # index of the next tissue sample at or after k
            last = np.where(ins[:, k], k, last); nxt[:, k] = last
        gap = (nxt[:, 1:] - idx[None, 1:]) * step        # length of the non-tissue run that starts at k + 1
        cand = ins[:, :-1] & ~ins[:, 1:] & (gap >= G["gap_min_mm"]) & (rr[None, :-1] > G["r_min_mm"])
        first = cand & (gap >= G["gap_first_mm"])
        r = np.where(first.any(1), rr[first.argmax(1)], np.nan); w = aw * np.isfinite(r)

        def pick(ref, lim):                              # candidate edge nearest to ref, if within lim
            dist = np.where(cand, np.abs(rr[None, :-1] - ref[:, None]), np.inf)
            j = dist.argmin(1)
            return np.where(dist[np.arange(len(d)), j] < lim, rr[j], np.nan)

        for it, Lk in enumerate(G["baseline_degrees"]):
            sel = deg <= Lk
            base = B[:, sel] @ _fit_sh(B[:, sel], deg[sel], np.nan_to_num(r), w, G["baseline_ridge"])
            r = pick(base, np.inf if it < 2 else 2 * band); res = r - base
            sc = max(0.3, 1.4826 * np.nanmedian(np.abs(res[np.isfinite(res)])))
            u = np.clip(np.abs(res) / (np.where(res > 0, 2.0, 3.0) * sc), 0, 1)
            w = aw * np.where(np.isfinite(res), (1 - u ** 2) ** 2, 0)
        fit = base.copy()
        for it in range(4):
            r = pick(fit, tol if it else band); r = np.where(np.abs(r - base) < band, r, np.nan)
            u = np.clip(np.abs(r - fit) / (2 * tol), 0, 1)
            w = aw * np.where(np.isfinite(r), (1 - u ** 2) ** 2, 0)
            fit = base + B @ _fit_sh(B, deg, np.nan_to_num(r - base), w, G["ridge"])
        r = pick(fit, tol); good = np.isfinite(r)
        rf = np.where(good, r, fit)
        if outer == 0:
            c = c + (0.75 * (rf ** 4 * aw)[:, None] * d).sum(0) / (rf ** 3 * aw).sum()
    nb = cKDTree(d).query_ball_point(d, 2 * np.sin(np.deg2rad(G["morph_deg"]) / 2))
    flat = np.concatenate(nb); st = np.r_[0, np.cumsum([len(q) for q in nb])[:-1]]
    ero = lambda x: np.minimum.reduceat(x[flat], st); dil = lambda x: np.maximum.reduceat(x[flat], st)
    q0 = rf - base
    q1 = dil(ero(q0)); q1 = np.where(q0 - q1 > G["morph_min_mm"], q1, q0)      # narrow outward bumps
    q2 = ero(dil(q1)); q2 = np.where(q2 - q1 > G["morph_min_mm"], q2, q1)      # narrow inward notches
    R = (base + q2).reshape(nth, nph)
    R = ndi.median_filter(np.concatenate([R[:, -3:], R, R[:, :3]], 1), size=3, mode="nearest")[:, 3:-3]
    good = np.isfinite(pick(R.ravel(), G["morph_min_mm"]))      # the final surface lies on a tissue edge
    return dict(center_mm=c, R=R, measured=good.reshape(nth, nph).astype(np.int8))


def _rays(nth, nph):
    th = (np.arange(nth) + 0.5) * np.pi / nth; ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
    TH, PH = (q.ravel() for q in np.meshgrid(th, ph, indexing="ij"))
    return np.stack([np.cos(TH), np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH)], 1)


def mask_to_star(gl, s, G):
    """A globe mask written as r(direction) around its centroid: the outermost mask voxel on each ray.
    Exact for a star-shaped mask; 'star_volume_ratio' in the returned dict says how close this one is."""
    nth, nph = int(round(180 / G["ray_step_deg"])), int(round(360 / G["ray_step_deg"]))
    d = _rays(nth, nph); c = np.argwhere(gl).mean(0) * s
    rr = np.arange(0, G["ray_max_mm"] + 6, 0.5 * s)
    pts = ((c[None, None] + rr[None, :, None] * d[:, None]) / s).reshape(-1, 3).T
    ins = ndi.map_coordinates(gl.astype(np.uint8), pts, order=0, mode="constant", cval=0).reshape(len(d), -1) > 0
    R = np.where(ins.any(1), rr[ins.shape[1] - 1 - ins[:, ::-1].argmax(1)], 0.0).reshape(nth, nph)
    surf = dict(center_mm=c, R=R, measured=np.ones((nth, nph), np.int8))
    surf["star_volume_ratio"] = round(float(star_mask(gl.shape, s, surf).sum() / gl.sum()), 3)
    return surf


def apply_marks(surf, marks_mm, G, n=None, s=None, thr=None):
    """Move the surface onto the points marked by hand as the true wall. Each mark has a residual (its radius minus
    the surface radius in its direction). A direction takes the Gaussian-weighted mean residual of the marks
    within marks_reach_deg; the Gaussian is narrow next to a mark (marks_sigma_min_deg) and widens with the
    distance to the nearest mark, so the surface goes through the marks and blends smoothly between the
    sections they were drawn on. The correction is applied in full up to marks_full_deg from the nearest mark
    and fades to zero at marks_reach_deg. Directions moved by more than 0.1 mm get measured = 2.
    Rows of kind 0 (4th column) are outline left untouched: they ask for no change and stop a correction
    from spreading sideways.
    A keep anchor is overruled by a mark that moves the wall on another section within marks_overrule_deg
    (5th column = section number): explicit corrections beat untouched outline.

    Overhang: r(direction) holds one radius per direction. Where the drawn wall is the underside of a flap of
    tissue leaning over a dent, the scan shows tissue just outside the mark and none just inside it (tested
    marks_side_mm either side along the ray, with the tissue threshold thr on the normalised volume n). The
    flap top is the existing outline, so such marks ask for no change (the pocket under the flap stays inside)."""
    nth, nph = surf["R"].shape; d = _rays(nth, nph)
    marks_mm = np.asarray(marks_mm, float)
    kind = marks_mm[:, 3] if marks_mm.shape[1] > 3 else np.ones(len(marks_mm))
    sec = marks_mm[:, 4] if marks_mm.shape[1] > 4 else np.zeros(len(marks_mm)); marks_mm = marks_mm[:, :3]
    q = marks_mm - surf["center_mm"]; rm = np.linalg.norm(q, axis=1); u = q / rm[:, None]
    res = np.where(kind > 0, rm - star_lookup(surf, marks_mm)[0], 0.0)     # kind 0: outline left untouched -> no change
    chord = lambda deg: 2 * np.sin(np.deg2rad(deg) / 2)
    use = res.copy(); n_flap = 0
    if n is not None:
        side = lambda sign: ndi.map_coordinates(n, ((surf["center_mm"] + u * (rm + sign * G["marks_side_mm"])[:, None]) / s).T,
                                                order=1, mode="constant", cval=0) > thr
        flap = side(+1) & ~side(-1) & (res < 0)
        use[flap] = 0.0; n_flap = int(flap.sum())
    # explicit beats implicit: a keep anchor gives way to a mark that moves the wall on ANOTHER section within
    # marks_overrule_deg (that place was corrected there and simply not redrawn on this section)
    mov = (kind > 0) & (np.abs(use) > G["marks_move_mm"]); drop = np.zeros(len(u), bool)
    if mov.any() and (kind == 0).any():
        um, sm = u[mov], sec[mov]; ia = np.flatnonzero(kind == 0)
        for k, idx in zip(ia, cKDTree(um).query_ball_point(u[ia], chord(G["marks_overrule_deg"]))):
            drop[k] = bool(idx) and bool((sm[idx] != sec[k]).any())
    u, use = u[~drop], use[~drop]
    tree = cKDTree(u)
    delta = np.zeros(len(d)); frac = np.zeros(len(d))
    for k, idx in enumerate(tree.query_ball_point(d, chord(G["marks_reach_deg"]))):
        if idx:
            ang = np.degrees(np.arccos(np.clip(u[idx] @ d[k], -1, 1)))
            w = np.exp(-0.5 * (ang / max(G["marks_sigma_min_deg"], G["marks_sigma_rel"] * ang.min())) ** 2)
            t = np.clip((ang.min() - G["marks_full_deg"]) / (G["marks_reach_deg"] - G["marks_full_deg"]), 0, 1)
            frac[k] = 0.5 * (1 + np.cos(np.pi * t)); delta[k] = frac[k] * (w @ use[idx]) / w.sum()
    surf["R"] = np.maximum(surf["R"] + delta.reshape(nth, nph), 1.0)
    surf["measured"] = np.where(np.abs(delta.reshape(nth, nph)) > 0.1, 2, surf["measured"]).astype(np.int8)
    m = kind > 0
    surf["marks"] = dict(n=int(m.sum()), n_keep_anchors=int((~m).sum()), n_keep_anchors_overruled=int(drop.sum()),
                         n_flap_underside=n_flap,
                         correction_mm_median=round(float(np.median(res[m])), 2),
                         correction_mm_p10_p90=[round(float(x), 2) for x in np.percentile(res[m], [10, 90])])
    return surf


def smooth_surface(surf, G, marks_mm=None):
    """Soft smoother: Gaussian average of r over the sphere (width smooth_deg), to take the voxel and
    hand-drawing jaggedness out of the outline. Records how far the marks end up from the final surface."""
    if G.get("smooth_deg", 0) > 0:
        nth, nph = surf["R"].shape; d = _rays(nth, nph); R = surf["R"].ravel(); out = np.empty_like(R)
        for k, idx in enumerate(cKDTree(d).query_ball_point(d, 2 * np.sin(np.deg2rad(2.5 * G["smooth_deg"]) / 2))):
            ang = np.degrees(np.arccos(np.clip(d[idx] @ d[k], -1, 1)))
            w = np.exp(-0.5 * (ang / G["smooth_deg"]) ** 2); out[k] = (w @ R[idx]) / w.sum()
        surf["R"] = out.reshape(nth, nph)
    if marks_mm is not None and "marks" in surf:
        marks_mm = np.asarray(marks_mm, float)
        marks_mm = marks_mm[marks_mm[:, 3] > 0, :3] if marks_mm.shape[1] > 3 else marks_mm
        left = np.abs(np.linalg.norm(marks_mm - surf["center_mm"], axis=1) - star_lookup(surf, marks_mm)[0])
        surf["marks"].update(left_after_mm_median_abs=round(float(np.median(left)), 2),
                             left_after_mm_p95_abs=round(float(np.percentile(left, 95)), 2))
    return surf


def star_lookup(surf, p_mm, order=1):
    """Surface radius (order 1) or measured flag (order 0: 0 bridged, 1 on a tissue edge, 2 set by the hand-drawn marks)
    in the direction of points p_mm (N, 3)."""
    q = np.asarray(p_mm, float) - surf["center_mm"]
    r = np.linalg.norm(q, axis=1) + 1e-9
    A = surf["R"] if order else surf["measured"].astype(np.float32)
    nth, nph = A.shape
    Ap = np.concatenate([A[:, -1:], A, A[:, :1]], 1)
    th = np.arccos(np.clip(q[:, 0] / r, -1, 1)); ph = np.mod(np.arctan2(q[:, 2], q[:, 1]), 2 * np.pi)
    return ndi.map_coordinates(Ap, [th / np.pi * nth - 0.5, ph / (2 * np.pi) * nph + 0.5], order=order, mode="nearest"), r


def star_plane(surf, origin_mm, a, b, t):
    """R(direction) - r on the plane origin + u a + v b (u, v in t): positive inside the surface. Its zero
    contour is the outline in that plane, free of the voxel staircase a mask shows where the plane grazes it."""
    U, V = np.meshgrid(t, t, indexing="ij")
    P = (np.asarray(origin_mm)[None, None] + U[..., None] * a + V[..., None] * b).reshape(-1, 3)
    Rv, r = star_lookup(surf, P)
    return (Rv - r).reshape(U.shape)


def star_mask(shape, s, surf):
    """Voxels inside the surface."""
    c = surf["center_mm"]; rm = float(surf["R"].max()) + 1
    lo = np.maximum(np.floor((c - rm) / s).astype(int), 0); hi = np.minimum(np.ceil((c + rm) / s).astype(int) + 1, shape)
    g = np.stack(np.meshgrid(*[np.arange(lo[a], hi[a]) * s for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
    Rv, r = star_lookup(surf, g)
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (r <= Rv).reshape(hi - lo)
    return m


def nerve_stump(lab, gl, surf, s, seed_mm, step=0.25, half_mm=5.5, length_mm=30.0, cap_factor=1.35, start_cap_mm=3.2):
    """Optic-nerve stump behind the globe, followed outward from the wall at the optic-nerve sensor (seed_mm).
    The globe is not touched. At each step along the current direction the cross-section of the solid outside the
    globe is taken; the blob on the centre line is the stump, its centroid bends the line, and its equivalent radius
    is capped at cap_factor x the median radius of the first 2 mm, so that where the stump touches holder plastic of
    the same density only a disc of the stump's own size is kept. The stump ends where the solid ends, where it thins to less than
    0.45 x its early radius (a thermistor lead or a streak, not nerve), or, looking back from the end, where the
    cross-section stayed more than twice the cap for the last 1.5 mm or more (it ran into the tube or a plate).
    Returns the mask and its length, diameter, volume."""
    solid = np.isin(lab, [6, 7, 8, 9, 12]) | ((lab == 5) & ~gl)
    c = surf["center_mm"]; d = (np.asarray(seed_mm) - c) / np.linalg.norm(np.asarray(seed_mm) - c)
    p = c + d * float(star_lookup(surf, (c + d * 5.0)[None])[0][0])                    # start on the wall
    t = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, t); u /= np.linalg.norm(u); w = np.cross(d, u)
    g = np.arange(-half_mm, half_mm + 1e-6, s); U, W = np.meshgrid(g, g, indexing="ij"); rad = np.hypot(U, W)
    radii, pts, big = [], [], []; n2 = int(2.0 / step)
    for k in range(int(length_mm / step)):
        q = p + d * step
        P = (q[None, None] + U[..., None] * u + W[..., None] * w) / s
        sec = ndi.map_coordinates(solid.astype(np.uint8), P.reshape(-1, 3).T, order=0, mode="constant").reshape(U.shape) > 0
        L, n = ndi.label(sec)
        dist = np.where(L > 0, rad, np.inf); i0 = np.unravel_index(dist.argmin(), dist.shape)
        if n == 0 or dist[i0] > 1.5:
            break
        blob = L == L[i0]; r_eq = float(np.sqrt(blob.sum() * s * s / np.pi))
        cap = cap_factor * float(np.median(radii[:n2])) if len(radii) >= n2 else start_cap_mm
        if len(radii) >= n2 and r_eq < 0.45 * float(np.median(radii[:n2])):
            break                                                         # thinned to a wire or a streak
        big.append(r_eq > 2.0 * cap)
        if r_eq > cap:
            blob &= rad <= cap; r_eq = cap; cen = np.zeros(2)
        else:
            cen = np.array([U[blob].mean(), W[blob].mean()])
        p = q + 0.5 * (cen[0] * u + cen[1] * w)
        radii.append(r_eq); pts.append(q[None] + U[blob][:, None] * u + W[blob][:, None] * w)
    k = len(big)
    while k > 0 and big[k - 1]:
        k -= 1
    if (len(big) - k) * step >= 1.5:                                      # the end ran into something much larger
        radii, pts = radii[:k], pts[:k]
    m = np.zeros(lab.shape, bool)
    if pts:
        ijk = np.round(np.concatenate(pts) / s).astype(int); ok = ((ijk >= 0) & (ijk < np.array(lab.shape))).all(1)
        m[tuple(ijk[ok].T)] = True
        m = ndi.binary_closing(m, iterations=2) & solid & ~gl & (lab != 5)
    info = dict(found=bool(m.any()), length_mm=round(len(radii) * step, 1),
                diameter_mm_median=round(2 * float(np.median(radii)), 2) if radii else None,
                vol_mm3=round(float(m.sum() * s ** 3), 1), reached_end_of_search=len(radii) >= int(length_mm / step),
                note="followed outward from the globe wall at the optic-nerve sensor; radius capped where it touches the holder")
    return m, info


def calibrate(v, s, scan):
    """Locate the globe as the thickest solid blob and sample tissue (T) and background (B) levels."""
    g = ndi.gaussian_filter(v, 1)
    nz = g[g > 0]
    lo, hi = np.percentile(nz, [0.5, 99.95])
    ts = threshold_multiotsu(np.clip(nz, lo, hi)[::5], classes=4)
    best = None
    for t in sorted(ts):
        m = g > t
        if scan.get("shell"):
            m = fill3(edt_close(m, 1.5 / s))
        m = ndi.binary_opening(ndi.binary_fill_holes(m), iterations=2)
        D = ndi.distance_transform_edt(m) * s
        if scan["group"] == "preserved":            # globe sits mid-tube; holder blocks at the tube ends
            N = D.shape[1]; D[:, :int(.3 * N)] = 0; D[:, int(.75 * N):] = 0
        r = D.max(); c = np.unravel_index(D.argmax(), D.shape)
        rmin = 4 if scan.get("shell") else 7          # collapsed (fixed) globes are thin shells
        if rmin < r < 14.5 and (best is None or r > best[1]):
            best = (t, r, c)
    if best is None:
        raise RuntimeError("calibration: no globe-sized blob found")
    t, r, c = best
    zz, yy, xx = np.ogrid[:g.shape[0], :g.shape[1], :g.shape[2]]
    ins = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) * s * s < (0.6 * r) ** 2
    T = float(np.median(g[ins]))
    h, b = np.histogram(nz[nz < T - 0.15 * (T - lo)], bins=400)
    B = float(b[h.argmax()])
    return dict(T=T, B=B, center=[int(x) for x in c], r_mm=float(r))


def segment(v, s, scan, cal, P):
    g = ndi.gaussian_filter(v, 1)
    n = (g - cal["B"]) / (cal["T"] - cal["B"])
    valid = v > 0
    if scan["bath"] == "air":
        M, ts = 0.0, 0.5
    else:
        h, b = np.histogram(n[(n > 0.3) & (n < 0.9) & valid], bins=60, range=(0.3, 0.9))
        M = float(b[h.argmax()] + 0.005); ts = (M + 1) / 2
    metal = ndi.binary_propagation(n > P["metal_n"], mask=n > P["metal_n_low"]) & \
        ndi.binary_dilation(n > P["metal_n"], iterations=2)
    solid = (n > ts) & ~metal
    base = solid | metal
    if scan.get("shell"):
        # close/fill the collapsed scleral shell only within 15 mm of the calibrated eye centre
        # (per-slice filling would otherwise fill whole tube cross-sections)
        cz = cal["center"]; zz, yy, xx = np.ogrid[:v.shape[0], :v.shape[1], :v.shape[2]]
        near = ((zz - cz[0]) ** 2 + (yy - cz[1]) ** 2 + (xx - cz[2]) ** 2) * s * s < 15.0 ** 2
        base = edt_close(base & near, 1.5 / s)
        f = fill3(base)
    else:
        f = base
    D = ndi.distance_transform_edt(ndi.binary_opening(f, iterations=2))
    if scan["group"] == "preserved":
        N = D.shape[1]; D[:, :int(.3 * N)] = 0; D[:, int(.75 * N):] = 0
    cen = np.unravel_index(D.argmax(), D.shape)
    lab, _ = ndi.label(edt_open(f, P["r_open_mm"] / s))
    gl = fill3(lab == lab[cen])
    G = P.get("globe_surface", {"method": "opening"}); surf = None
    method = "opening" if scan.get("shell") else G["method"]
    if method == "smooth_star":
        surf = star_surface(n, s, ts, np.argwhere(gl).mean(0), G)
    # Hand-drawn corrections, round by round: {1: marks, 2: marks, ...}. Round 1 was drawn on the automatic outline; the
    # outline then got the soft smoother (smooth_deg) and that is what the round-2 pictures showed, so round 2 is
    # applied after it. A last, slightly stronger smoother (smooth_final_deg) finishes every outline; rounds 3+
    # were drawn on that finished outline and come after it.
    rounds = scan.get("marks") if G.get("use_marks", True) else None
    rounds = rounds or {}
    if rounds and surf is None:
        surf = mask_to_star(gl, s, G); method += "+marks"
    if surf is None and G.get("smooth_deg", 0) > 0:
        surf = mask_to_star(gl, s, G)
    if surf is not None:
        log = {}
        if 1 in rounds:
            surf = apply_marks(surf, rounds[1], G, n, s, ts)
        surf = smooth_surface(surf, G, rounds.get(1))
        if "marks" in surf:
            log["round1"] = surf.pop("marks")
        if 2 in rounds:
            surf = apply_marks(surf, rounds[2], G, n, s, ts)
            log["round2"] = surf.pop("marks")
        if G.get("smooth_final_deg", 0) > 0:
            surf = smooth_surface(surf, dict(G, smooth_deg=G["smooth_final_deg"]))
        for k in sorted(q for q in rounds if q > 2):
            # drawn on the finished outline: applied after the final smoother; only the part this round moved is
            # smoothed again, so the outline already accepted stays exactly as it was
            before = surf["R"].copy()
            surf = apply_marks(surf, rounds[k], G, n, s, ts)
            log[f"round{k}"] = surf.pop("marks")
            if G.get("smooth_final_deg", 0) > 0:
                moved = np.clip(np.abs(surf["R"] - before) / 0.2, 0, 1); raw = surf["R"].copy()
                surf = smooth_surface(surf, dict(G, smooth_deg=G["smooth_final_deg"]))
                w = smooth_surface(dict(R=moved), dict(G, smooth_deg=G["smooth_final_deg"]))["R"]
                surf["R"] = w * surf["R"] + (1 - w) * raw
        for k, m in rounds.items():                          # how far each round's marks end up from the final outline
            mk = np.asarray(m, float); mk = mk[mk[:, 3] > 0, :3] if mk.shape[1] > 3 else mk
            left = np.abs(np.linalg.norm(mk - surf["center_mm"], axis=1) - star_lookup(surf, mk)[0])
            log[f"round{k}"].update(final_distance_mm_median=round(float(np.median(left)), 2),
                                    final_distance_mm_p95=round(float(np.percentile(left, 95)), 2))
        if log:
            surf["marks"] = log
        gl = star_mask(v.shape, s, surf)

    out = np.zeros(v.shape, np.uint8)
    if M > 0:
        out[valid & (n > 0.3) & ~solid & ~metal] = 1
    rest = solid & ~ndi.binary_dilation(gl, iterations=1)
    pl, _ = ndi.label(rest)
    parts = []
    for i, sl in enumerate(ndi.find_objects(pl)):
        m = pl[sl] == i + 1
        vol = m.sum() * s ** 3
        if vol < 1:
            out[sl][m] = 9; continue
        bb = np.array([q.stop - q.start for q in sl]) * s
        Pm = np.argwhere(m) * s
        sd = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(Pm.T))[::-1], 0))
        touch = bool((ndi.binary_dilation(m, iterations=2) & gl[sl]).any())
        if bb.max() > 0.8 * max(v.shape) * s and m.mean() < 0.25:
            cls = 6                                   # spans the field and hollow -> tube wall
        elif touch and sd[1] < 2.5 and sd[2] < 2.5 and sd[0] > 1.6 * sd[1] and scan["group"] != "preserved":
            cls = 8                                   # elongated cylinder on the globe -> nerve/muscle stump
        elif sd[2] < 1.5 or vol > 150:
            cls = 7                                   # flat or large -> holder plastic
        else:
            cls = 9
        out[sl][m] = cls
        if cls == 7 and touch and vol > 150 and scan["group"] != "preserved":
            # a nerve stump fused with a thin plate: erode the plate away (2 mm) and keep a thick elongated core
            core = edt_open(m, 2.0 / s)
            cl, nc = ndi.label(core)
            for j in range(1, nc + 1):
                cm = cl == j; cv = cm.sum() * s ** 3
                if cv < 40:
                    continue
                Pc = np.argwhere(cm) * s
                sdc = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(Pc.T))[::-1], 0))
                near_gl = (ndi.binary_dilation(cm, iterations=int(3.0 / s)) & gl[sl]).any()
                if near_gl and sdc[0] > 1.6 * sdc[1] and sdc[1] < 3.0:
                    grown = ndi.binary_dilation(cm, iterations=int(2.0 / s)) & m
                    out[sl][grown] = 8
                    Pg = np.argwhere(grown) * s
                    parts.append(dict(cls=NAMES[8], vol_mm3=round(float(grown.sum() * s ** 3), 1),
                                      bbox_mm=[], pca_sd_mm=sdc.round(2).tolist(), mean_n=None, touches_globe=True,
                                      centroid_mm=(Pg.mean(0) + np.array([q.start for q in sl]) * s).round(2).tolist(),
                                      note="split from a plate-fused part"))
        parts.append(dict(cls=NAMES[cls], vol_mm3=round(float(vol), 1), bbox_mm=bb.round(1).tolist(),
                          pca_sd_mm=sd.round(2).tolist(), mean_n=round(float(n[sl][m].mean()), 2),
                          touches_globe=touch,
                          centroid_mm=(Pm.mean(0) + np.array([q.start for q in sl]) * s).round(2).tolist()))
    out[gl & solid] = 2
    out[gl & ~solid & (n > 0.3)] = 3 if M > 0 else 4
    out[gl & (n <= 0.3)] = 4
    out[metal] = 5
    info = dict(medium_level=round(M, 3), solid_thr=round(ts, 3), center_vox=[int(x) for x in cen], parts=parts,
                globe_surface=dict(method=method))
    if surf is not None:
        aw = np.sin((np.arange(surf["R"].shape[0]) + 0.5) * np.pi / surf["R"].shape[0])[:, None] * np.ones_like(surf["R"])
        info["globe_surface"].update(center_mm=surf["center_mm"].round(3).tolist(),
                                     measured_fraction=round(float(((surf["measured"] == 1) * aw).sum() / aw.sum()), 3),
                                     marked_fraction=round(float(((surf["measured"] == 2) * aw).sum() / aw.sum()), 3),
                                     **{k: surf[k] for k in ("marks", "star_volume_ratio") if k in surf},
                                     radius_mm_min_mean_max=[round(float(x), 2) for x in
                                                             (surf["R"].min(), (surf["R"] * aw).sum() / aw.sum(), surf["R"].max())])
    return out, gl, n, info, surf
