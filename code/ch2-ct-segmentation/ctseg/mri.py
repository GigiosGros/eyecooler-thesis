"""Pig-eye MRI (gradient echo, 0.078 x 0.078 x 0.8 mm): globe + lens segmentation on an isotropic grid.

Contrast: vitreous white, aqueous light, lens grey cortex around a dark nucleus, eye wall, iris and ciliary body dark,
bath mid-grey, holder plastic and air without signal. Reworked 2026-10-07 (see segment()).
"""
import glob, os
import numpy as np
import pydicom
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from scipy.spatial import ConvexHull, cKDTree


def convex_hull_image(m):
    """3-D convex hull of a mask via the hull's facet equations (fast; voxel centres inside or on hull)."""
    surf = np.argwhere(m & ~ndi.binary_erosion(m))
    H = ConvexHull(surf)
    lo, hi = surf.min(0), surf.max(0) + 1
    out = np.zeros(m.shape, bool)
    zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    P = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1).astype(np.float32)
    inside = np.ones(len(P), bool)
    for eq in H.equations.astype(np.float32):
        inside &= P @ eq[:3] + eq[3] <= 0.5
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = inside.reshape(zz.shape)
    return out
from .segment import fill3, edt_open

ISO = 0.125      # mm, working grid (in-plane pixel 0.078 mm, slice 0.8 mm)

# Outline and lens surfaces use the same star-surface code as the CT (segment.star_surface); parameters here.
_G = dict(ray_step_deg=1.5, ray_max_mm=20.0, r_min_mm=3.0, gap_min_mm=0.3, gap_first_mm=1.5, baseline_degrees=[2, 2, 2, 4, 4, 4, 4],
          baseline_ridge=1e-4, degree=10, ridge=3e-4, tol_mm=0.6, band_mm=1.5, morph_deg=12.0, morph_min_mm=0.3, smooth_deg=3.0)
_GL = dict(_G, ray_step_deg=3.0, ray_max_mm=10.0, r_min_mm=1.0, gap_first_mm=0.6, baseline_degrees=[2] * 7, degree=6, ridge=1e-3,
           tol_mm=0.4, band_mm=1.0, morph_deg=15.0, morph_min_mm=0.2, smooth_deg=4.0)
WALL_MM = (0.1, 2.5)     # accepted range of the measured wall thickness
VOID = 0.25              # signal below VOID x bath level = no signal (holder plastic, air)
FAT_WATER_PPM = 3.5      # chemical shift of oil (CH2) against water, ppm
# The eyes were scanned immersed in oil (2026-10-07). Oil resonates 3.5 ppm from water, so the scanner draws the
# oil bath displaced along the readout direction by 3.5 ppm x frequency / bandwidth per pixel = 14.3 pixels = 1.1 mm
# (399.5 MHz, 97.7 Hz/pixel), towards lower row numbers ("up" in the stored slices). Water-bearing tissue is drawn in
# place. Result in the image: an empty band between the eye and the bath on the upper side, and bath signal drawn over
# the outer eye wall on the lower side. Everything bounded by oil (bath, holder) is moved back before it is used.


def load_series(d):
    ds = [pydicom.dcmread(f) for f in glob.glob(os.path.join(d, "*.dcm"))]
    o = np.array(ds[0].ImageOrientationPatient, float); nrm = np.cross(o[:3], o[3:])
    ds.sort(key=lambda q: float(np.dot(nrm, np.array(q.ImagePositionPatient, float))))
    v = np.stack([q.pixel_array.astype(np.float32) for q in ds])
    pos = np.array([q.ImagePositionPatient for q in ds], float)
    dz = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    ps = float(ds[0].PixelSpacing[0])
    f0, bw = float(ds[0].get("ImagingFrequency", 0) or 0), float(ds[0].get("PixelBandwidth", 0) or 0)
    shift = FAT_WATER_PPM * f0 / bw * ps if f0 and bw else 0.0             # Hz / (Hz per pixel) * mm per pixel
    meta = dict(n_slices=len(ds), pixel_mm=ps, slice_mm=dz, date=str(ds[0].get("StudyDate", "")),
                imaging_frequency_MHz=f0, pixel_bandwidth_Hz=bw, oil_shift_predicted_mm=round(shift, 3),
                TR=float(ds[0].get("RepetitionTime", 0) or 0), TE=float(ds[0].get("EchoTime", 0) or 0),
                flip=float(ds[0].get("FlipAngle", 0) or 0))
    return v, ps, dz, meta


def to_iso(v, ps, dz):
    v = ndi.gaussian_filter(v, (0, 0.6, 0.6))
    return ndi.zoom(v, (dz / ISO, ps / ISO, ps / ISO), order=1).astype(np.float32)


def _hull(m):
    """Convex hull of a mask, computed on every second voxel (fast) and brought back to the grid."""
    h = convex_hull_image(m[::2, ::2, ::2])
    h = np.repeat(np.repeat(np.repeat(h, 2, 0), 2, 1), 2, 2)[:m.shape[0], :m.shape[1], :m.shape[2]]
    out = np.zeros(m.shape, bool); out[:h.shape[0], :h.shape[1], :h.shape[2]] = h
    return out


def segment(v, collapsed=False, oil_shift_mm=0.0, slice_mm=0.8, slice_shift_mm=None):
    """Labels on the isotropic grid. Contrast (gradient echo): vitreous white, aqueous light, lens grey cortex around a
    dark nucleus, eye wall / iris / ciliary body dark, bath mid-grey, holder plastic and air without signal.

    What is found from the image contrast: vitreous, aqueous, lens (nucleus, cortex), the inner body of the eye
    (fluid + lens), no-signal parts (holder). What is assigned by position, because the tissues have the same signal:
    the globe outline = inner body + wall, with the wall thickness measured where the wall meets the bath and the
    local median elsewhere; cornea = wall over the aqueous, the rest of the wall = sclera with choroid and retina;
    iris and ciliary body = dark tissue inside the inner body next to the lens or the aqueous; other dark tissue
    inside = intraocular membranes (preserved eyes); optic-nerve stump = the largest tissue attached to the globe that
    reaches more than 4 mm away from it and is not in front.
    oil_shift_mm: displacement of the oil bath in the image (rows, towards lower row numbers); the bath and the holder
    are moved back by it, and the outer surface of the eye is taken where the corrected bath begins.
    slice_shift_mm: displacement of the oil across the slices; None = measured per eye in whole slices. The fresh
    eyes were all scanned with one protocol, so they must share one value: config mri.oil_slice_shift_mm.
    collapsed=True (preserved eyes: anterior chamber collapsed, retina detached and folded): aqueous, cornea and
    iris / ciliary body are NOT assigned, because position no longer tells them apart; bright fluid is one class and
    dark tissue inside the eye is "membranes"."""
    from . import segment as S
    s = ISO
    g = ndi.gaussian_filter(v, 0.5)
    t1, t2 = threshold_multiotsu(g[::3, ::3, ::3], classes=3)               # no signal | grey bath | white vitreous
    M = float(np.median(g[(g > t1) & (g < t2)])); W = float(np.median(g[g > t2]))
    bright = g > (M + W) / 2
    L, nl = ndi.label(edt_open(bright, 0.6 / s)); c = np.array(g.shape) / 2; best, score = 0, -1
    for i, sl in enumerate(ndi.find_objects(L)):                           # big bright blob near the centre
        m = L[sl] == i + 1; vol = m.sum() * s ** 3
        cen = np.array([(q.start + q.stop) / 2 for q in sl])
        sc = vol / (1 + np.linalg.norm((cen - c) * s) / 5)
        if vol > 500 and sc > score:
            best, score = i + 1, sc
    vit = L == best
    vhull = _hull(vit)
    # ---- lens: dark nucleus inside the vitreous hull, then its full extent out to the bright fluid around it
    dv = ndi.distance_transform_edt(~vit) * s
    Ll, _ = ndi.label(edt_open((g < 0.8 * M) & (dv < 6.0), 1.0 / s)); k, bs = 0, 0
    for i, sl in enumerate(ndi.find_objects(Ll)):
        m = Ll[sl] == i + 1; vol = m.sum() * s ** 3
        sc = vhull[sl][m].mean() * min(vol, 300) if vol >= 30 else 0
        if sc > bs:
            bs, k = sc, i + 1
    lens, linfo, lsurf = None, dict(found=False), None
    inner = vit.copy(); aq = np.zeros(g.shape, bool); nucleus = np.zeros(g.shape, bool)
    if bs > 15:
        nucleus = Ll == k
        lsurf = S.star_surface((~bright).astype(np.float32), s, 0.5, np.argwhere(nucleus).mean(0), _GL)
        # where the lens does not border bright fluid (against the cornea in a collapsed anterior chamber, or the
        # ciliary body), take the radius found on the opposite side: the lens is close to symmetric about its centre
        Rl, ml = lsurf["R"], lsurf["measured"] > 0; npn = Rl.shape[1]
        opp = lambda A: np.roll(A[::-1], npn // 2, axis=1)
        med = float(np.median(Rl[ml])) if ml.any() else float(np.median(Rl))
        lsurf["R"] = np.where(ml, Rl, np.where(opp(ml), opp(Rl), med))
        lsurf = S.smooth_surface(lsurf, _GL)
        lens = S.star_mask(g.shape, s, lsurf)
        near = ndi.binary_dilation(lens, iterations=int(round(1.0 / s)))
        La, _ = ndi.label(bright & ~vit)                                  # aqueous: bright pockets lying against the lens
        keep = np.unique(La[near & (La > 0)])
        aq = np.isin(La, keep[keep > 0]); inner |= lens | aq
        if collapsed:
            aq[:] = False                                                 # kept in the inner body, not named
        linfo = dict(found=True, trusted=True, contrast_mad=None, on_fluid_border_fraction=round(float(ml.mean()), 2),
                     note="MRI lens = smooth surface around the dark nucleus, fitted to where the lens meets the bright fluid; "
                          "the radius of the opposite side is used where it meets dark tissue")
    inner = fill3(S.edt_close(np.pad(inner, 12), 1.0 / s)[12:-12, 12:-12, 12:-12])
    # ---- the oil bath as drawn, and moved back to where it is
    # The edge of the bath against an empty band is taken at half the LOCAL bath level: the coil makes the bath
    # 15-25 % weaker on one side of the tube, and a fixed threshold put that edge up to 0.9 mm too far out there.
    sure = edt_open((g > 0.7 * M) & (g < 1.3 * M) & ~bright, 0.6 / s); sg = 3.0 / s
    wsum = ndi.gaussian_filter(sure.astype(np.float32), sg)
    level = np.where(wsum > 0.02, ndi.gaussian_filter(np.where(sure, g, 0).astype(np.float32), sg) / np.maximum(wsum, 1e-3), M)
    Lb, _ = ndi.label(edt_open((g >= 0.5 * level) & ~bright, 0.4 / s)); cnt = np.bincount(Lb.ravel()); cnt[0] = 0
    oil_obs = Lb == cnt.argmax()
    cz, cy, cx = (int(x) for x in np.argwhere(vit).mean(0)); half = int(3.0 / s)

    def gap(sign):                                                        # no-signal run next to the bath, up or down
        out = []
        for dx in range(-half, half + 1, max(1, half // 5)):
            col = g[cz, :, cx + dx][::-1] if sign < 0 else g[cz, :, cx + dx]
            col = col[(g.shape[1] - cy) if sign < 0 else cy:]
            w = np.flatnonzero(col > (M + W) / 2)
            if not len(w):
                continue
            q = col[w[-1]:w[-1] + int(7 / s)] < 0.3 * M; L, n = ndi.label(q)
            runs = [(L == i + 1).sum() for i in range(n) if (L == i + 1).sum() * s >= 0.3]
            out.append(runs[-1] * s if runs else 0.0)
        return float(np.median(out)) if out else 0.0
    gap_up, gap_down = gap(-1), gap(+1)
    nshift = int(round(oil_shift_mm / s)) * (1 if gap_up >= gap_down else -1)   # rows to move the oil back (down if the gap is up)

    def gap_z(sign):                                                      # the same band across the slices
        out = []
        for dx in range(-half, half + 1, max(1, half // 5)):
            col = g[:cz + 1, cy, cx + dx][::-1] if sign < 0 else g[cz:, cy, cx + dx]
            w = np.flatnonzero(col > (M + W) / 2)
            if not len(w):
                continue
            q = col[w[-1]:w[-1] + int(7 / s)] < 0.3 * M; L, n = ndi.label(q)
            runs = [(L == i + 1).sum() for i in range(n) if (L == i + 1).sum() * s >= 0.3]
            out.append(runs[-1] * s if runs else 0.0)
        return float(np.median(out)) if out else 0.0
    # Slice selection displaces the oil as well (it is excited in a neighbouring slice). The radio-frequency bandwidth
    # is not in the files, so this shift is MEASURED: the one-sided empty band across the slices, in whole slices.
    gz_lo, gz_hi = gap_z(-1), gap_z(+1); zshift = 0

    def dark_flank(sign):                                                 # dark wall along the columns, where no shift acts
        out = []
        for dy in range(-half, half + 1, max(1, half // 5)):
            col = g[cz, cy + dy, :cx + 1][::-1] if sign < 0 else g[cz, cy + dy, cx:]
            w = np.flatnonzero(col > (M + W) / 2)
            if not len(w):
                continue
            q = col[w[-1]:w[-1] + int(7 / s)] < 0.3 * M; L, n = ndi.label(q)
            runs = [(L == i + 1).sum() for i in range(n) if (L == i + 1).sum() * s >= 0.2]
            out.append(runs[-1] * s if runs else 0.0)
        return float(np.median(out)) if out else 0.0
    wall_dark = min(dark_flank(-1), dark_flank(+1))
    # the dark run across the slices is the real dark wall plus the empty band: take the wall (as seen along the
    # columns, where nothing is displaced) off before counting slices, or the shift comes out a slice too large
    if abs(gz_lo - gz_hi) >= 0.6:
        zshift = int(round(round(max(max(gz_lo, gz_hi) - wall_dark, 0) / slice_mm) * slice_mm / s)) * (1 if gz_lo > gz_hi else -1)
    if slice_shift_mm is not None:                                       # one value for the scan session (config)
        zshift = int(round(slice_shift_mm / s)) * (1 if gz_lo >= gz_hi else -1)
    unshift = lambda m: ndi.shift(m.astype(np.uint8), (zshift, nshift, 0), order=0, mode="constant", cval=0) > 0
    oil = unshift(oil_obs) if (nshift or zshift) else oil_obs
    # ---- globe = inner body + wall: inner radius per direction, wall thickness to where the corrected bath begins
    isurf = S.mask_to_star(inner, s, _G); cen = isurf["center_mm"]; Rin = isurf["R"].ravel()
    nth, nph = isurf["R"].shape; d = S._rays(nth, nph)
    tt = np.arange(0, 4.0, 0.1)
    P = (cen[None, None] + (Rin[:, None, None] + tt[None, :, None]) * d[:, None]) / s
    inoil = ndi.map_coordinates(oil.astype(np.uint8), P.reshape(-1, 3).T, order=0, mode="constant", cval=0).reshape(len(d), len(tt)) > 0
    th = np.full(len(d), np.nan)
    for i in range(len(d)):
        for j in np.flatnonzero(inoil[i]):
            if j + 6 <= len(tt) and inoil[i, j:j + 6].all():              # 0.6 mm of bath behind the wall
                th[i] = tt[j]; break
    ok = np.isfinite(th) & (th >= WALL_MM[0]) & (th <= WALL_MM[1]); th = np.where(ok, th, np.nan)
    nb = cKDTree(d).query_ball_point(d, 2 * np.sin(np.deg2rad(15) / 2)); gmed = float(np.nanmedian(th))
    loc = np.array([np.nanmedian(th[q]) if np.isfinite(th[q]).sum() >= 5 else gmed for q in nb])
    surf = dict(center_mm=cen, R=(Rin + np.where(ok, th, loc)).reshape(nth, nph), measured=ok.reshape(nth, nph).astype(np.int8))
    surf = S.smooth_surface(surf, _G)
    globe = S.star_mask(g.shape, s, surf)
    if lens is not None:
        lens &= globe; nucleus &= lens
        linfo.update(vol_mm3=round(float(lens.sum() * s ** 3), 1), nucleus_vol_mm3=round(float(nucleus.sum() * s ** 3), 1),
                     centroid_mm=(np.argwhere(lens).mean(0) * s).round(3).tolist())
    # ---- labels
    lab = np.zeros(g.shape, np.uint8)
    void = edt_open(g < VOID * M, 0.4 / s)
    Lv, _ = ndi.label(void); holder = np.zeros(g.shape, bool); edge = np.zeros(g.shape, bool)
    edge[:, 0, :] = edge[:, -1, :] = edge[:, :, 0] = edge[:, :, -1] = True
    for i, sl in enumerate(ndi.find_objects(Lv)):
        m = Lv[sl] == i + 1
        if m.sum() * s ** 3 > 20 and not edge[sl][m].any():               # inside the tube, not the space around it
            holder[sl][m] = True
    if nshift or zshift:                                                  # plastic is seen only as missing oil: move it back too
        holder = edt_open(unshift(holder) & ~oil, 0.4 / s)
    holder &= ~globe
    lab[oil & ~globe] = 1
    lab[holder] = 7
    # tissue outside the globe: neither bath nor holder (drawn in place: it holds water)
    # (bath for this purpose = the bath as drawn, at more than 0.75 of its local level, after a 1 mm opening, which
    #  keeps the grey inside of the nerve out of it; tissue holds water and is drawn in place)
    Lc, _ = ndi.label(edt_open((g >= 0.75 * level) & ~bright & ~globe, 1.0 / s)); cc = np.bincount(Lc.ravel()); cc[0] = 0
    core = Lc == cc.argmax()
    tis = ~core & ~holder & ~globe & (ndi.distance_transform_edt(~globe) * s < 14.0) & (g > VOID * M)
    tis &= ~ndi.binary_dilation(void & ~holder, iterations=2)
    tis = edt_open(tis, 0.5 / s)
    Lt, _ = ndi.label(tis); gc = np.argwhere(globe).mean(0)
    ax = (np.argwhere(lens).mean(0) - gc) if lens is not None else None
    stump_vol = 0.0; touch = ndi.binary_dilation(globe, iterations=int(round(1.5 / s))); cands = []
    for i, sl in enumerate(ndi.find_objects(Lt)):
        m = Lt[sl] == i + 1; vol = m.sum() * s ** 3
        if vol < 2 or not (touch[sl] & m).any():
            continue
        lab[sl][m] = 17
        p = np.argwhere(m) + np.array([q.start for q in sl]) - gc
        front = ax is not None and ((p @ ax) > 0).mean() >= 0.5
        far = np.linalg.norm(p, axis=1).max() * s
        if not front and vol > 30 and far > surf["R"].mean() + 4.0:        # reaches more than 4 mm away from the globe
            cands.append((vol, i))
    if cands:
        vol, i = max(cands); sl = ndi.find_objects(Lt)[i]
        lab[sl][Lt[sl] == i + 1] = 12; stump_vol = vol
    lab[globe] = 2                                                         # sclera with choroid and retina
    wall = globe & ~inner
    if aq.any():                                                           # cornea = wall whose nearest inner-body voxel is aqueous
        idx = ndi.distance_transform_edt(~inner, return_distances=False, return_indices=True)
        aqd = ndi.binary_dilation(aq, iterations=int(round(0.5 / s)))
        lab[wall & aqd[tuple(idx)]] = 11
    # thin layer of intermediate signal on the inside of the wall = retina with choroid; read only where the bath is
    # not drawn over the wall (upper side and flanks)
    din = ndi.distance_transform_edt(~inner) * s
    jj = np.arange(g.shape[1])[None, :, None] * s - cen[1]
    clear = (jj * np.sign(nshift if nshift else 1)) < 0.5 * surf["R"].mean()
    if not collapsed:
        lab[wall & (din <= 0.45) & (g >= 0.75 * M) & clear & (lab == 2)] = 18
    # ciliary body: the dark tongue of the wall that reaches in beside the lens equator, between aqueous and vitreous
    # (it is outside the "inner body", so it was labelled wall; its signal is intermediate, like the retina layer):
    # wall within 5 mm of the lens, within 1.6 mm of the fluid, and level with the lens along the eye axis
    if lens is not None and not collapsed:
        d_lens = ndi.distance_transform_edt(~lens) * s
        li = np.argwhere(lens); lcen = li.mean(0); axl = lcen - np.argwhere(globe).mean(0); axl /= np.linalg.norm(axl)
        zl = (li - lcen) @ axl * s; z0, z1 = float(zl.min()), float(zl.max())
        wi = np.argwhere(wall); zw = (wi - lcen) @ axl * s
        level_with_lens = np.zeros(g.shape, bool); level_with_lens[tuple(wi[(zw >= z0 - 0.5) & (zw <= z1)].T)] = True
        cil = wall & level_with_lens & (d_lens <= 5.0) & (din <= 1.6) & (g >= 0.3 * M) & np.isin(lab, [2, 18])   # sclera is darker
        lab[edt_open(cil, 0.2 / s)] = 14
    fluid = globe & bright & ~(lens if lens is not None else False)
    lab[fluid] = 3; lab[fluid & aq] = 13
    solid_in = inner & globe & ~bright & ~(lens if lens is not None else False)
    if lens is not None and not collapsed:
        dl = ndi.distance_transform_edt(~ndi.binary_dilation(lens | aq, iterations=1)) * s
        lab[solid_in & (dl <= 2.5)] = 14                                   # iris and ciliary body
        lab[solid_in & (dl > 2.5)] = 16                                    # intraocular membranes
        lab[lens] = 10; lab[nucleus] = 15
    else:
        lab[solid_in] = 16
        if lens is not None:
            lab[lens] = 10; lab[nucleus] = 15
    vol = lambda k: round(float((lab == k).sum() * s ** 3), 1)
    info = dict(medium_level=M, vitreous_mL=round(vol(3) / 1000, 3), aqueous_mm3=vol(13), cornea_mm3=vol(11),
                iris_ciliary_mm3=vol(14), membranes_mm3=vol(16), nerve_stump_mm3=round(stump_vol, 1), holder_mm3=vol(7),
                wall_thickness_median_mm=round(gmed, 2), wall_measured_fraction=round(float(ok.mean()), 2),
                retina_choroid_mm3=vol(18), oil_shift_applied_mm=round(abs(nshift) * s, 3),
                oil_shift_direction="towards lower rows" if nshift >= 0 else "towards higher rows",
                empty_band_measured_mm=dict(upper_side=round(gap_up, 2), lower_side=round(gap_down, 2)),
                oil_shift_across_slices_applied_mm=round(abs(zshift) * s, 3),
                empty_band_across_slices_mm=dict(lower_slices=round(gz_lo, 2), higher_slices=round(gz_hi, 2)),
                dark_wall_on_unshifted_flank_mm=round(wall_dark, 2),
                globe_surface=dict(method="inner body + wall thickness", center_mm=cen.round(3).tolist()))
    return lab, globe, lens, linfo, info
