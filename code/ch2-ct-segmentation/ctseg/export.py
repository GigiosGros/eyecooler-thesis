"""3D Slicer files (volume, segmentation, bead markups, axis line) and QC figures."""
import json
import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from .segment import NAMES, COLORS


def _hdr(s, origin_lps):
    return {"space": "left-posterior-superior", "space directions": np.eye(3) * s,
            "space origin": np.asarray(origin_lps, float), "kinds": ["domain"] * 3}


def slicer(outdir, sid, v, lab, s, f, s0, beads, anat):
    """Volume + labelmap on the downsampled grid, placed in the FULL-resolution scan frame
    (same LPS frame as the original Slicer nrrds: LPS = (col, row, slice) * s0)."""
    origin = np.full(3, (f - 1) / 2 * s0)
    nrrd.write(f"{outdir}/{sid}.nrrd", v.transpose(2, 1, 0).astype(np.float32), _hdr(s, origin))
    write_seg(outdir, sid, lab, s, f, s0)
    _markups(outdir, sid, f, s0, beads, anat)


def write_seg(outdir, sid, lab, s, f, s0):
    """The label map alone (used when only labels change)."""
    origin = np.full(3, (f - 1) / 2 * s0)
    h = _hdr(s, origin); j = 0
    for k in sorted(NAMES):
        if k > 0 and (lab == k).any():
            h.update({f"Segment{j}_ID": NAMES[k], f"Segment{j}_Name": NAMES[k], f"Segment{j}_LabelValue": str(k),
                      f"Segment{j}_Layer": "0", f"Segment{j}_Color": " ".join(map(str, COLORS[k]))}); j += 1
    nrrd.write(f"{outdir}/{sid}.seg.nrrd", lab.transpose(2, 1, 0), h)


def plate_to_holder(lab, s, ring, half_mm=2.0, radius_mm=15.0):
    """Voxels labelled tube wall (6) that lie in the slab of the holder ring plate, within radius_mm of its axis,
    become holder plastic (7). Returns how many."""
    n = np.array(ring["normal"]); c = np.array(ring["plate_centre_mm"])
    idx = np.argwhere(lab == 6); q = idx * s - c
    along = q @ n; rad = np.linalg.norm(q - np.outer(along, n), axis=1)
    sel = idx[(np.abs(along) <= half_mm) & (rad <= radius_mm)]
    lab[tuple(sel.T)] = 7
    return len(sel)


def point_markup(path, label, lps):
    mk = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
          "markups": [{"type": "Fiducial", "coordinateSystem": "LPS", "coordinateUnits": "mm",
                       "controlPoints": [{"id": "1", "label": label, "position": [float(x) for x in lps], "positionStatus": "defined"}]}]}
    json.dump(mk, open(path, "w"), indent=1)


def _markups(outdir, sid, f, s0, beads, anat):
    def cp(i, label, pos):
        return {"id": str(i), "label": label, "position": [float(x) for x in pos], "positionStatus": "defined"}
    pts = [cp(i + 1, b.get("pick_label", f"{sid}-bead{i + 1}"), b["lps_mm"]) for i, b in enumerate(beads)]
    mk = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
          "markups": [{"type": "Fiducial", "coordinateSystem": "LPS", "coordinateUnits": "mm", "controlPoints": pts}]}
    json.dump(mk, open(f"{outdir}/{sid}_thermistor_beads.mrk.json", "w"), indent=1)
    toL = lambda p_kji: (np.asarray(p_kji) + (f - 1) / 2 * s0)[::-1]
    ln = {"@schema": mk["@schema"], "markups": [{"type": "Line", "coordinateSystem": "LPS", "coordinateUnits": "mm",
          "controlPoints": [cp(1, "anterior_pole", toL(anat["anterior_pole_mm"])),
                            cp(2, "posterior_pole", toL(anat["posterior_pole_mm"]))]}]}
    json.dump(ln, open(f"{outdir}/{sid}_axial_length.mrk.json", "w"), indent=1)


def _plane(vol, o, a, b, s, half=16.0, step=None, order=1):
    step = step or s
    t = np.arange(-half, half, step)
    U, V = np.meshgrid(t, t, indexing="ij")
    P = (o[None, None] + U[..., None] * a + V[..., None] * b) / s
    return ndi.map_coordinates(vol, P.reshape(-1, 3).T, order=order, mode="constant").reshape(U.shape), t


LEGEND = {1: "bath liquid", 2: "globe tissue", 3: "fluid inside the globe outline", 4: "air inside the globe outline",
          5: "thermistor metal", 6: "tube wall", 7: "holder plastic", 8: "other attached tissue",
          9: "other small solid", 10: "lens", 11: "cornea", 12: "optic-nerve stump", 13: "aqueous",
          14: "iris and ciliary body", 15: "lens nucleus", 16: "membranes inside the eye", 17: "other tissue outside the globe"}
LEGEND_MRI = {1: "oil bath (moved back to its true place)", 2: "sclera (and wall not resolved further)", 3: "vitreous",
              7: "holder plastic (moved back with the bath)", 10: "lens cortex", 18: "retina with choroid (where readable)"}


def qc(path, sid, v, lab, gl, s, f, s0, anat, beads, info, surf=None):
    """Three planes through the globe centroid: grey values with the globe outline (top) and the labels (bottom).
    Every picture carries a legend. With a smooth_star outline (surf), the bridged parts are drawn in cyan."""
    from .segment import star_lookup, star_plane
    o = np.array(anat["frame"]["origin_mm"]); F = np.array(anat["frame"]["axes"])
    ex, ey, ez = F
    planes = [("plane through the axis: horizontal = axis (anterior right), vertical = e_x", ez, ex),
              ("plane through the axis: horizontal = axis (anterior right), vertical = e_y", ez, ey),
              ("plane through the centroid, perpendicular to the axis", ex, ey)]
    cmap = ListedColormap([COLORS[k] for k in range(len(COLORS))])
    fig, ax = plt.subplots(2, 3, figsize=(15, 11))
    gvals = v[(lab == 2)]
    lo, hi = np.percentile(gvals, [1, 99.5]) if gvals.size else (v.min(), v.max())
    seen = set()
    for c, (title, a, b) in enumerate(planes):
        img, t = _plane(v, o, a, b, s)
        lb, _ = _plane(lab.astype(np.float32), o, a, b, s, order=0)
        gm, _ = _plane(gl.astype(np.float32), o, a, b, s, order=0)
        ext = [t[0], t[-1], t[0], t[-1]]
        ax[0, c].imshow(img.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=ext)
        if surf is not None:                         # the surface itself, not its voxel mask (no staircase)
            gm = star_plane(surf, o, a, b, t) + 0.5
        cs = ax[0, c].contour(t, t, gm.T, [.5], colors="r", linewidths=.9)
        if surf is not None:
            xy = np.concatenate([q.vertices for q in cs.get_paths() if len(q.vertices)] or [np.zeros((0, 2))])
            if len(xy):
                flag = star_lookup(surf, o[None] + xy[:, :1] * a[None] + xy[:, 1:] * b[None], order=0)[0]
                ax[0, c].plot(xy[flag < 0.5, 0], xy[flag < 0.5, 1], ".", color="cyan", ms=2.2)
                ax[0, c].plot(xy[flag > 1.5, 0], xy[flag > 1.5, 1], ".", color="magenta", ms=2.2)
        ax[1, c].imshow(lb.T, cmap=cmap, vmin=-.5, vmax=len(COLORS) - .5, origin="lower", extent=ext, interpolation="nearest")
        seen |= set(np.unique(lb).astype(int).tolist())
        if c < 2:
            pa, pp = (np.array(anat[k]) - o for k in ("anterior_pole_mm", "posterior_pole_mm"))
            ax[0, c].plot([pa @ a, pp @ a], [pa @ b, pp @ b], "y-", lw=1)
            ax[0, c].plot(pa @ a, pa @ b, "y>", ms=8)
        for bd in beads:
            p = np.array(bd["full_mm_kji"]) - (f - 1) / 2 * s0 - o
            ax[0, c].plot(p @ a, p @ b, "o", mfc="none", mec="lime", ms=7)
        ax[0, c].set_title(title, fontsize=9); ax[0, c].set_xlabel("mm"); ax[1, c].set_xlabel("mm")
        ax[0, c].set_ylabel("mm"); ax[1, c].set_ylabel("mm")
    L = lambda **k: plt.Line2D([], [], **k)
    star = surf is not None and (surf["measured"] == 0).any()
    top = [L(color="r", lw=1.5, label="globe outline" + (", tissue edge found" if star else ""))]
    if star:
        top.append(L(color="cyan", lw=0, marker=".", ms=7,
                     label="globe outline, bridged (no usable edge: metal streak, bubble, holder or nerve contact)"))
    if surf is not None and (surf["measured"] == 2).any():
        top.append(L(color="magenta", lw=0, marker=".", ms=7, label="globe outline, set by LM's drawn corrections"))
    top.append(L(color="y", lw=1.2, marker=">", ms=7, label="axis of the eye frame (arrow = anterior end)"))
    if beads:
        top.append(L(color="lime", lw=0, marker="o", mfc="none", ms=7, label="thermistor bead (projected onto the plane)"))
    names = {**LEGEND, **LEGEND_MRI} if info.get("modality") == "MRI" else LEGEND
    bot = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[k], ec="0.4", lw=.5, label=names[k]) for k in sorted(seen) if k in names]
    bot.append(plt.Rectangle((0, 0), 1, 1, fc=COLORS[0], ec="0.4", lw=.5, label="air / outside"))
    fig.legend(handles=top, loc="lower center", bbox_to_anchor=(0.5, 0.058), ncol=2, fontsize=8, frameon=False,
               title="top row", title_fontsize=8)
    fig.legend(handles=bot, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(bot), 6), fontsize=8, frameon=False,
               title="bottom row (labels)", title_fontsize=8)
    gs = info.get("segmentation", {}).get("globe_surface", {}) if isinstance(info.get("segmentation"), dict) else {}
    outline = ""
    if "measured_fraction" in gs:
        mk = gs.get("marked_fraction", 0.0)
        outline = (f" | outline {100 * gs['measured_fraction']:.0f} % edge, {100 * (1 - gs['measured_fraction'] - mk):.0f} % bridged"
                   + (f", {100 * mk:.0f} % LM marks" if mk else ""))
    fig.suptitle(f"{sid}: globe {info['globe_vol_mL']} mL | axial {anat['axial_length_mm']} mm (axis from {anat['axis']['source']}) | "
                 f"equatorial {anat['equatorial']['max_mm']}/{anat['equatorial']['min_mm']} mm | beads {len(beads)}{outline}",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0.125, 1, 0.95]); plt.savefig(path, dpi=70); plt.close()
