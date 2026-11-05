"""Evidence that the globe outline follows the eye: the outline drawn on the scan in 15 parallel sections per eye.

For every eye in config.json, from the files run.py wrote (<ID>.nrrd, <ID>.seg.nrrd, <ID>_results.json and, for the
smooth-surface outline, <ID>_globe_surface.npz): three families of sections in the eye frame (perpendicular to the
axis, to e_y and to e_x), each at -8, -4, 0, +4, +8 mm from the globe centroid. Red = outline on a tissue edge,
cyan = outline bridged (no usable edge there). Preserved eyes have the opening-based outline: red only.

    python outline_evidence.py            # all eyes -> outputs/_outline_evidence/<ID>_outline.png
    python outline_evidence.py H2 F17     # selected eyes
"""
import json, os, sys
import numpy as np, nrrd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run
from ctseg import segment
from ctseg.export import _plane

OFFSETS = (-8.0, -4.0, 0.0, 4.0, 8.0)   # mm from the globe centroid


def one(od, sid, out_dir):
    r = json.load(open(f"{od}/{sid}/{sid}_results.json")); s = r["voxel_work_mm"]
    v, _ = nrrd.read(f"{od}/{sid}/{sid}.nrrd"); v = v.transpose(2, 1, 0)
    lab, _ = nrrd.read(f"{od}/{sid}/{sid}.seg.nrrd"); lab = lab.transpose(2, 1, 0)
    sf = f"{od}/{sid}/{sid}_globe_surface.npz"; surf = None
    if os.path.exists(sf):
        d = np.load(sf); surf = dict(center_mm=d["center_mm"], R=d["R_mm"], measured=d["measured"])
        gl = segment.star_mask(v.shape, s, surf)
    else:
        L, _ = ndi.label(np.isin(lab, [2, 3, 4])); gl = segment.fill3(L == np.bincount(L.ravel())[1:].argmax() + 1)
    o = np.array(r["anatomy"]["frame"]["origin_mm"]); ex, ey, ez = np.array(r["anatomy"]["frame"]["axes"])
    fam = [("section perpendicular to the axis", ez, ex, ey, "towards anterior"),
           ("section containing the axis (anterior right), perpendicular to e_y", ey, ez, ex, "along e_y"),
           ("section containing the axis (anterior right), perpendicular to e_x", ex, ez, ey, "along e_x")]
    g = v[lab == 2]; lo, hi = np.percentile(g, [1, 99.5])
    if r["bath"] == "liquid":           # tissue and bath are close in grey value: window on both
        lo = lo - 1.5 * (hi - lo)
    fig, ax = plt.subplots(3, 5, figsize=(20, 12.6))
    for i, (name, nrm, a, b, way) in enumerate(fam):
        for j, off in enumerate(OFFSETS):
            oo = o + off * nrm
            img, t = _plane(v, oo, a, b, s); gm, _ = _plane(gl.astype(np.float32), oo, a, b, s, order=0)
            A = ax[i, j]
            A.imshow(img.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=[t[0], t[-1], t[0], t[-1]])
            if surf is not None:                     # the surface itself, not its voxel mask (no staircase)
                gm = segment.star_plane(surf, oo, a, b, t) + 0.5
            if (gm > 0.5).any():
                cs = A.contour(t, t, gm.T, [.5], colors="r", linewidths=.9)
                if surf is not None:
                    xy = np.concatenate([q.vertices for q in cs.get_paths() if len(q.vertices)] or [np.zeros((0, 2))])
                    if len(xy):
                        fl = segment.star_lookup(surf, oo[None] + xy[:, :1] * a[None] + xy[:, 1:] * b[None], order=0)[0]
                        A.plot(xy[fl < 0.5, 0], xy[fl < 0.5, 1], ".", color="cyan", ms=2.2)
                        A.plot(xy[fl > 1.5, 0], xy[fl > 1.5, 1], ".", color="magenta", ms=2.2)
            A.set_title(f"{name}\n{off:+.0f} mm {way}" if j == 0 else f"{off:+.0f} mm {way}", fontsize=8, loc="left")
            A.set_xlabel("mm", fontsize=7); A.tick_params(labelsize=7)
    L2 = lambda **k: plt.Line2D([], [], **k)
    star = surf is not None and (surf["measured"] == 0).any()
    h = [L2(color="r", lw=1.5, label="globe outline" + (", on a tissue edge" if star else " (preserved eye: opening-based outline)"))]
    if star:
        h.append(L2(color="cyan", lw=0, marker=".", ms=8, label="globe outline, bridged (no usable edge: metal streak, bubble, holder or nerve contact)"))
    if surf is not None and (surf["measured"] == 2).any():
        h.append(L2(color="magenta", lw=0, marker=".", ms=8, label="globe outline, set by LM's drawn corrections"))
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=10, frameon=False)
    gs = r["segmentation"].get("globe_surface", {})
    mf = f" | {100 * gs['measured_fraction']:.0f} % of the outline on a tissue edge" if "measured_fraction" in gs and star else ""
    mf += f" | {100 * gs['marked_fraction']:.0f} % set by LM's corrections" if gs.get("marked_fraction") else ""
    fig.suptitle(f"{sid} ({r['group']}, {r['bath']} scan): globe outline on the scan | {r['globe_vol_mL']} mL | axial "
                 f"{r['anatomy']['axial_length_mm']} mm (axis from {r['anatomy']['axis']['source']}){mf}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); p = f"{out_dir}/{sid}_outline.png"; plt.savefig(p, dpi=75); plt.close()
    return p


if __name__ == "__main__":
    cfg = run.load_cfg(); od = cfg["out_dir"]; out_dir = os.path.join(od, "_outline_evidence"); os.makedirs(out_dir, exist_ok=True)
    for sid in (sys.argv[1:] or [sc["id"] for sc in cfg["scans"]]):
        print(one(od, sid, out_dir), flush=True)
