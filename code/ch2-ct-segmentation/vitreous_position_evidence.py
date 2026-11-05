"""Evidence for the vitreous screen: one micro-CT section per eye through the centre of the globe outline and the
vitreous sensor, with the outline, the centre-to-wall line and the sensor on it.

The section plane contains the line from the centre of the outline to the vitreous pick (drawn horizontal, wall to
the right) and, as far as possible, the axis of the eye frame. Read from the files run.py wrote.

    python vitreous_position_evidence.py   -> outputs/_vitreous_position_evidence/<ID>.npz  (data of the panel)
                                              outputs/_vitreous_position_evidence/ALL.png   (all eyes, for checking)

Each npz holds: image (uint8, grey window already applied), t_mm (axis of the image, both directions), inside
(R(direction) - r on the same grid; its zero contour is the outline), r_mm (centre to sensor), R_mm (centre to wall
in that direction), fraction, and the eye id / cohort. The thesis figure is drawn from these files
(UIC-BME-Thesis/code/figures/appendix_vitreous_position.py).
"""
import json, os, sys
import numpy as np, nrrd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run
from ctseg import segment
from ctseg.export import _plane

HALF = 16.0   # mm, half field of view


def one(od, sid, out_dir):
    r = json.load(open(f"{od}/{sid}/{sid}_results.json")); s = r["voxel_work_mm"]
    pk = next(p for p in r["thermistors"]["picks"] if p["sensor"] == "vitreous")
    d = np.load(f"{od}/{sid}/{sid}_globe_surface.npz"); surf = dict(center_mm=d["center_mm"], R=d["R_mm"], measured=d["measured"])
    c = surf["center_mm"]; w = np.array(pk["work_mm_kji"])
    a = (w - c) / np.linalg.norm(w - c)
    ez = np.array(r["anatomy"]["frame"]["axes"])[2]
    b = ez - (ez @ a) * a
    if np.linalg.norm(b) < 0.3:                      # sensor almost on the axis: use e_x for the second direction
        ex = np.array(r["anatomy"]["frame"]["axes"])[0]; b = ex - (ex @ a) * a
    b /= np.linalg.norm(b)
    v, _ = nrrd.read(f"{od}/{sid}/{sid}.nrrd"); v = v.transpose(2, 1, 0)
    lab, _ = nrrd.read(f"{od}/{sid}/{sid}.seg.nrrd"); lab = lab.transpose(2, 1, 0)
    g = v[lab == 2]; lo, hi = np.percentile(g, [1, 99.5])
    if r["bath"] == "liquid":
        lo = lo - 1.5 * (hi - lo)
    img, t = _plane(v, c, a, b, s, half=HALF)
    inside = segment.star_plane(surf, c, a, b, t)
    Rv, rr = segment.star_lookup(surf, w[None])
    np.savez_compressed(f"{out_dir}/{sid}.npz", image=np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8),
                        t_mm=t.astype(np.float32), inside=inside.astype(np.float16), r_mm=float(rr[0]), R_mm=float(Rv[0]),
                        fraction=float(rr[0] / Rv[0]), eye=sid, group=r["group"], bath=r["bath"],
                        outline=r["segmentation"]["globe_surface"]["method"])
    return sid, float(rr[0]), float(Rv[0])


def draw(A, z, limit=None):
    """One panel from an npz record (used here and by the thesis script)."""
    t = z["t_mm"]; r, R = float(z["r_mm"]), float(z["R_mm"])
    A.imshow(z["image"].T, cmap="gray", vmin=0, vmax=255, origin="lower", extent=[t[0], t[-1], t[0], t[-1]])
    A.contour(t, t, z["inside"].astype(np.float32).T, [0], colors="red", linewidths=0.9)
    A.plot([0, R], [0, 0], "-", color="yellow", lw=0.9)
    A.plot(0, 0, "+", color="yellow", ms=7, mew=1.2); A.plot(R, 0, "|", color="yellow", ms=9, mew=1.4)
    flagged = limit is not None and r / R > limit
    A.plot(r, 0, "D" if flagged else "o", mfc="none", mec="cyan", ms=8, mew=1.5)
    A.set_xticks([]); A.set_yticks([])
    return flagged


if __name__ == "__main__":
    cfg = run.load_cfg(); od = cfg["out_dir"]; out_dir = os.path.join(od, "_vitreous_position_evidence"); os.makedirs(out_dir, exist_ok=True)
    ids = sys.argv[1:] or [sc["id"] for sc in cfg["scans"]]
    for sid in ids:
        print("%s r %.2f mm, R %.2f mm" % one(od, sid, out_dir), flush=True)
    allid = [sc["id"] for sc in cfg["scans"] if os.path.exists(f"{out_dir}/{sc['id']}.npz")]
    fig, ax = plt.subplots(4, 5, figsize=(15, 12.4))
    for A in ax.flat:
        A.axis("off")
    for A, sid in zip(ax.flat, allid):
        z = np.load(f"{out_dir}/{sid}.npz"); A.axis("on"); draw(A, z, 0.75)
        A.set_title(f"{sid}: {float(z['fraction']):.2f}", fontsize=10)
    fig.suptitle("Vitreous sensor (cyan) on the line from the centre of the globe outline (+) to the wall (|); red = outline; "
                 "diamond = more than 0.75 of the way", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(f"{out_dir}/ALL.png", dpi=70); plt.close()
