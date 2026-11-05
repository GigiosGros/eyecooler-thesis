"""Read the corrections drawn by hand on the outline-evidence pictures and store them as 3-D points.

On 2026-10-06 I drew, in magenta, where the true globe wall runs on outputs/_outline_evidence/<ID>_outline.png.
The annotated pictures are kept in outputs/_outline_evidence_LM_marks_20261006/. Every picture is 15 known
sections of the eye frame that was current when it was drawn, so a magenta pixel is a point in the scan.
This script writes outputs/<ID>/<ID>_outline_marks_LM.csv (working-grid mm and Slicer LPS mm); run.py then makes
the globe outline pass through those points (segment.apply_marks).

The frame the pictures were drawn in is written into the csv header the first time and read back from there
afterwards, so the script can be re-run after run.py has changed <ID>_results.json. A csv must exist (or
results.json must still hold the drawing frame) for every annotated picture.
The untouched part of the outline in the annotated sections is stored too, as 'keep' anchors (kind 0): without
them a correction spreads sideways into wall already accepted.

    python outline_marks.py            # all annotated pictures of round 1
    python outline_marks.py H7 P12
    python outline_marks.py --round=2  # second round (green lines of 2026-10-07) -> <ID>_outline_marks_LM_round2.csv

Check printed per eye: the red outline pixels of the same picture, sent through the same pixel -> mm mapping,
must fall on the stored outline (median |error| well under one pixel = 0.13 mm).
"""
import csv, json, os, re, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run
from ctseg import segment
from outline_evidence import OFFSETS

MARKS_DIR = "_outline_evidence_LM_marks_20261006"
# Later rounds: drawn again (any colour) on the pictures of the CORRECTED outline. The untouched pictures and the
# frame each was drawn in were saved first, so the drawing is whatever differs from the untouched picture.
# Round 3 was drawn before untouched copies were made: its marks are the orange pixels (255, 127, 39), a colour the
# pictures do not otherwise contain; the frames were saved into the annotated folder right after.
ROUNDS = {2: dict(marks_dir="_outline_evidence_LM_marks_round2_20261007", pristine_dir="_outline_evidence_round2_pristine"),
          3: dict(marks_dir="_outline_evidence_LM_marks_round3_20261007", colour="orange")}
EDGE = 1.0      # px: the one-pixel axes frame at each side of a section (gives the smallest error in the mapping check)
KEEP_CLEAR_DEG = 3   # an outline pixel within this angle of a magenta pixel (seen from the section's outline centre) is replaced
KEEP_STEP = 3        # every 3rd untouched outline pixel becomes a keep anchor


def layout(im):
    """Pixel box (x0, x1, y_top, y_bottom) of the 3 x 5 sections, found in the picture itself: the fifteen large
    non-white squares (the axes frame is one pixel wide and belongs to the square)."""
    from scipy import ndimage as ndi
    L, n = ndi.label(ndi.binary_opening(im.sum(2) < 740, structure=np.ones((40, 40))))
    sl = [q for q in ndi.find_objects(L) if q[0].stop - q[0].start > 150]
    assert len(sl) == 15, f"expected 15 sections, found {len(sl)}"
    sl.sort(key=lambda q: (round(q[0].start / 100), q[1].start))
    return {(k // 5, k % 5): (q[1].start + EDGE, q[1].stop - EDGE, q[0].start + EDGE, q[0].stop - EDGE) for k, q in enumerate(sl)}


def pixels_to_points(mask, box, o, F, s, thin=True, step=1):
    """Pixels of a boolean picture mask -> points (working-grid mm), with the panel (row, column) of each."""
    ex, ey, ez = F
    fam = [(ez, ex, ey), (ey, ez, ex), (ex, ez, ey)]
    t1 = np.arange(-16.0, 16.0, s)[-1]                      # imshow extent of the sections is [-16, t1]
    out = []
    for (i, j), (x0, x1, y0, y1) in box.items():
        sub = mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        if sub.sum() < 5:
            continue
        q = np.argwhere(skeletonize(sub) if thin else sub)[::step]
        py = q[:, 0] + int(y0) + 0.5; px = q[:, 1] + int(x0) + 0.5
        u = -16 + (px - x0) / (x1 - x0) * (t1 + 16); v = -16 + (y1 - py) / (y1 - y0) * (t1 + 16)
        nrm, a, b = fam[i]
        P = o[None] + OFFSETS[j] * nrm[None] + u[:, None] * a[None] + v[:, None] * b[None]
        out.append(np.c_[P, np.full(len(P), i), np.full(len(P), j)])
    return np.concatenate(out) if out else np.zeros((0, 5))


def untouched(outline, mag, box):
    """Outline pixels left as they were. Within a section, an outline pixel counts as replaced when a magenta
    pixel lies in the same direction (within KEEP_CLEAR_DEG) seen from the middle of that section's outline;
    all other outline pixels of the annotated picture are approved as drawn and become 'keep' anchors."""
    keep = np.zeros_like(outline)
    for (x0, x1, y0, y1) in box.values():
        sl = (slice(int(y0), int(np.ceil(y1))), slice(int(x0), int(np.ceil(x1))))
        q = np.argwhere(outline[sl])
        if len(q) < 20:
            continue
        c = q.mean(0); ang = lambda z: np.degrees(np.arctan2(z[:, 0] - c[0], z[:, 1] - c[1])) % 360
        taken = np.zeros(360, bool); m = np.argwhere(mag[sl])
        if len(m):
            for k in range(-KEEP_CLEAR_DEG, KEEP_CLEAR_DEG + 1):
                taken[(ang(m).astype(int) + k) % 360] = True
        ok = q[~taken[ang(q).astype(int) % 360]]
        keep[sl][ok[:, 0], ok[:, 1]] = True
    return keep


def one(od, sid):
    fn = f"{od}/{MARKS_DIR}/{sid}_outline.png"
    if not os.path.exists(fn):
        return f"{sid}: no annotated picture"
    im = np.asarray(Image.open(fn).convert("RGB")).astype(int); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    box = layout(im)
    mag = (r > 170) & (b > 170) & (g < 140) & (r - g > 80) & (b - g > 80)
    red = (r > 200) & (g < 60) & (b < 60)
    outline = red | ((r < 80) & (g > 200) & (b > 200))       # red or cyan part of the outline as it was drawn
    res = json.load(open(f"{od}/{sid}/{sid}_results.json")); s, s0, f = res["voxel_work_mm"], res["voxel_full_mm"], res["downsample"]
    o = np.array(res["anatomy"]["frame"]["origin_mm"]); F = np.array(res["anatomy"]["frame"]["axes"])
    out = f"{od}/{sid}/{sid}_outline_marks_LM.csv"; frame_from = "results.json"
    if os.path.exists(out):                                  # the frame the pictures were drawn in, as first recorded
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e-?\d+)?|-?\d+", open(out).readlines()[1].split("origin_mm")[1])]
        o, F = np.array(nums[:3]), np.array(nums[3:12]).reshape(3, 3); frame_from = "the existing csv"
    P = pixels_to_points(mag, box, o, F, s)
    K = pixels_to_points(untouched(outline, mag, box), box, o, F, s, thin=False, step=KEEP_STEP) if len(P) else np.zeros((0, 5))
    msg = f"{sid}: {len(P)} mark points on {len(set(map(tuple, P[:, 3:].astype(int))))} sections, {len(K)} keep anchors (frame from {frame_from})"
    sf = f"{od}/{sid}/{sid}_globe_surface.npz"
    if os.path.exists(sf) and frame_from == "results.json":  # mapping check on the red outline (only while the surface is the drawn one)
        d = np.load(sf); surf = dict(center_mm=d["center_mm"], R=d["R_mm"], measured=d["measured"])
        Q = pixels_to_points(red, box, o, F, s, thin=False)
        if len(Q):
            Rv, rr = segment.star_lookup(surf, Q[:, :3]); msg += f" | mapping check on the red outline: median |error| {np.median(np.abs(rr - Rv)):.2f} mm"
        if len(P):
            Rv, rr = segment.star_lookup(surf, P[:, :3]); e = rr - Rv
            msg += f" | mark minus outline: median {np.median(e):+.2f}, 10th-90th pct {np.percentile(e, 10):+.2f} to {np.percentile(e, 90):+.2f} mm"
    if len(P):
        A = np.r_[np.c_[P, np.ones(len(P))], np.c_[K, np.zeros(len(K))]]
        with open(out, "w", newline="") as fh:
            fh.write(f"# Corrections drawn by LM on {MARKS_DIR}/{sid}_outline.png, read by outline_marks.py. kind 1 = magenta mark (the wall is here); "
                     f"kind 0 = outline LM left untouched in an annotated section (keep it).\n")
            fh.write(f"# Eye frame of the picture: origin_mm {o.round(3).tolist()} axes {F.round(4).tolist()}; working voxel {s} mm.\n")
            w = csv.writer(fh); w.writerow(["work_k_mm", "work_j_mm", "work_i_mm", "L_mm", "P_mm", "S_mm", "section_row", "section_col", "kind"])
            lps = (A[:, :3] + (f - 1) / 2 * s0)[:, ::-1]
            w.writerows(np.c_[A[:, :3].round(3), lps.round(3), A[:, 3:].astype(int)].tolist())
    return msg


def later_round(od, sid, k):
    """Round k >= 2: marks = pixels that differ from the untouched picture; keep anchors = the outline (red, cyan
    or magenta) left untouched in a section that was drawn on or looked at. Written to <ID>_outline_marks_LM_round<k>.csv.
    run.py applies the rounds in order, each on the outline the previous one produced."""
    R = ROUNDS[k]; fn = f"{od}/{R['marks_dir']}/{sid}_outline.png"; ref = R.get("pristine_dir", R["marks_dir"])
    pn = f"{od}/{ref}/{sid}_outline.png"
    if not (os.path.exists(fn) and os.path.exists(pn)):
        return f"{sid} round {k}: no annotated picture"
    im = np.asarray(Image.open(fn).convert("RGB")).astype(int); pr = np.asarray(Image.open(pn).convert("RGB")).astype(int)
    if "pristine_dir" in R:
        drawn = np.abs(im - pr).sum(2) > 150                   # a new stroke, not a re-compression shade
    else:                                                      # orange (255, 127, 39)
        drawn = (im[..., 0] > 230) & (im[..., 1] > 100) & (im[..., 1] < 155) & (im[..., 2] < 80)
    if drawn.sum() < 5:
        return f"{sid} round {k}: nothing drawn"
    fr = json.load(open(f"{od}/{ref}/frames.json"))["frames"][sid]
    o, F, s, s0, f = np.array(fr["origin_mm"]), np.array(fr["axes"]), fr["voxel_work_mm"], fr["voxel_full_mm"], fr["downsample"]
    box = layout(pr)
    r, g, b = pr[..., 0], pr[..., 1], pr[..., 2]
    outline = ((r > 200) & (g < 60) & (b < 60)) | ((r < 80) & (g > 200) & (b > 200)) | ((r > 200) & (g < 60) & (b > 200))
    P = pixels_to_points(drawn, box, o, F, s)
    K = pixels_to_points(untouched(outline & ~drawn, drawn, box), box, o, F, s, thin=False, step=KEEP_STEP)
    A = np.r_[np.c_[P, np.ones(len(P))], np.c_[K, np.zeros(len(K))]]
    with open(f"{od}/{sid}/{sid}_outline_marks_LM_round{k}.csv", "w", newline="") as fh:
        fh.write(f"# Round {k} corrections drawn by LM on {R['marks_dir']}/{sid}_outline.png ({'pixels that differ from ' + R['pristine_dir'] if 'pristine_dir' in R else R['colour'] + ' pixels'}), "
                 f"read by outline_marks.py. kind 1 = mark (the wall is here); kind 0 = outline LM left untouched (keep it). "
                 f"Applies to the outline as it stood after round {k - 1}.\n")
        fh.write(f"# Eye frame of the picture: origin_mm {o.round(3).tolist()} axes {F.round(4).tolist()}; working voxel {s} mm.\n")
        w = csv.writer(fh); w.writerow(["work_k_mm", "work_j_mm", "work_i_mm", "L_mm", "P_mm", "S_mm", "section_row", "section_col", "kind"])
        lps = (A[:, :3] + (f - 1) / 2 * s0)[:, ::-1]
        w.writerows(np.c_[A[:, :3].round(3), lps.round(3), A[:, 3:].astype(int)].tolist())
    return f"{sid} round {k}: {len(P)} mark points on {len(set(map(tuple, P[:, 3:].astype(int))))} sections, {len(K)} keep anchors"


if __name__ == "__main__":
    cfg = run.load_cfg(); od = cfg["out_dir"]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rnd = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--round=")), 1)
    for sid in (args or [sc["id"] for sc in cfg["scans"]]):
        print(one(od, sid) if rnd == 1 else later_round(od, sid, rnd), flush=True)
