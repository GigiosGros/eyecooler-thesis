"""Eye-wall layers on the stored MRI slices (0.078 mm in-plane), with the oil bath removed.

For each pig eye, on the five stored slices around the globe centre:
  1. the oil bath as it is DRAWN in the slice (the true bath of the label volume, displaced by the chemical shift
     along the rows and across the slices) is subtracted at its local level, which leaves a tissue-only image; the
     wall hidden under the displaced bath on the far side becomes readable (approximately: it is a subtraction);
  2. along rays from the globe centre, every 2 degrees: the edge of the fluid, the thin layer of intermediate signal
     against it (retina with choroid), and the end of the wall (where the tissue-only signal stays below OUT_THR
     for 0.5 mm);
  3. per eye: thickness of the retina-with-choroid layer and of the whole wall, by sector, median over rays and slices.

    python mri_layers.py            # all pig eyes -> outputs/<ID>/<ID>_MRI_layers.png, outputs/mri_wall_layers.csv
    python mri_layers.py F17

Needs the outputs of `run.py --mri-only` (label volume, results). Sectors are angles from the lens direction:
front 0 +- 40 deg (cornea), flanks 60-120 deg either side, back 140-220 deg.
"""
import csv, json, os, sys
import numpy as np, nrrd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run
from ctseg import mri

OUT_THR = 0.10        # tissue-only signal (bath = 1) below which there is no tissue
LIGHT = 0.75          # signal above which the layer against the fluid counts as retina with choroid
FLUID = 1.40


def native_of(vol_iso, k, shape, ps, dz, order=0):
    """Slice k of the stored series from a volume on the isotropic grid."""
    r, c = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    P = np.stack([np.full(r.shape, k * dz / mri.ISO), r * ps / mri.ISO, c * ps / mri.ISO])
    return ndi.map_coordinates(vol_iso, P.reshape(3, -1), order=order, mode="constant").reshape(r.shape)


def one(od, scan, rows):
    sid = scan["id"]; r = json.load(open(f"{od}/{sid}/{sid}_MRI_results.json")); g = r["segmentation"]
    v, ps, dz, meta = mri.load_series(scan["mri"]); M = g["medium_level"]
    lab, _ = nrrd.read(f"{od}/{sid}/{sid}_MRI.seg.nrrd"); lab = lab.transpose(2, 1, 0)
    ny = int(round(g["oil_shift_applied_mm"] / mri.ISO)) * (1 if g["oil_shift_direction"] == "towards lower rows" else -1)
    b = g["empty_band_across_slices_mm"]; nz = int(round(g["oil_shift_across_slices_applied_mm"] / mri.ISO)) * (1 if b["lower_slices"] > b["higher_slices"] else -1)
    drawn = ndi.shift((lab == 1).astype(np.uint8), (-nz, -ny, 0), order=0, mode="constant") > 0     # the bath where the scanner draws it
    gc = np.array(r["anatomy"]["frame"]["origin_mm"]); lc = np.array(r["anatomy"]["lens"]["centroid_mm"])
    k0 = int(round(gc[0] / dz)); vc = gc[1:] / ps; d0 = (lc[1:] - gc[1:]); d0 /= np.linalg.norm(d0); p0 = np.array([-d0[1], d0[0]])
    ang = np.arange(0, 360, 2.0); t = np.arange(0, 20, 0.02); keep = None
    fig, ax = plt.subplots(1, 3, figsize=(27, 9.6))
    for k in range(k0 - 2, k0 + 3):
        if not 0 <= k < v.shape[0]:
            continue
        img = v[k] / M
        D = native_of(drawn.astype(np.uint8), k, img.shape, ps, dz) > 0
        # local level of the bath: normalised smoothing of the image over the drawn bath well away from the eye
        far = D & (native_of((ndi.distance_transform_edt(lab != 2) * mri.ISO > 3.0).astype(np.uint8) if False else np.ones_like(lab, np.uint8), k, img.shape, ps, dz) > 0)
        ok = far & (img > 0.6) & (img < 1.3)
        sg = 3.0 / ps
        level = ndi.gaussian_filter(np.where(ok, img, 0.0), sg) / np.maximum(ndi.gaussian_filter(ok.astype(float), sg), 1e-3)
        level = np.where(ndi.gaussian_filter(ok.astype(float), sg) > 0.02, level, 1.0)
        tis = np.clip(img - level * D, 0, None)                          # tissue-only image
        sm = ndi.gaussian_filter(tis, 0.6); raw = ndi.gaussian_filter(img, 0.6)
        res = []
        for a in ang:
            dd = np.cos(np.deg2rad(a)) * d0 + np.sin(np.deg2rad(a)) * p0
            P = vc[:, None] + (t / ps)[None] * dd[:, None]
            p = ndi.map_coordinates(sm, P, order=1, mode="constant"); pr = ndi.map_coordinates(raw, P, order=1, mode="constant")
            w = np.flatnonzero(pr > FLUID)
            if not len(w):
                res.append((a, np.nan, np.nan, np.nan)); continue
            i0 = w[-1]; q = p[i0:i0 + 200]                               # 4 mm beyond the fluid
            low = q < OUT_THR; end = np.nan
            for j in np.flatnonzero(low):
                if low[j:j + 25].all() and j + 25 <= len(q):             # 0.5 mm without tissue
                    end = j * 0.02; break
            li = np.flatnonzero(q < LIGHT); light = (li[0] * 0.02) if len(li) else np.nan
            res.append((a, t[i0], light if np.isfinite(end) and light <= end else np.nan, end))
        res = np.array(res)
        for a, rf, light, end in res:
            sector = "front" if (a <= 40 or a >= 320) else "flank" if (60 <= a <= 120 or 240 <= a <= 300) else "back" if 140 <= a <= 220 else "oblique"
            rows.append(dict(eye=sid, group=scan["group"], slice=k - k0, angle_deg=a, sector=sector, fluid_edge_mm=round(float(rf), 2) if np.isfinite(rf) else "",
                             retina_choroid_mm=round(float(light), 2) if np.isfinite(light) else "", wall_mm=round(float(end), 2) if np.isfinite(end) else ""))
        if k == k0:
            keep = (img, tis, D, res)
    img, tis, D, res = keep; hi = 1.9
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=hi); ax[0].contour(D, [.5], colors="deepskyblue", linewidths=.6)
    ax[0].set_title(f"{sid} stored slice {k0}: as scanned; blue = where the oil bath is drawn", fontsize=13)
    ax[1].imshow(tis, cmap="gray", vmin=0, vmax=hi); ax[1].set_title("oil bath subtracted: tissue only (approximate under the bath)", fontsize=13)
    ax[2].imshow(tis, cmap="gray", vmin=0, vmax=hi)
    for col, lbl, off in (("yellow", "edge of the fluid", lambda q: q[:, 1]), ("orangered", "outer edge of retina with choroid", lambda q: q[:, 1] + q[:, 2]),
                          ("lime", "outer edge of the wall", lambda q: q[:, 1] + q[:, 3])):
        rr = off(res); okk = np.isfinite(rr); a = np.deg2rad(res[okk, 0])
        xy = vc[:, None] + (rr[okk] / ps)[None] * (np.cos(a)[None] * d0[:, None] + np.sin(a)[None] * p0[:, None])
        ax[2].plot(xy[1], xy[0], ".", color=col, ms=2.5, label=lbl)
    ax[2].legend(loc="lower right", fontsize=11, markerscale=4); ax[2].set_title("layers found along rays from the globe centre", fontsize=13)
    half = 15.5 / ps
    for A in ax:
        A.set_xlim(vc[1] - half, vc[1] + half); A.set_ylim(vc[0] + half, vc[0] - half); A.set_xticks([]); A.set_yticks([])
    plt.tight_layout(); plt.savefig(f"{od}/{sid}/{sid}_MRI_layers.png", dpi=55); plt.close()
    return sid


if __name__ == "__main__":
    cfg = run.load_cfg(); od = cfg["out_dir"]; rows = []
    scans = [sc for sc in cfg["scans"] if sc.get("mri") and (not sys.argv[1:] or sc["id"] in sys.argv[1:])]
    for sc in scans:
        print(one(od, sc, rows), flush=True)
    if not sys.argv[1:]:
        with open(f"{od}/mri_wall_layers_rays.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    d = pd.DataFrame(rows).replace("", np.nan); d[["retina_choroid_mm", "wall_mm"]] = d[["retina_choroid_mm", "wall_mm"]].astype(float)
    t = d[d.sector != "oblique"].groupby(["eye", "sector"]).agg(wall_mm=("wall_mm", "median"), wall_found=("wall_mm", lambda x: round(x.notna().mean(), 2)),
                                                          retina_choroid_mm=("retina_choroid_mm", "median")).round(2).reset_index()
    if not sys.argv[1:]:
        t.to_csv(f"{od}/mri_wall_layers.csv", index=False)
    print(t.pivot(index="eye", columns="sector", values=["wall_mm", "retina_choroid_mm", "wall_found"]).to_string())
