"""Three checks and tables built on the MRI and CT outputs (run after run.py / relabel_stump.py).

    python mri_extras.py measurements   -> outputs/mri_measurements.csv   (lens, anterior chamber, cornea, volumes per pig eye)
    python mri_extras.py drawing        -> outputs/_MRI_layer_marking/drawing_check.csv
                                           (hand-drawn coloured lines on fresh eye 17, slice 18, against the labels)
    python mri_extras.py overlay        -> outputs/<ID>/<ID>_CT_MRI_overlay.png, outputs/ct_mri_sensor_anatomy.csv
                                           (MRI structures brought onto the CT of the same eye; where each sensor sits)

Overlay: the two scans are aligned rigidly from three landmarks each: the globe centre, the eye axis (centre to
lens) and the direction of the optic-nerve stump. No scaling, no image registration. Two landmarks plus an axis
cannot tell a scan from its mirror image about the plane of the axis and the stump, and the eye changes shape
between the scans (different days, different holder), so positions off that plane are approximate.
"""
import csv, json, os, sys
import numpy as np, nrrd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run
from ctseg import segment, mri
from ctseg.export import _plane

PIGS = ["F15", "F16", "F17", "F18", "F19", "P11", "P12", "P13", "P14", "P15"]


def measurements(od):
    rows = []
    for e in PIGS:
        r = json.load(open(f"{od}/{e}/{e}_MRI_results.json")); g = r["segmentation"]; a = r["anatomy"]; m = r.get("measurements", {}); l = a["lens"]
        rows.append(dict(eye=e, cohort="fresh" if e[0] == "F" else "preserved (collapsed)", globe_vol_mL=r["globe_vol_mL"], axial_length_mm=a["axial_length_mm"],
                         equatorial_max_mm=a["equatorial"]["max_mm"], equatorial_min_mm=a["equatorial"]["min_mm"], vitreous_mL=g["vitreous_mL"],
                         aqueous_mm3=g["aqueous_mm3"], lens_vol_mm3=l.get("vol_mm3"), lens_nucleus_mm3=l.get("nucleus_vol_mm3"),
                         lens_diameter_mm=m.get("lens_diameter_mm"), lens_thickness_mm=m.get("lens_thickness_mm"),
                         anterior_chamber_depth_mm=m.get("anterior_chamber_depth_mm"), cornea_thickness_mm=m.get("cornea_thickness_mm"),
                         iris_ciliary_mm3=g["iris_ciliary_mm3"], retina_choroid_mm3=g.get("retina_choroid_mm3"), nerve_stump_mm3=g["nerve_stump_mm3"],
                         oil_shift_rows_mm=g["oil_shift_applied_mm"], oil_shift_slices_mm=g["oil_shift_across_slices_applied_mm"]))
    with open(f"{od}/mri_measurements.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).drop(columns=["cohort", "oil_shift_rows_mm", "oil_shift_slices_mm", "lens_nucleus_mm3"]).to_string(index=False))


def drawing(od):
    from PIL import Image
    D = f"{od}/_MRI_layer_marking"; mp = json.load(open(f"{D}/pristine/mapping.json"))
    a = np.asarray(Image.open(f"{D}/marked_round1/F17_MRI_slice18_for_marking.png").convert("RGB")).astype(int)
    p = np.asarray(Image.open(f"{D}/pristine/F17_MRI_slice18_for_marking.png").convert("RGB")).astype(int)
    ch = np.abs(a - p).sum(2) > 60
    lab, _ = nrrd.read(f"{od}/F17/F17_MRI.seg.nrrd"); lab = lab.transpose(2, 1, 0); s = mri.ISO
    r = json.load(open(f"{od}/F17/F17_MRI_results.json")); dz = r["acquisition"]["slice_mm"]; ps = mp["native_pixel_mm"]
    cols = {"yellow": ((240, 240, 0), "cornea", [11]), "light blue": ((0, 150, 220), "lens (its edge)", [10, 15]), "red": ((220, 30, 30), "iris", [14]),
            "pink": ((245, 170, 195), "ciliary body", [14]), "yellow-green": ((175, 220, 30), "choroid", [18]), "purple": ((150, 75, 150), "retina", [18]),
            "brown": ((175, 120, 75), "vitreous", [3]), "lavender": ((195, 170, 220), "optic nerve", [12]), "dark blue": ((50, 75, 200), None, None)}
    rows = []
    for name, (rgb, tissue, want) in cols.items():
        m = ch & (np.abs(a - np.array(rgb)).sum(2) < 90)
        L, n = ndi.label(ndi.binary_dilation(m, iterations=3))
        for i in range(1, n + 1):
            ys, xs = np.nonzero(m & (L == i))
            if len(ys) < 30:
                continue
            t, w = tissue, want
            if name == "dark blue":                              # two tissues in one colour: the short arc behind the cornea, the long strokes on the wall
                t, w = ("aqueous", [13]) if (xs.max() - xs.min()) < 330 and ys.mean() < 320 else ("sclera", [2])
            iso = np.stack([np.full(len(ys), mp["native_slice"] * dz / s), (mp["crop_rows"][0] + ys / mp["zoom"]) * ps / s, (mp["crop_cols"][0] + xs / mp["zoom"]) * ps / s])
            target = np.isin(lab, w)
            if "edge" in t:
                target = target & ~ndi.binary_erosion(target)
            dist = ndi.distance_transform_edt(~target) * s
            dd = ndi.map_coordinates(dist, iso, order=1)
            at = ndi.map_coordinates(lab, iso, order=0); names = [segment.NAMES[k] for k in np.bincount(at).argsort()[::-1][:2] if (at == k).any()]
            rows.append(dict(colour=name, tissue_as_read=t, stroke_px=len(ys), distance_to_my_label_mm_median=round(float(np.median(dd)), 2),
                             distance_mm_90th_pct=round(float(np.percentile(dd, 90)), 2), share_within_0p3mm=round(float((dd <= 0.3).mean()), 2),
                             my_labels_under_the_stroke=" / ".join(names)))
    with open(f"{D}/drawing_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))


def _frame(gc, ez, stump_c):
    ez = ez / np.linalg.norm(ez); d = stump_c - gc; ex = d - (d @ ez) * ez; ex /= np.linalg.norm(ex)
    return np.stack([ex, np.cross(ez, ex), ez]), float(np.degrees(np.arccos(np.clip((d / np.linalg.norm(d)) @ ez, -1, 1))))


def overlay(od):
    rows = []
    for e in PIGS:
        rc = json.load(open(f"{od}/{e}/{e}_results.json")); rm = json.load(open(f"{od}/{e}/{e}_MRI_results.json")); sc, sm = rc["voxel_work_mm"], mri.ISO
        vc, _ = nrrd.read(f"{od}/{e}/{e}.nrrd"); vc = vc.transpose(2, 1, 0)
        lc, _ = nrrd.read(f"{od}/{e}/{e}.seg.nrrd"); lc = lc.transpose(2, 1, 0)
        lm, _ = nrrd.read(f"{od}/{e}/{e}_MRI.seg.nrrd"); lm = lm.transpose(2, 1, 0)
        if not (lc == 12).any() or not (lm == 12).any() or not rm["anatomy"]["lens"].get("found"):
            print(e, "skipped: stump or lens missing in one scan"); continue
        gcc = np.array(rc["anatomy"]["frame"]["origin_mm"]); gcm = np.array(rm["anatomy"]["frame"]["origin_mm"])
        Fc, ac = _frame(gcc, np.array(rc["anatomy"]["frame"]["axes"])[2], np.argwhere(lc == 12).mean(0) * sc)
        Fm, am = _frame(gcm, np.array(rm["anatomy"]["frame"]["axes"])[2], np.argwhere(lm == 12).mean(0) * sm)
        to_m = lambda pc: gcm + (np.atleast_2d(pc) - gcc) @ Fc.T @ Fm            # CT point (working mm) -> MRI point (mm)
        dl = ndi.distance_transform_edt(~np.isin(lm, [10, 15])) * sm
        gm = np.isin(lm, [2, 3, 10, 11, 13, 14, 15, 16, 18]); dout = ndi.distance_transform_edt(gm) * sm
        for q in rc["thermistors"]["picks"]:
            pm = to_m(np.array(q["work_mm_kji"]))[0] / sm
            lbl = int(ndi.map_coordinates(lm, pm[:, None], order=0)[0])
            rows.append(dict(eye=e, sensor=q["sensor"], mri_structure_at_sensor=segment.NAMES[lbl], distance_to_lens_mm=round(float(ndi.map_coordinates(dl, pm[:, None], order=1)[0]), 2),
                             depth_inside_mri_globe_mm=round(float(ndi.map_coordinates(dout, pm[:, None], order=1)[0]), 2), depth_below_ct_outline_mm=q["depth_below_surface_mm"],
                             out_of_plane_mm=round(float((np.array(q["work_mm_kji"]) - gcc) @ Fc[1]), 2),
                             ct_axis=rc["anatomy"]["axis"]["source"], stump_angle_ct_deg=round(ac, 1), stump_angle_mri_deg=round(am, 1),
                             ct_minus_mri_axial_mm=round(rc["anatomy"]["axial_length_mm"] - rm["anatomy"]["axial_length_mm"], 2)))
        # figure: the plane of the axis and the stump
        t = np.arange(-17, 17, sc); U, V = np.meshgrid(t, t, indexing="ij")
        Pc = gcc[None, None] + U[..., None] * Fc[2] + V[..., None] * Fc[0]
        img = ndi.map_coordinates(vc, (Pc / sc).reshape(-1, 3).T, order=1, mode="constant").reshape(U.shape)
        Pm = to_m(Pc.reshape(-1, 3)); lm_on_ct = ndi.map_coordinates(lm, (Pm / sm).T, order=0, mode="constant").reshape(U.shape)
        lct = ndi.map_coordinates(lc, (Pc / sc).reshape(-1, 3).T, order=0, mode="constant").reshape(U.shape)
        g_ = vc[lc == 2]; lo, hi = np.percentile(g_, [1, 99.5]); lo -= 1.5 * (hi - lo) if rc["bath"] == "liquid" else 0
        fig, ax = plt.subplots(1, 2, figsize=(19, 9.6)); ext = [t[0], t[-1], t[0], t[-1]]
        ax[0].imshow(img.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=ext)
        for ks, col, name in (([2, 3, 10, 11, 13, 14, 15, 16, 18], "yellow", "MRI globe"), ([10, 15], "cyan", "MRI lens"), ([13], "deepskyblue", "MRI aqueous"),
                              ([11], "salmon", "MRI cornea"), ([12], "lime", "MRI nerve stump")):
            mk = np.isin(lm_on_ct, ks)
            if mk.any():
                ax[0].contour(t, t, mk.T, [.5], colors=col, linewidths=1.1); ax[0].plot([], [], color=col, label=name)
        ax[0].contour(t, t, np.isin(lct, [2, 3, 4, 5]).T, [.5], colors="red", linewidths=.9); ax[0].plot([], [], color="red", label="CT globe outline")
        for q in rc["thermistors"]["picks"]:
            d = np.array(q["work_mm_kji"]) - gcc
            ax[0].plot(d @ Fc[2], d @ Fc[0], "o", mfc="none", mec="white", ms=9, mew=1.6)
            ax[0].text(d @ Fc[2] + 0.5, d @ Fc[0] + 0.5, q["sensor"][:4], color="white", fontsize=9)
        ax[0].plot([], [], "o", mfc="none", mec="white", label="sensors (projected onto the plane)"); ax[0].legend(loc="upper left", fontsize=9)
        ax[0].set_title(f"{e}: CT section through the eye axis (right = front) and the nerve stump, MRI structures overlaid", fontsize=12)
        cm = plt.matplotlib.colors.ListedColormap([segment.COLORS[k] for k in range(len(segment.COLORS))])
        ax[1].imshow(lm_on_ct.T, cmap=cm, vmin=-.5, vmax=len(segment.COLORS) - .5, origin="lower", extent=ext, interpolation="nearest")
        ax[1].contour(t, t, np.isin(lct, [2, 3, 4, 5]).T, [.5], colors="red", linewidths=.9)
        ax[1].set_title(f"MRI labels in the same plane, CT globe outline in red. Stump {ac:.0f} deg (CT) vs {am:.0f} deg (MRI) from the axis", fontsize=12)
        for A in ax:
            A.set_xlabel("mm"); A.set_ylabel("mm")
        plt.tight_layout(); plt.savefig(f"{od}/{e}/{e}_CT_MRI_overlay.png", dpi=60); plt.close(); print(e, "overlay written", flush=True)
    with open(f"{od}/ct_mri_sensor_anatomy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import pandas as pd
    pd.set_option("display.width", 250); d = pd.DataFrame(rows)
    print(d.pivot(index="eye", columns="sensor", values="mri_structure_at_sensor").to_string())
    print(d.groupby("eye")[["stump_angle_ct_deg", "stump_angle_mri_deg", "ct_minus_mri_axial_mm"]].first().to_string())


if __name__ == "__main__":
    od = run.load_cfg()["out_dir"]
    {"measurements": measurements, "drawing": drawing, "overlay": overlay}[sys.argv[1]](od)
