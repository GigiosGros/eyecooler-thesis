"""Apply the optic-nerve stump labelling (segment.nerve_stump) to the existing CT outputs without re-running the
segmentation: reads <ID>.nrrd, <ID>.seg.nrrd, <ID>_globe_surface.npz and <ID>_results.json, relabels only voxels
outside the globe, and rewrites <ID>.seg.nrrd, <ID>_qc.png and the stump entries of <ID>_results.json. The globe
outline, the eye frame and every dimension stay exactly as they are. run.py does the same step in a full run.

    python relabel_stump.py            # all eyes, then the tables
    python relabel_stump.py H6 F17
"""
import json, os, sys
import numpy as np, nrrd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run
from ctseg import segment, export

if __name__ == "__main__":
    cfg = run.load_cfg(); od0 = cfg["out_dir"]
    for sid in (sys.argv[1:] or [sc["id"] for sc in cfg["scans"]]):
        od = f"{od0}/{sid}"; r = json.load(open(f"{od}/{sid}_results.json")); s, s0, f = r["voxel_work_mm"], r["voxel_full_mm"], r["downsample"]
        v, _ = nrrd.read(f"{od}/{sid}.nrrd"); v = v.transpose(2, 1, 0)
        lab, _ = nrrd.read(f"{od}/{sid}.seg.nrrd"); lab = np.ascontiguousarray(lab.transpose(2, 1, 0))
        z = np.load(f"{od}/{sid}_globe_surface.npz"); surf = dict(center_mm=z["center_mm"], R=z["R_mm"], measured=z["measured"])
        gl = segment.star_mask(v.shape, s, surf)
        pk = next((p for p in r["thermistors"]["picks"] if p["sensor"] == "optic"), None)
        if pk is None:
            print(sid, "no optic-nerve sensor pick"); continue
        lab[lab == 12] = 7                       # a previous stump goes back to the pool of solids before re-tracking
        ring = r["anatomy"]["axis"].get("ring")
        if ring and ring.get("accepted"):        # the ring plate fused with the tube was called "tube wall": it is holder
            n_fix = int(export.plate_to_holder(lab, s, ring)); r["segmentation"]["holder_plate_relabelled_mm3"] = round(n_fix * s ** 3, 1)
        stump, sinfo = segment.nerve_stump(lab, gl, surf, s, np.array(pk["work_mm_kji"]))
        lab[stump] = 12
        r["anatomy"]["nerve_stump"] = sinfo
        c = surf["center_mm"]; d = (np.array(pk["work_mm_kji"]) - c); d /= np.linalg.norm(d)
        site = c + d * float(segment.star_lookup(surf, (c + d * 5.0)[None])[0][0])       # on the wall, at the optic-nerve sensor
        r["anatomy"]["nerve_site_mm"] = site.round(3).tolist()
        export.point_markup(f"{od}/{sid}_nerve_site.mrk.json", "nerve_site", (site + (f - 1) / 2 * s0)[::-1])
        r["volumes_mm3"] = {segment.NAMES[k]: round(float((lab == k).sum() * s ** 3), 1) for k in segment.NAMES if k > 0}
        json.dump(r, open(f"{od}/{sid}_results.json", "w"), indent=1, default=float)
        export.write_seg(od, sid, lab, s, f, s0)
        export.qc(f"{od}/{sid}_qc.png", sid, v, lab, gl, s, f, s0, r["anatomy"], r["thermistors"]["beads"], r, surf)
        print(sid, sinfo["length_mm"], "mm long,", sinfo["diameter_mm_median"], "mm diameter,", sinfo["vol_mm3"], "mm3", flush=True)
    run.tables(cfg)
