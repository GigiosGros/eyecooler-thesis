"""Automated ex-vivo micro-CT eye segmentation + dimension/thermistor extraction.

Usage (from this folder):
    python run.py                 # all scans in config.json, 4 in parallel, then tables + MANIFEST
    python run.py --only H1 F15   # selected scans (tables are rebuilt from every per-eye JSON present)
    python run.py --jobs 1        # serial
    python run.py --tables-only   # rebuild CSVs + MANIFEST from existing per-eye results
    python run.py --refresh-cache # re-read the DICOM stacks instead of using the cached downsampled volumes

Outputs (out_dir in config.json):
    <ID>/<ID>_results.json, <ID>_qc.png, <ID>.nrrd, <ID>.seg.nrrd, <ID>_thermistor_beads.mrk.json, <ID>_axial_length.mrk.json
    <ID>/<ID>_globe_surface.npz   (smooth_star outline: centre, radius and measured flag per direction)
    eye_dimensions.csv, thermistor_beads.csv, attached_parts.csv, MANIFEST.json
"""
import argparse, csv, datetime, glob, hashlib, json, os, platform, sys, traceback
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_cfg():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    sub = lambda p: p.replace("{box}", cfg["box"]).replace("{ct}", cfg["ct"])
    for k in ("cache_dir", "out_dir", "picks_dir"):
        cfg[k] = sub(cfg[k])
    for sc in cfg["scans"]:
        for k in ("xdir", "local_xdir", "mri"):
            if k in sc:
                sc[k] = sub(sc[k])
    return cfg


def process(sid, refresh=False):
    import numpy as np
    from ctseg import io, segment, anatomy, therm, export
    cfg = load_cfg(); P = cfg["params"]
    scan = next(s for s in cfg["scans"] if s["id"] == sid)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    od = os.path.join(cfg["out_dir"], sid); os.makedirs(od, exist_ok=True)
    f = scan["f"]
    src = io.resolve_dir(scan); scan["read_from"] = src
    cache = os.path.join(cfg["cache_dir"], f"{sid}_f{f}.npz")
    if os.path.exists(cache) and not refresh:
        d = np.load(cache); v, s, s0, raw_max = d["v"], float(d["s"]), float(d["s0"]), int(d["raw_max"])
    else:
        v, s, s0, raw_max = io.load_downsampled(src, f)
        np.savez_compressed(cache, v=v, s=s, s0=s0, raw_max=raw_max)
    scan["marks"] = {}                                         # corrections drawn by hand, by round (outline_marks.py)
    for k, mf in [(1, f"{sid}_outline_marks_LM.csv")] + [(k, f"{sid}_outline_marks_LM_round{k}.csv") for k in range(2, 10)]:
        if os.path.exists(os.path.join(od, mf)):
            mk = np.loadtxt(os.path.join(od, mf), delimiter=",", comments="#", skiprows=3, ndmin=2)
            scan["marks"][k] = np.c_[mk[:, :3], mk[:, 8], 5 * mk[:, 6] + mk[:, 7]]      # point, kind, section number
    cal = segment.calibrate(v, s, scan)
    lab, gl, n, seginfo, surf = segment.segment(v, s, scan, cal, P)
    A = P.get("axis_ring", {})
    ring = anatomy.ring_axis(lab, gl, s, A) if scan["group"] in A.get("groups", []) else None
    beads, tinfo = therm.find_beads(scan, src, f, s0, s, gl, cal, P, raw_max)
    picks = therm.load_picks(cfg["picks_dir"], sid)
    tinfo["pick_validation"] = therm.match_picks(beads, picks)
    conv = (tinfo["pick_validation"] or {}).get("convention", "tf_forward")
    names = cfg["pick_order"]
    optic = None                                # optic-nerve sensor (working-grid mm): only to tell front from back
    if picks is not None:
        for lbl, pc in zip(picks["labels"], picks["candidates"].get(conv, picks["candidates"]["as_saved"])):
            num = str(lbl).rsplit("-", 1)[-1]
            if num.isdigit() and 1 <= int(num) <= len(names) and names[int(num) - 1] == "optic":
                optic = pc[::-1] - (f - 1) / 2 * s0
    anat, lens_m, corn_m = anatomy.analyse(n, lab, gl, s, scan["group"], seginfo["parts"], P["lens_dog_sigmas_mm"], ring=ring,
                                          opposite_mm=optic)
    if anat["axis"]["source"].startswith("pca") and picks is not None and "optic" in names:
        # last resort: the optic-nerve SENSOR marks the posterior pole (NOT independent of the sensors)
        pc = picks["candidates"].get(conv, picks["candidates"]["as_saved"])[names.index("optic")]
        w = pc[::-1] - (f - 1) / 2 * s0
        gcen = np.array(anat["frame"]["origin_mm"])
        anat, lens_m, corn_m = anatomy.analyse(n, lab, gl, s, scan["group"], seginfo["parts"], P["lens_dog_sigmas_mm"],
                                              axis_override=(gcen - w, "optic_sensor(not independent)"))
    pick_rows = therm.picks_in_eye_frame(picks, conv, names, gl, s, f, s0, anat)
    opt = next((q for q in pick_rows if q["sensor"] == "optic"), None)
    lim = P.get("axis_optic_sensor_min_polar_deg")
    if (scan["group"] == "human" and opt and lim and anat["axis"]["independent_of_sensors"]
            and opt["polar_from_anterior_pole_deg"] < lim):
        # the lens / nerve-stump cue points the wrong way: the optic-nerve sensor cannot sit near the equator
        rej = dict(source=anat["axis"]["source"], ez=anat["axis"]["ez"], optic_sensor_polar_deg=opt["polar_from_anterior_pole_deg"])
        gcen = np.array(anat["frame"]["origin_mm"]); ok = False
        if rej["source"].startswith("optic_nerve") and anat["lens"].get("trusted"):
            # the stump cue failed; a trusted lens that it had overruled gets its turn before the sensor fallback
            anat, lens_m, corn_m = anatomy.analyse(n, lab, gl, s, scan["group"], seginfo["parts"], P["lens_dog_sigmas_mm"], ring=ring,
                                                  opposite_mm=optic, nerve_over_lens=False)
            pick_rows = therm.picks_in_eye_frame(picks, conv, names, gl, s, f, s0, anat)
            o2 = next(q for q in pick_rows if q["sensor"] == "optic")
            ok = anat["axis"]["source"] == "lens" and o2["polar_from_anterior_pole_deg"] >= lim
        if not ok:
            anat, lens_m, corn_m = anatomy.analyse(n, lab, gl, s, scan["group"], seginfo["parts"], P["lens_dog_sigmas_mm"],
                                                  axis_override=(gcen - np.array(opt["work_mm_kji"]), "optic_sensor(not independent)"))
            pick_rows = therm.picks_in_eye_frame(picks, conv, names, gl, s, f, s0, anat)
        anat["axis"]["rejected_cue"] = rej
    beads = therm.eye_coords(beads, anat, f, s0)
    if P.get("label_lens_cornea"):       # off by default: the CT lens/cornea are estimates and mislead in the pictures
        if lens_m is not None and anat["lens"].get("trusted"):  # untrusted lens candidates are reported, not labelled
            lab[lens_m & gl & (lab != 5) & (lab != 4)] = 10
        if corn_m is not None:
            lab[corn_m & (lab == 2)] = 11
    if surf is not None and optic is not None:      # optic-nerve stump behind the globe (labels outside the globe only)
        if ring is not None and ring.get("accepted"):       # the ring plate fused with the tube was called tube wall
            seginfo["holder_plate_relabelled_mm3"] = round(int(export.plate_to_holder(lab, s, ring)) * s ** 3, 1)
        stump, sinfo = segment.nerve_stump(lab, gl, surf, s, optic)
        lab[stump] = 12; anat["nerve_stump"] = sinfo
        cs = surf["center_mm"]; dn = (optic - cs) / np.linalg.norm(optic - cs)
        site = cs + dn * float(segment.star_lookup(surf, (cs + dn * 5.0)[None])[0][0])   # on the wall, at the optic-nerve sensor
        anat["nerve_site_mm"] = site.round(3).tolist()
        export.point_markup(os.path.join(od, f"{sid}_nerve_site.mrk.json"), "nerve_site", (site + (f - 1) / 2 * s0)[::-1])
    if surf is not None:
        np.savez_compressed(os.path.join(od, f"{sid}_globe_surface.npz"), center_mm=surf["center_mm"], R_mm=surf["R"],
                            measured=surf["measured"],
                            note="R_mm[i, j]: radius along (cos t, sin t cos p, sin t sin p) in (slice, row, col) of the working "
                                 "grid, t = (i + 0.5) * pi / n_t, p = (j + 0.5) * 2 pi / n_p; center_mm on the working grid")
    gvox = np.argwhere(gl) * s; gc = gvox.mean(0)
    w, V = np.linalg.eigh(np.cov((gvox - gc).T)); pr = (gvox - gc) @ V[:, ::-1]
    gvol = float(gl.sum() * s ** 3)
    import pydicom
    scan_date = str(io.dcm(io.slice_files(src)[0], stop_before_pixels=True).get("StudyDate", ""))
    scan.pop("marks", None)
    info = dict(id=sid, group=scan["group"], bath=scan["bath"], source_dir=scan["xdir"], read_from=src, scan_date=scan_date, downsample=f,
                voxel_full_mm=s0, voxel_work_mm=s, raw_max=raw_max, calibration=cal,
                globe_vol_mL=round(gvol / 1000, 3), ESD_mm=round((6 * gvol / np.pi) ** (1 / 3), 2),
                pca_extent_mm=(pr.max(0) - pr.min(0) + s).round(2).tolist(),
                volumes_mm3={segment.NAMES[k]: round(float((lab == k).sum() * s ** 3), 1) for k in segment.NAMES if k > 0},
                segmentation=seginfo, anatomy=anat, thermistors=dict(info=tinfo, beads=beads, picks=pick_rows),
                coordinate_note="*_mm in (slice,row,col) on the working grid unless named lps_mm/full_mm_kji; "
                                "lps_mm = Slicer LPS of the original full-res scan nrrd (origin 0)")
    json.dump(info, open(os.path.join(od, f"{sid}_results.json"), "w"), indent=1, default=float)
    export.slicer(od, sid, v, lab, s, f, s0, beads, anat)
    export.qc(os.path.join(od, f"{sid}_qc.png"), sid, v, lab, gl, s, f, s0, anat, beads, info, surf)
    msg = f"{sid}: globe {info['globe_vol_mL']} mL, axial {anat['axial_length_mm']} mm ({anat['axis']['source']}), beads {len(beads)}"
    if scan.get("mri"):
        msg += " | " + process_mri(scan, od)
    return msg


def process_mri(scan, od):
    """Same geometry extraction on the pig MRI (isotropic 0.2 mm grid, MRI patient frame with origin 0)."""
    import numpy as np
    from ctseg import mri, anatomy, export, segment
    sid = scan["id"] + "_MRI"
    v, ps, dz, meta = mri.load_series(scan["mri"])
    vi = mri.to_iso(v, ps, dz)
    lab, gl, lens, linfo, minfo = mri.segment(vi, collapsed=bool(scan.get("shell")), oil_shift_mm=meta["oil_shift_predicted_mm"],
                                              slice_mm=dz, slice_shift_mm=load_cfg().get("mri", {}).get("oil_slice_shift_mm", {}).get(scan["group"]))
    an, _, _ = anatomy.analyse(None, lab, gl, mri.ISO, scan["group"], [], None, lens_given=(lens, linfo))
    s = mri.ISO; gvol = float(gl.sum() * s ** 3)
    # lengths along the eye axis, through the lens centre: lens, aqueous (anterior chamber depth), cornea
    meas = {}
    if lens is not None:
        from scipy import ndimage as ndi
        ez = np.array(an["frame"]["axes"])[2]; lc = np.array(linfo["centroid_mm"]); tt = np.arange(-12, 12, 0.02)
        ln = ndi.map_coordinates(lab, ((lc[None] + tt[:, None] * ez[None]) / s).T, order=0, mode="constant")
        run = lambda k: round(float((ln == k).sum() * 0.02), 2)
        meas = dict(lens_thickness_mm=round(run(10) + run(15), 2), anterior_chamber_depth_mm=run(13), cornea_thickness_mm=run(11),
                    lens_diameter_mm=linfo.get("diameter_mm"), note="along the eye axis through the lens centre")
    P = np.argwhere(gl) * s; gc = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - gc).T)); pr = (P - gc) @ V[:, ::-1]
    info = dict(id=sid, modality="MRI", source_dir=scan["mri"], acquisition=meta, voxel_work_mm=s,
                globe_vol_mL=round(gvol / 1000, 3), ESD_mm=round((6 * gvol / np.pi) ** (1 / 3), 2),
                pca_extent_mm=(pr.max(0) - pr.min(0) + s).round(2).tolist(), segmentation=minfo, anatomy=an, measurements=meas,
                volumes_mm3={segment.NAMES[k]: round(float((lab == k).sum() * s ** 3), 1) for k in segment.NAMES if k > 0 and (lab == k).any()},
                notes="globe = inner body (vitreous + aqueous + lens) + wall; wall thickness measured where the wall meets the "
                      "bath, local median elsewhere. Cornea, iris/ciliary body, membranes and nerve stump are assigned by "
                      "position (same signal as the rest of the wall); see ctseg/mri.py segment().")
    json.dump(info, open(os.path.join(od, f"{sid}_results.json"), "w"), indent=1, default=float)
    export.slicer(od, sid, vi, lab, s, 1, s, [], an)
    export.qc(os.path.join(od, f"{sid}_qc.png"), sid, vi, lab, gl, s, 1, s, an, [], info)
    return f"MRI globe {info['globe_vol_mL']} mL, axial {an['axial_length_mm']} mm"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tables(cfg):
    od = cfg["out_dir"]; rows, brow, prow, krow = [], [], [], []
    for sc in cfg["scans"]:
        fn = os.path.join(od, sc["id"], f"{sc['id']}_results.json")
        if not os.path.exists(fn):
            continue
        r = json.load(open(fn)); a = r["anatomy"]; le = a["lens"]; co = a["cornea"]; pv = r["thermistors"]["info"].get("pick_validation") or {}
        rows.append(dict(
            eye=r["id"], group=r["group"], bath=r["bath"], scan_date=r.get("scan_date", ""), voxel_work_mm=round(r["voxel_work_mm"], 4),
            globe_vol_mL=r["globe_vol_mL"], ESD_mm=r["ESD_mm"],
            axial_length_mm=a["axial_length_mm"], axis_source=a["axis"]["source"],
            axis_lens_vs_nerve_deg=a["axis"].get("lens_vs_nerve_deg", ""),
            nerve_cue_vs_axis_deg=a["axis"].get("nerve_cue_vs_axis_deg", ""), lens_cue_vs_axis_deg=a["axis"].get("lens_cue_vs_axis_deg", ""),
            equatorial_max_mm=a["equatorial"]["max_mm"], equatorial_min_mm=a["equatorial"]["min_mm"],
            equatorial_equiv_circle_mm=a["equatorial"]["equiv_circle_mm"],
            pca_D1_mm=r["pca_extent_mm"][0], pca_D2_mm=r["pca_extent_mm"][1], pca_D3_mm=r["pca_extent_mm"][2],
            scleral_sphere_radius_mm=a["scleral_sphere_radius_mm"],
            lens_found=le.get("found"), lens_trusted=le.get("trusted"), lens_vol_mm3=le.get("vol_mm3", ""),
            lens_diameter_mm=le.get("diameter_mm", ""), lens_thickness_mm=le.get("thickness_along_axis_mm", ""),
            apex_to_lens_front_mm=le.get("apex_to_lens_front_mm", ""),
            cornea_resolved=co["resolved"], cornea_diameter_mm=co.get("diameter_mm", ""),
            cornea_sagittal_height_mm=co.get("sagittal_height_mm", ""), cornea_radius_mm=co.get("radius_of_curvature_mm", ""),
            optic_nerve_stump=bool(a["nerve"]), globe_air_void_mm3=r["volumes_mm3"]["globe_air_void"],
            stump_length_mm=(a.get("nerve_stump") or {}).get("length_mm", ""), stump_diameter_mm=(a.get("nerve_stump") or {}).get("diameter_mm_median", ""),
            stump_vol_mm3=(a.get("nerve_stump") or {}).get("vol_mm3", ""),
            globe_outline=r["segmentation"].get("globe_surface", {}).get("method", "opening"),
            outline_measured_fraction=r["segmentation"].get("globe_surface", {}).get("measured_fraction", ""),
            outline_marked_fraction=r["segmentation"].get("globe_surface", {}).get("marked_fraction", ""),
            outline_n_marks=sum(m.get("n", 0) for m in r["segmentation"].get("globe_surface", {}).get("marks", {}).values()) or "",
            n_beads=len(r["thermistors"]["beads"]), metal_saturated_scan=r["thermistors"]["info"]["saturated_scan"],
            picks_convention=pv.get("convention", ""), picks_matched=f"{pv.get('n_matched', '')}/{pv.get('n_picks', '')}" if pv else "",
            picks_median_dist_mm=pv.get("median_dist_mm", "")))
        for i, b in enumerate(r["thermistors"]["beads"]):
            brow.append(dict(eye=r["id"], bead=i + 1, pick_label=b.get("pick_label", ""), pick_dist_mm=b.get("pick_dist_mm", ""),
                             L_mm=b["lps_mm"][0], P_mm=b["lps_mm"][1], S_mm=b["lps_mm"][2],
                             eye_x_mm=b["eye_xyz_mm"][0], eye_y_mm=b["eye_xyz_mm"][1], eye_z_mm=b["eye_xyz_mm"][2],
                             polar_from_anterior_pole_deg=b["polar_from_anterior_pole_deg"], azimuth_deg=b["azimuth_deg"],
                             axial_dist_from_anterior_pole_mm=b["axial_dist_from_anterior_pole_mm"],
                             depth_below_surface_mm=b["depth_below_surface_mm"], bead_diam_mm=b["bead_diam_mm"]))
        for q in r["thermistors"].get("picks", []):
            krow.append(dict(eye=r["id"], group=r["group"], sensor=q["sensor"], pick=q["pick"],
                             depth_below_surface_mm=q["depth_below_surface_mm"],
                             polar_from_anterior_pole_deg=q["polar_from_anterior_pole_deg"], azimuth_deg=q["azimuth_deg"],
                             radius_over_half_axial=q["radius_over_half_axial"],
                             axial_dist_from_anterior_pole_mm=q["axial_dist_from_anterior_pole_mm"],
                             eye_x_mm=q["eye_xyz_mm"][0], eye_y_mm=q["eye_xyz_mm"][1], eye_z_mm=q["eye_xyz_mm"][2],
                             work_k_mm=q["work_mm_kji"][0], work_j_mm=q["work_mm_kji"][1], work_i_mm=q["work_mm_kji"][2],
                             axial_length_mm=a["axial_length_mm"], axis_source=a["axis"]["source"],
                             axis_independent_of_sensors=a["axis"].get("independent_of_sensors", True)))
        for p in r["segmentation"]["parts"]:
            prow.append(dict(eye=r["id"], **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in p.items()}))
    crow = []
    for sc in cfg["scans"]:
        fc = os.path.join(od, sc["id"], f"{sc['id']}_results.json"); fm = os.path.join(od, sc["id"], f"{sc['id']}_MRI_results.json")
        if not (os.path.exists(fc) and os.path.exists(fm)):
            continue
        c, m = json.load(open(fc)), json.load(open(fm))
        row = dict(eye=sc["id"], group=sc["group"], CT_date=c.get("scan_date"), MRI_date=m["acquisition"]["date"])
        for k, get in (("globe_vol_mL", lambda r: r["globe_vol_mL"]), ("ESD_mm", lambda r: r["ESD_mm"]),
                       ("axial_length_mm", lambda r: r["anatomy"]["axial_length_mm"]),
                       ("equatorial_max_mm", lambda r: r["anatomy"]["equatorial"]["max_mm"]),
                       ("equatorial_min_mm", lambda r: r["anatomy"]["equatorial"]["min_mm"]),
                       ("pca_D1_mm", lambda r: r["pca_extent_mm"][0]), ("pca_D3_mm", lambda r: r["pca_extent_mm"][2])):
            a, b = get(c), get(m)
            row[f"CT_{k}"], row[f"MRI_{k}"], row[f"diff_CT_minus_MRI_{k}"] = a, b, round(a - b, 3)
        row["CT_axis_source"] = c["anatomy"]["axis"]["source"]; row["MRI_axis_source"] = m["anatomy"]["axis"]["source"]
        crow.append(row)
    for name, rr in (("eye_dimensions.csv", rows), ("thermistor_beads.csv", brow), ("attached_parts.csv", prow),
                     ("ct_vs_mri.csv", crow), ("sensor_picks_in_eye_frame.csv", krow)):
        if rr:
            with open(os.path.join(od, name), "w", newline="") as fh:
                keys = list(dict.fromkeys(k for row in rr for k in row))
                w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rr)
    # manifest
    import numpy, scipy, skimage, pydicom
    man = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
               packages=dict(numpy=numpy.__version__, scipy=scipy.__version__, skimage=skimage.__version__, pydicom=pydicom.__version__),
               code={os.path.relpath(p, HERE): sha(p) for p in sorted(glob.glob(os.path.join(HERE, "**", "*.py"), recursive=True)
                                                                   + [os.path.join(HERE, "config.json")])},
               inputs={}, outputs={})
    for sc in cfg["scans"]:
        fs = sorted(glob.glob(os.path.join(sc["xdir"], "*.dcm")))
        if fs:
            man["inputs"][sc["id"]] = dict(dir=sc["xdir"], n_files=len(fs), first=os.path.basename(fs[0]), last=os.path.basename(fs[-1]),
                                           first_bytes=os.path.getsize(fs[0]), first_sha256=sha(fs[0]), last_sha256=sha(fs[-1]))
    for p in sorted(glob.glob(os.path.join(od, "**", "*"), recursive=True)):
        if os.path.isfile(p) and not p.endswith("MANIFEST.json") and not p.endswith(".nrrd"):
            man["outputs"][os.path.relpath(p, od)] = sha(p)
        elif p.endswith(".nrrd"):
            man["outputs"][os.path.relpath(p, od)] = f"size {os.path.getsize(p)} (hash skipped)"
    json.dump(man, open(os.path.join(od, "MANIFEST.json"), "w"), indent=1)
    print(f"tables: {len(rows)} eyes, {len(brow)} beads -> {od}")


def _mri_only(sid):
    cfg = load_cfg(); scan = next(s for s in cfg["scans"] if s["id"] == sid)
    if not scan.get("mri"):
        return f"{sid}: no MRI"
    return f"{sid}: " + process_mri(scan, os.path.join(cfg["out_dir"], sid))


def _safe(args):
    sid, refresh = args
    try:
        return _mri_only(sid) if refresh == "mri" else process(sid, refresh)
    except Exception:
        return f"{sid}: FAILED\n{traceback.format_exc()}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*"); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--tables-only", action="store_true"); ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--mri-only", action="store_true", help="redo only the MRI segmentation of the pig eyes (CT outputs untouched)")
    a = ap.parse_args(); cfg = load_cfg()
    if a.mri_only:
        a.refresh_cache = "mri"
    if not a.tables_only:
        ids = a.only or [s["id"] for s in cfg["scans"]]
        if a.jobs > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                for msg in ex.map(_safe, [(i, a.refresh_cache) for i in ids]):
                    print(msg, flush=True)
        else:
            for i in ids:
                print(_safe((i, a.refresh_cache)), flush=True)
    tables(cfg)
