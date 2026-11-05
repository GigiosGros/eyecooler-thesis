"""Where each vitreous sensor sits between the globe centre and the wall (input of the vitreous screen).

For every eye: the vitreous Slicer pick (the point whose label number names the vitreous sensor), its distance r
from the centre of the globe outline, the radius R of the outline in that same direction, and the fraction r / R
(0 = centre, 1 = on the wall). Read from the files run.py wrote (<ID>_results.json, <ID>_globe_surface.npz).

    python vitreous_position.py     -> outputs/vitreous_position.csv

The thesis-analysis package screens the vitreous sensor on this fraction (qc.vitreous_max_centre_fraction); copy the
table to thesis-analysis/config/vitreous_position_ct.csv after the outlines change. Eyes without a surface file
(outline never written as r(direction)) are listed with empty values.
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run
from ctseg import segment

COHORT = {"human": ("Human", "Fresh"), "fresh": ("Porcine", "Fresh"), "preserved": ("Porcine", "Preserved")}

if __name__ == "__main__":
    cfg = run.load_cfg(); od = cfg["out_dir"]; rows = []
    for sc in cfg["scans"]:
        sid = sc["id"]; r = json.load(open(f"{od}/{sid}/{sid}_results.json"))
        pk = next((p for p in r["thermistors"]["picks"] if p["sensor"] == "vitreous"), None)
        species, status = COHORT[r["group"]]
        row = dict(species=species, status=status, marker_eye="E" + sid[1:], eye=sid, outline=r["segmentation"]["globe_surface"]["method"])
        sf = f"{od}/{sid}/{sid}_globe_surface.npz"
        if pk is not None and os.path.exists(sf):
            d = np.load(sf); surf = dict(center_mm=d["center_mm"], R=d["R_mm"], measured=d["measured"])
            Rv, rr = segment.star_lookup(surf, np.array([pk["work_mm_kji"]]))
            row.update(distance_from_centre_mm=round(float(rr[0]), 2), wall_radius_same_direction_mm=round(float(Rv[0]), 2),
                       centre_fraction=round(float(rr[0] / Rv[0]), 3), depth_below_outline_mm=pk["depth_below_surface_mm"])
        rows.append(row)
    keys = ["species", "status", "marker_eye", "centre_fraction", "distance_from_centre_mm", "wall_radius_same_direction_mm",
            "depth_below_outline_mm", "eye", "outline"]
    with open(f"{od}/vitreous_position.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    for q in sorted(rows, key=lambda q: -q.get("centre_fraction", 0)):
        print(q["eye"], q.get("centre_fraction", ""), q.get("distance_from_centre_mm", ""), q.get("wall_radius_same_direction_mm", ""))
