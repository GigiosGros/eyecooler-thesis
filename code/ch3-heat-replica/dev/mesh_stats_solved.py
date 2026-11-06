"""Statistics of the meshes the thesis results were solved on, for the mesh tables of Chapter 3.

    python dev/mesh_stats_solved.py

Eye model: the finest stored COMSOL mesh of each ring (Ring Comparison.mph), with the air and the
module (TEC) layer removed by the same rule as dev/new_runs.py (transform): elements of those domains
go, nodes used by no remaining element go, and a boundary triangle stays if all three of its nodes stay.
For comparison the same numbers are written for the mesh as stored, with the air and the module layer.

Heat exchanger: the "normal" (intermediate) mesh of each design, on which the W13 sweep was solved
(dev/wb_runs.py), read with dev/wb_common.load_mesh (triangles that bound no volume removed).

Element quality is COMSOL's skewness: the equiangular skew of the element,
    1 - max((a_max - a_e) / (180 - a_e), (a_e - a_min) / a_e),
taken over the angles of the element, with a_e the angle of the regular tetrahedron. COMSOL does not say
which angles; the script computes it with the face angles (a_e = 60 deg) and with the dihedral angles
(a_e = 70.53 deg) and checks both against COMSOL's own report of the stored meshes (mesh-statistics
screenshots, 2025-12-11): the measure that reproduces it is the one reported.

Writes results/mesh_stats_solved/mesh_stats.csv and MANIFEST.json. Reads meshes only, no solve.
"""
import csv
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "dev"))
import run_all                                          # noqa: E402
from replica.mphread import Mph                         # noqa: E402

OUT = os.path.join(HERE, "results", "mesh_stats_solved")
EYE_FILE = r"Ring Comparison\Ring Comparison.mph"
# COMSOL's report of the stored eye meshes (with air and module layer): min and average skewness
COMSOL_REPORT = {"v1": (0.094, 0.671), "v2": (0.014, 0.670), "v3": (0.019, 0.667)}
FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def face_angles(P):
    """The 12 face angles of each tetrahedron [deg]; P is (n, 4, 3)."""
    out = []
    for f in FACES:
        for i in range(3):
            a, b, c = P[:, f[i]], P[:, f[(i + 1) % 3]], P[:, f[(i + 2) % 3]]
            cosv = np.einsum("ij,ij->i", _unit(b - a), _unit(c - a))
            out.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
    return np.stack(out, 1)


def dihedral_angles(P):
    """The 6 dihedral angles of each tetrahedron [deg]."""
    out = []
    for i, j in EDGES:
        k, l = [m for m in range(4) if m not in (i, j)]
        e = _unit(P[:, j] - P[:, i])
        u = P[:, k] - P[:, i]
        v = P[:, l] - P[:, i]
        u = _unit(u - np.einsum("ij,ij->i", u, e)[:, None] * e)
        v = _unit(v - np.einsum("ij,ij->i", v, e)[:, None] * e)
        out.append(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, v), -1, 1))))
    return np.stack(out, 1)


def skew(ang, ideal):
    return 1.0 - np.maximum((ang.max(1) - ideal) / (180.0 - ideal), (ideal - ang.min(1)) / ideal)


def stats(points, tets, n_tris, chunk=200_000):
    vol = np.empty(len(tets))
    s_face = np.empty(len(tets))
    s_dih = np.empty(len(tets))
    for a in range(0, len(tets), chunk):
        P = points[tets[a:a + chunk]]
        x, y, z = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]
        vol[a:a + chunk] = np.abs(np.einsum("ij,ij->i", np.cross(x, y), z)) / 6
        s_face[a:a + chunk] = skew(face_angles(P), 60.0)
        s_dih[a:a + chunk] = skew(dihedral_angles(P), np.degrees(np.arccos(1 / 3)))
    return dict(vertices=len(points), tetrahedra=len(tets), boundary_triangles=int(n_tris),
                skew_face_min=round(float(s_face.min()), 3), skew_face_mean=round(float(s_face.mean()), 3),
                skew_dihedral_min=round(float(s_dih.min()), 3), skew_dihedral_mean=round(float(s_dih.mean()), 3),
                volume_cm3=round(float(vol.sum()) * 1e6, 2))


def eye_rows():
    path = run_all.mph_path(EYE_FILE)
    mph = Mph(path)
    rows = []
    for comp, info in mph.components().items():
        ring = info["label"].split()[-1]                       # "SCR v1" -> "v1"
        if info["geom"] not in mph.mesh_resources():
            continue
        m = mph.mesh(mph.mesh_resources()[info["geom"]]["resource"])
        names = {}
        for mat in mph.materials(comp, 0):
            if mat["dim"] == 3:
                for d in mat["entities"]:
                    names[d] = mat["name"]
        drop = [d for d, n in names.items() if "air" in n.lower() or "tec layer" in n.lower()]
        rows.append(dict(model="eye", design=ring, mesh="finest, as stored (with air and module layer)",
                         removed="", **stats(m["points"], m["tets"], len(m["tris"]))))
        keep_el = ~np.isin(m["tet_dom"], drop)
        tets = m["tets"][keep_el]
        used = np.unique(tets)
        new = np.full(len(m["points"]), -1, np.int64)
        new[used] = np.arange(len(used))
        keep_tri = (new[m["tris"]] >= 0).all(1)
        rows.append(dict(model="eye", design=ring, mesh="finest, as solved (air and module layer removed)",
                         removed="; ".join(sorted({names[d] for d in drop})),
                         **stats(m["points"][used], new[tets], keep_tri.sum())))
        print(rows[-2]["design"], {k: rows[-2][k] for k in ("vertices", "skew_face_min", "skew_face_mean",
                                                             "skew_dihedral_min", "skew_dihedral_mean")}, flush=True)
        del m
    return path, rows


def hx_rows():
    import wb_common as wb
    rows, paths = [], []
    for v in (1, 2, 3):
        p = wb.model_file(v)
        paths.append(p)
        m = wb.load_mesh(Mph(p), "normal")
        rows.append(dict(model="heat exchanger", design=f"v{v}", mesh="intermediate (normal), as solved",
                         removed="", **stats(m["points"], m["tets"], len(m["tris"]))))
        print("hx", rows[-1], flush=True)
    return paths, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    eye_path, rows = eye_rows()
    # which angle definition reproduces COMSOL's report of the stored meshes
    err = {}
    for kind in ("face", "dihedral"):
        err[kind] = max(max(abs(r[f"skew_{kind}_min"] - COMSOL_REPORT[r["design"]][0]),
                            abs(r[f"skew_{kind}_mean"] - COMSOL_REPORT[r["design"]][1]))
                        for r in rows if "as stored" in r["mesh"])
    best = min(err, key=err.get)
    print("largest difference from COMSOL's report:", err, "->", best, flush=True)
    hx_paths, hx = hx_rows()
    rows += hx
    for r in rows:
        r["quality_min"] = r[f"skew_{best}_min"]
        r["quality_mean"] = r[f"skew_{best}_mean"]
    with open(os.path.join(OUT, "mesh_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(dict(created=datetime.datetime.now().isoformat(timespec="seconds"), code="dev/mesh_stats_solved.py",
                       inputs=[eye_path] + hx_paths, outputs=["mesh_stats.csv"],
                       quality_measure=f"skewness with {best} angles",
                       check_against_comsol_report=dict(largest_abs_difference=err, comsol=COMSOL_REPORT)), f, indent=1)


if __name__ == "__main__":
    main()
