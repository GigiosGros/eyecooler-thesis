"""Chapter 3: temperature in the plane of symmetry after 35 min of cooling, rings v1-v3.

The values are COMSOL's own: the solution stored inside Ring Comparison.mph,
read without COMSOL by extract_comsol_plane_exact.py in
03 - Experiments & Data/Analysis Code/scr-heat-replica (the stored values are
located through the file's own numbering of unknowns; see that script and
the copied extraction manifest).  This script only snapshots the extracted
plane fields and draws them.

Source: Analysis Code/scr-heat-replica/results/comsol_plane/
Run:    python code/figures/aim2_temperature_field.py
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection

from provenance import FIG, SOURCES, THESIS, record

BOX = THESIS.parents[2]
ORIGIN = BOX / "03 - Experiments & Data" / "Analysis Code" / "scr-heat-replica" / "results" / "comsol_plane"
FILES = {"plane_fields_35min.npz": "comsol_plane__plane_fields_35min.npz",
         "MANIFEST.json": "comsol_plane__extraction_manifest.json"}
T_LO, T_HI, STEP = 4.0, 37.0, 2.0          # common color scale and isotherm spacing, in deg C

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def main():
    dst_dir = SOURCES / "data" / "model"
    dst_dir.mkdir(parents=True, exist_ok=True)
    snaps = []
    for src, name in FILES.items():
        if (ORIGIN / src).exists():                      # refresh the snapshot when the source is at hand
            shutil.copyfile(ORIGIN / src, dst_dir / name)
        snaps.append(dst_dir / name)
    d = np.load(snaps[0])
    versions = sorted({k.split("_")[0] for k in d.files})
    fig, axs = plt.subplots(1, len(versions), figsize=(7.4, 5.3), sharey=True, constrained_layout=True)
    levels = np.arange(T_LO + STEP, T_HI, STEP)
    for ax, letter, v in zip(axs, "ABC", versions):
        x, z, tris, T, dom = (d[f"{v}_{k}"] for k in ("x", "z", "tris", "T", "dom"))
        tri = mtri.Triangulation(x, z, tris)
        im = ax.tripcolor(tri, T, shading="gouraud", cmap="viridis", vmin=T_LO, vmax=T_HI, rasterized=True)
        ax.tricontour(tri, T, levels=levels, colors="white", linewidths=0.35, alpha=0.75)
        # outlines of the tissues and the ring: plane edges shared by two different domains
        # (the corners are listed per triangle, so points are identified by their coordinates)
        _, pid = np.unique(np.round(np.stack([x, z], 1), 6), axis=0, return_inverse=True)
        pid = pid.ravel()
        first = np.zeros(pid.max() + 1, int)
        first[pid] = np.arange(len(pid))
        pt = pid[tris]
        e = np.sort(np.concatenate([pt[:, [0, 1]], pt[:, [1, 2]], pt[:, [0, 2]]]), axis=1)
        ed = np.tile(dom, 3)
        o = np.lexsort((e[:, 1], e[:, 0]))
        e, ed = e[o], ed[o]
        inner = np.where((e[1:] == e[:-1]).all(1) & (ed[1:] != ed[:-1]))[0]
        i0, i1 = first[e[inner, 0]], first[e[inner, 1]]
        seg = np.stack([np.stack([x[i0], z[i0]], 1), np.stack([x[i1], z[i1]], 1)], 1)
        ax.add_collection(LineCollection(seg, colors="black", linewidths=0.45, alpha=0.8))
        ax.set_aspect("equal")
        ax.set_title(f"({letter}) {v}", loc="left", fontsize=10)
        ax.set_xlabel("x (mm)", fontsize=9)
    axs[0].set_ylabel("z (mm)", fontsize=9)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02, ticks=np.arange(5, 38, 5))
    cb.set_label("Temperature (°C)", fontsize=9)
    dst = FIG / "temperature-field.png"
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rel = str(ORIGIN.relative_to(BOX)).replace("\\", "/")
    record(__file__, [dst], snaps, origins={s.name: f"{rel}/{src}" for s, src in zip(snaps, FILES)},
           notes="COMSOL stored solution of Ring Comparison.mph at 35 min, read by extract_comsol_plane_exact.py "
                 "(scr-heat-replica). Color scale 4 to 37 C; white isotherms every 2 C; black lines are domain outlines. "
                 "Identification statistics per ring are in the copied extraction manifest.")
    print("wrote", dst.name)


if __name__ == "__main__":
    main()
