"""Boundary conditions of the three heat exchangers as the Python runs of the thesis set them (sweep W13 of
dev/wb_runs.py: heating="annulus", module_plane="air"), drawn for thesis Figure D.2 (nothing is solved).

    python -X utf8 dev/make_hx_boundaries_python.py [output folder]
    default: results/figures_2026-11-11/hx_boundaries_python

Differences from the COMSOL picture (dev/make_model_figures.py fig_hx_boundaries, drawn from COMSOL's heated faces):
  heated face   the module footprint, the annulus of radius 7 to 13 mm, found on the module plane by
                dev/wb_runs.correct_heating (each triangle carries the flux times the fraction of its area inside the
                annulus; triangles with at least half their area inside are drawn red, as in the read-outs)
  dashed lines  the circles of radius 7 and 13 mm; the red triangles follow them to within one triangle
  module plane  outside the annulus: outer wall exchanging heat with the air (module_plane="air")
Inlet, outlet and the other outer walls are those of the model files. Same mesh ("normal") and view as before.
Drawn 6.0 in wide, the text width of the thesis, so that the legend prints at 7.5 pt and the titles at 8 pt.

Writes hx-boundaries.png and MANIFEST.json (inputs, settings, heated areas) in the output folder.
"""
import hashlib
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICA = os.path.dirname(HERE)
sys.path.insert(0, REPLICA)
sys.path.insert(0, HERE)
import make_model_figures as mf  # noqa: E402
import wb_common as wb  # noqa: E402
import wb_runs  # noqa: E402

ANNULUS_MM = tuple(wb_runs.DEFAULTS["annulus_mm"])          # (7.0, 13.0) mm, the setting of sweep W13
COL = {"heated": "#d7301f", "inlet": "#238b45", "outlet": "#ff7f00", "air": "#c6dbef"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(out):
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(6.0, 2.75))
    info = {}
    for j, v in enumerate((1, 2, 3)):
        mph, prob = mf.load_hx(v, "normal")
        faces = wb.face_sets(mph, prob)
        _, heated_sel, hinfo = wb_runs.correct_heating(prob, dict(annulus_mm=ANNULUS_MM, alpha=1.0), dict(P_W=10.0), print)
        P = prob.points
        # every outer triangle: start from the model's sets, then recolour the module plane by the annulus
        # (keyed by the sorted corners, drawn with the triangle as stored so that its orientation and shading are kept)
        key, tri_of = {}, {}
        for name, k in (("Solid outer surface", "air"), ("HotSide", "air"), ("Inlet", "inlet"), ("Outlet", "outlet")):
            for t in faces[name][0]:
                kt = tuple(sorted(t)); key[kt] = k; tri_of.setdefault(kt, t)
        for t in prob.tris[heated_sel]:
            kt = tuple(sorted(t)); key[kt] = "heated"; tri_of.setdefault(kt, t)
        tris = np.array([tri_of[kt] for kt in key])
        cols = np.array([COL[k] for k in key.values()])
        heated_high = P[np.unique(faces["HotSide"][0])][:, 2].mean() > P[:, 2].mean()
        a = fig.add_subplot(1, 3, j + 1, projection="3d")
        a.add_collection3d(Poly3DCollection(P[tris] * 1e3, facecolors=cols, linewidths=0, antialiaseds=False,
                                            shade=True, rasterized=True))
        lo, hi = P.min(0) * 1e3, P.max(0) * 1e3
        th = np.linspace(0, 2 * np.pi, 361)                  # the exact edges of the annulus, on the module plane
        zh = hinfo["plane_z_mm"] + (0.05 if heated_high else -0.05)
        for r in ANNULUS_MM:
            a.plot(r * np.cos(th), r * np.sin(th), np.full_like(th, zh), color="k", lw=0.5, ls=(0, (3, 2)), zorder=10)
        a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_zlim(lo[2], hi[2])
        a.set_box_aspect(hi - lo)
        a.view_init(elev=38 if heated_high else -38, azim=55)
        a.set_axis_off()
        a.set_title(f"heat exchanger v{v}", fontsize=8, pad=0)
        from replica import fem
        drawn = float(fem.tri_area(P, prob.tris[heated_sel]).sum())
        info[f"v{v}"] = dict(hinfo, heated_drawn_mm2=round(drawn * 1e6, 1),
                             comsol_heated_face_mm2=round(float(faces["HotSide"][1].sum()) * 1e6, 1))
    fig.legend(handles=[Patch(facecolor=COL["heated"], label=f"heated annulus, radius {ANNULUS_MM[0]:g}–{ANNULUS_MM[1]:g} mm (module)"),
                        Patch(facecolor=COL["inlet"], label="inlet: velocity and temperature given"),
                        Patch(facecolor=COL["outlet"], label="outlet"),
                        Patch(facecolor=COL["air"], label="outer walls: air convection and radiation"),
                        Line2D([], [], color="k", lw=0.6, ls=(0, (3, 2)), label=f"circles of radius {ANNULUS_MM[0]:g} and {ANNULUS_MM[1]:g} mm")],
               loc="lower center", ncol=2, fontsize=7.5, frameon=False, handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0.25, wspace=0.02)
    png = os.path.join(out, "hx-boundaries.png")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    code = {f: sha(os.path.join(HERE, f)) for f in ("make_hx_boundaries_python.py", "make_model_figures.py", "wb_runs.py", "wb_common.py")}
    with open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(dict(made=time.strftime("%Y-%m-%d %H:%M"), script="dev/make_hx_boundaries_python.py",
                       code_sha256=code, settings=dict(mesh="normal", heating="annulus", annulus_mm=list(ANNULUS_MM),
                                                       module_plane="air", as_in="dev/wb_runs.py sweep W13"),
                       inputs=["dev/wb_inputs/Waterblock_v1/v2/v3_auxiliary_step.lite.mph (mesh normal)"],
                       outputs={"hx-boundaries.png": sha(png)}, heated_area=info), f, indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else os.path.join(REPLICA, "results", "figures_2026-11-11", "hx_boundaries_python"))
