"""Chapter 3: where the eye model is read out and where the sensors were, on the model geometry.

Three pictures on the section of the model geometry in its plane of symmetry (y = 0):
  probe-locations.png            the point probes of the COMSOL model
  sensor-locations.png           the sensors implanted in the nine human donor eyes
  probe-vs-sensor-locations.png  both, for comparison

Geometry  Computational model/Projects/SCR/Eye Refined v1 LM 250911 v25.step, the file
          imported by the model (dimensions in mm; optical axis along z, corneal apex
          near z = 0). A copy is kept in figures/aim2/sources/cad/.
Probes    Coordinates of the Domain Point Probes of the v1 parametric-sweep model, read
          from its model summary (Computational model/Model Summaries/
          SCR_Parametric_Sweep_Ring_SCR_v1_parametric_sweep.html), in m, here in mm.
          All lie in the plane of symmetry except the SCR probe, which is at
          (0, 10, 0) mm, 10 mm from the axis on the rim of the ring; because the ring
          is a body of revolution it is drawn turned into the plane, at (-10, 0).
Sensors   Ex-Vivo/CT Segmentation/outputs/sensor_picks_in_eye_frame.csv (human eyes H1-H9)
          and, for the optic nerve, the per-eye <ID>_results.json of the same folder. A copy
          of the table and the derived optic_sensor_nerve_reference.csv are kept in
          figures/aim2/sources/data/. Each sensor is placed on the MODEL globe from what was
          measured in its own eye, in mm, against an anatomical reference:
            scleral and vitreous sensors -- on the outer surface of the model sclera at the
              same fraction of the axial length from the anterior pole, then inward along
              the normal by the depth below the globe surface measured by micro-CT;
            optic-nerve sensor, nerve stump identified in the scan (H3, H4, H6) -- at its
              measured distance along the nerve from the point where the nerve joins the
              globe, and its measured distance from the axis of the nerve;
            optic-nerve sensor, nerve not identified (other eyes; in H7 and H8 the part
              labelled "nerve" by the pipeline lies about 130 deg from the anterior pole and
              is not the nerve) -- on the model surface at its measured straight-line
              distance from the posterior pole, on the side of the nerve, outward by its
              measured height above the surface. Drawn hollow; not in the mean.
          Sensors excluded from the analysis by the spatial screen of Chapter 2 (thesis-analysis
          results/01_quality_control/spatial_screen.csv, column flagged; copy in sources/data)
          are drawn as a black-edged cross and are left out of the mean position.
          All sensors are turned about the optical axis into the plane of the picture
          (azimuth dropped). The model globe (25.9 mm) is not the donor eyes (22.2-26.6 mm).

The section is cut with the OpenCASCADE kernel of Gmsh (pip install gmsh): each body is
intersected with the plane and the resulting faces are triangulated for filling.

Run:  python code/figures/aim2_probe_locations.py
"""
import json
import shutil

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D

from provenance import FIG, SOURCES, THESIS, record

BOX = THESIS.parents[2]
ORIGIN = (BOX / "03 - Experiments & Data" / "Computational model" / "Projects" / "SCR"
          / "Eye Refined v1 LM 250911 v25.step")
SNAP = SOURCES / "cad" / "Eye Refined v1 LM 250911 v25.step"
PICKS_ORIGIN = (BOX / "03 - Experiments & Data" / "Ex-Vivo" / "CT Segmentation" / "outputs"
                / "sensor_picks_in_eye_frame.csv")
PICKS = SOURCES / "data" / "sensor_picks_in_eye_frame.csv"

Z_APEX, Z_POSTERIOR = 0.0, -25.9          # model globe on the optical axis, mm (from the section)
CENTRE_Z, HALF_AXIAL = (Z_APEX + Z_POSTERIOR) / 2, (Z_APEX - Z_POSTERIOR) / 2
AXIAL = Z_APEX - Z_POSTERIOR
CT_OUT = PICKS_ORIGIN.parent
NERVE_REF = SOURCES / "data" / "optic_sensor_nerve_reference.csv"
SCREEN_ORIGIN = (BOX / "03 - Experiments & Data" / "Analysis Code" / "thesis-analysis" / "results"
                 / "01_quality_control" / "spatial_screen.csv")
SCREEN = SOURCES / "data" / "spatial_screen.csv"
CHANNEL = {"Anterior Sclera": "anterior", "Equatorial Sclera": "equatorial", "Vitreous": "vitreous",
           "Posterior Sclera": "posterior", "Optic Nerve": "optic"}
NERVE_MIN_POLAR_DEG = 155.0               # a "nerve" further than this from the posterior pole is not the nerve

# name in the model, (x, z) in mm, label, colour (site colours of Chapter 2), label offset in points
PROBES = [
    ("SCR 1", (-10.0, 0.0), "SCR", "#1f77b4", (-16, 22)),
    ("Anteriror Sclera Probe 1", (11.53, -8.95), "Anterior sclera", "#9467bd", (10, 4)),
    ("Equatorial Sclera Probe 1", (12.25, -15.0), "Equatorial sclera", "#ff7f0e", (10, -3)),
    ("Vitreous Humor Probe 1", (0.0, -15.0), "Vitreous", "#2ca02c", (9, -3)),
    ("Posterior Sclera Probe 1", (9.6, -21.05), "Posterior sclera", "#8c564b", (10, -8)),
    ("Optic Nerve Head Probe 1", (-1.86, -24.6), "Proximal optic nerve", "#d62728", (9, -12)),
    ("Optic Nerve Probe 1", (-3.0, -34.6), "Distal optic nerve", "#7f7f7f", (9, -3)),
]
# sensor name in the CT table -> (label, colour, side of the picture, label offset in points)
SENSORS = {
    "anterior": ("Anterior sclera", "#9467bd", 1, (12, 10)),
    "equatorial": ("Equatorial sclera", "#ff7f0e", 1, (12, 0)),
    "vitreous": ("Vitreous", "#2ca02c", 1, (-10, 12)),
    "posterior": ("Posterior sclera", "#8c564b", 1, (12, -8)),
    "optic": ("Proximal optic nerve", "#d62728", -1, (12, -14)),
}
# body name in the STEP file -> (fill colour, drawing order)
BODIES = {
    "Air (v1)": ("#ffffff", 0), "Fat (v1)": ("#f7efd9", 0), "Eye Lid": ("#f3d9cc", 1),
    "Extraocular Muscles": ("#e9bdb0", 1), "Sclera": ("#e6e6e6", 2), "Choroid": ("#c98d86", 3),
    "Retina": ("#f2c94c", 4), "Vitreous": ("#eaf4fb", 2), "Cloquet’s Canal": ("#d7e9f5", 3),
    "Lens": ("#cfdcec", 3), "Ciliary Body": ("#d9b8a6", 3), "Acqueous": ("#dbeef9", 2),
    "Cornea": ("#c9dff0", 3), "Ring": ("#9aa3ad", 4), "TEC LAyer": ("#5d6770", 4),
}
TEXT = [("Vitreous humor", (-6.2, -12.0)), ("Lens", (0, -5.9)), ("Ring", (10.3, -0.6)),
        ("Eyelid", (-15.6, -3.2)), ("Eyelid", (15.6, -3.2)), ("Orbital fat", (-11.5, -38.5)),
        ("Retina domain\n(includes the optic nerve)", (6.4, -39.2)),
        ("Muscle", (-9.6, -29.5)),
        ("Muscle", (9.6, -29.5))]


def section():
    """{body: (triangles (n, 3, 2), outline segments (m, 2, 2))} in the plane y = 0, mm."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(SNAP))
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    names = {t: gmsh.model.getEntityName(3, t).split("/")[-1] for _, t in vols}
    faces = {}
    for _, t in vols:
        rect = gmsh.model.occ.addRectangle(-30, -60, 0, 60, 90)
        gmsh.model.occ.rotate([(2, rect)], 0, 0, 0, 1, 0, 0, np.pi / 2)   # into the plane y = 0
        try:
            res, _ = gmsh.model.occ.intersect(gmsh.model.occ.copy([(3, t)]), [(2, rect)],
                                              removeObject=True, removeTool=True)
        except Exception:
            continue                                                       # body does not meet the plane
        faces.setdefault(names[t], []).extend(s for d, s in res if d == 2)
    gmsh.model.occ.remove(vols, recursive=True)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
    gmsh.model.mesh.generate(2)
    out = {}
    for name, surfs in faces.items():
        tris, segs = [], []
        for s in surfs:
            tags, coords, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=True)
            xz = dict(zip(tags.tolist(), np.asarray(coords).reshape(-1, 3)[:, [0, 2]]))
            _, _, nodes = gmsh.model.mesh.getElements(2, s)
            if len(nodes):
                tris.extend([[xz[a], xz[b], xz[c]] for a, b, c in np.asarray(nodes[0]).reshape(-1, 3).tolist()])
            for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
                lo, hi = gmsh.model.getParametrizationBounds(1, c)
                p = np.asarray(gmsh.model.getValue(1, c, np.linspace(lo[0], hi[0], 80).tolist())).reshape(-1, 3)[:, [0, 2]]
                segs.extend(np.stack([p[:-1], p[1:]], axis=1).tolist())
        out[name] = (np.asarray(tris), np.asarray(segs))
    gmsh.finalize()
    return out


def base(sec, labels=True, xmax=19.6):
    fig, ax = plt.subplots(figsize=(5.2 * (xmax + 19.6) / 39.2, 5.25))   # same scale, wider for labels
    sec = {n: v for n, v in sec.items() if n not in ("Air (v1)", "TEC LAyer")}   # as solved: no air, no module layer
    for name, (tris, segs) in sorted(sec.items(), key=lambda kv: BODIES.get(kv[0], ("#fff", 0))[1]):
        colour, order = BODIES.get(name, ("#ffffff", 0))
        if len(tris):
            ax.add_collection(PolyCollection(tris, facecolors=colour, edgecolors=colour, linewidths=0.2,
                                             zorder=1 + order))
        if len(segs):
            ax.add_collection(LineCollection(segs, colors="0.25", linewidths=0.45, zorder=10))
    if labels:
        for label, (x, z) in TEXT:
            ax.text(x, z, label, fontsize=6.5, color="0.3", ha="center", va="center", zorder=11)
    ax.plot([13.5, 18.5], [-44.0, -44.0], color="k", lw=1.6, zorder=20)       # 5 mm scale bar
    ax.text(16.0, -43.2, "5 mm", fontsize=7, ha="center", va="bottom", zorder=20)
    ax.set_xlim(-19.6, xmax)
    ax.set_ylim(-46.2, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    return fig, ax


def tag(ax, xy, text, colour, off):
    ax.annotate(text, xy, xytext=off, textcoords="offset points", fontsize=7.5, color=colour,
                fontweight="bold", ha="left" if off[0] > 0 else "right", va="center", zorder=21,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def nerve_reference():
    """Optic-nerve sensor of every human eye against the nerve stump and the posterior pole of
    its own eye, from the CT results (eye frame, mm). Written to NERVE_REF and returned."""
    rows = []
    for d in sorted(CT_OUT.glob("H[0-9]*")):
        j = json.loads((d / (d.name + "_results.json")).read_text(encoding="utf-8"))
        an = j["anatomy"]
        o, ax = np.array(an["frame"]["origin_mm"]), np.array(an["frame"]["axes"])
        eye = lambda p: (np.array(p) - o) @ ax.T
        q = [k for k in j["thermistors"]["picks"] if k["sensor"] == "optic"][0]
        sp = np.array(q["eye_xyz_mm"])
        row = dict(eye=d.name, axis_source=an["axis"]["source"],
                   depth_below_surface_mm=q["depth_below_surface_mm"],
                   sensor_to_posterior_pole_mm=round(float(np.linalg.norm(sp - eye(an["posterior_pole_mm"]))), 3),
                   nerve_detected=False, nerve_polar_deg=np.nan, nerve_used=False,
                   sensor_along_nerve_mm=np.nan, sensor_off_nerve_axis_mm=np.nan)
        n = an.get("nerve") or {}
        if n.get("attach_mm") and n.get("centroid_mm"):
            at, c = eye(n["attach_mm"]), eye(n["centroid_mm"])
            u = (c - at) / np.linalg.norm(c - at)
            along = float((sp - at) @ u)
            pol = float(np.degrees(np.arccos(at[2] / np.linalg.norm(at))))
            row.update(nerve_detected=True, nerve_polar_deg=round(pol, 1),
                       nerve_used=bool(pol >= NERVE_MIN_POLAR_DEG), sensor_along_nerve_mm=round(along, 3),
                       sensor_off_nerve_axis_mm=round(float(np.linalg.norm(sp - at - along * u)), 3))
        rows.append(row)
    ref = pd.DataFrame(rows)
    NERVE_REF.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(NERVE_REF, index=False)
    return ref


class Globe:
    """Outer surface of the model sclera in the section, on the side x > 0, as a curve from
    the front of the sclera to the posterior pole (the nerve sheath is not part of it)."""

    def __init__(self, sec):
        seg = sec["Sclera"][1]
        pts = np.concatenate([seg[:, 0] + (seg[:, 1] - seg[:, 0]) * f for f in np.linspace(0, 1, 12)])
        pts = pts[pts[:, 0] > 0]
        th = np.degrees(np.arctan2(pts[:, 0], pts[:, 1] - CENTRE_Z))
        r = np.hypot(pts[:, 0], pts[:, 1] - CENTRE_Z)
        edges = np.arange(np.floor(th.min()) + 1, 172.5, 1.0)
        k = np.digitize(th, edges)
        used = [i for i in range(1, len(edges)) if (k == i).any()]
        tb = np.array([(edges[i - 1] + edges[i]) / 2 for i in used])
        rb = np.array([r[k == i].max() for i in used])
        t = np.arange(tb.min(), 180.0001, 0.25)
        rr = np.interp(t, np.append(tb, 180.0), np.append(rb, CENTRE_Z - Z_POSTERIOR))
        self.xz = np.column_stack([rr * np.sin(np.radians(t)), CENTRE_Z + rr * np.cos(np.radians(t))])
        tan = np.gradient(self.xz, axis=0)
        nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        inward = np.array([0.0, CENTRE_Z]) - self.xz
        self.n_in = nrm * np.sign((nrm * inward).sum(1))[:, None]
        self.chord = np.hypot(self.xz[:, 0], self.xz[:, 1] - Z_POSTERIOR)

    def _at(self, i, depth, side):
        p = self.xz[i] + depth * self.n_in[i]
        return side * p[0], p[1]

    def at_axial(self, frac, depth, side=1):
        """Point at a fraction of the axial length from the anterior pole, `depth` mm inside."""
        return self._at(int(np.argmin(np.abs(self.xz[:, 1] - (Z_APEX - frac * AXIAL)))), depth, side)

    def from_pole(self, chord, depth, side=1):
        """Point at a straight-line distance from the posterior pole, `depth` mm inside."""
        return self._at(int(np.argmin(np.abs(self.chord - chord))), depth, side)

    def nerve(self):
        """Where the axis of the model nerve (through its two probes) leaves the globe, the
        unit vector along the nerve away from the globe, and the unit vector across it,
        away from the optical axis."""
        a, b = np.array(PROBES[5][1]), np.array(PROBES[6][1])
        u = (b - a) / np.linalg.norm(b - a)
        w = np.array([u[1], -u[0]])
        w = w if w[0] < 0 else -w
        m = (self.xz * [-1, 1])[self.chord < 8]                  # the globe on the side of the nerve
        return m[int(np.argmin(np.abs((m - a) @ w)))], u, w


def sensors_on_model(globe, ref):
    """Human sensors placed on the model section: eye, sensor, x, z, referenced, excluded (by
    the spatial screen), and the coordinates used (fraction of the axial length, depth)."""
    sc = pd.read_csv(SCREEN)
    sc = sc[(sc["species"] == "Human") & (sc["flagged"].astype(str).str.lower() == "true")]
    flagged = {(e.replace("E", "H"), CHANNEL[c]) for e, c in zip(sc["marker_eye"], sc["channel"])}
    d = pd.read_csv(PICKS)
    d = d[d["group"] == "human"].copy()
    d["axial_fraction"] = d["axial_dist_from_anterior_pole_mm"] / d["axial_length_mm"]
    at, u, w = globe.nerve()
    ref = ref.set_index("eye")
    out = []
    for r in d.itertuples():
        known = True
        if r.sensor != "optic":
            x, z = globe.at_axial(r.axial_fraction, r.depth_below_surface_mm)
        elif ref.loc[r.eye, "nerve_used"]:
            x, z = at + ref.loc[r.eye, "sensor_along_nerve_mm"] * u + ref.loc[r.eye, "sensor_off_nerve_axis_mm"] * w
        else:
            x, z = globe.from_pole(ref.loc[r.eye, "sensor_to_posterior_pole_mm"], r.depth_below_surface_mm, side=-1)
            known = False
        out.append(dict(eye=r.eye, sensor=r.sensor, x=x, z=z, referenced=known,
                        excluded=(r.eye, r.sensor) in flagged,
                        axial_fraction=r.axial_fraction, depth=r.depth_below_surface_mm))
    return pd.DataFrame(out)


def mean_position(globe, ref, g):
    """Mean of one site in the coordinates it was placed by, on the model, without the
    sensors excluded by the spatial screen."""
    g = g[~g["excluded"]]
    if g["sensor"].iloc[0] != "optic":
        return globe.at_axial(g["axial_fraction"].mean(), g["depth"].mean())
    at, u, w = globe.nerve()
    k = ref[ref["nerve_used"]]
    return tuple(at + k["sensor_along_nerve_mm"].mean() * u + k["sensor_off_nerve_axis_mm"].mean() * w)


def draw_probes(ax, labels=True):
    for _, (x, z), label, colour, off in PROBES:
        ax.plot(x, z, "o", ms=6.5, mfc=colour, mec="white", mew=0.9, zorder=20)
        if labels:
            tag(ax, (x, z), label, colour, off)


def draw_sensors(ax, s, globe, ref, labels=True):
    means = {}
    for name, g in s.groupby("sensor"):
        label, colour, _, off = SENSORS[name]
        x, g = g[g["excluded"]], g[~g["excluded"]]
        ax.plot(x["x"], x["z"], "X", ms=7.5, mfc=colour, mec="black", mew=0.7, zorder=21)
        k, u = g[g["referenced"]], g[~g["referenced"]]
        ax.plot(k["x"], k["z"], "o", ms=3.4, mfc=colour, mec="none", alpha=0.6, zorder=18)
        ax.plot(u["x"], u["z"], "o", ms=3.6, mfc="none", mec=colour, mew=0.9, zorder=18)
        mx, mz = mean_position(globe, ref, g)
        means[name] = (mx, mz)
        ax.plot(mx, mz, "D", ms=6.5, mfc=colour, mec="black", mew=0.8, zorder=22)
        if labels:
            tag(ax, (mx, mz), label, colour, off)
    return means


SENSOR_LEGEND = [
    Line2D([], [], marker="o", ls="", ms=4, mfc="0.4", mec="none", alpha=0.6, label="one sensor"),
    Line2D([], [], marker="o", ls="", ms=4, mfc="none", mec="0.4", mew=0.9,
           label="optic-nerve sensor, nerve not\nidentified in the scan"),
    Line2D([], [], marker="X", ls="", ms=7, mfc="0.4", mec="black", mew=0.7,
           label="sensor excluded by the spatial\nscreen (not in the mean)"),
    Line2D([], [], marker="D", ls="", ms=6, mfc="0.4", mec="black", mew=0.8,
           label="mean position of the\nincluded sensors"),
]


def main():
    for snap, origin in ((SNAP, ORIGIN), (PICKS, PICKS_ORIGIN), (SCREEN, SCREEN_ORIGIN)):
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, snap) if snap is not SNAP or not snap.exists() else None
    sec = section()
    globe = Globe(sec)
    ref = nerve_reference()
    s = sensors_on_model(globe, ref)
    notes = ("Section of the model geometry at y = 0 (Gmsh / OpenCASCADE). Probes: Domain Point Probes of the "
             "v1 parametric-sweep model; SCR probe (0, 10, 0) mm drawn turned into the plane. Sensors: human "
             "eyes H1-H9, placed on the model globe by fraction of the axial length and measured depth below "
             "the surface; optic-nerve sensor relative to the nerve stump where identified (H3, H4, H6), "
             "otherwise by its distance from the posterior pole.")
    origins = {SNAP.name: str(ORIGIN.relative_to(BOX)).replace("\\", "/"),
               PICKS.name: str(PICKS_ORIGIN.relative_to(BOX)).replace("\\", "/"),
               SCREEN.name: str(SCREEN_ORIGIN.relative_to(BOX)).replace("\\", "/"),
               NERVE_REF.name: "derived by this script from "
                               + str(CT_OUT.relative_to(BOX)).replace("\\", "/") + "/H*/H*_results.json"}

    # 1. the probes of the model
    fig, ax = base(sec)
    draw_probes(ax)
    out1 = FIG / "probe-locations.png"
    fig.savefig(out1, dpi=400)
    plt.close(fig)

    # 2. the sensors implanted in the human donor eyes, with the probes of the model
    fig, ax = base(sec, labels=False, xmax=23.5)
    means2 = draw_sensors(ax, s, globe, ref)
    draw_probes(ax, labels=False)
    for name, i in (("anterior", 1), ("equatorial", 2), ("vitreous", 3), ("posterior", 4), ("optic", 5)):
        ax.plot([PROBES[i][1][0], means2[name][0]], [PROBES[i][1][1], means2[name][1]],
                color=PROBES[i][3], lw=1.1, zorder=17)
    tag(ax, PROBES[0][1], "SCR (model)", PROBES[0][3], PROBES[0][4])
    tag(ax, PROBES[6][1], "Distal optic nerve (model)", PROBES[6][3], PROBES[6][4])
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=6, mfc="0.4", mec="white", mew=0.9,
                              label="model probe")] + SENSOR_LEGEND,
              loc="upper center", bbox_to_anchor=(0.0, 14.6), bbox_transform=ax.transData, ncol=1,
              fontsize=7, frameon=False)
    out2 = FIG / "sensor-locations.png"
    fig.savefig(out2, dpi=400)
    plt.close(fig)

    # 3. comparison
    fig, ax = base(sec, labels=False, xmax=30.0)
    means = draw_sensors(ax, s, globe, ref, labels=False)
    draw_probes(ax, labels=False)
    pair = {"anterior": 1, "equatorial": 2, "vitreous": 3, "posterior": 4, "optic": 5}
    for name, (mx, mz) in means.items():
        _, (px, pz), label, colour, off = PROBES[pair[name]]
        ax.plot([px, mx], [pz, mz], color=colour, lw=1.1, zorder=17)
        tag(ax, (px, pz), "%s (%.1f mm)" % (label, np.hypot(px - mx, pz - mz)), colour, off)
    for i in (0, 6):
        _, (px, pz), label, colour, off = PROBES[i]
        tag(ax, (px, pz), label + " (model only)" if i == 6 else label, colour, off)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=6, mfc="0.4", mec="white", mew=0.9,
                              label="model probe")] + SENSOR_LEGEND,
              loc="upper center", bbox_to_anchor=(0.0, 14.6), bbox_transform=ax.transData, ncol=1,
              fontsize=7, frameon=False)
    out3 = FIG / "probe-vs-sensor-locations.png"
    fig.savefig(out3, dpi=400)
    plt.close(fig)

    record(__file__, [out1], [SNAP], origins=origins, notes=notes)
    record(__file__, [out2, out3], [SNAP, PICKS, NERVE_REF, SCREEN], origins=origins, notes=notes)
    for o in (out1, out2, out3):
        print("written:", o.name)
    # numbers for the text
    print("excluded by the spatial screen:", s[s["excluded"]][["eye", "sensor", "axial_fraction", "depth"]].values.tolist())
    print(ref.to_string(index=False))
    for i, name in ((1, "anterior"), (2, "equatorial"), (3, "vitreous"), (4, "posterior"), (5, "optic")):
        _, (px, pz), label, _, _ = PROBES[i]
        g = s[(s["sensor"] == name) & ~s["excluded"]]
        mx, mz = means[name]
        print("%-22s n = %d, model probe axial fraction %.2f | sensors %.2f (%.2f-%.2f), depth %.2f mm (%.2f to %.2f) | "
              "probe to mean of the sensors %.1f mm" % (
                  label, len(g), (Z_APEX - pz) / AXIAL, g["axial_fraction"].mean(), g["axial_fraction"].min(),
                  g["axial_fraction"].max(), g["depth"].mean(), g["depth"].min(), g["depth"].max(),
                  np.hypot(px - mx, pz - mz)))
    at, u, w = globe.nerve()
    k = ref[ref["nerve_used"]]
    print("model nerve leaves the globe at (%.2f, %.2f) mm; nerve-head probe %.2f mm inside it along the nerve; "
          "sensors (n = %d) %.2f mm outside (%.2f to %.2f), %.2f mm from the axis of the nerve (%.2f to %.2f)" % (
              at[0], at[1], (at - np.array(PROBES[5][1])) @ u, len(k), k["sensor_along_nerve_mm"].mean(),
              k["sensor_along_nerve_mm"].min(), k["sensor_along_nerve_mm"].max(),
              k["sensor_off_nerve_axis_mm"].mean(), k["sensor_off_nerve_axis_mm"].min(),
              k["sensor_off_nerve_axis_mm"].max()))


if __name__ == "__main__":
    main()
