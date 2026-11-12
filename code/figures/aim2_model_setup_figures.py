"""Chapter 3 figures of the computational models themselves: domains, boundary conditions and meshes.

The pictures are drawn outside this repository, directly from the COMSOL model files, by
  03 - Experiments & Data/Analysis Code/scr-heat-replica/dev/make_model_figures_comsol.py --thesis
(repository GigiosGros/scr-heat-replica; nothing is solved there: the script reads the geometry, the three
meshes, the domains and the boundary selections of "Ring Comparison.mph", its two mesh-study files and the
three water-block files, and draws the plane of symmetry and sections).  Its output folder holds a
MANIFEST.json of its own.

This script copies the six pictures unchanged: first as snapshots into figures/aim2/sources/model_setup/,
then under their thesis names into figures/aim2/, and records both in figures/aim2/MANIFEST.json.

Run:  python code/figures/aim2_model_setup_figures.py            (from the snapshots)
      python code/figures/aim2_model_setup_figures.py --refresh  (re-copy the snapshots first)
"""
import shutil
import sys

from provenance import FIG, SOURCES, record

BOX = FIG.parents[4]  # .../Eye Cooler
ORIGIN = BOX / "03 - Experiments & Data" / "Analysis Code" / "scr-heat-replica" / "results" / "figures_2026-10-12" / "model_setup_comsol_thesis"
SNAP = SOURCES / "model_setup"
NAMES = {"eye-geometry.png": "model-domains.png",
         "eye-boundaries.png": "model-boundary-conditions.png",
         "eye-mesh.png": "model-meshes.png",
         "hx-geometry.png": "heat-exchanger-sections.png",
         "hx-boundaries.png": "heat-exchanger-boundary-conditions.png",
         "hx-mesh.png": "heat-exchanger-meshes.png"}

SNAP.mkdir(parents=True, exist_ok=True)
if "--refresh" in sys.argv or not all((SNAP / n).exists() for n in NAMES):
    for n in list(NAMES) + ["MANIFEST.json", "sizes.csv"]:
        shutil.copy2(ORIGIN / n, SNAP / n)
for src, dst in NAMES.items():
    if src == "hx-boundaries.png":
        continue                         # 2026-11-11: Figure D.2 is now the Python set-up picture (below)
    shutil.copy2(SNAP / src, FIG / dst)
    record(__file__, [FIG / dst], [SNAP / src, SNAP / "MANIFEST.json"],
           origins={src: "scr-heat-replica/results/figures_2026-10-12/model_setup_comsol_thesis/" + src,
                    "MANIFEST.json": "the manifest written by scr-heat-replica/dev/make_model_figures_comsol.py"},
           notes="copied unchanged; drawn from the COMSOL model files by scr-heat-replica/dev/make_model_figures_comsol.py --thesis")
    print("wrote", dst)

# 2026-10-31: the domains and boundary conditions in the thesis are those of the corrected Python runs (no air domain,
# no module layer), drawn on the meshes solved, by scr-heat-replica/dev/make_model_figures_solved.py.
ORIGIN_PY = ORIGIN.parents[1] / "figures_2026-10-31" / "model_setup_solved"
SNAP_PY = SOURCES / "model_setup_python"
SNAP_PY.mkdir(parents=True, exist_ok=True)
for src, dst in (("eye-geometry.png", "model-domains.png"), ("eye-boundaries.png", "model-boundary-conditions.png"),
                 ("eye-mesh.png", "model-meshes.png")):
    if "--refresh" in sys.argv or not (SNAP_PY / src).exists():
        shutil.copy2(ORIGIN_PY / src, SNAP_PY / src)
    shutil.copy2(SNAP_PY / src, FIG / dst)
    record(__file__, [FIG / dst], [SNAP_PY / src],
           origins={src: "scr-heat-replica/results/figures_2026-10-31/model_setup_solved/" + src},
           notes="copied unchanged; corrected Python set-up, drawn by scr-heat-replica/dev/make_model_figures_solved.py")
    print("wrote", dst)

# 2026-11-11: the boundary conditions of the heat exchangers as the Python runs of Chapter 3 set them
# (sweep W13: heat on the annulus of radius 7 to 13 mm, the rest of the module plane an outer wall), drawn by
# scr-heat-replica/dev/make_hx_boundaries_python.py (6.0 in wide, legend 7.5 pt). Replaces the COMSOL picture
# hx-boundaries.png, which showed COMSOL's heated faces.
ORIGIN_HX = ORIGIN.parents[1] / "figures_2026-11-11" / "hx_boundaries_python"
for n in ("hx-boundaries.png", "MANIFEST.json"):
    if "--refresh" in sys.argv or not (SNAP_PY / ("hx_" + n if n == "MANIFEST.json" else n)).exists():
        shutil.copy2(ORIGIN_HX / n, SNAP_PY / ("hx_" + n if n == "MANIFEST.json" else n))
shutil.copy2(SNAP_PY / "hx-boundaries.png", FIG / "heat-exchanger-boundary-conditions.png")
record(__file__, [FIG / "heat-exchanger-boundary-conditions.png"], [SNAP_PY / "hx-boundaries.png", SNAP_PY / "hx_MANIFEST.json"],
       origins={"hx-boundaries.png": "scr-heat-replica/results/figures_2026-11-11/hx_boundaries_python/hx-boundaries.png",
                "hx_MANIFEST.json": "scr-heat-replica/results/figures_2026-11-11/hx_boundaries_python/MANIFEST.json"},
       notes="copied unchanged; Python set-up of the heat-exchanger runs (heated annulus 7-13 mm, module_plane air, mesh normal), "
             "drawn by scr-heat-replica/dev/make_hx_boundaries_python.py. Superseded COMSOL version kept in "
             "figures/aim2/_superseded_2026-11-11/heat-exchanger-boundary-conditions.png")
print("wrote heat-exchanger-boundary-conditions.png")
