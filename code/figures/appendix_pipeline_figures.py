"""Appendix flow diagrams of the two analysis packages (analysis_pipeline.png, model_pipeline.png).

Drawn here from the stage texts below, which state what the packages do as configured:
  thesis-analysis        config/config.toml  (calibration, qc, statistics, uncertainty sections)
  thesis-model-analysis  config/config.toml  (corrections, superposition sections)
The SHA-256 of both config files is recorded in figures/appendix/MANIFEST.json, so a change of
method that is not reflected in the text below shows up as a changed input hash.

Run:  python code/figures/appendix_pipeline_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from provenance import THESIS, record  # noqa: E402

BOX = THESIS.parents[2]
CODE = BOX / "03 - Experiments & Data" / "Analysis Code"
OUT = THESIS / "figures" / "appendix"
SRC = OUT / "sources"

ANALYSIS = [
    ("1  Raw recordings", "Human and porcine .xlsx\ncalibration files\nmicro-CT sensor markers", False),
    ("2  Calibration", "Human: line through the three\nplateaus of each channel\nPorcine: fixed linear transfer", True),
    ("3  Denoising", "Median filter, 51 samples\nreflection at cooling onset\nzero-phase Butterworth 0.01 Hz", True),
    ("4  Quality control", "Hardware gate, fit gate\n(R² ≥ 0.85), micro-CT spatial\nscreen, manual exclusions", True),
    ("5  Endpoints", "T_initial, T_final, ΔT\nt50, t90, plateau SD", True),
    ("6  Statistics", "Per site: paired or Welch t,\nHolm, Hedges g, donor repeat\n90 % CI of each difference", True),
    ("7  Outputs", "results/ folders 00–08\ntables, figures, run log\nSHA-256 of every recording file", False),
]
MODEL = [
    ("1  COMSOL exports", "SCR: mesh, parametric, material,\nsilicone, reheat, energy files\nWater block: stationary, sweep", False),
    ("2  Run registry", "523 runs from 32 files\nunits °C and min, 35-min cut\nv3 eye-flux sign restored", True),
    # P88 (2026-10-19): the text reports the change to the next finer mesh, not the GCI
    ("3  Verification", "Change to the next finer\nmesh (°C), SCR and water-\nblock energy closure", True),
    ("4  Metrics", "T_final, ΔT, t50, t90, τ\nenergy split, R_th, R_WB\nReynolds numbers", True),
    ("5  Effectiveness", "Superposition fit: ε per site\nring temperature for targets\ndesign differences vs mesh change", True),
    ("6  Ex-vivo comparison", "ε and t90 vs human\nex-vivo results\n(Al, ceramic, reduced bulk)", True),
    ("7  Outputs", "results/ folders 00–11\nworkbook, report, run log\nSHA-256 of each COMSOL export", False),
]


def draw(stages, fill, edge, path):
    fig, ax = plt.subplots(figsize=(14.83, 5.61), dpi=200)
    fig.subplots_adjust(0, 0, 1, 1)                                    # the axes fill the page
    ax.set_xlim(0, 14.83); ax.set_ylim(0, 5.61); ax.axis("off")
    w, h = 3.25, 2.0
    top = [(0.2 + i * 3.61, 3.4) for i in range(4)]
    bottom = [(2.0 + i * 3.61, 0.25) for i in range(3)]
    for (x, y), (title, body, coloured) in zip(top + bottom, stages):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", lw=1.6,
                                    ec=edge, fc=fill if coloured else "#F1F1F1"))
        ax.text(x + 0.18, y + h - 0.36, title, fontsize=14.5, fontweight="bold", va="center")
        ax.text(x + 0.18, y + h - 0.72, body, fontsize=11, va="top", linespacing=1.25)
    arrow = dict(arrowstyle="-|>", color=edge, lw=1.8, mutation_scale=14)
    for (x, y) in top[:-1]:
        ax.annotate("", (x + 3.61 - 0.05, y + h / 2), (x + w + 0.03, y + h / 2), arrowprops=arrow)
    for (x, y) in bottom[:-1]:
        ax.annotate("", (x + 3.61 - 0.05, y + h / 2), (x + w + 0.03, y + h / 2), arrowprops=arrow)
    xl, yl = top[-1][0] + w / 2, 2.82                                  # elbow from box 4 to box 5
    ax.plot([xl, xl, bottom[0][0] + w / 2], [top[-1][1], yl, yl], color=edge, lw=1.8)
    ax.annotate("", (bottom[0][0] + w / 2, bottom[0][1] + h + 0.03), (bottom[0][0] + w / 2, yl), arrowprops=arrow)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    jobs = [("analysis_pipeline.png", ANALYSIS, "#E9F0F8", "#2F435A", "thesis-analysis"),
            ("model_pipeline.png", MODEL, "#E9F2EA", "#2F4F3C", "thesis-model-analysis")]
    for name, stages, fill, edge, pkg in jobs:
        snap = SRC / f"{pkg}__config.toml"
        snap.write_bytes((CODE / pkg / "config" / "config.toml").read_bytes())   # the configuration the text describes
        draw(stages, fill, edge, OUT / name)
        record(__file__, [OUT / name], [snap], manifest=OUT / "MANIFEST.json",
               origins={snap.name: f"03 - Experiments & Data/Analysis Code/{pkg}/config/config.toml"})
        print("wrote", name)


if __name__ == "__main__":
    main()
