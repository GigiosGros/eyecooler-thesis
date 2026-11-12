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


# 2026-11-11: drawn at the printed width (6.0 in = textwidth; included at width=\textwidth) so that the text
# prints at 8 pt (titles) and 7 pt (body); it was 14.83 in wide (4.5 and 5.9 pt in print). Data units are printed
# points. The stage texts above are unchanged; the lines too wide for a box at 7 pt are rewrapped below (same words,
# checked in draw()), and draw() stops if any line is still wider than the box.
W_PT = 432.0                       # 6.0 in
T_TITLE, T_BODY = 8.0, 7.0
PAD, GAP = 4.0, 8.0                # inner padding of a box; horizontal gap between boxes (holds the arrow)
REWRAP = {
    "2  Calibration": "Human: line through the\nthree plateaus of each\nchannel\nPorcine: fixed linear\ntransfer",
    "3  Denoising": "Median filter, 51 samples\nreflection at cooling onset\nzero-phase Butterworth\n0.01 Hz",
    "4  Quality control": "Hardware gate, fit gate\n(R² ≥ 0.85), micro-CT\nspatial screen, manual\nexclusions",
    "6  Statistics": "Per site: paired or\nWelch t, Holm, Hedges g,\ndonor repeat\n90 % CI of each difference",
    "7  Outputs|a": "results/ folders 00–08\ntables, figures, run log\nSHA-256 of every\nrecording file",
    "1  COMSOL exports": "SCR: mesh, parametric,\nmaterial, silicone, reheat,\nenergy files\nWater block: stationary,\nsweep",
    "2  Run registry": "523 runs from 32 files\nunits °C and min,\n35-min cut\nv3 eye-flux sign restored",
    "3  Verification": "Change to the next finer\nmesh (°C), SCR and\nwater-block energy\nclosure",
    "5  Effectiveness": "Superposition fit:\nε per site\nring temperature\nfor targets\ndesign differences\nvs mesh change",
    "6  Ex-vivo comparison": "ε and t90 vs human\nex-vivo results\n(Al, ceramic, reduced\nbulk)",
    "7  Outputs|m": "results/ folders 00–11\nworkbook, report, run log\nSHA-256 of each\nCOMSOL export",
}
TITLE_WRAP = {"6  Ex-vivo comparison": ["6  Ex-vivo", "comparison"]}   # second line aligned after the number


def _width(fig, text, size, bold=False):
    t = fig.text(0, 0, text, fontsize=size, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    t.remove()
    return w


def _words(s):
    return s.replace("-\n", "-").split()


def draw(stages, fill, edge, path):
    w = (W_PT - 2 - 3 * GAP) / 4                                       # four boxes on the top row
    fig = plt.figure(figsize=(W_PT / 72, 3), dpi=300)
    inner = w - 2 * PAD
    tag = "|a" if stages is ANALYSIS else "|m"
    texts = []
    for title, body, coloured in stages:
        new = REWRAP.get(title + tag, REWRAP.get(title, body))
        assert _words(new) == _words(body), (title, new)                # same words as the stage text
        t_lines = TITLE_WRAP.get(title, [title])
        b_lines = new.split("\n")
        num_w = _width(fig, title.split("  ")[0] + "  ", T_TITLE, True)
        widest = max([_width(fig, l, T_BODY) for l in b_lines]
                     + [_width(fig, l, T_TITLE, True) + (num_w if k else 0) for k, l in enumerate(t_lines)])
        assert widest <= inner, (title, widest, inner)
        texts.append((t_lines, b_lines, coloured, num_w))
    lh_t, lh_b = T_TITLE * 1.2, T_BODY * 1.25
    h = 2 * PAD + max(len(t) * lh_t + 3 + len(b) * lh_b for t, b, _, _ in texts)
    elbow = 16.0                                                         # room for the elbow between the rows
    H = 2 * h + elbow + 2
    fig.set_size_inches(W_PT / 72, H / 72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT); ax.set_ylim(0, H); ax.axis("off")
    step = w + GAP
    top = [(1 + i * step, h + elbow + 1) for i in range(4)]
    x0 = (W_PT - (3 * w + 2 * GAP)) / 2                                  # bottom row centred, as before
    bottom = [(x0 + i * step, 1) for i in range(3)]
    for (x, y), (t_lines, b_lines, coloured, num_w) in zip(top + bottom, texts):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", lw=0.7,
                                    ec=edge, fc=fill if coloured else "#F1F1F1"))
        for k, line in enumerate(t_lines):
            ax.text(x + PAD + (num_w if k else 0), y + h - PAD - k * lh_t, line, fontsize=T_TITLE,
                    fontweight="bold", va="top")
        ax.text(x + PAD, y + h - PAD - len(t_lines) * lh_t - 3, "\n".join(b_lines), fontsize=T_BODY, va="top",
                linespacing=1.25)
    arrow = dict(arrowstyle="-|>", color=edge, lw=0.75, mutation_scale=6, shrinkA=0, shrinkB=0)
    for row in (top, bottom):
        for (x, y) in row[:-1]:
            ax.annotate("", (x + step - 0.5, y + h / 2), (x + w + 0.5, y + h / 2), arrowprops=arrow)
    xl, yl = top[-1][0] + w / 2, h + 1 + elbow / 2                      # elbow from box 4 to box 5
    ax.plot([xl, xl, bottom[0][0] + w / 2], [top[-1][1], yl, yl], color=edge, lw=0.75)
    ax.annotate("", (bottom[0][0] + w / 2, bottom[0][1] + h + 0.5), (bottom[0][0] + w / 2, yl), arrowprops=arrow)
    fig.savefig(path, dpi=300, facecolor="white")
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
