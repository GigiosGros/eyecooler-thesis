"""Chapter 2 Methods: flowchart of the experimental protocol (protocol.png).

Redrawn from Methods-manuscript Fig. 7 (a 1010 x 254 pixel image, too coarse for print):
the same ten steps, in the same order and layout, drawn here as a high-resolution figure.
No data input; the step names are the ones in the manuscript figure.

Run:  python code/figures/aim1_protocol.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from provenance import THESIS, record

OUT = THESIS / "figures" / "aim1" / "methods"
DST = OUT / "protocol.png"
ROWS = [["Cleaning", "MRI", "Preparation\nof oil bath", "Implantation", "Calibration"],
        ["Positioning", "Intraocular\npressure\nadjustment", "Equilibration", "Cooling\nprotocol",
         "Computed\ntomography"]]
FACE, EDGE, LINE = "#eaf0fa", "#8fa3cf", "#222222"
W, H, GAP = 1.55, 0.86, 0.72          # box width, height, horizontal gap (inches of drawing space)
Y = [1.55, 0.0]                        # row centres


def main():
    pitch = W + GAP
    width = 5 * W + 4 * GAP
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * (Y[0] + H + 0.1) / (width + 0.1)))
    ax.set_xlim(-0.05, width + 0.05)
    ax.set_ylim(-H / 2 - 0.05, Y[0] + H / 2 + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    arrow = dict(arrowstyle="-|>", mutation_scale=9, lw=1.0, color=LINE, shrinkA=0, shrinkB=0)
    for r, names in enumerate(ROWS):
        for c, name in enumerate(names):
            x0 = c * pitch
            ax.add_patch(FancyBboxPatch((x0, Y[r] - H / 2), W, H,
                                        boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=FACE, ec=EDGE, lw=1.0))
            ax.text(x0 + W / 2, Y[r], name, ha="center", va="center", fontsize=8,
                    linespacing=1.15)
            if c < 4:
                ax.add_patch(FancyArrowPatch((x0 + W, Y[r]), (x0 + pitch, Y[r]), **arrow))
    # from the last box of the first row to the first box of the second row
    x_last, x_first, y_mid = 4 * pitch + W / 2, W / 2, (Y[0] + Y[1]) / 2
    ax.plot([x_last, x_last, x_first], [Y[0] - H / 2, y_mid, y_mid], color=LINE, lw=1.0,
            solid_joinstyle="round", solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x_first, y_mid), (x_first, Y[1] + H / 2), **arrow))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(DST, dpi=600)
    plt.close(fig)
    record(__file__, [DST], [], manifest=OUT.parent / "MANIFEST.json",
           notes="Redrawn flowchart; replaces protocol.jpg (Methods manuscript Fig. 7, "
                 "1010 x 254 pixels). Same ten steps and layout; no data input.")
    print("written:", DST)


if __name__ == "__main__":
    main()
