"""Appendix: necropsy photographs of animal 3 (died in recovery from session 3, 2026-10-19).

Chosen on 2026-11-08 from the 11 photographs in Live Pigs Safety Study/Pig 3/Reference/Necropsy 261019
(cause of death): A = 20261019_201821303_iOS.jpg (neck opened, larynx and trachea), B = 20261019_201928014_iOS.jpg
(heart, opened, with a scalpel for scale). The originals are copied unchanged to figures/aim3/sources/necropsy/
and only rotated by their EXIF orientation here.

Output: figures/aim3/necropsy-animal3.png
Run:  python code/figures/aim3_necropsy_figure.py
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from provenance import THESIS, record
from aim3_invivo_figures import STUDY

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "necropsy"
OUT = FIG3 / "necropsy-animal3.png"
PICS = STUDY / "Pig 3" / "Reference" / "Necropsy 261019 (cause of death)"
CHOSEN = [("A", "20261019_201821303_iOS.jpg"), ("B", "20261019_201928014_iOS.jpg")]
plt.rcParams["font.family"] = "Arial"


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    imgs, srcs = [], []
    for _, name in CHOSEN:
        dst = SRC / name
        if not dst.exists():
            shutil.copy2(PICS / name, dst)
        srcs.append(dst)
        im = ImageOps.exif_transpose(Image.open(dst)).convert("RGB")
        im.thumbnail((1600, 1600))
        imgs.append(np.asarray(im))
    h = max(i.shape[0] / i.shape[1] for i in imgs)
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 6.0 / 2 * h + 0.3), dpi=300, gridspec_kw={"wspace": 0.04})
    for ax, (letter, _), img in zip(axs, CHOSEN, imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.0, 1.01, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    record(__file__, [OUT], srcs, origins={s.name: str(PICS / s.name) for s in srcs},
           notes="photographs chosen 2026-11-08 (necropsy images of animal 3 in the appendix)",
           manifest=FIG3 / "MANIFEST.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
