"""Chapter 4: photograph of the 3D-printed guide used to trim the fixed globes for histology.

Chosen on 2026-10-29 from the pictures of 2026-10-23 (Live Pigs Safety Study/Admin &
Reference/Pictures): A = 20261023_201739788_iOS.heic (front), B = 20261023_201754021_iOS.heic
(side). The originals are copied unchanged to figures/aim3/sources/histology_guide/ and
converted here (EXIF orientation applied, cropped to the guide).

Output: figures/aim3/histology-guide.png
Run:  python code/figures/aim3_histology_guide.py
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from provenance import THESIS, record
from aim3_invivo_figures import STUDY

pillow_heif.register_heif_opener()
FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "histology_guide"
OUT = FIG3 / "histology-guide.png"
PICS = STUDY / "Admin & Reference" / "Pictures"
CHOSEN = [("A", "20261023_201739788_iOS.heic"), ("B", "20261023_201754021_iOS.heic")]
# crop to the guide itself, fractions of each photograph (left, right, top, bottom); 2026-10-31
CROP = {"A": (0.26, 0.78, 0.33, 0.70), "B": (0.33, 0.84, 0.30, 0.71)}
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
        l, r, u, b = CROP[_]
        imgs.append(np.asarray(im.crop((int(l * im.width), int(u * im.height), int(r * im.width), int(b * im.height)))))
    h = max(i.shape[0] / i.shape[1] for i in imgs)
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 5.0 / 2 * h + 0.3), dpi=400,
                            gridspec_kw={"wspace": 0.04})
    for ax, (letter, _), img in zip(axs, CHOSEN, imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.0, 1.01, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")
    fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    record(__file__, [OUT], srcs, origins={s.name: str(PICS / s.name) for s in srcs},
           notes="photographs chosen 2026-10-29 (pictures 21 and 22 of the 10-09..10-26 sheet)",
           manifest=FIG3 / "MANIFEST.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
