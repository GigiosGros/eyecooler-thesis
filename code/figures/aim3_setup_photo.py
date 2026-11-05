"""Chapter 4, Figure fig:aim3-setup: the devices on the eyes of an anesthetized pig.

Chosen on 2026-10-29 from the contact sheet of aim3_setup_candidates.py:
  A  candidate 2 (20260706_163241342_iOS.heic): both devices in place (animal 1, session 1)
  B  candidate 7 (20260706_163721742_iOS.heic): a Jet ERG being recorded through the
     central opening of the device (animal 1, session 1)
  C  candidate 12 (20260824_145241123_iOS.heic): the RETeval screen during a recording
     (animal 2, session 3)
The originals (iPhone HEIC, Live Pigs Safety Study/Admin & Reference/Pictures) are copied
unchanged to figures/aim3/sources/setup/ and converted here (EXIF orientation applied,
no other change).

Output: figures/aim3/setup-photo.png
Run:  python code/figures/aim3_setup_photo.py
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
SRC = FIG3 / "sources" / "setup"
OUT = FIG3 / "setup-photo.png"
PICS = STUDY / "Admin & Reference" / "Pictures"
CHOSEN = [("A", "20260706_163241342_iOS.heic"), ("B", "20260706_163721742_iOS.heic"),
          ("C", "20260824_145241123_iOS.heic")]
plt.rcParams["font.family"] = "Arial"


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    imgs, srcs = [], []
    for _, name in CHOSEN:
        dst = SRC / name
        if not dst.exists():
            shutil.copy2(PICS / name, dst)
        srcs.append(dst)
        imgs.append(np.asarray(ImageOps.exif_transpose(Image.open(dst)).convert("RGB")))
    h = max(i.shape[0] / i.shape[1] for i in imgs)
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 6.3 / 3 * h + 0.3), dpi=400,
                            gridspec_kw={"wspace": 0.04})
    for ax, (letter, _), img in zip(axs, CHOSEN, imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.0, 1.01, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")
    fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    record(__file__, [OUT], srcs,
           origins={s.name: str(PICS / s.name) for s in srcs},
           notes="A = candidate 2, B = candidate 7, C = candidate 12 of setup_candidates.png (chosen 2026-10-29)",
           manifest=FIG3 / "MANIFEST.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
