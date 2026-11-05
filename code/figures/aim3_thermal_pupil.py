"""Chapter 4, Figure fig:aim3-thermal-pupil: thermal images and the RETeval pupil camera view.

Added 2026-10-30.
  A  animal 6, session 1: thermal image of the device on the eye (20261027_160724293_iOS.jpg)
  B  animal 4, session 3: thermal image of the device on the eye
  C  animal 5, session 1: RETeval pupil camera frame through the device
The thermal originals are phone-camera exports with the read-outs burned in; capture times are
not in the files. Originals are copied unchanged to figures/aim3/sources/thermal_pupil/.

Output: figures/aim3/thermal-pupil.png
Run:  python code/figures/aim3_thermal_pupil.py
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from provenance import THESIS, record
from aim3_invivo_figures import STUDY

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "thermal_pupil"
OUT = FIG3 / "thermal-pupil.png"
CHOSEN = [
    ("A", STUDY / "Pig 6" / "Session 1 - 30min" / "09 - Thermal Imaging"
     / "Pig 6 S1 thermal (export 2026-10-27, capture times not in the files)" / "20261027_160724293_iOS.jpg"),
    ("B", STUDY / "Pig 4" / "Session 3 - 2hr" / "09 - Thermal Imaging"
     / "Pig4_AOT00000308-261013_S3-2hr_Thermal_110452-658_LM_261020.jpg"),
    ("C", STUDY / "Pig 5" / "Reference" / "Pig 5 ERG Original Files" / "RETeval exports (pupil video, images)"
     / "261026-1_160101_261026081306-RightEye-1.png"),
]
plt.rcParams["font.family"] = "Arial"


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    srcs, imgs = [], []
    for _, p in CHOSEN:
        dst = SRC / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        srcs.append(dst)
        imgs.append(np.asarray(Image.open(dst).convert("RGB")))
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 3.0), dpi=400,
                            gridspec_kw={"wspace": 0.04, "width_ratios": [0.75, 0.75, 1.57]})
    for ax, (letter, _), img in zip(axs, CHOSEN, imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.0, 1.01, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="bottom", ha="left")
    fig.savefig(OUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    record(__file__, [OUT], srcs, origins={s.name: str(p) for s, (_, p) in zip(srcs, CHOSEN)},
           notes="Thermal A/B and pupil frame C chosen 2026-10-30; thermal capture times unknown",
           manifest=FIG3 / "MANIFEST.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
