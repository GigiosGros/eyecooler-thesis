"""Chapter 4: photograph of the animals in their pen at the BRL.

Chosen on 2026-10-29: 20261009_165720002_iOS.heic (Live Pigs Safety Study/Admin &
Reference/Pictures), taken 2026-10-09, the day after animals 3 and 4 arrived. The original is
copied unchanged to figures/aim3/sources/housing/ and converted here (EXIF orientation applied,
no other change).

Output: figures/aim3/housing-photo.png
Run:  python code/figures/aim3_housing_photo.py
"""
import shutil

import pillow_heif
from PIL import Image, ImageOps

from provenance import THESIS, record
from aim3_invivo_figures import STUDY

pillow_heif.register_heif_opener()
FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "housing"
OUT = FIG3 / "housing-photo.png"
PICS = STUDY / "Admin & Reference" / "Pictures"
NAME = "20261009_165720002_iOS.heic"


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    dst = SRC / NAME
    if not dst.exists():
        shutil.copy2(PICS / NAME, dst)
    img = ImageOps.exif_transpose(Image.open(dst)).convert("RGB")
    img.thumbnail((2000, 2000))
    img.save(OUT, dpi=(300, 300), optimize=True)
    record(__file__, [OUT], [dst], origins={NAME: str(PICS / NAME)},
           notes="chosen 2026-10-29 (picture 2 of the 10-09..10-26 sheet)",
           manifest=FIG3 / "MANIFEST.json")
    print("wrote", OUT, img.size)


if __name__ == "__main__":
    main()
