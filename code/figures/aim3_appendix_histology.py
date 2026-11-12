"""Appendix: slide 5 of the H&E sections of the test and control eyes of each animal.

Output (figures/aim3/appendix_histology/):
  Pig<N>_slide5.jpg    test eye (left) and control eye (right), slide 5 of each, as scanned (scale bar
                       of the scanner kept); one figure per animal
  MANIFEST.json        provenance of figures/aim3 (provenance.record)

Inputs (copied and reduced by --refresh into figures/aim3/sources/histology/):
  Pig <N> Test_5.jpg, Pig <N> Control_5.jpg   whole-slide scans from
      Pig N/Session 4 - End of Life/07 - Histology/H&E Staining/{Test Eye,Control Eye}/
      (about 19,000 px a side), longest side reduced to MAX_SIDE pixels. The scans are not cropped,
      rotated or color corrected.

Animal 1 has no slide 5 (its slides are 1, 9 and 10): slide 10 is used, the scan showing
the whole section of each eye (Pig 1 Test_10_200X_FullEye.jpg; Pig 1 Control_10_200X_HalfEye.jpg, whose
"HalfEye" scan holds the whole section). Animals 5 and 6 are added with
--refresh once their sections are scanned.

Run:  python code/figures/aim3_appendix_histology.py [--refresh]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from provenance import THESIS, record

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources" / "histology"
OUT = FIG3 / "appendix_histology"
STUDY = Path(os.environ.get("LIVE_PIG_STUDY", THESIS / "data" / "Live Pigs Safety Study"))
SLIDE = 5
SLIDE_OF = {1: 10}                  # animal -> slide, where slide 5 does not exist
SLIDE_OF_EYE = {(3, "Test"): 10}     # 2026-11-12: slide chosen by me for one eye (animal 3 test eye: slide 10)
SUFFIX_OF = {(1, "Test"): "_200X_FullEye", (1, "Control"): "_200X_HalfEye"}   # file-name suffix of those scans
MAX_SIDE = 3000
Image.MAX_IMAGE_PIXELS = None       # the slide scans are about 350 Mpx


def slide(pig, eye):
    return SLIDE_OF_EYE.get((pig, eye), SLIDE_OF.get(pig, SLIDE))


def name(pig, eye):
    return f"Pig {pig} {eye}_{slide(pig, eye)}{SUFFIX_OF.get((pig, eye), '')}.jpg"


def study_file(pig, eye):
    return (STUDY / f"Pig {pig}" / "Session 4 - End of Life" / "07 - Histology" / "H&E Staining"
            / f"{eye} Eye" / name(pig, eye))


def refresh():
    SRC.mkdir(parents=True, exist_ok=True)
    for pig in range(1, 7):
        for eye in ("Test", "Control"):
            f = study_file(pig, eye)
            if not f.exists():
                continue
            im = Image.open(f)
            im.draft("RGB", (im.size[0] // 4, im.size[1] // 4))   # JPEG decode at reduced size
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            im.save(SRC / f.name, quality=90)
            print("copied", f.name, im.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.refresh:
        refresh()
    OUT.mkdir(parents=True, exist_ok=True)
    outs, ins, origins = [], [], {}
    for pig in range(1, 7):
        files = [SRC / name(pig, eye) for eye in ("Test", "Control")]
        if not all(f.exists() for f in files):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
        for ax, f, title in zip(axes, files, ("Test eye", "Control eye")):
            ax.imshow(Image.open(f))
            ax.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        for ax, title in zip(axes, ("Test eye", "Control eye")):
            b = ax.get_position()
            fig.text((b.x0 + b.x1) / 2, 0.96, title, ha="center", va="center", fontsize=12)
        st, sc = slide(pig, "Test"), slide(pig, "Control")
        out = OUT / (f"Pig{pig}_slide{st}.jpg" if st == sc else f"Pig{pig}_test{st}_control{sc}.jpg")
        fig.savefig(out, dpi=300, pil_kwargs={"quality": 90})
        plt.close(fig)
        outs.append(out)
        for eye, f in zip(("Test", "Control"), files):
            ins.append(f)
            origins[f.name] = str(study_file(pig, eye))
        print("wrote", out.name)
    record(__file__, outs, ins, origins=origins, manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
