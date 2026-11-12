"""Figure 4.13: H&E sections of the test and control eyes of animal 4 (slide 5), the whole section and three
close-ups: cornea, sclera behind the limbus (under the ring), and posterior retina.

Output: figures/aim3/histology.png (+ entry in figures/aim3/MANIFEST.json)
Inputs: the whole-slide scans Pig 4 {Test,Control}_5.jpg in
        Pig 4/Session 4 - End of Life/07 - Histology/H&E Staining/{Test Eye,Control Eye}/ (about 19,000 px a side).
        The close-ups are cut from the scans decoded at half resolution and saved, unaltered, to
        figures/aim3/sources/histology_crops/ (--refresh); the overviews are figures/aim3/sources/histology/.
Crop centres are fractions of the scan width and height, chosen by eye on the overview; side in mm. Scale from the
scanner's own 1 mm bar: about 910 px/mm at full resolution (920 and 905 measured on the two scans).

Run:  python code/figures/aim3_histology_figure.py [--refresh]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from provenance import THESIS, record

FIG3 = THESIS / "figures" / "aim3"
OVR = FIG3 / "sources" / "histology"
CROPS = FIG3 / "sources" / "histology_crops"
STUDY = Path(os.environ.get("LIVE_PIG_STUDY", THESIS / "data" / "Live Pigs Safety Study"))
PIG, SLIDE = 4, 5
PX_PER_MM = 910 / 2          # at the half resolution the crops are taken from
SIDE_MM = 1.6
Image.MAX_IMAGE_PIXELS = None

# region -> (centre x, centre y) as fractions of the scan
REGIONS = {
    "Test": {"Cornea": (0.76, 0.90), "Sclera under the ring": (0.925, 0.47), "Posterior retina": (0.085, 0.55)},
    "Control": {"Cornea": (0.952, 0.50), "Sclera under the ring": (0.835, 0.12), "Posterior retina": (0.31, 0.69)},
}


def scan(eye):
    return (STUDY / f"Pig {PIG}" / "Session 4 - End of Life" / "07 - Histology" / "H&E Staining" / f"{eye} Eye"
            / f"Pig {PIG} {eye}_{SLIDE}.jpg")


def crop_name(eye, region):
    return f"Pig {PIG} {eye}_{SLIDE} - {region}.png"


def refresh():
    CROPS.mkdir(parents=True, exist_ok=True)
    half = SIDE_MM * PX_PER_MM / 2
    for eye, regs in REGIONS.items():
        im = Image.open(scan(eye))
        W, H = im.size
        im.draft("RGB", (W // 2, H // 2))
        im = im.convert("RGB")
        w, h = im.size
        for region, (fx, fy) in regs.items():
            cx, cy = fx * w, fy * h
            box = tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
            im.crop(box).save(CROPS / crop_name(eye, region))
            print("cropped", eye, region, box)
        del im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    if ap.parse_args().refresh:
        refresh()
    regions = list(REGIONS["Test"])
    # 2026-11-11: drawn at the printed width (6.0 in, included at width=\textwidth) with text of 8 pt; it was
    # 12 in wide (titles 10-11 pt, scale labels 8 pt, i.e. 4-5.5 pt in print). Same crops and layout; dpi raised
    # so the PNG keeps its 3000 px width.
    fig, axs = plt.subplots(2, 4, figsize=(6.0, 3.2), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    ins = []
    for row, eye in enumerate(("Test", "Control")):
        ov = OVR / f"Pig {PIG} {eye}_{SLIDE}.jpg"
        ins.append(ov)
        o = Image.open(ov)
        ax = axs[row, 0]
        ax.imshow(o)
        ax.set_axis_off()
        ax.set_title(f"{eye} eye", fontsize=8.5, loc="left")
        for k, region in enumerate(regions):
            fx, fy = REGIONS[eye][region]
            s = SIDE_MM * 910 * o.size[0] / FULL_W[eye]
            ax.add_patch(Rectangle((fx * o.size[0] - s / 2, fy * o.size[1] - s / 2), s, s, fill=False, ec="black", lw=0.6))
            ax.text(fx * o.size[0] + s / 2 + 15, fy * o.size[1], "abc"[k], fontsize=8, va="center", fontweight="bold")
            c = CROPS / crop_name(eye, region)
            ins.append(c)
            a = axs[row, k + 1]
            img = Image.open(c)
            a.imshow(img)
            a.set_axis_off()
            if row == 0:
                a.set_title(f"({'abc'[k]}) {region}", fontsize=8)
            bar = 0.2 * PX_PER_MM          # 200 um
            y0 = img.size[1] * 0.93
            a.plot([img.size[0] * 0.95 - bar, img.size[0] * 0.95], [y0, y0], color="black", lw=1.6)
            a.text(img.size[0] * 0.95, y0 - img.size[1] * 0.03, "200 µm", ha="right", va="bottom", fontsize=8)
    fig.tight_layout(pad=0.3)
    out = FIG3 / "histology.png"
    fig.savefig(out, dpi=500)
    plt.close(fig)
    record(__file__, [out], ins, origins={f"Pig {PIG} {e}_{SLIDE}.jpg": str(scan(e)) for e in ("Test", "Control")},
           manifest=FIG3 / "MANIFEST.json")
    print("wrote", out.name)


FULL_W = {"Test": 19581, "Control": 18466}   # full-resolution scan widths (px)

if __name__ == "__main__":
    main()
