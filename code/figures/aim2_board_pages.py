"""One PNG per page of the control-board PDFs, for the appendix figures.

The appendix used to include the Altium exports directly, one page of the PDF
per figure (\\includegraphics[page=N]{...pdf}). LaTeXML ignores the page number
when it builds the accessible EPUB, so every schematic figure showed the same
sheet there. Each page is therefore rendered to its own PNG, which both
pdflatex and LaTeXML include correctly and which keeps its alt text.

Needs pdftoppm (ships with MiKTeX and with Poppler).

The sheets were placed one per page, turned sideways, and most of each sheet
is empty frame. The script now also writes composite pages that stack several
sheets, each cropped to its drawn content and labelled with its sheet name
(the label replaces the title block, which is cropped away), at the full
resolution of the single-sheet PNGs, which are kept:
    control-board-schematics-1-2.png   sheets 1 and 2
    control-board-schematics-3-5.png   sheets 3, 4 and 5
    control-board-drawings-1-2.png     assembly and fabrication drawings

    python code/figures/aim2_board_pages.py
"""
import pathlib
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

from provenance import FIG, record

DPI = 300
# (source PDF, number of pages, output name pattern)
JOBS = [
    ("control-board-schematics.pdf", 5, "control-board-schematics-sheet{n}.png"),
    ("control-board-drawings.pdf", 2, "control-board-drawings-page{n}.png"),
]


def pdftoppm():
    exe = shutil.which("pdftoppm")
    if exe:
        return exe
    miktex = pathlib.Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdftoppm.exe"
    if miktex.exists():
        return str(miktex)
    raise SystemExit("pdftoppm not found; install MiKTeX or Poppler")


def main():
    exe = pdftoppm()
    for pdf_name, pages, pattern in JOBS:
        pdf = FIG / pdf_name
        outputs = []
        for n in range(1, pages + 1):
            out = FIG / pattern.format(n=n)
            stem = out.with_suffix("")          # pdftoppm appends ".png" itself
            subprocess.run([exe, "-png", "-r", str(DPI), "-f", str(n), "-l", str(n),
                            "-singlefile", str(pdf), str(stem)], check=True)
            outputs.append(out)
            print("wrote", out.name)
        record(__file__, outputs, [pdf],
               notes=f"pdftoppm -png -r {DPI}, one page of {pdf_name} per file "
                     "(page number = number in the file name)")
    compose()


# Composite pages. Regions as fractions of the sheet (x0, y0, x1, y1): the area
# inside the frame, and the boxes inside it to ignore when finding the drawn
# content (revision table and title block of the Altium templates).
SCH = dict(inner=(0.018, 0.026, 0.982, 0.972),
           ignore=[(0.465, 0.0, 1.0, 0.17), (0.59, 0.71, 1.0, 1.0)])
DRW = dict(inner=(0.010, 0.013, 0.990, 0.987),
           ignore=[(0.81, 0.0, 1.0, 0.12), (0.675, 0.895, 1.0, 1.0)])
COMPOSITES = [
    ("control-board-schematics-1-2.png", SCH, [
        ("control-board-schematics-sheet1.png", "Sheet 1 of 5: Power"),
        ("control-board-schematics-sheet2.png", "Sheet 2 of 5: ADN8831 controller and output stage")]),
    ("control-board-schematics-3-5.png", SCH, [
        ("control-board-schematics-sheet3.png", "Sheet 3 of 5: Peripherals"),
        ("control-board-schematics-sheet4.png", "Sheet 4 of 5: Thermistor channels"),
        ("control-board-schematics-sheet5.png", "Sheet 5 of 5: Microcontroller")]),
    ("control-board-drawings-1-2.png", DRW, [
        ("control-board-drawings-page1.png", "Assembly drawing"),
        ("control-board-drawings-page2.png", "Fabrication drawing")]),
]
PAD = 40            # px kept around the drawn content
GAP = 120           # px between stacked sheets
FONT_PX = 72        # label height, about 6 mm at 300 dpi


def content_crop(path, regions):
    """The sheet cropped to its drawn content (frame, revision table and title block left out)."""
    import numpy as np
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    h, w, _ = a.shape
    px = lambda f: (int(f[0] * w), int(f[1] * h), int(f[2] * w), int(f[3] * h))
    x0, y0, x1, y1 = px(regions["inner"])
    bg = a[y0 + 5, x0 + 5]
    ink = np.abs(a - bg).max(axis=2) > 60
    keep = np.zeros_like(ink)
    keep[y0:y1, x0:x1] = True
    draw = ImageDraw.Draw(im)
    for box in regions["ignore"]:
        bx0, by0, bx1, by1 = px(box)
        keep[by0:by1, bx0:bx1] = False
        # blank it too: the crop is a rectangle and can still reach into the box
        draw.rectangle((bx0, by0, bx1, by1), fill=tuple(int(v) for v in bg))
    ys, xs = np.nonzero(ink & keep)
    box = (max(xs.min() - PAD, 0), max(ys.min() - PAD, 0),
           min(xs.max() + PAD, w), min(ys.max() + PAD, h))
    return im.crop(box), tuple(int(v) for v in bg)


def compose():
    try:
        font = ImageFont.truetype("arial.ttf", FONT_PX)
    except OSError:
        font = ImageFont.load_default()
    for out_name, regions, sheets in COMPOSITES:
        parts = [(content_crop(FIG / f, regions), label) for f, label in sheets]
        width = max(c.width for (c, _), _ in parts)
        height = sum(c.height + FONT_PX + 30 for (c, _), _ in parts) + GAP * (len(parts) - 1)
        page = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(page)
        y = 0
        for (crop, bg), label in parts:
            draw.text((0, y), label, fill="black", font=font)
            y += FONT_PX + 30
            page.paste(crop, ((width - crop.width) // 2, y))  # same scale for every sheet
            y += crop.height + GAP
        out = FIG / out_name
        page.save(out, dpi=(DPI, DPI))
        print("wrote", out.name, page.size)
        record(__file__, [out], [FIG / f for f, _ in sheets],
               notes="sheets cropped to their drawn content (frame, revision table and "
                     "title block left out), labelled, stacked at full resolution")


if __name__ == "__main__":
    main()
