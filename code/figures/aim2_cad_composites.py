"""Figures 3.1 and 3.2: compose the CAD screenshots of the rings and heat exchangers.

Inputs (figures/aim2/sources/cad/, copied unchanged from my screenshots,
taken 2026-10-01 in the CAD model):
  ring cut-aways   154112 = v1, 154102 = v2, 154050 = v3 (v3 = lens over the cornea)
  HX cut-aways     154234 = v1, 154243 = v2, 154258 = v3 (pipe diameter, channel height, lid)
  HX top views     155100 = one CAD view of all three blocks at a common scale

Processing: white margins trimmed; ring panels set to a common height; heat-exchanger
cut-aways scaled to a common 35 px/mm using the half-width of the central hole
(8 / 8 / 7.75 mm, measured as 232 / 272 / 348 px), because the cut faces were viewed at
different zoom (the pixel widths were measured by hand on the screenshots;
the hard-coded layout scale is styling); top views split into the three objects and scaled together. No pixel
content is drawn or altered other than resampling and the panel letters.

Run:  python code/figures/aim2_cad_composites.py
"""
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from provenance import FIG, SOURCES, record

CAD = SOURCES / "cad"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"
FONT_R = "C:/Windows/Fonts/arial.ttf"


def trim(im):
    bg = Image.new("RGB", im.size, (255, 255, 255))
    mask = ImageChops.difference(im, bg).convert("L").point(lambda p: 255 if p > 18 else 0)
    return im.crop(mask.getbbox())


def shot(stamp):
    return CAD / f"Screenshot 2026-10-01 {stamp}.png"


def rings():
    order = [("154112", "A", "v1"), ("154102", "B", "v2"), ("154050", "C", "v3")]
    H, gap, top = 1000, 60, 80
    panels = []
    for stamp, lab, v in order:
        im = trim(Image.open(shot(stamp)).convert("RGB"))
        panels.append((im.resize((int(im.width * H / im.height), H), Image.LANCZOS), lab, v))
    W = sum(p.width for p, _, _ in panels) + gap * (len(panels) + 1)
    out = Image.new("RGB", (W, H + top + 30), "white")
    d = ImageDraw.Draw(out)
    fb, fr = ImageFont.truetype(FONT_B, 52), ImageFont.truetype(FONT_R, 44)
    x = gap
    for p, lab, v in panels:
        out.paste(p, (x, top))
        d.text((x, 10), lab, font=fb, fill="black")
        d.text((x + 50, 16), v, font=fr, fill="black")
        x += p.width + gap
    dst = FIG / "ring-variants-model.png"
    out.save(dst, dpi=(300, 300))
    record(__file__, [dst], [shot(s) for s, _, _ in order],
           origins={str(shot(s).name): "my CAD screenshots, 2026-10-01"
                    for s, _, _ in order},
           notes="Figure 3.1 (fig:aim2-eye-model). Mapping v1/v2/v3 as listed in the script docstring.")


def heat_exchangers():
    order = [("154234", "A", "v1", 232 / 8.0), ("154243", "B", "v2", 272 / 8.0),
             ("154258", "C", "v3", 348 / 7.75)]
    target = 35.0  # px per mm
    row1 = []
    for stamp, lab, v, ppmm in order:
        im = trim(Image.open(shot(stamp)).convert("RGB"))
        k = target / ppmm
        row1.append((im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS), lab, v))
    # row 2: split the top-view screenshot into its three objects by empty columns
    topv = Image.open(shot("155100")).convert("RGB")
    ink = np.asarray(topv).astype(int).min(axis=2) < 230
    cols = ink.any(axis=0)
    runs, x, W = [], 0, len(cols)
    while x < W:
        if cols[x]:
            s = x
            while x < W and cols[x]:
                x += 1
            runs.append((s, x))
        else:
            x += 1
    runs = [r for r in runs if r[1] - r[0] > 60]
    objs = [trim(topv.crop((s, 0, e, topv.height))) for s, e in runs]
    assert len(objs) == 3, f"expected 3 objects in the top view, found {len(objs)}"
    f2 = row1[0][0].width / objs[0].width * 0.80
    objs = [o.resize((int(o.width * f2), int(o.height * f2)), Image.LANCZOS) for o in objs]
    gap, lab_h = 80, 150
    colw = [max(p.width, o.width) for (p, _, _), o in zip(row1, objs)]
    W = sum(colw) + gap * 4
    H1 = max(p.height for p, _, _ in row1)
    H2 = max(o.height for o in objs)
    out = Image.new("RGB", (W, lab_h + H1 + 60 + H2 + 20), "white")
    d = ImageDraw.Draw(out)
    fb, fr = ImageFont.truetype(FONT_B, 110), ImageFont.truetype(FONT_R, 95)
    x = gap
    for (p, lab, v), o, cw in zip(row1, objs, colw):
        d.text((x, 15), lab, font=fb, fill="black")
        d.text((x + 95, 27), v, font=fr, fill="black")
        out.paste(p, (x + (cw - p.width) // 2, lab_h + (H1 - p.height) // 2))
        out.paste(o, (x + (cw - o.width) // 2, lab_h + H1 + 60 + (H2 - o.height) // 2))
        x += cw + gap
    dst = FIG / "heat-exchanger-variants.png"
    out.save(dst, dpi=(300, 300))
    ins = [shot(s) for s, *_ in order] + [shot("155100")]
    record(__file__, [dst], ins,
           origins={p.name: "my CAD screenshots, 2026-10-01" for p in ins},
           notes="Figure 3.2 (fig:aim2-hx-model). Row 1 scaled to 35 px/mm from the central "
                 "hole; row 2 = top views at one common scale.")


if __name__ == "__main__":
    rings()
    heat_exchangers()
    print("wrote ring-variants-model.png, heat-exchanger-variants.png")
