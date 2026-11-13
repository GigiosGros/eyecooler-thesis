"""Appendix: the selected observation and staining photographs of every session, with the grades.

Output (figures/aim3/appendix_surface/):
  Pig<N>_S<K>.jpg      cooling session: rows redness photograph and fluorescein photograph; columns test
                       eye before, test eye after, control eye before, control eye after.
                       End-of-life session: the same rows, one time point, test and control eyes.
                       Each panel carries the grades of record.
  appendix_panels.csv  the session, slot, file and grades drawn (read by the thesis caption script)
  MANIFEST.json        provenance of figures/aim3 (provenance.record)

Inputs (copied and reduced by --refresh into figures/aim3/sources/surface_all/):
  Pig<N>_S<K>_<kind>_<Tp>_<Eye>.jpg   the frame of record of each slot, Pig N/Reference/_picks.csv
                                      (checked 2026-11-01 against the session folders), longest
                                      side reduced to MAX_SIDE pixels
  ../master_scores.csv                grades of record, all mine; for animals 1-2 the staining
                                      grades are my later re-grade ('Stain Ian' column)

Crops: the redness photographs are cropped to a close-up square around the eye (2026-11-12): side
EYE_FRAC of the shorter side of the frame, centred on the darkest spot (pupil and iris) that has skin
around it (eye_centre); where that missed the eye the centre is read off a 10 x 10 grid of the full
frame and stored in sources/surface_eye_overrides.csv; the centres used are written to
sources/surface_eye_crops.csv. The staining frames lose the instrument's banner and grey bar.

Run:  python code/figures/aim3_appendix_surface.py [--refresh]
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from provenance import THESIS, record

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources"
OUT = FIG3 / "appendix_surface"
IMG = SRC / "surface_all"
STUDY = Path(os.environ.get("LIVE_PIG_STUDY", THESIS / "data" / "Live Pigs Safety Study"))
MAX_SIDE = 700
BANNER, BAR = 24 / 1024, 16 / 1024     # staining frame: fraction of the height lost top and bottom
KIND = {"Observation": "Obs", "Corneal Staining": "Stain"}
TP = {"Pre": "before", "Post": "after"}
plt.rcParams.update({"font.size": 7})


def sess_no(s):
    return int(re.search(r"Session (\d)", s).group(1))


EYE_FRAC = 0.55                         # redness close-up: side as a fraction of the shorter side of the frame
OVERRIDES = SRC / "surface_eye_overrides.csv"
CROPS = SRC / "surface_eye_crops.csv"


def eye_centre(im):
    """Centre of the eye as fractions of width and height: the darkest spot against its surroundings
    (difference of Gaussians) that has some skin around it, away from the edges of the frame."""
    s = im.copy()
    s.thumbnail((400, 400))
    rgb = np.asarray(s, float)
    a = rgb.mean(axis=2)
    h, w = a.shape
    m = min(h, w)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    skin = gaussian_filter(((R > 110) & (R - G > 12) & (R - B > 5)).astype(float), 0.12 * m)
    score = gaussian_filter(a, 0.03 * m) - gaussian_filter(a, 0.15 * m) + 200 * (skin < 0.2)
    b = int(0.12 * m)
    score[:b] = score[-b:] = 1e9
    score[:, :b] = score[:, -b:] = 1e9
    y, x = np.unravel_index(np.argmin(score), score.shape)
    return x / w, y / h


def eye_crop(im, cx, cy, frac):
    w, h = im.size
    s = int(frac * min(w, h))
    x0 = min(max(int(cx * w - s / 2), 0), w - s)
    y0 = min(max(int(cy * h - s / 2), 0), h - s)
    return im.crop((x0, y0, x0 + s, y0 + s))


def refresh():
    IMG.mkdir(parents=True, exist_ok=True)
    ov = {(r.pig, r.session, r.timepoint, r.eye): r for r in pd.read_csv(OVERRIDES).itertuples()}
    crops = []
    for n in range(1, 7):
        pig = STUDY / f"Pig {n}"
        p = pd.read_csv(pig / "Reference" / "_picks.csv")
        p = p[p.Test.isin(KIND)]
        for r in p.itertuples():
            hits = [q for q in (pig / r.Session).rglob(r.File) if "_Archive" not in q.parts]
            assert len(hits) == 1, (n, r.File, hits)
            im = Image.open(hits[0]).convert("RGB")
            if r.Test == "Observation":
                im.thumbnail((MAX_SIDE * 4, MAX_SIDE * 4))
                o = ov.get((n, sess_no(r.Session), r.Timepoint, r.Eye))
                if o is None:
                    (cx, cy), frac, how = eye_centre(im), EYE_FRAC, "automatic"
                else:
                    cx, cy, frac, how = o.cx, o.cy, (EYE_FRAC if pd.isna(o.frac) else o.frac), "override"
                im = eye_crop(im, cx, cy, frac)
                crops.append(dict(pig=n, session=sess_no(r.Session), timepoint=r.Timepoint, eye=r.Eye, file=r.File,
                                  cx=round(cx, 3), cy=round(cy, 3), frac=frac, centre=how))
            else:
                im.thumbnail((MAX_SIDE * 2, MAX_SIDE * 2))
                w, h = im.size
                im = im.crop((0, int(h * BANNER), w, int(h * (1 - BAR))))
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(IMG / f"Pig{n}_S{sess_no(r.Session)}_{KIND[r.Test]}_{r.Timepoint}_{r.Eye}.jpg", quality=85)
    pd.DataFrame(crops).to_csv(CROPS, index=False)
    used = {(c["pig"], c["session"], c["timepoint"], c["eye"]) for c in crops}
    assert set(ov) <= used, set(ov) - used


def grades():
    d = pd.read_csv(SRC / "master_scores.csv")
    g = {}
    names = {"bulbar": "bulbar", "limbal": "limbal", "corneal": "stain"}
    for r in d.itertuples():
        k = names[r.Measure.split()[0]]
        for tp, eye, v in (("Pre", "Test", r.Pre_Test), ("Pre", "Control", r.Pre_Control),
                           ("Post", "Test", r.Post_Test), ("Post", "Control", r.Post_Control)):
            if pd.notna(v):
                g[(r.Pig, sess_no(r.Session), tp, eye, k)] = v
    return g


def fmt(v):
    return "not graded" if v is None else f"{v:g}"


def session_figure(n, k, g, rows):
    files = sorted(IMG.glob(f"Pig{n}_S{k}_*.jpg"))
    eol = any("_EoL_" in f.name for f in files)
    slots = [("EoL", "Test"), ("EoL", "Control")] if eol else [("Pre", "Test"), ("Post", "Test"), ("Pre", "Control"), ("Post", "Control")]
    # the end-of-life grades sit in the Pre columns of master_scores
    src = {"EoL": "Pre"}
    w = 3.4 if eol else 6.5
    fig, axs = plt.subplots(2, len(slots), figsize=(w, 3.25), squeeze=False)
    for c, (tp, eye) in enumerate(slots):
        for r, kind in enumerate(("Obs", "Stain")):
            ax = axs[r][c]
            ax.axis("off")
            f = IMG / f"Pig{n}_S{k}_{kind}_{tp}_{eye}.jpg"
            gt = lambda s: g.get((n, k, src.get(tp, tp), eye, s))
            if kind == "Obs":
                txt = f"Bulbar {fmt(gt('bulbar'))}, limbal {fmt(gt('limbal'))}"
                vals = dict(bulbar=gt("bulbar"), limbal=gt("limbal"))
            else:
                txt = f"Staining {fmt(gt('stain'))}"
                vals = dict(stain=gt("stain"))
            if f.exists():
                ax.imshow(Image.open(f))
            else:
                ax.text(0.5, 0.5, "no photograph", ha="center", va="center", transform=ax.transAxes)
            ax.set_title((f"{eye} eye" + ("" if eol else f", {TP[tp]}") + "\n" if r == 0 else "") + txt, fontsize=7)
            rows.append(dict(pig=n, session=k, kind=kind, timepoint=tp, eye=eye, **vals))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.005, wspace=0.03, hspace=0.12)
    out = OUT / f"Pig{n}_S{k}.jpg"
    fig.savefig(out, dpi=220, pil_kwargs={"quality": 85})
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.refresh:
        refresh()
    OUT.mkdir(parents=True, exist_ok=True)
    g = grades()
    rows, outs = [], []
    for n, k in sorted({tuple(map(int, re.match(r"Pig(\d)_S(\d)", f.name).groups())) for f in IMG.glob("*.jpg")}):
        outs.append(session_figure(n, k, g, rows))
    csv = OUT / "appendix_panels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    inputs = sorted(IMG.glob("*.jpg")) + [SRC / "master_scores.csv", OVERRIDES, CROPS]
    record(__file__, outs + [csv], inputs, origins={"figures/aim3/sources/surface_all": "Pig N/Reference/_picks.csv frames, reduced; redness frames cropped around the eye (surface_eye_crops.csv)"},
           notes="Selected frames of every session with the grades of record.", manifest=FIG3 / "MANIFEST.json")
    print(len(outs), "figures")


if __name__ == "__main__":
    main()
