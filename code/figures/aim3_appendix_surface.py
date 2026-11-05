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

Crops: the redness photographs are cropped to a centered square, the staining frames lose the
instrument's banner and grey bar. No hand-picked crops: this is the record, not an illustration.

Run:  python code/figures/aim3_appendix_surface.py [--refresh]
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

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


def refresh():
    IMG.mkdir(parents=True, exist_ok=True)
    for n in range(1, 7):
        pig = STUDY / f"Pig {n}"
        p = pd.read_csv(pig / "Reference" / "_picks.csv")
        p = p[p.Test.isin(KIND)]
        for r in p.itertuples():
            hits = [q for q in (pig / r.Session).rglob(r.File) if "_Archive" not in q.parts]
            assert len(hits) == 1, (n, r.File, hits)
            im = Image.open(hits[0]).convert("RGB")
            im.thumbnail((MAX_SIDE * 2, MAX_SIDE * 2))
            w, h = im.size
            if r.Test == "Observation":
                s = min(w, h)
                im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            else:
                im = im.crop((0, int(h * BANNER), w, int(h * (1 - BAR))))
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            im.save(IMG / f"Pig{n}_S{sess_no(r.Session)}_{KIND[r.Test]}_{r.Timepoint}_{r.Eye}.jpg", quality=85)


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
    inputs = sorted(IMG.glob("*.jpg")) + [SRC / "master_scores.csv"]
    record(__file__, outs + [csv], inputs, origins={"figures/aim3/sources/surface_all": "Pig N/Reference/_picks.csv frames, reduced"},
           notes="Selected frames of every session with the grades of record.", manifest=FIG3 / "MANIFEST.json")
    print(len(outs), "figures")


if __name__ == "__main__":
    main()
