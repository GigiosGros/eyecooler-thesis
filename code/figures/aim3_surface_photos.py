"""Chapter 4: Figure fig:aim3-surface-photos (ocular surface before and after one session).

Output (figures/aim3/):
  surface-photos.png   2 x 4 grid: row 1 the redness photographs (Observation), row 2 the
                       fluorescein staining photographs (Corneal Staining); columns test eye
                       before, test eye after, control eye before, control eye after. Each
                       panel is labelled with eye, time point and the grade(s) of record.
  MANIFEST.json        provenance (provenance.record)

Parameters:  --pig N --session K   (default: animal 2, session 3)
  Session chosen (2026-10-29): animal 2, session 3 (120 min). Of the nine sessions of animals
  2-4 it is the one with the clearest staining change in the test eye: grade 2 -> 4 in the
  test eye against 3 -> 1 in the control eye (Test_worse = yes, change T-C = +4; master_scores),
  the only session of animals 2-4 in which the test eye's staining got worse relative to the
  control eye's. UPDATE-PIGS56: re-check the choice with animals 5 and 6.

Inputs (copied by --refresh into figures/aim3/sources/surface/, origins in origins.json):
  Pig<N>_S<K>_<Test>_<Tp>_<Eye>.<jpg|tif>
                   the frame of record of each slot: the one I left in the session folder
                   (03 - Observation / 05 - Corneal Staining, 01 - Pre / 02 - Post
                   Experiment), as listed in Pig N/Reference/_picks.csv (1 pick per slot)
  ../master_scores.csv
                   grades of record (the session scoring files, via Live Pigs Master Analysis;
                   all mine; for animals 1-2 the staining grades are my later re-grade,
                   'Stain Ian' column)

Crops: every redness photograph is cropped to a square of CROP_SIDE_FRAC of the image width
centred on the eye (eye centres set by hand below, in fractions of width and height, read
from the frames), which leaves out the lab background; the staining frames lose only the
instrument's information banner (top) and grey bar (bottom). No person is identifiable in the
crops (gloved fingers holding the lids may show).

Run:  python code/figures/aim3_surface_photos.py [--pig 2 --session 3] [--refresh]
"""
import argparse
import csv
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

from provenance import THESIS, record
from aim3_invivo_figures import STUDY, SESSIONS

FIG3 = THESIS / "figures" / "aim3"
SRC = FIG3 / "sources"
SURF = SRC / "surface"
OUT = FIG3 / "surface-photos.png"
SCORES = SRC / "master_scores.csv"
TESTS = {"Observation": ("03 - Observation", "redness"),
         "Corneal Staining": ("05 - Corneal Staining", "staining")}
TP_DIR = {"Pre": "01 - Pre Experiment", "Post": "02 - Post Experiment"}
CROP_SIDE_FRAC = 0.56
# eye centre (x, y) in fractions of the (EXIF-upright) image, read from the frames by hand
EYE_CENTRE = {
    (2, 3, "Pre", "Test"): (0.15, 0.50),
    (2, 3, "Pre", "Control"): (0.65, 0.55),
    (2, 3, "Post", "Test"): (0.43, 0.56),
    (2, 3, "Post", "Control"): (0.61, 0.58),
}
STAIN_TRIM = (50, 40)          # rows removed at the top (banner) and bottom (grey bar)


def picks(pig, session):
    f = STUDY / f"Pig {pig}" / "Reference" / "_picks.csv"
    out = {}
    with open(f, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["Session"] == SESSIONS[session] and r["Test"] in TESTS:
                out[(r["Test"], r["Timepoint"], r["Eye"])] = r["File"]
    return out


def local_name(pig, session, test, tp, eye, src_name):
    tag = "Observation" if test == "Observation" else "Staining"
    return f"Pig{pig}_S{session}_{tag}_{tp}_{eye}{pathlib_suffix(src_name)}"


def pathlib_suffix(name):
    return "." + name.rsplit(".", 1)[-1].lower()


def refresh(pig, session):
    SURF.mkdir(parents=True, exist_ok=True)
    origins = {}
    for (test, tp, eye), name in picks(pig, session).items():
        folder, _ = TESTS[test]
        src = STUDY / f"Pig {pig}" / SESSIONS[session] / folder / TP_DIR[tp] / name
        if not src.exists():
            raise SystemExit(f"pick not in the session folder: {src}")
        dst = SURF / local_name(pig, session, test, tp, eye, name)
        shutil.copy2(src, dst)
        origins[str(dst.relative_to(THESIS)).replace("\\", "/")] = (
            str(src) + " (pick of record, Reference/_picks.csv)")
    old = SRC / "origins.json"
    merged = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    merged.update(origins)
    old.write_text(json.dumps(merged, indent=1), encoding="utf-8")


def grades(pig, session):
    d = pd.read_csv(SCORES)
    d = d[(d["Pig"] == pig) & (d["Session"] == SESSIONS[session])]
    g = {}
    for _, r in d.iterrows():
        key = {"bulbar redness (0-4)": "bulbar", "limbal redness (0-4)": "limbal",
               "corneal staining (0-4)": "staining"}.get(r["Measure"])
        if key:
            for tp in ("Pre", "Post"):
                for eye in ("Test", "Control"):
                    g[(key, tp, eye)] = r[f"{tp}_{eye}"]
    return g


def _fmt(v):
    return "–" if pd.isna(v) else f"{int(v)}" if float(v).is_integer() else f"{v}"


def load_panel(pig, session, test, tp, eye):
    f = next(SURF.glob(f"Pig{pig}_S{session}_{'Observation' if test == 'Observation' else 'Staining'}"
                       f"_{tp}_{eye}.*"))
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    W, H = im.size
    if test == "Observation":
        cx, cy = EYE_CENTRE.get((pig, session, tp, eye), (0.5, 0.5))
        side = int(CROP_SIDE_FRAC * W)
        x0 = min(max(int(cx * W - side / 2), 0), W - side)
        y0 = min(max(int(cy * H - side / 2), 0), H - side)
        im = im.crop((x0, y0, x0 + side, y0 + side))
    else:
        im = im.crop((0, STAIN_TRIM[0], W, H - STAIN_TRIM[1]))
    im.thumbnail((1400, 1400))
    return f, im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pig", type=int, default=2)
    ap.add_argument("--session", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.refresh:
        refresh(a.pig, a.session)
    g = grades(a.pig, a.session)
    cols = [("Test", "Pre"), ("Test", "Post"), ("Control", "Pre"), ("Control", "Post")]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.75),
                             gridspec_kw={"height_ratios": [1, 0.8]})
    inputs = []
    for c, (eye, tp) in enumerate(cols):
        when = "before" if tp == "Pre" else "after"
        for r, test in enumerate(("Observation", "Corneal Staining")):
            f, im = load_panel(a.pig, a.session, test, tp, eye)
            inputs.append(f)
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if test == "Observation":
                txt = (f"bulbar {_fmt(g.get(('bulbar', tp, eye)))}, "
                       f"limbal {_fmt(g.get(('limbal', tp, eye)))}")
                ax.set_title(f"{eye} eye, {when}", fontsize=8.5)
            else:
                txt = f"staining {_fmt(g.get(('staining', tp, eye)))}"
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.5, color="white",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.65))
    axes[0, 0].set_ylabel("Redness", fontsize=8.5)
    axes[1, 0].set_ylabel("Fluorescein\nstaining", fontsize=8.5)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print(f"animal {a.pig} session {a.session}: grades {g}")
    origins_f = SRC / "origins.json"
    allo = json.loads(origins_f.read_text(encoding="utf-8")) if origins_f.exists() else {}
    rel = [str(p.relative_to(THESIS)).replace("\\", "/") for p in inputs]
    origins = {k: allo[k] for k in rel if k in allo}
    origins["figures/aim3/sources/master_scores.csv"] = allo.get(
        "figures/aim3/sources/master_scores.csv", "")
    record(__file__, [OUT], inputs + [SCORES], origins=origins,
           notes=f"Animal {a.pig}, session {a.session}: picks of record, grades of record "
                 "(master_scores). UPDATE-PIGS56: re-check the session choice.",
           manifest=FIG3 / "MANIFEST.json")


if __name__ == "__main__":
    main()
