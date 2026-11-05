"""Chapter 2 figures: copied from the thesis-analysis package output, margins trimmed.

The package (03 - Experiments & Data/Analysis Code/thesis-analysis) draws these figures
itself from the raw recordings; this script only selects them, trims the white margin
(20 px kept) and saves them under the thesis names. Site names in the figures come from
the package config [figures.channel_display_names], which matches the thesis text.

Package run that produced the inputs:
  python run_analysis.py --data <UploadExperiment> --out results
  data = 00 - Inbox (to sort)/UploadExperiment-20260915T110439Z-1-001.zip, extracted
The package is not under git, so the manifest records the SHA-256 of its code and config
files and of the data zip, in addition to every input and output image.

Run:  python code/figures/aim1_package_figures.py
"""
import pathlib

from PIL import Image, ImageChops

from provenance import THESIS, record, sha256

BOX = THESIS.parents[2]  # .../Eye Cooler
PKG = BOX / "03 - Experiments & Data" / "Analysis Code" / "thesis-analysis"
RES = PKG / "results"
DATA_ZIP = BOX / "03 - Experiments & Data" / "Ex-Vivo" / "UploadExperiment-20260915T110439Z-1-001.zip"
OUT = THESIS / "figures" / "aim1"
SRC = OUT / "sources"

FIGS = {
    "aim1-spatial-screen.png": "01_quality_control/spatial_screen_print.png",
    "aim1-excluded-traces.png": "01_quality_control/trace_exclusions.png",
    "aim1-preserved-porcine-curves.png": "02_groups/Porcine_preserved__aluminum_original/curves.png",
    "aim1-fresh-porcine-curves.png": "02_groups/Porcine_fresh__aluminum_original/curves.png",
    "aim1-human-curves.png": "02_groups/Human__aluminum_original/curves.png",
    "aim1-human-vs-porcine.png":
        "03_comparisons/Human__aluminum_original__vs__Porcine_fresh__aluminum_original/curves.png",
    "aim1-preserved-vs-fresh.png":
        "03_comparisons/Porcine_fresh__aluminum_original__vs__Porcine_preserved__aluminum_original/curves.png",
}


def trim(im, pad=20):
    b = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    return im.crop((max(b[0] - pad, 0), max(b[1] - pad, 0),
                    min(b[2] + pad, im.width), min(b[3] + pad, im.height)))


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    pkg_files = sorted(list((PKG / "scec_analysis").glob("*.py")) + [PKG / "run_analysis.py"]
                       + list((PKG / "config").glob("*.toml")))
    pkg_hashes = {str(p.relative_to(PKG)).replace("\\", "/"): sha256(p) for p in pkg_files}
    for name, rel in FIGS.items():
        src = RES / rel
        snap = SRC / ("package__" + rel.replace("/", "__"))
        snap.write_bytes(src.read_bytes())  # unchanged copy of the package output
        dst = OUT / name
        trim(Image.open(snap).convert("RGB")).save(dst, dpi=(200, 200))
        record(__file__, [dst], [snap], manifest=OUT / "MANIFEST.json",
               origins={snap.name: "thesis-analysis/results/" + rel},
               notes={"package_code_sha256": pkg_hashes,
                      "data_zip": DATA_ZIP.name, "data_zip_sha256": sha256(DATA_ZIP)})
        print("wrote", name)


if __name__ == "__main__":
    main()
