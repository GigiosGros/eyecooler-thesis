"""One PNG per page of the control-board PDFs, for the appendix figures.

The appendix used to include the Altium exports directly, one page of the PDF
per figure (\\includegraphics[page=N]{...pdf}). LaTeXML ignores the page number
when it builds the accessible EPUB, so every schematic figure showed the same
sheet there. Each page is therefore rendered to its own PNG, which both
pdflatex and LaTeXML include correctly and which keeps its alt text.

Needs pdftoppm (ships with MiKTeX and with Poppler).

    python code/figures/aim2_board_pages.py
"""
import pathlib
import shutil
import subprocess

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


if __name__ == "__main__":
    main()
