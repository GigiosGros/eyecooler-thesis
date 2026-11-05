#!/usr/bin/env python3
"""
Format-agnostic reader for RETeval ERG exports.

The device exports the same table either as a workbook (.xlsx) or as a CSV, and
both share one layout: 14 metadata rows (col A = field name, col B = the device's
own description of it, then one column PAIR per recording), a blank row, the
"Reported Waveform" header row, and then the ms / uV samples.

    row  1  PatientID
    row  2  PatientBirthdate
    row  3  TestingDate           <- groups the recordings of one timepoint
    row  4  TestedEye             <- RightEye / LeftEye / BothEyes
    row  5  ElectrodePackageType
    row  6  Flash (Td.s or Td)
    row  7  Flash (cd.s/m2 or cd/m2)   <- the flash level we key on
    row  8  FlashColor
    rows 9-12 background / stimulus frequency
    row 13  (blank)
    row 14  "Reported Waveform"  ...  ms | uV | ms | uV ...
    row 15+ samples

The CSV is the only form the device wrote for the later sessions, so everything
downstream (split_erg -> extract_erg / erg_all) goes through here instead of
calling openpyxl directly. A CSV cell arrives as text, so values are coerced to
the types the .xlsx path produced (float / datetime / date / None) - otherwise
the "is this a number?" checks in the waveform readers would reject every sample.

Also holds erg_folder(), which finds a session's ERG folder whatever number
prefix it carries ("06 - ERG Recording" in the cooling sessions, "05 - ..." in
the End-of-Life one).
"""
import time
import csv
import datetime
import re
from pathlib import Path

META_ROWS = 14          # metadata rows before the waveform samples (DEFAULT layout only)
# 2026-10-22: exports made with more fields ticked in RFF Extractor (the device's own
# a/b-wave times and amplitudes, PhNR, percentiles, oscillatory-potential cursors...)
# put those rows between Stimulus Frequency and "Reported Waveform", so the samples
# start later than row 15 (row 44 in Pig 2's 10/21 re-export, later still for Pigs
# 3-4). Readers must locate the header with waveform_start(), never assume row 14.
WAVEFORM_HEADER = "Reported Waveform"
CSV_ENCODING = "latin-1"   # device writes cp1252-ish text; latin-1 never raises

_DT = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_D = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def coerce(s):
    """CSV text -> the value type the .xlsx export would have carried."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if _DT.match(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    if _D.match(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    try:
        return float(s)
    except ValueError:
        return s


def iter_rows(path: Path, max_row=None):
    """Yield every row of a RETeval export as a tuple of values (.xlsx or .csv).

    max_row stops the read early - the metadata scan only needs the first 14
    rows, and the waveform readers only the averaged segment, so a full pass
    over ~118k sample rows is avoided where it isn't needed."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, encoding=CSV_ENCODING, newline="") as fh:
            for i, row in enumerate(csv.reader(fh), 1):
                if max_row and i > max_row:
                    return
                yield tuple(coerce(c) for c in row)
    else:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        try:
            for row in ws.iter_rows(max_row=max_row, values_only=True):
                yield row
        finally:
            wb.close()


def waveform_start(rows):
    """0-based index of the first SAMPLE row (the row after the "Reported Waveform"
    header) in a list of export rows, whatever metadata the export carries.
    Falls back to the default layout (row 15) only when no header is found."""
    for i, r in enumerate(rows):
        if r and isinstance(r[0], str) and r[0].strip() == WAVEFORM_HEADER:
            return i + 1
    return META_ROWS


def head(path: Path, n=META_ROWS):
    """First n rows, padded to a common width - the metadata block."""
    rows = list(iter_rows(path, max_row=n))
    while len(rows) < n:
        rows.append(())
    ncol = max((len(r) for r in rows), default=0)
    return [tuple(r) + (None,) * (ncol - len(r)) for r in rows], ncol


def recordings(path: Path):
    """[{ms_col, dev_time, eye, flash}] - one entry per recording column pair.

    A recording is a column with both a TestedEye (row 4) and a TestingDate
    (row 3); its samples live in that column (ms) and the next one (uV). The
    'BothEyes' columns the device emits per timepoint carry no waveform and no
    flash level, so they fall out downstream on the flash check."""
    rows, ncol = head(path)

    def cell(rn, c):
        return rows[rn - 1][c - 1] if c - 1 < len(rows[rn - 1]) else None

    out = []
    for c in range(3, ncol + 1):
        if cell(4, c) not in (None, "") and cell(3, c) not in (None, ""):
            out.append(dict(ms_col=c, dev_time=cell(3, c), eye=cell(4, c),
                            flash=cell(7, c)))
    return out


def timepoints(path: Path):
    """Sorted distinct TestingDate values - one per ERG timepoint."""
    return sorted({r["dev_time"] for r in recordings(path)})


# End-of-Life sessions have no pre/post: their one split file sits directly in
# the ERG folder as "<source> - End of Life.xlsx". Internally the tools still key it as phase
# "Pre" (it is the session's only/baseline recording set), but nothing on disk or in a sheet
# header says "Pre" for it.
EOL_SUFFIX = " - End of Life.xlsx"


def is_eol(session) -> bool:
    return "End of Life" in str(session)


def phase_files(erg: Path, phase: str, sub: str, session=None):
    """Split workbooks for one phase: <erg>/<sub>/*.xlsx, or for an End-of-Life session's
    'Pre' phase <erg>/*  - End of Life.xlsx."""
    if is_eol(session if session is not None else erg.parent):
        if phase != "Pre":
            return []
        return sorted(f for f in listing(erg, "*" + EOL_SUFFIX) if not f.name.startswith("~$"))
    return sorted(f for f in listing(erg / sub, "*.xlsx") if not f.name.startswith("~$"))


def listing(folder: Path, pattern: str, tries: int = 4, wait_s: float = 1.5):
    """folder.glob(pattern), asked again when it comes back empty for a folder that exists.
    Box Drive can return an empty listing for a folder it hasn't finished listing; without
    this a phase was skipped with no message (2026-11-05). Still empty after the retries:
    printed as a warning when the folder isn't empty, so a skip is never silent."""
    folder = Path(folder)
    for i in range(tries):
        found = list(folder.glob(pattern))
        if found or not folder.is_dir():
            return found
        time.sleep(wait_s)
    if any(folder.iterdir()):
        print(f"  ! WARNING: no {pattern} in {folder} after {tries} tries, but the folder is not empty")
    return found


def erg_folder(session_folder: Path):
    """The session's ERG folder, whatever its number prefix ('06 - ERG
    Recording' in the cooling sessions, '05 - ...' at End of Life). None if the
    session has none."""
    session_folder = Path(session_folder)
    if not session_folder.is_dir():
        return None
    for _ in range(4):
        for d in sorted(session_folder.iterdir()):
            if d.is_dir() and "erg" in d.name.lower():
                return d
        time.sleep(1.5)                 # Box Drive: listing not ready yet (see listing())
    return None
