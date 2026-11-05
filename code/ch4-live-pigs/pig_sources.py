#!/usr/bin/env python3
"""
Source-file readers for the Live Pigs study -- the layer that knows WHERE each
measurement lives on disk and HOW to address it as an Excel external reference.

This is a library, not a script. It was extracted from the retired
build_pig_master.py, whose workbook-building half is gone: build_pig_TC_analysis
now reads the sources directly (VirtualMaster) instead of going through a
"Pig N Master.xlsx" intermediate, so only the readers were still live.

Each reader takes a timepoint folder and returns {row-label: excel-formula},
including only the cells that actually hold a value in the source (an external
reference to an empty cell reads as 0, so blanks must be omitted, not linked).

Nothing here copies data: every formula is a live external reference, e.g.
    ='C:\\...\\04 - Tonometry\\01 - Pre Experiment\\[Tonometry.xlsx]Sheet1'!B2

Folders are resolved by POSITION (number prefix stripped), so the tools keep
working after Rename-ExperimentFiles changes file names, and whether a test sits
at "04 - Tonometry" (cooling session) or "03 - Tonometry" (End of Life, which is
renumbered because it has no Calibration/Cooling step).
"""
import re
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

# the study root is the folder holding _Tools, so renaming the study folder
# (e.g. "Live Pigs" -> "Live Pigs Safety Study") cannot break the default
DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

MON_SAMPLE_ROWS = 100000   # upper bound for monitoring sample ranges


# ---------------------------------------------------------------- helpers
def cooling_dir(sess: Path) -> Path:
    """The session's device-temperature folder: "07 - Cooling" in a cooling session,
    a later number in an End of Life session (07 there is Histology)."""
    for d in sorted(sess.glob("[0-9]* - Cooling")):
        if d.is_dir():
            return d
    return sess / "07 - Cooling"


def find_xlsx(folder: Path):
    """First real .xlsx in a folder (skip Excel lock files and master output)."""
    if folder is None or not folder.exists():
        return None
    for f in sorted(folder.glob("*.xlsx")):
        if f.name.startswith("~$"):
            continue
        if f.name.endswith("Master.xlsx"):
            continue
        # a "..._BACKUP_<stamp>.xlsx" copy (the earlier grade set) shares the folder
        # with the scoring file of record; it was avoided only because "." sorts
        # before "_". Skip it explicitly.
        if "_backup_" in f.name.lower():
            continue
        return f
    return None


def open_src(folder, sheet=None):
    """Return (file_path, worksheet) for a folder's data file, or (None, None)."""
    f = find_xlsx(folder)
    if not f:
        return None, None
    try:
        wb = openpyxl.load_workbook(f, data_only=True, read_only=True)
    except Exception as e:
        print(f"    ! could not read {f.name}: {e}")
        return None, None
    if sheet is None:
        return f, wb.worksheets[0]
    return (f, wb[sheet]) if sheet in wb.sheetnames else (f, None)


def ext_ref(f: Path, sheet: str, ref: str):
    """Excel external reference formula to <ref> (cell or range) in a source file."""
    return f"='{f.parent}\\[{f.name}]{sheet}'!{ref}"


def ext_agg(f: Path, sheet: str, rng: str, func: str):
    """Aggregate formula over an external range, e.g. =AVERAGE('...'!B2:D2)."""
    return f"={func}(" + ext_ref(f, sheet, rng)[1:] + ")"


def has(ws, coord):
    return ws[coord].value not in (None, "")


def find_test_folder(sess_path: Path, suffix: str):
    """Locate a test folder by the name after its number prefix, so it works
    whether the folder is '04 - Tonometry' (normal session) or '03 - Tonometry'
    (End of Life, renumbered)."""
    if not sess_path.exists():
        return None
    pat = re.compile(r"^\d+\s*-\s*" + re.escape(suffix) + r"$", re.I)
    for d in sorted(sess_path.iterdir()):
        if d.is_dir() and pat.match(d.name):
            return d
    p = sess_path / suffix
    return p if p.exists() else None


def timepoints_for(base: Path):
    """[(label, folder)] for a test folder: Pre/Post if those subfolders exist,
    else a single flat timepoint (the End-of-Life case)."""
    if base is None:
        return []
    pre  = base / "01 - Pre Experiment"
    post = base / "02 - Post Experiment"
    if pre.exists() or post.exists():
        out = []
        if pre.exists():
            out.append(("Pre", pre))
        if post.exists():
            out.append(("Post", post))
        return out
    return [("once", base)]


# ---------------------------------------------------------------- per-test readers
# Each reader returns {row-label: excel-formula-string} for the cells that have a
# source value (missing/empty source cells are omitted -> left blank downstream).
def read_setup(folder):
    f, ws = open_src(folder)
    if not ws:
        return {}
    st = ws.title
    out = {}
    if has(ws, "B2"): out["Room Temperature (°C)"] = ext_ref(f, st, "B2")
    if has(ws, "B3"): out["Room Humidity (%)"]      = ext_ref(f, st, "B3")
    return out

SETUP_ROWS = ["Room Temperature (°C)", "Room Humidity (%)"]


def read_tono(folder):
    f, ws = open_src(folder)
    if not ws:
        return {}
    st = ws.title
    out = {}
    reads = {"Test — Reading 1": "B2", "Test — Reading 2": "C2", "Test — Reading 3": "D2",
             "Control — Reading 1": "B3", "Control — Reading 2": "C3", "Control — Reading 3": "D3"}
    for label, coord in reads.items():
        if has(ws, coord):
            out[label] = ext_ref(f, st, coord)
    if any(has(ws, c) for c in ("B2", "C2", "D2")):
        out["Test — Average (mmHg)"]    = ext_agg(f, st, "B2:D2", "AVERAGE")
    if any(has(ws, c) for c in ("B3", "C3", "D3")):
        out["Control — Average (mmHg)"] = ext_agg(f, st, "B3:D3", "AVERAGE")
    return out

TONO_ROWS = ["Test — Average (mmHg)", "Test — Reading 1", "Test — Reading 2",
             "Test — Reading 3", "Control — Average (mmHg)", "Control — Reading 1",
             "Control — Reading 2", "Control — Reading 3"]


def read_obs(folder):
    f, ws = open_src(folder)
    if not ws:
        return {}
    st = ws.title
    # Tarsal redness was row 4 until 2026-08-21; it is no longer scored (the inner
    # lid was never everted, so every image returned NA) and the row is gone from
    # the scoring template.
    cells = {"Bulbar redness — Test": "B2", "Bulbar redness — Control": "C2",
             "Limbal redness — Test": "B3", "Limbal redness — Control": "C3"}
    return {lbl: ext_ref(f, st, c) for lbl, c in cells.items() if has(ws, c)}

OBS_ROWS = ["Bulbar redness — Test", "Bulbar redness — Control",
            "Limbal redness — Test", "Limbal redness — Control"]


def read_corneal(folder):
    f, ws = open_src(folder)
    if not ws:
        return {}
    st = ws.title
    cells = {"Corneal staining — Test": "B2", "Corneal staining — Control": "C2"}
    return {lbl: ext_ref(f, st, c) for lbl, c in cells.items() if has(ws, c)}

CORNEAL_ROWS = ["Corneal staining — Test", "Corneal staining — Control"]


# sheet name -> (test-folder suffix, reader, row labels)
TESTS = [
    ("Setup",            "Setup",            read_setup,   SETUP_ROWS),
    ("Tonometry",        "Tonometry",        read_tono,    TONO_ROWS),
    ("Observation",      "Observation",      read_obs,     OBS_ROWS),
    ("Corneal Staining", "Corneal Staining", read_corneal, CORNEAL_ROWS),
]

SI_FIELDS = ["Animal # (PG)", "Order #", "Ear Tag / Tattoo", "Weight (kg)",
             "Session Date", "Session (1/2/3 - length)", "Investigator",
             "Surgeon", "Protocol #", "Account #", "Procedure",
             "Anesthesia Start", "Anesthesia End"]

MON_COLS = [("HR (bpm)", 2), ("SAP mean", 5), ("RR", 6),
            ("EtCO2", 7), ("SpO2 (%)", 8), ("Temp (F)", 13)]


def read_systemic(folder):
    """(info, mon) dicts of Excel formulas. info = live refs to Session Info
    cells; mon = MIN/AVERAGE/MAX formulas over the Monitoring source columns."""
    info, mon = {}, {}
    f = find_xlsx(folder)
    if not f:
        return info, mon
    try:
        wb = openpyxl.load_workbook(f, data_only=True, read_only=True)
    except Exception as e:
        print(f"    ! could not read {f.name}: {e}")
        return info, mon
    if "Session Info" in wb.sheetnames:
        ws = wb["Session Info"]
        rowmap = {}
        for r in range(2, ws.max_row + 1):
            k = ws.cell(r, 1).value
            if isinstance(k, str) and ws.cell(r, 2).value not in (None, ""):
                rowmap[k.strip()] = r
        for field in SI_FIELDS:
            if field in rowmap:
                info[field] = ext_ref(f, "Session Info", f"B{rowmap[field]}")
    if "Monitoring" in wb.sheetnames:
        ws = wb["Monitoring"]
        maxr = min(ws.max_row, MON_SAMPLE_ROWS)
        for name, ci in MON_COLS:
            col = get_column_letter(ci)
            has_val = any(isinstance(ws.cell(r, ci).value, (int, float))
                          for r in range(2, ws.max_row + 1))
            if not has_val:
                continue
            rng = f"{col}2:{col}{maxr}"
            mon[f"{name} — min"]  = ext_agg(f, "Monitoring", rng, "MIN")
            mon[f"{name} — mean"] = ext_agg(f, "Monitoring", rng, "AVERAGE")
            mon[f"{name} — max"]  = ext_agg(f, "Monitoring", rng, "MAX")
    return info, mon
