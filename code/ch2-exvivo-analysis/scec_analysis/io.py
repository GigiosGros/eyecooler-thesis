"""Discover and load the ex-vivo recordings.

File conventions (one flat data folder):
  Ex_E<n>_<ring>_LM_<date>.xlsx      human run; ring A (aluminum original, A2 = repeat),
                                     LB (aluminum reduced-bulk), C (ceramic original)
  Calibration_HE<n>_LM_<date>.xlsx   human per-eye calibration plateaus (37 / 20.5 / 4 degC)
  P_E<n>_<ring>_LM_<date>.csv        human intraocular-pressure sidecar
  Fresh_Eye_<date>_<seq>.xlsx        fresh porcine run (aluminum original ring)
  Preserved_Eye_<date>_<seq>.xlsx    preserved porcine run (aluminum original ring)
  H<n>/F<seq>/P<seq>.mrk.json        co-registered micro-CT sensor positions
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openpyxl import load_workbook

HUMAN_RUN_RE = re.compile(r"^Ex_E(?P<eye>\d+)_(?P<ring>A2?|LB|C)_LM(?:_\d+)?\.xlsx$", re.I)
HUMAN_CAL_RE = re.compile(r"^Calibration_HE(?P<eye>\d+)_LM(?:_\d+)?\.xlsx$", re.I)
PRESSURE_RE = re.compile(r"^P_E(?P<eye>\d+)_(?P<ring>A2?|LB|C)_LM(?:_\d+)?\.csv$", re.I)
PIG_RE = re.compile(r"^(?P<status>Fresh|Preserved)_Eye_\d{6}_(?P<seq>\d+)\.xlsx$", re.I)
MARKER_RE = re.compile(r"^(?P<grp>[HPF])(?P<eye>\d+)(?:\.mrk|_mrk)\.json$", re.I)

RING_META = {  # ring code in the file name -> (material, geometry, display code)
    "A": ("Aluminum", "Original", "A"),
    "LB": ("Aluminum", "Reduced Bulk", "RB"),
    "C": ("Ceramic", "Original", "C"),
}
MARKER_GROUP = {"H": ("Human", "Fresh"), "F": ("Porcine", "Fresh"), "P": ("Porcine", "Preserved")}


@dataclass
class Run:
    run_id: str
    species: str
    status: str
    ring_material: str
    ring: str
    ring_code: str
    eye: int
    file: str
    trial: str = "A"
    source_seq: int | None = None          # porcine file sequence number (marker link)
    room_temp_C: float | None = None
    volts: pd.DataFrame | None = None      # raw acquisition, V (current in A for porcine)
    temp: pd.DataFrame | None = None       # calibrated, degC (+ current, A)
    median: pd.DataFrame | None = None     # median-filtered temperature
    filtered: pd.DataFrame | None = None   # median + onset-preserving low-pass
    fit: dict = field(default_factory=dict)
    exclusions: dict = field(default_factory=dict)
    also_broken: dict = field(default_factory=dict)
    pressure: pd.DataFrame | None = None
    pressure_offsets: dict = field(default_factory=dict)

    @property
    def eye_id(self) -> str:
        return f"E{self.eye}"

    @property
    def group(self) -> tuple:
        return (self.species, self.status, self.ring_material, self.ring)


def group_label(key: tuple) -> str:
    species, status, material, ring = key
    return f"{species} {status} · {material} {ring}"


def read_human_xlsx(path: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_excel(path)
    names = list(df.columns)
    names[: len(columns)] = columns[: len(names)]
    df.columns = names
    return df.set_index("Relative Time")


def _room_temp(cell) -> float | None:
    m = re.search(r"([\d.]+)\s*[°º]?\s*C", str(cell)) if cell is not None else None
    return float(m.group(1)) if m else None


def read_pig_xlsx(path: str, temp_channels: list[str], current: str) -> tuple[pd.DataFrame, float | None]:
    """Porcine workbook: row 0 sensor names, row 1 'Volt (V)' / 'Temp (ºC)' sub-headers.
    Returns the sensor VOLTAGES (temperature is recomputed from them) plus the
    pre-calibrated TEC current in A, which the acquisition stored in the
    'Temp' column under 'Current for TEC'."""
    wb = load_workbook(path, data_only=True, read_only=True)
    ws = wb["Sheet1"] if "Sheet1" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header, sub, data = rows[0] or (), rows[1] or (), rows[2:]
    rename = {"ECR": "SCR"}
    volt_cols, name, current_col = {}, None, None
    for j, s in enumerate(sub):
        if j < len(header) and header[j] is not None and str(header[j]).strip():
            name = str(header[j]).strip()
        s = str(s).strip() if s else ""
        if name and "Current" in name and s and not s.startswith("Volt"):
            current_col = j          # sub-header 'Current (A)' (older files: 'Temp')
        elif name and s.startswith("Volt"):
            volt_cols[rename.get(name, name)] = j
    t_idx = next((j for j, s in enumerate(sub) if s and ("Relative" in str(s) or s == "Seconds")), 0)

    def col(j):
        return np.array([(r[j] if j < len(r) else np.nan) for r in data], dtype=float)

    t = col(t_idx)
    out = {ch: col(j) for ch, j in volt_cols.items() if ch in temp_channels}
    if current_col is not None:
        out[current] = col(current_col)
    return pd.DataFrame(out, index=t), _room_temp(rows[0][0])


def read_pressure_csv(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["Seconds"], errors="coerce").notna()].copy()
    df["Seconds"] = pd.to_numeric(df["Seconds"], errors="coerce")
    df = df.set_index("Seconds").rename(columns={"PSI_A": "Inflow", "PSI_B": "Outflow"})
    offsets = {}
    with open(path) as f:
        for line in f.readlines()[-30:]:
            m = re.search(r"Offset\s+([AB])\s*:\s*([\d.+-]+)", line, re.I)
            if m:
                offsets[m.group(1).upper()] = float(m.group(2))
    return df[[c for c in ("Inflow", "Outflow") if c in df.columns]], offsets


def read_markers(path: str, ids: dict[str, str], key_by: str = "id") -> dict[str, np.ndarray]:
    """{channel: xyz}. key_by = "label": the sensor number is the trailing number of the
    label the operator gave the point in Slicer (H8-3 -> 3), falling back to the internal
    control-point id and then to the list position. key_by = "id" (notebook v4.28) tries
    the internal id first. The two differ only in H8.mrk.json (ids 2 and 3 carry the
    labels H8-3 and H8-2)."""
    with open(path) as f:
        points = json.load(f)["markups"][0]["controlPoints"]
    out = {}
    for k, cp in enumerate(points, start=1):
        m = re.search(r"-(\d+)$", str(cp.get("label", "")))
        by_label = m.group(1) if m and m.group(1) in ids else None
        by_id = str(cp["id"]) if cp.get("id") is not None and str(cp["id"]) in ids else None
        key = (by_label or by_id) if key_by == "label" else (by_id or by_label)
        key = key or str(k)
        if key in ids:
            out[ids[key]] = np.asarray(cp["position"], dtype=float)
    return out


def load_dataset(data_dir: str, cfg: dict):
    """Returns (runs, calibration_files, markers).

    runs        {run_id: Run} after de-duplication (a repeated human run keeps the
                last trial collected, e.g. E1_A2 replaces E1_A)
    calibration {eye_id: DataFrame of calibration plateaus, V}
    markers     {(species, status): {marker_eye: {channel: xyz}}}
    """
    ch_cfg = cfg["channels"]
    temp_ch, current = ch_cfg["temperature"], ch_cfg["current"]
    columns = cfg["acquisition"]["human_columns"]
    files = sorted(os.listdir(data_dir))
    runs, calibration, markers = {}, {}, {}
    pigs = {"Fresh": [], "Preserved": []}

    for fname in files:
        path = os.path.join(data_dir, fname)
        if m := HUMAN_RUN_RE.match(fname):
            eye, code = int(m.group("eye")), m.group("ring").upper()
            base = code.rstrip("0123456789") or code
            material, geometry, display = RING_META[base]
            rid = f"H_Fresh_E{eye}_{code}"
            runs[rid] = Run(rid, "Human", "Fresh", material, geometry, display, eye, path,
                            trial=code if code != base else "A",
                            volts=read_human_xlsx(path, columns))
        elif m := HUMAN_CAL_RE.match(fname):
            calibration[f"E{int(m.group('eye'))}"] = read_human_xlsx(path, columns)
        elif m := PIG_RE.match(fname):
            pigs[m.group("status").capitalize()].append((int(m.group("seq")), path))
        elif m := MARKER_RE.match(fname):
            key = MARKER_GROUP[m.group("grp").upper()]
            pts = read_markers(path, cfg["markers"]["ids"], cfg["markers"].get("key_by", "id"))
            if pts:
                markers.setdefault(key, {})[f"E{int(m.group('eye'))}"] = pts

    # Porcine eyes are numbered 1..N per status in file-sequence order.
    for status, items in pigs.items():
        for eye, (seq, path) in enumerate(sorted(items), start=1):
            volts, room = read_pig_xlsx(path, temp_ch, current)
            rid = f"P_{status}_E{eye}_A"
            runs[rid] = Run(rid, "Porcine", status, "Aluminum", "Original", "A", eye, path,
                            source_seq=seq, room_temp_C=room, volts=volts)

    # De-duplicate: one run per (eye, ring); keep the last trial collected.
    buckets = {}
    for rid, r in runs.items():
        buckets.setdefault((r.species, r.status, r.eye, r.ring_code), []).append(rid)
    for rids in buckets.values():
        for rid in sorted(rids)[:-1]:
            runs.pop(rid)

    # Attach human pressure sidecars.
    for fname in files:
        if m := PRESSURE_RE.match(fname):
            rid = f"H_Fresh_E{int(m.group('eye'))}_{m.group('ring').upper()}"
            if rid in runs:
                runs[rid].pressure, runs[rid].pressure_offsets = read_pressure_csv(
                    os.path.join(data_dir, fname))
    return runs, calibration, markers
