"""Load the COMSOL CSV exports into a run registry (notebook Steps 4a–4c, 5, 5a).

The registry is a manifest DataFrame (one row per simulation run) plus a dict of
signal frames keyed by run_id. Temperatures are in °C, time in minutes. Load order,
run ids and frame layouts are identical to the notebook's, so every downstream
table matches it row for row.
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

K = 273.15  # replaced by cfg["units"]["k_to_c"] in build_registry (same value)


# ── primitives ─────────────────────────────────────────────────────────────
def read_comsol_csv(path, skiprows=4, max_skip=8):
    """COMSOL preamble is 4 '%' lines; try that first, then other depths."""
    last_err = None
    for k in [skiprows] + [i for i in range(max_skip + 1) if i != skiprows]:
        try:
            df = pd.read_csv(path, skiprows=k, low_memory=False)
            if df.shape[1] >= 2:
                df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("% ").strip()})
                return df, k
        except Exception as e:  # noqa: BLE001 — mirror the notebook's permissive probe
            last_err = e
    raise RuntimeError(f"Could not parse {path}: {last_err}")


def read_comsol_table(path):
    """COMSOL Evaluation-Group CSV whose header is the '%' line holding Q0/alpha."""
    with open(path) as fh:
        lines = fh.readlines()
    hi = next(i for i, l in enumerate(lines) if l.startswith("%") and ("Q0" in l or "alpha" in l.lower()))
    hdr = [c.lstrip("% ").strip() for c in next(csv.reader([lines[hi]]))]
    return pd.read_csv(path, skiprows=hi + 1, header=None, names=hdr)


def k_to_c(x):
    return pd.to_numeric(x, errors="coerce") - K


def parse_variant_probe_col(col):
    """'Optic Nerve Head 1 (K), Point: …' -> ('Optic Nerve Head', 1);
    'Temperature (K), Domain Probe Eye Avg 1' -> ('Eye Avg', 1). Eye and Retina stay distinct."""
    m = re.match(r"^(.+?)\s+(\d+)\s*\(K\)", col)
    if m:
        return m.group(1).strip(), int(m.group(2))
    m = re.match(r"Temperature \(K\),\s*Domain Probe (Eye|Retina) (Avg|Max|Min)\s+(\d+)$", col)
    if m:
        return f"{m.group(1)} {m.group(2)}", int(m.group(3))
    return None, None


def parse_timeprobe_col(col):
    m = re.match(r"Time=([-+]?\d*\.?\d+)\s*min,\s*(.+)$", col)
    if not m:
        return None, None, None
    probe, var = parse_variant_probe_col(m.group(2))
    return float(m.group(1)), probe, var


def unpivot_time_x_probe(df_wide, variant_filter=None):
    """Wide sweep export (one row per sweep point, columns 'Time=t min, probe N (K)')
    -> [(coord_dict, frame indexed by time, one column per probe, °C)]."""
    coord_cols = [c for c in df_wide.columns if not c.startswith("Time=")]
    parsed = []
    for c in (c for c in df_wide.columns if c.startswith("Time=")):
        t, probe, var = parse_timeprobe_col(c)
        if probe is not None:
            parsed.append((c, t, probe, var))
    out = []
    for _, row in df_wide.iterrows():
        coord = {c: row[c] for c in coord_cols}
        rec = defaultdict(dict)
        for c, t, probe, var in parsed:
            if variant_filter is not None and var != variant_filter:
                continue
            try:
                v = float(row[c])
            except Exception:  # noqa: BLE001
                v = np.nan
            rec[t][probe] = v
        if not rec:
            continue
        times = sorted(rec)
        probes = sorted({p for r in rec.values() for p in r})
        data = np.full((len(times), len(probes)), np.nan)
        for i, t in enumerate(times):
            for j, p in enumerate(probes):
                if p in rec[t]:
                    data[i, j] = rec[t][p]
        frame = pd.DataFrame(data, index=times, columns=probes)
        frame.index.name = "Time (min)"
        out.append((coord, frame - K))
    return out


def empty_row():
    return dict(run_id=None, family=None, block=None, variant=None, mesh=None, material=None,
                silicone=None, interface=None, flow=None, T0_K=np.nan, Tb_K=np.nan, Tc_K=np.nan,
                Q0=np.nan, Tin_K=np.nan, Qh=np.nan, alpha=np.nan, signal_kind="time_series",
                channels=tuple(), file=None, exclusion_status="Included", exclusion_reason="")


def _sil_tag(s):
    return s.replace(" ", "").replace(".", "p")


# ── loaders (one per file family) ──────────────────────────────────────────
RE_RING = re.compile(r"^Ring[ _]+Comparison[ _]+Probe[ _]+\(?Stationary[ _]+Start\)?[ _]+(?P<mesh>.+)\.csv$")


def _per_variant_probe_frames(df):
    df = df.loc[:, ~df.columns.str.contains(r"\.\d+$", regex=True)]   # drop pandas '.1' duplicate blocks
    t = pd.to_numeric(df[df.columns[0]], errors="coerce")
    per_variant = defaultdict(dict)
    for c in df.columns[1:]:
        probe, var = parse_variant_probe_col(c)
        if probe is None or var not in (1, 2, 3):
            continue
        per_variant[var][probe] = k_to_c(df[c])
    for var_idx, probes in per_variant.items():
        frame = pd.DataFrame({p: s.values for p, s in probes.items()}, index=t.values)
        frame.index.name = "Time (min)"
        yield var_idx, frame


def load_ring_comparison(path, cfg):
    fname = os.path.basename(path)
    m = RE_RING.match(fname)
    if not m:
        return []
    mesh = m.group("mesh").strip()
    df, _ = read_comsol_csv(path)
    out = []
    for vi, frame in _per_variant_probe_frames(df):
        row = empty_row()
        row.update(run_id=f"scr_mesh_V{vi}_{mesh}", family="SCR", block="scr_mesh_convergence",
                   variant=f"V{vi}", mesh=mesh, channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


RE_SIL_TEMP = re.compile(r"^Ring[ _]+(?P<sil>05|1)mm[ _]+Silicone[ _]+Layer[ _]+Temperatures\.csv$", re.IGNORECASE)


def load_scr_silicone_temperatures(path, cfg):
    fname = os.path.basename(path)
    m = RE_SIL_TEMP.match(fname)
    if not m:
        return []
    silicone = "0.5 mm" if m.group("sil") == "05" else "1.0 mm"
    df, _ = read_comsol_csv(path)
    out = []
    for vi, frame in _per_variant_probe_frames(df):
        row = empty_row()
        row.update(run_id=f"scr_silicone_internal_V{vi}_{_sil_tag(silicone)}", family="SCR",
                   block="scr_silicone_internal", variant=f"V{vi}", silicone=silicone,
                   channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


RE_ENERGIES_SIL = re.compile(r"^Ring[ _]+(?P<sil>05|1)mm[ _]+silicone[ _]+layer[ _]+Energies[ _]+and[ _]+"
                             r"Thermal[ _]+Resistance[ _]+Data\.csv$", re.IGNORECASE)
RE_ENERGIES_BASE = re.compile(r"^Ring[ _]+Comparison[ _]+Energies[ _]+and[ _]+Thermal[ _]+Resistance[ _]+"
                              r"Data[ _]+Normal[ _]+Mesh\.csv$", re.IGNORECASE)
_RE_E_FLUX = re.compile(r"Normal total heat flux SCR (?P<iface>Air|Eyelids|Eye) v(?P<v>\d)\s*\((?P<unit>W/m\^2|W)\)")
_RE_E_TC = re.compile(r"Temperature T_cold v(?P<v>\d)")
_RE_E_TH = re.compile(r"Temperature T_hot v(?P<v>\d)")
_RE_E_QTEC = re.compile(r"Normal total heat flux q_TEC v(?P<v>\d)")


def load_scr_energies(path, cfg):
    """Per variant x interface: q_total (W/m^2) and q_total_W (COMSOL area-integrated, half model);
    per variant: Tcold / Thot / HeatFlux_TEC. v3's SCR-Eye normal is reversed at source, so its
    sign is flipped here (open at COMSOL source; energies use |q| so magnitudes are unaffected)."""
    fname = os.path.basename(path)
    ms, mb = RE_ENERGIES_SIL.match(fname), RE_ENERGIES_BASE.match(fname)
    if ms:
        silicone = "0.5 mm" if ms.group("sil") == "05" else "1.0 mm"
        flux_block, tr_block = "scr_silicone_flux", "scr_silicone_tr"
    elif mb:
        silicone, flux_block, tr_block = "Baseline", "scr_flux", "scr_tr"
    else:
        return []
    df, _ = read_comsol_csv(path)
    t = pd.to_numeric(df[df.columns[0]], errors="coerce")
    flux = defaultdict(dict)
    tc, th, qtec = {}, {}, {}
    for c in df.columns[1:]:
        mm = _RE_E_FLUX.search(c)
        if mm:
            flux[(mm.group("iface"), int(mm.group("v")))][mm.group("unit")] = c
            continue
        for rx, store in ((_RE_E_TC, tc), (_RE_E_TH, th), (_RE_E_QTEC, qtec)):
            mm = rx.search(c)
            if mm:
                store[int(mm.group("v"))] = c
                break
    out = []
    for (iface, vi), units in flux.items():
        frame = pd.DataFrame(index=t.values)
        frame.index.name = "Time (min)"
        sign = -1.0 if (iface == "Eye" and vi == 3) else 1.0
        if "W/m^2" in units:
            frame["q_total"] = sign * pd.to_numeric(df[units["W/m^2"]], errors="coerce").values
        if "W" in units:
            frame["q_total_W"] = sign * pd.to_numeric(df[units["W"]], errors="coerce").values
        if "q_total" not in frame.columns:
            continue
        row = empty_row()
        row.update(run_id=f"{flux_block}_V{vi}_{iface}_{_sil_tag(silicone)}", family="SCR", block=flux_block,
                   variant=f"V{vi}", interface=iface, silicone=silicone,
                   channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    for vi in sorted(set(tc) | set(th) | set(qtec)):
        frame = pd.DataFrame(index=t.values)
        frame.index.name = "Time (min)"
        if vi in tc:
            frame["Tcold"] = k_to_c(df[tc[vi]]).values
        if vi in th:
            frame["Thot"] = k_to_c(df[th[vi]]).values
        if vi in qtec:
            frame["HeatFlux_TEC"] = pd.to_numeric(df[qtec[vi]], errors="coerce").values
        if frame.shape[1] == 0:
            continue
        row = empty_row()
        row.update(run_id=f"{tr_block}_V{vi}_{_sil_tag(silicone)}", family="SCR", block=tr_block,
                   variant=f"V{vi}", silicone=silicone, signal_kind="scalar_ts",
                   channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


def load_wb_fine_energies(path, cfg):
    """Fine Energies v{n}.csv: operating point + HotSide_W = |alpha=1 hot-side flux (W)|."""
    fname = os.path.basename(path)
    m = re.match(r"Fine[ _]Energies[ _]v(\d)\.csv", fname, re.IGNORECASE)
    if not m:
        return []
    df, _ = read_comsol_csv(path)
    keys = ["Q0 (m^3/s)", "Tin (K)", "T0 (K)", "Qh (W/m^2)"]
    if not all(k in df.columns for k in keys):
        return []
    hot = [c for c in df.columns if c.startswith("alpha=1,") and c.endswith("(W)") and "wet" not in c.lower()]
    if not hot:
        return []
    frame = df[keys].copy()
    for k in keys:
        frame[k] = pd.to_numeric(frame[k], errors="coerce")
    frame["HotSide_W"] = pd.to_numeric(df[hot[0]], errors="coerce").abs()
    frame = frame.dropna().reset_index(drop=True)
    if frame.empty:
        return []
    row = empty_row()
    row.update(run_id=f"wb_fine_energies_V{m.group(1)}", family="Waterblock", block="wb_fine_energies",
               variant=f"V{m.group(1)}", signal_kind="steady_state", file=fname)
    return [(row, frame)]


def load_scr_material_sweep(path, cfg):
    fname = os.path.basename(path)
    m = re.match(r"SCR[ _]v(\d)[ _]material[ _]sweep\.csv", fname)
    if not m:
        return []
    variant = f"V{m.group(1)}"
    df, _ = read_comsol_csv(path)
    mats = {int(k): v for k, v in cfg["vocabulary"]["materials"].items()}
    out = []
    for coord, frame in unpivot_time_x_probe(df, variant_filter=int(m.group(1))):
        try:
            sw = int(round(float(list(coord.values())[0])))
        except Exception:  # noqa: BLE001
            sw = None
        material = mats.get(sw, f"switch_{sw}")
        row = empty_row()
        row.update(run_id=f"scr_material_{variant}_{material.replace(' ', '')}", family="SCR",
                   block="scr_material", variant=variant, material=material,
                   channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


def load_scr_parametric(path, cfg):
    fname = os.path.basename(path)
    m = re.match(r"SCR[ _]v(\d)[ _]Parametric[ _]Sweep\.csv", fname)
    if not m:
        return []
    variant = f"V{m.group(1)}"
    df, _ = read_comsol_csv(path)
    out = []
    for coord, frame in unpivot_time_x_probe(df, variant_filter=int(m.group(1))):
        cv = list(coord.values())
        try:
            T0, Tb, Tc = (float(cv[i]) for i in range(3))
        except Exception:  # noqa: BLE001
            T0 = Tb = Tc = np.nan
        row = empty_row()
        row.update(run_id=f"scr_param_{variant}_T0={T0:.1f}_Tb={Tb:.1f}_Tc={Tc:.1f}".replace(" ", ""),
                   family="SCR", block="scr_parametric", variant=variant, T0_K=T0, Tb_K=Tb, Tc_K=Tc,
                   channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


RE_REHEAT_FILE = re.compile(r"^SCR[ _]v(\d+)[ _]ReHeat\.csv$", re.IGNORECASE)


def _reheat_channel_name(col):
    c = str(col)
    m = re.search(r"Domain Probe (Eye|Retina) (Avg|Max|Min)", c)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = re.match(r"^(.+?)\s+\d+\s*\(K\)", c)
    return m.group(1).strip() if m else c


def load_scr_reheat(path, cfg):
    """Long-format reheat export -> one run per (T0, Tb, Tc)."""
    fname = os.path.basename(path)
    m = RE_REHEAT_FILE.match(fname)
    if not m:
        return []
    variant = f"V{int(m.group(1))}"
    df, _ = read_comsol_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    cols = list(df.columns)
    find = lambda key: next((c for c in cols if str(c).strip().lower().startswith(key.lower())), None)  # noqa: E731
    c_T0, c_Tb, c_Tc, c_t = find("T0"), find("Tb"), find("Tc"), find("Time")
    if None in (c_T0, c_Tb, c_Tc, c_t):
        return []
    probe_cols = [c for c in cols if c not in (c_T0, c_Tb, c_Tc, c_t)]
    names = {c: _reheat_channel_name(c) for c in probe_cols}
    out = []
    for (T0, Tb, Tc), sub in df.groupby([df[c_T0].round(3), df[c_Tb].round(3), df[c_Tc].round(3)]):
        sub = sub.sort_values(c_t)
        t = pd.to_numeric(sub[c_t], errors="coerce").to_numpy(dtype=float)
        frame = pd.DataFrame({names[c]: pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float) - K
                              for c in probe_cols}, index=t)
        frame.index.name = "Time (min)"
        row = empty_row()
        row.update(run_id=f"scr_reheat_{variant}_T0={float(T0):.1f}_Tb={float(Tb):.1f}_Tc={float(Tc):.1f}".replace(" ", ""),
                   family="SCR", block="scr_reheat", variant=variant, T0_K=float(T0), Tb_K=float(Tb),
                   Tc_K=float(Tc), channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


def load_wb_aux(path, cfg):
    """V{n}_Auxiliary_{Coarse,Normal}.csv — single operating point, probes vs alpha."""
    fname = os.path.basename(path)
    m = re.match(r"V(\d)_Auxiliary_(Coarse|Normal)\.csv", fname)
    if not m:
        return []
    df, _ = read_comsol_csv(path)
    acol = df.columns[0]
    rename = {}
    for c in df.columns[1:]:
        for p in ("TEC", "INw", "OUTw", "Waterblock"):
            if p in c:
                rename[c] = p
                break
    frame = df.rename(columns={**rename, acol: "alpha"})
    frame = frame[["alpha"] + [v for v in rename.values() if v in frame.columns]]
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    for p in cfg["probes"]["waterblock"]:
        if p in frame.columns:
            frame[p] = k_to_c(frame[p])
    frame = frame.set_index("alpha").sort_index()
    frame.index.name = "alpha"
    row = empty_row()
    row.update(run_id=f"wb_aux_V{m.group(1)}_{m.group(2)}", family="Waterblock", block="wb_aux",
               variant=f"V{m.group(1)}", mesh=m.group(2), signal_kind="steady_state",
               channels=tuple(frame.columns), file=fname)
    return [(row, frame)]


_WB_KEYS = ["Q0 (m^3/s)", "Tin (K)", "T0 (K)", "Qh (W/m^2)"]


def _wb_rid_coords(coord):
    return (float(coord.get("Q0 (m^3/s)", np.nan)), float(coord.get("Tin (K)", np.nan)),
            float(coord.get("T0 (K)", np.nan)), float(coord.get("Qh (W/m^2)", np.nan)))


def load_wb_aux_fine_sweep(path, cfg):
    """Fine mesh: every (Q0, Tin, T0, Qh) is one run holding probes vs alpha."""
    fname = os.path.basename(path)
    m = re.match(r"V(\d)_Auxiliary_Fine_Sweep\.csv", fname)
    if not m:
        return []
    variant = f"V{m.group(1)}"
    df, _ = read_comsol_csv(path)
    rename = {}
    for c in df.columns:
        for p in ("TEC", "INw", "OUTw", "Waterblock"):
            if p in c and "Boundary Probe" in c:
                rename[c] = p
                break
    df = df.rename(columns=rename)
    probes = cfg["probes"]["waterblock"]
    for p in probes:
        if p in df.columns:
            df[p] = k_to_c(df[p])
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    coord_cols = [c for c in _WB_KEYS if c in df.columns]
    if not coord_cols:
        return []
    out = []
    for keys, g in df.groupby(coord_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        Q0, Tin, T0, Qh = _wb_rid_coords(dict(zip(coord_cols, keys)))
        frame = g.set_index("alpha")[[p for p in probes if p in g.columns]].sort_index()
        frame.index.name = "alpha"
        row = empty_row()
        row.update(run_id=f"wb_aux_{variant}_Fine_Q0={Q0:.2e}_Tin={Tin:.1f}_T0={T0:.1f}_Qh={Qh:.1f}".replace(" ", ""),
                   family="Waterblock", block="wb_aux", variant=variant, mesh="Fine",
                   Q0=Q0, Tin_K=Tin, T0_K=T0, Qh=Qh, signal_kind="steady_state",
                   channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


def load_wb_sweep(path, cfg):
    """V{n}_SWEEP_(Normal_Turbulent).csv — transient; one run per operating point."""
    fname = os.path.basename(path)
    m = re.match(r"V(\d)[ _]SWEEP[ _]\(?Normal[ _]Turbulent\)?\.csv", fname)
    if not m:
        return []
    variant = f"V{m.group(1)}"
    df, _ = read_comsol_csv(path)
    rename = {}
    for c in df.columns:
        if c.startswith("Time"):
            rename[c] = "Time (min)"
        else:
            for p in ("TEC", "INw", "OUTw", "Waterblock"):
                if p in c:
                    rename[c] = p
                    break
    df = df.rename(columns=rename)
    coord_cols = [c for c in _WB_KEYS if c in df.columns]
    if not coord_cols:
        return []
    probes = cfg["probes"]["waterblock"]
    out = []
    for keys, g in df.groupby(coord_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        Q0, Tin, T0, Qh = _wb_rid_coords(dict(zip(coord_cols, keys)))
        frame = g.set_index("Time (min)")[[p for p in probes if p in g.columns]]
        frame.index = pd.to_numeric(frame.index, errors="coerce")
        for p in probes:
            if p in frame.columns:
                frame[p] = k_to_c(frame[p])
        row = empty_row()
        row.update(run_id=f"wb_sweep_{variant}_Q0={Q0:.2e}_Tin={Tin:.1f}_T0={T0:.1f}_Qh={Qh:.1f}".replace(" ", ""),
                   family="Waterblock", block="wb_sweep", variant=variant, mesh="Normal", flow="Turbulent",
                   Q0=Q0, Tin_K=Tin, T0_K=T0, Qh=Qh, channels=tuple(frame.columns), file=fname)
        out.append((row, frame))
    return out


# Dispatch table, in the notebook's order (first match wins).
LOADER_DISPATCH = [
    (re.compile(r"^Ring[ _]+Comparison[ _]+Probe[ _]+\(?Stationary[ _]+Start\)?[ _]+.+\.csv$"), load_ring_comparison),
    (re.compile(r"^Ring[ _]+(05|1)mm[ _]+Silicone[ _]+Layer[ _]+Temperatures\.csv$", re.IGNORECASE), load_scr_silicone_temperatures),
    (re.compile(r"^Ring[ _]+.*Energies[ _]+and[ _]+Thermal[ _]+Resistance[ _]+Data.*\.csv$", re.IGNORECASE), load_scr_energies),
    (re.compile(r"^Fine[ _]Energies[ _]v\d\.csv$", re.IGNORECASE), load_wb_fine_energies),
    (re.compile(r"^SCR[ _]v\d[ _]material[ _]sweep\.csv$"), load_scr_material_sweep),
    (re.compile(r"^SCR[ _]v\d[ _]Parametric[ _]Sweep\.csv$"), load_scr_parametric),
    (RE_REHEAT_FILE, load_scr_reheat),
    (re.compile(r"^V\d[ _]Auxiliary[ _](Coarse|Normal)\.csv$"), load_wb_aux),
    (re.compile(r"^V\d[ _]Auxiliary[ _]Fine[ _]Sweep\.csv$"), load_wb_aux_fine_sweep),
    (re.compile(r"^V\d[ _]SWEEP[ _]\(?Normal[ _]Turbulent\)?\.csv$"), load_wb_sweep),
]
# Single-variant TEC fragments superseded by the Energies file; deliberately not loaded.
SUPERSEDED = re.compile(r"^(?:V\d[ _])?(Tcold|Thot|HeatFlux[ _]Trough[ _]TEC)\.csv$", re.IGNORECASE)
TIME_SERIES_KINDS = {"time_series", "scalar_ts"}


@dataclass
class Registry:
    manifest: pd.DataFrame
    signals: dict
    data_dir: str
    skipped: list = field(default_factory=list)
    n_truncated: int = 0
    files: list = field(default_factory=list)


def build_registry(data_dir: str, cfg: dict) -> Registry:
    global K
    K = cfg["units"]["k_to_c"]
    tmax = float(cfg["units"]["max_time_min"])
    rows, signals, skipped, n_trunc = [], {}, [], 0
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    for f in files:
        loader = next((ldr for rgx, ldr in LOADER_DISPATCH if rgx.match(f)), None)
        if loader is None:
            skipped.append((f, "superseded by the Ring_Comparison Energies file" if SUPERSEDED.match(f)
                            else "no matching loader"))
            continue
        path = os.path.join(data_dir, f)
        # Corrected ring energy tables (from COMSOL's stored boundary fluxes; see
        # Model_data/stored_flux_energies/README.md) replace the exports of the same name.
        alt = os.path.join(data_dir, "stored_flux_energies", f)
        if loader is load_scr_energies and cfg["corrections"].get("use_stored_flux_energies", False) and os.path.exists(alt):
            path = alt
        try:
            results = loader(path, cfg)
        except Exception as e:  # noqa: BLE001 — report, don't abort (as the notebook does)
            skipped.append((f, f"loader failed: {type(e).__name__}: {e}"))
            continue
        if not results:
            skipped.append((f, "loader returned 0 runs"))
            continue
        for row, frame in results:
            rid = row["run_id"]
            if rid in signals:
                k = 2
                while f"{rid}#{k}" in signals:
                    k += 1
                rid = row["run_id"] = f"{rid}#{k}"
            if row.get("signal_kind") in TIME_SERIES_KINDS:
                n0 = len(frame)
                frame = frame.loc[frame.index <= tmax + 1e-6]
                n_trunc += len(frame) < n0
            signals[rid] = frame
            rows.append(row)
    return Registry(pd.DataFrame(rows), signals, data_dir, skipped, n_trunc, files)


def integrity_audit(reg: Registry, nan_warn_frac=0.5) -> pd.DataFrame:
    """Step 5a: one row per run with an empty frame, an all-NaN column or >= 50 % NaN.
    An all-NaN TEC column on a waterblock Auxiliary mesh run is expected (no TEC probe there)."""
    rows = []
    for _, r in reg.manifest.iterrows():
        fr = reg.signals[r.run_id]
        if fr is None or fr.size == 0:
            rows.append(dict(run_id=r.run_id, issue="empty frame", columns="", expected=False))
            continue
        empty = [c for c in fr.columns if fr[c].isna().all()]
        if empty:
            expected = str(r.run_id).startswith("wb_aux") and empty == ["TEC"]
            rows.append(dict(run_id=r.run_id, issue="all-NaN column", columns=", ".join(empty), expected=expected))
        frac = float(fr.isna().to_numpy().mean())
        if frac >= nan_warn_frac:
            rows.append(dict(run_id=r.run_id, issue=f"{frac:.0%} NaN", columns="", expected=False))
    return pd.DataFrame(rows, columns=["run_id", "issue", "columns", "expected"])


def manifest_table(reg: Registry) -> pd.DataFrame:
    out = reg.manifest.copy()
    out["channels"] = out["channels"].apply(lambda t: ", ".join(t) if isinstance(t, tuple) else str(t))
    return out


def filter_manifest(manifest, **filters):
    m = manifest
    for k, v in filters.items():
        if v is None or v == "All" or k not in m.columns:
            continue
        m = m[m[k].isin(list(v))] if isinstance(v, (list, tuple, set)) else m[m[k] == v]
    return m
