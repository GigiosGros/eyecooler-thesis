#!/usr/bin/env python3
"""
Oscillatory potentials (OPs) - check of the primary OP values against the device.

OPs are a PRIMARY ERG measure (no longer exploratory, since 2026-10-23). The primary
values - and every statistic on them - come from erg_cohort.py (OP_sum_uV in
erg_cohort_recordings.csv; measure "op_amp" in the effects / summary / restoration
tables), measured by erg_all.op_metrics as the ISCEV 2022 standard sets out (see
erg_all.OP_*). This script checks those values against two independent references,
read from the RETeval .rffx originals (Pig N/Reference/Pig N ERG Original Files/
RETeval originals (rff)/):

  reported   the same ISCEV measurement (erg_all.op_metrics, fixed 10-70 ms window)
             applied to the device's REPORTED waveform instead of our raw average
  device     the RETeval's own OP values (OscillatoryPotentialTotalAmplitude and the
             per-wavelet amplitudes / implicit times; the device computes them at the
             8 cd.s/m2 step only)

Steps flagged DoNotReport are skipped; a repeated step keeps the last one. Recordings
are mapped to phase / electrode / eye through each pig's Reference/erg_plan.csv.

Writes ERG Cohort Analysis/Oscillatory potentials/: erg_ops_check.csv,
OP device check.xlsx (per-recording table + agreement summary) and
fig_ops_check_flash8.png / _flash25.png (primary vs reported vs device).

    python erg_ops.py          # after erg_cohort.py
"""
from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import numpy as np

import erg_all as EA

TOOLS = Path(__file__).resolve().parent
STUDY = TOOLS.parent
COHORT = STUDY / "ERG Cohort Analysis"
OUT = COHORT / "Oscillatory potentials"


def plan_map(pig_dir):
    """{(device date, 'HH:MM'): (session, group 'Pre Jet' ...)} from erg_plan.csv."""
    f = pig_dir / "Reference" / "erg_plan.csv"
    if not f.exists():
        return {}
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out, by = {}, {}
    for r in rows:
        by.setdefault(r["Session"], []).append(r)
    for sess, rs in by.items():
        npre, nintra = int(rs[0]["NPre"]), int(rs[0]["NIntra"])
        for i, r in enumerate(sorted(rs, key=lambda r: r["DeviceTime"])):
            ph = "Pre" if i < npre else ("Intra" if i < npre + nintra else "Post")
            out[(r["ErgDate"], r["DeviceTime"].strip())] = (sess, f"{ph} {r['Electrode']}")
    return out


def steps(rffx):
    rep = ET.parse(rffx).getroot().find("Report")
    tdate = rep.findtext("TestingDate") or ""
    for t in rep.findall("Test"):
        if t.findtext("TestStepType") != "TestStep_FlashElectroretinogram":
            continue
        if (t.findtext("DoNotReport") or "").strip().lower() == "true":
            continue
        cfg = t.find("TestConfiguration")
        fl = float((cfg.findtext("GanzfeldFlashColor") or ",nan").split(",")[1])
        w = t.find("ElectrodePotentialReportWaveform")
        if w is None or not (w.findtext("DoubleStreamData") or "").strip():
            continue
        per = float(w.findtext("SamplePeriod")); off = float(w.findtext("StimulusTimeOffset") or 0)
        y = np.array(w.findtext("DoubleStreamData").split(), float) * 1e6
        t_ms = (np.arange(len(y)) * per - off) * 1e3
        num = lambda tag: [float(x) for x in (t.findtext(".//" + tag) or "").split()]   # nested; flash 8 only
        yield dict(tdate=tdate, eye=t.findtext("TestedEye"), flash=fl, t=t_ms, uv=y, per=per,
                   dev_amp=[x * 1e6 for x in num("OscillatoryPotentialAmplitudes")],
                   dev_time=[x * 1e3 for x in num("OscillatoryPotentialImplicitTimes")],
                   dev_total=((num("OscillatoryPotentialTotalAmplitude") or [None])[0] or 0) * 1e6 or None)


def primary_values():
    """{(pig, session, group, eye, flash 'g'): OP_sum_uV} from the cohort recordings table."""
    f = COHORT / "erg_cohort_recordings.csv"
    out = {}
    if not f.exists():
        return out
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("OP_sum_uV") not in (None, ""):
                out[(int(r["Pig"]), r["Session"], r["Group"], r["Eye"], f"{float(r['Flash']):g}")] = float(r["OP_sum_uV"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prim = primary_values()
    rows = []
    for pig_dir in sorted(STUDY.glob("Pig *"), key=lambda d: int(d.name.split()[1])):
        src = pig_dir / "Reference" / f"{pig_dir.name} ERG Original Files" / "RETeval originals (rff)"
        pm = plan_map(pig_dir)
        if not src.exists() or not pm:
            continue
        pig = int(pig_dir.name.split()[1])
        last = {}
        for f in sorted(src.glob("*.rffx")):
            for s in steps(f):
                if f"{s['flash']:g}" not in EA.OP_FLASHES:
                    continue
                sess, grp = pm.get((s["tdate"][:10], s["tdate"][11:16]), (None, None))
                if sess is not None:
                    last[(sess, grp, s["eye"], s["flash"])] = (f.stem, s)      # repeated step: keep the last
        for (sess, grp, eye, fl), (stem, s) in sorted(last.items()):
            role = "Test" if eye == "RightEye" else "Control"
            op = EA.op_metrics(s["t"], s["uv"], s["per"])
            row = dict(Pig=pig, Session=sess, Group=grp, Eye=role, Flash=fl, Recording=stem,
                       Primary_OP_sum_uV=prim.get((pig, sess, grp, role, f"{fl:g}")),
                       Reported_OP_sum_uV=(round(op["sum"], 1) if op and op["sum"] is not None else None),
                       Device_total_uV=(round(s["dev_total"], 1) if s["dev_total"] is not None else None))
            for k in range(EA.OP_N):
                row[f"Reported_OP{k + 1}_ms"] = round(op["times"][k], 1) if op and k < len(op["times"]) else None
                row[f"Device_OP{k + 1}_uV"] = round(s["dev_amp"][k], 1) if k < len(s["dev_amp"]) else None
                row[f"Device_OP{k + 1}_ms"] = round(s["dev_time"][k], 1) if k < len(s["dev_time"]) else None
            p = row["Primary_OP_sum_uV"]
            row["Primary_vs_reported_pct"] = (round(100 * (p / row["Reported_OP_sum_uV"] - 1), 1)
                                              if p and row["Reported_OP_sum_uV"] else None)
            row["Primary_vs_device_pct"] = (round(100 * (p / row["Device_total_uV"] - 1), 1)
                                            if p and row["Device_total_uV"] else None)
            rows.append(row)
    with open(OUT / "erg_ops_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summ = agreement(rows)
    figs = [check_figure(rows, fl) for fl in (8.0, 25.0)]
    wb = build_workbook(rows, summ, [f for f in figs if f])
    print(f"{len(rows)} OP flash steps checked -> {wb.name}")
    for r in summ:
        print(f"  flash {r['Flash']:g} vs {r['Reference']}: n {r['n']}, r {r['Pearson_r']}, "
              f"median diff {r['Median_diff_pct']} %, within 15 %: {r['Within_15pct']}")


def agreement(rows):
    out = []
    for fl in (8.0, 25.0):
        for ref, col in (("reported trace", "Reported_OP_sum_uV"), ("device", "Device_total_uV")):
            pr = [(r["Primary_OP_sum_uV"], r[col]) for r in rows
                  if r["Flash"] == fl and r["Primary_OP_sum_uV"] and r[col]]
            if len(pr) < 3:
                continue
            a, b = np.array(pr).T
            d = 100 * (a / b - 1)
            out.append(dict(Flash=fl, Reference=ref, n=len(pr), Pearson_r=round(float(np.corrcoef(a, b)[0, 1]), 3),
                            Median_diff_pct=round(float(np.median(d)), 1),
                            Median_abs_diff_pct=round(float(np.median(np.abs(d))), 1),
                            Within_15pct=f"{int((np.abs(d) <= 15).sum())} of {len(d)}"))
    return out


def check_figure(rows, fl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr = [r for r in rows if r["Flash"] == fl and r["Primary_OP_sum_uV"]]
    if not pr:
        return None
    refs = [("Reported_OP_sum_uV", "Same method on the device's reported trace (µV)")]
    if any(r["Device_total_uV"] for r in pr):
        refs.append(("Device_total_uV", "RETeval's own OP total (µV)"))
    fig, axes = plt.subplots(1, len(refs), figsize=(4.6 * len(refs), 4.2), squeeze=False)
    for ax, (col, lab) in zip(axes[0], refs):
        xy = [(r[col], r["Primary_OP_sum_uV"], r["Pig"]) for r in pr if r[col]]
        for x, y, p in xy:
            ax.plot(x, y, "o", ms=4, alpha=0.75, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                                        "#9467bd", "#8c564b"][(p - 1) % 6])
        hi = max(max(x for x, *_ in xy), max(y for _, y, _ in xy)) * 1.05 if xy else 1
        ax.plot([0, hi], [0, hi], "k--", lw=0.8)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel(lab, fontsize=8); ax.set_ylabel("Primary OP_sum (raw average, µV)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Oscillatory potentials, flash {fl:g} cd·s/m²: primary values vs references "
                 f"(dashed = identity; colour = pig)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    name = f"fig_ops_check_flash{fl:g}.png"
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    return name


def build_workbook(rows, summ, figs):
    import openpyxl
    import xlsx_style as XS
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    XS.write_table(wb, "Agreement", summ,
                   "Primary OP_sum (erg_cohort, ISCEV 2022 method on our raw average without the first flash) "
                   "against the same method on the device's reported trace, and against the RETeval's own OP "
                   "total (flash 8 only). Difference = primary / reference - 1.")
    XS.write_table(wb, "Per recording", rows,
                   "Every flash-8 and flash-25 step from the .rffx originals (DoNotReport skipped; repeated "
                   "steps: the last). Times in ms from the flash.")
    fws = wb.create_sheet("Figures")
    fws.sheet_view.showGridLines = False
    rr = 1
    for name in figs:
        fws.cell(rr, 1, name).font = XS.TITLE
        rr += 1 + XS.add_image(fws, OUT / name, f"A{rr + 1}", width_px=880)
    out = OUT / "OP device check.xlsx"
    wb.save(out)
    return out


if __name__ == "__main__":
    main()
