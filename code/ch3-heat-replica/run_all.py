"""Re-solve every scleral-contact-ring COMSOL study with the independent replica and compare.

    python run_all.py                         # representative subset of the sweeps
    python run_all.py --full                  # every sweep case (about 2 h on 5 workers)
    python run_all.py --studies ring_comparison mesh_study
    python run_all.py --figures-only          # redraw figures from stored results

Studies: ring_comparison, mesh_study, material_sweep, silicone,
parametric_sweep, reheat.  Each case reads its model definition and mesh from
the .mph archive, solves the stationary start and the transient(s), and is
compared with the probe table COMSOL exported.
"""
import os

os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import argparse
import csv
import datetime
import itertools
import json
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = os.path.normpath(os.path.join(HERE, "..", "..", "Computational model"))
LOCAL = r"D:\COMSOL\Projects\SCR"          # local copies of the large files, used when present
REPORTS = os.path.join(BOX, "Reports", "SCR")


# Local copies that are NOT the same problem as the Box originals (older saves: different meshes or
# solutions).  Checked with dev/compare_model_definitions.py on 2026-10-07: the parametric-sweep
# copies define the same problem as Box (same meshes, parameters, features, materials); the
# ring-comparison copy is the December 2025 model (older meshes, solved to 60 min), not the one
# re-solved in May 2026 that the report tables come from.
STALE_LOCAL = ("Silicone", "Ring Comparison")


def mph_path(rel):
    """The model file in Box, or its local copy where that is known to define the same problem."""
    a = os.path.join(LOCAL, rel)
    if os.path.exists(a) and not rel.startswith(STALE_LOCAL):
        return a
    return os.path.join(BOX, "Projects", "SCR", rel)


# ----------------------------------------------------------------- reference
def norm(name):
    """Common key for a probe name in the model tree and in an exported table."""
    s = name.lower().replace("anteriror", "anterior")
    s = re.sub(r"^time=[\d.e+-]+ min,", "", s).strip()
    s = re.sub(r"^temperature \(k\),", "", s)
    s = re.sub(r"\(k\).*$", "", s)
    s = s.replace("temperature", "").replace("domain probe", "").replace("probe", "").replace(",", " ")
    return " ".join(s.split())


def _table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = list([r for r in rows if r and r[0].startswith("%")][-1])
    head[0] = head[0].lstrip("% ")
    data = np.array([[float(x) if x not in ("", "NaN") else np.nan for x in r] for r in rows
                     if r and not r[0].startswith("%")])
    return head, data


def ref_long(path, nkey=0, key=None):
    """Table with a Time column (after nkey parameter columns) -> (t, {probe: values})."""
    head, data = _table(path)
    if nkey:
        data = data[np.all(np.isclose(data[:, :nkey], key, atol=1e-6), axis=1)]
    return data[:, nkey], {norm(h): data[:, i] for i, h in enumerate(head) if i > nkey}


def ref_wide(path, nkey, key):
    """Sweep table: one row per case, columns 'Time=t min, probe' -> (t, {probe: values})."""
    head, data = _table(path)
    row = data[np.all(np.isclose(data[:, :nkey], key, atol=1e-6), axis=1)]
    if len(row) != 1:
        return None, {}
    out = {}
    for i, h in enumerate(head):
        m = re.match(r"Time=([\d.eE+-]+) min, (.*)$", h)
        if m:
            out.setdefault(norm(m.group(2)), []).append((float(m.group(1)), row[0, i]))
    t = np.array([a for a, _ in next(iter(out.values()))])
    return t, {k: np.array([b for _, b in v]) for k, v in out.items()}


def ref_keys(path, nkey):
    _, data = _table(path)
    return sorted({tuple(np.round(r[:nkey], 6)) for r in data})


# --------------------------------------------------------------------- cases
def _subset(keys, study):
    """Representative part of a temperature sweep: the nominal point and the
    two opposite extremes of the (T0, Tb, Tc) box; for reheat, the nominal point."""
    nominal = lambda k: np.allclose(k, (298.15, 310.15, 277.15))
    if study == "reheat":
        return [k for k in keys if nominal(k)]
    lo = tuple(min(k[i] for k in keys) for i in range(3))
    hi = tuple(max(k[i] for k in keys) for i in range(3))
    return [k for k in keys if nominal(k) or np.allclose(k, lo) or np.allclose(k, hi)]


def build_cases(studies, full=False):
    cases = []

    def add(study, cid, rel, comp, ref, **kw):
        cases.append(dict(study=study, id=cid, mph=mph_path(rel), comp=comp, ref=ref, **kw))

    if "ring_comparison" in studies:
        # the table exported from the Box file's own solution (May 2026); the older local table
        # "... 01step.csv" belongs to the December 2025 model and is not used
        ref = os.path.join(REPORTS, "Ring Comparison", "Ring Comparison Probe (Stationary Start) Normal.csv")
        for c in ("comp1", "comp2", "comp3"):
            add("ring_comparison", f"v{c[-1]}", r"Ring Comparison\Ring Comparison.mph", c, ("long", ref, 0, None))
    if "mesh_study" in studies:
        for mesh in ("Coarse", "Coarser"):
            ref = os.path.join(REPORTS, "Ring Comparison", f"Ring Comparison Probe (Stationary Start) {mesh}.csv")
            for c in ("comp1", "comp2", "comp3"):
                add("mesh_study", f"{mesh.lower()}_v{c[-1]}", rf"Ring Comparison\Ring Comparison Mesh Study {mesh}.mph",
                    c, ("long", ref, 0, None))
    if "silicone" in studies:
        for label, fn, rn in (("0.5mm", "Ring 0.5mm silicone layer.mph", "Ring 05mm Silicone Layer Temperatures.csv"),
                              ("1mm", "Ring 1mm silicone layer.mph", "Ring 1mm Silicone Layer Temperatures.csv")):
            for c in ("comp1", "comp2", "comp3"):
                add("silicone", f"{label}_v{c[-1]}", rf"Silicone\{fn}", c,
                    ("long", os.path.join(REPORTS, "Silicone Study", rn), 0, None))
    for v in (1, 2, 3):
        c = f"comp{v}"
        if "material_sweep" in studies:
            ref = os.path.join(REPORTS, "Material Sweep", f"SCR v{v} material sweep.csv")
            for (idx,) in ref_keys(ref, 1):
                add("material_sweep", f"v{v}_material{int(idx)}", rf"Material Sweep\Ring SCR v{v} material sweep.mph", c,
                    ("wide", ref, 1, (idx,)), switch_index=int(idx) - 1)
        if "parametric_sweep" in studies:
            ref = os.path.join(REPORTS, "Parametric Sweep", f"SCR v{v} Parametric Sweep.csv")
            for key in (ref_keys(ref, 3) if full else _subset(ref_keys(ref, 3), "parametric_sweep")):
                add("parametric_sweep", "v%d_T0_%g_Tb_%g_Tc_%g" % ((v,) + key),
                    rf"Parametric Sweep\Ring SCR v{v} parametric sweep.mph", c, ("wide", ref, 3, key),
                    overrides=dict(zip(("T0", "Tb", "Tc"), key)))
        if "reheat" in studies:
            ref = os.path.join(REPORTS, "ReHeat", f"SCR v{v} ReHeat.csv")
            for key in (ref_keys(ref, 3) if full else _subset(ref_keys(ref, 3), "reheat")):
                add("reheat", "v%d_T0_%g_Tb_%g_Tc_%g" % ((v,) + key), rf"ReHeat\Ring SCR v{v} ReHeat.mph", c,
                    ("long", ref, 3, key), overrides=dict(zip(("T0", "Tb", "Tc"), key)), reheat=True)
    return cases


# ---------------------------------------------------------------------- solve
def stages(t_end):
    """6 s steps (COMSOL's own step) for the first 5 min, 15 s to 15 min, 30 s afterwards."""
    return [(min(300.0, t_end), 6.0), (min(900.0, t_end), 15.0), (t_end, 30.0)]


def run_case(case, out_root):
    if case.get("variant"):
        out_root = os.path.join(out_root, case["variant"])
    from scipy.interpolate import PchipInterpolator
    from replica import model, solve
    from replica.mphread import Mph
    t0 = time.time()
    out_dir = os.path.join(out_root, case["study"], case["id"])
    os.makedirs(out_dir, exist_ok=True)
    log = []
    mph = Mph(case["mph"])
    info = mph.components()[case["comp"]]
    import sensor_probes
    extra = sensor_probes.sensor_probes()
    prob = model.build(mph, component=case["comp"], heat=info["heat"], geom=info["geom"],
                       overrides=case.get("overrides"), switch_index=case.get("switch_index", 0),
                       extra_probes=extra, insulation_slit=not case.get("no_slit"))
    extra.update(sensor_probes.snap_to_sclera(prob, extra))
    prob.probes.update(extra)
    log += [f"note: {n}" for n in prob.notes]
    h = solve.Heat(prob, log=log.append, flow=bool(case.get("flow")))
    located = h.probe_setup()
    names = list(prob.probes)
    with open(os.path.join(out_dir, "extra_probe_points.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["probe", "x_mm", "z_mm", "model_domain", "moved_to_reach_sclera_mm"])
        for n in extra:
            w.writerow([n, round(extra[n]["xyz"][0] * 1e3, 3), round(extra[n]["xyz"][2] * 1e3, 3),
                        prob.domain_names[located[n][3]], round(extra[n].get("moved_mm", 0.0), 3)])
    steps = mph.study_steps()
    stat = next(k for k, v in steps.items() if v["op"] == "Stationary")
    trans = [k for k, v in steps.items() if v["op"] == "Transient"]
    T = h.stationary(step=stat)

    # surface temperatures at COMSOL's output times, for the viewer
    snap = case["study"] in ("ring_comparison", "silicone", "reheat")
    surf = np.unique(prob.tris)
    snaps = {}

    def march(T, step, release=False):
        rows = []
        tl = np.array([float(x) for x in steps[step]["tlist"].split()]) * 60.0
        t_end = tl[-1]

        def cb(t, T):
            rows.append([t / 60.0] + [h.probes(T)[n] for n in names])
            if snap and np.abs(tl - t).min() < 1e-6:
                snaps.setdefault(step, []).append((t / 60.0, T[surf].astype(np.float32)))

        st = stages(t_end)
        if release and os.environ.get("SCR_RELEASE_STEPS") == "1":
            # (2026-10-22) short steps across the release, as dev/new_runs.py release_steps;
            # opt-in, so the default runs are unchanged
            st = [(6.0, 0.5), (18.0, 1.0), (60.0, 3.0)] + [(tu, dt) for tu, dt in st if tu > 60.0]
        T = h.transient(T, t_end, st, step=step, callback=cb, tol=1e-5)
        return T, np.array(rows)

    T, res = march(T, trans[0])
    phases = {"cooling": res}
    if case.get("reheat"):
        T, phases["reheat"] = march(T, trans[1], release=True)
    for ph, r in phases.items():
        with open(os.path.join(out_dir, f"probes_{ph}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Time (min)"] + [f"{n} (K)" for n in names])
            w.writerows(r.tolist())

    if snap:
        for ph, step in zip(phases, trans):
            name = "surface_T.npz" if ph == "cooling" else f"surface_T_{ph}.npz"
            np.savez_compressed(os.path.join(out_dir, name), nodes=surf,
                                t_min=np.array([a for a, _ in snaps[step]]),
                                T_K=np.stack([b for _, b in snaps[step]]))

    # ---- comparison at COMSOL's own time points
    kind, path, nkey, key = case["ref"]
    t_ref, ref = (ref_long if kind == "long" else ref_wide)(path, nkey, key) if nkey else ref_long(path)
    res = phases["reheat" if case.get("reheat") else "cooling"]
    rows = []
    if t_ref is not None:
        keep = t_ref <= res[-1, 0] + 1e-9
        for j, n in enumerate(names):
            k = norm(n)
            if k not in ref or np.isnan(ref[k]).all():
                continue
            rep = PchipInterpolator(res[:, 0], res[:, j + 1])(t_ref[keep])
            d = rep - ref[k][keep]
            rows.append(dict(study=case["study"], case=case["id"], probe=n,
                             comsol_start_K=ref[k][keep][0], replica_start_K=rep[0], diff_start_K=d[0],
                             comsol_end_K=ref[k][keep][-1], replica_end_K=rep[-1], diff_end_K=d[-1],
                             max_abs_diff_K=np.nanmax(np.abs(d)), t_of_max_min=t_ref[keep][np.nanargmax(np.abs(d))],
                             rms_diff_K=float(np.sqrt(np.nanmean(d ** 2))),
                             comsol_range_K=float(np.nanmax(ref[k][keep]) - np.nanmin(ref[k][keep]))))
        np.savez_compressed(os.path.join(out_dir, "comsol_reference.npz"), t=t_ref[keep],
                            **{norm(n).replace(" ", "_"): ref[norm(n)][keep] for n in names if norm(n) in ref})
    log.append(f"done in {time.time() - t0:.0f} s; {len(rows)} probes compared")
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in log) + "\n")
        f.write(f"code {code_version()}\n")
    return rows


def code_version():
    """Short SHA-256 over run_all.py and replica/*.py as they are on disk, to tell which code made a result."""
    import glob
    import hashlib
    h = hashlib.sha256()
    for path in [os.path.join(HERE, "run_all.py")] + sorted(glob.glob(os.path.join(HERE, "replica", "*.py"))):
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode() + f.read())
    return h.hexdigest()[:12] + " " + time.strftime("%Y-%m-%d %H:%M")


def _safe(case, out_root):
    try:
        return case, run_case(case, out_root), None
    except Exception:
        return case, [], traceback.format_exc()


# -------------------------------------------------------------------- figures
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]   # Okabe-Ito


def short(p):
    return re.sub(r" \d$", "", p.replace(" Probe", "").replace("Domain ", "").replace("Anteriror", "Anterior"))


def figures(out_root):
    """One validation figure per study, drawn from the stored comparison tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = os.path.join(out_root, "comparison_all.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    made = []
    for study in sorted({r["study"] for r in rows}):
        rs = [r for r in rows if r["study"] == study]
        probes = list(dict.fromkeys(short(r["probe"]) for r in rs))
        pts = [p for p in probes if not p.startswith(("Eye", "Retina"))]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.25]))
        # left: parity of end-of-run temperatures at the point probes
        for c, p in zip(COLORS, pts):
            sel = [r for r in rs if short(r["probe"]) == p]
            ax[0].plot([float(r["comsol_end_K"]) - 273.15 for r in sel], [float(r["replica_end_K"]) - 273.15 for r in sel],
                       ls="none", marker="o", ms=5, mfc=c, mec="white", mew=0.6, label=p)
        lim = [min(ax[0].get_xlim()[0], ax[0].get_ylim()[0]), max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
        ax[0].plot(lim, lim, color="0.6", lw=0.8, zorder=0)
        ax[0].set(xlabel="COMSOL, end of run (°C)", ylabel="Replica, end of run (°C)",
                  title=f"{study.replace('_', ' ')}: {len({r['case'] for r in rs})} cases")
        ax[0].legend(frameon=False, fontsize=8)
        # right: largest difference at any time, every case, by probe
        rng = np.random.default_rng(0)
        for i, p in enumerate(probes):
            v = np.array([float(r["max_abs_diff_K"]) for r in rs if short(r["probe"]) == p])
            ax[1].plot(np.maximum(v, 1e-5), i + rng.uniform(-0.25, 0.25, len(v)), ls="none", marker="o", ms=4,
                       mfc="#0072B2", mec="white", mew=0.4, alpha=0.8)
        ax[1].set_yticks(range(len(probes)), probes, fontsize=8)
        ax[1].invert_yaxis()
        ax[1].set_xscale("log")
        ax[1].set(xlabel="Largest |replica − COMSOL| at any time (°C)", title="Each dot is one case")
        for a in ax:
            a.grid(color="0.9", lw=0.6)
            a.spines[["top", "right"]].set_visible(False)
        out = os.path.join(out_root, study, "validation.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        made.append(out)
    return made


def summarize(out_root):
    with open(os.path.join(out_root, "comparison_all.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for study in sorted({r["study"] for r in rows}):
        rs = [r for r in rows if r["study"] == study]
        pt = [r for r in rs if "Domain" not in r["probe"]]
        f = lambda sel, k: max((abs(float(r[k])) for r in sel), default=float("nan"))
        worst = max(rs, key=lambda r: float(r["max_abs_diff_K"]))
        out.append(dict(study=study, cases=len({r["case"] for r in rs}), probes_compared=len(rs),
                        worst_point_probe_K=round(f(pt, "max_abs_diff_K"), 4),
                        worst_any_probe_K=round(f(rs, "max_abs_diff_K"), 4),
                        worst_end_of_run_K=round(f(rs, "diff_end_K"), 4),
                        worst_case=worst["case"], worst_probe=worst["probe"]))
    with open(os.path.join(out_root, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def main():
    all_studies = ["ring_comparison", "mesh_study", "silicone", "material_sweep", "parametric_sweep", "reheat"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", nargs="*", default=all_studies, choices=all_studies)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--full", action="store_true",
                    help="every case of the temperature sweeps (81 + 22) instead of the representative subset")
    ap.add_argument("--flow", action="store_true",
                    help="also solve the Laminar Flow interface (buoyant air, aqueous through-flow)")
    ap.add_argument("--no-slit", action="store_true",
                    help="thermal insulation on interior boundaries leaves the temperature continuous "
                         "(instead of allowing a jump across the face)")
    ap.add_argument("--case", nargs="*", help="run only cases whose study/id contains one of these strings")
    ap.add_argument("--only-missing", action="store_true", help="skip cases that already have results")
    ap.add_argument("--limit", type=int, help="run only the first N cases (testing)")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    allcsv = os.path.join(args.out, "comparison_all.csv")

    if not args.figures_only:
        cases = build_cases(args.studies, args.full)[:args.limit]
        if args.case:
            cases = [c for c in cases if any(s in f"{c['study']}/{c['id']}" for s in args.case)]
        variant = "_".join(v for v, on in (("with_flow", args.flow), ("no_slit", args.no_slit)) if on)
        if variant:                        # results go to results/<variant>/<study>/<case>
            for c in cases:
                c["flow"], c["no_slit"], c["variant"] = args.flow, args.no_slit, variant
            allcsv = os.path.join(args.out, variant, "comparison_all.csv")
            os.makedirs(os.path.dirname(allcsv), exist_ok=True)
        old, old_all = [], []
        if os.path.exists(allcsv):                      # keep results of studies not being re-run
            with open(allcsv, newline="", encoding="utf-8") as f:
                ids = {(c["study"], c["id"]) for c in cases}
                old_all = list(csv.DictReader(f))
                old = [r for r in old_all if (r["study"], r["case"]) not in ids]
        if args.only_missing:
            have = {(r["study"], r["case"]) for r in old_all}
            cases = [c for c in cases if (c["study"], c["id"]) not in have]
            old = old_all
        print(f"{len(cases)} cases, {args.workers} workers", flush=True)
        rows, failed, t0 = list(old), [], time.time()
        # one case per process: the direct solver's memory is only fully returned when the process ends
        with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
            futs = [ex.submit(_safe, c, args.out) for c in cases]
            for k, fu in enumerate(as_completed(futs), 1):
                case, r, err = fu.result()
                rows += r
                if err:
                    failed.append((case["study"], case["id"], err))
                worst = max((x["max_abs_diff_K"] for x in r), default=float("nan"))
                print(f"[{k}/{len(cases)} {time.time() - t0:6.0f} s] {case['study']}/{case['id']}: "
                      + ("FAILED\n" + err if err else f"worst |diff| {worst:.4f} K"), flush=True)
                with open(allcsv, "w", newline="", encoding="utf-8") as f:
                    if rows:
                        w = csv.DictWriter(f, fieldnames=list(rows[0]))
                        w.writeheader()
                        w.writerows(sorted(rows, key=lambda x: (x["study"], x["case"])))
        manifest = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                        code="run_all.py + replica/ (see README.md)",
                        settings=dict(time_scheme="BDF2", stages="6 s to 5 min, 15 s to 15 min, then 30 s",
                                      elements="linear tetrahedra on the mesh stored in each .mph"),
                        inputs=sorted({c["mph"] for c in cases} | {c["ref"][1] for c in cases}),
                        cases=[dict(study=c["study"], id=c["id"], mph=os.path.basename(c["mph"]), component=c["comp"],
                                    overrides=c.get("overrides"), switch_index=c.get("switch_index", 0)) for c in cases],
                        failed=[f[:2] for f in failed])
        with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
    variant = "_".join(v for v, on in (("with_flow", args.flow), ("no_slit", args.no_slit)) if on)
    root = os.path.join(args.out, variant) if variant else args.out
    for s in summarize(root):
        print(s)
    for p in figures(root):
        print("figure:", p)


if __name__ == "__main__":
    main()
