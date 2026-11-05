"""Tables, figures and MANIFEST.json of the outer-boundary package, from the cached runs of dev/outer_boundary.py.

    python -X utf8 dev/outer_boundary.py tables        (or: python -X utf8 dev/outer_boundary_report.py)

Writes to results/overnight_2026-10-08/outer_boundary/: summary_tables.md, all_runs.csv, fig_*.png, MANIFEST.json.
Nothing is solved here; a run that is missing is left out of its table and named at the end of the file.
"""
import csv
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import outer_boundary as ob  # noqa: E402

NAMES = [n for _, n in ob.SITES]
SHORT = {"anterior sclera": "ant. sclera", "equatorial sclera": "equat. sclera", "vitreous": "vitreous",
         "posterior sclera": "post. sclera", "proximal optic nerve": "prox. optic nerve",
         "distal optic nerve": "dist. optic nerve", "average retina": "avg retina"}
MISSING = []


def load(tag):
    p = os.path.join(ob.RUNS, tag + ".json")
    if not os.path.exists(p):
        MISSING.append(tag)
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def d(run, ref, name, what="end"):
    return run[name][what] - ref[name][what]


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def head(cells):
    return row(cells) + "\n" + row(["---"] * len(cells))


def equivalent_h(run, ref, uniform, name="proximal optic nerve"):
    """Uniform shell coefficient that gives the same change at 35 min at one site (log interpolation)."""
    hs = np.array(sorted(uniform))
    dv = np.array([d(uniform[h], ref, name) for h in hs])
    x = d(run, ref, name)
    if x <= dv.min():
        return f"<{hs[0]:g}"
    if x >= dv.max():
        return f">{hs[-1]:g}"
    return f"{float(np.exp(np.interp(x, dv, np.log(hs)))):.0f}"


def main():
    os.makedirs(ob.OUT, exist_ok=True)
    L = []
    w = L.append
    fixed = load("fixed")
    uniform = {h: load(f"uniform_h{h}") for h in ob.H_UNIFORM}
    uniform = {h: r for h, r in uniform.items() if r}
    w("# Outer boundary of the in-vivo model: tables\n")
    w(f"Made by `dev/outer_boundary_report.py` on {time.strftime('%Y-%m-%d %H:%M')} from the cached runs of "
      "`dev/outer_boundary.py`. All runs: v1 ring unless stated, aluminium, body 37 C, room 25 C, set point 4 C, "
      "COMSOL's coarser mesh (73,011 elements in the new set-up; 83,631 to 100,623 with added tissue). Temperatures in C. "
      "\"Change\" = run minus the fixed-shell run on the same mesh.\n")

    # ---- 1. properties
    w("## 1. Properties of what lies beyond the shell (as used)\n")
    w(head(["Material", "k W/mK", "rho*cp MJ/m3K", "perfusion term W/m3K", "time constant rho*cp/w, min",
            "depth sqrt(k/w), mm", "h steady sqrt(k*w), W/m2K", "h 5 / 15 / 35 min after a step"]))
    mats = [("orbital fat (the model's)", ob.MAT["fat"]), ("cortical bone", ob.MAT["bone"]),
            ("cortical bone, measured k", ob.MAT["bone_hi"]), ("resting skeletal muscle (temporalis)", ob.MAT["muscle"]),
            ("extraocular muscle (the model's)", dict(k=0.49, rc=3728890.0, w=0.0045 * ob.RHOC_BLOOD)),
            ("brain (grey and white mixed)", ob.MAT["brain"]), ("sinus mucosa", ob.MAT["mucosa"])]
    for n, m in mats:
        w(row([n, f"{m['k']:.3g}", f"{m['rc'] / 1e6:.2f}", f"{m['w']:.0f}", f"{m['rc'] / m['w'] / 60:.1f}",
               f"{(m['k'] / m['w']) ** 0.5 * 1e3:.1f}", f"{ob.h_semi(m):.0f}",
               " / ".join(f"{ob.h_semi(m, t):.0f}" for t in (300, 900, 2100))]))
    w("\nSemi-infinite perfused tissue, surface stepped at t = 0: h(t) = sqrt(k w) erf(sqrt(t/tau)) + "
      "sqrt(k rho c / (pi t)) exp(-t/tau), tau = rho c / w; where k is the conductivity, w the perfusion term "
      "(blood density x blood heat capacity x perfusion rate), rho c the volumetric heat capacity of the tissue and "
      "t the time since the step. Air-filled sinus: 8 W/m2K across the cavity (radiation about 6, conduction and "
      "weak convection about 2), entered as a solid layer of that conductance.\n")

    # ---- 2. layered walls
    w("## 2. Equivalent coefficient of layered walls (one-dimensional, `layers_1d`)\n")
    w(head(["Wall, from the orbit outward", "steady", "5 min", "15 min", "35 min"]))
    stacks = [("fat, 60 mm (check: formula gives 18.8 steady, 19.2 at 35 min)", [(60, "fat")], "fixed"),
              ("brain, 40 mm (check: formula 132)", [(40, "brain")], "fixed"),
              ("bone 0.3 mm, mucosa 0.5 mm, sinus air", [(0.3, "bone"), (0.8, "mucosa"), (20, "cavity")], "fixed"),
              ("bone 0.5 mm, mucosa 1 mm, sinus air", [(0.5, "bone"), (1.5, "mucosa"), (20, "cavity")], "fixed"),
              ("bone 0.5 mm, mucosa 0.3 mm, sinus air", [(0.5, "bone"), (0.8, "mucosa"), (20, "cavity")], "fixed"),
              ("bone 0.5 mm, sinus air, no mucosa", [(0.5, "bone"), (20, "cavity")], "fixed"),
              ("bone 1 mm, brain", [(1, "bone"), (40, "brain")], "fixed"),
              ("bone 2 mm, brain", [(2, "bone"), (40, "brain")], "fixed"),
              ("bone 3 mm, brain", [(3, "bone"), (40, "brain")], "fixed"),
              ("bone (k 0.68) 2 mm, brain", [(2, "bone_hi"), (40, "brain")], "fixed"),
              ("bone 3 mm, resting muscle", [(3, "bone"), (60, "muscle")], "fixed"),
              ("bone 5 mm, resting muscle", [(5, "bone"), (60, "muscle")], "fixed"),
              ("bone (k 0.68) 5 mm, resting muscle", [(5, "bone_hi"), (60, "muscle")], "fixed"),
              ("bone, 80 mm (orbital margin)", [(80, "bone")], "fixed"),
              ("bone 6 mm, then 10 W/m2K (skin and air, taken at 37 C)", [(6, "bone")], 10.0),
              ("3 mm more fat, bone 1 mm, brain", [(3, "fat"), (4, "bone"), (40, "brain")], "fixed"),
              ("3 mm more fat, bone 0.3 mm, mucosa 0.5 mm, sinus air",
               [(3, "fat"), (3.3, "bone"), (3.8, "mucosa"), (20, "cavity")], "fixed")]
    for n, st, outer in stacks:
        out, hs = ob.layers_1d(ob.stack_with_cavity(st), outer)
        w(row([n, f"{hs:.0f}"] + [f"{v:.0f}" for v in out.values()]))
    w("")

    # ---- 3. best estimate per region
    w("## 3. Best-estimate coefficient per region of the shell (W/m2K to 37 C)\n")
    why = {"rim lateral": "zygomatic and frontal bone of the margin, skin", "rim sup-inf": "frontal / maxillary margin, skin",
           "rim medial": "frontal process of maxilla, lacrimal bone, nose",
           "mid lateral": "zygomatic bone 3-5 mm, temporalis muscle", "mid sup-inf": "roof 1-3 mm + frontal lobe (about 75) "
           "and floor 0.5-1 mm + maxillary sinus (about 22): one region in the half model",
           "mid medial": "lamina papyracea, ethmoid air cells", "deep lateral": "greater wing of sphenoid 3 mm, temporal lobe",
           "deep sup-inf": "roof + frontal lobe; floor + maxillary sinus", "deep medial": "posterior ethmoid / sphenoid sinus",
           "apex": "sphenoid, optic canal, cavernous sinus, brain"}
    areas = {}
    try:
        m = ob.Model("v1")
        _, prob, _ = m.problem(1.0)
        tr = next(c for c in prob.convective if c["name"] == "Body shell (soft)")["tris"]
        P = prob.points[tr]
        a = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1) / 2 * 1e6
        reg = ob.region_of(P.mean(1) * 1e3)
        areas = {n: float(a[reg == i].sum()) for i, n in enumerate(ob.REGIONS)}
    except Exception as e:                                     # tables can still be made without the model files
        w(f"(areas not computed: {e})\n")
    w(head(["Region (z range, mm)", "area mm2 (half model)", "what lies beyond", "low", "best", "high"]))
    zr = {"rim": "z > -14.3, eyelid part", "mid": "-27 to -14.3", "deep": "-38 to -27", "apex": "z < -38"}
    for n in ob.REGIONS:
        w(row([f"{n} ({zr[n.split()[0]]})", f"{areas.get(n, float('nan')):.0f}", why[n], ob.LOW[n], ob.BEST[n], ob.HIGH[n]]))
    w("\nSectors by angle about the axis: lateral (temporal) 0-60 deg, sup-inf 60-120, medial (nasal) 120-180.\n")

    # ---- 4. uniform sweep
    if fixed:
        w("## 4. Uniform shell coefficient: change against the fixed shell\n")
        for what, label in (("end", "after 35 min of cooling"), ("steady", "steady state with the ring at 4 C"),
                            ("start", "start state (before cooling)")):
            w(f"**{label}**\n")
            w(head(["h W/m2K"] + [SHORT[n] for n in NAMES]))
            w(row(["fixed shell, value"] + [f"{fixed[n][what]:.2f}" for n in NAMES]))
            for h, r in uniform.items():
                w(row([h] + [f"{d(r, fixed, n, what):+.2f}" for n in NAMES]))
            w("")
        w("How far from steady after 35 min (35-min value minus steady value):\n")
        w(head(["h W/m2K"] + [SHORT[n] for n in NAMES]))
        w(row(["fixed"] + [f"{fixed[n]['end'] - fixed[n]['steady']:+.3f}" for n in NAMES]))
        for h, r in uniform.items():
            w(row([h] + [f"{r[n]['end'] - r[n]['steady']:+.3f}" for n in NAMES]))
        w("")

    # ---- 5. regions
    ref = load("uniform_h5000")
    u20 = load("uniform_h20")
    if ref and u20:
        w("## 5. Which part of the shell matters: one part at 20 W/m2K, the rest at 5,000\n")
        w("Change after 35 min against all of the shell at 5,000 W/m2K.\n")
        w(head(["Soft part", "area mm2"] + [SHORT[n] for n in NAMES]))
        tot = {n: 0.0 for n in NAMES}
        for name in ob.REGIONS:
            r = load(f"region_soft_{name.replace(' ', '_')}")
            if r:
                w(row([name, f"{areas.get(name, float('nan')):.0f}"] + [f"{d(r, ref, n):+.2f}" for n in NAMES]))
                for n in NAMES:
                    tot[n] += d(r, ref, n)
        w(row(["sum of the ten", ""] + [f"{tot[n]:+.2f}" for n in NAMES]))
        w(row(["whole shell at 20", ""] + [f"{d(u20, ref, n):+.2f}" for n in NAMES]))
        for lab, tags in (("band", ("rim", "mid", "deep")), ("sector", ob.SECTORS)):
            for t in tags:
                r = load(f"{lab}_soft_{t}")
                if r:
                    w(row([f"whole {lab}: {t}", ""] + [f"{d(r, ref, n):+.2f}" for n in NAMES]))
        w("")
        w("The single parts do not add up to the whole (heat finds the parts still held at 37 C), so the other way "
          "round: the whole shell at 20 W/m2K and one part held at 37 C (5,000). Change after 35 min against the "
          "whole shell at 20 (positive = that part, if it really were at 37 C, would warm the site by this much).\n")
        w(head(["Part held at 37 C", "area mm2"] + [SHORT[n] for n in NAMES]))
        for name in ob.REGIONS:
            r = load(f"region_stiff_{name.replace(' ', '_')}")
            if r:
                w(row([name, f"{areas.get(name, float('nan')):.0f}"] + [f"{d(r, u20, n):+.2f}" for n in NAMES]))
        for lab, tags in (("band", ("rim", "mid", "deep")), ("sector", ob.SECTORS)):
            for t in tags:
                r = load(f"{lab}_stiff_{t}")
                if r:
                    w(row([f"whole {lab}: {t}", ""] + [f"{d(r, u20, n):+.2f}" for n in NAMES]))
        w(row(["whole shell at 5,000", ""] + [f"{d(ref, u20, n):+.2f}" for n in NAMES]))
        w("")

    # ---- 6. extended domain
    if fixed:
        w("## 6. Tissue added beyond the shell (extruded layers), change after 35 min against the fixed shell\n")
        w(head(["Case", "elements"] + [SHORT[n] for n in NAMES] + ["35 min minus steady, optic nerve",
                                                                   "same change at the optic nerve with uniform h ="]))
        lab = [("ext5_check_perfect_conductor", "CHECK: 5 mm of a perfect conductor, 37 C outside (must be 0)")]
        for T in (5, 10, 20):
            lab += [(f"ext{T}_fat_fixed", f"{T} mm more orbital fat, 37 C outside"),
                    (f"ext{T}_fat_semi", f"{T} mm more orbital fat, then fat without end (19 W/m2K)")]
        lab += [("ext10_fat_insulated", "10 mm more orbital fat, insulated outside")]
        lab += [(f"ext{T}_bone_fixed", f"{T} mm of bone all round, 37 C outside") for T in (5, 10, 20)]
        lab += [("ext10_all_brain", "everywhere: bone 2 mm, brain"), ("ext10_all_bone_muscle", "everywhere: bone 4.5 mm, resting muscle"),
                ("ext10_all_sinus", "everywhere: bone 0.5 mm, mucosa 0.5 mm, sinus air"),
                ("ext10_anat_roof", "anatomical walls, sup-inf = roof and brain"),
                ("ext10_anat_floor", "anatomical walls, sup-inf = floor and maxillary sinus"),
                ("ext10_anat_roof_bonehi", "anatomical, roof, bone k 0.68"), ("ext10_anat_floor_bonehi", "anatomical, floor, bone k 0.68")]
        for tag, name in lab:
            r = load(tag)
            if r:
                w(row([name, r["elements"]] + [f"{d(r, fixed, n):+.2f}" for n in NAMES]
                      + [f"{r['proximal optic nerve']['end'] - r['proximal optic nerve']['steady']:+.2f}",
                         equivalent_h(r, fixed, uniform)]))
        w("\nAnatomical walls (10 mm of added tissue, 37 C at its outside): medial = bone 0.5 mm, mucosa 0.5 mm, sinus air; "
          "lateral = bone 4.5 mm then resting muscle (mid), bone 3 mm then brain (deep); margin (eyelid part) = bone 6 mm "
          "then soft tissue; apex = bone 2 mm then brain; sup-inf = roof (bone 2 mm, brain) or floor (as medial).\n")

    # ---- 7. consequences
    w("## 7. Consequences: temperature after 35 min, v1 aluminium\n")
    w("Production-mesh fixed-shell value plus the coarser-mesh difference (boundary run minus fixed-shell run).\n")
    cq = {k: load(f"cq_v1_{k}") for k in ("fixed", "best", "low", "high", "anat_roof", "anat_floor", "h20", "h35", "h50", "h100")}
    if cq["fixed"]:
        labels = [("fixed", "fixed 37 C shell (present baseline)"), ("high", "regional coefficients, high (stiff) end"),
                  ("best", "regional coefficients, best estimate"), ("low", "regional coefficients, low (soft) end"),
                  ("anat_roof", "added tissue, anatomical, roof"), ("anat_floor", "added tissue, anatomical, floor"),
                  ("h100", "uniform 100"), ("h50", "uniform 50"), ("h35", "uniform 35"), ("h20", "uniform 20")]
        w(head(["Boundary"] + [f"{SHORT[n]}" for n in ob.DEEP] + ["cooling of the optic nerve (start minus end)",
                                                                   "uniform h with the same optic-nerve change"]))
        for k, name in labels:
            r = cq[k]
            if r:
                w(row([name] + [f"{ob.PRODUCTION[n] + d(r, cq['fixed'], n):.2f} ({d(r, cq['fixed'], n):+.2f})" for n in ob.DEEP]
                      + [f"{r['proximal optic nerve']['start'] - r['proximal optic nerve']['end']:.2f}",
                         "-" if k == "fixed" else equivalent_h(r, fixed, uniform) if fixed else ""]))
        w("")
        w("Start state (before cooling), coarser mesh:\n")
        w(head(["Boundary"] + [SHORT[n] for n in NAMES]))
        for k, name in labels[:6]:
            if cq[k]:
                w(row([name] + [f"{cq[k][n]['start']:.2f}" for n in NAMES]))
        w("")

    # ---- 8. perfusion
    w("## 8. Boundary x perfusion (all perfusion terms scaled), v1: proximal optic nerve after 35 min\n")
    w("Coarser-mesh values (the fixed-shell coarser value at nominal perfusion is 33.72; production 33.77).\n")
    cols = ("fixed", "high", "best", "low", "anat_roof", "anat_floor", "h100", "h20")
    w(head(["Perfusion"] + list(cols) + ["best minus fixed"]))
    for f, sfx in ((0.0, "v1_noperf"), (0.25, "v1_perf0.25"), (0.5, "v1_perf0.5"), (1.0, "v1"), (2.0, "v1_perf2")):
        rr = {c: load(f"cq_{sfx}_{c}") if not ((f == 0.25 and c not in ("fixed", "best", "low")) or (f == 0.0 and c not in ("fixed", "best"))) else None for c in cols}
        if rr["fixed"]:
            w(row([f"x{f:g}"] + [f"{rr[c]['proximal optic nerve']['end']:.2f}" if rr[c] else "" for c in cols]
                  + [f"{d(rr['best'], rr['fixed'], 'proximal optic nerve'):+.2f}" if rr["best"] else ""]))
    w("")
    w("Same, average retina:\n")
    w(head(["Perfusion"] + list(cols) + ["best minus fixed"]))
    for f, sfx in ((0.0, "v1_noperf"), (0.25, "v1_perf0.25"), (0.5, "v1_perf0.5"), (1.0, "v1"), (2.0, "v1_perf2")):
        rr = {c: load(f"cq_{sfx}_{c}") if not ((f == 0.25 and c not in ("fixed", "best", "low")) or (f == 0.0 and c not in ("fixed", "best"))) else None for c in cols}
        if rr["fixed"]:
            w(row([f"x{f:g}"] + [f"{rr[c]['average retina']['end']:.2f}" if rr[c] else "" for c in cols]
                  + [f"{d(rr['best'], rr['fixed'], 'average retina'):+.2f}" if rr["best"] else ""]))
    w("")

    # ---- 9. rings
    w("## 9. Rings: v2 and v3 minus v1 after 35 min, per boundary (coarser mesh)\n")
    w(head(["Boundary", "ring"] + [SHORT[n] for n in NAMES]))
    for k in ("fixed", "high", "best", "low", "anat_roof", "anat_floor", "h20"):
        r1 = load(f"cq_v1_{k}")
        for ring in ("v2", "v3"):
            r = load(f"cq_{ring}_{k}")
            if r and r1:
                w(row([k, f"{ring} - v1"] + [f"{d(r, r1, n):+.2f}" for n in NAMES]))
    w("")
    w("Change caused by the boundary (best estimate minus fixed) for each ring:\n")
    w(head(["ring"] + [SHORT[n] for n in NAMES]))
    for ring in ("v1", "v2", "v3"):
        a_, b_ = load(f"cq_{ring}_best"), load(f"cq_{ring}_fixed")
        if a_ and b_:
            w(row([ring] + [f"{d(a_, b_, n):+.2f}" for n in NAMES]))
    w("")

    # ---- csv of everything
    tags = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ob.RUNS, "*.json")))
    with open(os.path.join(ob.OUT, "all_runs.csv"), "w", newline="", encoding="utf-8") as f:
        c = csv.writer(f)
        c.writerow(["run", "ring", "settings", "shell", "added tissue mm", "elements"]
                   + [f"{n}, {x} (C)" for n in NAMES for x in ("start", "35 min", "steady")])
        for t in tags:
            r = load(t)
            c.writerow([t, r["ring"], json.dumps(r["settings"]), json.dumps(r["shell"]), r["ext"], r["elements"]]
                       + [f"{r[n][x]:.4f}" for n in NAMES for x in ("start", "end", "steady")])

    figs = figures(fixed, uniform, ref, cq) if fixed else []
    if MISSING:
        w("## Runs not found when this file was made\n")
        w(", ".join(sorted(set(MISSING))) + "\n")
    with open(os.path.join(ob.OUT, "summary_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    def sha(p):
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    import shutil
    os.makedirs(os.path.join(ob.OUT, "runs"), exist_ok=True)   # copy of the run cache next to the figures
    for t in tags:
        shutil.copyfile(os.path.join(ob.RUNS, t + ".json"), os.path.join(ob.OUT, "runs", t + ".json"))
    import run_all
    man = dict(made=time.strftime("%Y-%m-%d %H:%M:%S"), package="outer_boundary",
               command="python -X utf8 dev/outer_boundary.py all",
               code={os.path.basename(p): sha(p) for p in (os.path.join(HERE, "outer_boundary.py"), os.path.abspath(__file__),
                                                              os.path.join(HERE, "new_runs.py"), os.path.join(HERE, "mesh_adapt.py"))},
               solver_code=run_all.code_version(),
               inputs=dict(model_file=run_all.mph_path(ob.nr.MESH_FILE["coarser"]), mesh="COMSOL coarser, components SCR v1 / v2 / v3",
                           settings="DEFAULTS of dev/new_runs.py with shell_h, perf_all, ring as listed per run in all_runs.csv",
                           run_cache=ob.RUNS, run_cache_copy="runs/ (one json per run: read-outs at start, 35 min, steady, and time courses)",
                           runs=len(tags)),
               outputs=["summary_tables.md", "all_runs.csv", "cases_outer_boundary.json", "cases_outer_boundary_needs_patch.json",
                        "new_runs_outer_boundary.patch.md"] + figs,
               figures={f: "from the cached runs listed in all_runs.csv" for f in figs})
    with open(os.path.join(ob.OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    print("wrote", os.path.join(ob.OUT, "summary_tables.md"), "missing:", sorted(set(MISSING)))


def figures(fixed, uniform, ref, cq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
    COL = {"proximal optic nerve": "#2a78d6", "posterior sclera": "#eb6834", "average retina": "#1baf7a",
           "vitreous": "#eda100", "equatorial sclera": "#4a3aa7"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
    out = []

    # 1. uniform sweep
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    hs = sorted(uniform)
    for n, c in COL.items():
        y = [d(uniform[h], fixed, n) for h in hs]
        ax.plot(hs, y, "-o", color=c, lw=2, ms=4, label=n)
        ax.annotate(n, (hs[0], y[0]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", color=INK, fontsize=8)
    for lo, hi, lab in ((9, 35, "air sinus\nbehind thin bone"), (20, 50, ""), (60, 110, "brain\nbehind bone")):
        if lab:
            ax.axvspan(lo, hi, color="#f0efec", zorder=0)
            ax.text((lo * hi) ** 0.5, 0.05, lab, ha="center", va="bottom", color=MUTED, fontsize=7.5,
                    transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_xlim(1.2, 6000)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.set_xlabel("Coefficient of the outer shell to 37 C (W/m2K); 5,000 = fixed shell")
    ax.set_ylabel("Change after 35 min against the fixed shell (C)")
    ax.set_title("A softer orbit wall leaves every site colder", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(ob.OUT, "fig_uniform_sweep.png"), dpi=200); plt.close(fig)
    out.append("fig_uniform_sweep.png")

    # 2. region sensitivity (optic nerve, posterior sclera, average retina, vitreous)
    if ref:
        u20 = load("uniform_h20")
        rr = {n: load(f"region_stiff_{n.replace(' ', '_')}") for n in ob.REGIONS}
        if all(rr.values()):
            fig, axs = plt.subplots(1, 4, figsize=(9.6, 3.6), sharey=True, sharex=True)
            y = np.arange(len(ob.REGIONS))[::-1]
            for ax, n in zip(axs, ob.DEEP):
                v = [d(rr[r], u20, n) for r in ob.REGIONS]
                ax.barh(y, v, height=0.6, color=COL[n])
                for yy, vv in zip(y, v):
                    ax.text(vv + 0.02, yy, f"{vv:+.2f}", ha="left", va="center", fontsize=7, color=INK)
                ax.set_title(n, loc="left", fontsize=9, color=INK)
                ax.axvline(0, color=MUTED, lw=0.8)
                ax.grid(axis="x", color="#e6e5e1", lw=0.6)
                ax.set_xlabel("change (C)")
            axs[0].set_yticks(y); axs[0].set_yticklabels(ob.REGIONS)
            axs[0].set_xlim(-0.02, max(0.6, axs[0].get_xlim()[1] * 1.3))
            fig.suptitle("Warming after 35 min when one part of a 20 W/m2K shell is held at 37 C: where the shell matters",
                         x=0.01, ha="left", fontsize=10, color=INK)
            fig.tight_layout(); fig.savefig(os.path.join(ob.OUT, "fig_region_sensitivity.png"), dpi=200); plt.close(fig)
            out.append("fig_region_sensitivity.png")

    # 3. time course at the proximal optic nerve
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    lines = [("fixed", "fixed 37 C shell", "#2a78d6"), ("best", "regional coefficients, best estimate", "#eb6834"),
             ("low", "regional coefficients, soft end", "#eda100"), ("anat_roof", "10 mm of added tissue (anatomical, roof)", "#1baf7a")]
    for k, lab, c in lines:
        r = cq.get(k)
        if r:
            s = r["proximal optic nerve"]
            ax.plot(s["t"] + [35.0], s["series"] + [s["end"]], color=c, lw=2, label=lab)
            ax.annotate(f"{s['end']:.2f}", (35.0, s["end"]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
    ax.axhline(32.0, color=MUTED, lw=0.8, ls="--"); ax.text(0.3, 32.03, "32 C", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Time since the start of cooling (min)"); ax.set_ylabel("Proximal optic nerve (C), coarser mesh")
    ax.set_xlim(0, 38.5); ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.set_title("Optic nerve: the boundary changes the start, the end and the time to settle", loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ob.OUT, "fig_optic_nerve_time.png"), dpi=200); plt.close(fig)
    out.append("fig_optic_nerve_time.png")
    return out


if __name__ == "__main__":
    os.chdir(ob.REPLICA)
    main()
