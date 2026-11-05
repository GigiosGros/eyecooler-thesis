"""The new eye runs of the simulation plan, on COMSOL's stored meshes (route A).

    python dev/new_runs.py --list                       # every case and its state
    python dev/new_runs.py --id B_v1_Al RW_B_v1_Al      # exactly these case ids (an unknown id is an error)
    python dev/new_runs.py --case B_v1_Al               # SUBSTRING match: B_v1_Al also selects RW_B_v1_Al ...
                                                        # (warns when a pattern selects more than one case; use --id)
    python dev/new_runs.py --block A B F1               # whole blocks
    python dev/new_runs.py --all                        # everything not yet done
    python dev/new_runs.py --index 7                    # case number 7 of --list (for a cluster job array)
    python dev/new_runs.py --case B_v1_Al --mesh coarser --dry-run

Each case starts from the problem that replica.model.build reads out of the
COMSOL model file (mesh, materials, perfusion, probes) and changes it to the
new set-up; the solver files in replica/ are not touched.

  removed    air and module (TEC) layer; ex vivo also fat, muscles, eyelids
  merged     Cloquet's canal into the vitreous
  start      stationary state with the cooling face free
  cooling    the ring face that touched the module layer goes from its own
             starting temperature to the set point along the model's ramp
  surfaces   that touched air: coefficient + radiation to the room;
             ex vivo, globe surfaces that touched orbit tissue: oil bath
  rewarming  optional continuation with the cooling face released

Every case runs in its own process.  A case that fails writes its error to
STATUS.json and the next one starts; a finished case is skipped on the next
call unless --redo is given.  status.csv in the output folder lists them all.

Safeguards added 2026-10-22:
  --id         exact case ids; --case stays a substring match and prints a warning when one pattern selects
               several cases (on the cluster ob_489763_1 ran 16 cases through one --case pattern)
  lock         each case folder gets RUNNING.lock, created with O_CREAT | O_EXCL before the solve and removed after
               it; a second process finds the file and skips the case (it is also the "running" marker of an
               interrupted --redo).  A lock left by a killed process names its host, pid and start time; remove it by
               hand after checking that the process is gone.
  --dry-run    builds the problem into <out>/_dry/<case>, never into the folder of a real run, so a finished case's
               STATUS.json, settings.json and console.log are never touched
  overrides    a property override ("prop") whose name matches no kept domain, a lone rho or cp, or a key other
               than k / rho / cp raises.  The 530 historical cases whose tear-film override is a no-op on the v1/v2
               meshes are listed in dev/inputs/inert_overrides.json; they run unchanged and --list labels them
  cold-side    for h_face > 0 the reaction at the cooling face also carried the face-to-room term
  heat         2 h_face A_face (T_room - T_face); it is now subtracted (stored columns of E7_* and XG_* from before
               DRIVER_VERSION 3 still contain it: +0.40 W at 50 W/m2K, +1.59 W at 200 W/m2K)
  record       settings.json also holds the applied per-domain properties, perfusion, metabolic heat, blood
               temperature, ring emissivity, the SHA-256 of the CT sensor picks, the code stamp of dev/manifest.py
               (run_all.py + replica/ + dev/, CR LF normalised) and the point probes that lie outside the mesh
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPLICA = os.path.dirname(HERE)
sys.path.insert(0, REPLICA)

os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np  # noqa: E402

K0 = 273.15
OUT_DEFAULT = os.path.join(REPLICA, "results", "new_runs")

# ------------------------------------------------------------------ settings
DEFAULTS = dict(
    ring="v1", material="Al", silicone=None, ex_vivo=False, mesh="production",
    Tb=37.0, T0=25.0, Tc=4.0,            # body, room, set point (C)
    h_air=10.0,                          # W/m2K on surfaces that touched air
    evap=-12.0,                          # W/m2 on the cornea (ignored in v3: the lens covers it; none ex vivo)
    ramp_s=90.0,                         # duration of the set-point ramp
    workbook=True,                       # tissue properties of the workbook (False = the old model's)
    eps_from_material=True,              # ring emissivity by material (False = 0.05 for every ring, as solved before)
    perf_all=1.0, perf_choroid=1.0, perf_retina=1.0, perf_orbit=1.0,   # perfusion scale factors
    perf_ciliary=1.0,                    # (2026-10-28) ciliary body alone (Geiser 2004 has about 20x our flow)
    prop={},                             # extra overrides {"Sclera": {"k": 0.6}}
    t_cool_min=35.0, rewarm_min=0.0,
    release_steps=False,                 # True: short time steps in the first minute after the ring is released
    exact_stages=False,                  # True: a time stage that is not a whole number of steps ends exactly on its
                                         # end time (dev/transient_fixed.py); no case defined on 2026-10-16 needs it
    # ex vivo
    h_oil=30.0, T_bath=37.0, T_room=23.4, oil_fixed=False, choroid_perfusion=False,
    stump_mm=None,                       # mm: optic-nerve stump cut to this length, cut face in the oil (dev/nerve_stump.py);
    stump_rule="centroid",               # None = the model's 17.8 mm. Rule for the elements at the cut: centroid / inside
    # verification switches
    dt_scale=1.0,                        # time steps multiplied by this (0.5 = half steps)
    no_cooling=False,                    # cooling face never switched on: the start state must not drift
    shell_h=None,                        # W/m2K: body-temperature shell as a soft condition instead of a fixed temperature
    ramp_tau=None,                       # s: if set, the set point is approached exponentially with this time constant
                                         # (the measured ring falls fastest at t = 0) instead of along the smooth step
    mesh_file=None,                      # a mesh .npz in dev/inputs/ used in place of the mesh stored in the model file
                                         # (same domain and boundary numbers; made by dev/mesh_adapt.py)
    refine=0,                            # times every element is split into eight (same geometry, half the spacing)
    h_face=0.0,                          # W/m2K from the cooling face to the room while it is not driven: the unpowered
                                         # module and heat exchanger standing on the ring (0 = face perfectly free)
    shell_regions=None,                  # {region: W/m2K}: body-temperature shell as a coefficient that differs by region
                                         # (regions of dev/outer_boundary.py; not for extended meshes)
    ext_stacks=None,                     # material set (dev/outer_boundary.STACK_SETS) of the tissue that an extended
                                         # mesh_file adds beyond the shell; without it that tissue is orbital fat
)

DRIVER_VERSION = 3                       # 2: oil_fixed as a large coefficient; energy balance logged in every run
                                         # 3 (2026-10-22): cold-side heat without the h_face room term; overrides
                                         #   checked; applied properties and code stamp in settings.json
INERT_FILE = os.path.join(HERE, "inputs", "inert_overrides.json")
_INERT = None


def inert_overrides():
    """{case id: {"names": [...], "whole_case_inert": bool}} of dev/inputs/inert_overrides.json."""
    global _INERT
    if _INERT is None:
        try:
            with open(INERT_FILE, encoding="utf-8") as f:
                _INERT = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        except OSError:
            _INERT = {}
    return _INERT


def label_of(cid):
    """Short label for tables: '' or the no-op note of an inert override."""
    v = inert_overrides().get(cid)
    if not v:
        return ""
    return ("NO-OP (override of " + ", ".join(v["names"]) + " matches no domain of this mesh: equals its baseline)"
            if v["whole_case_inert"] else "column inert: " + ", ".join(v["names"]) + " (no such domain on this mesh)")
H_OIL_FIXED = 1.0e4                      # W/m2K standing for "surface held at bath temperature"
MATERIAL_INDEX = {"Al": 0, "alumina": 1, "silica": 2}
EPS = {"Al": 0.05, "alumina": 0.95, "silica": 0.90}          # see the simulation plan, open item 1
# workbook choices that differ from the old model (audit of 2026-10-01): k, rho, cp
# Stated bias (kept, no rerun): the composition densities of the workbook (and of
# the COMSOL files, so of every case) use rho = [phi_w + (1 - phi_w) rho_s] x 1000, linear in the water MASS fraction
# phi_w (rho_s: density of the solids, g/cm3).  The volume-additive rule 1 / rho = phi_w / 1000 + (1 - phi_w) / (1000 rho_s)
# gives densities 1.5-2.0 % lower (sclera 1082.4 against 1102, lens 1103.6 / 1125.8, ciliary body 1069.3 / 1087.5,
# choroid 1057.1 / 1074, cornea 1054.1 / 1070.3 kg/m3).  Effect at 35 min <= 0.0003 C; early time constants (prop. to
# rho cp) up to about 2 %.
WORKBOOK = {"Sclera": dict(k=0.545), "Eyelids": dict(k=0.34, rho=1070.0, cp=3660.0),
            "Fat": dict(k=0.2245, rho=920.0, cp=2500.0), "Aqueous Humor": dict(k=0.593, rho=1005.0, cp=4159.0)}
MESH_FILE = {"production": r"Ring Comparison\Ring Comparison.mph",
             "coarse": r"Ring Comparison\Ring Comparison Mesh Study Coarse.mph",
             "coarser": r"Ring Comparison\Ring Comparison Mesh Study Coarser.mph"}
SILICONE_FILE = {"0.5mm": r"Silicone\Ring 0.5mm silicone layer.mph", "1mm": r"Silicone\Ring 1mm silicone layer.mph"}
ORBIT = ("Fat", "Eyelids", "Extraocular muscles")


_CASES = None


def cases():
    global _CASES
    if _CASES is None:
        _CASES = _build_cases()
    return _CASES


def _build_cases():
    """The run table of the simulation plan (section 4) as {id: settings that differ from DEFAULTS}."""
    c = {}
    rings, mats = ("v1", "v2", "v3"), ("Al", "alumina", "silica")
    for r in rings:
        c[f"A_{r}_bridge"] = dict(ring=r, workbook=False, eps_from_material=False)
    for r in rings:
        for m in mats:
            c[f"B_{r}_{m}"] = dict(ring=r, material=m, rewarm_min=35.0 if m == "Al" else 0.0)
    for r in rings:
        for s in ("0.5mm", "1mm"):
            c[f"C_{r}_silicone{s}"] = dict(ring=r, silicone=s)
    for r in rings:
        for tag, kw in (("Tb35", dict(Tb=35.0)), ("T0_10", dict(T0=10.0)), ("Tc12", dict(Tc=12.0)),
                        ("warm", dict(Tb=39.0, T0=35.0, Tc=12.0)), ("cold", dict(Tb=35.0, T0=10.0, Tc=4.0))):
            c[f"D_{r}_{tag}"] = dict(ring=r, rewarm_min=35.0, **kw)
    for h in (5.0, 20.0):
        c[f"E1_hair{h:g}"] = dict(h_air=h)
    for q in (0.0, -40.0, -100.0):
        c[f"E2_evap{abs(q):g}"] = dict(evap=q)
    for tag, key in (("choroid", "perf_choroid"), ("retina", "perf_retina"), ("orbit", "perf_orbit")):
        for f in (0.5, 2.0):
            c[f"E4_perf_{tag}_x{f:g}"] = {key: f}
    c["E5_noperfusion"] = dict(perf_all=0.0)
    c["E6_alumina_eps0.3"] = dict(material="alumina", prop={"_eps": 0.3})
    for r in rings:
        for f in (0.0, 0.25, 0.5, 2.0, 4.0):
            if not (r == "v1" and f == 0.0):
                c[f"P1_{r}_all_x{f:g}"] = dict(ring=r, perf_all=f)
        for f in (0.25, 0.5, 2.0, 4.0):
            if not (r == "v1" and f in (0.5, 2.0)):
                c[f"P2_{r}_choroid_x{f:g}"] = dict(ring=r, perf_choroid=f)
    for r in rings:
        for m in mats:
            block = "F1" if (r, m) in (("v1", "Al"), ("v2", "Al"), ("v1", "alumina")) else "F2"
            c[f"{block}_{r}_{m}_exvivo"] = dict(ring=r, material=m, ex_vivo=True)
    for h in (15.0, 60.0, 120.0):
        c[f"G1_hoil{h:g}"] = dict(ex_vivo=True, h_oil=h)
    c["G1_oilfixed"] = dict(ex_vivo=True, oil_fixed=True)
    c["G4_exvivo_choroid_perfusion"] = dict(ex_vivo=True, choroid_perfusion=True)
    for r in rings:
        for m in ("coarse", "coarser"):
            c[f"H1_{r}_{m}"] = dict(ring=r, mesh=m)
    for m in ("coarse", "coarser"):
        c[f"H3_exvivo_{m}"] = dict(ex_vivo=True, mesh=m)
    done = dict(c)                                   # ---- everything below was added for the complete study
    for r in rings:                                  # all 27 operating points, with rewarming
        for Tb in (35, 37, 39):
            for T0 in (10, 25, 35):
                for Tc in (4, 8, 12):
                    c[f"D27_{r}_Tb{Tb}_T0{T0}_Tc{Tc}"] = dict(ring=r, Tb=float(Tb), T0=float(T0), Tc=float(Tc),
                                                               rewarm_min=35.0)
    for r in rings:                                  # set-point curve
        for Tc in range(0, 21, 2):
            c[f"S_{r}_Tc{Tc}"] = dict(ring=r, Tc=float(Tc))
    for r in rings:                                  # other ring materials with the silicone layer
        for m in ("alumina", "silica"):
            for sl in ("0.5mm", "1mm"):
                c[f"M_{r}_{m}_silicone{sl}"] = dict(ring=r, material=m, silicone=sl)
    for r in ("v2", "v3"):                           # sensitivities on the other two rings
        for h in (5.0, 20.0):
            c[f"E1_{r}_hair{h:g}"] = dict(ring=r, h_air=h)
        if r != "v3":
            for q in (0.0, -40.0, -100.0):
                c[f"E2_{r}_evap{abs(q):g}"] = dict(ring=r, evap=q)
    for r in rings:
        for tag, key in (("retina", "perf_retina"), ("orbit", "perf_orbit")):
            for f in (0.25, 0.5, 2.0, 4.0):
                if not (r == "v1" and f in (0.5, 2.0)):
                    c[f"P_{r}_{tag}_x{f:g}"] = {"ring": r, key: f}
    for r in rings:                                  # ex vivo: every ring and material at every oil condition
        for m in mats:
            for h in (15.0, 60.0, 120.0, None):
                if (r, m) != ("v1", "Al"):
                    c[f"X_{r}_{m}_" + (f"hoil{h:g}" if h else "oilfixed")] = dict(
                        ring=r, material=m, ex_vivo=True, **(dict(h_oil=h) if h else dict(oil_fixed=True)))
    for r, m in (("v1", "Al"), ("v2", "Al"), ("v1", "alumina")):      # ex vivo, the three experiments
        base = dict(ring=r, material=m, ex_vivo=True)
        for tag, kw in (("room21", dict(T_room=21.0)), ("room25", dict(T_room=25.0)), ("hair5", dict(h_air=5.0)),
                        ("hair20", dict(h_air=20.0)), ("bath37.3", dict(T_bath=37.3)), ("ramp60", dict(ramp_s=60.0)),
                        ("ramp120", dict(ramp_s=120.0)), ("ramp180", dict(ramp_s=180.0))):
            c[f"XS_{r}_{m}_{tag}"] = dict(base, **kw)
    for r in rings:                                  # mesh levels for the other materials and the experiments
        for m in ("alumina", "silica"):
            for lvl in ("coarse", "coarser"):
                c[f"H_{r}_{m}_{lvl}"] = dict(ring=r, material=m, mesh=lvl)
    for r, m in (("v2", "Al"), ("v1", "alumina")):
        for lvl in ("coarse", "coarser"):
            c[f"H_{r}_{m}_exvivo_{lvl}"] = dict(ring=r, material=m, ex_vivo=True, mesh=lvl)
    for r in rings:                                  # 90 minutes, to show the steady state
        c[f"L_{r}_invivo_90min"] = dict(ring=r, t_cool_min=90.0)
        c[f"L_{r}_exvivo_90min"] = dict(ring=r, ex_vivo=True, t_cool_min=90.0)
    for ex in (False, True):                         # verification: time step, start state, far boundary
        tag = "exvivo" if ex else "invivo"
        for f in (0.5, 0.25):
            c[f"V5_{tag}_dt_x{f:g}"] = dict(ex_vivo=ex, dt_scale=f)
        c[f"V8_{tag}_start_drift"] = dict(ex_vivo=ex, no_cooling=True)
    for h in (20.0, 100.0):
        c[f"V10_shell_h{h:g}"] = dict(shell_h=h)
    done2 = dict(c)
    # ex vivo at lower bath temperatures (2026-10-08: the porcine runs started near 34.5 C); human geometry
    for Tbath in (33.0, 34.5, 36.0):
        for h in (30.0, 60.0, 120.0, None):
            c[f"XP_bath{Tbath:g}_" + (f"hoil{h:g}" if h else "oilfixed")] = dict(
                ex_vivo=True, T_bath=Tbath, **(dict(h_oil=h) if h else dict(oil_fixed=True)))
    assert list(c)[:len(done2)] == list(done2)
    done3 = dict(c)
    # finer steps of the oil coefficient for the three experiments (the first sweep put the best value between 30 and 120)
    for r, m in (("v1", "Al"), ("v2", "Al"), ("v1", "alumina")):
        for h in (40.0, 45.0, 50.0, 55.0, 65.0, 70.0, 80.0, 100.0):
            c[f"XF_{r}_{m}_hoil{h:g}"] = dict(ring=r, material=m, ex_vivo=True, h_oil=h)
    assert list(c)[:len(done3)] == list(done3)
    done4 = dict(c)
    # oil coefficient x air coefficient x coupling of the unpowered device to the room, for the three experiments:
    # the start state depends on all three, the cooling mostly on the oil (2026-10-08)
    for r, m in (("v1", "Al"), ("v2", "Al"), ("v1", "alumina")):
        for ho in (45.0, 60.0, 80.0):
            for ha in (10.0, 20.0, 40.0):
                for hf in (0.0, 50.0, 200.0):
                    if not (ha == 10.0 and hf == 0.0 and ho in (45.0, 80.0)):       # those are XF cases
                        c[f"XG_{r}_{m}_oil{ho:g}_air{ha:g}_face{hf:g}"] = dict(
                            ring=r, material=m, ex_vivo=True, h_oil=ho, h_air=ha, h_face=hf)
    for r in rings:                                  # the same coupling in vivo
        for hf in (50.0, 200.0):
            c[f"E7_{r}_face{hf:g}"] = dict(ring=r, h_face=hf)
    assert list(c)[:len(done4)] == list(done4)
    done5 = dict(c)
    # V6: systematic refinement (every element split into eight) of each of COMSOL's meshes
    for r in rings:
        for lvl in ("coarser", "coarse", "production"):
            c[f"V6_{r}_{lvl}_x8"] = dict(ring=r, mesh=lvl, refine=1)
    for lvl in ("coarser", "coarse", "production"):
        c[f"V6_exvivo_{lvl}_x8"] = dict(ex_vivo=True, mesh=lvl, refine=1)
    c["V6_v1_coarser_x64"] = dict(mesh="coarser", refine=2)
    c["V6_exvivo_coarser_x64"] = dict(ex_vivo=True, mesh="coarser", refine=2)
    assert list(c)[:len(done5)] == list(done5)
    done6 = dict(c)
    # follow-up of the ex-vivo validation (the 2026-10-08 exvivo_validation check, section 3):
    # the starts are about 1.6 C high at oil 60 at every deep site, the ring falls exponentially, and the
    # alumina ring starts colder than measured
    EXP = (("v1", "Al", 54.0), ("v2", "Al", 36.0), ("v1", "alumina", 76.0))     # tau = measured ring t90 / ln(10)
    for r, m, tau in EXP:
        base = dict(ring=r, material=m, ex_vivo=True)
        for Tbath in (35.0, 35.5, 36.0):                     # a bath (or the oil at the eye) cooler than 37 C
            for ho in (50.0, 60.0, 70.0):
                c[f"XB_{r}_{m}_bath{Tbath:g}_oil{ho:g}"] = dict(base, T_bath=Tbath, h_oil=ho)
        for ho in (30.0, 60.0):                              # exponential set point with the measured time constant
            c[f"XR_{r}_{m}_exp{tau:g}_oil{ho:g}"] = dict(base, ramp_tau=tau, h_oil=ho)
        c[f"XR_{r}_{m}_exp{tau:g}_oil60_bath35.5"] = dict(base, ramp_tau=tau, h_oil=60.0, T_bath=35.5)
    for eps in (0.3, 0.8):                                   # alumina emissivity, ex vivo
        for ho in (30.0, 60.0):
            c[f"XE_v1_alumina_eps{eps:g}_oil{ho:g}"] = dict(ring="v1", material="alumina", ex_vivo=True, h_oil=ho,
                                                            prop={"_eps": eps})
    for r in rings:                                          # the same exponential set point in vivo
        c[f"E8_{r}_exp54"] = dict(ring=r, ramp_tau=54.0)
    assert list(c)[:len(done6)] == list(done6)
    done7 = dict(c)
    # V10 follow-up (results/gates_check_2026-10-08.txt): the deep sites depend on the outer 37 C shell, so map it
    for r in rings:
        for h in (50.0, 300.0, 1000.0, 5000.0):
            c[f"V10_{r}_shell_h{h:g}"] = dict(ring=r, shell_h=h)
    for r in ("v2", "v3"):
        for h in (20.0, 100.0):
            c[f"V10_{r}_shell_h{h:g}"] = dict(ring=r, shell_h=h)
    for h in (20.0, 100.0, 1000.0):                  # and with perfusion halved or doubled, since the two interact
        for f in (0.5, 2.0):
            c[f"V10_v1_shell_h{h:g}_perf_x{f:g}"] = dict(shell_h=h, perf_all=f)
    assert list(c)[:len(done7)] == list(done7)
    # further cases from data files, in file-name order (dev/inputs/cases_*.json: {case id: settings}); these come
    # last, so the numbers of the cases above never change
    import glob
    for path in sorted(glob.glob(os.path.join(HERE, "inputs", "cases_*.json"))):
        with open(path, encoding="utf-8") as f:
            extra = json.load(f)
        for cid, st in extra.items():
            if cid in c:
                raise ValueError(f"{os.path.basename(path)}: case id {cid} is already defined")
            unknown = set(st) - set(DEFAULTS)
            if unknown:
                raise ValueError(f"{os.path.basename(path)}: case {cid} has unknown settings {sorted(unknown)}")
            c[cid] = st
    assert list(c)[:len(done)] == list(done)         # earlier case numbers must not change
    return c


# not yet runnable from this file, listed so that --list shows the whole plan
# (2026-10-22, the property ranges, X5, V6, V3, V4 and G5 are implemented since 2026-10-08/-16 and
# were removed from this list; V4 is logged by every run since DRIVER_VERSION 2, older runs need dev/post_energy.py)
PENDING = {"bridge step with only the start state changed (V9)": "needs the air and module layer kept",
           "silicone mesh levels": "no coarser silicone meshes are stored",
           "perfusion falling with temperature (P3)": "law not chosen"}


def mesh_variant():
    """The whole case list on another set of meshes: environment variable SCR_MESH_VARIANT names a set of
    dev/inputs/mesh_variants.json, {set: {"<ring>|<silicone or none>|<invivo or exvivo>": mesh file in dev/inputs}}.
    Returns (name, table) or (None, {})."""
    name = os.environ.get("SCR_MESH_VARIANT")
    if not name:
        return None, {}
    with open(os.path.join(HERE, "inputs", "mesh_variants.json"), encoding="utf-8") as f:
        return name, json.load(f)[name]


def variant_key(s):
    return f"{s['ring']}|{s['silicone'] or 'none'}|{'exvivo' if s['ex_vivo'] else 'invivo'}"


def variant_file(s):
    """Mesh file of the variant for these settings; None if the case keeps its own mesh (it is itself a mesh
    check: other mesh level, own mesh file or refinement) or the variant has no mesh for this geometry."""
    name, table = mesh_variant()
    if not name or s["mesh"] != "production" or s["mesh_file"] or s["refine"]:
        return None
    return table.get(variant_key(s))


def settings_of(cid, mesh=None):
    s = dict(DEFAULTS)
    s.update(cases()[cid])
    if mesh:
        s["mesh"] = mesh
    vf = variant_file(s)
    if vf:
        s["mesh_file"] = vf
    return s


# ---------------------------------------------------------------- transform
def _doms(prob, *keys):
    return sorted(d for d, n in prob.domain_names.items() if any(k.lower() == n.lower() or k.lower() in n.lower()
                                                                  for k in keys))


def inert_names_for(s):
    """Override names listed as known no-ops (dev/inputs/inert_overrides.json) for a case with these settings:
    matched on ring and the exact prop dict, so callers that pass settings only (dev/post_energy.py, mesh_adapt)
    rebuild the registered cases unchanged."""
    out = set()
    for cid, v in inert_overrides().items():
        c = cases().get(cid)
        if c and c.get("ring", DEFAULTS["ring"]) == s["ring"] and c.get("prop", {}) == s["prop"]:
            out |= set(v["names"])
    return out


def transform(prob, s, log, inert=None):
    """Turn the problem read from the COMSOL file into the new set-up. Returns (prob, info).
    inert: override names known to match no domain of this mesh (default: inert_names_for(s)); any other
    unmatched name raises."""
    if inert is None:
        inert = inert_names_for(s) if s["prop"] else set()
    names = prob.domain_names
    air, tec, ring = _doms(prob, "Air"), _doms(prob, "TEC Layer"), _doms(prob, "Ring")
    canal, vit = _doms(prob, "Cloquet"), _doms(prob, "Vitreous")
    orbit = _doms(prob, *ORBIT)
    if not (tec and ring and vit):
        raise RuntimeError(f"expected domains not found: TEC {tec}, ring {ring}, vitreous {vit}")
    drop = set(air + tec + (orbit if s["ex_vivo"] else []))
    tris, up, dn = prob.tris, prob.tri_up, prob.tri_down

    def faces(a, b):            # triangles between a domain of set a and a domain of set b
        a, b = list(a), list(b)
        return (np.isin(up, a) & np.isin(dn, b)) | (np.isin(up, b) & np.isin(dn, a))

    kept = [d for d in names if d not in drop]
    f_cool = faces(ring, tec)
    f_air = faces(kept, air)
    globe = [d for d in kept if d not in ring and names[d] != "Silicone"]
    f_oil = np.zeros(len(tris), bool)
    f_ring_lid = np.zeros(len(tris), bool)
    if s["ex_vivo"]:
        f_oil = faces(globe, orbit)
        f_ring_lid = faces(ring, orbit)
        # faces of the globe on the old outer boundary but not on the plane of symmetry (end of the nerve stump)
        ext = ((up == 0) | (dn == 0)) & (np.isin(up, globe) | np.isin(dn, globe))
        off_plane = np.abs(prob.points[tris][:, :, 1]).max(1) > 1e-7
        f_oil |= ext & off_plane

    # ---- remove elements and renumber nodes
    keep_el = ~np.isin(prob.dom, list(drop))
    tets, dom = prob.tets[keep_el], prob.dom[keep_el].copy()
    used = np.unique(tets)
    new = np.full(prob.points.shape[0], -1, np.int64)
    new[used] = np.arange(len(used))
    pts = prob.points[used]
    tets = new[tets].astype(prob.tets.dtype)
    for d in canal:
        dom[dom == d] = vit[0]

    def tri_map(mask_or_tris):
        t = tris[mask_or_tris] if mask_or_tris.dtype == bool else mask_or_tris
        t = new[t]
        return t[(t >= 0).all(1)]

    def node_map(nodes):
        n = new[np.asarray(nodes)]
        return n[n >= 0]

    info = dict(nodes=len(pts), elements=len(tets), removed=sorted(names[d] for d in drop),
                cooling_face_triangles=int(f_cool.sum()), air_faces=int(f_air.sum()), oil_faces=int(f_oil.sum()))
    log(f"  removed {info['removed']}; {len(tets)} elements, {len(pts)} nodes left")
    log(f"  cooling face {f_cool.sum()} triangles, air {f_air.sum() + f_ring_lid.sum()}, oil {f_oil.sum()}")
    if not f_cool.any():
        raise RuntimeError("no face between the ring and the module layer: cooling face not found")

    Tb, T0, Tc = s["Tb"] + K0, s["T0"] + K0, s["Tc"] + K0
    room = (s["T_room"] + K0) if s["ex_vivo"] else T0
    mat = s["material"]
    eps_ring = (s["prop"].get("_eps", EPS[mat]) if s["eps_from_material"] else 0.05)

    # ---- fixed temperatures: body shell stays (in vivo); air wall and module conditions go
    dirichlet = []
    shell = []
    if not s["ex_vivo"]:
        for d in prob.dirichlet:
            if d["name"] == "Temperature Body":
                if s["shell_h"] or s["shell_regions"]:
                    shell.append(tri_map(np.isin(prob.tri_bnd, d["boundaries"])))
                else:
                    dirichlet.append(dict(name=d["name"], nodes=node_map(d["nodes"]), value=(lambda t, v=Tb: v),
                                          steps=["start", "cool", "rewarm"]))
        if not dirichlet and not shell:
            raise RuntimeError("body-temperature boundary not found")
    cool_nodes = np.unique(tri_map(f_cool))
    cool = dict(name="Cooling face", nodes=cool_nodes, value=None,                     # value set after the start state
                steps=[] if s["no_cooling"] else ["cool"])
    dirichlet.append(cool)

    # ---- radiation and convection on surfaces in air
    air_tris = tri_map(f_air | f_ring_lid)
    ring_new = set(np.unique(tets[np.isin(dom, ring)]).tolist())
    on_ring = np.array([all(n in ring_new for n in t) for t in air_tris.tolist()], bool) if len(air_tris) else np.zeros(0, bool)
    radiation = []
    if on_ring.any():
        radiation.append(dict(name="Radiation ring", tris=air_tris[on_ring], eps=eps_ring, t_amb=room))
    if (~on_ring).any():
        radiation.append(dict(name="Radiation tissue", tris=air_tris[~on_ring], eps=0.98, t_amb=room))
    convective = [dict(name="Air", tris=air_tris, h=(lambda Ts, h=s["h_air"]: np.full_like(Ts, h)), t_ext=room)]
    if s["h_face"] > 0:
        convective.append(dict(name="Module and heat exchanger (unpowered)", tris=tri_map(f_cool),
                               h=(lambda Ts, h=s["h_face"]: np.full_like(Ts, h)), t_ext=room))
    for tr in shell:
        if s["shell_regions"]:
            sys.path.insert(0, HERE)
            import outer_boundary as ob
            reg = ob.region_of(pts[tr].mean(1) * 1e3)
            for r, name in enumerate(ob.REGIONS):
                if (reg == r).any():
                    convective.append(dict(name=f"Body shell ({name})", tris=tr[reg == r],
                                           h=(lambda Ts, h=float(s["shell_regions"][name]): np.full_like(Ts, h)),
                                           t_ext=Tb))
        else:
            convective.append(dict(name="Body shell (soft)", tris=tr,
                                   h=(lambda Ts, h=s["shell_h"]: np.full_like(Ts, h)), t_ext=Tb))
    bath = s["T_bath"] + K0
    if s["ex_vivo"] and f_oil.any():
        oil_tris = tri_map(f_oil)
        # "fixed" = a very large coefficient (H_OIL_FIXED).  Fixing the temperature of the oil-face nodes themselves
        # also fixed the ring's nodes on the line where sclera, ring and oil meet (found by the energy balance,
        # 2026-10-08: 115 W through the cooling face), so results of oil_fixed cases from before DRIVER_VERSION 2
        # are invalid and are run again.
        h_oil = H_OIL_FIXED if s["oil_fixed"] else s["h_oil"]
        convective.append(dict(name="Oil bath", tris=oil_tris, h=(lambda Ts, h=h_oil: np.full_like(Ts, h)), t_ext=bath))

    # ---- tear evaporation
    source = []
    if not s["ex_vivo"]:
        for src in prob.surface_source:
            if "evap" in src["name"].lower() and s["evap"] != 0.0:
                source.append(dict(name=src["name"], tris=tri_map(src["tris"]), q=float(s["evap"])))

    # ---- materials
    k = {d: v for d, v in prob.k.items() if d not in drop and d not in canal}
    rho_cp = {d: v for d, v in prob.rho_cp.items() if d not in drop and d not in canal}
    over = {}
    if s["workbook"]:
        over.update({n: dict(v) for n, v in WORKBOOK.items()})
    case_names = [n for n in s["prop"] if not n.startswith("_")]
    for n in case_names:
        v = s["prop"][n]
        bad = set(v) - {"k", "rho", "cp"}
        if bad or ("rho" in v) != ("cp" in v):
            # rho without cp, or cp without rho, was silently ignored before 2026-10-22
            raise ValueError(f"property override {n}: {sorted(v)}; give k and/or rho together with cp")
        over.setdefault(n, {}).update(v)
    inert_used = []
    for n, v in over.items():
        hit = [d for d in _doms(prob, n) if d not in drop]
        if not hit and n in case_names:
            if n in inert:
                inert_used.append(n)
                log(f"  override of '{n}' matches no domain of this mesh: no-op (dev/inputs/inert_overrides.json)")
                continue
            raise ValueError(f"property override '{n}' matches no kept domain of this mesh "
                             f"({sorted(names[d] for d in names if d not in drop)}); it would be a silent no-op")
        for d in hit:
            if "k" in v:
                k[d] = float(v["k"])
            if "rho" in v and "cp" in v:
                rho_cp[d] = float(v["rho"]) * float(v["cp"])

    # (2026-10-24) a constant override of k (or of rho*cp) on a temperature-dependent material left the other
    # property a function, and replica.solve.Heat then treated the domain as constant and failed (float(function)).
    # The constant one is wrapped as a function of T, so the domain stays temperature-dependent with that value fixed.
    for d in list(k):
        if callable(k[d]) != callable(rho_cp.get(d)) and d in rho_cp:
            if not callable(k[d]):
                k[d] = (lambda T, _v=float(k[d]): np.full(np.shape(T), _v))
            else:
                rho_cp[d] = (lambda T, _v=float(rho_cp[d]): np.full(np.shape(T), _v))

    # ---- perfusion
    perfusion, t_blood, q_met = {}, {}, {}
    for d, w in prob.perfusion.items():
        if d in drop:
            continue
        n = names[d]
        f = s["perf_all"]
        f *= s["perf_choroid"] if n == "Choroid" else s["perf_retina"] if n == "Retina" else \
            s["perf_ciliary"] if n == "Ciliary Body" else s["perf_orbit"] if n in ORBIT else 1.0
        on =(not s["ex_vivo"]) or (s["choroid_perfusion"] and n == "Choroid")
        if on and f > 0:
            perfusion[d], t_blood[d] = w * f, Tb
            q_met[d] = prob.q_met[d] if s["perf_all"] > 0 and not s["ex_vivo"] else 0.0

    # ---- probes
    probes = {}
    for n, p in prob.probes.items():
        if p["kind"] == "point":
            probes[n] = p
        else:
            ds = sorted({vit[0] if d in canal else d for d in p["domains"] if d not in drop})
            if ds:
                probes[n] = dict(kind=p["kind"], domains=ds)
    for kind, label in (("average", "Ring average"), ("minimum", "Ring minimum"), ("maximum", "Ring maximum")):
        probes[label] = dict(kind=kind, domains=ring)

    t_init = np.where(np.isnan(prob.t_init[used]), Tb, prob.t_init[used])
    t_init[:] = bath if s["ex_vivo"] else Tb

    keep_tri = (new[tris] >= 0).all(1)
    from replica.model import Problem
    out = Problem(points=pts, tets=tets, dom=dom, n_original=len(pts), k=k, rho_cp=rho_cp,
                  variable={d for d in prob.variable if d not in drop and callable(k[d])},
                  perfusion=perfusion, t_blood=t_blood, q_met=q_met, t_init=t_init, dirichlet=dirichlet,
                  radiation=radiation, surface_source=source, probes=probes,
                  domain_names={d: n for d, n in names.items() if d not in drop and d not in canal},
                  tris=new[tris[keep_tri]], tri_bnd=prob.tri_bnd[keep_tri],
                  # (2026-10-22) the merged Cloquet's canal takes the vitreous id in the face
                  # arrays too, so canal | vitreous faces are interior faces of one domain, not walls (the heat
                  # equation does not read tri_up / tri_down; the flow solver and mesh.npz readers do)
                  tri_up=np.where(np.isin(up[keep_tri], list(drop)), 0,
                                  np.where(np.isin(up[keep_tri], canal), vit[0], up[keep_tri])),
                  tri_down=np.where(np.isin(dn[keep_tri], list(drop)), 0,
                                    np.where(np.isin(dn[keep_tri], canal), vit[0], dn[keep_tri])),
                  flow=None, convective=convective, notes=list(prob.notes), params=dict(prob.params), selections={})
    info.update(eps_ring=eps_ring, room_C=room - K0, perfused=sorted(names[d] for d in perfusion))
    # what was applied, per domain, not only the labels of the settings
    def val(x):
        return float(x) if isinstance(x, (int, float, np.floating)) else "temperature dependent (model expression)"
    ct = tri_map(f_cool)
    face_area = float(0.5 * np.linalg.norm(np.cross(pts[ct[:, 1]] - pts[ct[:, 0]], pts[ct[:, 2]] - pts[ct[:, 0]]),
                                           axis=1).sum()) if len(ct) else 0.0
    info.update(applied={names[d]: dict(k_W_mK=val(k[d]), rho_cp_J_m3K=val(rho_cp[d]),
                                        perfusion_W_m3K=float(perfusion.get(d, 0.0)), T_blood_C=(t_blood[d] - K0) if d in t_blood else None,
                                        Q_met_W_m3=float(q_met.get(d, 0.0)))
                         for d in sorted(k)},
                ring_source=dict(material=mat, switch_index=MATERIAL_INDEX[mat], eps=eps_ring,
                                 eps_rule="material" if s["eps_from_material"] else "0.05 for every ring"),
                cooling_face_area_half_m2=face_area, air_faces_with_ring_lid=int(f_air.sum() + f_ring_lid.sum()),
                inert_overrides=inert_used,
                metabolism_removed=bool(s["perf_all"] == 0.0 and not s["ex_vivo"]))
    if s["perf_all"] == 0.0 and not s["ex_vivo"]:
        log("  perf_all = 0: perfusion AND metabolic heat removed (E5 / P1 x0 are 'no blood flow, no metabolism')")
    return out, cool, info


# ------------------------------------------------------------------ one case
def run_one(cid, out_root, mesh=None, dry=False):
    import run_all
    from replica import model, solve
    from replica.mphread import Mph
    import sensor_probes

    s = settings_of(cid, mesh)
    if mesh and (s["silicone"] or s["mesh_file"]):
        # the override would label the folder __<mesh> but solve the case's own mesh
        raise ValueError(f"--mesh {mesh} has no effect on {cid} (silicone or mesh_file case): refused")
    out_dir = os.path.join(out_root, cid if not mesh else f"{cid}__{mesh}")
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def log(m):
        lines.append(m)
        print(m, flush=True)

    t0 = time.time()
    rel = SILICONE_FILE[s["silicone"]] if s["silicone"] else MESH_FILE[s["mesh"]]
    path = run_all.mph_path(rel)
    if not os.path.exists(path):                 # Linux: no drive D:, and the sub-folder is written with a backslash
        path = os.path.join(run_all.BOX, "Projects", "SCR", *rel.split("\\"))
    log(f"case {cid}: {os.path.basename(path)}, ring {s['ring']}, {s['material']}, "
        + ("ex vivo" if s["ex_vivo"] else "in vivo"))
    mph = Mph(path)
    comp = next(c for c, v in mph.components().items() if v["label"].endswith(s["ring"]))
    info = mph.components()[comp]
    extra = sensor_probes.sensor_probes()
    donor = None
    cad = None                                   # a mesh made from the STEP files (dev/cad_mesh.py): see dev/cad_base.py
    if s["mesh_file"]:
        dp = os.path.join(HERE, "inputs", *s["mesh_file"].split("/"))
        with np.load(dp, allow_pickle=True) as z:
            if "domain_names" in z.files:
                cad = {k: z[k] for k in z.files}
            else:
                donor = {k: z[k] for k in ("points", "tets", "tet_dom", "tris", "tri_bnd", "tri_up", "tri_down")}
                ext = {k: z[k] for k in ("ext_layer", "ext_region", "ext_depths_mm") if k in z.files}
        log(f"  mesh taken from {s['mesh_file']}: {len((cad or donor)['tets'])} elements"
            + (" (made from the STEP file)" if cad else " (see dev/inputs/README_mesh_adapted.md)"))
    if cad is not None and info["geom"] not in mph.mesh_resources():
        donor = dict(np.load(os.path.join(HERE, "inputs", f"mesh_silicone{s['silicone']}_{s['ring']}.npz")))
    elif cad is not None:
        pass                                     # the stored COMSOL mesh gives materials and sensor points only
    elif s["mesh_file"]:
        pass
    elif info["geom"] not in mph.mesh_resources():               # this file stores no mesh for this ring
        dp = os.path.join(HERE, "inputs", f"mesh_silicone{s['silicone']}_{s['ring']}.npz")
        if not os.path.exists(dp):
            raise RuntimeError(f"{os.path.basename(path)} stores no mesh for {s['ring']} and {dp} does not exist")
        donor = dict(np.load(dp))
        log(f"  mesh taken from {os.path.basename(dp)} (see dev/inputs/README.md)")
    # the mesh on which the scleral sensor points are placed: the stored one, so they do not move when the
    # mesh is refined or replaced (None = the mesh in use is the stored one)
    snap_mesh = None
    if s["refine"] or s["mesh_file"]:
        if info["geom"] in mph.mesh_resources():
            snap_mesh = mph.mesh(mph.mesh_resources()[info["geom"]]["resource"])
        elif not s["mesh_file"]:
            snap_mesh = donor
    if s["refine"] and cad is not None:
        sys.path.insert(0, HERE)
        import metric_tools
        keep = {k: cad[k] for k in ("domain_ids", "domain_names")}
        for _ in range(int(s["refine"])):
            cad = metric_tools.refine_uniform({k: cad[k] for k in ("points", "tets", "tet_dom", "tris", "tri_bnd", "tri_up", "tri_down")})
        cad = dict(cad, **keep)
        log(f"  every element split into eight, {s['refine']} time(s): {len(cad['tets'])} elements")
    elif s["refine"]:
        sys.path.insert(0, HERE)
        import metric_tools
        donor = donor or mph.mesh(mph.mesh_resources()[info["geom"]]["resource"])
        for _ in range(int(s["refine"])):
            donor = metric_tools.refine_uniform(donor)
        log(f"  every element split into eight, {s['refine']} time(s): {len(donor['tets'])} elements")
    base = model.build(mph, component=comp, heat=info["heat"], geom=info["geom"], insulation_slit=False, mesh=donor,
                       overrides=dict(T0=s["T0"] + K0, Tb=s["Tb"] + K0, Tc=s["Tc"] + K0),
                       switch_index=MATERIAL_INDEX[s["material"]], extra_probes=extra)
    if snap_mesh is None or cad is not None:     # (CAD: base is on the stored COMSOL mesh at this point)
        extra.update(sensor_probes.snap_to_sclera(base, extra))
    else:
        import types
        extra.update(sensor_probes.snap_to_sclera(types.SimpleNamespace(
            domain_names=base.domain_names, points=snap_mesh["points"], n_original=len(snap_mesh["points"]),
            tris=snap_mesh["tris"], tri_up=snap_mesh["tri_up"], tri_down=snap_mesh["tri_down"]), extra))
    base.probes.update(extra)
    comsol_base = base                           # the set-point ramp is read from the COMSOL problem
    if cad is not None:
        sys.path.insert(0, HERE)
        import cad_base
        base = cad_base.base_from_cad(cad, base, log)
    prob, cool, tinfo = transform(base, s, log)
    if s["ext_stacks"]:
        if s["refine"] or not s["mesh_file"]:
            raise RuntimeError("ext_stacks needs an extended mesh_file and refine = 0")
        sys.path.insert(0, HERE)
        import outer_boundary as ob
        ob.apply_stacks(prob, ext["ext_layer"], ext["ext_region"], ext["ext_depths_mm"], s["ext_stacks"],
                        _doms(base, "Fat")[0], s["Tb"])
        log(f"  tissue beyond the shell: set {s['ext_stacks']}, domains {sorted(d for d in prob.k if d > 100)}")
    if s["stump_mm"]:
        if not s["ex_vivo"]:
            raise RuntimeError("stump_mm is for ex-vivo cases (dev/nerve_stump.py)")
        sys.path.insert(0, HERE)
        import nerve_stump
        prob, cool, tinfo["stump"] = nerve_stump.shorten(prob, cool, float(s["stump_mm"]), rule=s["stump_rule"], log=log)
    sys.path.insert(0, HERE)
    import manifest
    picks = getattr(sensor_probes, "PICKS", None)
    record = dict(case=cid, settings=s, model_file=os.path.basename(path), component=comp,
                  driver_version=DRIVER_VERSION, label=label_of(cid),
                  code_stamp_dev=manifest.code_stamp(), code_stamp_run_all=run_all.code_version().split()[0],
                  sensor_picks=dict(path=picks, sha256=manifest.sha256(picks) if picks and os.path.isfile(picks) else None),
                  mesh_file_sha256=(manifest.sha256(os.path.join(HERE, "inputs", *s["mesh_file"].split("/")))
                                    if s["mesh_file"] else None), **tinfo)
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    if dry:
        log("  dry run: problem built, not solved")
        return

    h = solve.Heat(prob, log=log)
    h.probe_setup()
    # a point probe outside the kept mesh is read by extrapolation from the nearest element: say so
    outside = {n: float(v[4]) for n, v in getattr(h, "_pp", {}).items() if v and v[0] == "point" and v[4] < -1e-6}
    if outside:
        log(f"  WARNING: {len(outside)} point probe(s) outside the mesh (smallest barycentric coordinate): "
            + ", ".join(f"{n} {w:.3g}" for n, w in sorted(outside.items())))
    record["probes_outside_mesh"] = outside
    with open(os.path.join(out_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    names = list(prob.probes)
    T = h.stationary(step="start")
    face0 = float(T[cool["nodes"]].mean())
    Tc = s["Tc"] + K0
    # the model's own ramp shape (Temperature TEC (t): T0 - (T0 - Tc) * step(t / t_ramp)), stretched to ramp_s
    old = next(d for d in comsol_base.dirichlet if "step" in d.get("expr", ""))
    T0k, t_ramp = s["T0"] + K0, float(base.params.get("t_ramp", 90.0))
    span = (T0k - Tc) if abs(T0k - Tc) > 1e-9 else 1.0

    def ramp(t, f=t_ramp / s["ramp_s"]):
        return min(1.0, max(0.0, (T0k - old["value"](t * f)) / span)) if abs(T0k - Tc) > 1e-9 else float(t > 0)

    if s["ramp_tau"]:
        cool["value"] = lambda t, a=face0, b=Tc, tau=float(s["ramp_tau"]): float(b + (a - b) * np.exp(-t / tau))
    else:
        cool["value"] = lambda t, a=face0, b=Tc: float(a + (b - a) * ramp(t))
    start = h.probes(T)
    log(f"  start state: cooling face {face0 - K0:.2f} C, ring average {start['Ring average'] - K0:.2f} C")

    # the whole temperature field is kept at every time step, so nothing has to be chosen in advance
    fields = {"T_start": T.astype(np.float32)}

    def march(T, step, minutes):
        rows, last, snaps = [], {}, []

        def cb(t, Tn):
            p = h.probes(Tn)
            # heat through the cooling face (whole device, W): reaction of the constraint, first-order in time
            q = float("nan")
            if step == "cool" and "T" in last and t > last["t"]:
                Kv, Mv = h.variable_matrices(Tn)
                fr, _ = h.radiation(Tn)
                r = (h.K0 + Kv) @ Tn + (h.M0 + Mv) @ ((Tn - last["T"]) / (t - last["t"])) - h.F0 - fr
                q = -2.0 * float(r[cool["nodes"]].sum())
                if s["h_face"] > 0:
                    # the face-to-room coefficient also acts on the driven face, so the reaction
                    # carries q_room = 2 h_face A (T_room - T_face), heat from the room and not from the eye:
                    # h_face in W/m2K, A the half-model face area (m2), temperatures in K, 2 for the whole device
                    q -= 2.0 * s["h_face"] * tinfo["cooling_face_area_half_m2"] * (
                        tinfo["room_C"] + K0 - float(Tn[cool["nodes"]].mean()))
            last["T"], last["t"] = Tn.copy(), t
            rows.append([t / 60.0] + [p[n] for n in names] + [q])
            snaps.append(Tn.astype(np.float32))

        stages = run_all.stages(minutes * 60.0)
        if step != "cool" and s["release_steps"]:
            # the cooling-face condition is switched off at t = 0 of this phase: one 6 s step across that leaves the
            # ring 0.2-0.3 C too cold at 6 s (reheat check of 2026-10-09); with these steps 0.005 C
            stages = [(6.0, 0.5), (18.0, 1.0), (60.0, 3.0)] + [(tu, dt) for tu, dt in stages if tu > 60.0]
        stages = [(tu, dt * s["dt_scale"]) for tu, dt in stages]
        if s["exact_stages"]:
            sys.path.insert(0, HERE)
            import transient_fixed
            stages, changed = transient_fixed.exact_stages(stages, minutes * 60.0)
            for c in changed:
                log(f"  {step}: stage {c[0]:g}-{c[1]:g} s is not a whole number of {c[2]:g} s steps: "
                    f"{c[3]} steps, then one of {c[4]:.6g} s")
        T = h.transient(T, minutes * 60.0, stages, step=step, callback=cb, tol=1e-5)
        with open(os.path.join(out_dir, f"probes_{'cooling' if step == 'cool' else 'rewarm'}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Time (min)"] + [f"{n} (K)" for n in names] + ["Cold-side heat, whole device (W)"])
            w.writerows(rows)
        fields[f"time_{step}_s"] = np.array([r[0] * 60.0 for r in rows])
        fields[f"T_{step}"] = np.stack(snaps)                      # (time steps, nodes)
        return T, rows

    T, rows = march(T, "cool", s["t_cool_min"])
    end = dict(zip(names, rows[-1][1:-1]))
    log(f"  end of cooling: ring average {end['Ring average'] - K0:.2f} C "
        f"(min {end['Ring minimum'] - K0:.2f}, max {end['Ring maximum'] - K0:.2f}), cold-side heat {rows[-1][-1]:.2f} W")
    if s["rewarm_min"] > 0:
        T, _ = march(T, "rewarm", s["rewarm_min"])
    np.savez_compressed(os.path.join(out_dir, "fields.npz"), **fields)
    np.savez_compressed(os.path.join(out_dir, "mesh.npz"), points=prob.points, tets=prob.tets, dom=prob.dom,
                        tris=prob.tris, tri_up=prob.tri_up, tri_down=prob.tri_down, cooling_face_nodes=cool["nodes"],
                        domain_ids=np.array(sorted(prob.domain_names)),
                        domain_names=np.array([prob.domain_names[d] for d in sorted(prob.domain_names)]))
    try:                                         # gate V4: energy balance and heat split from the saved fields
        sys.path.insert(0, HERE)
        import post_energy
        eb = post_energy.analyse(out_dir, out_dir, log=log, built=(prob, cool, s, cid, h))
        with open(os.path.join(out_dir, "energy_balance.json"), "w", encoding="utf-8") as f:
            json.dump(eb, f, indent=1)
        log(f"  energy balance: out through the cooling face {eb['W']['cooling face (out)']:.3f} W at the end, "
            f"{eb['J']['cooling face (out)']:.0f} J over the run; closure {eb['closure_W_pct']:+.3f} % at the end, "
            f"{eb['closure_J_pct']:+.3f} % over the run (gate V4: 1 %)")
    except Exception as e:                       # the run itself is finished and saved; only the check is missing
        log(f"  energy balance not computed: {type(e).__name__}: {e}")
    log(f"done in {time.time() - t0:.0f} s")
    log("code " + run_all.code_version())
    log(f"code+dev {record['code_stamp_dev']} (dev/manifest.py: run_all.py + replica/ + dev/, LF) driver {DRIVER_VERSION}")
    with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _child(cid, out_root, mesh, dry):
    """Entry point of the per-case process: never lets an error escape without a STATUS.json."""
    if dry:
        out_root = dry_root(out_root)
    out_dir = os.path.join(out_root, cid if not mesh else f"{cid}__{mesh}")
    if dry and state_of(cid, out_root, mesh)[0] == "ok":         # cannot happen under _dry; never replace ok by dry
        raise RuntimeError(f"{out_dir} holds a finished run: a dry run may not write there")
    os.makedirs(out_dir, exist_ok=True)
    st = dict(case=cid, mesh=mesh, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    try:
        run_one(cid, out_root, mesh, dry)
        st.update(state="dry" if dry else "ok")
    except BaseException as e:                                   # includes MemoryError and KeyboardInterrupt
        st.update(state="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(st["traceback"], flush=True)
    st.update(seconds=round(time.time() - t0), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out_dir, "STATUS.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    return 0 if st["state"] != "failed" else 1


# --------------------------------------------------------------------- batch
def dry_root(out_root):
    """dry runs write under <out>/_dry, never into the folder of a real run."""
    return out_root if os.path.basename(os.path.normpath(out_root)) == "_dry" else os.path.join(out_root, "_dry")


def take_lock(out_dir):
    """O_EXCL lock of one case folder; returns the lock path or None if another process holds it."""
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "RUNNING.lock")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    import socket
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                                job=os.environ.get("SLURM_JOB_ID"), task=os.environ.get("SLURM_ARRAY_TASK_ID"))) + "\n")
    return p


def duplicates():
    """{case id: first earlier case id with identical settings}."""
    seen, dup = {}, {}
    for cid in cases():
        key = json.dumps({**DEFAULTS, **cases()[cid]}, sort_keys=True, default=str)
        if key in seen:
            dup[cid] = seen[key]
        else:
            seen[key] = cid
    return dup


def state_of(cid, out_root, mesh=None):
    p = os.path.join(out_root, cid if not mesh else f"{cid}__{mesh}", "STATUS.json")
    if not os.path.exists(p):
        return "", None
    try:
        with open(p, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("state") == "ok" and cases().get(cid, {}).get("oil_fixed"):
            try:
                with open(os.path.join(os.path.dirname(p), "settings.json"), encoding="utf-8") as f:
                    if json.load(f).get("driver_version", 1) < 2:
                        return "stale", st           # made before the fixed-oil correction: run again
            except Exception:
                return "stale", st
        return st.get("state", "?"), st
    except Exception:
        return "unreadable", None


def write_status(out_root, mesh=None):
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "status.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "case", "state", "seconds", "finished", "error"])
        for i, cid in enumerate(cases()):
            state, st = state_of(cid, out_root, mesh)
            st = st or {}
            w.writerow([i, cid, state, st.get("seconds", ""), st.get("finished", ""), st.get("error", "")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--id", nargs="*", help="exact case ids (an unknown id is an error)")
    ap.add_argument("--case", nargs="*", help="substring match (warns when a pattern selects several cases)")
    ap.add_argument("--skip", nargs="*", default=[], help="leave out cases whose id contains one of these strings")
    ap.add_argument("--block", nargs="*")
    ap.add_argument("--index", type=int, nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mesh", choices=list(MESH_FILE), help="override the mesh level (testing)")
    ap.add_argument("--dry-run", action="store_true", help="build the problem, do not solve")
    ap.add_argument("--redo", action="store_true", help="run cases again even if they finished")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--run-one", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.run_one:
        sys.exit(_child(a.run_one, a.out, a.mesh, a.dry_run))

    ids = list(cases())
    if a.list:
        dup = duplicates()
        for i, cid in enumerate(ids):
            note = "  ".join(x for x in (label_of(cid), f"same settings as {dup[cid]}" if cid in dup else "") if x)
            print(f"{i:3d}  {cid:34s} {state_of(cid, a.out, a.mesh)[0]}" + (f"   [{note}]" if note else ""))
        print(f"{len(ids)} cases defined.  Not yet runnable:")
        for kk, v in PENDING.items():
            print(f"     {kk}: {v}")
        return
    if a.dry_run:
        a.out = dry_root(a.out)
        print(f"dry run: output under {a.out}", flush=True)
    todo = []
    if a.all:
        todo = ids
    if a.id:
        unknown = [x for x in a.id if x not in cases()]
        if unknown:
            sys.exit(f"unknown case id(s): {unknown}")
        todo += list(a.id)
    if a.case:
        for x in a.case:
            hit = [c for c in ids if x in c]
            if len(hit) > 1:
                print(f"WARNING: --case {x} is a substring match and selects {len(hit)} cases: "
                      + ", ".join(hit[:8]) + (" ..." if len(hit) > 8 else "") + "  (use --id for exact ids)", flush=True)
        todo += [c for c in ids if any(x in c for x in a.case)]
    if a.block:
        todo += [c for c in ids if c.split("_")[0] in a.block]
    if a.index:
        todo += [ids[i] for i in a.index]
    todo = list(dict.fromkeys(todo))
    todo = [c for c in todo if not any(x in c for x in a.skip)]
    if mesh_variant()[0]:                        # only the cases that this mesh set replaces the mesh of
        todo = [c for c in todo if (lambda s0: variant_file(s0))({**DEFAULTS, **cases()[c]})]
    if not a.redo and not a.dry_run:
        todo = [c for c in todo if state_of(c, a.out, a.mesh)[0] != "ok"]
    if not todo:
        print("nothing to run (use --list, --case, --block, --index or --all)")
        return
    print(f"{len(todo)} case(s), one at a time", flush=True)
    failed, t0 = [], time.time()
    for n, cid in enumerate(todo, 1):
        cmd = [sys.executable, "-u", "-X", "utf8", os.path.abspath(__file__), "--run-one", cid, "--out", a.out]
        cmd += (["--mesh", a.mesh] if a.mesh else []) + (["--dry-run"] if a.dry_run else [])
        case_dir = os.path.join(a.out, cid if not a.mesh else f"{cid}__{a.mesh}")
        log_path = os.path.join(case_dir, "console.log")
        lock = take_lock(case_dir)
        if lock is None:
            print(f"[{n}/{len(todo)}] {cid}: skipped, {os.path.join(case_dir, 'RUNNING.lock')} is held by another process",
                  flush=True)
            failed.append(cid)
            continue
        try:
            if not a.redo and not a.dry_run and state_of(cid, a.out, a.mesh)[0] == "ok":
                print(f"[{n}/{len(todo)}] {cid}: finished meanwhile by another process, skipped", flush=True)
                continue
            try:
                with open(log_path, "w", encoding="utf-8") as lf:
                    rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPLICA).returncode
            except Exception as e:                               # the process could not even be started
                rc = -1
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\ncould not run: {e}\n")
        finally:
            os.remove(lock)
        state, st = state_of(cid, a.out, a.mesh)
        if rc != 0 and state != "failed":                        # killed or crashed without writing its status
            state = "crashed"
            with open(os.path.join(os.path.dirname(log_path), "STATUS.json"), "w", encoding="utf-8") as f:
                json.dump(dict(case=cid, state="crashed", error=f"process ended with code {rc}; see console.log",
                               finished=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)
        if state not in ("ok", "dry"):
            failed.append(cid)
        print(f"[{n}/{len(todo)} {time.time() - t0:6.0f} s] {cid}: {state}"
              + (f" - {(st or {}).get('error', 'see console.log')}" if state not in ("ok", "dry") else ""), flush=True)
        write_status(a.out, a.mesh)
    print(f"finished: {len(todo) - len(failed)} ok, {len(failed)} failed" + (": " + ", ".join(failed) if failed else ""))


if __name__ == "__main__":
    main()
