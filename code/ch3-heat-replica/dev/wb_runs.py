"""The water-block runs (simulation plan, section 7) on COMSOL's stored meshes.

    python -X utf8 dev/wb_runs.py --list                      # every case, its size class and its state
    python -X utf8 dev/wb_runs.py --case W1_v1_coarse         # cases by substring (several allowed)
    python -X utf8 dev/wb_runs.py --block W1                  # whole blocks
    python -X utf8 dev/wb_runs.py --index 7 12                # by number (for a cluster job array)
    python -X utf8 dev/wb_runs.py --all --dry-run             # build every problem, report sizes, solve nothing
    python -X utf8 dev/wb_runs.py --indices --block W1 --size fine     # the numbers of those cases, comma separated
    python -X utf8 dev/wb_runs.py --case W1_v1_coarse_base_a1 --test-steps 3      # loop test: 3 flow steps, then heat

Blocks
  W1  validation against COMSOL's laminar stationary tables: three designs; coarse and normal meshes at the one
      point COMSOL solved there (alpha 1 and 0.5); fine mesh at five points of the 36-point sweep (lowest flow,
      highest flow, hottest, ...) and one alpha 0.5 row.  Viscosity from the case's own temperature field.
  W2  the 36-point sweep per design on the NORMAL mesh (W2f: the same on the fine mesh, several times the work
      per flow field): one flow field per (flow rate, inlet temperature), then the
      four ambient x heat-flux variants on it.  The flow is solved with the viscosity at the inlet temperature;
      the hottest variant is then re-solved with the viscosity of its own temperature field and the change of
      every read-out is recorded (that is the size of the assumption).
  W3  (W3f on the fine mesh) the corrected set-up on the same flow fields (read from W2's stored flows): the same heat
      in watts on the three designs (8, 10, 12, 13, 15.5 W), put on the module footprint (annulus 7-13 mm, by
      radius, in code), module face not cooled by air; v3 as it is (no plate: heat into water and aluminium).
  W4  mesh and scheme: the five points on the coarse and normal meshes; the coarse mesh with every element split
      in eight; streamline diffusion halved and quartered; developed (parabolic) inlet profile; buoyancy.
  W5  is the flow steady?  The steady solution is disturbed by 1 % and time-stepped; reported as steady or
      unsteady with amplitude and period.
  W6  (2026-10-12) the corrected set-up at flow rates below the sweep: 0.025, 0.050, 0.075 L/min; buoyancy at the lowest.
  W7  (2026-10-12) three unsteady flows time-stepped longer (mean over 3 transits, 9 transits) and at half the step.
  W8  (2026-10-12) the corrected set-up at 0.10 and 0.25 L/min, inlet 10 C, on the coarse and the fine mesh.
  W9  (2026-10-13) buoyancy, repaired.  The W6 buoyancy cases solved the flow at the uniform inlet temperature, so
      the force (proportional to T - T_inlet) was zero by construction.  Here the flow is re-solved with the case's
      own temperature field ("coupled"), once with and once without the force; steady flows only (the time
      stepping carries no buoyancy).
  W10 (2026-10-13) unsteady flows with the temperature marched in time together with the flow (dev/wb_heat_td.py),
      against the stationary temperature on the mean flow of the same time window: what the mean-flow shortcut of
      all other unsteady cases leaves out.  One variant: 13 W, ambient equal to the inlet.
  W11 (2026-10-14) the corrected set-up on our own meshes made from the STEP files (dev/wb_cad_mesh.py: four
      levels with thin wall layers, level 2 also with two layers and without; mesh "cad:m2", "cad:m2c2", ...;
      folder $WB_CAD_MESHES, default wb_cad_problem.MESH_DIR), 0.10 and 0.25 L/min, inlet 10 C, 13 W, ambient
      equal to the inlet: does the heated-face temperature settle with the mesh once the walls are resolved?
      Run with the checked linear solves (dev/robust_solve.py).
  W12 (2026-10-15) the heat-exchanger table again where the flow is unsteady, with the temperature marched in time
      with the flow for all ten variants at once (heat 4, 6, 8, 9.5, 20 W on the footprint; ambient 10 and
      35 C), the module plane outside the footprint cooled by the air on all three designs (the air
      exchange is kept; W3 had it on v2 only), on COMSOL's normal mesh and on our level-2 mesh.  Steady cases (most of v1)
      keep the stationary temperature, which is then exact.
  W13 (2026-10-15) the expanded sweep over more parameters and variables, normal mesh, module plane
      cooled by the air, temperature marched where the flow is unsteady.  Case parameters (each needs its own flow):
      design; flow 0.025 to 1.0 L/min (v1 to 1.0, v2 to 0.75, v3 to 0.50: the pressure drop of v3 at 1 L/min would
      be about 0.7 MPa); coolant and inlet temperature: water at 4, 10, 20 C and 50 % propylene glycol at 4 and
      10 C.  Variant parameters (same flow, cheap): 10 W with ambient = inlet (the rise per watt); no heat with
      ambient 25 C above the inlet (the air term); 10 W with the air coefficient x0 and x3; 10 W with the solid's
      conductivity 167 W/m/K (alloy 6061-T6) and 390 W/m/K (copper); 10 W at ambient 22 C (room) and 35 C (warm
      room / near the face).  Any other load and ambient follows by superposition of the first two,
      T - T_in = R P + c (T_amb - T_in); the two ambient variants test that superposition (the air coefficient may
      depend on the temperature).
  W14 (2026-10-15) the W12 cases of v2 and v3 at 0.10 and 0.25 L/min with the streamline diffusion at 0.5 and 0.25 of
      its value: the verification (cylinder, Re 100) found that factor 1 lowers the shedding frequency by 7 % and the
      amplitude by a third, and factor 0.25 brings both within 1 to 2 % of the reference.

Every case runs in its own process.  States in STATUS.json / status.csv:
  ok             flow converged to a steady solution; temperatures solved
  unsteady       the flow keeps oscillating (amplitude and period in STATUS.json); temperatures on the mean flow
  not_converged  no steady solution found and the time stepping did not settle; temperatures on the last field
  failed/crashed error (traceback in STATUS.json / console.log)
Finished cases (ok, unsteady) are skipped unless --redo; not_converged and failed ones are tried again only
with --retry.  Saved per case: settings.json, run.log, flow_history.csv (residual history), monitors.csv (time
stepping), readouts.csv, fields.npz (velocity at vertices and mid-sides, pressure, temperature per variant),
mesh.npz, STATUS.json.
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
sys.path.insert(0, HERE)
sys.path.insert(0, REPLICA)

os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np  # noqa: E402

import wb_common as wb  # noqa: E402

K0 = wb.K0
OUT_DEFAULT = os.path.join(REPLICA, "results", "waterblock_runs")

DEFAULTS = dict(
    design=1, mesh="fine", refine=0,
    Q0=4.2e-6, Tin=293.15, alpha=1.0,            # flow rate m3/s, inlet temperature K, COMSOL's load parameter
    variants=[dict(T0=298.15, Qh=22000.0)],      # each: ambient T0 (K) and heat as flux Qh (W/m2) or power P_W (W)
    heating="comsol",                            # "comsol": the model's heated faces; "annulus": by radius, see below
    annulus_mm=(7.0, 13.0),                      # module footprint (the CAD's module layer is 379 mm2)
    coolant="water",                             # "water" (the model's) or "pg50" (50 % propylene glycol, ASHRAE)
    module_plane="as_model",                     # faces of the module plane outside the footprint: "as_model" (W3:
                                                 # air-cooled where the COMSOL model cools them) or "air" (all cooled)
    inlet="plug",                                # "plug" (COMSOL: uniform) or "parabolic" (developed pipe flow)
    upwind=1.0,                                  # factor on the streamline-diffusion parameter
    stab="sd",                                   # "sd" (streamline term only) or "supg" (consistent; dev/wb_flow.py)
    buoyancy=False,                              # Boussinesq force from water's density (COMSOL: gravity is on)
    viscosity="inlet",                           # "inlet": flow solved at the inlet temperature (+ check on the
                                                 # hottest variant); "coupled": two passes with the case's own field
    march=None,                                  # W5: dict(steps_per_transit, transits, perturb)
    heat_march=None,                             # W10: dict(steps_per_transit, spinup_transits, average_transits,
                                                 # solid_capacity, variant) for dev/wb_heat_td.march_heat
    max_steps=80,                                # flow steps per call; the rise to full flow may use three times this
    cfl0=20.0, reuse_flow=True,
)

POINTS = {  # five points of COMSOL's fine sweep: Q0 (m3/s), Tin, T0 (K), Qh (W/m2)
    "lowcold": (1.7e-6, 277.15, 283.15, 22000.0),      # lowest Reynolds number of the sweep
    "hot": (1.7e-6, 293.15, 308.15, 34000.0),          # hottest face of the sweep (thesis: 23.2 C on v1)
    "mid": (4.2e-6, 283.15, 308.15, 34000.0),
    "highcold": (8.3e-6, 277.15, 283.15, 22000.0),
    "high": (8.3e-6, 293.15, 308.15, 34000.0),         # highest Reynolds number of the sweep
}
P_WATTS = (8.0, 10.0, 12.0, 13.0, 15.5)                # W3: see REPORT.md for where each comes from


def cases():
    """{id: settings that differ from DEFAULTS}.  New cases are appended only: numbers must not change."""
    c = {}
    D = (1, 2, 3)

    def point(name, **kw):
        Q, Tin, T0, Qh = POINTS[name]
        return dict(Q0=Q, Tin=Tin, variants=[dict(T0=T0, Qh=Qh)], **kw)
    for d in D:                                                        # ---- W1 validation
        for m in ("coarse", "normal"):
            for a in (1.0, 0.5):
                c[f"W1_v{d}_{m}_base_a{a:g}"] = dict(design=d, mesh=m, alpha=a, viscosity="coupled")
        for name in POINTS:
            c[f"W1_v{d}_fine_{name}"] = point(name, design=d, viscosity="coupled")
        c[f"W1_v{d}_fine_lowcold_a0.5"] = point("lowcold", design=d, alpha=0.5, viscosity="coupled")
    S = wb.SWEEP
    four = [dict(T0=T0, Qh=Qh) for T0 in S["T0"] for Qh in S["Qh"]]
    for d in D:                                                        # ---- W2 the sweep
        for Q in S["Q0"]:
            for Tin in S["Tin"]:
                c[f"W2_v{d}_Q{Q * wb.LPM:.2f}_Tin{Tin - K0:02.0f}"] = dict(design=d, mesh="normal", Q0=Q, Tin=Tin, variants=four)
    ten = [dict(T0=T0, P_W=P) for T0 in S["T0"] for P in P_WATTS]
    for d in D:                                                        # ---- W3 corrected set-up
        for Q in S["Q0"]:
            for Tin in S["Tin"]:
                c[f"W3_v{d}_Q{Q * wb.LPM:.2f}_Tin{Tin - K0:02.0f}"] = dict(design=d, mesh="normal", Q0=Q, Tin=Tin, variants=ten, heating="annulus")
    for d in D:                                                        # ---- W4 mesh and scheme
        for m in ("coarse", "normal"):
            for name in POINTS:
                c[f"W4_v{d}_{m}_{name}"] = point(name, design=d, mesh=m, viscosity="coupled")
        for name in ("mid", "high"):
            c[f"W4_v{d}_coarse_x8_{name}"] = point(name, design=d, mesh="coarse", refine=1, viscosity="coupled")
            for f in (0.5, 0.25):
                c[f"W4_v{d}_normal_sd{f:g}_{name}"] = point(name, design=d, mesh="normal", upwind=f, viscosity="coupled")
            c[f"W4_v{d}_normal_parabolic_{name}"] = point(name, design=d, mesh="normal", inlet="parabolic", viscosity="coupled")
        for name in ("lowcold", "hot"):
            c[f"W4_v{d}_normal_buoyancy_{name}"] = point(name, design=d, mesh="normal", buoyancy=True, viscosity="coupled")
    for d in D:                                                        # ---- W5 steadiness
        for m, names in (("coarse", ("mid", "high")), ("normal", ("high",))):
            for name in names:
                c[f"W5_v{d}_{m}_{name}_march"] = point(name, design=d, mesh=m, march=dict(
                    steps_per_transit=200 if m == "coarse" else 100, transits=6.0 if m == "coarse" else 4.0, perturb=0.01))
    for d in D:                                                        # ---- W2f, W3f: the same two blocks on the fine mesh
        for Q in S["Q0"]:
            for Tin in S["Tin"]:
                c[f"W2f_v{d}_Q{Q * wb.LPM:.2f}_Tin{Tin - K0:02.0f}"] = dict(design=d, Q0=Q, Tin=Tin, variants=four)
    for d in D:
        for Q in S["Q0"]:
            for Tin in S["Tin"]:
                c[f"W3f_v{d}_Q{Q * wb.LPM:.2f}_Tin{Tin - K0:02.0f}"] = dict(design=d, Q0=Q, Tin=Tin, variants=ten, heating="annulus")
    # ---- added 2026-10-12 (appended: the numbers of the cases above do not change); all in the corrected set-up
    for d in D:                                                        # ---- W6 flow rates below the sweep
        for q in (0.025, 0.050, 0.075):                                # L/min
            for Tin in S["Tin"]:
                c[f"W6_v{d}_Q{q:.3f}_Tin{Tin - K0:02.0f}"] = dict(design=d, mesh="normal", Q0=q / wb.LPM, Tin=Tin, variants=ten, heating="annulus")
        c[f"W6_v{d}_Q0.025_Tin10_buoyancy"] = dict(design=d, mesh="normal", Q0=0.025 / wb.LPM, Tin=283.15, variants=ten, heating="annulus", buoyancy=True)
    for d, Q in ((1, 8.3e-6), (2, 4.2e-6), (3, 4.2e-6)):               # ---- W7 unsteady flows: longer average, shorter time step
        base = dict(design=d, mesh="normal", Q0=Q, Tin=283.15, variants=ten, heating="annulus")
        c[f"W7_v{d}_Q{Q * wb.LPM:.2f}_Tin10_long"] = dict(base, march=dict(steps_per_transit=200, transits=9.0, window_transits=3.0, min_transits=3.0, perturb=0.01))
        c[f"W7_v{d}_Q{Q * wb.LPM:.2f}_Tin10_dt"] = dict(base, march=dict(steps_per_transit=400, transits=4.0, perturb=0.01))
    for d in D:                                                        # ---- W8 the corrected set-up on the coarse and the fine mesh
        for m in ("coarse", "fine"):
            for Q in (1.7e-6, 4.2e-6):
                c[f"W8_v{d}_{m}_Q{Q * wb.LPM:.2f}_Tin10"] = dict(design=d, mesh=m, Q0=Q, Tin=283.15, variants=ten, heating="annulus")
    # ---- added 2026-10-13 (appended; simulation plan, section 9a)
    one = [dict(T0=283.15, P_W=15.5)]                                  # highest load; ambient = inlet at 10 C
    for d, q, Tin in ((1, 0.025, 283.15), (1, 0.100, 283.15), (2, 0.025, 283.15), (3, 0.025, 277.15)):   # ---- W9 buoyancy (steady flows)
        for m in ("coarse", "normal"):
            for tag, b in (("coupled", False), ("buoyancy", True)):
                c[f"W9_v{d}_{m}_Q{q:.3f}_Tin{Tin - K0:02.0f}_{tag}"] = dict(
                    design=d, mesh=m, Q0=q / wb.LPM, Tin=Tin, variants=[dict(T0=Tin, P_W=15.5)], heating="annulus",
                    buoyancy=b, viscosity="coupled", reuse_flow=False)
    hm = dict(steps_per_transit=200, spinup_transits=1.0, average_transits=4.0, solid_capacity=0.02)
    mar = dict(steps_per_transit=200, transits=6.0, perturb=0.01)
    w10 = [dict(T0=283.15, P_W=13.0)]
    for d in D:                                                        # ---- W10 temperature marched with the flow
        for m in ("coarse", "normal"):
            for Q in ((4.2e-6, 8.3e-6) if d == 1 else (1.7e-6, 4.2e-6, 8.3e-6)):
                short = m == "normal"                                  # normal mesh: fewer transits, to stay inside 24 h
                c[f"W10_v{d}_{m}_Q{Q * wb.LPM:.2f}_Tin10"] = dict(
                    design=d, mesh=m, Q0=Q, Tin=283.15, variants=w10, heating="annulus",
                    march=dict(mar, transits=4.0) if short else dict(mar),
                    heat_march=dict(hm, average_transits=3.0) if short else dict(hm))
        c[f"W10_v{d}_coarse_Q0.25_Tin10_cap0.2"] = dict(design=d, mesh="coarse", Q0=4.2e-6, Tin=283.15, variants=w10, heating="annulus",
                                                       march=dict(mar), heat_march=dict(hm, solid_capacity=0.2, spinup_transits=3.0))
        c[f"W10_v{d}_coarse_Q0.25_Tin10_dt"] = dict(design=d, mesh="coarse", Q0=4.2e-6, Tin=283.15, variants=w10, heating="annulus",
                                                   march=dict(mar, steps_per_transit=400), heat_march=dict(hm, steps_per_transit=400))
    for d in D:                                                        # ---- W11 our own meshes (added 2026-10-14)
        for m in ("m1", "m2", "m3", "m4", "m2c0", "m2c2"):
            for Q in (1.7e-6, 4.2e-6):
                c[f"W11_v{d}_cad{m}_Q{Q * wb.LPM:.2f}_Tin10"] = dict(design=d, mesh=f"cad:{m}", Q0=Q, Tin=283.15,
                                                                     variants=[dict(T0=283.15, P_W=13.0)], heating="annulus")
    P_NEW = (4.0, 6.0, 8.0, 9.5, 20.0)                                 # ---- W12 (added 2026-10-15)
    tenb = [dict(T0=T0, P_W=P) for T0 in S["T0"] for P in P_NEW]
    for d, flows in ((1, (0.25, 0.50)), (2, (0.05, 0.10, 0.25, 0.50)), (3, (0.05, 0.10, 0.25, 0.50))):
        for m in ("normal", "cad:m2"):
            for q in flows:
                c[f"W12_v{d}_{m.replace(':', '')}_Q{q:.2f}_Tin10"] = dict(
                    design=d, mesh=m, Q0=q / wb.LPM, Tin=283.15, variants=tenb, heating="annulus", module_plane="air",
                    march=dict(steps_per_transit=200, transits=4.0, perturb=0.01),
                    heat_march=dict(steps_per_transit=200, spinup_transits=1.0, average_transits=3.0, solid_capacity=0.02, all_variants=True))
    for d, qmax in ((1, 1.0), (2, 0.75), (3, 0.50)):                 # ---- W13 the expanded sweep (added 2026-10-15)
        for q in (0.025, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0):
            if q > qmax + 1e-9:
                continue
            for cool, Tin_C in (("water", 4), ("water", 10), ("water", 20), ("pg50", 4), ("pg50", 10)):
                Tin = Tin_C + K0
                var = [dict(T0=Tin, P_W=10.0), dict(T0=Tin + 25.0, P_W=0.0), dict(T0=295.15, P_W=10.0), dict(T0=308.15, P_W=10.0),
                       dict(T0=Tin, P_W=10.0, h_factor=0.0),
                       dict(T0=Tin, P_W=10.0, h_factor=3.0), dict(T0=Tin, P_W=10.0, k_solid=167.0), dict(T0=Tin, P_W=10.0, k_solid=390.0)]
                c[f"W13_v{d}_Q{q:.3f}_{cool}_Tin{Tin_C:02d}"] = dict(
                    design=d, mesh="normal", Q0=q / wb.LPM, Tin=Tin, coolant=cool, variants=var, heating="annulus", module_plane="air",
                    march=dict(steps_per_transit=200, transits=4.0, perturb=0.01),
                    heat_march=dict(steps_per_transit=200, spinup_transits=1.0, average_transits=3.0, solid_capacity=0.02, all_variants=True))
    for d in (2, 3):                                                   # ---- W14 less streamline diffusion (added 2026-10-15)
        for q in (0.10, 0.25):
            for f in (0.5, 0.25):
                c[f"W14_v{d}_normal_Q{q:.2f}_Tin10_sd{f:g}"] = dict(
                    design=d, mesh="normal", Q0=q / wb.LPM, Tin=283.15, variants=tenb, heating="annulus", module_plane="air", upwind=f,
                    march=dict(steps_per_transit=200, transits=5.0, perturb=0.01),
                    heat_march=dict(steps_per_transit=200, spinup_transits=1.0, average_transits=3.0, solid_capacity=0.02, all_variants=True))
    # ---- W15 consistent SUPG (added 2026-10-15): the W1 base case and the W10 marched case of v2, as stored with
    # "sd", on the coarse mesh (PC), the normal mesh and our level 2 (cluster); W10 also for v3
    c1 = dict(c["W1_v2_coarse_base_a1"])
    c["W15_v2_coarse_base_a1_supg"] = dict(c1, stab="supg")
    for d in (2, 3):
        for m in ("coarse", "normal", "cad:m2"):
            base = dict(c[f"W10_v{d}_coarse_Q0.25_Tin10"], mesh=m)
            if m != "coarse":
                base.update(march=dict(base["march"], transits=5.0), heat_march=dict(base["heat_march"], average_transits=3.0))
            c[f"W15_v{d}_{m.replace(':', '')}_Q0.25_Tin10_supg"] = dict(base, stab="supg")
    return c


def settings_of(cid):
    s = dict(DEFAULTS)
    s.update(cases()[cid])
    return s


def size_class(s):
    """coarse / normal / fine by the size of the flow problem (the coarse mesh split in eight counts as fine)."""
    m = str(s["mesh"])
    if m.startswith("cad:"):                                   # by the memory measured in the dry runs (meshes/REPORT.md)
        return {"cad:m1": "coarse", "cad:m2c0": "coarse", "cad:m4": "fine"}.get(m, "normal")
    return "fine" if (m == "fine" or s["refine"]) else m


# Sizes measured on 2026-10-08 with PARDISO's analysis phase (dev/wb_factor_estimate.py; REPORT.md section 4).
CLUSTER_GFLOP_PER_S = 100.0          # ASSUMED speed of one factorisation on a Lakeshore node (16-32 cores); not measured.
                                     # This PC did 14 GFlop/s on 3 threads next to another job (loop test, 2026-10-08)


def estimate(unknowns, design=1):
    """(factor memory GB, memory to request GB, seconds per flow step) for a flow problem of this size."""
    try:
        with open(os.path.join(HERE, "wb_inputs", "factor_estimate.json"), encoding="utf-8") as f:
            fit = json.load(f)["fit"]
        dd = fit["designs"].get(str(design)) or fit["designs"]["1"]
    except Exception:
        fit = dict(p=1.47, flops_p=2.10, pardiso_bytes_per_factor_nonzero=9.5, python_bytes_per_unknown=3800.0,
                   assemble_seconds_per_unknown_one_thread=2.4e-5)
        dd = dict(c=304e6 / 217811 ** 1.47, flops_c=428e9 / 217811 ** 2.10)
    factor_gb = dd["c"] * unknowns ** fit["p"] * fit["pardiso_bytes_per_factor_nonzero"] / 1e9
    total = 1.15 * factor_gb + (fit["python_bytes_per_unknown"] + 2500.0) * unknowns / 1e9 + 2.0
    seconds = dd["flops_c"] * unknowns ** fit["flops_p"] / 1e9 / CLUSTER_GFLOP_PER_S + fit["assemble_seconds_per_unknown_one_thread"] * unknowns
    return factor_gb, total, seconds


# ------------------------------------------------------------ pieces of a run
def correct_heating(prob, s, variant, log):
    """Heat on the module footprint, by radius: returns (list of surface sources, heated triangles, info).

    The faces of the stored meshes do not follow the circle r = outer radius (the triangles there are 1-1.6 mm
    wide on v1 and v2), so every triangle of the heated plane gets the flux times the fraction of its area that
    lies inside the annulus (found from 210 points spread evenly over the triangle).  The total is the stated
    power exactly; the edge of the heated patch is sharp to within one triangle."""
    from replica import fem
    old = [x for x in prob.surface_source if "TEC" in x["name"]]
    if len(old) != 1:
        raise RuntimeError(f"expected one heat-flux feature on the module face, found {[x['name'] for x in prob.surface_source]}")
    z = prob.points[np.unique(old[0]["tris"])][:, 2]
    z_hot = float(np.median(z))
    P = prob.points[prob.tris]
    on_plane = (prob.tri_up == 0) & (np.abs(P[:, :, 2] - z_hot).max(1) < 1e-7)
    ri, ro = (x * 1e-3 for x in s["annulus_mm"])
    n = 20
    lam = np.array([(a, b, n - a - b) for a in range(n + 1) for b in range(n + 1 - a)], float)
    lam = (lam + 1.0 / 3.0) / (n + 1.0)                                 # 231 interior points, evenly spread
    idx = np.where(on_plane)[0]
    xy = np.einsum("sk,tkd->tsd", lam, P[idx][:, :, :2])
    rr = np.hypot(xy[..., 0], xy[..., 1])
    frac = ((rr >= ri) & (rr <= ro)).mean(1)
    keep = frac > 0
    tris, frac = prob.tris[idx[keep]], frac[keep]
    area_t = fem.tri_area(prob.points, tris)
    area = float((area_t * frac).sum())
    q0 = float(variant["P_W"]) / area if "P_W" in variant else float(variant["Qh"]) * s["alpha"]
    sel = np.zeros(len(prob.tris), bool)
    sel[idx[keep][frac >= 0.5]] = True                                  # the triangles counted as "heated face" in the read-outs
    info = dict(heated_area_mm2=round(area * 1e6, 3), exact_annulus_mm2=round(np.pi * (ro ** 2 - ri ** 2) * 1e6, 3),
                flux_W_m2=q0, plane_z_mm=round(z_hot * 1e3, 4), triangles=int(keep.sum()), triangles_cut_by_the_edge=int(((frac > 0) & (frac < 1)).sum()))
    return [dict(name="Module heat (annulus by radius)", tris=tris, q=q0 * frac)] + [x for x in prob.surface_source if x is not old[0]], sel, info


def module_plane_outside(prob, s):
    """Outer boundary triangles of the module plane that lie wholly outside the footprint (radius above the outer
    radius of the annulus at all their corners): the ring and the faces towards the pipes."""
    old = [x for x in prob.surface_source if "TEC" in x["name"] or "annulus" in x["name"]]
    z_hot = float(np.median(prob.points[np.unique(old[0]["tris"])][:, 2]))
    P = prob.points[prob.tris]
    on_plane = (prob.tri_up == 0) & (np.abs(P[:, :, 2] - z_hot).max(1) < 1e-7)
    rr = np.hypot(P[:, :, 0], P[:, :, 1])
    return prob.tris[on_plane & (rr.min(1) >= s["annulus_mm"][1] * 1e-3 - 1e-9)]


# 50 % (by volume) propylene glycol in water: ASHRAE Handbook - Fundamentals 2009 (SI), chapter 31, as tabulated in
# pygld (github.com/jnsebgosselin/pygld, pygld/api/data/proptables_propglycol.npy, fraction 0.5); copied 2026-10-15.
PG50 = dict(
    T_C=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0),
    rho=(1056.85, 1054.84, 1052.71, 1050.44, 1048.04, 1045.52, 1042.87, 1040.09, 1037.18, 1034.15, 1030.98),   # kg/m3
    cp=(3416.0, 3435.0, 3455.0, 3474.0, 3493.0, 3513.0, 3532.0, 3551.0, 3570.0, 3590.0, 3609.0),             # J/(kg K)
    k=(0.319, 0.323, 0.327, 0.331, 0.334, 0.338, 0.341, 0.344, 0.347, 0.350, 0.353),                         # W/(m K)
    mu_mPas=(34.51, 24.81, 18.28, 13.77, 10.59, 8.30, 6.62, 5.36, 4.41, 3.68, 3.10),                         # mPa s
)


def coolant_props(name):
    """(rho, cp, k, mu) as functions of the temperature in K (arrays allowed); None for the model's water."""
    if name == "water":
        return None
    if name != "pg50":
        raise ValueError(name)
    T = np.array(PG50["T_C"]) + K0
    lin = lambda col: (lambda x, y=np.array(PG50[col]): np.interp(x, T, y))
    lmu = np.log(np.array(PG50["mu_mPas"]) * 1e-3)
    return lin("rho"), lin("cp"), lin("k"), (lambda x: np.exp(np.interp(x, T, lmu)))


def apply_coolant(pv, s, spec=None):
    """Replace the water of problem pv (and of the flow settings spec) by the coolant of the case.  The volume flow
    stays the one asked for: the inlet's mass flow is scaled by the ratio of the densities at the inlet temperature."""
    c = coolant_props(s["coolant"])
    if c is None:
        return
    rho, cp, k, mu = c
    for d in wb.fluid_domains(pv):
        pv.k[d] = k
        pv.rho_cp[d] = (lambda x: rho(x) * cp(x))
        pv.variable.add(d)
    if spec is not None:
        Tin = float(s["Tin"])
        rho_w = float(spec["rho_ref"])
        spec["mu"], spec["rho_ref"] = mu, float(rho(Tin))
        spec["beta"] = -(float(rho(Tin + 0.5)) - float(rho(Tin - 0.5))) / float(rho(Tin)) if spec.get("beta") else 0.0
        spec["inlets"] = [(e, ("mass_flow", v[1] * spec["rho_ref"] / rho_w) if isinstance(v, tuple) else v) for e, v in spec["inlets"]]


def apply_variant_physics(pv, var):
    """Heat-only parameters of a variant: factor on the air coefficient, conductivity of the solid."""
    if "h_factor" in var:
        for c in pv.convective:
            c["h"] = (lambda x, f=c["h"], a=float(var["h_factor"]): a * f(x))
    if "k_solid" in var:
        for d in pv.k:
            if d not in wb.fluid_domains(pv):
                pv.k[d] = (lambda x, v=float(var["k_solid"]): np.full(np.shape(x), v) if np.ndim(x) else v)


def carried_heat(prob, r, T, t_in):
    """Heat leaving with the coolant [W]: integral of rho*cp*(T - T_inlet)*(u.n) over inlet and outlet faces."""
    a, b, w1, w2 = 0.445948490915965, 0.091576213509771, 0.223381589678011, 0.109951743655322
    pts = np.array([(a, a, 1 - 2 * a), (a, 1 - 2 * a, a), (1 - 2 * a, a, a), (b, b, 1 - 2 * b), (b, 1 - 2 * b, b), (1 - 2 * b, b, b)])
    wts = np.array([w1] * 3 + [w2] * 3)
    rc = prob.rho_cp[r.spec["domains"][0]]
    total = 0.0
    for f in (r.inlet_faces, r.outlet_faces):
        if f is None:
            continue
        Tn = T[r.nodes[f["v"]]] - t_in                                   # (nf, 3)
        uv = np.einsum("fkd,fd->fk", r.u[f["v"]], f["n"])                 # u.n * area at vertices
        um = np.einsum("fkd,fd->fk", r.u[f["m"]], f["n"])                 # at mid-sides (0-1, 1-2, 0-2)
        c = rc(T[r.nodes[f["v"]]].mean(1)) if callable(rc) else np.full(len(Tn), rc)
        for l, w in zip(pts, wts):
            un = sum(l[i] * (2 * l[i] - 1) * uv[:, i] for i in range(3)) + 4 * (l[0] * l[1] * um[:, 0] + l[1] * l[2] * um[:, 1] + l[0] * l[2] * um[:, 2])
            total += float((w * c * un * (Tn @ l)).sum())
    return total


def readouts(prob, h, r, T, faces, variant_T_in, s):
    from replica import fem
    out = {}
    for name, face in faces.items():
        out[f"{name} average (K)"] = wb.face_average(T, face)
    for name in ("TEC", "HotSide", "HotSide on aluminium", "HotSide on water", "Solid outer surface", "Heated face"):
        if name in faces:
            out[f"{name} maximum (K)"] = wb.face_max(T, faces[name])
    solid = ~np.isin(prob.dom, wb.fluid_domains(prob))
    out["Aluminium maximum (K)"] = float(T[np.unique(prob.tets[solid])].max())
    out["Water maximum (K)"] = float(T[np.unique(prob.tets[~solid])].max())
    out["Heat in (W)"] = float(sum((np.asarray(x["q"]) * fem.tri_area(prob.points, x["tris"])).sum() for x in prob.surface_source))
    out["Heat carried by the water (W)"] = carried_heat(prob, r, T, variant_T_in)
    loss = 0.0
    for c in h.conv_bc:
        Tf = T[c["tris"]].mean(1)
        loss += float((c["h"](Tf) * c["area"] * (Tf - c["t_ext"])).sum())
    out["Heat to the air (W)"] = loss
    out["Heat balance error (W)"] = out["Heat in (W)"] - out["Heat carried by the water (W)"] - loss
    dp = r.pressure_drop()
    Q = r.inflow * r.inflow_scale
    out["Pressure drop inlet to outlet (Pa)"] = dp
    out["Pumping power (mW)"] = dp * Q * 1e3
    out["Volume flow (L/min)"] = Q * wb.LPM
    out["Outflow / inflow"] = r.face_flow(r.outlet_faces) / Q
    out["Largest speed (m/s)"] = float(np.linalg.norm(r.u, axis=1).max())
    return out


# ------------------------------------------------------------------ one case
def run_one(cid, out_root, dry=False, test_steps=0, max_hours=None):
    from replica import fem, solve
    import wb_flow

    s = settings_of(cid)
    # (2026-10-22) opt-in corrections for NEW runs only.  Without the environment variables every
    # case is set up exactly as before (stored results stay reproducible, flow-cache keys unchanged); with them the
    # choice is written into settings.json and into the flow-cache key, so no flow is shared across the two set-ups.
    #   WB_RHO_REF=inlet           fluid reference density at the inlet temperature (model: at the ambient T0)
    #   WB_BUOYANCY_DENSITY=local  buoyancy force [rho(T) - rho(T_ref)] g per point instead of Boussinesq at T_ref
    #   WB_SKIP_STEADY_MARCH=1     (2026-10-20) skip the stability march of a converged steady flow; now recorded here and
    #                              honoured only up to an inlet-pipe Reynolds number of 100
    if os.environ.get("WB_RHO_REF") == "inlet":
        s["rho_ref"] = "inlet"
    if os.environ.get("WB_BUOYANCY_DENSITY") == "local":
        s["buoyancy_density"] = "local"
    if os.environ.get("WB_SKIP_STEADY_MARCH") == "1":
        s["skip_steady_march"] = "requested (honoured up to inlet-pipe Reynolds number 100)"
    out_dir = os.path.join(out_root, cid)
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    t_start = time.time()
    deadline = t_start + 3600.0 * max_hours if max_hours else None

    def log(m):
        lines.append(m)
        print(m, flush=True)

    def flush_log():
        with open(os.path.join(out_dir, "run.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    stamp = wb.code_stamp()
    v = s["design"]
    mph = wb.open_model(v)
    log(f"case {cid}: design v{v}, {os.path.basename(wb.model_file(v))}, mesh {s['mesh']}" + (f" split in eight x{s['refine']}" if s["refine"] else "")
        + f", flow {s['Q0'] * s['alpha'] * wb.LPM:.3f} L/min, inlet {s['Tin'] - K0:.1f} C, alpha {s['alpha']:g}, {len(s['variants'])} variant(s)")
    log(f"code: solver {stamp['replica']}, water-block files {stamp['waterblock']}; threads {os.environ.get('MKL_NUM_THREADS')}")
    if str(s["mesh"]).startswith("cad:"):                                 # a mesh made from the STEP files
        import re
        import wb_cad_problem
        lv, suffix = re.fullmatch(r"cad:m(\d)(.*)", s["mesh"]).groups()
        mesh, crep = wb_cad_problem.load_cad_mesh(v, int(lv), folder=os.environ.get("WB_CAD_MESHES", wb_cad_problem.MESH_DIR), suffix=suffix)
        log(f"  mesh from the STEP file: {crep['file']}, {crep['nodes']} nodes, {crep['elements']} elements; "
            f"boundary numbers: largest area difference {crep['boundary_numbers']['largest_area_difference_mm2']} mm2")
    else:
        mesh = wb.load_mesh(mph, s["mesh"], s["refine"])
    if mesh["dropped"]["nodes"]:
        log(f"  mesh: faces that bound no volume were dropped: {mesh['dropped']}")

    def op_of(var):
        return dict(Q0=s["Q0"], Tin=s["Tin"], T0=var["T0"], Qh=var.get("Qh", 0.0))

    def t_in_of(var):                                                     # COMSOL: T0 - (T0 - Tin) * alpha
        return var["T0"] - (var["T0"] - s["Tin"]) * s["alpha"]

    prob = wb.build_problem(mph, mesh, op_of(s["variants"][0]), s["alpha"])
    for n in prob.notes:
        log("  note: " + n)
    water = wb.fluid_domains(prob)
    vol, g = fem.tet_geometry(prob.points, prob.tets)
    spec = dict(prob.flow["regions"][0])
    if s["coolant"] != "water":
        apply_coolant(prob, s, spec)
        log(f"  coolant {s['coolant']}: density {spec['rho_ref']:.1f} kg/m3, viscosity {float(spec['mu'](s['Tin'])) * 1e3:.3f} mPa s at the inlet; "
            f"inlet mass flow {spec['inlets'][0][1]}")
    gvec = prob.flow["gravity"]
    def fluid_rho():
        """Density of the coolant (kg/m3) as a function of T (K, arrays allowed): the model's water material, tabulated
        every 0.25 K over 263-363 K, or the coolant's table."""
        cp_ = coolant_props(s["coolant"])
        if cp_ is not None:
            return cp_[0]
        from replica.expr import Scope
        mats = [m for m in mph.materials(next(iter(mph.components()))) if m["dim"] == 3 and water[0] in (m["entities"] or [])]
        ms = Scope(mph.parameters()).child(mats[0]["funcs"])
        Tg = np.arange(263.15, 363.15 + 1e-9, 0.25)
        rg = np.array([float(ms.eval(mats[0]["props"]["density"], T=float(x), pA=101325.0)) for x in Tg])
        return lambda T: np.interp(T, Tg, rg)

    if s["buoyancy"]:
        from replica.expr import Scope  # noqa: F401  (density function comes with the material; evaluated below)
        mats = [m for m in mph.materials(next(iter(mph.components()))) if m["dim"] == 3 and water[0] in (m["entities"] or [])]
        ms = Scope(mph.parameters()).child(mats[0]["funcs"])
        rho = lambda T: float(ms.eval(mats[0]["props"]["density"], T=T, pA=101325.0))
        if s["coolant"] != "water":                                       # (2026-10-22) the coolant's own density, not water's
            rho = lambda T, f=coolant_props(s["coolant"])[0]: float(f(T))   # (no stored case has a coolant with buoyancy)
        Tr = float(s["Tin"])
        spec["beta"], spec["t_ref"] = -(rho(Tr + 0.5) - rho(Tr - 0.5)) / rho(Tr), Tr
        log(f"  buoyancy on: expansion coefficient {spec['beta']:.3e} 1/K at {Tr - K0:.1f} C, gravity {gvec}")
        if s.get("buoyancy_density") == "local":
            spec["rho_fn"] = fluid_rho()
            log("  buoyancy force from the local density difference [rho(T) - rho(T_ref)] g (WB_BUOYANCY_DENSITY=local)")
    if s.get("rho_ref") == "inlet" and s["coolant"] == "water":           # reference density at the inlet (the coolant path already uses rho(T_in))
        rho_in = float(fluid_rho()(float(s["Tin"])))
        old = float(spec["rho_ref"])
        spec["inlets"] = [(e, ("mass_flow", val[1] * rho_in / old) if isinstance(val, tuple) else val) for e, val in spec["inlets"]]
        spec["rho_ref"] = rho_in
        log(f"  reference density {old:.2f} -> {rho_in:.2f} kg/m3 (inlet temperature; WB_RHO_REF=inlet); inlet mass flow scaled "
            f"by the same ratio, so the volume flow seen by the solver is unchanged")
    t0 = time.time()
    r = wb_flow.WBFlow(prob, spec, vol, g)
    r.upwind = float(s["upwind"])
    r.stab = s["stab"]
    r.set_inlet_profile(s["inlet"])
    r.tasm = fem.Assembler(prob.tets[r.el], prob.points.shape[0])
    r.set_temperature(np.full(prob.points.shape[0], t_in_of(s["variants"][0])))      # viscosity at the inlet temperature
    unknowns = int(sum(len(x) for x in r.free_c) + r.n1)
    fgb, mem, sec = estimate(unknowns, v)
    nu = float(np.mean(r.mu_e) / r.rho)
    f_in = r.inlet_faces
    d_in = 2.0 * np.sqrt(f_in["area"].sum() / np.pi)
    size = dict(nodes=int(prob.points.shape[0]), elements=int(len(prob.tets)), water_elements=int(len(r.el)),
                velocity_nodes=int(r.n2), pressure_unknowns=int(r.n1), flow_unknowns=unknowns,
                velocity_block_nonzeros=int(r.asm22.nnz), newton_matrix_nonzeros_estimate=int(9 * r.asm22.nnz + 6 * r.B[0].nnz),
                estimated_factor_GB=round(fgb, 1), estimated_memory_GB=round(mem, 1), estimated_seconds_per_flow_step=round(sec, 1),
                inlet_speed_m_s=round(r.inlet_mean_speed, 4), inlet_diameter_mm=round(d_in * 1e3, 3),
                reynolds_inlet_pipe=round(r.inlet_mean_speed * d_in / nu),
                transit_time_s=round(float(r.vol.sum() / r.inflow), 4))
    log(f"  problem built in {time.time() - t0:.1f} s: {size['nodes']} nodes, {size['elements']} elements ({size['water_elements']} water); "
        f"flow {unknowns} unknowns ({r.n2} velocity nodes, {r.n1} pressures), Newton matrix about {size['newton_matrix_nonzeros_estimate'] / 1e6:.1f} M non-zeros")
    log(f"  estimate: factor {fgb:.1f} GB, memory to request {mem:.0f} GB, {sec:.0f} s per flow step; inlet speed {r.inlet_mean_speed:.3f} m/s, "
        f"inlet-pipe Reynolds number {size['reynolds_inlet_pipe']}, channel transit {size['transit_time_s']} s")
    step_tag = next(k for k, x in mph.study_steps().items() if x["op"] == "Stationary")

    # ---- heat problems of the variants (assembled now, so a dry run exercises everything but the solves)
    heats = []
    for i, var in enumerate(s["variants"]):
        pv = prob if i == 0 else wb.build_problem(mph, mesh, op_of(var), s["alpha"])
        faces = wb.face_sets(mph, pv)
        hinfo = {}
        if s["heating"] == "annulus":
            src, heated, hinfo = correct_heating(pv, s, var, log)
            pv.surface_source = src
            ht = pv.tris[heated]
            faces["Heated face"] = (ht, fem.tri_area(pv.points, ht))
            key = {tuple(sorted(t)) for t in ht.tolist()}
            for c in pv.convective:                                       # the module face is not cooled by the air
                keep = np.array([tuple(sorted(t)) not in key for t in c["tris"].tolist()], bool)
                if not keep.all():
                    hinfo["removed_from_air_cooling_mm2"] = round(float(fem.tri_area(pv.points, c["tris"][~keep]).sum()) * 1e6, 3)
                    c["tris"] = c["tris"][keep]
            if s["module_plane"] == "air" and pv.convective:
                ring = module_plane_outside(pv, s)
                cooled = {tuple(sorted(t)) for c in pv.convective for t in c["tris"].tolist()}
                add = np.array([t for t in ring.tolist() if tuple(sorted(t)) not in cooled and tuple(sorted(t)) not in key], np.int64).reshape(-1, 3)
                if len(add):
                    c = max(pv.convective, key=lambda c: float(fem.tri_area(pv.points, c["tris"]).sum()))      # the block's air boundary
                    c["tris"] = np.vstack([c["tris"], add.astype(c["tris"].dtype)])
                hinfo["added_to_air_cooling_mm2"] = round(float(fem.tri_area(pv.points, add).sum()) * 1e6, 3) if len(add) else 0.0
        elif "P_W" in var:
            raise RuntimeError("a heat load in watts needs heating='annulus'")
        apply_coolant(pv, s)                                             # before Heat: it copies the air boundaries
        apply_variant_physics(pv, var)
        h = solve.Heat(pv, log=lambda m: None, flow=False, supg_heat=True)
        h.regions = [r]
        q_in = float(sum((np.asarray(x["q"]) * fem.tri_area(pv.points, x["tris"])).sum() for x in pv.surface_source))
        label = f"T0{var['T0'] - K0:02.0f}_" + (f"P{var['P_W']:g}W" if "P_W" in var else f"q{var['Qh'] / 1000:g}k")
        label += (f"_h{var['h_factor']:g}" if "h_factor" in var else "") + (f"_k{var['k_solid']:g}" if "k_solid" in var else "")
        heats.append(dict(label=label, var=var, prob=pv, heat=h, faces=faces, info=hinfo, q_in=q_in, t_in=t_in_of(var)))
        log(f"  variant {label}: heat in {q_in:.3f} W, inlet {t_in_of(var) - K0:.2f} C, ambient {var['T0'] - K0:.1f} C"
            + (f"; heated area {hinfo['heated_area_mm2']} mm2 (exact annulus {hinfo['exact_annulus_mm2']}), flux {hinfo['flux_W_m2']:.0f} W/m2"
               + (f", {hinfo['removed_from_air_cooling_mm2']} mm2 taken out of the air cooling" if "removed_from_air_cooling_mm2" in hinfo else "")
               if hinfo else ""))
    areas = {k: round(float(a.sum()) * 1e6, 3) for k, (_, a) in heats[0]["faces"].items()}
    log(f"  read-out faces (mm2): {areas}")
    wb.write_json(os.path.join(out_dir, "settings.json"),
                  dict(case=cid, settings=s, model_file=os.path.basename(wb.model_file(v)), code=stamp, size=size,
                       dropped=mesh["dropped"], faces_mm2=areas, variants=[dict(label=x["label"], heat_in_W=x["q_in"], **x["info"]) for x in heats]))
    if dry:
        log("  dry run: problem built, nothing solved")
        flush_log()
        return dict(state="dry", **size)

    # ---- flow
    T_visc = np.full(prob.points.shape[0], heats[0]["t_in"])
    r.set_temperature(T_visc, gvec if s["buoyancy"] else None)
    sig = (f"v{v}_{str(s['mesh']).replace(':', '-')}_r{s['refine']}_Q{s['Q0'] * s['alpha'] * 1e9:.0f}_T{heats[0]['t_in']:.2f}_{s['inlet']}_sd{s['upwind']:g}"
           + ("_buoy" if s["buoyancy"] else "") + ("_supg" if s["stab"] != "sd" else "")
           # (2026-10-19) the coolant belongs in the key: without it the glycol cases of W13 at 0.025 and 0.05 L/min took
           # the water flow of W6 (results/waterblock_research_2026-10-19/REPORT.md); water keys are unchanged
           + (f"_{s['coolant']}" if s.get("coolant", "water") != "water" else "")
           # (2026-10-22) the opt-in set-ups (WB_RHO_REF, WB_BUOYANCY_DENSITY) get their own flows; default keys unchanged
           + ("_rhoin" if s.get("rho_ref") == "inlet" else "") + ("_bden" if s["buoyancy"] and s.get("buoyancy_density") == "local" else ""))
    # (2026-10-22) fingerprint of the flow field, stored next to the cache file as <key>.json and
    # compared on reuse: a mismatch of the physics, the mesh size or the imposed inflow refuses the reuse; a different
    # solver or code version is logged (not refused, so existing caches stay usable).  Cache files written before
    # 2026-10-22 have no fingerprint: they are reused as before, with the inflow check below logged.
    fingerprint = dict(key=sig, design=v, mesh=str(s["mesh"]), refine=s["refine"], flow_m3_s=float(s["Q0"] * s["alpha"]),
                       t_in_K=round(float(heats[0]["t_in"]), 6), inlet=s["inlet"], upwind=float(s["upwind"]), stab=s["stab"],
                       buoyancy=bool(s["buoyancy"]), coolant=s.get("coolant", "water"), rho_ref=s.get("rho_ref", "model"),
                       buoyancy_density=s.get("buoyancy_density", "boussinesq_at_inlet"), flow_unknowns=unknowns,
                       imposed_inflow_m3_s=float(r.inflow), model_file=os.path.basename(wb.model_file(v)),
                       solver=str(stamp.get("replica")), linear_solves="dev/robust_solve.py", code=str(stamp.get("waterblock")))
    fp_physics = [k for k in fingerprint if k not in ("solver", "linear_solves", "code", "imposed_inflow_m3_s")]
    flow_dir = os.path.join(out_root, "_flows")
    os.makedirs(flow_dir, exist_ok=True)
    flow_file, lock = os.path.join(flow_dir, sig + ".npz"), os.path.join(flow_dir, sig + ".lock")
    hist, flow_res, monitors = [], None, None
    hist_head = ["stage", "step", "pseudo_time_step_s", "residual", "relative_residual", "velocity_change", "max_speed_m_s", "pressure_drop_Pa",
                 "step_length_0_if_rejected", "factor_seconds"]

    def write_hist():
        with open(os.path.join(out_dir, "flow_history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(hist_head)
            w.writerows(hist)

    def steady(stage, **kw):
        """One call of WBFlow.steady from the stored field (viscosity passes, loop test); rows go to the history."""
        rows = []
        opts = dict(max_steps=test_steps or s["max_steps"], cfl0=s["cfl0"], log=log, deadline=deadline, history=rows)
        opts.update(kw)
        res = r.steady(**opts)
        hist.extend([[stage] + x for x in rows])
        write_hist()
        return res

    def to_full_flow(start):
        """Steady solution at full flow through reduced flow rates; the field is kept after every stage."""
        rows = []
        n_before = len(hist)

        def cb(scale, u, p):
            np.savez(os.path.join(out_dir, "checkpoint.npz"), u=u, p=p, scale=scale)
            del hist[n_before:]
            hist.extend([[f"flow x {x[0]:.4g}"] + x[1:] for x in rows])
            write_hist()
            flush_log()
        res = r.continuation(start=start, max_steps=test_steps or s["max_steps"] * 3, log=log, deadline=deadline, history=rows, callback=cb,
                             cfl0=s["cfl0"])
        del hist[n_before:]
        hist.extend([[f"flow x {x[0]:.4g}"] + x[1:] for x in rows])
        write_hist()
        return res

    def lock_alive():
        """Is the case that holds the lock still working?  (It may have been killed by the time or memory limit.)"""
        try:
            with open(lock, encoding="utf-8") as f:
                owner = f.read().strip()
            od = os.path.join(out_root, owner)
            if os.path.exists(os.path.join(od, "STATUS.json")) and os.path.getmtime(os.path.join(od, "STATUS.json")) > os.path.getmtime(lock):
                return False                                              # it finished without leaving the flow field
            newest = max([os.path.getmtime(lock)] + [os.path.getmtime(os.path.join(od, n)) for n in ("run.log", "flow_history.csv", "console.log")
                                                      if os.path.exists(os.path.join(od, n))])
            return time.time() - newest < 3 * 3600                        # no sign of life for three hours: stale
        except OSError:
            return False

    waited = 0
    while s["reuse_flow"] and not test_steps and not os.path.exists(flow_file) and os.path.exists(lock) and lock_alive():
        if waited == 0:
            log(f"  another process is solving this flow field ({sig}); waiting for it")
        time.sleep(60)
        waited += 60
    if s["reuse_flow"] and not test_steps and os.path.exists(flow_file):
        fpf = flow_file[:-4] + ".json"
        if os.path.exists(fpf):
            old_fp = json.load(open(fpf, encoding="utf-8"))
            bad = [k for k in fp_physics if k in old_fp and old_fp[k] != fingerprint[k]]
            if "imposed_inflow_m3_s" in old_fp and abs(old_fp["imposed_inflow_m3_s"] / fingerprint["imposed_inflow_m3_s"] - 1) > 1e-9:
                bad.append("imposed_inflow_m3_s")
            if bad:
                raise RuntimeError(f"flow cache {sig}.npz does not match this case ({', '.join(f'{k}: {old_fp[k]} != {fingerprint[k]}' for k in bad)}); "
                                   f"not reused.  Move the cache file aside by hand or run with reuse_flow off")
            for k in ("solver", "code"):
                if old_fp.get(k) != fingerprint[k]:
                    log(f"  note: flow cache written with {k} {old_fp.get(k)}, this run {fingerprint[k]}")
        else:
            log(f"  note: flow cache {sig}.npz has no fingerprint (written before 2026-10-22)")
        z = np.load(flow_file, allow_pickle=True)
        if z["u"].shape == r.u.shape:
            r.u, r.p = z["u"].astype(float), z["p"].astype(float)
            r.u[r.fixed_u] = r.value_u[r.fixed_u]
            flow_res = json.loads(str(z["result"]))
            flow_res["from"] = os.path.basename(flow_file)
            log(f"  flow field read from _flows/{sig}.npz ({flow_res['status']}, from case {flow_res.get('case')})")
            ratio = r.face_flow(r.outlet_faces) / r.inflow                  # (2026-10-22) does the stored field carry this inflow?
            flow_res["reused_outflow_over_imposed_inflow"] = float(ratio)
            if abs(ratio - 1.0) > 1e-6:
                log(f"  WARNING: the reused flow carries {ratio:.6f} of this case's imposed inflow (cache key does not hold the inflow)")
    if flow_res is None:
        with open(lock, "w", encoding="utf-8") as f:                  # this case now owns the flow field
            f.write(cid)
        ck = os.path.join(out_dir, "checkpoint.npz")
        start = None
        if os.path.exists(ck) and not test_steps:
            z = np.load(ck, allow_pickle=True)
            if z["u"].shape == r.u.shape and "scale" in z.files:
                start = (float(z["scale"]), z["u"], z["p"])
                log(f"  resuming from the checkpoint: steady solution at {start[0]:.4g} of the full flow")
        if start is not None and start[0] >= 1.0:
            r.u, r.p = start[1], start[2]
            flow_res = steady("full flow", restart=True, cfl0=np.inf)
            flow_res["reached"] = 1.0
        else:
            log("  flow: steady solutions at rising flow rate, up to the full flow" + (f" (loop test: {test_steps} steps only)" if test_steps else ""))
            flow_res = to_full_flow(start)
            if flow_res["status"] != "steady":
                log(f"  no steady solution at full flow: the steady branch was followed to {flow_res.get('reached', 0.0):.4g} of the full flow "
                    f"({flow_res['status']}, {flow_res['steps']} steps)")
        flow_res["case"] = cid
        if flow_res["status"] == "steady" and s["viscosity"] == "inlet" and not s["march"]:
            np.savez(flow_file, u=r.u, p=r.p, result=json.dumps(flow_res, default=float))
            wb.write_json(flow_file[:-4] + ".json", dict(fingerprint, case=cid))      # 2026-10-22: fingerprint
        try:
            os.remove(lock)
        except OSError:
            pass
    scale_now = r.inflow_scale                                            # 1 unless the flow iteration stopped short of the full flow
    log(f"  flow: {flow_res['status']}" + (f" in {flow_res['steps']} steps (+{flow_res.get('rejected', 0)} rejected), relative residual "
                                             f"{flow_res['relative_residual']:.2e}" if "steps" in flow_res else "")
        + f"; pressure drop {r.pressure_drop():.2f} Pa, largest speed {np.linalg.norm(r.u, axis=1).max():.4f} m/s, "
        f"outflow/inflow {r.face_flow(r.outlet_faces) / (r.inflow * scale_now):.9f}")

    state = "ok"
    march_res = None
    if test_steps:
        state = "test"
    elif flow_res["status"] == "time_limit":
        state = "not_converged"                                          # the checkpoint stays; --retry resumes from it
        log(f"  the flow iteration ran out of time; temperatures below are for {scale_now:.4g} of the requested flow or an unconverged field")
    elif flow_res["status"] == "steady" and s["march"] and os.environ.get("WB_SKIP_STEADY_MARCH") == "1" \
            and size["reynolds_inlet_pipe"] <= 100:                       # (2026-10-22) guard: low Reynolds number only
        # (2026-10-20) opt-in, for reruns on the PC: the march only tests whether a converged stationary flow stays steady,
        # and the temperatures use that stationary solution when it does.  W13_v1_Q0.025_pg50_Tin04 spent 13 of its 15.4 h
        # on 600 steps whose pressure drop never changed (1e-13 per transit).  Recorded in STATUS.json.
        march_res = dict(status="skipped", note="WB_SKIP_STEADY_MARCH=1: stationary flow converged, march not run", field="steady solution")
        log("  time stepping skipped (WB_SKIP_STEADY_MARCH=1): temperatures use the converged stationary flow")
    elif flow_res["status"] != "steady" or s["march"]:
        m = dict(steps_per_transit=200, transits=6.0, perturb=0.0)
        m.update(s["march"] or {})
        if flow_res["status"] != "steady":
            m.update(perturb=0.0, ramp_transits=1.0)                     # no steady field to start from: ramp up from rest
        scale_now = 1.0
        log("  time stepping at a fixed step to see whether the flow settles")
        u_steady, p_steady = (r.u.copy(), r.p.copy()) if flow_res["status"] == "steady" else (None, None)
        march_res, monitors, names = r.march(log=log, deadline=deadline, **m)
        if u_steady is not None and march_res["status"] != "unsteady":   # the disturbance died out (or the run was too short to tell):
            r.u, r.p = u_steady, p_steady                                # temperatures on the steady solution itself
            march_res["field"] = "steady solution"
        with open(os.path.join(out_dir, "monitors.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(names)
            w.writerows(monitors.tolist())
        log(f"  time stepping: {march_res['status']} after {march_res['steps']} steps ({march_res['transits']:.2f} transits); "
            f"pressure-drop amplitude {march_res.get('amplitude_Pa', 0.0):.3g} Pa ({march_res.get('amplitude_relative', 0.0):.2e} of the mean), "
            f"period {march_res.get('period_s')} s; temperatures use the {march_res['field']}")
        if march_res["status"] == "unsteady":
            state = "unsteady"
        elif flow_res["status"] != "steady":
            state = "not_converged"
            if march_res["status"] == "steady":                          # settled in time: finish with Newton on the stationary equations
                res3 = steady("after the time stepping", restart=True, cfl0=np.inf, max_steps=test_steps or 20)
                log(f"  Newton iteration from the settled field: {res3['status']} in {res3['steps']} steps")
                if res3["status"] == "steady":
                    flow_res, state = dict(res3, case=cid, reached=1.0, found_by="time stepping, then Newton"), "ok"
    if test_steps and s["march"]:
        march_res, monitors, names = r.march(log=log, steps_per_transit=200, transits=test_steps / 200.0, perturb=0.01, every=1)
    flush_log()

    # ---- temperatures
    rows, fields = [], {}
    worst_change = {}
    for x in heats:
        t0 = time.time()
        T = x["heat"].stationary(step=step_tag, flow_passes=0, tol=1e-6)
        ro = readouts(x["prob"], x["heat"], r, T, x["faces"], x["t_in"], s)
        x["T"], x["ro"] = T, ro
        log(f"  {x['label']}: " + ", ".join(f"{k.replace(' average (K)', '')} {val - K0:.3f}" for k, val in ro.items() if k.endswith("average (K)")
                                           and k.split(" average")[0] in ("TEC", "INw", "OUTw", "Waterblock", "HotSide"))
            + f" C; hottest aluminium {ro['Aluminium maximum (K)'] - K0:.3f} C; heat in {ro['Heat in (W)']:.3f} W, with the water "
            f"{ro['Heat carried by the water (W)']:.3f} W, to the air {ro['Heat to the air (W)']:.3f} W; {time.time() - t0:.1f} s")
    # viscosity from the temperature field: every variant when "coupled", else the hottest one as a check
    which = heats if s["viscosity"] == "coupled" else [max(heats, key=lambda x: x["ro"]["Water maximum (K)"])]
    if state in ("ok", "test") and not s["march"]:
        for x in which:
            before = dict(x["ro"])
            u0, p0 = r.u.copy(), r.p.copy()
            res2 = None
            for _ in range(1 if test_steps else 2):
                r.set_temperature(x["T"], gvec if s["buoyancy"] else None)
                res2 = steady(f"viscosity of {x['label']}", restart=True, cfl0=np.inf, scale=scale_now)
                x["T"] = x["heat"].stationary(step=step_tag, flow_passes=0, tol=1e-6)
                if res2["status"] != "steady":
                    break
            after = readouts(x["prob"], x["heat"], r, x["T"], x["faces"], x["t_in"], s)
            ch = {k: after[k] - before[k] for k in after if k.endswith("(K)") or k.startswith("Pressure")}
            big = max(abs(val) for k, val in ch.items() if k.endswith("(K)"))
            log(f"  {x['label']}: flow re-solved with the viscosity of its own temperature field ({res2['status']}, {res2['steps']} steps): "
                f"largest change of a read-out temperature {big:.4f} C, pressure drop {before['Pressure drop inlet to outlet (Pa)']:.2f} -> "
                f"{after['Pressure drop inlet to outlet (Pa)']:.2f} Pa")
            worst_change[x["label"]] = dict(largest_temperature_change_K=big, changes=ch, status=res2["status"])
            if s["viscosity"] == "coupled":
                x["ro"] = after
                if res2["status"] != "steady" and not test_steps:
                    state = "not_converged"
                fields[f"u_{x['label']}"], fields[f"p_{x['label']}"] = r.u.astype(np.float32), r.p.astype(np.float32)
            else:
                x["ro_coupled_check"] = after
                r.u, r.p = u0, p0
                r.set_temperature(T_visc, gvec if s["buoyancy"] else None)
    heat_march_out = None
    # also where the march ended "not settled" with a fluctuation above 0.1 % (the verification of 2026-10-15 found
    # shedding whose amplitude was still growing after 5 transits called "not settled")
    fluct = bool(march_res) and march_res.get("status") == "not_settled" and march_res.get("amplitude_relative", 0.0) > 1e-3
    if s["heat_march"] and (state == "unsteady" or fluct or test_steps) and getattr(r, "last_state", None) is not None:
        import wb_heat_td
        hm = dict(s["heat_march"])
        allv = hm.pop("all_variants", False)
        x = heats[hm.pop("variant", -1)] if not allv else heats[0]
        if test_steps:
            hm.update(steps_per_transit=200, spinup_transits=0.0, average_transits=test_steps / 200.0)
        log(f"  temperature marched with the flow ({'all ' + str(len(heats)) + ' variants' if allv else x['label']}): {hm}")
        if allv:
            hres = wb_heat_td.march_heat(r, [y["heat"] for y in heats], [y["T"] for y in heats], step_tag, x["faces"], log=log,
                                         deadline=deadline, every=1 if test_steps else 20, **hm)
        else:
            hres = wb_heat_td.march_heat(r, x["heat"], x["T"], step_tag, x["faces"], log=log, deadline=deadline,
                                         every=1 if test_steps else 20, **hm)
        with open(os.path.join(out_dir, "heat_monitors.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(hres["names"])
            w.writerows(hres["monitors"].tolist())
        heat_march_out = dict(hres["info"], variant=x["label"])
        if allv and hres["T_on_mean_flows"][0] is not None:
            heat_march_out["marched_minus_mean_flow_K"] = {}
            for y, Tm, Ts in zip(heats, hres["T_means"], hres["T_on_mean_flows"]):
                ro_m = readouts(y["prob"], y["heat"], r, Tm, y["faces"], y["t_in"], s)
                ro_s = readouts(y["prob"], y["heat"], r, Ts, y["faces"], y["t_in"], s)
                for tag, ro in (("marched", ro_m), ("mean flow, same window", ro_s)):
                    for k, val in ro.items():
                        rows.append([cid, f"{y['label']}|{tag}", k, repr(float(val))])
                fields[f"T_{y['label']}_marched"] = Tm
                heat_march_out["marched_minus_mean_flow_K"][y["label"]] = {k: ro_m[k] - ro_s[k] for k in ro_m if k.endswith("(K)")}
                key = "Heated face average (K)"
                log(f"  {y['label']}: heated face above the inlet, marched {ro_m[key] - y['t_in']:.4f} C, on the mean flow of the same window "
                    f"{ro_s[key] - y['t_in']:.4f} C")
        elif hres["T_on_mean_flow"] is not None:
            ro_m = readouts(x["prob"], x["heat"], r, hres["T_mean"], x["faces"], x["t_in"], s)
            ro_s = readouts(x["prob"], x["heat"], r, hres["T_on_mean_flow"], x["faces"], x["t_in"], s)
            for tag, ro in (("marched", ro_m), ("mean flow, same window", ro_s)):
                for k, val in ro.items():
                    rows.append([cid, f"{x['label']}|{tag}", k, repr(float(val))])
            fields[f"T_{x['label']}_marched"], fields[f"T_{x['label']}_meanflow"] = hres["T_mean"], hres["T_on_mean_flow"]
            ch = {k: ro_m[k] - ro_s[k] for k in ro_m if k.endswith("(K)")}
            heat_march_out["marched_minus_mean_flow_K"] = ch
            key = "Heated face average (K)" if "Heated face average (K)" in ro_m else next(k for k in ro_m if k.endswith("average (K)"))
            log(f"  {x['label']}: {key.replace(' (K)', '')} above the inlet, marched {ro_m[key] - x['t_in']:.4f} C, stationary on the mean flow "
                f"of the same window {ro_s[key] - x['t_in']:.4f} C, on the mean flow of the flow-only march {x['ro'][key] - x['t_in']:.4f} C; "
                f"drift over the window {hres['info']['drift_over_window_K'][key]:.4f} C, fluctuation (sd) {hres['info']['fluctuation_sd_K'][key]:.4f} C")
        flush_log()
    for x in heats:
        for k, val in x["ro"].items():
            rows.append([cid, x["label"], k, repr(float(val))])
        fields[f"T_{x['label']}"] = x["T"]
    with open(os.path.join(out_dir, "readouts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "variant", "readout", "value"])
        w.writerows(rows)
    fields.update(u=r.u.astype(np.float32), p=r.p.astype(np.float32))
    np.savez_compressed(os.path.join(out_dir, "fields.npz"), **fields)
    np.savez_compressed(os.path.join(out_dir, "mesh.npz"), points=prob.points, tets=prob.tets, dom=prob.dom, tris=prob.tris,
                        tri_bnd=prob.tri_bnd, tri_up=prob.tri_up, tri_down=prob.tri_down, water_nodes=r.nodes,
                        water_elements=r.el, velocity_cells=r.cells2, velocity_node_xyz=r.xyz, edge_nodes=r.edge_nodes,
                        domain_ids=np.array(sorted(prob.domain_names)),
                        domain_names=np.array([prob.domain_names[d] for d in sorted(prob.domain_names)]))
    ck = os.path.join(out_dir, "checkpoint.npz")
    if os.path.exists(ck) and state in ("ok", "unsteady"):
        os.remove(ck)
    log(f"done in {time.time() - t_start:.0f} s: {state}")
    flush_log()
    out = dict(state=state, flow=flow_res, viscosity_check=worst_change, **size)
    if march_res:
        out["time_stepping"] = march_res
    if heat_march_out:
        out["heat_march"] = heat_march_out
    import robust_solve                                                    # 2026-10-14: what the checked solves needed
    out["linear_solves"] = dict(robust_solve.STATS)
    log(f"  linear solves: {robust_solve.STATS}")
    return out


def _child(cid, out_root, dry, test_steps, max_hours):
    out_dir = os.path.join(out_root, cid)
    os.makedirs(out_dir, exist_ok=True)
    st = dict(case=cid, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()
    try:
        st.update(run_one(cid, out_root, dry, test_steps, max_hours))
    except BaseException as e:                                   # includes MemoryError and KeyboardInterrupt
        st.update(state="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(st["traceback"], flush=True)
    st.update(seconds=round(time.time() - t0), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import psutil
        mi = psutil.Process().memory_info()
        st["peak_memory_GB"] = round(getattr(mi, "peak_wset", getattr(mi, "rss", 0)) / 1e9, 2)
        if not hasattr(mi, "peak_wset"):
            import resource
            st["peak_memory_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    except Exception:
        pass
    name = "STATUS.json" if not (dry or test_steps) else ("STATUS_dry.json" if dry else "STATUS_test.json")
    wb.write_json(os.path.join(out_dir, name), st)
    return 0 if st["state"] != "failed" else 1


# --------------------------------------------------------------------- batch
def state_of(cid, out_root, name="STATUS.json"):
    p = os.path.join(out_root, cid, name)
    if not os.path.exists(p):
        return "", None
    try:
        with open(p, encoding="utf-8") as f:
            st = json.load(f)
        return st.get("state", "?"), st
    except Exception:
        return "unreadable", None


def write_status(out_root, name="STATUS.json"):
    os.makedirs(out_root, exist_ok=True)
    target = "status.csv" if name == "STATUS.json" else name.replace("STATUS", "status").replace(".json", ".csv")
    with open(os.path.join(out_root, target), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "case", "size", "state", "seconds", "peak_memory_GB", "flow_unknowns", "estimated_memory_GB",
                    "estimated_seconds_per_flow_step", "flow_status", "flow_steps", "unsteady_amplitude_Pa", "unsteady_period_s", "finished", "error"])
        for i, cid in enumerate(cases()):
            state, st = state_of(cid, out_root, name)
            st = st or {}
            fl, ts = st.get("flow") or {}, st.get("time_stepping") or {}
            w.writerow([i, cid, size_class(settings_of(cid)), state, st.get("seconds", ""), st.get("peak_memory_GB", ""), st.get("flow_unknowns", ""),
                        st.get("estimated_memory_GB", ""), st.get("estimated_seconds_per_flow_step", ""), fl.get("status", ""), fl.get("steps", ""),
                        ts.get("amplitude_Pa", ""), ts.get("period_s", ""), st.get("finished", ""), st.get("error", "")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--indices", action="store_true", help="print the numbers of the selected cases, comma separated")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--block", nargs="*")
    ap.add_argument("--index", type=int, nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--size", choices=["coarse", "normal", "fine"], nargs="*", help="keep only cases of these size classes")
    ap.add_argument("--pending", action="store_true", help="keep only cases that were tried and failed, crashed or did not converge")
    ap.add_argument("--dry-run", action="store_true", help="build the problem, do not solve")
    ap.add_argument("--test-steps", type=int, default=0, help="loop test: this many flow steps, then the heat solves")
    ap.add_argument("--max-hours", type=float, default=None, help="stop the flow iteration of a case after this long and save")
    ap.add_argument("--redo", action="store_true", help="run cases again even if they finished")
    ap.add_argument("--retry", action="store_true", help="run again the cases that failed or did not converge")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--run-one", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.run_one:
        sys.exit(_child(a.run_one, a.out, a.dry_run, a.test_steps, a.max_hours))

    ids = list(cases())
    todo = []
    if a.all or (a.list and not (a.case or a.block or a.index)) or (a.indices and not (a.case or a.block or a.index)):
        todo = list(ids)
    if a.case:
        todo += [c for c in ids if any(x in c for x in a.case)]
    if a.block:
        todo += [c for c in ids if c.split("_")[0] in a.block]
    if a.index:
        todo += [ids[i] for i in a.index]
    todo = list(dict.fromkeys(todo))
    if a.size:
        todo = [c for c in todo if size_class(settings_of(c)) in a.size]
    if a.pending:
        todo = [c for c in todo if state_of(c, a.out)[0] in ("failed", "crashed", "not_converged", "unreadable")]
    if a.indices:
        print(",".join(str(ids.index(c)) for c in todo))
        return
    if a.list:
        for cid in todo:
            s = settings_of(cid)
            print(f"{ids.index(cid):3d}  {cid:36s} {size_class(s):6s} {len(s['variants']):2d} variant(s)  {state_of(cid, a.out)[0] or 'not_run'}")
        n = {k: sum(size_class(settings_of(c)) == k for c in ids) for k in ("coarse", "normal", "fine")}
        print(f"{len(ids)} cases defined: {n}")
        return
    status_name = "STATUS.json" if not (a.dry_run or a.test_steps) else ("STATUS_dry.json" if a.dry_run else "STATUS_test.json")
    if not (a.redo or a.dry_run or a.test_steps):
        skip = ("ok", "unsteady") if a.retry else ("ok", "unsteady", "not_converged", "failed", "crashed")
        todo = [c for c in todo if state_of(c, a.out)[0] not in skip]
    if not todo:
        print("nothing to run (use --list, --case, --block, --index or --all; --retry for failed cases)")
        return
    print(f"{len(todo)} case(s), one at a time", flush=True)
    failed, t0 = [], time.time()
    for n, cid in enumerate(todo, 1):
        cmd = [sys.executable, "-u", "-X", "utf8", os.path.abspath(__file__), "--run-one", cid, "--out", a.out]
        cmd += (["--dry-run"] if a.dry_run else []) + (["--test-steps", str(a.test_steps)] if a.test_steps else [])
        cmd += (["--max-hours", str(a.max_hours)] if a.max_hours else [])
        log_path = os.path.join(a.out, cid, "console_dry.log" if a.dry_run else "console.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPLICA).returncode
        except Exception as e:                                   # the process could not even be started
            rc = -1
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\ncould not run: {e}\n")
        state, st = state_of(cid, a.out, status_name)
        if rc != 0 and state != "failed":                        # killed or crashed without writing its status
            state = "crashed"
            wb.write_json(os.path.join(os.path.dirname(log_path), status_name),
                          dict(case=cid, state="crashed", error=f"process ended with code {rc}; see {os.path.basename(log_path)}",
                               finished=time.strftime("%Y-%m-%d %H:%M:%S")))
        if state not in ("ok", "dry", "test", "unsteady"):
            failed.append(cid)
        extra = ""
        if st and state == "dry":
            extra = f" - {st['flow_unknowns']} flow unknowns, about {st['estimated_memory_GB']} GB, {st['estimated_seconds_per_flow_step']} s per flow step"
        elif state not in ("ok", "test"):
            extra = f" - {(st or {}).get('error', (st or {}).get('time_stepping', 'see console.log'))}"
        print(f"[{n}/{len(todo)} {time.time() - t0:6.0f} s] {cid}: {state}{extra}", flush=True)
        write_status(a.out, status_name)
    print(f"finished: {len(todo) - len(failed)} done, {len(failed)} failed or not converged" + (": " + ", ".join(failed) if failed else ""))


if __name__ == "__main__":
    main()
