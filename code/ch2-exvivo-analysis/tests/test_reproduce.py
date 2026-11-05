"""The package must reproduce analysis notebook v4.28 exactly.

reference_v428.json was produced by running the notebook headlessly on the same
data and calling the notebook's own functions. Run:

    python tests/test_reproduce.py path/to/UploadExperiment
"""
import json
import os
import sys
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from run_analysis import analyse  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def main(data_dir):
    ref = json.load(open(os.path.join(HERE, "reference", "reference_v428.json"), encoding="utf-8"))
    cfg = tomllib.load(open(os.path.join(ROOT, "config", "config.toml"), "rb"))
    cfg["calibration"]["human_curve"] = "steinhart_hart"  # notebook v4.28 human conversion
    cfg["calibration"]["plateau_method"] = "kmeans"       # notebook v4.28 plateau levels
    cfg["markers"]["key_by"] = "id"        # notebook v4.28 reads the Slicer control-point id
    cfg["acquisition"].pop("channel_swaps", None)    # the notebook takes every channel as recorded
    cfg["acquisition"].pop("column_exchanges", None)  # and every workbook column with its own curve
    cfg["qc"].pop("hardware_exclusions", None)       # and has no documented hardware list
    cfg["qc"].pop("hardware_railed_fraction", None)  # the notebook has no rail test
    cfg["statistics"]["mixed_model"] = True  # notebook v4.28 also fits the mixed model
    cfg["uncertainty"]["method"] = "notebook"  # notebook v4.28 budget (circuit model)
    cfg["qc"]["apply_manual"] = False      # the notebook has no manual exclusions
    cfg["qc"]["final_window_s"] = 15       # notebook v4.28 endpoint definitions
    cfg["qc"]["plateau_window"] = "t90"
    cfg["qc"]["spatial_fence"] = "two-sided"  # notebook v4.28 spatial screen (Tukey on the constellation deviation)
    res = analyse(data_dir, cfg)
    runs = res["runs"]

    # 1. Same traces excluded, for the same reasons.
    got = {f"{rid}|{ch}": v for rid, r in runs.items() for ch, v in r.exclusions.items() if v}
    check(got == ref["exclusions"], f"exclusions differ: only package {sorted(set(got) - set(ref['exclusions']))}, "
                                    f"only notebook {sorted(set(ref['exclusions']) - set(got))}, "
                                    f"reason mismatches {[k for k in got if k in ref['exclusions'] and got[k] != ref['exclusions'][k]]}")

    # 2. Same group membership.
    for name, ids in ref["groups"].items():
        check(sorted(res["groups"].get(name, [])) == sorted(ids), f"group {name}: {res['groups'].get(name)} vs {ids}")

    # 3. Group summary tables (notebook prints mean (SD) at fixed precision).
    key = {"HumanAlOrig": ("Human", "Fresh", "Aluminum", "Original"),
           "HumanAlRB": ("Human", "Fresh", "Aluminum", "Reduced Bulk"),
           "PigFresh": ("Porcine", "Fresh", "Aluminum", "Original"),
           "PigPres": ("Porcine", "Preserved", "Aluminum", "Original")}
    s = res["summary"]
    cols = {"Init T (°C)": ("T_initial", 1), "Final T (°C)": ("T_final", 1), "ΔT (°C)": ("dT", 1),
            "t₅₀ (s) μ(σ)": ("t50", 0), "t₉₀ (s) μ(σ)": ("t90", 0), "Plateau SD (°C) μ(σ)": ("plateau_SD", 2)}
    for tname, rows in ref["tables"].items():
        sp, st, mat, ring = key[tname]
        for row in rows:
            g = s[(s.species == sp) & (s.status == st) & (s.ring_material == mat) & (s.ring == ring)
                  & (s.channel == row["Sensor"])]
            check(len(g) == 1, f"{tname}/{row['Sensor']}: missing in package summary")
            if len(g) != 1:
                continue
            for col, (base, dec) in cols.items():
                mine = f"{g[base + '_mean'].iloc[0]:.{dec}f} ({g[base + '_sd'].iloc[0]:.{dec}f})"
                check(mine == row[col], f"{tname}/{row['Sensor']}/{col}: {mine} vs {row[col]}")
            check(f"{g['CV_pct'].iloc[0]:.1f}" == row["CV (%)"], f"{tname}/{row['Sensor']}/CV")

    # 4. All pairwise comparisons: models and per-channel statistics.
    for pair, want in ref["stats"].items():
        a, b = pair.split(" || ")
        models, table = res["comparisons"][(a, b)]
        for i, mname in enumerate(["interaction", "additive_tissue", "scr"]):
            w = want["models"][mname]
            m = models.iloc[i]
            if w is None:
                check(np.isnan(m["effect"]), f"{pair}/{mname}: expected no fit")
                continue
            for k, v in zip(["effect", "ci_low", "ci_high", "p"], w):
                check(close(m[k], v, 1e-6), f"{pair}/{mname}/{k}: {m[k]} vs {v}")
        for ch, w in want["channels"].items():
            r = table[table.channel == ch].iloc[0]
            if w is None:
                check("diff_C" not in r or np.isnan(r.get("diff_C", np.nan)), f"{pair}/{ch}: expected no test")
                continue
            check(r["design"] == (f"paired k={w['n'][0]}" if w["paired"] else f"Welch {w['n'][0]},{w['n'][1]}"),
                  f"{pair}/{ch}: design {r['design']}")
            for k, v in (("diff_C", w["diff"]), ("p_raw", w["p"]), ("ci90_low", w["ci"][0]),
                         ("ci90_high", w["ci"][1]), ("hedges_g", w["g"]), ("g_ci_low", w["gci"][0]),
                         ("g_ci_high", w["gci"][1])):
                check(close(r[k], v, 1e-9), f"{pair}/{ch}/{k}: {r[k]} vs {v}")
            if "p_holm" in w:
                check(close(r["p_holm"], w["p_holm"], 1e-9), f"{pair}/{ch}/p_holm")

    # 5. Uncertainty (the notebook rounds each term to 4-5 decimals before summing).
    check(close(res["noise_95"], ref["noise"], 1e-9), f"noise {res['noise_95']} vs {ref['noise']}")
    check(abs(res["u_abs"][2] - ref["U_abs"]) < 2e-3, f"U_abs {res['u_abs'][2]} vs {ref['U_abs']}")
    check(abs(res["u_diff"][2] - ref["U_diff"]) < 1e-6, f"U_diff {res['u_diff'][2]} vs {ref['U_diff']}")

    n_checks = "exclusions, groups, summary tables, 10 comparisons, uncertainty"
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS[:60]:
            print("  -", f)
        sys.exit(1)
    print(f"PASSED — package reproduces notebook v4.28 ({n_checks}).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SCEC_DATA", "UploadExperiment"))
