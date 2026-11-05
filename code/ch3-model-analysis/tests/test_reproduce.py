"""The package must reproduce analysis notebook v3.8.0 exactly.

reference/reference_v380.json was produced by running the notebook headlessly on the same
data and calling the notebook's own functions (tests/make_reference.py). With every
[corrections] switch off, every number must match to 1e-9 (relative). Run:

    python tests/test_reproduce.py path/to/Model_data
"""
import json
import os
import sys
import tomllib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from run_analysis import analyse  # noqa: E402

TOL = 1e-9
FAILS, N = [], [0]


def close(a, b, tol=TOL):
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        na = a is None or (isinstance(a, float) and np.isnan(a))
        nb = b is None or (isinstance(b, float) and np.isnan(b))
        return na and nb
    if isinstance(a, (str, bool, np.bool_)) or isinstance(b, (str, bool)):
        return a == b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def check(cond, msg):
    N[0] += 1
    if not cond:
        FAILS.append(msg)


def cmp_rows(got, ref, keys, what):
    """got: DataFrame; ref: list of dicts. Compare every ref column present in got, row by row."""
    check(len(got) == len(ref), f"{what}: {len(got)} rows vs {len(ref)}")
    for i, (g, r) in enumerate(zip(got.to_dict("records"), ref)):
        for k, v in r.items():
            gk = keys.get(k, k)
            if gk is None:
                continue
            check(gk in g, f"{what}[{i}]: column {gk} missing")
            if gk in g:
                gv = g[gk]
                gv = None if (isinstance(gv, float) and np.isnan(gv)) else gv
                check(close(gv, v), f"{what}[{i}].{k}: {gv} vs {v}")


def main(data_dir):
    ref = json.load(open(os.path.join(HERE, "reference", "reference_v380.json"), encoding="utf-8"))
    cfg = tomllib.load(open(os.path.join(ROOT, "config", "config.toml"), "rb"))
    cfg["corrections"] = {k: False for k in cfg["corrections"]}          # notebook behaviour
    res = analyse(data_dir, cfg)
    reg = res["registry"]

    # 1. Same runs, same frames.
    check(len(reg.manifest) == ref["registry"]["n_runs"], "run count")
    check(sorted(reg.manifest.run_id) == ref["registry"]["run_ids"], "run ids differ")
    got_blocks = {f"{b}|{v}": int(n) for (b, v), n in reg.manifest.groupby(["block", "variant"]).size().items()}
    check(got_blocks == ref["registry"]["by_block"], "runs by block differ")
    for rid, s in ref["signals"].items():
        f = reg.signals[rid]
        check(list(f.shape) == s["shape"], f"{rid} shape")
        check(close(float(np.nansum(f.index.values.astype(float))), s["index_sum"], 1e-12), f"{rid} index")
        for c, v in s["cols"].items():
            check(c in f.columns and close(float(np.nansum(pd.to_numeric(f[c], errors="coerce").values)), v, 1e-12),
                  f"{rid}.{c} checksum")

    # 2. Mesh convergence.
    g = res["gci"]
    for r in ref["gci"]:
        m = g[(g.family == r["family"]) & (g.variant == r["variant"]) & (g.channel == r["channel"])]
        check(len(m) == 1, f"gci row {r['family']} {r['variant']} {r['channel']}")
        if len(m) != 1:
            continue
        m = m.iloc[0]
        order = cfg["mesh"]["scr_order"] if r["family"] == "SCR" else cfg["mesh"]["wb_order"]
        for k, col in (("coarse", f"T_{order[0]} (C)"), ("medium", f"T_{order[1]} (C)"), ("fine", f"T_{order[2]} (C)"),
                       ("p", "p"), ("gci_pct", "GCI (%)"), ("asymp", "asymptotic_ratio"), ("status", "status")):
            v = m[col]
            check(close(None if (isinstance(v, float) and np.isnan(v)) else v, r[k]), f"gci {r['variant']} {r['channel']} {k}: {v} vs {r[k]}")
    check(len(g) == len(ref["gci"]), f"gci rows {len(g)} vs {len(ref['gci'])}")
    check(res["wb_ref_run"] == ref["wb_ref_run"], "WB fine reference run")
    gt = res["gci_time"]
    for r in ref["gci_time"]:
        m = gt[(gt.variant == r["variant"]) & (gt.channel == r["channel"])].iloc[0]
        for k, col in (("n", "n_samples"), ("early", "early mean (%)"), ("mid", "mid mean (%)"), ("late", "late mean (%)"),
                       ("n_mono", "n_monotonic"), ("p_mean", "mean p")):
            v = m[col]
            check(close(None if (isinstance(v, float) and np.isnan(v)) else v, r[k]), f"gci_time {r['variant']} {r['channel']} {k}")
    series_sum = {k: float(np.nansum(v[1])) for k, v in res["gci_series"].items()}
    for r in ref["gci_time"]:
        check(close(series_sum[(r["variant"], r["channel"])], r["gci_sum"]), f"gci_time series {r['variant']} {r['channel']}")

    # 3. Energy.
    cmp_rows(res["energy"], ref["energy"], {}, "energy")
    cmp_rows(res["efficiency"], ref["efficiency"], {}, "efficiency")
    cmp_rows(res["wb_dissipation"], ref["wb_dissipation"], {}, "wb_dissipation")
    cmp_rows(res["wb_closure"], ref["wb_closure"], {}, "wb_closure")

    # 4. Thermal resistance (notebook reference = half-model values).
    rt = res["scr_rth"].set_index("Variant")
    for v, r in ref["scr_rth"].items():
        check(close(rt.loc[v, "R_steady_raw_export (K/W)"], r["R_steady"]), f"R_steady {v} (raw export, as in the notebook)")
        # this test runs with every correction off, so q_TEC is not doubled here: the notebook's values
        check(close(rt.loc[v, "R_steady_half_model (K/W)"], r["R_steady"]), f"R_half {v} (corrections off)")
        check(close(rt.loc[v, "R_steady_full_device (K/W)"], r["R_steady"] / 2), f"R_full {v} (corrections off)")
        for k, col in (("R_overall_mean", "R_overall_mean_half (K/W)"), ("R_min", "R_min_half (K/W)"),
                       ("R_max", "R_max_half (K/W)"), ("t_ss_start", "steady_window_start (min)")):
            check(close(rt.loc[v, col], r[k]), f"{k} {v}")
    cmp_rows(res["wb_rth"], ref["wb_rth"], {}, "wb_rth")

    # 5. Dimensionless numbers.
    cmp_rows(res["dimensionless"], ref["dimensionless"], {}, "dimensionless")
    cmp_rows(res["reynolds_nominal"], ref["reynolds_nominal"], {}, "reynolds_nominal")

    # 6. Master metrics (row order identical).
    cmp_rows(res["master"], ref["master_metrics"], {}, "master_metrics")

    # 7. Sensitivity.
    sens = {(e["family"], e["mode"], e["variant"], e["metric"]): e for e in res["sensitivity"]}
    check(len(sens) == len(ref["sensitivity"]), f"sensitivity cases {len(sens)} vs {len(ref['sensitivity'])}")
    for r in ref["sensitivity"]:
        e = sens.get((r["family"], r["mode"], r["variant"], r["metric"]))
        tag = f"{r['family']} {r['mode']} {r['variant']} {r['metric']}"
        check(e is not None, f"sensitivity {tag} missing")
        if e is None:
            continue
        check(len(e["X"]) == r["n"], f"{tag} n")
        for probe, col in r["pearson"].items():
            for inp, v in col.items():
                check(close(e["pearson"].loc[inp, probe], v), f"{tag} pearson {inp}->{probe}")
        check(set(e["anova"]) == set(r["anova"]), f"{tag} anova probes")
        for probe, rows in r["anova"].items():
            if probe in e["anova"]:
                a = e["anova"][probe].set_index("Factor")
                for row in rows:
                    check(row["Factor"] in a.index, f"{tag} anova {probe} factor {row['Factor']} missing")
                    if row["Factor"] not in a.index:
                        continue
                    for k in ("df", "eta^2 (%)", "omega^2 (%)"):
                        check(close(a.loc[row["Factor"], k], row[k], 1e-7), f"{tag} anova {probe} {row['Factor']} {k}")
        if r["pca"] is None:
            check(e["pca"] is None, f"{tag} pca")
        elif e["pca"] is not None:
            for i, v in enumerate(r["pca"]["evr"]):
                check(close(e["pca"]["evr"][i], v, 1e-7), f"{tag} pca evr {i}")
            for pc, col in r["pca"]["loadings_abs"].items():
                for probe, v in col.items():
                    check(close(abs(e["pca"]["loadings"].loc[probe, pc]), v, 1e-6), f"{tag} loading {probe} {pc}")
            for pc, col in r["pca"]["reg_abs"].items():
                for inp, v in col.items():
                    check(close(abs(e["pca"]["reg"].loc[inp, pc]), v, 1e-6), f"{tag} pca reg {inp} {pc}")

    # 8. Step-13 per-run thermal summaries.
    from comsol_analysis.metrics import thermal_summary_run
    for rid, rows in ref["thermal_summary_13"].items():
        cmp_rows(thermal_summary_run(reg.signals[rid], cfg), rows, {}, f"summary13 {rid}")

    print(f"{N[0]} checks, {len(FAILS)} failures")
    for f in FAILS[:40]:
        print("  FAIL", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "Model_data")))
