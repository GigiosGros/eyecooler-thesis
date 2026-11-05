"""Test of the DEFAULT configuration — the one that produces the thesis numbers.

    python tests/test_default.py path/to/data            compare with the frozen reference
    python tests/test_default.py path/to/data --update   re-freeze after an intended change

test_reproduce.py pins the notebook v4.28 settings; this test covers config/config.toml
as shipped. Two parts:

1. Independent checks, no reference needed: every trace is accounted for; each
   per-channel test (difference, 90 % CI, p, Holm, equivalence, donor-level repeat) is
   recomputed here from the per-trace table with scipy / statsmodels.
2. Regression against tests/reference/reference_default.json: exclusions and reasons,
   group membership, group summary, per-channel statistics, uncertainty. A difference
   means the results changed; if the change is intended, re-freeze with --update and
   say why in the commit / README.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from run_analysis import analyse  # noqa: E402
from scec_analysis import qc  # noqa: E402

REF = os.path.join(HERE, "reference", "reference_default.json")
FAILS: list[str] = []
KEYS = ["species", "status", "ring_material", "ring"]


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def close(a, b, tol=1e-9):
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return (a is None or (isinstance(a, float) and np.isnan(a))) and (b is None or (isinstance(b, float) and np.isnan(b)))
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def independent(res, cfg):
    """Recompute every per-channel test from the per-trace table."""
    runs, kin = res["runs"], res["kinetics"]
    alpha, margin = cfg["statistics"]["alpha"], cfg["statistics"]["equivalence_margin_C"]
    donors = cfg["statistics"]["donor_of_human_eye"]
    excl = qc.exclusion_table(runs, cfg)
    check(len(excl) == len(runs) * len(cfg["channels"]["temperature"]), "exclusion table does not list every trace")
    check(int((excl.reason == "").sum()) == len(kin), "included traces and kinetics rows differ")
    device = cfg["channels"]["device"]
    for (a, b), (_, table) in res["comparisons"].items():
        ka = kin[kin.run_id.isin(res["groups"][a])]
        kb = kin[kin.run_id.isin(res["groups"][b])]
        for _, r in table.iterrows():
            if pd.isna(r.get("diff_C", np.nan)):
                continue
            ch = r["channel"]
            xa = ka[ka.channel == ch].set_index("eye").dT
            xb = kb[kb.channel == ch].set_index("eye").dT
            paired = ka.species.iloc[0] == kb.species.iloc[0] and ka.status.iloc[0] == kb.status.iloc[0]
            shared = xa.index.intersection(xb.index)
            paired = paired and len(shared) >= 2
            tag = f"{a} - {b} / {ch}"
            if paired:
                d = (xa[shared] - xb[shared]).to_numpy()
                diff, se, df = d.mean(), d.std(ddof=1) / np.sqrt(d.size), d.size - 1
                p = stats.ttest_1samp(d, 0.0).pvalue
            else:
                x, y = xa.to_numpy(), xb.to_numpy()
                va, vb = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
                diff, se = x.mean() - y.mean(), np.sqrt(va + vb)
                df = (va + vb) ** 2 / (va ** 2 / (x.size - 1) + vb ** 2 / (y.size - 1))
                p = stats.ttest_ind(x, y, equal_var=False).pvalue
            q = stats.t.ppf(1 - alpha, df)
            check(r["design"].startswith("paired") == paired, f"{tag}: design {r['design']}")
            for name, mine in (("diff_C", diff), ("p_raw", p), ("ci90_low", diff - q * se), ("ci90_high", diff + q * se)):
                check(close(r[name], mine), f"{tag}: {name} {r[name]} vs independent {mine}")
            check(bool(r["equivalent"]) == bool(diff - q * se > -margin and diff + q * se < margin), f"{tag}: equivalence")
            # donor level
            if pd.notna(r.get("diff_donor_C", np.nan)):
                don = lambda idx: [donors.get(e.lstrip("E"), e) if sp == "Human" else f"{sp}{st}{e}"  # noqa: E731
                                   for e, sp, st in idx]
                if paired:
                    g = pd.Series((xa[shared] - xb[shared]).to_numpy(),
                                  index=don([(e, "Human", "") for e in shared])).groupby(level=0).mean().to_numpy()
                    dd, dse, ddf = g.mean(), g.std(ddof=1) / np.sqrt(g.size), g.size - 1
                else:
                    ga = pd.Series(xa.to_numpy(), index=don([(e, ka.species.iloc[0], ka.status.iloc[0]) for e in xa.index])).groupby(level=0).mean().to_numpy()
                    gb = pd.Series(xb.to_numpy(), index=don([(e, kb.species.iloc[0], kb.status.iloc[0]) for e in xb.index])).groupby(level=0).mean().to_numpy()
                    va, vb = ga.var(ddof=1) / ga.size, gb.var(ddof=1) / gb.size
                    dd, dse = ga.mean() - gb.mean(), np.sqrt(va + vb)
                    ddf = (va + vb) ** 2 / (va ** 2 / (ga.size - 1) + vb ** 2 / (gb.size - 1))
                dq = stats.t.ppf(1 - alpha, ddf)
                for name, mine in (("diff_donor_C", dd), ("ci90_donor_low", dd - dq * dse), ("ci90_donor_high", dd + dq * dse)):
                    check(close(r[name], mine), f"{tag}: {name} {r[name]} vs independent {mine}")
        fam = table[~table.channel.isin(device) & table.get("p_raw", pd.Series(dtype=float)).notna()]
        if len(fam):
            mine = multipletests(fam.p_raw.to_numpy(), method="holm")[1]
            check(np.allclose(fam.p_holm.to_numpy(), mine, atol=1e-12), f"{a} - {b}: Holm differs from statsmodels")


def snapshot(res, cfg):
    runs = res["runs"]
    snap = {"exclusions": {f"{rid}|{ch}": v for rid, r in sorted(runs.items()) for ch, v in sorted(r.exclusions.items()) if v},
            "groups": {k: sorted(v) for k, v in res["groups"].items()},
            "summary": {}, "stats": {},
            "U_abs": res["u_abs"][2], "U_diff": res["u_diff"][2], "noise_95": res["noise_95"],
            "config": {k: cfg[k] for k in ("calibration", "filter", "markers", "statistics")}
            | {"qc": {k: v for k, v in cfg["qc"].items() if k != "manual_exclusions"},
               "manual_exclusions": [[m["run_id"], m["channel"]] for m in cfg["qc"].get("manual_exclusions", [])]}}
    for _, r in res["summary"].iterrows():
        snap["summary"][" | ".join([*(r[k] for k in KEYS), r["channel"]])] = {
            c: (None if pd.isna(r[c]) else float(r[c])) for c in res["summary"].columns if c not in KEYS + ["channel"]}
    cols = ["design", "diff_C", "p_raw", "p_holm", "ci90_low", "ci90_high", "equivalent", "hedges_g",
            "n_donors", "diff_donor_C", "ci90_donor_low", "ci90_donor_high", "equivalent_donor"]
    for (a, b), (_, table) in res["comparisons"].items():
        for _, r in table.iterrows():
            snap["stats"][f"{a} || {b} || {r['channel']}"] = {
                c: (None if (c not in r or pd.isna(r[c])) else (r[c] if isinstance(r[c], str) else
                                                             bool(r[c]) if isinstance(r[c], (bool, np.bool_)) else float(r[c])))
                for c in cols}
    return json.loads(json.dumps(snap, default=str))


def compare(got, want, path=""):
    if isinstance(want, dict):
        for k in sorted(set(want) | set(got if isinstance(got, dict) else {})):
            if not isinstance(got, dict) or k not in got:
                check(False, f"{path}/{k}: missing in the package results")
            elif k not in want:
                check(False, f"{path}/{k}: not in the reference")
            else:
                compare(got[k], want[k], f"{path}/{k}")
    elif isinstance(want, float) or isinstance(got, float):
        check(close(got, want, 1e-8), f"{path}: {got} vs reference {want}")
    else:
        check(got == want, f"{path}: {got} vs reference {want}")


def main(data_dir, update):
    cfg = tomllib.load(open(os.path.join(ROOT, "config", "config.toml"), "rb"))
    res = analyse(data_dir, cfg)
    independent(res, cfg)
    n_ind = len(FAILS)
    snap = snapshot(res, cfg)
    if update:
        with open(REF, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=1, sort_keys=True)
        print(f"reference written: {REF}")
    elif not os.path.exists(REF):
        check(False, f"no reference yet ({REF}); run with --update")
    else:
        compare(snap, json.load(open(REF, encoding="utf-8")))
    if FAILS:
        print(f"FAILED ({len(FAILS)}; {n_ind} independent, {len(FAILS) - n_ind} against the reference):")
        for f in FAILS[:60]:
            print("  -", f)
        sys.exit(1)
    n = sum(pd.notna(t.get("diff_C", pd.Series(dtype=float))).sum() for _, t in res["comparisons"].values())
    print(f"PASSED — default configuration: {len(res['kinetics'])} included traces, {n} per-channel tests recomputed "
          f"independently; {'reference re-frozen' if update else 'matches the frozen reference'}.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--update"]
    main(args[0] if args else os.environ.get("SCEC_DATA", "UploadExperiment"), "--update" in sys.argv)
