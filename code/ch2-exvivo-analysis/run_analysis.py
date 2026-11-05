"""Run the complete SCEC ex-vivo thermal analysis.

    python run_analysis.py --data path/to/UploadExperiment --out results

Writes to --out:
  00_summary/           report.md (readable summary), run_log.json (parameters, software
                        versions, SHA-256 of every input file, trace counts)
  01_quality_control/   exclusions.csv, spatial_screen.csv, temperature_fence.csv
                        (report only, never applied), spatial_screen.png
  02_groups/            all_groups_kinetics.csv, all_groups_summary.csv, all_groups.png
    <group>/            kinetics.csv, summary.csv, curves.png (all channels on one plot)
  03_comparisons/
    <A>__vs__<B>/       channels.csv (per-channel tests; models.csv only with the legacy mixed model),
                        curves.png (one panel per channel), overlay.png (all channels on one plot)
  04_auxiliary/
    pressure/  current/  ambient/    per_run.csv, by_group.csv
  05_uncertainty/       absolute.csv, differential.csv
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import tomllib
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scec_analysis
from scec_analysis.safe_write import save_csv, save_text
from scec_analysis import (auxiliary, calibration, diagnostics, endpoints, figures, io, processing, qc,
                           statistics, uncertainty)

SHORT = {("Human", "Fresh", "Aluminum", "Original"): "Human, aluminum original",
         ("Human", "Fresh", "Aluminum", "Reduced Bulk"): "Human, aluminum reduced-bulk",
         ("Human", "Fresh", "Ceramic", "Original"): "Human, ceramic original",
         ("Porcine", "Fresh", "Aluminum", "Original"): "Porcine fresh, aluminum original",
         ("Porcine", "Preserved", "Aluminum", "Original"): "Porcine preserved, aluminum original"}


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analyse(data_dir: str, cfg: dict):
    """Everything except writing files; returns a dict of results (used by tests)."""
    runs, cal_files, markers = io.load_dataset(data_dir, cfg)
    sh = calibration.calibrate(runs, cal_files, cfg)
    processing.process(runs, cfg)
    spatial, temp_fence = qc.run_qc(runs, markers, cfg)
    kin = endpoints.kinetics_table(runs, cfg)
    summary = endpoints.group_summary(kin, cfg)

    groups = {}
    for rid, r in sorted(runs.items()):
        groups.setdefault(SHORT.get(r.group, io.group_label(r.group)), []).append(rid)
    order = [n for n in SHORT.values() if n in groups] + [n for n in groups if n not in SHORT.values()]
    groups = {n: groups[n] for n in order}

    comparisons = {}
    for a, b in itertools.combinations(groups, 2):
        comparisons[(a, b)] = statistics.compare(runs, (a, groups[a]), (b, groups[b]), cfg)

    if cfg["uncertainty"].get("method", "notebook") == "measured":
        sd = uncertainty.plateau_noise_sd(cal_files, cfg)
        noise = 1.96 * sd                                          # reported as a 95 % half-width, as before
        u = dict(cfg["uncertainty"])
        if u.get("linear_conversion") == "human":                  # 2026-10-19: human lines, never below the bench values
            h = uncertainty.human_line_terms(cal_files, cfg)
            u["linear_interpolation_error_C"] = max(u["linear_interpolation_error_C"], h["abs"])
            u["linear_interpolation_dT_small_C"] = max(u["linear_interpolation_dT_small_C"], h["small"])
            u["linear_interpolation_dT_large_C"] = max(u["linear_interpolation_dT_large_C"], h["large"])
            u["linear_source"] = "human calibration lines"
        u_abs = uncertainty.measured_absolute_budget(u, sd)
        run_sd = uncertainty.run_scatter_sd(runs, cfg)
        u_diff = uncertainty.measured_differential_budget(u, sd, run_sd)
        # porcine cohort (2026-10-20): fixed transfer function = the 2022 bench
        # line (its own linearity terms), trimmed at 20.5 and 37 C (span 16.5 C)
        up = dict(cfg["uncertainty"])
        up["linear_source"] = "2022 bench table, porcine transfer function"
        u_abs_p = uncertainty.measured_absolute_budget(up, sd)
        u_diff_p = uncertainty.measured_differential_budget(up, sd, run_sd, cal_span_C=37.0 - 20.5)
    else:                                                          # notebook v4.28 budget (circuit model)
        noise = uncertainty.plateau_noise(cal_files, sh, cfg)
        u_abs = uncertainty.absolute_budget(cfg["uncertainty"], noise)
        u_diff = uncertainty.differential_budget(cfg["uncertainty"], noise)
        u_abs_p = u_diff_p = None
    return dict(u_abs_porcine=u_abs_p, u_diff_porcine=u_diff_p, runs=runs, markers=markers, groups=groups, spatial=spatial, temp_fence=temp_fence,
                kinetics=kin, summary=summary, comparisons=comparisons, noise_95=noise,
                u_abs=u_abs, u_diff=u_diff, calibration_files=cal_files, steinhart_hart=sh,
                current_fit=calibration.fit_pooled_current(cal_files, cfg))


def write_outputs(res: dict, cfg: dict, data_dir: str, out: str, cfg_path: str):
    runs, groups = res["runs"], res["groups"]

    def folder(*parts):
        path = os.path.join(out, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def csv(df, *parts):
        save_csv(df, os.path.join(folder(*parts[:-1]), parts[-1]))

    # 01 — quality control
    excl = qc.exclusion_table(runs, cfg)
    csv(excl, "01_quality_control", "exclusions.csv")
    csv(res["spatial"], "01_quality_control", "spatial_screen.csv")
    csv(res["temp_fence"], "01_quality_control", "temperature_fence.csv")
    figures.plot_spatial(res["spatial"], cfg, os.path.join(folder("01_quality_control"), "spatial_screen.png"))
    diagnostics.spatial_outlier_planes(res["spatial"], cfg, folder("01_quality_control"), slug)
    diagnostics.manual_exclusions_figure(res["runs"], cfg, os.path.join(folder("01_quality_control"), "manual_exclusions.png"),
                                         lambda g: figures.display_name(SHORT.get(g, " ".join(g)), cfg))
    # thesis Aim 1 cohorts only (aluminum original ring)
    original = {g for g in SHORT if g[3] == "Original" and g[2] == "Aluminum"}
    diagnostics.trace_exclusions_figure(res["runs"], cfg, os.path.join(folder("01_quality_control"), "trace_exclusions.png"),
                                        lambda g: figures.display_name(SHORT.get(g, " ".join(g)), cfg), original)
    figures.plot_spatial_print(res["spatial"], cfg, os.path.join(folder("01_quality_control"), "spatial_screen_print.png"))
    diagnostics.temperature_fence_figure(res["temp_fence"], cfg,
                                         os.path.join(folder("01_quality_control"), "temperature_fence.png"),
                                         lambda g: SHORT.get(g, " · ".join(g)))

    # 02 — groups (all together, then one folder per group)
    kin, summ = res["kinetics"], res["summary"]
    csv(kin, "02_groups", "all_groups_kinetics.csv")
    csv(summ, "02_groups", "all_groups_summary.csv")
    figures.plot_groups(runs, groups, cfg, os.path.join(folder("02_groups"), "all_groups.png"),
                        "All groups (mean ± SD)", color_by="group")
    for name, ids in groups.items():
        key = runs[ids[0]].group
        sel = lambda df: df[(df.species == key[0]) & (df.status == key[1])
                            & (df.ring_material == key[2]) & (df.ring == key[3])]
        csv(sel(kin), "02_groups", slug(name), "kinetics.csv")
        csv(sel(summ), "02_groups", slug(name), "summary.csv")
        curves = []                                               # group mean +/- SD per channel (thesis figures)
        for ch in cfg["channels"]["temperature"]:
            res_ch = figures.group_mean(runs, ids, ch)
            if res_ch is not None:
                mu, sd, n = res_ch
                curves.append(pd.DataFrame({"channel": ch, "time (s)": mu.index.to_numpy(float),
                                            "mean (C)": mu.to_numpy(float), "sd (C)": sd.to_numpy(float), "n": n}))
        if curves:
            csv(pd.concat(curves, ignore_index=True), "02_groups", slug(name), "mean_curves.csv")
        figures.plot_overlay(runs, {name: ids}, cfg,
                             os.path.join(folder("02_groups", slug(name)), "curves.png"),
                             f"{figures.display_name(name, cfg)} ({len(ids)} runs)")
        diagnostics.individual_traces_figure(runs, ids, cfg,
                                             os.path.join(folder("02_groups", slug(name)), "individual_traces.png"),
                                             f"{figures.display_name(name, cfg)} — every included trace (color = eye)")
        diagnostics.fft_figure(runs, ids, cfg, os.path.join(folder("02_groups", slug(name)), "fft.png"),
                               f"{figures.display_name(name, cfg)} — mean amplitude spectrum of the calibrated traces")
        for rid in ids:                                          # 08 — one figure per run
            diagnostics.run_figure(runs[rid], cfg, os.path.join(folder("08_runs", slug(name)), f"{rid}.png"))

    # 03 — pairwise comparisons, one folder each
    for (a, b), (models, table) in res["comparisons"].items():
        pair = f"{slug(a)}__vs__{slug(b)}"
        if not models.empty:                                      # legacy mixed model (statistics.mixed_model)
            csv(models, "03_comparisons", pair, "models.csv")
        csv(table, "03_comparisons", pair, "channels.csv")
        figures.plot_groups(runs, {a: groups[a], b: groups[b]}, cfg,
                            os.path.join(folder("03_comparisons", pair), "curves.png"),
                            figures.comparison_title([a, b], cfg))
        figures.plot_overlay(runs, {a: groups[a], b: groups[b]}, cfg,
                             os.path.join(folder("03_comparisons", pair), "overlay.png"),
                             figures.comparison_title([a, b], cfg))

    # 03 — post hoc sensitivity of the species comparison (config [sensitivity])
    sens = statistics.species_sensitivity(runs, groups, cfg)
    if not sens.empty:
        csv(sens, "03_comparisons", "sensitivity_species_primary_endpoint.csv")

    # 03 — the visual (manual) exclusions switched off: every comparison re-run with those
    # traces retained, as the thesis says it was (2026-10-19). One row per
    # comparison and channel whose input changed; columns _excl = as reported, _kept =
    # with the visually excluded traces retained.
    if cfg.get("sensitivity", {}).get("manual_exclusions_off", False) and cfg["qc"].get("apply_manual", True):
        import copy
        cfg2 = copy.deepcopy(cfg)
        cfg2["qc"]["apply_manual"] = False
        res2 = analyse(data_dir, cfg2)
        keep = ["design", "diff_C", "ci90_low", "ci90_high", "p_raw", "p_holm"]
        rows = []
        for (a, b), (_, t1) in res["comparisons"].items():
            if (a, b) not in res2["comparisons"]:
                continue
            t2 = res2["comparisons"][(a, b)][1]
            m = t1[["channel"] + keep].merge(t2[["channel"] + keep], on="channel",
                                            suffixes=("_excl", "_kept"))
            m = m[(m["design_excl"] != m["design_kept"]) |
                  ((m["diff_C_excl"] - m["diff_C_kept"]).abs() > 1e-12)]
            if not m.empty:
                m.insert(0, "comparison", f"{a} - {b}")
                rows.append(m)
        if rows:
            csv(pd.concat(rows, ignore_index=True), "03_comparisons",
                "sensitivity_manual_exclusions_off.csv")

    # 03 — post hoc contrast of one ring against the mean of others (config [geometry_contrast])
    geom = statistics.geometry_contrast(runs, groups, cfg)
    if not geom.empty:
        csv(geom, "03_comparisons", "geometry_contrast.csv")

    # 04 — auxiliary recordings
    p = auxiliary.pressure_summary(runs, cfg)
    c = auxiliary.current_summary(runs, cfg)
    amb = auxiliary.ambient_summary(runs)
    for name, df, cols in (("pressure", p, [k for k in p.columns if k.startswith("mean_")]),
                           ("current", c, ["mean_A", "first_60s_A", "last_60s_A"]),
                           ("ambient", amb, ["room_temp_C"])):
        csv(df, "04_auxiliary", name, "per_run.csv")
        if not df.empty:
            csv(auxiliary.group_means(df, cols), "04_auxiliary", name, "by_group.csv")

    # 05 — measurement uncertainty
    ua, ud = res["u_abs"], res["u_diff"]
    csv(ua[0], "05_uncertainty", "absolute.csv")
    csv(ud[0], "05_uncertainty", "differential.csv")
    if res.get("u_diff_porcine") is not None:                     # porcine budget
        csv(res["u_abs_porcine"][0], "05_uncertainty", "absolute_porcine.csv")
        dp = res["u_diff_porcine"][0].copy()
        dp.attrs = {}
        csv(dp, "05_uncertainty", "differential_porcine.csv")
    diagnostics.uncertainty_figure(ua, os.path.join(folder("05_uncertainty"), "absolute.png"),
                                   "Absolute uncertainty of a single reading (contributions at k = 2)")
    diagnostics.uncertainty_figure(ud, os.path.join(folder("05_uncertainty"), "differential.png"),
                                   "Differential uncertainty of ΔT (contributions at k = 2)")

    # 06 — calibration, 07 — sensor positions
    diagnostics.calibration_figures(res["calibration_files"], res["steinhart_hart"], res["current_fit"],
                                    cfg, folder("06_calibration"), slug)
    diagnostics.spatial_figures(res["spatial"], cfg, folder("07_spatial"), slug)

    # 00 — summary and provenance
    inputs = {r.file: sha256(r.file) for r in runs.values()}
    counts = excl["reason"].replace("", "included").value_counts().to_dict()
    log = {"package_version": scec_analysis.__version__, "run_at": datetime.now().isoformat(timespec="seconds"),
           "python": platform.python_version(),
           "libraries": {m: __import__(m).__version__ for m in ("numpy", "pandas", "scipy", "statsmodels", "sklearn")},
           "data_dir": os.path.abspath(data_dir), "config_file": os.path.abspath(cfg_path), "config": cfg,
           "input_sha256": {os.path.basename(k): v for k, v in sorted(inputs.items())},
           "groups": groups, "trace_counts": counts,
           "noise_95CI_C": res["noise_95"], "U_absolute_k2_C": ua[2], "U_differential_k2_C": ud[2]}
    save_text(json.dumps(log, indent=2, default=str), os.path.join(folder("00_summary"), "run_log.json"))
    write_report(res, cfg, counts, os.path.join(folder("00_summary"), "report.md"))


def _true(x) -> bool:
    """True only for a real True (a missing test is NaN, which bool() would call True)."""
    return bool(pd.notna(x) and x)


def write_report(res: dict, cfg: dict, counts: dict, path: str):
    s, margin = res["summary"], cfg["statistics"]["equivalence_margin_C"]
    L = [f"# SCEC ex-vivo thermal analysis — report (package {scec_analysis.__version__})", "",
         f"Traces: {counts}. Measurement uncertainty (k = 2): absolute ±{res['u_abs'][2]:.2f} °C, "
         f"differential (ΔT) ±{res['u_diff'][2]:.3f} °C"
         + (f" for a small excursion (optic nerve), ±{res['u_diff'][0].attrs['U_large']:.2f} °C for a large one (ring)"
            if "U_large" in res["u_diff"][0].attrs else "")
         + f". Equivalence margin ±{margin:g} °C.", "",
         "## Groups", ""]
    for name, ids in res["groups"].items():
        L.append(f"- **{name}**: {len(ids)} runs")
    L += ["", "## Cooling excursion ΔT, mean (SD) °C", "",
          "| Group | " + " | ".join(cfg["channels"]["temperature"]) + " |",
          "|---|" + "---|" * len(cfg["channels"]["temperature"])]
    for key, g in s.groupby(["species", "status", "ring_material", "ring"], sort=False):
        cells = []
        for ch in cfg["channels"]["temperature"]:
            r = g[g.channel == ch]
            cells.append("—" if r.empty else f"{r.dT_mean.iloc[0]:.1f} ({r.dT_sd.iloc[0]:.1f}), n={r.n.iloc[0]}")
        L.append(f"| {SHORT.get(key, ' '.join(key))} | " + " | ".join(cells) + " |")
    L += ["", "## Comparisons (A − B) at the optic nerve", ""]
    for (a, b), (models, table) in res["comparisons"].items():
        on = table[table.channel == cfg["channels"]["primary_endpoint"]].iloc[0]
        line = (f"- **{a} − {b}** ({on['design']}): optic nerve Δ = {on.get('diff_C', np.nan):+.2f} °C, "
                f"90% CI [{on.get('ci90_low', np.nan):+.2f}, {on.get('ci90_high', np.nan):+.2f}] "
                f"({'equivalent' if _true(on.get('equivalent')) else 'not shown equivalent'} at ±{margin:g} °C)")
        if pd.notna(on.get("diff_donor_C", np.nan)):
            line += (f"; by donor ({on['n_donors']}) {on['diff_donor_C']:+.2f} "
                     f"[{on['ci90_donor_low']:+.2f}, {on['ci90_donor_high']:+.2f}] "
                     f"({'equivalent' if _true(on['equivalent_donor']) else 'not shown equivalent'})")
        if not models.empty:
            m0, m1 = models.iloc[0], models.iloc[1]
            line += (f"; mixed model at optic nerve {m0['effect']:+.2f} [{m0['ci_low']:+.2f}, {m0['ci_high']:+.2f}], "
                     f"tissue average {m1['effect']:+.2f} [{m1['ci_low']:+.2f}, {m1['ci_high']:+.2f}]")
        L.append(line)
    save_text("\n".join(L) + "\n", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with the .xlsx/.csv/.mrk.json files")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "config.toml"))
    args = ap.parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    res = analyse(args.data, cfg)
    write_outputs(res, cfg, args.data, args.out, args.config)
    print(f"Done. Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
