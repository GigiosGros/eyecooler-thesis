"""Run the complete COMSOL model analysis (SCR ring + waterblock).

    python run_analysis.py --data path/to/Model_data --out results

Writes to --out:
  00_summary/           report.md (readable summary), run_log.json (parameters, software versions,
                        SHA-256 of every input file, run counts), model_analysis.xlsx (all tables)
  01_data/              manifest.csv (one row per simulation run), integrity.csv, skipped_files.csv
  02_mesh_convergence/  gci.csv (final value, SCR + waterblock), gci_time.csv, figures
  03_energy/            scr_energy.csv, scr_efficiency.csv, area_check.csv, scr_energy_balance.csv,
                        wb_dissipation.csv, wb_closure.csv (+ per point), figures
  04_thermal_resistance/ scr_rth.csv (full device + half model), scr_rth_timeseries.csv,
                        wb_rth.csv, wb_rth_summary.csv, wb_sweep_vs_stationary.csv, figures
  05_dimensionless/     reynolds_nominal.csv, entrance_length.csv, dimensionless.csv
  06_metrics/           master_metrics.csv (every run x probe)
  07_sensitivity/       dominant_factors_<metric>.csv, <family>_<mode>/<variant>_<metric>/
                        pearson.csv, anova.csv, pca.csv, figures
  08_comparisons/       <set>/effects.csv, summary.csv, overlay.png
  09_effectiveness/     superposition_fit.csv (ε per variant x site), required_ring_temperature.csv,
                        run_epsilon.csv (material / silicone / mesh runs), kinetics.csv + kinetics_summary.csv,
                        reheat_persistence.csv, design_across_conditions.csv, figures
  10_exvivo_comparison/ model_vs_exvivo.csv, site_ordering.csv, figure (different conditions — not a validation)
  11_waterblock/        flow_model_runs.csv, flow_model_gci.csv, hot_side_resistance.csv, resistance_vs_flow.csv,
                        max_surface_temperature.csv, figures
"""
from __future__ import annotations

import argparse
import hashlib
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

import comsol_analysis  # noqa: E402
from comsol_analysis import rescale  # noqa: E402
from comsol_analysis.sensitivity_sheets import sensitivity_sheets  # noqa: E402
from comsol_analysis import (comparisons, dimensionless, effectiveness, energy, exvivo, figures, io,  # noqa: E402
                             mesh, metrics, resistance, sensitivity, waterblock)
from comsol_analysis.safe_write import save_csv, save_text  # noqa: E402


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analyse(data_dir: str, cfg: dict) -> dict:
    """Everything except writing files; returns a dict of results (used by tests)."""
    reg = io.build_registry(data_dir, cfg)
    gci = mesh.gci_table(reg, cfg)
    gci_time, gci_series = mesh.gci_time_summary(reg, cfg)
    E = energy.scr_energies(reg, cfg)
    closure, closure_pts = energy.wb_closure(reg, cfg, per_point=True)
    rth, rth_series = resistance.scr_rth_table(reg, cfg)
    wb_rth = resistance.wb_thermal_resistance(reg, cfg)
    re_nom, entrance = dimensionless.reynolds_nominal(cfg)
    sens = sensitivity.run_all(reg, cfg)
    sets = comparisons.comparison_sets(reg, cfg)
    eps = effectiveness.superposition_fit(reg, cfg)
    kin = effectiveness.kinetics(reg, cfg)
    exp = exvivo.load_experiment(os.path.normpath(os.path.join(HERE, cfg["exvivo"]["summary_csv"])))
    cmp = exvivo.compare(reg, exp, cfg)
    flow_runs, flow_gci = waterblock.flow_model_study(data_dir, cfg)
    hot = waterblock.hot_side_resistance(reg, cfg)
    sup_on = cfg.get("superposition", {}).get("enabled", False)   # derived tables, off by default
    added = dict(body_start=rescale.from_body_start(reg, cfg) if sup_on else pd.DataFrame(),
                 exvivo_rescaled=rescale.exvivo_like_for_like(reg, exp, cfg) if sup_on else pd.DataFrame(),
                 epsilon=eps, required_ring=effectiveness.required_ring_temperature(eps, cfg),
                 run_epsilon=effectiveness.run_epsilon(reg, cfg), kinetics=kin,
                 kinetics_summary=effectiveness.kinetics_summary(kin), persistence=effectiveness.persistence(reg, cfg),
                 design=effectiveness.design_across_conditions(reg, gci, cfg),
                 exvivo=cmp, exvivo_material=exvivo.material_comparison(reg, cfg), exvivo_order=exvivo.site_ordering(cmp) if not cmp.empty else pd.DataFrame(),
                 flow_runs=flow_runs, flow_gci=flow_gci, hot_side=hot, r_vs_flow=waterblock.resistance_vs_flow(hot),
                 wb_diff_band=waterblock.difference_mesh_band(reg, flow_runs, cfg),
                 wb_max=waterblock.max_tec_temperature(hot, cfg) if not hot.empty else pd.DataFrame())
    return dict(**added,
        registry=reg, integrity=io.integrity_audit(reg), gci=gci, gci_time=gci_time, gci_series=gci_series,
        wb_ref_run={v: mesh.wb_ref_run(reg, v, cfg) for v in cfg["vocabulary"]["variants"]},
        energy=E, efficiency=energy.efficiencies(E), area_check=energy.area_check(E),
        energy_balance=energy.scr_energy_balance(reg, cfg), wb_dissipation=energy.wb_dissipation(reg, cfg),
        wb_closure=closure, wb_closure_points=closure_pts, scr_rth=rth, scr_rth_series=rth_series,
        wb_rth=wb_rth, wb_rth_summary=resistance.wb_rth_summary(wb_rth),
        wb_consistency=resistance.wb_study_consistency(reg, cfg),
        reynolds_nominal=re_nom, entrance=entrance, dimensionless=dimensionless.dimensionless_table(reg, cfg),
        master=metrics.master_metrics(reg, cfg), sensitivity=sens,
        dominant={m: sensitivity.dominant_factors(sens, m) for m in ("dT", "Tfinal")},
        comparison_sets=sets,
        comparisons={k: (comparisons.effect_table(reg, g, cfg), comparisons.group_summaries(reg, g, cfg))
                     for k, g in sets.items()})


def write_outputs(res: dict, cfg: dict, data_dir: str, out: str, cfg_path: str):
    reg = res["registry"]
    scr_chans = cfg["probes"]["scr_point"] + cfg["probes"]["retina_agg"] + cfg["probes"]["eye_agg"]

    def folder(*parts):
        path = os.path.join(out, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def csv(df, *parts):
        save_csv(df, os.path.join(folder(*parts[:-1]), parts[-1]))

    def fig(*parts):
        return os.path.join(folder(*parts[:-1]), parts[-1])

    # 01 — data
    manifest = io.manifest_table(reg)
    csv(manifest, "01_data", "manifest.csv")
    csv(res["integrity"], "01_data", "integrity.csv")
    csv(pd.DataFrame(reg.skipped, columns=["file", "reason"]), "01_data", "skipped_files.csv")

    # 02 — mesh convergence
    csv(res["gci"], "02_mesh_convergence", "gci.csv")
    csv(res["gci_time"], "02_mesh_convergence", "gci_time.csv")
    figures.mesh_convergence(res["gci"], "SCR", cfg["mesh"]["scr_order"], fig("02_mesh_convergence", "scr_convergence.png"))
    figures.mesh_convergence(res["gci"], "Waterblock", cfg["mesh"]["wb_order"], fig("02_mesh_convergence", "wb_convergence.png"))
    figures.gci_time(res["gci_series"], cfg["probes"]["main"] + ["Optic Nerve"], fig("02_mesh_convergence", "gci_time_resolved.png"))

    # 03 — energy
    csv(res["energy"], "03_energy", "scr_energy.csv")
    csv(res["efficiency"], "03_energy", "scr_efficiency.csv")
    csv(res["area_check"], "03_energy", "area_check.csv")
    csv(res["energy_balance"], "03_energy", "scr_energy_balance.csv")
    csv(res["wb_dissipation"], "03_energy", "wb_dissipation.csv")
    csv(res["wb_closure"], "03_energy", "wb_closure.csv")
    csv(res["wb_closure_points"], "03_energy", "wb_closure_per_point.csv")
    figures.scr_energy(res["energy"], fig("03_energy", "scr_energy.png"))
    figures.wb_closure(res["wb_closure"], fig("03_energy", "wb_closure.png"))
    figures.energy_balance(reg, fig("03_energy", "scr_energy_balance.png"))

    # 04 — thermal resistance
    csv(res["scr_rth"], "04_thermal_resistance", "scr_rth.csv")
    ts = pd.concat({v: d for v, d in res["scr_rth_series"].items()}, names=["Variant"]).reset_index()
    csv(ts, "04_thermal_resistance", "scr_rth_timeseries.csv")
    csv(res["wb_rth"], "04_thermal_resistance", "wb_rth.csv")
    csv(res["wb_rth_summary"], "04_thermal_resistance", "wb_rth_summary.csv")
    csv(res["wb_consistency"], "04_thermal_resistance", "wb_sweep_vs_stationary.csv")
    figures.scr_rth(res["scr_rth_series"], fig("04_thermal_resistance", "scr_rth.png"))
    figures.wb_rth(res["wb_rth"], fig("04_thermal_resistance", "wb_rth.png"))

    # 05 — dimensionless numbers
    csv(res["reynolds_nominal"], "05_dimensionless", "reynolds_nominal.csv")
    csv(res["entrance"], "05_dimensionless", "entrance_length.csv")
    csv(res["dimensionless"], "05_dimensionless", "dimensionless.csv")
    csv(effectiveness.timeseries(reg, cfg), "09_effectiveness", "timeseries_thesis.csv")

    # 06 — master metrics
    csv(res["master"], "06_metrics", "master_metrics.csv")

    # 07 — sensitivity
    for m, df in res["dominant"].items():
        csv(df, "07_sensitivity", f"dominant_factors_{m}.csv")
    for e in res["sensitivity"]:
        sub = (f"{e['family']}_{slug(e['mode'])}", f"{e['variant']}_{e['metric']}")
        pr = e["pearson"].reset_index().rename(columns={"index": "input"})
        csv(pr, "07_sensitivity", *sub, "pearson.csv")
        if e["anova"]:
            csv(pd.concat([a.assign(probe=p) for p, a in e["anova"].items()], ignore_index=True),
                "07_sensitivity", *sub, "anova.csv")
        if e["pca"] is not None:
            p = e["pca"]
            tab = p["loadings"].copy()
            tab.loc["explained_variance_ratio"] = p["evr"]
            csv(tab.reset_index().rename(columns={"index": "probe"}), "07_sensitivity", *sub, "pca_loadings.csv")
            csv(p["reg"].reset_index().rename(columns={"index": "input"}), "07_sensitivity", *sub, "pca_input_regression.csv")
        if e["variant"] == "All" and e["metric"] in ("dT", "Tfinal"):
            figures.pearson_heatmap(e["pearson"], f"{e['family']} {e['mode']} — Pearson ρ, input → {sensitivity.METRICS[e['metric']]}",
                                    fig("07_sensitivity", *sub, "pearson.png"))
            key = "Optic Nerve Head" if e["family"] == "SCR" else "TEC"
            if key in e["anova"]:
                figures.anova_bars(e["anova"][key], f"{e['family']} {e['mode']} — η² on {key} ({e['metric']})",
                                   fig("07_sensitivity", *sub, f"anova_{slug(key)}.png"))
        if e["family"] == "SCR" and e["variant"] in cfg["vocabulary"]["variants"] and e["metric"] in ("dT", "Tfinal"):
            figures.contour(e["X"], e["Y"], "Optic Nerve Head",
                            f"SCR {e['variant']} — Optic Nerve Head {sensitivity.METRICS[e['metric']]} over (T0, Tb)",
                            fig("07_sensitivity", *sub, "contour_optic_nerve_head.png"))

    # 08 — design comparisons
    for name, (eff, summ) in res["comparisons"].items():
        csv(eff, "08_comparisons", name, "effects.csv")
        csv(summ, "08_comparisons", name, "summary.csv")
        figures.group_overlay(reg, res["comparison_sets"][name], cfg["probes"]["main"],
                              f"{name.replace('_', ' ')}: " + " vs ".join(res["comparison_sets"][name]),
                              fig("08_comparisons", name, "overlay.png"))

    # 09 — effectiveness, kinetics, persistence, design comparison across conditions
    for k, name in (("epsilon", "superposition_fit"), ("required_ring", "required_ring_temperature"),
                    ("run_epsilon", "run_epsilon"), ("kinetics", "kinetics"), ("kinetics_summary", "kinetics_summary"),
                    ("persistence", "reheat_persistence"), ("design", "design_across_conditions")):
        csv(res[k], "09_effectiveness", f"{name}.csv")
    figures.epsilon_bars(res["epsilon"], res["run_epsilon"], fig("09_effectiveness", "effectiveness.png"))
    # 10 — ex-vivo comparison
    if not res["exvivo"].empty:
        csv(res["exvivo"], "10_exvivo_comparison", "model_vs_exvivo.csv")
        csv(res["exvivo_order"], "10_exvivo_comparison", "site_ordering.csv")
        if not res["exvivo_material"].empty:
            csv(res["exvivo_material"], "10_exvivo_comparison", "material_aluminum_vs_ceramic.csv")
        figures.exvivo(res["exvivo"], fig("10_exvivo_comparison", "model_vs_exvivo.png"))
    # 12 — runs rescaled by superposition (only with [superposition] enabled)
    for k, name in (("body_start", "from_body_temperature_start"), ("exvivo_rescaled", "model_vs_exvivo_like_for_like")):
        if not res[k].empty:
            csv(res[k], "12_superposition", f"{name}.csv")
    # 11 — waterblock sizing
    for k, name in (("flow_runs", "flow_model_runs"), ("flow_gci", "flow_model_gci"), ("hot_side", "hot_side_resistance"),
                    ("r_vs_flow", "resistance_vs_flow"), ("wb_max", "max_surface_temperature"),
                    ("wb_diff_band", "difference_mesh_band")):
        if not res[k].empty:
            csv(res[k], "11_waterblock", f"{name}.csv")
    if not res["flow_runs"].empty:
        figures.flow_model(res["flow_runs"], fig("11_waterblock", "laminar_vs_turbulent_mesh.png"))
    if not res["hot_side"].empty:
        figures.wb_resistance(res["hot_side"], fig("11_waterblock", "resistance_vs_flow.png"))

    # 00 — Excel workbook (the notebook's Step 14 sheets + the new diagnostics), summary, provenance
    sheets = {"Master_Metrics": res["master"], "Manifest": manifest, "GCI": res["gci"], "GCI_time": res["gci_time"],
              "SCR_Energy": res["energy"], "SCR_Efficiency": res["efficiency"], "SCR_Energy_Balance": res["energy_balance"],
              "WB_Dissipation": res["wb_dissipation"], "WB_Closure": res["wb_closure"], "SCR_Thermal_R": res["scr_rth"],
              "WB_Thermal_R": res["wb_rth"], "WB_Sweep_vs_Stationary": res["wb_consistency"], "Dimensionless": res["dimensionless"], "Reynolds_nominal": res["reynolds_nominal"],
              "Effectiveness": res["epsilon"], "Required_ring_T": res["required_ring"], "Run_effectiveness": res["run_epsilon"],
              "Kinetics_summary": res["kinetics_summary"], "Reheat_persistence": res["persistence"],
              "Design_across_conditions": res["design"], "ExVivo_comparison": res["exvivo"],
              "WB_flow_model_GCI": res["flow_gci"], "WB_hot_side_R": res["hot_side"], "WB_R_vs_flow": res["r_vs_flow"]}
    sheets.update(sensitivity_sheets(out))      # 07_sensitivity tables (written above), long format
    xlsx = os.path.join(folder("00_summary"), "model_analysis.xlsx")
    tmp = xlsx.replace(".xlsx", ".tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        for k, df in sheets.items():
            if not df.empty:
                df.to_excel(xw, sheet_name=k[:31], index=False)
    os.replace(tmp, xlsx)

    inputs = {f: sha256(os.path.join(data_dir, f)) for f in reg.files}
    log = {"package_version": comsol_analysis.__version__, "run_at": datetime.now().isoformat(timespec="seconds"),
           "python": platform.python_version(),
           "libraries": {m: __import__(m).__version__ for m in ("numpy", "pandas", "scipy", "statsmodels", "sklearn")},
           "data_dir": os.path.abspath(data_dir), "config_file": os.path.abspath(cfg_path), "config": cfg,
           "input_sha256": inputs, "n_runs": len(reg.manifest), "n_files_dispatched": int(reg.manifest.file.nunique()),
           "skipped_files": reg.skipped, "n_time_truncated": reg.n_truncated,
           "runs_by_block": {f"{b}|{v}": int(n) for (b, v), n in reg.manifest.groupby(["block", "variant"]).size().items()}}
    save_text(json.dumps(log, indent=2, default=str), os.path.join(folder("00_summary"), "run_log.json"))
    write_report(res, cfg, os.path.join(folder("00_summary"), "report.md"))


def _md(df, floatfmt=".3g"):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                    else ("—" if isinstance(v, (float, np.floating)) else str(v)) for v in r) + " |")
    return L


def write_report(res: dict, cfg: dict, path: str):
    reg, c = res["registry"], cfg["corrections"]
    main = cfg["probes"]["main"]
    sites = ["Anterior Sclera", "Equatorial Sclera", "Vitreous Humor", "Retina Avg", "Posterior Sclera", "Optic Nerve Head"]
    g, eps, rq, eb = res["gci"], res["epsilon"], res["required_ring"], res["energy_balance"]
    ex, mat, fg, d = res["exvivo"], res["exvivo_material"], res["flow_gci"], res["design"]

    def e(v, s_):
        return float(eps[(eps.variant == v) & (eps.site == s_)]["epsilon (w_Tc)"].iloc[0])

    rq37 = rq[rq["Tb (C)"] == 37.0]

    def need(s_, t):
        return float(rq37[(rq37.variant == "V1") & (rq37.site == s_) & (rq37["target (C)"] == t)]["required_ring_T (C)"].iloc[0])

    onh_at4 = float(rq37[(rq37.variant == "V1") & (rq37.site == "Optic Nerve Head")]["T_site_at_4C_ring (C)"].iloc[0])
    main_g = g[(g.family == "SCR") & g.channel.isin(main)]
    closed = eb[(eb.closure_if_interfaces_halved - 1).abs() < 0.01]
    spread = eps[eps.site.isin(sites)].groupby("site")["epsilon (w_Tc)"].agg(lambda x: x.max() - x.min()).max()
    L = [f"# COMSOL model analysis — report (package {comsol_analysis.__version__})", "",
         f"{len(reg.manifest)} simulation runs from {reg.manifest.file.nunique()} COMSOL exports. Method of notebook v3.8.0, "
         "reproduced exactly; corrections are listed at the end. Temperatures in °C, times in minutes.", "",
         "## Key findings", "",
         "- **Steady state is one number per site.** Final temperatures are an exact weighted mean of body and ring "
         f"temperature (27-run sweep, residual ≤ {eps['max_abs_residual (K)'].max():.3f} K). Effectiveness "
         f"ε = (Tb − T_site)/(Tb − T_ring), V1: optic nerve head {e('V1', 'Optic Nerve Head'):.2f}, posterior sclera "
         f"{e('V1', 'Posterior Sclera'):.2f}, retina (avg) {e('V1', 'Retina Avg'):.2f}, vitreous {e('V1', 'Vitreous Humor'):.2f}. "
         "The starting (ambient) temperature does not affect the steady state.",
         f"- **Targets (Tb = 37 °C).** The optic nerve head reaches 35 °C with the ring at ≤ {need('Optic Nerve Head', 35.0):.0f} °C; "
         f"32 °C would need {need('Optic Nerve Head', 32.0):.0f} °C, which is not reachable above the 4 °C floor "
         f"(with a 4 °C ring it settles at {onh_at4:.1f} °C). The retina average reaches 32 °C with the ring at "
         f"≤ {need('Retina Avg', 32.0):.0f} °C.",
         f"- **Ring designs are nearly equivalent.** ε differs by ≤ {spread:.3f} between V1, V2 and V3; the largest T_final "
         f"difference at any of the 27 operating points is {d[d.site.isin(sites)][['dT_final min (K)', 'dT_final max (K)']].abs().max().max():.2f} K. "
         "Material matters more (§3)."]
    re0 = res["run_epsilon"]
    mt = re0[(re0.study == "Material") & re0.site.isin(sites)].pivot_table(index=["variant", "site"], columns="material", values="epsilon")
    if {"Aluminum", "Ceramic", "Glass"} <= set(mt.columns):
        rc, rg = mt.Ceramic / mt.Aluminum, mt.Glass / mt.Aluminum
        L.append(f"- **Ring material.** Relative to aluminum, a ceramic (Al2O3) ring keeps {100 * rc.min():.0f}–{100 * rc.max():.0f} % "
                 f"of the effectiveness at every site; glass (SiO2) keeps only {100 * rg.min():.0f}–{100 * rg.max():.0f} %.")
    if not ex.empty:
        deep = ex[ex.site != "Anterior Sclera"]["eps ratio exp/model"]
        L.append("- **Model vs ex vivo (human eyes, different conditions).** The model ranks the sites exactly as the experiment "
                 f"does; ex vivo, sites away from the ring receive {deep.min():.1f}–{deep.max():.1f}x more of the ring's cold than "
                 "in the model — consistent with the model's perfusion and eyelid heat load, both absent ex vivo — while the "
                 "anterior sclera matches.")
    if not mat.empty:
        mat_t = mat[mat.site != "SCR"]
        onh = mat[mat.site == "Optic Nerve Head"].iloc[0]
        ant = mat[mat.site == "Anterior Sclera"].iloc[0]
        L.append("- **Aluminum vs ceramic.** The model predicts a small aluminum advantage in ΔT (optic nerve head "
                 f"{onh['model d(dT) Al-ceramic (K)']:+.2f} K, anterior sclera {ant['model d(dT) Al-ceramic (K)']:+.2f} K). The paired "
                 f"human experiment cannot resolve it (all Holm p ≥ {mat['exp p (Holm)'].min():.2f}) and trends toward ceramic at the "
                 f"tissue sites; the model lies inside the experimental 90 % CI at {int(mat_t['model inside exp 90%CI'].sum())} of "
                 f"{len(mat_t)} tissue sites ({', '.join(mat_t[mat_t['model inside exp 90%CI']].site.str.lower())}).")
    L.append(f"- **Model integrity.** Mesh uncertainty at the main SCR sites ≤ {main_g['U_mesh (K)'].max():.2f} °C. The SCR energy "
             f"balance closes to 1.000 in {len(closed)} of {len(eb)} runs once the exported q_TEC is doubled (interior face); the "
             f"waterblock closes to {100 * res['wb_closure'].ratio.min():.1f}–{100 * res['wb_closure'].ratio.max():.1f} %. The turbulent "
             "waterblock study is not mesh-converged; the laminar one is used. Its absolute temperatures are converged, but the "
             "surface-minus-coolant differences that define the hot-side resistances are not, except V1 at the TEC face (see section 5).")
    if not res["wb_max"].empty:
        L.append(f"- **Waterblock safety.** Hottest TEC / waterblock surface over the whole sweep {res['wb_max']['max (C)'].max():.1f} °C "
                 f"(≥ {res['wb_max']['margin_to_43C (K)'].min():.0f} °C below the 43 °C IEC 60601-1 limit).")

    L += ["", "## 1. Model credibility", "", "### Mesh convergence (three-grid GCI, finest mesh)", "",
          "U_mesh is the absolute uncertainty band (K); GCI % is relative to the value in °C and exaggerates probes near 0–8 °C.", ""]
    L += _md(main_g[["variant", "channel", "p", "GCI (%)", "U_mesh (K)", "asymptotic_ratio", "status"]])
    L += ["", "### SCR energy balance (Σ interface heat / TEC heat, steady state)", "",
          "The raw exports give exactly 2.000 because the TEC cold face is an interior boundary: the exported q_TEC is the mean "
          "of its two sides, one of which carries no heat, i.e. half the heat through the face. With q_TEC doubled the balance "
          "closes; where it does not, heat enters through faces missing from the interface selections. The complete division "
          "for those configurations comes from COMSOL's stored boundary fluxes (scr-heat-replica/extract_comsol_energy.py), "
          "which close in 8 of 9 configurations (the ninth has no stored mesh).", ""]
    L += _md(eb[["variant", "silicone", "steady_power_ratio_last5min", "closure_if_interfaces_halved", "heat_not_in_interface_selections (%)"]])
    L += ["", "### Waterblock energy closure (coolant enthalpy out / hot-side heat in)", ""]
    L += _md(res["wb_closure"])
    if not fg.empty:
        L += ["", "### Waterblock flow model: laminar vs turbulent (SST), same operating point, three meshes", ""]
        L += _md(fg)

    L += ["", "## 2. SCR cooling performance", "", "### Effectiveness ε per site (27-run superposition fit)", ""]
    L += _md(eps[eps.site.isin(sites + ["Optic Nerve"])].pivot(index="site", columns="variant", values="epsilon (w_Tc)").reset_index())
    L += ["", "### Ring temperature needed (Tb = 37 °C)", ""]
    L += _md(rq37[rq37.site.isin(sites)][["variant", "site", "target (C)", "required_ring_T (C)",
                                          "reachable_above_safety_floor", "T_site_at_4C_ring (C)"]])
    L += ["", "### Kinetics (median over the 27 runs)", ""]
    L += _md(res["kinetics_summary"][res["kinetics_summary"].site.isin(sites)])
    ps = res["persistence"]
    ps = (ps[ps.site.isin(sites) & ~ps.never_cooled_below_35].groupby(["variant", "site"])["minutes_below_35_after_off"]
          .agg(["min", "median", "max"]).reset_index())
    L += ["", "### Persistence after switch-off (reheat runs: minutes a site stays below 35 °C)", ""]
    L += _md(ps)

    L += ["", "## 3. Design comparisons", "", "### Ring variants — T_final difference at all 27 operating points", ""]
    L += _md(d[d.site.isin(sites)][["A", "B", "site", "dT_final mean (K)", "dT_final min (K)", "dT_final max (K)",
                                     "same_sign_everywhere", "U_mesh combined (K)", "exceeds_mesh_band"]])
    re_ = res["run_epsilon"]
    L += ["", "### Materials (baseline operating point, 30 min) — ε per site", ""]
    L += _md(re_[(re_.study == "Material") & re_.site.isin(sites)]
             .pivot_table(index=["variant", "site"], columns="material", values="epsilon").reset_index())
    L += ["", "### Silicone layer — ε per site (energy split for V2/V3 0.5 mm and V2 1.0 mm is incomplete, see §1)", ""]
    sil = re_[re_.study.isin(["Silicone", "Parametric baseline"]) & re_.site.isin(sites)]
    L += _md(sil.pivot_table(index=["variant", "site"], columns="silicone", values="epsilon").reset_index())
    L += ["", "### Energy delivered and efficiency (full device, 35 min, interface double counting removed)", ""]
    L += _md(res["efficiency"][["Variant", "Silicone", "E_eye (J)", "E_eyelids (J)", "E_air (J)",
                                "eta_eye (%) [J]", "eta_body (%) [J]", "Eye/Body (%) [J]"]])

    if not ex.empty:
        L += ["", "## 4. Model vs ex vivo (human eyes)", "",
              "Different conditions: the model has perfusion, metabolism and eyelids; the ex-vivo eye sits in a 37 °C bath "
              "with none of these. This is a comparison, not a validation. V1 = original ring, V2 = reduced bulk (V3, reduced "
              "bulk with corneal lens, has no ex-vivo counterpart); the ceramic original ring is compared with the V1 ceramic "
              "run. ε_bath = (37 − T_final)/(37 − T_ring) with each side's own ring temperature; experimental 95 % CI of the mean.", ""]
        L += _md(ex[["group", "site", "n_eyes", "eps_bath model", "eps_bath exp", "eps_bath exp 95%CI low",
                     "eps_bath exp 95%CI high", "eps ratio exp/model", "t90 model (min)", "t90 exp (min)"]])
        L += ["", "Site ranking agreement (Spearman ρ of ε across the five sites):", ""] + _md(res["exvivo_order"])
        if not mat.empty:
            L += ["", "### Aluminum vs ceramic (original ring): model V1 runs vs the paired ex-vivo comparison", "",
                  "ΔT = T_initial − T_final, aluminum minus ceramic (positive = aluminum cools more). Experiment: paired within "
                  "the same eyes, 90 % CI, Holm-adjusted p. The rings were applied in sequence (ceramic last), so an order "
                  "effect cannot be excluded on the experimental side.", ""]
            L += _md(mat[["site", "model d(dT) Al-ceramic (K)", "exp d(dT) Al-ceramic (K)", "exp 90%CI low", "exp 90%CI high",
                          "exp design", "exp p (Holm)", "model inside exp 90%CI", "same direction"]])

    if not res["hot_side"].empty:
        L += ["", "## 5. Waterblock", "",
              "### Hot-side resistance vs flow (laminar stationary study, Fine mesh; V3 has no TEC probe)", ""]
        L += _md(res["r_vs_flow"][["Variant", "Q0 (L/min)", "R_TEC (K/W) mean", "R_WB (K/W) mean",
                                   "TEC_rise_above_inlet (K/W) mean", "n_points"]])
        db = res["wb_diff_band"]
        if not db.empty:
            st = db[db.study == "stationary auxiliary"]
            L += ["", "### Mesh band of the differences behind the resistances", "",
                  "A resistance is a surface-minus-coolant temperature difference divided by the imposed hot-side heat, so it "
                  "carries the mesh band of that difference. The differences are small "
                  f"({st[[c for c in st.columns if c.startswith('dT_')]].min().min():.2f}–"
                  f"{st[[c for c in st.columns if c.startswith('dT_')]].max().max():.2f} °C at the mesh reference point), "
                  "so the hot side stays within about 1 °C of the coolant on every mesh; as resistances, only the rows marked "
                  "CONVERGED are quantitative, the others are an order of magnitude and do not support a ranking of the variants.", ""]
            L += _md(db[["variant", "study", "difference"] + [c for c in db.columns if c.startswith("dT_")]
                        + ["change Normal to Fine (%)", "band (% of dT)", "U_mesh (C)", "band_method", "status"]])
        L += ["", "### Hottest surface over the sweep", ""] + _md(res["wb_max"])
        L += ["", "Channel Reynolds numbers are 1 000–5 000 (`05_dimensionless/`); Dittus-Boelter is outside its validity "
              "range everywhere. Pressure drop was not exported, so pumping power cannot be assessed.", ""]

    L += ["## 6. Corrections to notebook v3.8.0 and notes", ""]
    L += [f"- `{k}`: {'on' if v else 'OFF'}" for k, v in c.items()]
    L += ["- SCR R_th = (T_hot − T_cold)/q_TEC is the resistance of the modelled TEC layer (a passive aluminium layer); "
          f"full device V1/V2/V3 = {', '.join(f'{x:.3f}' for x in res['scr_rth']['R_steady_full_device (K/W)'])} K/W.",
          "- The turbulent waterblock sweep (Normal mesh) is kept in `04_thermal_resistance/` for reference only; its TEC "
          "temperatures are not mesh-converged.",
          "- ANOVA / PCA on ΔT (`07_sensitivity/`) mostly reflect the starting temperature; use ε and the T_final tables.", ""]
    save_text("\n".join(L) + "\n", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with the COMSOL .csv exports (flat)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=os.path.join(HERE, "config", "config.toml"))
    args = ap.parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    res = analyse(args.data, cfg)
    write_outputs(res, cfg, args.data, args.out, args.config)
    print(f"Done. {len(res['registry'].manifest)} runs. Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
