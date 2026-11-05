"""Chapter 3: result tables of the eye model from the runs of the Python program.

Reads the probe histories of the program's runs (scr-heat-replica, results/<study>/<case>/
probes_*.csv) and, with the same code, the COMSOL histories stored beside them
(comsol_reference.npz), and writes the tables the Chapter 3 figures and text use. No solve is
started and no value is typed in. Definitions are those of thesis-model-analysis
(comsol_analysis/effectiveness.py and exvivo.py): the same formulas, run on another source.

Output, under figures/aim2/sources/data/python/:
  run_epsilon.csv, superposition_fit.csv, required_ring_temperature.csv, design_27points.csv,
  kinetics.csv, kinetics_summary.csv, reheat_persistence.csv, timeseries_thesis.csv,
  model_vs_exvivo.csv, material_vs_exvivo.csv, geometry_vs_exvivo.csv, mesh_steps.csv,
  site_values.csv, agreement.csv, plane_fields_35min.npz, retina_profile_v{1,2,3}.csv
Each table carries a column source = "python" or "comsol", except model_vs_exvivo, which keeps the
package's columns for the Python model (the COMSOL model columns are in
comsol_model_vs_exvivo_reference.csv).

Run:  python code/figures/aim2_python_results.py
      SCR_RESULTS=<scr-heat-replica/results> overrides the default path.
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd

from provenance import SOURCES, THESIS, sha256

DEFAULT = THESIS.parents[2] / "03 - Experiments & Data" / "Analysis Code" / "scr-heat-replica" / "results"
RES = pathlib.Path(os.environ.get("SCR_RESULTS", DEFAULT))
OUT = SOURCES / "data" / "python"
EXV = SOURCES / "data" / "exvivo"
K = 273.15
Tb0, T00, Tc0 = 37.0, 25.0, 4.0
SITES = {"Optic Nerve Head": ("Optic Nerve Head Probe", "optic_nerve_head"),
         "Vitreous Humor": ("Vitreous Humor Probe", "vitreous_humor"),
         "Anterior Sclera": ("Anteriror Sclera Probe", "anterior_sclera"),
         "Posterior Sclera": ("Posterior Sclera Probe", "posterior_sclera"),
         "Equatorial Sclera": ("Equatorial Sclera Probe", "equatorial_sclera"),
         "Optic Nerve": ("Optic Nerve Probe", "optic_nerve"),
         "Retina Avg": ("Domain Probe Retina Avg", "retina_avg"),
         "Retina Max": ("Domain Probe Retina Max", "retina_max"),
         "Retina Min": ("Domain Probe Retina Min", "retina_min"),
         "Eye Avg": ("Domain Probe Eye Avg", "eye_avg"),
         "Eye Max": ("Domain Probe Eye Max", "eye_max"),
         "Eye Min": ("Domain Probe Eye Min", "eye_min")}
ALL = {**SITES, "SCR": ("SCR", "scr")}
USED = set()


def py(study, case, which="cooling"):
    p = RES / study / case / f"probes_{which}.csv"
    USED.add(p)
    d = pd.read_csv(p)
    n = None
    for c in d.columns:
        if c.startswith("SCR "):
            n = c.split()[1]
    out = {"t": d.iloc[:, 0].to_numpy()}
    for s, (a, _) in ALL.items():
        out[s] = d[f"{a} {n} (K)"].to_numpy() - K
    return pd.DataFrame(out).set_index("t")


def ref(study, case):
    p = RES / study / case / "comsol_reference.npz"
    if not p.exists():
        return None
    USED.add(p)
    d = np.load(p, allow_pickle=True)
    out = {"t": d["t"]}
    n = [k for k in d.files if k.startswith("scr_")][0].split("_")[1]
    for s, (_, b) in ALL.items():
        key = f"{b}_{n}"
        if key not in d.files:
            return None
        out[s] = d[key] - K
    return pd.DataFrame(out).set_index("t")


def t2v(t, y, target, decreasing=True):
    """First stored time at which y reaches target: metrics.time_to_value of the package."""
    i = np.where(y <= target)[0] if decreasing else np.where(y >= target)[0]
    return float(t[i[0]]) if len(i) else np.nan


def t_interp(t, y, target, decreasing=True):
    """The same crossing, interpolated between stored times (independent of the output step)."""
    i = np.where(y <= target)[0] if decreasing else np.where(y >= target)[0]
    if not len(i):
        return np.nan
    j = i[0]
    if j == 0:
        return float(t[0])
    return float(t[j - 1] + (target - y[j - 1]) * (t[j] - t[j - 1]) / (y[j] - y[j - 1]))


PARAM = [(v, T0, Tb, Tc) for v in (1, 2, 3) for T0 in (283.15, 298.15, 308.15)
         for Tb in (308.15, 310.15, 312.15) for Tc in (277.15, 281.15, 285.15)]


def pcase(v, T0, Tb, Tc):
    return f"v{v}_T0_{T0:.2f}_Tb_{Tb:.2f}_Tc_{Tc:.2f}"


def sources():
    for name, get in (("python", py), ("comsol", ref)):
        yield name, get


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SIT = [s for s in SITES]
    eps, kin, sup, des, vals = [], [], [], [], []
    runs = {}
    for src, get in sources():
        P = {}
        for c in PARAM:
            P[c] = get("parametric_sweep", pcase(*c))
        runs[src] = P
        base = {v: P[(v, 298.15, 310.15, 277.15)] for v in (1, 2, 3)}
        # effectiveness of single runs
        def add(study, v, material, silicone, s):
            if s is None:
                return
            Tr = float(s["SCR"].iloc[-1])
            for site in SIT:
                y = s[site]
                eps.append(dict(source=src, study=study, variant=f"V{v}", material=material, silicone=silicone,
                                site=site, **{"t_end (min)": float(y.index[-1]), "T_ring_final (C)": Tr,
                                              "T_initial (C)": float(y.iloc[0]), "T_final (C)": float(y.iloc[-1]),
                                              "epsilon": (Tb0 - y.iloc[-1]) / (Tb0 - Tr)}))
        for v in (1, 2, 3):
            add("Parametric baseline", v, "Aluminum", "Baseline", base[v])
            for m, name in ((1, "Aluminum"), (2, "Ceramic"), (3, "Glass")):
                add("Material", v, name, "Baseline", get("material_sweep", f"v{v}_material{m}"))
            for t, name in (("0.5mm", "0.5 mm"), ("1mm", "1.0 mm")):
                c = f"{t}_v{v}"
                add("Silicone", v, "Aluminum", name, get("silicone", c) if (RES / "silicone" / c / "probes_cooling.csv").exists() else None)
        # superposition over the 27 points
        for v in (1, 2, 3):
            cs = [c for c in PARAM if c[0] == v]
            X = np.array([[c[1] - K, c[2] - K, c[3] - K, 1.0] for c in cs])
            for site in SIT:
                y = np.array([P[c][site].iloc[-1] for c in cs])
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                sup.append(dict(source=src, variant=f"V{v}", site=site, w_T0=b[0], w_Tb=b[1],
                                **{"epsilon (w_Tc)": b[2], "intercept (C)": b[3],
                                   "max_abs_residual (K)": float(np.abs(y - X @ b).max())}))
        # designs across the 27 points: excursion and final temperature
        for a, b_ in ((1, 2), (1, 3), (2, 3)):
            for site in SIT:
                dT, dF, ratio = [], [], []
                for c in PARAM:
                    if c[0] != a:
                        continue
                    A, B = P[c], P[(b_,) + c[1:]]
                    da = A[site].iloc[0] - A[site].iloc[-1]
                    db = B[site].iloc[0] - B[site].iloc[-1]
                    dT.append(da - db)
                    dF.append(A[site].iloc[-1] - B[site].iloc[-1])
                    ratio.append(db / da if da else np.nan)
                des.append(dict(source=src, A=f"V{a}", B=f"V{b_}", site=site,
                                exc_diff_min=min(dT), exc_diff_max=max(dT), final_diff_min=min(dF), final_diff_max=max(dF),
                                ratio_min=np.nanmin(ratio), ratio_max=np.nanmax(ratio)))
        # kinetics
        for c in PARAM:
            s = P[c]
            for site in SIT:
                t, y = s.index.values, s[site].values
                d = y[0] - y[-1]
                kin.append(dict(source=src, variant=f"V{c[0]}", T0=c[1] - K, Tb=c[2] - K, Tc=c[3] - K, site=site,
                                t90_first_stored=t2v(t, y, y[0] - .9 * d, d > 0),
                                t90_interpolated=t_interp(t, y, y[0] - .9 * d, d > 0)))
    pd.DataFrame(eps).to_csv(OUT / "run_epsilon.csv", index=False)
    sf = pd.DataFrame(sup)
    sf.to_csv(OUT / "superposition_fit.csv", index=False)
    pd.DataFrame(des).to_csv(OUT / "design_27points.csv", index=False)
    kd = pd.DataFrame(kin)
    kd.to_csv(OUT / "kinetics.csv", index=False)
    kd.groupby(["source", "variant", "site"]).agg(
        t90_median_stored=("t90_first_stored", "median"), t90_median_interp=("t90_interpolated", "median"),
        t90_min=("t90_interpolated", "min"), t90_max=("t90_interpolated", "max")).reset_index().to_csv(OUT / "kinetics_summary.csv", index=False)

    # ring temperature needed
    rows = []
    for _, r in sf.iterrows():
        e = r["epsilon (w_Tc)"]
        for Tb in (35., 37., 39.):
            for tgt in (35., 32.):
                rows.append(dict(source=r.source, variant=r.variant, site=r.site, epsilon=e, **{
                    "Tb (C)": Tb, "target (C)": tgt, "required_ring_T (C)": Tb - (Tb - tgt) / e,
                    "T_site_at_4C_ring (C)": Tb - e * (Tb - 4.0)}))
    pd.DataFrame(rows).to_csv(OUT / "required_ring_temperature.csv", index=False)

    # rewarming: time 0 = switch-off. reheat/ holds 22 cases (the set of the model file)
    rows = []
    for d in sorted((RES / "reheat").iterdir()):
        if not d.is_dir():
            continue
        v = int(d.name[1])
        Tc = float(d.name.split("_Tc_")[1]) - K
        Tb = float(d.name.split("_Tb_")[1].split("_")[0]) - K
        s = py("reheat", d.name, "reheat")
        for site in SIT:
            y = s[site].values
            t = s.index.values
            above = bool(y[0] >= 35)
            rows.append(dict(source="python", variant=f"V{v}", Tb=Tb, Tc=Tc, site=site, T_off=y[0], T_end=y[-1],
                             never_below=above, minutes_first_stored=np.nan if above else t2v(t, y, 35., False),
                             minutes_interp=np.nan if above else t_interp(t, y, 35., False)))
        p = d / "comsol_reference.npz"
    pd.DataFrame(rows).to_csv(OUT / "reheat_persistence.csv", index=False)

    # time courses (same selection as effectiveness.timeseries)
    rows = []
    chans = ["SCR"] + SIT

    def ts(study, v, material, Tb, Tc, s, rid):
        for ch in chans:
            rows.append(pd.DataFrame({"study": study, "variant": f"V{v}", "material": material, "Tb (C)": Tb, "Tc (C)": Tc,
                                      "run_id": rid, "time (min)": s.index.values, "channel": ch, "T (C)": s[ch].values}))
    for v in (1, 2, 3):
        ts("Cooling, baseline point", v, "Aluminum", 37., 4., runs["python"][(v, 298.15, 310.15, 277.15)], pcase(v, 298.15, 310.15, 277.15))
    for m, name in ((1, "Aluminum"), (2, "Ceramic"), (3, "Glass")):
        ts("Material, V1", 1, name, np.nan, np.nan, py("material_sweep", f"v1_material{m}"), f"v1_material{m}")
    for Tc in (277.15, 281.15, 285.15):
        c = pcase(1, 298.15, 310.15, Tc)
        if (RES / "reheat" / c).exists():
            ts("Re-heat, V1", 1, "Aluminum", 37., Tc - K, py("reheat", c, "reheat"), c)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "timeseries_thesis.csv", index=False)

    # model against the ex-vivo measurements: experimental columns unchanged from the package table
    src_c = OUT / "comsol_model_vs_exvivo_reference.csv"
    mv = pd.read_csv(src_c)
    USED.add(src_c)
    OM = {"Human Fresh Aluminum Original": ("parametric_sweep", pcase(1, 298.15, 310.15, 277.15)),
          "Human Fresh Aluminum Reduced Bulk": ("parametric_sweep", pcase(2, 298.15, 310.15, 277.15)),
          "Human Fresh Ceramic Original": ("material_sweep", "v1_material2")}
    for i, r in mv.iterrows():
        s = py(*OM[r.group])
        ring = float(s["SCR"].iloc[-1])
        t, v = s.index.values, s[r.site].values
        d = v[0] - v[-1]
        mv.loc[i, "model_run"] = OM[r.group][1]
        mv.loc[i, "T_final model (C)"] = v[-1]
        mv.loc[i, "eps_bath model"] = (37 - v[-1]) / (37 - ring)
        mv.loc[i, "eps ratio exp/model"] = r["eps_bath exp"] / mv.loc[i, "eps_bath model"]
        mv.loc[i, "eps_start model"] = d / (v[0] - ring)
        mv.loc[i, "eps_start ratio exp/model"] = r["eps_start exp"] / mv.loc[i, "eps_start model"]
        mv.loc[i, "T_initial model (C)"] = v[0]
        mv.loc[i, "dT model (C)"] = d
        # interpolated between stored times: the program stores every 6 to 30 s, COMSOL's exports every minute
        mv.loc[i, "t50 model (min)"] = t_interp(t, v, v[0] - .5 * d)
        mv.loc[i, "t90 model (min)"] = t_interp(t, v, v[0] - .9 * d)
        mv.loc[i, "model within exp 95%CI"] = bool(r["eps_bath exp 95%CI low"] <= mv.loc[i, "eps_bath model"] <= r["eps_bath exp 95%CI high"])
        mv.loc[i, "model within exp 95%CI (eps_start)"] = bool(r["eps_start exp 95%CI low"] <= mv.loc[i, "eps_start model"] <= r["eps_start exp 95%CI high"])
    mv.to_csv(OUT / "model_vs_exvivo.csv", index=False)

    # paired ring differences, model against the experimental interval
    MAP = {"SCR": "SCR", "Anterior Sclera": "Anterior Sclera", "Equatorial Sclera": "Equatorial Sclera", "Vitreous": "Vitreous Humor",
           "Posterior Sclera": "Posterior Sclera", "Optic Nerve": "Optic Nerve Head"}
    a = py("material_sweep", "v1_material1"); c = py("material_sweep", "v1_material2")
    p1 = runs["python"][(1, 298.15, 310.15, 277.15)]; p2 = runs["python"][(2, 298.15, 310.15, 277.15)]
    for fn, name, A, B in (("Human__aluminum_original__vs__Human__ceramic_original__channels.csv", "material_vs_exvivo.csv", a, c),
                           ("Human__aluminum_original__vs__Human__aluminum_reduced_bulk__channels.csv", "geometry_vs_exvivo.csv", p1, p2)):
        e = pd.read_csv(EXV / fn)
        USED.add(EXV / fn)
        rows = []
        for _, x in e.iterrows():
            s = MAP.get(x.channel)
            if s is None or s == "SCR":
                continue
            d = (A[s].iloc[0] - A[s].iloc[-1]) - (B[s].iloc[0] - B[s].iloc[-1])
            rows.append(dict(site=s, model_diff=d, exp_diff=x.diff_C, ci90_low=x.ci90_low, ci90_high=x.ci90_high,
                             inside=bool(x.ci90_low <= d <= x.ci90_high), same_sign=bool(np.sign(d) == np.sign(x.diff_C))))
        pd.DataFrame(rows).to_csv(OUT / name, index=False)

    # mesh steps: coarser and coarse = mesh_study runs (at 35 min), finest = parametric baseline run
    rows = []
    for src, get in sources():
        for v in (1, 2, 3):
            c0, c1 = get("mesh_study", f"coarser_v{v}"), get("mesh_study", f"coarse_v{v}")
            f = runs[src][(v, 298.15, 310.15, 277.15)]
            for site in SIT + ["SCR"]:
                a0 = float(np.interp(35, c0.index, c0[site])); a1 = float(np.interp(35, c1.index, c1[site]))
                rows.append(dict(source=src, variant=f"V{v}", site=site, T_coarser=a0, T_coarse=a1, T_finest=float(f[site].iloc[-1]),
                                 step_a=a1 - a0, step_b=float(f[site].iloc[-1]) - a1))
    pd.DataFrame(rows).to_csv(OUT / "mesh_steps.csv", index=False)

    # plane of symmetry and retina line: COMSOL's nodes, the program's temperatures at those nodes
    fields = {}
    for v in ("v1", "v2", "v3"):
        s = RES / "ring_comparison" / v / "surface_T.npz"
        USED.add(s)
        d = np.load(s)
        idx = {int(n): i for i, n in enumerate(d["nodes"])}
        it = int(np.argmin(np.abs(d["t_min"] - 35.0)))
        assert abs(d["t_min"][it] - 35.0) < 1e-6
        Tn = d["T_K"][it] - K
        p = RES / "comsol_plane" / f"plane_{v}_35min.csv"
        USED.add(p)
        pl = pd.read_csv(p)
        T = np.array([Tn[idx[int(n)]] for n in pl.node], float)
        ntri = len(pl) // 3
        tris = np.arange(3 * ntri).reshape(-1, 3)
        dom = pl.domain.values[::3]
        # Domain 6 is the module layer, whose insulated faces carry two values on one node; the surface file
        # of the program keeps one of them, so the layer cannot be redrawn from it and is left out. It is not a result.
        keep = dom != 6
        tris, dom = tris[keep], dom[keep]
        fields.update({f"{v}_x": pl.x_mm.values, f"{v}_z": pl.z_mm.values, f"{v}_tris": tris, f"{v}_T": T, f"{v}_dom": dom})
        dif = pd.Series(T - pl.T_comsol_C.values).groupby(pl.domain.values).agg(lambda x: float(np.abs(x).max()))
        print("plane", v, "largest difference from COMSOL by domain:", dif.round(3).to_dict())
        src = RES / "comsol_retina" / f"retina_profile_{v}.csv"
        USED.add(src)
        rp = pd.read_csv(src)
        rp = rp[[c for c in rp.columns if not c.startswith("T_replica")]]
        for tm in (0, 15, 35):
            i = int(np.argmin(np.abs(d["t_min"] - tm)))
            assert abs(d["t_min"][i] - tm) < 1e-6
            rp[f"T_python_{tm}min_C"] = [float(d["T_K"][i][idx[int(n)]]) - K for n in rp.node]
            print("retina", v, tm, "min, largest difference from COMSOL",
                  round(float((rp[f"T_python_{tm}min_C"] - rp[f"T_comsol_{tm}min_C"]).abs().max()), 4))
        rp.to_csv(OUT / f"retina_profile_{v}.csv", index=False)
    np.savez_compressed(OUT / "plane_fields_35min.npz", **fields)

    # agreement over the validation set, from the stored comparison tables
    c = pd.read_csv(RES / "comparison_all.csv")
    r = pd.read_csv(RES / "reheat_release_2026-10-22" / "comparison_all.csv")
    USED.update({RES / "comparison_all.csv", RES / "reheat_release_2026-10-22" / "comparison_all.csv"})
    rows = []
    cool = c[c.study != "reheat"]
    for g, d in list(cool.groupby("study")) + [("ALL cooling", cool)] + [("reheat, 6 s release step", c[c.study == "reheat"]), ("reheat, release steps", r)]:
        rows.append(dict(set=g, cases=d[["study", "case"]].drop_duplicates().shape[0] if "study" in d else d.case.nunique(), probes=len(d),
                         worst_any_time_C=d.max_abs_diff_K.max(), worst_at_end_C=d.diff_end_K.abs().max()))
    pd.DataFrame(rows).to_csv(OUT / "agreement.csv", index=False)

    man = {"script": "code/figures/aim2_python_results.py", "results_folder": str(RES),
           "inputs_sha256": {str(p.relative_to(RES)) if RES in p.parents else p.name: sha256(p) for p in sorted(USED)}}
    (OUT / "inputs_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("wrote", OUT, len(USED), "inputs")


if __name__ == "__main__":
    main()
