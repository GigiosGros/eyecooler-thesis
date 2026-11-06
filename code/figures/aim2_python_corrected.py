"""Chapter 3: result tables of the eye model from the corrected runs of the Python program.

The validation tables (agreement with COMSOL, mesh steps) stay in aim2_python_results.py, which reads
the program's runs of the same cases as COMSOL. This script reads the corrected cases, on COMSOL's
production meshes (scr-heat-replica, tables of 2026-10-13, original/new_runs):
  B_v{1,2,3}_{Al,alumina,silica}  baseline, ring material
  C_v{1,2,3}_silicone{0.5,1}mm    insulating layer
  D27_v*_Tb*_T0*_Tc*              27 operating points per ring
  RW_B_*, RW_D27_*                the same runs with rewarming (short steps after the release)
The eye starts at rest with the device in place and switched off, there is no air domain (surface
coefficients instead), and the properties are those of the property workbook. Heat comes from
results/energy_balance_2026-10-22/heat_split.csv.

Two steps, so that the tables can be rebuilt without the 1.2 GB archive:
  python code/figures/aim2_python_corrected.py --store   (needs SCR_RUNS, SCR_FIELDS, SCR_RESULTS)
      copies the probes, the plane of symmetry and the retina line into data/python/corrected/
  python code/figures/aim2_python_corrected.py
      writes the tables from that store. No solve is started and no value is typed in.

SCR_RUNS   folder new_runs, extracted from Lakeshore cluster runs (2026-10-21)/tables_2026-10-13.tar.gz
SCR_FIELDS Lakeshore tables snapshot (2026-10-13)/fields/original_mesh_baseline_cases
SCR_RESULTS scr-heat-replica/results
"""
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

from provenance import SOURCES, sha256

PY = SOURCES / "data" / "python"
OUT = PY / "corrected"
EXV = SOURCES / "data" / "exvivo"
STORE = OUT / "run_store.npz"
K = 273.15
Tb0 = 37.0
SITES = {"Optic Nerve Head": "Optic Nerve Head Probe 1 (K)", "Vitreous Humor": "Vitreous Humor Probe 1 (K)",
         "Anterior Sclera": "Anteriror Sclera Probe 1 (K)", "Posterior Sclera": "Posterior Sclera Probe 1 (K)",
         "Equatorial Sclera": "Equatorial Sclera Probe 1 (K)", "Optic Nerve": "Optic Nerve Probe 1 (K)",
         "Retina Avg": "Domain Probe Retina Avg 1 (K)", "Retina Max": "Domain Probe Retina Max 1 (K)",
         "Retina Min": "Domain Probe Retina Min 1 (K)", "Eye Avg": "Domain Probe Eye Avg 1 (K)",
         "Eye Max": "Domain Probe Eye Max 1 (K)", "Eye Min": "Domain Probe Eye Min 1 (K)"}
ALL = {**SITES, "SCR": "SCR 1 (K)", "Ring average": "Ring average (K)"}
SIT = list(SITES)
POINTS = [(v, T0, Tb, Tc) for v in (1, 2, 3) for T0 in (10, 25, 35) for Tb in (35, 37, 39) for Tc in (4, 8, 12)]


def d27(v, T0, Tb, Tc):
    return f"D27_v{v}_Tb{Tb}_T{T0:03d}_Tc{Tc}"


def case_list():
    c = [d27(*p) for p in POINTS]
    c += ["RW_" + x for x in c]
    c += [f"B_v{v}_{m}" for v in (1, 2, 3) for m in ("Al", "alumina", "silica")]
    c += [f"RW_B_v{v}_Al" for v in (1, 2, 3)]
    c += [f"C_v{v}_silicone{t}mm" for v in (1, 2, 3) for t in ("0.5", "1")]
    return c


def build_store():
    runs = pathlib.Path(os.environ["SCR_RUNS"])
    fields = pathlib.Path(os.environ["SCR_FIELDS"])
    res = pathlib.Path(os.environ["SCR_RESULTS"])
    OUT.mkdir(parents=True, exist_ok=True)
    data, used = {}, {}
    names = list(ALL)
    for c in case_list():
        for which in ("cooling", "rewarm"):
            p = runs / c / f"probes_{which}.csv"
            if which == "rewarm" and not c.startswith("RW_"):
                continue
            d = pd.read_csv(p)
            data[f"{c}|{which}|t"] = d.iloc[:, 0].to_numpy(float)
            d.columns = [" ".join(("1" if w.isdigit() else w) for w in x.split(" ")) if " Probe " in x or x.startswith("SCR ") else x for x in d.columns]
            data[f"{c}|{which}|T"] = np.array([d[ALL[n]].to_numpy(float) - K for n in names], float).T
            used[f"{c}/{p.name}"] = sha256(p)
    np.savez_compressed(STORE, names=np.array(names), **data)

    # plane of symmetry and retina line at COMSOL's own output points: nodes matched by position
    fld = {}
    for v in ("v1", "v2", "v3"):
        m = np.load(fields / f"B_{v}_Al" / "mesh.npz", allow_pickle=True)
        f = np.load(fields / f"B_{v}_Al" / "fields.npz")
        P = m["points"] * (1e3 if np.abs(m["points"]).max() < 1 else 1.0)
        on = np.where(np.abs(P[:, 1]) < 1e-6)[0]
        from scipy.spatial import cKDTree
        tree = cKDTree(P[on][:, [0, 2]])
        tc = f["time_cool_s"] / 60.0
        idx = {tm: int(np.argmin(np.abs(tc - tm))) for tm in (15, 35)}
        assert all(abs(tc[i] - tm) < 1e-6 for tm, i in idx.items())
        pl = pd.read_csv(res / "comsol_plane" / f"plane_{v}_35min.csv")
        dist, j = tree.query(np.c_[pl.x_mm, pl.z_mm])
        T = f["T_cool"][idx[35]][on[j]] - (K if f["T_cool"].max() > 200 else 0)
        tri = np.arange(len(pl)).reshape(-1, 3)
        dom = pl.domain.values[::3]
        ok = (dist[tri] < 1e-3).all(axis=1) & (dom != 6)
        print(v, "plane points matched", int((dist < 1e-3).sum()), "of", len(pl), "domains dropped",
              sorted(set(dom[~ok].tolist())))
        fld.update({f"{v}_x": pl.x_mm.values, f"{v}_z": pl.z_mm.values, f"{v}_tris": tri[ok], f"{v}_T": T, f"{v}_dom": dom[ok]})
        rp = pd.read_csv(res / "comsol_retina" / f"retina_profile_{v}.csv")
        rp = rp[["node", "x_mm", "z_mm", "arc_mm"]].copy()
        dist, j = tree.query(np.c_[rp.x_mm, rp.z_mm])
        assert dist.max() < 1e-3, dist.max()
        sub = K if f["T_cool"].max() > 200 else 0
        rp["T_python_0min_C"] = f["T_start"][on[j]] - sub
        for tm, i in idx.items():
            rp[f"T_python_{tm}min_C"] = f["T_cool"][i][on[j]] - sub
        rp.to_csv(OUT / f"retina_profile_{v}.csv", index=False)
        used[f"fields/B_{v}_Al/fields.npz"] = sha256(fields / f"B_{v}_Al" / "fields.npz")
    np.savez_compressed(OUT / "plane_fields_35min.npz", **fld)
    shutil.copy2(res / "energy_balance_2026-10-22" / "heat_split.csv", OUT / "heat_split.csv")
    used["energy_balance_2026-10-22/heat_split.csv"] = sha256(OUT / "heat_split.csv")
    (OUT / "inputs_manifest.json").write_text(json.dumps(
        {"script": "code/figures/aim2_python_corrected.py", "runs": str(runs), "fields": str(fields),
         "archive": "Computational model/Simulation results/Lakeshore cluster runs (2026-10-21)/tables_2026-10-13.tar.gz",
         "inputs_sha256": used}, indent=1), encoding="utf-8")
    print("store written", len(used), "inputs")


def t_interp(t, y, target, decreasing=True):
    i = np.where(y <= target)[0] if decreasing else np.where(y >= target)[0]
    if not len(i):
        return np.nan
    j = i[0]
    if j == 0:
        return float(t[0])
    return float(t[j - 1] + (target - y[j - 1]) * (t[j] - t[j - 1]) / (y[j] - y[j - 1]))


def main():
    st = np.load(STORE)
    names = list(st["names"])

    def run(c, which="cooling"):
        return pd.DataFrame(st[f"{c}|{which}|T"], index=st[f"{c}|{which}|t"], columns=names)

    P = {p: run(d27(*p)) for p in POINTS}
    B = {v: run(f"B_v{v}_Al") for v in (1, 2, 3)}
    for v in (1, 2, 3):
        print("B against D27 at the baseline, largest end difference",
              round(float((B[v].iloc[-1] - P[(v, 25, 37, 4)].iloc[-1]).abs().max()), 5))

    eps = []

    def add(study, v, material, silicone, s):
        Tr = float(s["SCR"].iloc[-1])
        for site in SIT:
            y = s[site]
            eps.append(dict(source="python", study=study, variant=f"V{v}", material=material, silicone=silicone, site=site,
                            **{"t_end (min)": float(y.index[-1]), "T_ring_final (C)": Tr, "T_ring_average_final (C)": float(s["Ring average"].iloc[-1]),
                               "T_initial (C)": float(y.iloc[0]), "T_final (C)": float(y.iloc[-1]), "epsilon": (Tb0 - y.iloc[-1]) / (Tb0 - Tr)}))
    for v in (1, 2, 3):
        add("Parametric baseline", v, "Aluminum", "Baseline", B[v])
        for m, name in (("Al", "Aluminum"), ("alumina", "Ceramic"), ("silica", "Glass")):
            add("Material", v, name, "Baseline", run(f"B_v{v}_{m}"))
        for t, name in (("0.5", "0.5 mm"), ("1", "1.0 mm")):
            add("Silicone", v, "Aluminum", name, run(f"C_v{v}_silicone{t}mm"))
    pd.DataFrame(eps).to_csv(OUT / "run_epsilon.csv", index=False)

    sup, des, kin = [], [], []
    for v in (1, 2, 3):
        cs = [c for c in POINTS if c[0] == v]
        X = np.array([[c[1], c[2], c[3], 1.0] for c in cs], float)
        for site in SIT:
            y = np.array([P[c][site].iloc[-1] for c in cs])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            sup.append(dict(source="python", variant=f"V{v}", site=site, w_T0=b[0], w_Tb=b[1],
                            **{"epsilon (w_Tc)": b[2], "intercept (C)": b[3], "max_abs_residual (K)": float(np.abs(y - X @ b).max())}))
    sf = pd.DataFrame(sup)
    sf.to_csv(OUT / "superposition_fit.csv", index=False)
    for a, b_ in ((1, 2), (1, 3), (2, 3)):
        for site in SIT:
            dT, dF, ratio = [], [], []
            for c in POINTS:
                if c[0] != a:
                    continue
                A, Bb = P[c], P[(b_,) + c[1:]]
                da, db = A[site].iloc[0] - A[site].iloc[-1], Bb[site].iloc[0] - Bb[site].iloc[-1]
                dT.append(da - db); dF.append(A[site].iloc[-1] - Bb[site].iloc[-1]); ratio.append(db / da if da else np.nan)
            des.append(dict(source="python", A=f"V{a}", B=f"V{b_}", site=site, exc_diff_min=min(dT), exc_diff_max=max(dT),
                            final_diff_min=min(dF), final_diff_max=max(dF), ratio_min=np.nanmin(ratio), ratio_max=np.nanmax(ratio)))
    pd.DataFrame(des).to_csv(OUT / "design_27points.csv", index=False)
    for c in POINTS:
        s = P[c]
        for site in SIT:
            t, y = s.index.values, s[site].values
            d = y[0] - y[-1]
            kin.append(dict(source="python", variant=f"V{c[0]}", T0=c[1], Tb=c[2], Tc=c[3], site=site,
                            t90_interpolated=t_interp(t, y, y[0] - .9 * d, d > 0)))
    kd = pd.DataFrame(kin)
    kd.to_csv(OUT / "kinetics.csv", index=False)
    kd.groupby(["source", "variant", "site"]).agg(t90_median_interp=("t90_interpolated", "median"), t90_min=("t90_interpolated", "min"),
                                                  t90_max=("t90_interpolated", "max")).reset_index().to_csv(OUT / "kinetics_summary.csv", index=False)
    kd.groupby("site").t90_interpolated.agg(["median", "min", "max"]).reset_index().to_csv(OUT / "kinetics_all_rings.csv", index=False)

    rows = []
    for _, r in sf.iterrows():
        e = r["epsilon (w_Tc)"]
        for Tb in (35., 37., 39.):
            for tgt in (35., 32.):
                rows.append(dict(source="python", variant=r.variant, site=r.site, epsilon=e,
                                 **{"Tb (C)": Tb, "target (C)": tgt, "required_ring_T (C)": Tb - (Tb - tgt) / e,
                                    "T_site_at_4C_ring (C)": Tb - e * (Tb - 4.0)}))
    pd.DataFrame(rows).to_csv(OUT / "required_ring_temperature.csv", index=False)

    # rewarming: minutes from the release to 35 C, ambient 25 C
    rows = []
    for v in (1, 2, 3):
        for Tb in (35, 37, 39):
            for Tc in (4, 8, 12):
                s = run("RW_" + d27(v, 25, Tb, Tc), "rewarm")
                t = s.index.values - s.index.values[0]
                for site in SIT:
                    y = s[site].values
                    above = bool(y[0] >= 35)
                    # cooling run of the same point: minutes from switch-on until the site first falls below 35 C
                    c = P[(v, 25, Tb, Tc)]
                    tc = c.index.values - c.index.values[0]
                    yc = c[site].values
                    rows.append(dict(source="python", variant=f"V{v}", Tb=Tb, Tc=Tc, site=site, T_off=y[0], T_end=y[-1], T_start_cooling=float(yc[0]),
                                     never_below=above, minutes_interp=np.nan if above else t_interp(t, y, 35., False),
                                     minutes_to_35_cooling=np.nan if yc.min() >= 35 else t_interp(tc, yc, 35., True)))
    pd.DataFrame(rows).to_csv(OUT / "reheat_persistence.csv", index=False)

    rows = []
    chans = ["SCR"] + SIT

    def ts(study, v, material, Tc, s, rid):
        for ch in chans:
            rows.append(pd.DataFrame({"study": study, "variant": f"V{v}", "material": material, "Tb (C)": 37., "Tc (C)": Tc, "run_id": rid,
                                      "time (min)": s.index.values, "channel": ch, "T (C)": s[ch].values}))
    for v in (1, 2, 3):
        ts("Cooling, baseline point", v, "Aluminum", 4., run(f"RW_B_v{v}_Al"), f"RW_B_v{v}_Al")
    for m, name in (("Al", "Aluminum"), ("alumina", "Ceramic"), ("silica", "Glass")):
        ts("Material, V1", 1, name, 4., run(f"B_v1_{m}"), f"B_v1_{m}")
    s = run("RW_B_v1_Al", "rewarm")
    s = s.set_axis(s.index.values - s.index.values[0], axis=0)
    ts("Re-heat, V1", 1, "Aluminum", 4., s, "RW_B_v1_Al")
    pd.concat(rows, ignore_index=True).to_csv(OUT / "timeseries_thesis.csv", index=False)

    mv = pd.read_csv(PY / "comsol_model_vs_exvivo_reference.csv")
    OM = {"Human Fresh Aluminum Original": "B_v1_Al", "Human Fresh Aluminum Reduced Bulk": "B_v2_Al", "Human Fresh Ceramic Original": "B_v1_alumina"}
    for i, r in mv.iterrows():
        s = run(OM[r.group])
        ring = float(s["SCR"].iloc[-1])
        t, v = s.index.values, s[r.site].values
        d = v[0] - v[-1]
        mv.loc[i, "model_run"] = OM[r.group]
        mv.loc[i, "T_final model (C)"] = v[-1]
        mv.loc[i, "eps_bath model"] = (37 - v[-1]) / (37 - ring)
        mv.loc[i, "eps ratio exp/model"] = r["eps_bath exp"] / mv.loc[i, "eps_bath model"]
        mv.loc[i, "eps_start model"] = d / (v[0] - ring)
        mv.loc[i, "eps_start ratio exp/model"] = r["eps_start exp"] / mv.loc[i, "eps_start model"]
        mv.loc[i, "T_initial model (C)"] = v[0]
        mv.loc[i, "dT model (C)"] = d
        mv.loc[i, "t50 model (min)"] = t_interp(t, v, v[0] - .5 * d)
        mv.loc[i, "t90 model (min)"] = t_interp(t, v, v[0] - .9 * d)
        mv.loc[i, "model within exp 95%CI (eps_start)"] = bool(r["eps_start exp 95%CI low"] <= mv.loc[i, "eps_start model"] <= r["eps_start exp 95%CI high"])
    mv.to_csv(OUT / "model_vs_exvivo.csv", index=False)

    MAP = {"Anterior Sclera": "Anterior Sclera", "Equatorial Sclera": "Equatorial Sclera", "Vitreous": "Vitreous Humor",
           "Posterior Sclera": "Posterior Sclera", "Optic Nerve": "Optic Nerve Head"}
    for fn, name, A, Bb in (("Human__aluminum_original__vs__Human__ceramic_original__channels.csv", "material_vs_exvivo.csv", B[1], run("B_v1_alumina")),
                            ("Human__aluminum_original__vs__Human__aluminum_reduced_bulk__channels.csv", "geometry_vs_exvivo.csv", B[1], B[2])):
        e = pd.read_csv(EXV / fn)
        rows = []
        for _, x in e.iterrows():
            s = MAP.get(x.channel)
            if s is None:
                continue
            d = (A[s].iloc[0] - A[s].iloc[-1]) - (Bb[s].iloc[0] - Bb[s].iloc[-1])
            rows.append(dict(site=s, model_diff=d, exp_diff=x.diff_C, ci90_low=x.ci90_low, ci90_high=x.ci90_high,
                             inside=bool(x.ci90_low <= d <= x.ci90_high), same_sign=bool(np.sign(d) == np.sign(x.diff_C))))
        pd.DataFrame(rows).to_csv(OUT / name, index=False)

    # heat over 35 min, whole device. Set "ring" for all cases: the layer counts as a neighbour of the ring,
    # and the heat it passes on comes from the eyelids and the air.
    h = pd.read_csv(OUT / "heat_split.csv")
    rows = []
    for v in (1, 2, 3):
        for lab, c in (("Baseline", f"B_v{v}_Al"), ("0.5 mm", f"C_v{v}_silicone0.5mm"), ("1.0 mm", f"C_v{v}_silicone1mm")):
            r = h[(h.case == c) & (h.set == "ring")].iloc[0]
            eye = r["eye (J)"] + r["tear_cornea (J)"]
            lid = r["eyelids_orbit (J)"] + r["silicone (J)"]
            air = r["air_convection (J)"] + r["radiation (J)"]
            tot = r["sum drawn from neighbours (J)"]
            rows.append(dict(variant=f"V{v}", silicone=lab, case=c, face_J=r["cooling face out (J)"], stored_J=r["release of heat stored in the set (J)"],
                             eye_J=eye, eyelids_J=lid, air_J=air, drawn_J=tot, eye_share_pct=100 * eye / tot, face_W_end=r["cooling face out (W)"]))
    e = pd.DataFrame(rows)
    e.to_csv(OUT / "energy_summary.csv", index=False)
    print(e.round(2).to_string())


if __name__ == "__main__":
    if "--store" in sys.argv:
        build_store()
    else:
        main()
