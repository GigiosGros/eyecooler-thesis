"""Independent recomputation from the raw CSVs — no package / notebook code.

    python tests/independent_check.py [Model_data folder] [results folder]

Checks the results of the DEFAULT configuration (corrections as set in config/config.toml):
the interior-boundary halving of the SCR interface energies and the cap of the GCI order
at the formal order are applied here too, each in one line of its own."""
import numpy as np, pandas as pd, math, re, os, sys, tomllib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "Model_data")
R = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results")
COR = tomllib.load(open(os.path.join(ROOT, "config", "config.toml"), "rb"))["corrections"]
P_FORMAL = tomllib.load(open(os.path.join(ROOT, "config", "config.toml"), "rb"))["mesh"].get("p_formal", 2.0)
raw = lambda f: pd.read_csv(os.path.join(D, f), skiprows=4)
res = lambda *p: pd.read_csv(os.path.join(R, *p))
out = []
def rep(name, mine, pkg, tol=1e-6):
    ok = abs(mine - pkg) <= tol * max(1, abs(pkg)); out.append(ok)
    print(f"{'OK ' if ok else 'XX '} {name:60s} independent={mine:.6g}  package={pkg:.6g}")

# (a) SCR energy, baseline, per interface/variant: 2 * ∫|P| dt from the raw W columns
# the ring energy table: the corrected one (from COMSOL stored boundary fluxes) when the package is set to use it
_en = "Ring Comparison Energies and Thermal Resistance Data Normal Mesh.csv"
_alt = os.path.join(D, "stored_flux_energies", _en)
E = pd.read_csv(_alt, skiprows=4) if COR.get("use_stored_flux_energies", False) and os.path.exists(_alt) else raw(_en)
t = E.iloc[:, 0].values * 60
pe = res("03_energy", "scr_energy.csv")
for v in (1, 2, 3):
    for iface in ("Eye", "Eyelids", "Air"):
        col = [c for c in E.columns if f"SCR {iface} v{v} (W)" in c and "W/m" not in c][0]
        # x2 for the half-symmetry model; /2 because COMSOL counts the flux on an interior boundary twice
        mine = (1 if COR["interface_flux_interior_double"] else 2) * np.trapezoid(np.abs(E[col].values), t)
        pkg = pe[(pe.variant == f"V{v}") & (pe.silicone == "Baseline") & (pe.interface == iface)].energy_J.iloc[0]
        rep(f"SCR energy V{v} {iface} (J)", mine, pkg)
# (b) SCR R_th: mean over last 20 % of samples of (Th-Tc)/|q|
rt = res("04_thermal_resistance", "scr_rth.csv").set_index("Variant")
for v in (1, 2, 3):
    th = E[[c for c in E.columns if f"T_hot v{v}" in c][0]].values
    tc = E[[c for c in E.columns if f"T_cold v{v}" in c][0]].values
    q = E[[c for c in E.columns if f"q_TEC v{v}" in c][0]].values
    Rt = (th - tc) / np.abs(q); n = int(len(Rt) * 0.2)
    tf = 2.0 if COR.get("tec_flux_interior_average", False) else 1.0      # exported q_TEC is half the face's heat
    rep(f"SCR R_th V{v} half model (K/W)", Rt[-n:].mean() / tf, rt.loc[f"V{v}", "R_steady_half_model (K/W)"])
# (c) WB closure: Fine Energies vs Aux Fine Sweep at alpha=1, water props interpolated at Tin
wc = res("03_energy", "wb_closure.csv").set_index("Variant")
T_C, rho_t, cp_t = [4, 10, 25], [1000.0, 999.7, 997.0], [4204, 4192, 4180]
for v in (1, 2, 3):
    fe = raw(f"Fine Energies v{v}.csv"); fe.columns = [c.lstrip("% ").strip() for c in fe.columns]
    hot = [c for c in fe.columns if c.startswith("alpha=1,") and c.endswith("(W)") and "wet" not in c.lower()][0]
    lines = open(os.path.join(D, f"V{v}_Auxiliary_Fine_Sweep.csv")).read().splitlines()
    hdr_i = max(i for i, l in enumerate(lines) if l.startswith("%"))
    A = pd.read_csv(os.path.join(D, f"V{v}_Auxiliary_Fine_Sweep.csv"), skiprows=hdr_i)
    A.columns = [c.lstrip("% ").strip() for c in A.columns]
    A = A[A["alpha"] == 1.0]
    inw = [c for c in A.columns if "INw" in c][0]; outw = [c for c in A.columns if "OUTw" in c][0]
    Tin = A["Tin (K)"] - 273.15
    A = A.assign(P=np.interp(Tin, T_C, rho_t) * A["Q0 (m^3/s)"] * np.interp(Tin, T_C, cp_t) * (A[outw] - A[inw]))
    keys = ["Q0 (m^3/s)", "Tin (K)", "T0 (K)", "Qh (W/m^2)"]
    m = fe[keys + [hot]].merge(A[keys + ["P"]], on=keys)
    rep(f"WB closure V{v} ratio (mean/mean)", m.P.mean() / m[hot].abs().mean(), wc.loc[f"V{v}", "ratio"])
# (d) GCI: textbook Celik (2008) procedure, fixed-point on p
def celik(f1, f2, f3, N1, N2, N3, Fs=1.25):
    r21, r32 = (N1 / N2) ** (1 / 3), (N2 / N3) ** (1 / 3)
    e21, e32 = f2 - f1, f3 - f2
    s = np.sign(e32 / e21); p = 2.0
    for _ in range(200):
        p = abs(math.log(abs(e32 / e21)) + math.log((r21 ** p - s) / (r32 ** p - s))) / math.log(r21)
    p_used = min(p, P_FORMAL) if COR["gci_cap_p_formal"] else p      # conservative: never above the formal order
    return 100 * Fs * abs((f1 - f2) / f1) / (r21 ** p_used - 1), p
g = res("02_mesh_convergence", "gci.csv")
dof = {"V1": (179228, 39952, 16873), "V2": (176051, 38494, 16047), "V3": (185132, 39063, 16175)}
for v in ("V1", "V2", "V3"):
    for ch in ("Optic Nerve Head", "Vitreous Humor", "Posterior Sclera"):
        f = {}
        for mesh in ("Coarser", "Coarse", "Normal"):
            P = raw(f"Ring Comparison Probe (Stationary Start) {mesh}.csv")
            col = [c for c in P.columns if re.match(rf"^{ch} {v[1]} \(K\)", c)][0]
            f[mesh] = P.loc[P.iloc[:, 0] <= 35.0 + 1e-9, col].iloc[-1] - 273.15
        r = g[(g.family == "SCR") & (g.variant == v) & (g.channel == ch)].iloc[0]
        if str(r.get("band_method", "GCI")).startswith("GCI"):
            pct, p = celik(f["Normal"], f["Coarse"], f["Coarser"], *dof[v])
            rep(f"GCI {v} {ch} (%)", round(pct, 4), r["GCI (%)"], 1e-3)
        else:   # oscillatory or diverging sequence: the band is the spread of the three meshes (or larger)
            spread = max(f.values()) - min(f.values())
            ok = r["U_mesh (K)"] >= spread - 1e-9; out.append(ok)
            print(f"{'OK ' if ok else 'XX '} {'band ' + v + ' ' + ch + ' >= 3-mesh spread (C)':60s} spread={spread:.6g}  package={r['U_mesh (K)']:.6g}")
# (e) parametric baseline run V1 = mesh-study Normal run (same operating point)? + t50 on ONH
Pm = raw("SCR v1 Parametric Sweep.csv")
row = Pm[(Pm.iloc[:, 0] == 298.15) & (Pm.iloc[:, 1] == 310.15) & (Pm.iloc[:, 2] == 277.15)].iloc[0]
cols = [(float(re.match(r"Time=([\d.]+) min", c).group(1)), c) for c in Pm.columns if c.startswith("Time=") and re.search(r", Optic Nerve Head 1 \(K\)", c)]
tt = np.array([a for a, _ in cols]); yy = np.array([row[c] for _, c in cols]) - 273.15
o = np.argsort(tt); tt, yy = tt[o], yy[o]; msk = tt <= 35; tt, yy = tt[msk], yy[msk]
mm = res("06_metrics", "master_metrics.csv")
mr = mm[(mm.study == "Parametric Sweep") & (mm.variant == "V1") & (mm.probe == "Optic Nerve Head")
        & np.isclose(mm["T0_input (C)"], 25) & np.isclose(mm["Tb_input (C)"], 37) & np.isclose(mm["Tc_input (C)"], 4)].iloc[0]
rep("param V1 ONH T_final (C)", yy[-1], mr["Tf_simulated (C)"])
rep("param V1 ONH dT (K)", yy[0] - yy[-1], mr["dT (K)"])
target = yy[0] - 0.5 * (yy[0] - yy[-1]); rep("param V1 ONH t50 (min)", tt[np.argmax(yy <= target)], mr["t50 (min)"])
print("   (mesh-study Normal ONH V1 final = 33.6887 °C; parametric baseline final =", round(yy[-1], 4), ")")
# (f) Reynolds, V3 inlet pipe at 8.0e-6 m^3/s, 25 °C — Reynolds report quotes 8090
rn = res("05_dimensionless", "reynolds_nominal.csv")
v_ = 8.0e-6 / (math.pi / 4 * (1.41e-3) ** 2); rep("Re V3 inlet pipe (report 8090)", v_ * 1.41e-3 / 0.893e-6,
    rn[(rn.Variant == "V3") & (rn.Region == "Inlet pipe")].Re.iloc[0])
print(f"\n{sum(out)}/{len(out)} independent checks agree")
# non-zero exit status if any check disagrees
sys.exit(0 if all(out) else 1)
