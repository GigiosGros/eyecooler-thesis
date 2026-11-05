"""Reduced steady-state heat-conduction model of the globe: ex-vivo bath vs perfused eye.

Axisymmetric (r, z) finite-volume model, no COMSOL needed. Purpose: test which
heat-resupply condition reproduces the measured ex-vivo cooling pattern, and
whether a perfused choroid explains why the COMSOL model cools less at depth.

Geometry (human, simplified): sphere R = 12 mm centred at z = -15 mm (anterior
pole z = -3), sclera 0.7 mm, choroid 0.25 mm, retina 0.25 mm, vitreous inside;
cornea zone theta < 28.6 deg (corneal radius 5.75 mm); optic nerve r < 1.5 mm,
13 mm long behind the posterior pole. Ring: 4 degC on the spherical band from
the limbus giving ~390 mm^2 of contact. Cornea: air, h_air to 21 degC.
Rest of the surface (and the nerve): either the 37 degC bath with coefficient
h_bath (ex vivo), or a 37 degC orbital boundary h_orbit (perfused case).
Choroid perfusion: Pennes term w*rho_b*c_b*(37 - T), w = 0.0927 1/s (as in the
COMSOL model). Outcome: eps = (37 - T_site)/(37 - T_ring) at the five sites.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, json, os, sys

D = 0.1e-3                       # grid step (m)
R, ZC = 12e-3, -15e-3
T_RING, T_HOT, T_AIR = 4.0, 37.0, 21.0
K = dict(sclera=0.50, choroid=0.53, retina=0.55, vitreous=0.60, cornea=0.56, nerve=0.50)
RHOC_B = 1068 * 3682
W_CHOROID = 0.0927
TH_CORNEA = np.arcsin(5.75 / 12)                      # 28.6 deg
H_BAND = 390e-6 / (2 * np.pi * R)                      # ring band height along z
TH_RING2 = np.arccos(np.cos(TH_CORNEA) - H_BAND / R)   # posterior edge of ring contact

nr, nz = int(14e-3 / D), int(43e-3 / D)
rc = (np.arange(nr) + .5) * D
zc = -(np.arange(nz) + .5) * D                        # z from -0.05 mm down to -43 mm
Rg, Zg = np.meshgrid(rc, zc, indexing="ij")
rho = np.hypot(Rg, Zg - ZC)
theta = np.arctan2(Rg, Zg - ZC)                        # 0 = anterior pole

globe = rho < R
nerve = (Rg < 1.5e-3) & (Zg < ZC - R + 1.0e-3) & (Zg > ZC - R - 13e-3)
inside = globe | nerve
mat = np.full(Rg.shape, "", dtype=object)
mat[inside] = "vitreous"
mat[nerve & ~globe] = "nerve"
mat[globe & (rho >= R - 0.7e-3)] = "sclera"
mat[globe & (rho >= R - 0.95e-3) & (rho < R - 0.7e-3) & (theta > np.radians(40))] = "choroid"
mat[globe & (rho >= R - 1.2e-3) & (rho < R - 0.95e-3) & (theta > np.radians(40))] = "retina"
mat[globe & (theta < TH_CORNEA) & (rho >= R - 0.55e-3)] = "cornea"
kmap = np.zeros(Rg.shape)
for m, k in K.items():
    kmap[mat == m] = k

idx = -np.ones(Rg.shape, int)
idx[inside] = np.arange(inside.sum())
N = inside.sum()


def site_cells():
    def at(th_deg, rad):
        th = np.radians(th_deg)
        r, z = rad * np.sin(th), ZC + rad * np.cos(th)
        return int(r / D), int(-z / D)
    mid = R - 0.35e-3
    return {"Anterior sclera": at(62.3, mid), "Equatorial sclera": at(90, mid),
            "Posterior sclera": at(122.2, mid), "Vitreous": (0, int(15e-3 / D)),
            "Optic nerve (head)": (int(1.86e-3 / D), int(24.6e-3 / D))}


SITES = site_cells()


def solve(h_out, w_choroid=0.0, h_air=10.0, ring_on=True):
    """Steady state. h_out: coefficient to 37 degC on the non-ring, non-corneal surface."""
    rows, cols, vals = [], [], []
    b = np.zeros(N)
    diag = np.zeros(N)
    for i in range(nr):
        for j in range(nz):
            p = idx[i, j]
            if p < 0:
                continue
            r = rc[i]
            vol = r * D * D
            # Pennes in choroid
            if mat[i, j] == "choroid" and w_choroid:
                g = w_choroid * RHOC_B * vol
                diag[p] += g; b[p] += g * T_HOT
            for di, dj, area in ((1, 0, (r + D / 2) * D), (-1, 0, (r - D / 2) * D), (0, 1, r * D), (0, -1, r * D)):
                ii, jj = i + di, j + dj
                if di == -1 and i == 0:
                    continue                      # axis: no flux
                if 0 <= ii < nr and 0 <= jj < nz and idx[ii, jj] >= 0:
                    k = 2 * kmap[i, j] * kmap[ii, jj] / (kmap[i, j] + kmap[ii, jj])
                    g = k * area / D
                    diag[p] += g; rows.append(p); cols.append(idx[ii, jj]); vals.append(-g)
                    continue
                # boundary face
                th = theta[i, j]
                gk = kmap[i, j] * area / (D / 2)
                if globe[i, j] and th < TH_CORNEA:
                    h, Tinf = h_air, T_AIR
                elif globe[i, j] and th <= TH_RING2 and ring_on:
                    h, Tinf = None, T_RING
                else:
                    h, Tinf = h_out, T_HOT
                g = gk if h is None else 1 / (1 / gk + 1 / (h * area))
                diag[p] += g; b[p] += g * Tinf
    A = sp.csr_matrix((vals + list(diag), (rows + list(range(N)), cols + list(range(N)))), shape=(N, N))
    T = spla.spsolve(A, b)
    full = np.full(Rg.shape, np.nan); full[inside] = T
    return full


def eps(full):
    return {s: (T_HOT - full[c]) / (T_HOT - T_RING) for s, c in SITES.items()}


if __name__ == "__main__":
    out = {}
    hs = [20, 40, 60, 80, 120, 200, 400, 1000]
    out["exvivo_no_perfusion"] = {h: eps(solve(h)) for h in hs}
    out["perfused_choroid"] = {h: eps(solve(h, W_CHOROID)) for h in hs}
    out["baseline_no_ring_exvivo"] = {h: {s: float(solve(h, ring_on=False)[c]) for s, c in SITES.items()} for h in (40, 80, 200)}
    out["baseline_no_ring_perfused"] = {h: {s: float(solve(h, W_CHOROID, ring_on=False)[c]) for s, c in SITES.items()} for h in (40, 80, 200)}
    out["ring_band_deg"] = [float(np.degrees(TH_CORNEA)), float(np.degrees(TH_RING2))]
    # Thesis Table H (tab:code-conduction) rows, stored explicitly:
    # h = 30 = the no-perfusion case closest to the human measurements; h = 66 and 83 = the
    # no-perfusion range that matches the COMSOL (perfused) model at the deep sites, i.e.
    # how strong a surface exchange stands in for choroidal perfusion; perfusion x0.5 / x2
    # at h = 30 for the sensitivity sentence.
    out["table_rows"] = {
        "no_perfusion_h30": eps(solve(30)),
        "no_perfusion_h66": eps(solve(66)),
        "no_perfusion_h83": eps(solve(83)),
        "perfused_h20": eps(solve(20, W_CHOROID)),
        "perfused_h40": eps(solve(40, W_CHOROID)),
        "perfused_x0.5_h30": eps(solve(30, 0.5 * W_CHOROID)),
        "perfused_x1_h30": eps(solve(30, W_CHOROID)),
        "perfused_x2_h30": eps(solve(30, 2 * W_CHOROID)),
    }
    out["geometry"] = dict(globe_diameter_mm=2e3 * R, sclera_mm=0.7, choroid_mm=0.25, retina_mm=0.25,
                           choroid_retina_from_deg=40, cornea_half_angle_deg=float(np.degrees(TH_CORNEA)),
                           nerve_radius_mm=1.5, nerve_length_mm=13, ring_area_mm2=390, grid_mm=1e3 * D,
                           k_W_per_mK=K, perfusion_per_s=W_CHOROID, rho_c_blood=RHOC_B,
                           h_air=10.0, T_air=T_AIR, T_ring=T_RING, T_body=T_HOT,
                           optic_nerve_site_mm=dict(r=1.86, depth_inside_globe=round(1e3 * (R - (24.6e-3 + ZC)), 2)))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda v: round(float(v), 3)))
