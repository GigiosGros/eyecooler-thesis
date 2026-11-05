"""Temperature marched in time together with an unsteady flow (simulation plan, section 9a, step R3).

Imported by dev/wb_runs.py (setting `heat_march`).  Why: where the flow does not settle, dev/wb_runs.py solves the
temperature once on the time-averaged velocity.  That leaves out the heat carried by the fluctuations: with
u = U + u' and T = Tm + T' (time mean + fluctuation), the mean of the convective term is U.grad(Tm) plus the mean
of u'.grad(T'), and the second part is missing.  Here flow and temperature are stepped together and the temperature
is averaged, so both parts are in.  The same run also averages the velocity over the same time window and solves
the stationary temperature on that mean, so the two answers differ by the missing part and by nothing else.

Equations, per time step of length dt (second-order backward differences; first step first order):

    flow         rho (a0 u_new + a1 u + a2 u_old) + rho (w.grad) u_new = -grad p + div(mu grad u_new),  div u_new = 0
    temperature  M (a0 T_new + a1 T + a2 T_old) + (K + C(u_new)) T_new = F

where u is the velocity, p the pressure, rho the density and mu the viscosity of the water, w = 2 u - u_old the
extrapolated convecting velocity, a0 = 1.5/dt, a1 = -2/dt, a2 = 0.5/dt, T the temperature at the mesh nodes,
M the heat-capacity matrix, K the conduction matrix with the exchange with the air, C(u) the convection matrix of
the water for the velocity u (with the same streamline term as in the stationary runs), and F the heat put in
plus the exchange with the air.  The flow does not depend on the temperature here (viscosity at the inlet
temperature, no buoyancy), as in the runs it is compared with.

Solid heat capacity: the aluminium answers in tens of channel transit times, the water in about one.  The time
mean of the settled state does not depend on how fast the solid gets there, so M of the solid is multiplied by
`solid_capacity` (default 0.02) to reach the settled state in a few transits.  The solid's heat capacity also
sets how much the wall temperature follows the fluctuations of the water, which could change the mean slightly;
run a case twice with different factors to see that (cases W10 ..._cap0.2).  The drift of the heated-face
temperature over the averaging window is reported: it must be small against the difference being measured.
"""
import time

import numpy as np
import scipy.sparse as sp

from replica import fem
from robust_solve import factorize

import wb_common as wb


def _solid_mass(h, T):
    """Heat-capacity matrix of the solid elements alone (linear tetrahedra), at the temperature field T."""
    p = h.p
    solid = np.flatnonzero(~np.isin(p.dom, wb.fluid_domains(p)))
    tets = p.tets[solid]
    vol, _ = fem.tet_geometry(p.points, tets)
    rc = np.empty(len(solid))
    Te = T[tets].mean(1)
    for d in np.unique(p.dom[solid]):
        m = p.dom[solid] == d
        x = p.rho_cp[d]
        rc[m] = x(Te[m]) if callable(x) else x
    loc = (rc * vol / 20.0)[:, None, None] * (np.ones((4, 4)) + np.eye(4))[None]
    return fem.Assembler(tets, p.points.shape[0]).matrix(loc)


def march_heat(r, h, T_start, step_tag, faces, steps_per_transit=200, spinup_transits=1.0, average_transits=4.0,
               solid_capacity=0.02, log=None, deadline=None, every=20):
    """Step flow `r` (a WBFlow that has just been marched and found unsteady) and heat problem `h` together.

    Returns dict(T_mean, T_on_mean_flow, u_mean, p_mean, monitors, names, info).  r.u, r.p are left at the mean
    of the averaging window (so read-outs of the flow are those of the mean flow, as in the other runs)."""
    log = log or (lambda s: None)
    hs = list(h) if isinstance(h, (list, tuple)) else [h]                # several variants: same mesh and flow
    Ts = list(T_start) if isinstance(T_start, (list, tuple)) else [T_start]
    single = not isinstance(h, (list, tuple))
    h = hs[0]
    n = h.n
    if getattr(r, "last_state", None) is None:
        raise RuntimeError("march_heat needs the instantaneous state left by WBFlow.march")
    u, u_old, p, dt0 = r.last_state
    transit = r.vol.sum() / r.inflow
    dt = transit / steps_per_transit
    if u_old is not None and abs(dt - dt0) > 1e-12 * dt:
        u_old = None                                      # another step length: start again with a first-order step
    r.u, r.p = u.copy(), p.copy()
    r.u[r.fixed_u] = r.value_u[r.fixed_u]
    cons = [hv.constraints(step_tag, 0.0) for hv in hs]
    fixed = cons[0][0]
    for fx, _ in cons[1:]:
        if not np.array_equal(fx, fixed):
            raise RuntimeError("variants with different fixed-temperature nodes cannot share a factorisation")
    free = np.setdiff1d(np.arange(n), fixed)
    Tc, Tp = [], [None] * len(hs)
    for (fx, vals), T0 in zip(cons, Ts):
        x = T0.copy()
        x[fixed] = vals
        Tc.append(x)
    Ms = _solid_mass(h, Tc[0])
    names = ["t_s", "pressure_drop_Pa"] + [f"{k} average (K)" for k in faces] + ["Water maximum (K)"]
    water_nodes = np.unique(h.p.tets[np.isin(h.p.dom, wb.fluid_domains(h.p))])
    rows = []
    n_spin = int(round(spinup_transits * steps_per_transit))
    n_avg = int(round(average_transits * steps_per_transit))
    acc_T = [np.zeros(n) for _ in hs]
    acc_u, acc_p, nacc = np.zeros_like(r.u), np.zeros_like(r.p), 0
    t, t0 = 0.0, time.time()
    worst_defect = 0.0
    info = dict(time_step_s=dt, transit_s=transit, solid_capacity=solid_capacity, steps_per_transit=steps_per_transit, variants=len(hs))
    for nstep in range(1, n_spin + n_avg + 1):
        if deadline and time.time() > deadline:
            info["stopped"] = "time_limit"
            break
        t += dt
        un = r.u.copy()
        if u_old is None:
            a, hist = 1.0 / dt, -un / dt
        else:
            a, hist = 1.5 / dt, (-2.0 * un + 0.5 * u_old) / dt
            fr_ = ~r.fixed_u
            r.u[fr_] = (2.0 * un - u_old)[fr_]
        r.step(r._T, r.spec["t_ref"], None, a, hist, upwind=r.upwind)
        u_old = un
        if Tp[0] is None:
            a0, a1, a2 = 1.0 / dt, -1.0 / dt, 0.0
        else:
            a0, a1, a2 = 1.5 / dt, -2.0 / dt, 0.5 / dt
        lu = None
        for i, hv in enumerate(hs):
            T, T_prev = Tc[i], Tp[i]
            Tx = T if T_prev is None else 2.0 * T - T_prev
            Kv, Mv = hv.variable_matrices(Tx)
            M = (hv.M0 + Mv - (1.0 - solid_capacity) * Ms).tocsr()
            A = (a0 * M + hv.K0 + Kv + hv.convection(Tx)).tocsr()
            Tn = Tx.copy()
            Tn[fixed] = cons[i][1]
            hist_T = M @ (a1 * T + (a2 * T_prev if T_prev is not None else 0.0))
            # exchange with the air (and radiation, if any): load `sur` at Tn and minus its derivative `Bs`, as in
            # Heat.stationary.  The first variant's matrix is factorised; the others are solved with that factor by
            # defect correction against their own matrix (equal unless properties or air coefficients depend on T)
            sur, Bs = hv.radiation(Tn)
            res_T = A @ Tn + hist_T - hv.F0 - sur
            scale = max(float(np.abs((hv.F0 + sur - hist_T)[free]).max()), 1e-300)
            if lu is None:                                   # first variant: one Newton step, as for a single variant
                lu = factorize((A + Bs).tocsr()[free][:, free])
                Tn[free] -= lu(res_T[free])
                rn = 0.0
            else:                                            # other variants: defect correction with the shared factor
                for k in range(40):                         # a different matrix (air factor, solid conductivity) needs more
                    Tn[free] -= lu(res_T[free])
                    sur, Bs = hv.radiation(Tn)
                    res_T = A @ Tn + hist_T - hv.F0 - sur
                    rn = float(np.abs(res_T[free]).max()) / scale
                    if rn < 1e-10:
                        break
            worst_defect = max(worst_defect, rn)
            if not np.isfinite(Tn).all():
                raise FloatingPointError("temperature time step diverged")
            Tp[i], Tc[i] = T, Tn
        if hasattr(lu, "free"):
            lu.free()
        T = Tc[0]
        rows.append([t, r.pressure_drop()] + [wb.face_average(T, f) for f in faces.values()] + [float(T[water_nodes].max())])
        if nstep > n_spin:
            for i in range(len(hs)):
                acc_T[i] += Tc[i]
            acc_u += r.u
            acc_p += r.p
            nacc += 1
        if nstep % every == 0 or nstep == 1:
            log(f"    heat march t = {t:8.4f} s ({t / transit:5.2f} transits)  pressure drop {rows[-1][1]:9.3f} Pa  "
                + ", ".join(f"{k.replace(' average (K)', '')} {v - wb.K0:.3f}" for k, v in zip(names[2:], rows[-1][2:]) if "Heated" in k or "TEC" in k or "HotSide" == k.split(" average")[0])
                + f" C  ({(time.time() - t0) / nstep:.1f} s per step, {len(hs)} variant(s))")
    mon = np.array(rows)
    info["largest_relative_defect_of_shared_factor"] = worst_defect
    if nacc == 0:
        info["status"] = "no_average"
        out = dict(T_mean=Tc[0], T_on_mean_flow=None, monitors=mon, names=names, info=info)
        if not single:
            out.update(T_means=Tc, T_on_mean_flows=[None] * len(hs))
        return out
    T_means = [x / nacc for x in acc_T]
    r.u, r.p = acc_u / nacc, acc_p / nacc
    r.u[r.fixed_u] = r.value_u[r.fixed_u]
    T_on = [hv.stationary(step=step_tag, flow_passes=0, tol=1e-6) for hv in hs]    # stationary temperature on the mean flow of the same window
    # drift and fluctuation of each face average over the averaging window (first variant)
    w = mon[-nacc:]
    tt = w[:, 0] - w[:, 0].mean()
    drift, fluct = {}, {}
    for j, k in enumerate(names[2:], start=2):
        slope = float((tt * (w[:, j] - w[:, j].mean())).sum() / (tt * tt).sum())
        drift[k] = slope * float(np.ptp(w[:, 0]))
        fluct[k] = float(np.std(w[:, j] - w[:, j].mean() - slope * tt))
    # (2026-10-22) "ok" only when the planned averaging window was completed; a march cut by the time
    # limit is "short_window" (9 cases of the 10-19 pull averaged 0.2-2.7 of 3 transits and were reported "ok").
    # The readers (dev/wb_research_read.py via dev/wb_void.short_window) flag stored "ok" cases by averaged_transits.
    info.update(status="ok" if nacc >= n_avg else "short_window", requested_average_transits=average_transits,
                steps=len(rows), averaged_steps=nacc, averaged_transits=nacc / steps_per_transit,
                drift_over_window_K=drift, fluctuation_sd_K=fluct, seconds_per_step=(time.time() - t0) / max(len(rows), 1))
    # (2026-10-22) validity screen: a diverged flow (W14_v3_normal_Q0.25_Tin10_sd0.25: mean pressure drop
    # -255,892 kPa) was reported "ok".  Column 1 of the monitors is the pressure drop (Pa).
    dp_w = w[:, 1]
    info["mean_pressure_drop_over_window_Pa"] = float(np.mean(dp_w))
    if not np.isfinite(dp_w).all() or float(np.mean(dp_w)) <= 0.0 or float(np.ptp(dp_w)) >= 2.0 * abs(float(np.mean(dp_w))):
        info["status"] = "diverged"
    out = dict(T_mean=T_means[0], T_on_mean_flow=T_on[0], monitors=mon, names=names, info=info)
    if not single:
        out.update(T_means=T_means, T_on_mean_flows=T_on)
    return out
