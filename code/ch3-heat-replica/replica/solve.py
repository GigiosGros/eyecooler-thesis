"""Stationary and time-dependent solution of the heat problem.

Weak form, for every test function v (T continuous, linear on tetrahedra):

    int rho*cp dT/dt v + int k grad(T).grad(v) + int w_b (T - T_b) v
        = int Q_met v + oint q_s v + oint eps*sigma (T_amb^4 - T^4) v

with fixed temperatures imposed at the nodes of the temperature boundaries.
Time stepping is second-order BDF with a fixed step inside each stage; the
radiation term is solved by Newton iteration and the temperature-dependent
material properties are evaluated at the extrapolated temperature.
"""
import time

import numpy as np
import scipy.sparse as sp

from . import fem

try:
    import pypardiso

    def factorize(A):
        s = pypardiso.PyPardisoSolver()
        A = sp.csr_matrix(A)
        s.factorize(A)
        f = lambda b: s.solve(A, b)
        f.free = lambda: s.free_memory(everything=True)     # phase -1: the only call that returns all of it
        return f
except ImportError:                                    # pragma: no cover
    from scipy.sparse.linalg import splu

    def factorize(A):
        return splu(sp.csc_matrix(A)).solve


class Heat:
    def __init__(self, prob, log=print, flow=False, supg_heat=False, flow_element="p1"):
        """flow=True also solves the Laminar Flow interface and adds convection to the heat equation.

        flow_element: "p1" linear velocity and pressure (natural convection in the eye models),
        "p2" quadratic velocity and linear pressure (forced flow; see flow2.py)."""
        self.p = p = prob
        self.log = log
        self.flow_tol, self.flow_maxit = 1e-3, 3      # per time step: a few linearised solves from the extrapolated flow
        self.supg_heat = supg_heat       # streamline-upwind term in the heat equation (needed at high Peclet number)
        self.n = n = p.points.shape[0]
        vol, g = fem.tet_geometry(p.points, p.tets)
        self.vol = vol
        var = np.isin(p.dom, list(p.variable))
        self.var_el = np.where(var)[0]

        # element coefficient arrays for the constant-property domains
        kc = np.zeros(len(vol)); cc = np.zeros(len(vol)); wb = np.zeros(len(vol)); src = np.zeros(len(vol))
        for d in np.unique(p.dom):
            m = p.dom == d
            if d not in p.variable:
                kc[m] = p.k[d]; cc[m] = p.rho_cp[d]
            if d in p.perfusion:
                wb[m] = p.perfusion[d]
                src[m] = p.q_met[d] + p.perfusion[d] * p.t_blood[d]
        t0 = time.time()
        asm = fem.Assembler(p.tets, n)
        stiff = np.einsum("eik,ejk->eij", g, g) * vol[:, None, None]
        mass = fem.MASS_REF[None] * vol[:, None, None]
        self.K0 = asm.matrix(stiff * kc[:, None, None] + mass * wb[:, None, None])
        self.M0 = asm.matrix(mass * cc[:, None, None])
        self.F0 = fem.load(p.tets, np.repeat((src * vol / 4)[:, None], 4, 1), n)
        # variable-property elements keep their own (small) pattern
        self.vt = p.tets[self.var_el]
        self.vdom = p.dom[self.var_el]
        self.vstiff = stiff[self.var_el]
        self.vmass = mass[self.var_el]
        self.vasm = fem.Assembler(self.vt, n) if len(self.var_el) else None
        del stiff, mass

        # fluid regions; only those with something driving a flow are solved
        self.regions = []
        if flow and p.flow:
            from .flow import Region
            if flow_element == "p2":
                from .flow2 import TaylorHood as Region
            for spec in p.flow["regions"]:
                r = Region(p, spec, vol, g)
                r.tasm = fem.Assembler(p.tets[r.el], n)
                if spec["beta"] != 0.0 or np.abs(r.value_u).max() > 0:
                    self.regions.append(r)
                    log(f"  flow region {spec['name']}: {r.n} nodes, {len(r.el)} elements, "
                        f"beta {spec['beta']:.3e} 1/K, inflow {np.abs(r.value_u).max():.2e} m/s")
                else:
                    log(f"  flow region {spec['name']}: constant density and no inlet, nothing drives a flow -> at rest")

        # boundary terms
        for s in p.surface_source:
            a = fem.tri_area(p.points, s["tris"])
            self.F0 += fem.load(s["tris"], np.repeat((s["q"] * a / 3)[:, None], 3, 1), n)
        rt = [r["tris"] for r in p.radiation]
        self.rad_tris = np.concatenate(rt) if rt else np.zeros((0, 3), int)
        self.rad_eps = np.concatenate([np.full(len(r["tris"]), r["eps"]) for r in p.radiation]) if rt else np.zeros(0)
        self.rad_tamb = np.concatenate([np.full(len(r["tris"]), r["t_amb"]) for r in p.radiation]) if rt else np.zeros(0)
        self.rad_area = fem.tri_area(p.points, self.rad_tris)
        self.rasm = fem.Assembler(self.rad_tris, n) if rt else None
        self.conv_bc = [dict(c, area=fem.tri_area(p.points, c["tris"]), asm=fem.Assembler(c["tris"], n))
                        for c in p.convective]
        log(f"  assembled {n} nodes, {len(vol)} elements in {time.time() - t0:.1f} s "
            f"({len(self.var_el)} elements with temperature-dependent properties)")

    # ------------------------------------------------------------ pieces
    def variable_matrices(self, T):
        """Stiffness and mass of the temperature-dependent domains at temperature field T."""
        if self.vasm is None:
            z = sp.csr_matrix((self.n, self.n))
            return z, z
        Te = T[self.vt].mean(1)
        k = np.empty(len(Te)); c = np.empty(len(Te))
        for d in self.p.variable:
            m = self.vdom == d
            k[m] = self.p.k[d](Te[m]); c[m] = self.p.rho_cp[d](Te[m])
        return (self.vasm.matrix(self.vstiff * k[:, None, None]),
                self.vasm.matrix(self.vmass * c[:, None, None]))

    def convection(self, T):
        """Matrix of the convective term rho*cp*(u.grad T) for the current velocity fields."""
        C = sp.csr_matrix((self.n, self.n))
        for r in self.regions:
            rc = self.p.rho_cp[r.spec["domains"][0]]
            rc = rc(T[self.p.tets[r.el]].mean(1)) if callable(rc) else rc
            if hasattr(r, "heat_convection"):                                    # quadratic velocity: integrated exactly
                kk = self.p.k[r.spec["domains"][0]]
                kk = kk(T[self.p.tets[r.el]].mean(1)) if callable(kk) else kk
                C = C + r.tasm.matrix(r.heat_convection(rc, kk, upwind=self.supg_heat))
                continue
            ug = np.einsum("ek,eik->ei", r.element_velocity(), r.g)             # u.grad(phi_j)
            loc = (rc * r.vol / 4)[:, None, None] * np.ones((1, 4, 1)) * ug[:, None, :]
            if self.supg_heat:
                # Streamline-upwind term, needed when convection dominates conduction
                # on the scale of an element.  Known limit: with linear velocity and
                # pressure the discrete flow conserves mass only on average, so at
                # high Reynolds number on a coarse mesh this term does not balance
                # heat exactly (see README, water-block test).
                kk = self.p.k[r.spec["domains"][0]]
                kk = kk(T[self.p.tets[r.el]].mean(1)) if callable(kk) else kk
                un = np.linalg.norm(r.element_velocity(), axis=1)
                tau = 1.0 / np.sqrt((2 * un / r.h) ** 2 + (4 * kk / (rc * r.h ** 2)) ** 2)
                loc = loc + (tau * rc * r.vol)[:, None, None] * ug[:, :, None] * ug[:, None, :]
            C = C + r.tasm.matrix(loc)
        return C

    def radiation(self, T):
        """Nodal radiative load and its (negative) derivative matrix.

        The surface term eps*sigma*(T_amb^4 - T^4) is integrated with the
        temperature interpolated linearly and a three-point rule at the edge
        midpoints (exact for quadratics).
        """
        f_c, B_c = np.zeros(self.n), sp.csr_matrix((self.n, self.n))
        for c in self.conv_bc:               # convective heat flux h(T)*(T_ext - T), h at the face temperature
            Tf = T[c["tris"]].mean(1)
            hA = c["h"](Tf) * c["area"]
            B = c["asm"].matrix(fem.TRI_MASS_REF[None] * hA[:, None, None])
            f_c += fem.load(c["tris"], np.repeat((hA * c["t_ext"] / 3)[:, None], 3, 1), self.n) - B @ T
            B_c = B_c + B
        if self.rasm is None:
            return f_c, B_c
        Tn = T[self.rad_tris]                                    # (nt, 3)
        N = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5]])    # shape values at the 3 points
        Tq = Tn @ N.T                                            # (nt, 3 points)
        w = (self.rad_area / 3)[:, None] * self.rad_eps[:, None] * fem.SIGMA
        q = w * (self.rad_tamb[:, None] ** 4 - Tq ** 4)
        f = fem.load(self.rad_tris, q @ N, self.n)
        dq = w * 4 * Tq ** 3
        loc = np.einsum("tq,qi,qj->tij", dq, N, N)
        return f + f_c, self.rasm.matrix(loc) + B_c

    def constraints(self, step, t):
        """Fixed-temperature nodes and values for a study step.

        A node that lies on several active temperature conditions (the edge where the
        body-temperature boundary meets the air wall) gets the mean of their values.
        That is what COMSOL's stored solution holds there: 31.00 C between 37 C and
        25 C, read exactly from the file (dev/test_conflicting_temperatures.py).
        Letting the later condition win instead leaves those 227 nodes 6 K off and
        shifts a floating ring by 0.17 K.
        """
        tot, cnt = np.zeros(self.n), np.zeros(self.n)
        for d in self.p.dirichlet:
            if step in d["steps"]:
                np.add.at(tot, d["nodes"], d["value"](t) * np.ones(len(d["nodes"])))
                np.add.at(cnt, d["nodes"], 1.0)
        idx = np.where(cnt > 0)[0]
        return idx, tot[idx] / cnt[idx]

    # -------------------------------------------------------- stationary
    def stationary(self, step="stat", tol=1e-8, maxit=30, flow_passes=None, t=0.0):
        """flow_passes: how many times the flow is re-solved with the new temperature
        (None = every iteration, for buoyancy-driven flow; a small number for forced
        flow, where temperature only enters through the viscosity)."""
        T = self.p.t_init.copy()
        fixed, vals = self.constraints(step, t)       # t: time at which time-dependent conditions are taken
        free = np.setdiff1d(np.arange(self.n), fixed)
        T[fixed] = vals
        for it in range(maxit):
            if flow_passes is None or it < flow_passes:
                for r in self.regions:
                    r.solve(T, r.spec["t_ref"], self.p.flow["gravity"], relax=0.7 if it else 1.0,
                            **({} if flow_passes is None else dict(maxit=80, tol=1e-5)))
            Kv, _ = self.variable_matrices(T)
            fr, Br = self.radiation(T)
            A = (self.K0 + Kv + self.convection(T)).tocsr()
            R = A @ T - self.F0 - fr
            J = (A + Br).tocsr()
            dT = np.zeros(self.n)
            lu = factorize(J[free][:, free])
            dT[free] = lu(-R[free])
            if hasattr(lu, "free"):
                lu.free()
            T += dT
            err = np.abs(dT).max()
            umax = max([np.abs(r.u).max() for r in self.regions], default=0.0)
            self.log(f"  stationary iteration {it + 1}: max change {err:.3e} K"
                     + (f", max |u| {umax:.3e} m/s" if self.regions else ""))
            if err < (1e-6 if self.regions else tol):
                break
        else:
            raise RuntimeError("stationary solution did not converge")
        self._log_flow_maxit()
        return T

    # ----------------------------------------------------- time dependent
    def transient(self, T0, t_end, stages, step="time", callback=None, tol=1e-7):
        """BDF2 from T0.  stages: list of (t_until, dt).  callback(t, T) after every step."""
        T_prev, T = None, T0.copy()
        t = 0.0
        fixed, vals = self.constraints(step, 0.0)
        free = np.setdiff1d(np.arange(self.n), fixed)
        solve = None
        sig = None
        dt_old = None
        nfact = nstep = niter = 0
        if callback:
            callback(t, T)
        for t_until, dt in stages:
            t_until = min(t_until, t_end)
            nst = int(round((t_until - t) / dt))
            for _ in range(nst):
                if T_prev is None:                      # first step: backward Euler
                    a0, a1, a2 = 1.0 / dt, -1.0 / dt, 0.0
                    Tx = T.copy()
                else:
                    w = dt / dt_old
                    a0 = (1 + 2 * w) / (1 + w) / dt
                    a1 = -(1 + w) / dt
                    a2 = w * w / (1 + w) / dt
                    Tx = T + w * (T - T_prev)           # extrapolated temperature
                t_new = t + dt
                _, vals = self.constraints(step, t_new)
                for r in self.regions:                  # flow, driven by the extrapolated temperature
                    un = r.u.copy()
                    have_prev = T_prev is not None and r.u_prev is not None
                    hist = a1 * un + (a2 * r.u_prev if have_prev else 0.0)
                    if have_prev:                       # start the iteration from the extrapolated velocity
                        free_u = ~r.fixed_u
                        r.u[free_u] = (un + (dt / dt_old) * (un - r.u_prev))[free_u]
                    r.solve(Tx, r.spec["t_ref"], self.p.flow["gravity"], a=a0, hist=hist,
                            tol=self.flow_tol, maxit=self.flow_maxit,
                            log=self.log if getattr(self, "debug_flow", None) and self.debug_flow(t_new) else None)
                    r.u_prev = un
                Kv, Mv = self.variable_matrices(Tx)
                M = self.M0 + Mv
                K = self.K0 + Kv + (self.convection(Tx) if self.regions else 0)
                hist = M @ (a1 * T + (a2 * T_prev if T_prev is not None else 0.0))
                A = (a0 * M + K).tocsr()
                Tn = Tx.copy()
                Tn[fixed] = vals
                for k in range(25):
                    fr, Br = self.radiation(Tn)
                    R = A @ Tn + hist - self.F0 - fr
                    if solve is None or sig != (a0,) or k == 8 or (self.regions and k == 0 and nstep % 5 == 0):
                        if solve is not None and hasattr(solve, "free"):
                            solve.free()
                        solve = factorize((A + Br)[free][:, free])
                        sig = (a0,)
                        nfact += 1
                    d = solve(-R[free])
                    Tn[free] += d
                    niter += 1
                    if np.abs(d).max() < tol:
                        break
                else:
                    raise RuntimeError(f"time step at t={t_new:.1f} s did not converge")
                T_prev, T, t, dt_old = T, Tn, t_new, dt
                nstep += 1
                if callback:
                    callback(t, T)
        if solve is not None and hasattr(solve, "free"):
            solve.free()
        self.log(f"  {nstep} time steps, {niter} iterations, {nfact} factorizations")
        self._log_flow_maxit()
        self.last_T = T
        return T

    def _log_flow_maxit(self):
        """(2026-10-28) Say how many flow solves of each region stopped at their iteration limit since
        the last report, and reset the count.  In the transient this is expected (flow_maxit = 3 linearised solves
        per step, the coupling is converged by the step's own loop); in a stationary solve it is a warning."""
        for r in getattr(self, "regions", []):
            n = getattr(r, "n_maxit", 0)
            if n:
                self.log(f"  flow region {r.spec.get('name', '?')}: {n} flow solve(s) stopped at maxit, "
                         f"worst relative change {getattr(r, 'worst_rel_change', float('nan')):.2e}")
                r.n_maxit, r.worst_rel_change = 0, 0.0

    # ------------------------------------------------------------ outputs
    def probe_setup(self):
        p = self.p
        self._pp = {}
        for name, pr in p.probes.items():
            if pr["kind"] == "point":
                e, lam, worst = fem.locate(p.points, p.tets, pr["xyz"])[0]
                if worst < -1e-6:
                    # (2026-10-28) outside every element: evaluating with these coordinates was a
                    # silent linear extrapolation.  Clamp to the nearest point of the best element and log it.
                    lam = np.clip(np.asarray(lam, float), 0.0, None)
                    lam = lam / lam.sum()
                    self.log(f"  probe {name}: outside the mesh (lambda_min {worst:.2e}); clamped onto its nearest "
                             f"element ({p.domain_names.get(int(p.dom[e]), int(p.dom[e])) if hasattr(p, 'domain_names') else int(p.dom[e])})")
                self._pp[name] = ("point", p.tets[e], lam, int(p.dom[e]), worst)
            else:
                m = np.isin(p.dom, pr["domains"])
                self._pp[name] = (pr["kind"], p.tets[m], self.vol[m])
        return self._pp

    def probes(self, T):
        out = {}
        for name, pr in self._pp.items():
            if pr[0] == "point":
                out[name] = float(T[pr[1]] @ pr[2])
            elif pr[0] == "average":
                out[name] = float((T[pr[1]].mean(1) * pr[2]).sum() / pr[2].sum())
            elif pr[0] == "maximum":
                out[name] = float(T[pr[1]].max())
            elif pr[0] == "minimum":
                out[name] = float(T[pr[1]].min())
        return out
