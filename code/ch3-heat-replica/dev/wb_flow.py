"""Steady forced flow for the water block: pseudo-time stepping with an exact Newton linearisation, and a
time-accurate march that can tell a steady flow from one that keeps oscillating.

Imported by dev/wb_runs.py; tested by dev/wb_test_flow.py.  replica/flow2.py and dev/flow2_td.py are used
unchanged: `WBFlow` is a subclass of `TaylorHoodTD` and solves the SAME discrete equations as
`replica.flow2.TaylorHood.solve` (quadratic velocity, linear pressure, streamline diffusion with the same
parameter), so a converged `WBFlow.steady` is the solution that iteration was heading for.

Why the stationary iteration of flow2.py stalls (it reached 80 iterations without converging at half load):
its "Newton" matrix leaves out the derivative of the streamline-diffusion term, which on these meshes is
10-40 times the physical viscosity along the flow, so the iteration is a fixed-point iteration with a poor
matrix; and it has no globalisation except a cap on the step.  Here

  * the matrix is the exact derivative of the residual (convection, streamline diffusion and its parameter);
    checked against finite differences in dev/wb_test_flow.py;
  * each step solves  (rho M / dt + J(x)) dx = R(x)  (one backward-Euler step, linearised).  dt starts at a
    few element transit times and is multiplied by the factor by which the residual fell (switched evolution
    relaxation); a step that raises the residual is rejected and dt is cut.  As dt grows the step becomes a
    pure Newton step, so the end of the iteration converges quadratically;
  * a large dt damps every mode, growing or not, so this finds a steady solution even where the real flow
    would not stay steady (as COMSOL's stationary solver does).  Whether the flow is steady is a separate
    question, answered by `march`: second-order time stepping at a fixed step, with the pressure drop, the
    kinetic energy and the velocity at a ring of points recorded every step; `classify` reads those series and
    returns "steady", "unsteady" (with amplitude and period) or "not settled".

Stabilisation (`stab`, added 2026-10-15).  "sd" (default, as before): the streamline term
rho*tau*(w.grad v).(w.grad u) only, i.e. the convective part of the momentum residual tested with tau*(w.grad v).
"supg": the whole strong momentum residual is tested (consistent SUPG),

    sum_e  int_e  tau (u.grad v_a) . R(u, p)  dV,      R = rho (u.grad) u + grad p - mu lap(u) - f

where v_a is the quadratic test function of velocity node a, u the velocity (also the convecting velocity in the
stationary equations), p the pressure (linear: grad p constant per element), mu the viscosity (constant per
element), lap(u) the Laplacian of the quadratic velocity (constant per element: lap(phi_i) = 4 |g_i|^2 for a
vertex function and 8 g_i.g_j for the function of edge ij, g_i the gradient of barycentric coordinate i), f the
buoyancy force per volume (-rho beta (T - T_ref) gravity) and tau the same parameter as for "sd".  The added terms
vanish for the exact solution, so the stabilisation no longer adds diffusion along the streamlines; the
verification of 2026-10-15 (cylinder, Re 100) showed that the "sd" term lowers the shedding frequency by 7 % and
the amplitude by a third.  The Newton matrix includes every added term exactly (checked against finite differences
in dev/wb_test_flow.py); the time step (dev/flow2_td.py) adds the time-derivative part rho (a u + hist) too.

Memory: the residual is formed from the blocks (no global matrix); the Newton matrix is assembled directly
on the free unknowns.  The factorisation (PARDISO, through replica.solve.factorize) is the large item.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from flow2_td import TaylorHoodTD  # noqa: E402
from replica.flow import outward_normals  # noqa: E402
from replica.flow2 import QP  # noqa: E402
from robust_solve import factorize  # noqa: E402  (2026-10-14: checked solves; see dev/robust_solve.py)


class WBFlow(TaylorHoodTD):
    upwind = 1.0                                   # factor on the streamline-diffusion parameter (1 = flow2.py)
    stab = "sd"                                    # "sd" (streamline term only, as before) or "supg" (consistent)

    def __init__(self, prob, spec, vol, grad):
        super().__init__(prob, spec, vol, grad)
        self.free_c = [np.where(~self.fixed_u[:, c])[0] for c in range(3)]
        self.mu_e = None
        self.Fb = np.zeros((self.n2, 3))
        self.he = self.h / 2.0
        self.W = self.w                                                      # (ne, nq) weights * volume
        self.phi_t = np.ascontiguousarray(self.phi.T)                        # (10, nq)
        self._faces(prob)
        self.inlet_mean_speed = float(self.inflow / self.inlet_faces["area"].sum()) if self.inlet_faces is not None else 0.0
        self.set_temperature(np.full(prob.points.shape[0], spec["t_ref"]))

    # ------------------------------------------------------------ set-up
    def _faces(self, prob):
        """Inlet and outlet faces as local vertex / mid-side unknowns, with normals and areas."""
        spec = self.spec
        inside = np.isin(prob.tri_up, spec["domains"]).astype(int) + np.isin(prob.tri_down, spec["domains"]).astype(int)
        surface = inside == 1
        glob = np.full(prob.points.shape[0], -1, np.int64)
        glob[self.nodes] = np.arange(self.n1)
        self.xyz = np.concatenate([prob.points[self.nodes], prob.points[self.nodes][self.edge_nodes].mean(1)])

        def pack(mask):
            ft = prob.tris[mask]
            if not len(ft):
                return None
            nn = outward_normals(prob.points, prob.tets[self.el], ft)
            v = glob[ft]
            m = np.array([[self.edge_of[tuple(sorted((int(r[i]), int(r[j]))))] for i, j in ((0, 1), (1, 2), (0, 2))] for r in v], int)
            return dict(v=v, m=m, n=nn, area=np.linalg.norm(nn, axis=1))
        self.inlet_faces = pack(surface & np.isin(prob.tri_bnd, [b for bl, _ in spec["inlets"] for b in bl]))
        self.outlet_faces = pack(surface & np.isin(prob.tri_bnd, spec["outlets"]))

    def set_temperature(self, T_glob, gvec=None):
        """Viscosity (per element) and buoyancy load for a temperature field."""
        Te = T_glob[self.nodes][self.cells1].mean(1)
        self._T = T_glob
        mu = self.spec["mu"]
        self.mu_e = np.asarray(mu(Te), float) if callable(mu) else np.full(len(self.vol), float(mu))
        self.Fb[:] = 0.0
        self.fq = None                                                       # buoyancy force per volume at the points
        beta = self.spec.get("beta", 0.0)
        rho_fn = self.spec.get("rho_fn")
        if beta and gvec is not None and rho_fn is not None:
            # (2026-10-22, opt-in: spec["rho_fn"], set by dev/wb_runs.py only with WB_BUOYANCY_DENSITY=local)
            # force per volume f = [rho(T) - rho(T_ref)] g at each quadrature point, where rho is the coolant density
            # (kg/m3) at the local temperature T (K), T_ref the reference temperature (K) and g gravity (m/s2); the
            # Boussinesq form below linearises rho at T_ref (13x too weak at 4 C against the secant over 4-31 C).
            Tq_abs = np.einsum("qi,ei->eq", QP, T_glob[self.nodes][self.cells1])
            drho = np.asarray(rho_fn(Tq_abs), float) - float(rho_fn(self.spec["t_ref"]))
            self.fq = drho[:, :, None] * np.asarray(gvec, float)[None, None, :]
            fe = np.einsum("eq,qa->ea", self.W * drho, self.phi)
            for d in range(3):
                if gvec[d]:
                    self.Fb[:, d] = np.bincount(self.cells2.ravel(), (fe * gvec[d]).ravel(), self.n2)
        elif beta and gvec is not None:
            Tq = np.einsum("qi,ei->eq", QP, T_glob[self.nodes][self.cells1]) - self.spec["t_ref"]
            self.fq = -self.rho * beta * Tq[:, :, None] * np.asarray(gvec, float)[None, None, :]
            fe = -self.rho * beta * np.einsum("eq,qa->ea", self.W * Tq, self.phi)
            for d in range(3):
                if gvec[d]:
                    self.Fb[:, d] = np.bincount(self.cells2.ravel(), (fe * gvec[d]).ravel(), self.n2)

    def set_inlet_profile(self, kind):
        """'plug' (uniform, as in the COMSOL model) or 'parabolic' (developed pipe flow) with the same volume flow."""
        if kind == "plug":
            return
        if kind != "parabolic":
            raise ValueError(kind)
        f = self.inlet_faces
        idx = np.unique(np.concatenate([f["v"].ravel(), f["m"].ravel()]))
        idx = idx[self.fixed_u[idx].all(1) & (np.abs(self._inlet_value[idx]).sum(1) > 0)]      # inlet nodes off the wall
        unit = -(f["n"] / f["area"][:, None]).mean(0)
        unit /= np.linalg.norm(unit)
        allf = np.unique(np.concatenate([f["v"].ravel(), f["m"].ravel()]))
        c = (self.xyz[f["v"]].mean(1) * f["area"][:, None]).sum(0) / f["area"].sum()
        R = np.sqrt(f["area"].sum() / np.pi)
        val = np.zeros((self.n2, 3))
        r = np.linalg.norm(np.cross(self.xyz[allf] - c, unit), axis=1)
        val[allf] = np.clip(1.0 - (r / R) ** 2, 0.0, None)[:, None] * unit
        val[np.setdiff1d(allf, idx)] = 0.0
        q = sum((a / 3.0) * (val[m] @ (-n_ / a)).sum() for a, n_, m in zip(f["area"], f["n"], f["m"]))
        new = self._inlet_value.copy()
        new[allf] = val[allf] * (self.inflow / q)
        self._inlet_value = new
        self.set_inflow(self.inflow_scale)

    # ----------------------------------------------------------- algebra
    def _kin(self, u):
        U = u[self.cells2]                                                   # (ne, 10, 3)
        uq = np.einsum("qa,ead->eqd", self.phi, U)                           # (ne, nq, 3)
        wgrad = np.einsum("eqd,eqad->eqa", uq, self.dphi)                    # u.grad(phi_a)
        sp_q = np.linalg.norm(uq, axis=2)
        s = sp_q.mean(1)
        al, be = (2.0 / self.he) ** 2, (4.0 * (self.mu_e / self.rho) / self.he ** 2) ** 2
        tau = self.upwind / np.sqrt(al * s * s + be)
        return U, uq, wgrad, sp_q, s, tau, al

    # --------------------------------------------------- consistent SUPG
    def lapphi(self):
        """Laplacian of the ten quadratic basis functions, constant per element: (ne, 10)."""
        if getattr(self, "_lapphi", None) is None:
            g = self.g
            out = np.empty((len(g), 10))
            for i in range(4):
                out[:, i] = 4.0 * np.einsum("ed,ed->e", g[:, i], g[:, i])
            for k, (i, j) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
                out[:, 4 + k] = 8.0 * np.einsum("ed,ed->e", g[:, i], g[:, j])
            self._lapphi = out
        return self._lapphi

    def supg_extra(self, U, p):
        """The part of the strong momentum residual that "sd" leaves out, at the quadrature points: (ne, nq, 3),
        E = grad p - mu lap(u) - f."""
        gp = np.einsum("ek,ekd->ed", p[self.cells1], self.g)                 # grad p, constant per element
        lu = np.einsum("eb,ebd->ed", self.lapphi(), U)                       # lap(u)
        E = np.repeat((gp - self.mu_e[:, None] * lu)[:, None, :], self.W.shape[1], axis=1)
        if getattr(self, "fq", None) is not None:
            E = E - self.fq
        return E

    def supg_pressure_blocks(self, wgrad, tau):
        """d(SUPG term)/d(pressure) for the three momentum components: three (n2, n1) matrices."""
        wt = self.W * tau[:, None]
        rows = np.repeat(self.cells2, 4, axis=1).ravel()
        cols = np.tile(self.cells1, (1, 10)).ravel()
        out = []
        for d in range(3):
            loc = np.einsum("eq,eqa,ek->eak", wt, wgrad, self.g[:, :, d])     # (ne, 10, 4)
            out.append(sp.csr_matrix((loc.ravel(), (rows, cols)), shape=(self.n2, self.n1)))
        return out

    def _supg_visc_local(self, wgrad, tau):
        """d(SUPG term)/d(u_d) through -mu lap(u_d): element matrices (ne, 10, 10), the same for the three components."""
        t1 = np.einsum("eq,eqa->ea", self.W * tau[:, None], wgrad)
        return -(self.mu_e[:, None, None] * t1[:, :, None]) * self.lapphi()[:, None, :]

    def _picard_local(self, wgrad, tau, a=0.0):
        """Element matrices of rho*a*M + mu*grad.grad + rho*(phi + tau*w.grad phi)(w.grad .)."""
        test = (self.W[:, :, None] * (self.phi[None] + tau[:, None, None] * wgrad)).transpose(0, 2, 1)   # (ne, 10, nq)
        loc = self.rho * np.matmul(test, wgrad) + self.mu_e[:, None, None] * self.lap
        if a:
            loc += (self.rho * a) * self.mass
        return loc

    def residual(self, u, p):
        """R = F - K(u) x on every unknown (fixed ones included): (n2, 3) for momentum, (n1) for continuity."""
        U, _, wgrad, _, _, tau, _ = self._kin(u)
        A = self.asm22.matrix(self._picard_local(wgrad, tau))
        ru = self.Fb - A @ u
        for d in range(3):
            ru[:, d] += self.B[d].T @ p
        if self.stab == "supg":
            E = self.supg_extra(U, p)
            loc = np.einsum("eq,eqa,eqd->ead", self.W * tau[:, None], wgrad, E)
            for d in range(3):
                ru[:, d] -= np.bincount(self.cells2.ravel(), loc[:, :, d].ravel(), self.n2)
        rp = -(self.B[0] @ u[:, 0] + self.B[1] @ u[:, 1] + self.B[2] @ u[:, 2])
        return ru, rp

    def _pack(self, ru, rp):
        return np.concatenate([ru[self.free_c[c], c] for c in range(3)] + [rp])

    def jacobian(self, u, a=0.0, newton=True, p=None):
        """d(K(u) x)/dx (+ rho*a*M on the velocity blocks) on the free unknowns, as one CSR matrix.
        p: the pressure at which the Newton matrix is taken ("supg" only: its residual depends on p non-linearly
        through tau(u) and the test function u.grad v); default the stored pressure."""
        U, uq, wgrad, sp_q, s, tau, al = self._kin(u)
        Aloc = self._picard_local(wgrad, tau, a)
        supg = self.stab == "supg"
        if supg:
            Aloc = Aloc + self._supg_visc_local(wgrad, tau)
            Gp = self.supg_pressure_blocks(wgrad, tau)
            E = self.supg_extra(U, self.p if p is None else p)
        fc = self.free_c
        blocks = [[None] * 4 for _ in range(4)]
        if newton:
            dudx = np.einsum("eci,eqcj->eqij", U, self.dphi)                 # (ne, nq, 3, 3): d u_i / d x_j
            S = np.einsum("eqj,eqij->eqi", uq, dudx)                         # (u.grad) u_i
            test = self.phi[None] + tau[:, None, None] * wgrad               # (ne, nq, 10)
            Wr = self.W * self.rho
            RS = self.rho * S + E if supg else self.rho * S                  # the residual the test tau*(u.grad v) multiplies
            g = np.einsum("eq,eqa,eqi->eai", self.W, wgrad, RS)              # d(stabilisation term)/d tau
            ds = np.einsum("eqj,qb->ebj", uq / np.maximum(sp_q, 1e-300)[:, :, None], self.phi) / self.phi.shape[0]
            dtau = -(tau ** 3) * al * s / (self.upwind ** 2) if self.upwind else np.zeros_like(tau)
        for i in range(3):
            for j in range(3):
                loc = Aloc if i == j else None
                if newton:
                    C = (Wr * dudx[:, :, i, j])[:, :, None] * test + (self.W * tau[:, None] * RS[:, :, i])[:, :, None] * self.dphi[:, :, :, j]
                    N = np.matmul(C.transpose(0, 2, 1), self.phi[None])      # (ne, 10, 10)
                    N += g[:, :, i][:, :, None] * (dtau[:, None, None] * ds[:, :, j][:, None, :])
                    loc = N + Aloc if i == j else N
                    del C, N
                if loc is not None:
                    blocks[i][j] = self.asm22.matrix(loc)[fc[i]][:, fc[j]]
            blocks[i][3] = ((-self.B[i].T + Gp[i]) if supg else -self.B[i].T).tocsr()[fc[i]]
            blocks[3][i] = self.B[i][:, fc[i]]
        blocks[3][3] = sp.csr_matrix((self.n1, self.n1))
        return sp.bmat(blocks, format="csr")

    def _apply(self, u, p, dx):
        u, p = u.copy(), p.copy()
        o = 0
        for c in range(3):
            k = len(self.free_c[c])
            u[self.free_c[c], c] += dx[o:o + k]
            o += k
        p += dx[o:]
        return u, p

    def _norm(self, ru, rp, u_ref):
        """Size of the residual on the free unknowns: momentum in N, continuity made comparable by rho * u_ref."""
        return float(np.sqrt(sum((ru[self.free_c[c], c] ** 2).sum() for c in range(3)) + ((self.rho * u_ref * rp) ** 2).sum()))

    def linear_step(self, u, p, ru, rp, a=0.0, newton=True, info=None):
        t0 = time.time()
        J = self.jacobian(u, a, newton, p=p)
        if info is not None:
            info.update(unknowns=J.shape[0], matrix_nonzeros=int(J.nnz), assemble_seconds=time.time() - t0)
        t0 = time.time()
        lu = factorize(J)
        dx = lu(self._pack(ru, rp))
        if hasattr(lu, "free"):
            lu.free()
        if info is not None:
            info["factor_seconds"] = time.time() - t0
        del J
        return dx

    # ------------------------------------------------------- monitors
    def kinetic_energy(self, u=None):
        u = self.u if u is None else u
        return 0.5 * self.rho * float(sum(u[:, c] @ (self.M @ u[:, c]) for c in range(3)))

    def face_pressure(self, faces, p=None):
        p = self.p if p is None else p
        return float((p[faces["v"]].mean(1) * faces["area"]).sum() / faces["area"].sum()) if faces is not None else 0.0

    def face_flow(self, faces, u=None):
        """Volume flow out through a set of faces (exact for the quadratic velocity)."""
        u = self.u if u is None else u
        return float(np.einsum("fmd,fd->", u[faces["m"]], faces["n"]) / 3.0) if faces is not None else 0.0

    def pressure_drop(self, u=None, p=None):
        return self.face_pressure(self.inlet_faces, p) - self.face_pressure(self.outlet_faces, p)

    def probe_nodes(self, n=12):
        """Velocity nodes spread through the region (nearest to n points picked by farthest-point sampling)."""
        X = self.xyz[~self.fixed_u.any(1)]
        ids = np.where(~self.fixed_u.any(1))[0]
        pick = [int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))]
        d = np.linalg.norm(X - X[pick[0]], axis=1)
        for _ in range(n - 1):
            k = int(d.argmax())
            pick.append(k)
            d = np.minimum(d, np.linalg.norm(X - X[k], axis=1))
        return ids[pick]

    # ---------------------------------------------------------- steady
    def steady(self, tol=1e-7, tol_u=1e-6, max_steps=80, cfl0=20.0, growth=4.0, dt_max_transits=1e6, restart=False,
               log=None, deadline=None, history=None, scale=1.0, callback=None):
        """Pseudo-time stepping to a solution of the stationary equations.

        Starts from the Stokes solution (or from the stored field if restart).  cfl0 = inf starts with pure
        Newton steps (for a restart close to the solution).  Converged when the residual is below tol times the
        momentum flux of the inlet, rho * U^2 * A, and the largest velocity change of the last step is below tol_u
        of the largest speed.  Returns a dict:
        status "steady" | "no_steady" | "diverged" | "time_limit", steps, residual (relative to the first), and
        the history rows [step, pseudo time step s, residual, relative residual, largest velocity change / largest
        speed, largest speed, pressure drop Pa, step length (0 = rejected), factor seconds]."""
        log = log or (lambda s: None)
        hist = history if history is not None else []
        self.set_inflow(scale)                      # scale < 1: a reduced flow rate, as a stepping stone (restart from it)
        u, p = self.u.copy(), self.p.copy()
        u[self.fixed_u] = self.value_u[self.fixed_u]
        u_ref = max(self.inlet_mean_speed * scale, 1e-12)
        info = {}
        if not restart:
            u[~self.fixed_u] = 0.0
            p[:] = 0.0
            ru, rp = self.residual(u, p)
            dx = self.linear_step(u, p, ru, rp, newton=False, info=info)             # Stokes flow
            u, p = self._apply(u, p, dx)
            log(f"    flow problem: {info['unknowns']} unknowns, matrix {info['matrix_nonzeros']} non-zeros, "
                f"first factorisation {info['factor_seconds']:.1f} s")
        ru, rp = self.residual(u, p)
        r = r0 = self._norm(ru, rp, u_ref)
        r_ref = self.rho * u_ref ** 2 * (float(self.inlet_faces["area"].sum()) if self.inlet_faces is not None else float(np.median(self.h)) ** 2)
        transit = self.vol.sum() / max(self.inflow * scale, 1e-300)
        dt_max = dt_max_transits * transit
        dt = min(cfl0 * float(np.median(self.h)) / u_ref, dt_max)
        out = dict(status="no_steady", steps=0, rejected=0, first_residual=r0, reference_residual=r_ref, scale=scale)
        k = rejected = 0
        r0 = max(r0, 1e-300)
        log(f"    start: residual {r0:.3e} ({r0 / r_ref:.2e} of the inlet momentum flux), pseudo time step "
            f"{'inf' if dt >= dt_max else '%.3g ms' % (dt * 1e3)} (channel transit {transit:.3f} s)")
        if r0 < tol * r_ref:
            out["status"] = "steady"
            max_steps = 0
        h_over_u = float(np.median(self.h)) / u_ref
        while k < max_steps:
            if deadline and time.time() > deadline:
                out["status"] = "time_limit"
                break
            a = 0.0 if dt >= dt_max else 1.0 / dt
            dx = self.linear_step(u, p, ru, rp, a=a, info=info)
            # step length: the full step unless it raises the residual by more than a factor two (Newton's method
            # often passes through a larger residual on its way; demanding a decrease at every step stalls it);
            # otherwise a half or a quarter step that does not raise it (no new factorisation needed)
            lam, rn, un, pn = 1.0, np.inf, u, p
            good = False
            if np.isfinite(dx).all():
                for lam in (1.0, 0.5, 0.25):
                    un, pn = self._apply(u, p, lam * dx)
                    run, rpn = self.residual(un, pn)
                    rn = self._norm(run, rpn, u_ref)
                    if np.isfinite(rn) and rn <= (2.0 if lam == 1.0 else 1.0) * r:
                        good = True
                        break
            if not good:
                rejected += 1
                new = 100.0 * h_over_u if a == 0.0 else 0.25 * dt
                hist.append([k, dt, rn, rn / r_ref, np.nan, np.nan, np.nan, 0, info.get("factor_seconds", np.nan)])
                log(f"    step {k + 1}: rejected (residual {r:.2e} -> {rn:.2e} even at a quarter of the step), pseudo time step "
                    f"{'inf' if a == 0.0 else '%.3g' % (dt * 1e3)} -> {new * 1e3:.3g} ms")
                dt = new
                if dt < 1e-4 * h_over_u or rejected > 25:
                    out["status"] = "diverged"
                    break
                continue
            k += 1
            umax = float(np.linalg.norm(un, axis=1).max())
            du = float(np.abs(un - u).max()) / max(umax, 1e-300)
            u, p, ru, rp = un, pn, run, rpn
            dp = self.pressure_drop(u, p)
            hist.append([k, dt, rn, rn / r_ref, du, umax, dp, lam, info.get("factor_seconds", np.nan)])
            log(f"    step {k}: dt {'inf' if a == 0 else '%.3g ms' % (dt * 1e3)}" + (f", step length {lam:.3g}" if lam < 1 else "")
                + f", residual {rn:.3e} ({rn / r_ref:.2e} of the inlet momentum flux), "
                f"velocity change {du:.2e}, max speed {umax:.4f} m/s, pressure drop {dp:.2f} Pa, "
                f"assemble {info['assemble_seconds']:.1f} s, factor and solve {info['factor_seconds']:.1f} s")
            if callback:
                callback(k, u, p)
            if rn < tol * r_ref and du < tol_u:
                out["status"] = "steady"
                r = rn
                break
            # the pseudo time step grows with the fall of the residual, and by at least 1.5 per accepted full step
            dt = min(dt * float(np.clip(r / max(rn, 1e-300), 1.5 if lam == 1.0 else 1.0, growth)), dt_max)
            r = rn
        self.u, self.p = u, p
        out.update(steps=k, rejected=rejected, residual=r, relative_residual=r / r_ref,
                   unknowns=info.get("unknowns"), matrix_nonzeros=info.get("matrix_nonzeros"),
                   factor_seconds=info.get("factor_seconds"), pressure_drop_Pa=self.pressure_drop(),
                   max_speed=float(np.linalg.norm(u, axis=1).max()),
                   assemble_seconds=info.get("assemble_seconds"),
                   outflow_over_inflow=self.face_flow(self.outlet_faces) / (self.inflow * scale) if self.outlet_faces is not None else None)
        return out

    def continuation(self, first=0.1, factor=1.6, stage_steps=25, max_steps=200, start=None, log=None, deadline=None,
                     history=None, callback=None, **kw):
        """To the steady solution at full flow through steady solutions at reduced flow rates.

        The flow rate is raised from `first` times the full value by `factor` per stage; each stage starts from
        the previous solution scaled to the new flow rate (pure Newton first, pseudo-time steps if that is
        rejected).  A stage that does not converge in stage_steps is tried again with half the increase; when the
        increase falls below 0.5 % of the full flow the steady branch is given up.  start = (scale, u, p) resumes.
        history rows get the flow-rate scale as first column.  Returns the dict of the last stage with
        status "steady" (at full flow) | "no_steady" | "time_limit", `reached` (largest converged scale), total steps."""
        log = log or (lambda s: None)
        hist = history if history is not None else []
        done, sc, steps, rej = 0.0, first, 0, 0
        keep = None
        if start is not None:
            done, keep = float(start[0]), (start[1].copy(), start[2].copy())
            sc = min(done * factor, 1.0)
        res = dict(status="no_steady")
        ratio = factor                                                      # current increase per stage; halved (in excess of 1) after a failure
        while True:
            sc = min(sc, 1.0)
            if keep is not None:
                self.u, self.p = keep[0] * (sc / done), keep[1] * (sc / done) ** 2
            rows = []
            log(f"   flow x {sc:.4g}" + (f" (from the solution at x {done:.4g})" if done else " (from the Stokes solution)"))
            opts = dict(kw)
            if keep is not None:
                opts.update(cfl0=np.inf)
            res = self.steady(scale=sc, restart=keep is not None, max_steps=min(stage_steps, max(max_steps - steps, 1)), log=log,
                              deadline=deadline, history=rows, **opts)
            hist.extend([[sc] + x for x in rows])
            steps += res["steps"]
            rej += res["rejected"]
            if res["status"] == "steady":
                done, keep = sc, (self.u.copy(), self.p.copy())
                if callback:
                    callback(done, self.u, self.p)
                if sc >= 1.0:
                    break
                if res["steps"] <= 6 and res["rejected"] == 0:
                    ratio = min(factor, 1.0 + 1.5 * (ratio - 1.0))          # easy stage: lengthen the next one
                sc = done * ratio
            elif res["status"] == "time_limit" or steps >= max_steps:
                break
            else:
                if keep is None:                                            # even the first stage failed: start lower
                    sc = sc / 4.0
                    if sc < 1e-3:
                        break
                else:
                    ratio = 1.0 + 0.5 * (ratio - 1.0)
                    sc = done * ratio
                    if sc - done < 0.005:
                        break
        if keep is not None and res["status"] != "steady":
            self.u, self.p = keep                                           # leave the last converged field
            self.set_inflow(done)
        res = dict(res, reached=done, steps=steps, rejected=rej)
        if res["status"] != "time_limit" and done < 1.0:
            res["status"] = "no_steady"
        return res

    # ----------------------------------------------------- time accurate
    def march(self, steps_per_transit=200, transits=6.0, min_transits=2.0, window_transits=1.0, perturb=0.0, ramp_transits=0.0,
              log=None, deadline=None, every=20, T_glob=None, seed=0):
        """Second-order time stepping at a fixed step (convecting velocity extrapolated; one factorisation per step).

        Starts from the stored field (perturb > 0 adds a divergence-free-ish random disturbance of that relative
        size to the free velocities, to see whether a steady solution holds) or, with ramp_transits > 0, from
        rest with the inlet raised smoothly.  Records the monitors every step and keeps the mean field of the last
        window.  Stops when `classify` says steady or unsteady in two successive windows, or at `transits`.
        Returns (result dict, monitor array, names); self.u, self.p are left at the mean of the last window
        if the flow is unsteady, else at the last step."""
        log = log or (lambda s: None)
        T_glob = self._T if T_glob is None else T_glob
        transit = self.vol.sum() / self.inflow
        dt = transit / steps_per_transit
        probes = self.probe_nodes()
        names = ["t_s", "pressure_drop_Pa", "kinetic_energy_J", "max_speed_m_s", "relative_change"] + [f"speed_probe_{i}" for i in range(len(probes))]
        if ramp_transits > 0:
            self.u[~self.fixed_u] = 0.0
            self.p[:] = 0.0
            self.set_inflow(0.0)
        elif perturb > 0:
            rng = np.random.default_rng(seed)
            free = ~self.fixed_u
            speed = np.linalg.norm(self.u, axis=1)                          # disturbance scaled with the local speed
            self.u[free] += perturb * rng.standard_normal(int(free.sum())) * (speed[:, None] * np.ones((1, 3)))[free]
        rows, u_old, t = [], None, 0.0
        nwin = max(int(round(window_transits * steps_per_transit)), 8)
        acc_u, acc_p, nacc = np.zeros_like(self.u), np.zeros_like(self.p), 0
        mean_u = mean_p = None
        verdicts = []
        res = dict(status="not_settled", time_step_s=dt, transit_s=transit)
        nstep = 0
        nmax = int(round(transits * steps_per_transit))
        while nstep < nmax:
            if deadline and time.time() > deadline:
                res["stopped"] = "time_limit"
                break
            t += dt
            if ramp_transits > 0:
                s = min(t / (ramp_transits * transit), 1.0)
                self.set_inflow(3 * s * s - 2 * s ** 3)
            un = self.u.copy()
            if u_old is None:
                a, h_ = 1.0 / dt, -un / dt
            else:
                a, h_ = 1.5 / dt, (-2.0 * un + 0.5 * u_old) / dt
                free = ~self.fixed_u
                self.u[free] = (2.0 * un - u_old)[free]
            self.step(T_glob, self.spec["t_ref"], None, a, h_, upwind=self.upwind)
            u_old = un
            nstep += 1
            sp_ = np.linalg.norm(self.u, axis=1)
            umax = float(sp_.max())
            rel = float(np.abs(self.u - un).max()) / max(umax, 1e-300) * steps_per_transit      # change per transit / max speed
            rows.append([t, self.pressure_drop(), self.kinetic_energy(), umax, rel] + sp_[probes].tolist())
            acc_u += self.u
            acc_p += self.p
            nacc += 1
            if log and (nstep % every == 0 or nstep == 1):
                log(f"    t = {t:8.4f} s ({t / transit:5.2f} transits)  pressure drop {rows[-1][1]:9.3f} Pa  max speed {umax:.4f} m/s  "
                    f"change per transit {rel:.2e}")
            if nstep % nwin == 0:
                mean_u, mean_p = acc_u / nacc, acc_p / nacc
                acc_u[:], acc_p[:], nacc = 0.0, 0.0, 0
                if t >= max(min_transits, ramp_transits + window_transits) * transit - 0.5 * dt:      # (2026-10-15: tolerance; the
                    # accumulated time fell a rounding error short of the first window and that window was skipped)
                    v = classify(np.array(rows)[-nwin:, 0], np.array(rows)[-nwin:, 1:])
                    verdicts.append(v)
                    log(f"    window ending {t / transit:.2f} transits: {v['status']}, pressure-drop amplitude {v['amplitude_relative']:.2e} "
                        f"of its mean, period {v['period_s']} s, drift {v['drift_relative']:.2e} per window")
                    if len(verdicts) >= 2 and verdicts[-1]["status"] == verdicts[-2]["status"] != "not_settled":
                        if verdicts[-1]["status"] == "steady" or abs(verdicts[-1]["amplitude_relative"] - verdicts[-2]["amplitude_relative"]) \
                                < 0.3 * verdicts[-1]["amplitude_relative"]:
                            res.update(verdicts[-1])
                            break
        if verdicts and res["status"] == "not_settled":
            res.update({k: v for k, v in verdicts[-1].items() if k != "status"})
        if res["status"] == "unsteady" and len(rows) >= 3 * nwin:
            # amplitude and period from the last three windows together (more cycles than one window holds).
            # (2026-10-22) never reaching back over the start-up ramp: the wide window starts at least one
            # transit after the end of the ramp.  Before, a march from rest (ramp_transits 1, 3 transits) measured over
            # the ramp from zero: amplitude about 9x too large, mean pressure drop 14-21 % low, period None.  Stored
            # STATUS values of earlier runs are recomputed from monitors.csv by dev/wb_fluct_check.py.
            arr = np.array(rows)
            t_ok = (ramp_transits + window_transits) * transit - 0.5 * dt if ramp_transits > 0 else -np.inf
            first = max(len(arr) - 3 * nwin, int(np.searchsorted(arr[:, 0], t_ok)))
            last = arr[first:]
            if len(last) >= nwin:
                wide = classify(last[:, 0], last[:, 1:])
                res.update(amplitude_Pa=wide["amplitude_Pa"], amplitude_relative=wide["amplitude_relative"], period_s=wide["period_s"],
                           mean_pressure_drop_Pa=wide["mean_pressure_drop_Pa"],
                           measured_over=f"the last {len(last) / nwin:.2f} windows, after the ramp + 1 transit" if ramp_transits > 0
                           else "the last three windows")
        res.update(steps=nstep, time_s=t, transits=t / transit)
        if res["status"] != "unsteady" and len(verdicts) >= 2 and all(v["status"] == "unsteady" for v in verdicts[-2:]):
            res["status"] = "unsteady"                               # oscillating, with an amplitude that still changes between windows
            res["note"] = "amplitude not yet constant from window to window"
        elif res["status"] != "unsteady" and len(verdicts) == 1 and verdicts[0]["status"] == "unsteady"                 and verdicts[0]["drift_relative"] < 0.1 * verdicts[0]["amplitude_relative"]:
            res["status"] = "unsteady"                               # (2026-10-15) one window only, oscillating without drift
            res["note"] = "one window only"
        # the last instantaneous state (velocity, velocity one step before, pressure): dev/wb_heat_td.py goes on from it
        self.last_state = (self.u.copy(), None if u_old is None else u_old.copy(), self.p.copy(), dt)
        if res["status"] == "unsteady" and mean_u is not None:
            self.u, self.p = mean_u, mean_p
            self.u[self.fixed_u] = self.value_u[self.fixed_u]
            res["field"] = "mean of the last window"
        else:
            res["field"] = "last step"
        return res, np.array(rows), names


def classify(t, y, steady_tol=1e-4, amp_tol=1e-3):
    """Read a window of monitor series (columns: pressure drop, kinetic energy, max speed, relative change, probes...).

    steady:      the pressure drop and the kinetic energy each stay within steady_tol of their mean (peak to peak / 2);
    unsteady:    one of them oscillates by more than amp_tol of its mean and the oscillation is not a drift
                 (the straight-line trend over the window explains less than half of the range);
    not_settled: anything else (still drifting towards a state).
    Returns status, amplitude (half the peak-to-peak of the detrended pressure drop, absolute and relative),
    period (dominant period of the detrended pressure drop, from its spectrum; None if under two cycles), drift."""
    t = np.asarray(t, float)
    out = dict(status="not_settled", amplitude_Pa=0.0, amplitude_relative=0.0, period_s=None, drift_relative=0.0,
               kinetic_energy_amplitude_relative=0.0)
    amps, drifts = [], []
    for col in (0, 1):
        s = np.asarray(y[:, col], float)
        mean = abs(s.mean()) or 1.0
        A = np.stack([t - t.mean(), np.ones_like(t)], 1)
        coef = np.linalg.lstsq(A, s, rcond=None)[0]
        det = s - A @ coef
        amps.append(0.5 * np.ptp(det) / mean)
        drifts.append(abs(coef[0]) * np.ptp(t) / mean)
        if col == 0:
            out["amplitude_Pa"] = float(0.5 * np.ptp(det))
            out["mean_pressure_drop_Pa"] = float(s.mean())
            f = np.abs(np.fft.rfft(det * np.hanning(len(det))))
            if len(f) > 3 and f[1:].max() > 0:
                k = int(f[1:].argmax()) + 1
                if k >= 2:                                         # at least two cycles in the window
                    out["period_s"] = float(np.ptp(t) * (len(t) / (len(t) - 1)) / k)
    out["amplitude_relative"], out["kinetic_energy_amplitude_relative"] = float(amps[0]), float(amps[1])
    out["drift_relative"] = float(max(drifts))
    # a regular oscillation of the pressure drop (2026-10-15): upward zero crossings of the detrended series at
    # intervals that agree within 1 %.  Shedding whose amplitude still grows drifts in the kinetic energy and was
    # called "not settled"; it is called "unsteady" with a note.
    s0 = np.asarray(y[:, 0], float)
    A0 = np.stack([t - t.mean(), np.ones_like(t)], 1)
    det0 = s0 - A0 @ np.linalg.lstsq(A0, s0, rcond=None)[0]
    up = np.where((det0[:-1] < 0) & (det0[1:] >= 0))[0]
    regular = False
    if len(up) >= 4:
        tc = t[up] - det0[up] * (t[up + 1] - t[up]) / (det0[up + 1] - det0[up])          # crossing times, interpolated
        per = np.diff(tc)
        out["period_spread"] = float(per.std() / per.mean())
        regular = out["period_spread"] < 0.01
        h = len(det0) // 2
        growth = float(np.ptp(det0[h:]) / max(np.ptp(det0[:h]), 1e-300))
        out["amplitude_growth"] = growth
    if max(amps) < steady_tol and max(drifts) < steady_tol:
        out["status"] = "steady"
    elif max(amps) > amp_tol and max(drifts) < max(amps):
        out["status"] = "unsteady"
    elif regular and amps[0] > amp_tol:
        out["status"] = "unsteady"
        out["note"] = "regular oscillation, amplitude still growing" if out.get("amplitude_growth", 1.0) > 1.05 else "regular oscillation with drift"
    return out
