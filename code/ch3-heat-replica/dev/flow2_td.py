"""Time stepping for the quadratic-velocity flow element (dev version; replica/flow2.py is left alone).

TaylorHoodTD adds to replica.flow2.TaylorHood:

  * a time step: when the heat solver passes the coefficients of its time
    discretisation (a, hist), one linear system is solved per step, with the
    convecting velocity taken from the velocity the solver hands over (its
    extrapolation from the previous steps).  No inner iteration.
  * an inlet that can be ramped: `inflow_scale` multiplies the inlet velocity.
  * `march_to_steady`: time-steps the flow alone until it stops changing.
    This is the robust way to a steady forced flow when Newton's method on
    the stationary equations does not converge (water block at full flow).

The stationary solve (a is None) is inherited unchanged.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replica.flow2 import QP, TaylorHood
from robust_solve import factorize   # 2026-10-14: checked solves (PARDISO with explicit settings; see dev/robust_solve.py)


class TaylorHoodTD(TaylorHood):
    stab = "sd"          # "sd": streamline term only (as before); "supg": consistent, see dev/wb_flow.py (needs lapphi)

    def __init__(self, prob, spec, vol, grad):
        super().__init__(prob, spec, vol, grad)
        self.u_prev = None
        self.inflow_scale = 1.0
        self._inlet_value = self.value_u.copy()
        self.mass = np.einsum("eq,qa,qb->eab", self.w, self.phi, self.phi)             # (ne, 10, 10)
        self.M = self.asm22.matrix(self.mass)

    def set_inflow(self, scale):
        """Scale the inlet velocity (1 = the value in the model)."""
        self.inflow_scale = scale
        self.value_u = self._inlet_value * scale
        self.u[self.fixed_u] = self.value_u[self.fixed_u]

    def solve(self, T_glob, T_ref=None, gvec=None, a=None, hist=None, **kw):
        if a is None:
            return super().solve(T_glob, T_ref, gvec, **kw)
        return self.step(T_glob, T_ref, gvec, a, hist, upwind=kw.get("upwind", 1.0))

    def step(self, T_glob, T_ref, gvec, a, hist, upwind=1.0):
        """One time step: rho*(a*u + hist) + rho*(w.grad)u = -grad p + div(mu grad u) + f, div u = 0,
        with w the velocity currently stored (the caller's extrapolation)."""
        n1, n2, rho = self.n1, self.n2, self.rho
        Te = T_glob[self.nodes][self.cells1].mean(1)
        mu = self.spec["mu"](Te) if callable(self.spec["mu"]) else np.full(len(self.vol), self.spec["mu"])
        wq = np.einsum("qa,ead->eqd", self.phi, self.u[self.cells2])                    # convecting velocity
        wgrad = np.einsum("eqd,eqad->eqa", wq, self.dphi)
        conv = rho * np.einsum("eq,qa,eqb->eab", self.w, self.phi, wgrad)
        speed = np.linalg.norm(wq, axis=2).mean(1)
        he = self.h / 2.0
        tau = upwind / np.sqrt((2 * a) ** 2 + (2 * speed / he) ** 2 + (4 * (mu / rho) / he ** 2) ** 2)
        sd = rho * tau[:, None, None] * np.einsum("eq,eqa,eqb->eab", self.w, wgrad, wgrad)
        loc = rho * a * self.mass + mu[:, None, None] * self.lap + conv + sd
        supg = self.stab == "supg"
        if supg:
            # consistent SUPG: the rest of the strong residual, rho*(a u + hist) + grad p - mu lap(u) - f, tested with
            # tau (w.grad v); here the parts linear in the new u and p go to the matrix, hist and f to the right side
            wt = self.w * tau[:, None]
            t1 = np.einsum("eq,eqa->ea", wt, wgrad)
            loc = loc + rho * a * np.einsum("eq,eqa,qb->eab", wt, wgrad, self.phi) - (mu[:, None] * t1)[:, :, None] * self.lapphi()[:, None, :]
            G = self.supg_pressure_blocks(wgrad, tau)
        A = self.asm22.matrix(loc)
        Z = None
        PB = [(-self.B[d].T + G[d]) if supg else -self.B[d].T for d in range(3)]
        K = sp.bmat([[A, Z, Z, PB[0]], [Z, A, Z, PB[1]], [Z, Z, A, PB[2]],
                     [self.B[0], self.B[1], self.B[2], sp.csr_matrix((n1, n1))]], format="csr")
        F = np.zeros(3 * n2 + n1)
        beta = self.spec.get("beta", 0.0)
        for d in range(3):
            F[d * n2:(d + 1) * n2] = -rho * (self.M @ hist[:, d])
        if supg:
            hq = np.einsum("qa,ead->eqd", self.phi, hist[self.cells2])                    # history term at the points
            sl = -rho * np.einsum("eq,eqa,eqd->ead", wt, wgrad, hq)
            for d in range(3):
                F[d * n2:(d + 1) * n2] += np.bincount(self.cells2.ravel(), sl[:, :, d].ravel(), n2)
        if beta and gvec is not None:
            Tq = np.einsum("qi,ei->eq", QP, T_glob[self.nodes][self.cells1]) - T_ref
            fe = -rho * beta * np.einsum("eq,qa->ea", self.w * Tq, self.phi)
            if supg:
                fe = fe - rho * beta * np.einsum("eq,eqa->ea", wt * Tq, wgrad)        # f tested with tau (w.grad v)
            for d in range(3):
                if gvec[d]:
                    F[d * n2:(d + 1) * n2] += np.bincount(self.cells2.ravel(), (fe * gvec[d]).ravel(), n2)
        x = np.concatenate([self.u.T.ravel(), self.p])
        x[:3 * n2][self.fixed_u.T.ravel()] = self.value_u.T.ravel()[self.fixed_u.T.ravel()]
        r = F - K @ x
        lu = factorize(K[self.free][:, self.free])
        dx = lu(r[self.free])
        if hasattr(lu, "free"):
            lu.free()
        if not np.isfinite(dx).all():
            raise FloatingPointError("flow time step diverged")
        x[self.free] += dx
        self.u, self.p = x[:3 * n2].reshape(3, n2).T, x[3 * n2:]
        return 1

    def march_to_steady(self, T_glob, dt, ramp=0.0, t_max=60.0, tol=1e-4, T_ref=None, gvec=None, log=None, every=10):
        """Time-step the flow alone (second-order BDF) until the velocity changes by less than
        tol (relative to the largest speed) per unit time scale h/U; the inlet rises smoothly
        from the current scale to 1 over `ramp` seconds.  Returns (time reached, steady?)."""
        t, s0 = 0.0, self.inflow_scale if ramp > 0 else 1.0
        if ramp <= 0:
            self.set_inflow(1.0)
        u_old = None
        nstep = 0
        while t < t_max - 1e-12:
            t += dt
            if ramp > 0:
                s = min(t / ramp, 1.0)
                self.set_inflow(s0 + (1.0 - s0) * (3 * s * s - 2 * s ** 3))
            un = self.u.copy()
            if u_old is None:
                a, hist = 1.0 / dt, -un / dt
            else:
                a, hist = 1.5 / dt, (-2.0 * un + 0.5 * u_old) / dt
                free = ~self.fixed_u
                self.u[free] = (2.0 * un - u_old)[free]                                # extrapolated convecting velocity
            self.step(T_glob, T_ref, gvec, a, hist)
            u_old = un
            nstep += 1
            umax = np.linalg.norm(self.u, axis=1).max()
            rate = np.abs(self.u - un).max() / dt                                       # largest acceleration
            rel = rate * (np.median(self.h) / max(umax, 1e-12)) / max(umax, 1e-12)      # change over one transit of an element
            if log and (nstep % every == 0 or nstep == 1):
                log(f"    t = {t:7.3f} s  inlet x {self.inflow_scale:.3f}  max speed {umax:.4f} m/s  relative change {rel:.2e}")
            if t >= ramp and rel < tol:
                return t, True
        return t, False
