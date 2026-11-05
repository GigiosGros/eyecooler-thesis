"""Laminar flow in the fluid domains, coupled to the heat equation (COMSOL: Laminar Flow + Nonisothermal Flow).

Incompressible Navier-Stokes with the Boussinesq buoyancy term, for velocity
u and reduced pressure p (the hydrostatic part rho_ref*g.x is taken out):

    rho_ref (du/dt + (u.grad)u) = -grad p + div(mu grad u) - rho_ref*beta*(T - T_ref)*g
    div u = 0

with no-slip walls, zero normal velocity on the mirror plane, a prescribed
normal inflow velocity on inlets and zero pressure on outlets.  rho_ref and
beta = -(1/rho) d(rho)/dT come from the material's density function at the
reference temperature, so a material with constant density has no buoyancy.

Discretisation: linear velocity and linear pressure on the same tetrahedra
(COMSOL's P1+P1), stabilised with streamline-upwind and pressure-stabilising
terms (SUPG/PSPG).  COMSOL's own streamline/crosswind stabilisation uses a
different formula, so the two discrete flow fields are not bit-identical.
"""
import numpy as np
import scipy.sparse as sp

from . import fem
from .solve import factorize


class FlowNotConverged(RuntimeError):
    """A flow solve ended at its iteration limit with strict = True."""


def _not_converged(region, maxit, tol, log, strict):
    """Record (and with strict raise) a flow solve that used all maxit iterations without meeting tol.

    Fewer iterations than needed can be intended (the transient does a fixed few linearised solves per time step and
    converges the coupling in its own loop), so by default this only records: region.converged = False, region.n_maxit
    += 1, and the worst relative change seen, region.worst_rel_change = max(last_change / last_scale)."""
    region.converged = False
    region.n_maxit = getattr(region, "n_maxit", 0) + 1
    rel = getattr(region, "last_change", float("nan")) / max(getattr(region, "last_scale", 1.0), 1e-300)
    region.worst_rel_change = max(getattr(region, "worst_rel_change", 0.0), rel)
    msg = f"    flow solve stopped at maxit = {maxit}: relative change {rel:.2e} > tol {tol:.1e}"
    if log:
        log(msg)
    if strict:
        raise FlowNotConverged(msg.strip())
    return maxit


def outward_normals(points, tets, faces):
    """Area-weighted normal of each boundary face (n, 3), pointing out of the element that owns it."""
    n = int(tets.max()) + 1

    def key(f):
        f = np.sort(f.astype(np.int64), axis=1)
        return (f[:, 0] * n + f[:, 1]) * n + f[:, 2]
    combos = ([0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0])
    fk = np.concatenate([key(tets[:, c[:3]]) for c in combos])
    opp = np.concatenate([tets[:, c[3]] for c in combos])
    order = np.argsort(fk, kind="stable")
    fk, opp = fk[order], opp[order]
    pos = np.searchsorted(fk, key(faces))
    x = points[faces]
    nn = 0.5 * np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    inward = points[opp[pos]] - x.mean(1)
    nn[np.einsum("ij,ij->i", nn, inward) > 0] *= -1
    return nn


class Region:
    """One connected fluid region (e.g. the air, or the aqueous humour)."""

    def __init__(self, prob, spec, vol, grad):
        self.spec = spec
        el = np.where(np.isin(prob.dom, spec["domains"]))[0]
        self.el = el
        self.nodes, loc = np.unique(prob.tets[el], return_inverse=True)
        self.cells = loc.reshape(-1, 4)
        self.n = n = len(self.nodes)
        self.vol, self.g = vol[el], grad[el]
        self.h = (12.0 * self.vol / np.sqrt(2.0)) ** (1.0 / 3.0)      # edge of the regular tetrahedron of equal volume
        self.asm = fem.Assembler(self.cells, n)
        self.rho = spec["rho_ref"]
        self.u = np.zeros((n, 3))
        self.u_prev = None
        self.p = np.zeros(n)
        glob = {int(g): i for i, g in enumerate(self.nodes)}
        to_local = lambda arr: np.array([glob[int(a)] for a in np.unique(arr) if int(a) in glob], int)

        # ---- boundary conditions on the velocity
        tris, up, down = prob.tris, prob.tri_up, prob.tri_down
        inside = np.isin(up, spec["domains"]).astype(int) + np.isin(down, spec["domains"]).astype(int)
        surface = inside == 1                                         # faces bounding the region
        is_sym = np.isin(prob.tri_bnd, spec["symmetry"])
        is_in = np.isin(prob.tri_bnd, [b for bl, _ in spec["inlets"] for b in bl])
        is_out = np.isin(prob.tri_bnd, spec["outlets"])
        wall = to_local(tris[surface & ~is_sym & ~is_in & ~is_out])
        fixed = np.zeros((n, 3), bool)
        value = np.zeros((n, 3))
        sym_nodes = to_local(tris[surface & is_sym])
        normals = np.zeros((n, 3))
        P = prob.points
        for t in tris[surface & is_sym]:
            x = P[t]
            nn = np.cross(x[1] - x[0], x[2] - x[0])
            for a in t:
                normals[glob[int(a)]] += nn
        if len(sym_nodes):
            ax = np.abs(normals[sym_nodes]).sum(0).argmax()
            if not np.allclose(np.abs(normals[sym_nodes]).sum(0)[[i for i in range(3) if i != ax]], 0, atol=1e-12):
                raise ValueError("mirror plane is not a coordinate plane")
            fixed[sym_nodes, ax] = True
        # inlets: normal inflow velocity U0, direction from the area-weighted nodal normal
        inlet_faces = []
        rt = prob.tets[el]
        ot = tris[surface & is_out]
        self.outlet_tris = ot                                         # global node numbers
        self.outlet_normals = outward_normals(P, rt, ot) if len(ot) else np.zeros((0, 3))
        self.u_sub = np.zeros((len(el), 3))                           # see solve(): velocity correction per element
        for bl, U0 in spec["inlets"]:
            ft = tris[surface & np.isin(prob.tri_bnd, bl)]
            nrm = np.zeros((n, 3))
            fn = -outward_normals(P, rt, ft) if len(ft) else np.zeros((0, 3))     # pointing into the fluid
            for t, nn in zip(ft, fn):
                for a in t:
                    nrm[glob[int(a)]] += nn
            idx = to_local(ft)
            if not len(idx):
                continue
            if isinstance(U0, tuple):                                 # ("mass_flow", kg/s) -> uniform normal velocity
                area = sum(0.5 * np.linalg.norm(np.cross(P[t][1] - P[t][0], P[t][2] - P[t][0])) for t in ft)
                U0 = U0[1] / (spec["rho_ref"] * area)
            nrm[idx] /= np.linalg.norm(nrm[idx], axis=1)[:, None]
            fixed[idx] = True
            value[idx] = U0 * nrm[idx]
            inlet_faces.append((ft, fn, U0))
        fixed[wall] = True
        value[wall] = 0.0
        # The nodes on the rim of an inlet are wall nodes (no slip), so the
        # discrete inflow is smaller than U0 x area, markedly so on a coarse
        # inlet.  Scale the inlet velocities so the volume flow is exact.
        for ft, fn, U0 in inlet_faces:
            q = area = 0.0
            for t, nn in zip(ft, fn):
                q += value[[glob[int(a)] for a in t]].mean(0) @ nn
                area += np.linalg.norm(nn)
            idx = np.setdiff1d(to_local(ft), wall)
            if q > 0 and len(idx):
                value[idx] *= U0 * area / q
        self.fixed_u, self.value_u = fixed, value
        # pressure level: fixed at one node unless an outlet sets it
        self.p_fix = None if spec["outlets"] else int(np.setdiff1d(np.arange(n), wall)[0])
        dof_fixed = np.concatenate([fixed.T.ravel(), np.zeros(n, bool)])
        if self.p_fix is not None:
            dof_fixed[3 * n + self.p_fix] = True
        self.free = np.where(~dof_fixed)[0]
        self.u[fixed] = value[fixed]
        self.forced = bool(spec["beta"] != 0.0 or spec["inlets"])

    # ------------------------------------------------------------------
    def solve(self, T_glob, T_ref, gvec, a=None, hist=None, tol=1e-6, maxit=25, relax=1.0, log=None, strict=False):
        """One (time) step: Picard iteration on the convecting velocity.

        (2026-10-28) Ending at maxit without meeting tol is no longer silent: the region records
        converged = False, last_change / last_scale and counts it in n_maxit, logs it when log is given, and raises
        FlowNotConverged when strict = True.  The return value is unchanged (iterations used).

        a: BDF coefficient of the new velocity (None for a stationary solve);
        hist: (n, 3) the remaining part of the discrete time derivative.
        """
        n, V, g, cells = self.n, self.vol, self.g, self.cells
        Te = T_glob[self.nodes][cells].mean(1)
        mu = self.spec["mu"](Te) if callable(self.spec["mu"]) else np.full(len(V), self.spec["mu"])
        rho = self.rho
        # buoyancy force density at the nodes
        fb = -rho * self.spec["beta"] * (T_glob[self.nodes] - T_ref)[:, None] * np.asarray(gvec)[None, :]
        f_e = fb[cells].mean(1)                                        # (ne, 3)
        if a is not None:
            f_e = f_e - rho * hist[cells].mean(1)
        a0 = 0.0 if a is None else a
        mass = fem.MASS_REF[None] * V[:, None, None]
        lap = np.einsum("eik,ejk->eij", g, g) * V[:, None, None]
        Mmat = self.asm.matrix(mass)
        for it in range(maxit):
            w = self.u[cells].mean(1)                                  # element convecting velocity
            wn = np.linalg.norm(w, axis=1)
            nu = mu / rho
            tau = 1.0 / np.sqrt((2 * a0) ** 2 + (2 * wn / self.h) ** 2 + (4 * nu / self.h ** 2) ** 2)
            wg = np.einsum("ek,eik->ei", w, g)                         # w.grad(phi_i)
            conv = (V / 4)[:, None, None] * np.ones((1, 4, 1)) * wg[:, None, :]          # phi_i w.grad(phi_j)
            supg = tau[:, None, None] * rho * (a0 * (V / 4)[:, None, None] * wg[:, :, None]
                                               + V[:, None, None] * wg[:, :, None] * wg[:, None, :])
            A_uu = self.asm.matrix(rho * a0 * mass + rho * conv + mu[:, None, None] * lap + supg)
            blocks = [[None] * 4 for _ in range(4)]
            rhs = np.zeros(4 * n)
            for d in range(3):
                blocks[d][d] = A_uu
                grad_p = -(V / 4)[:, None, None] * g[:, :, d][:, :, None] * np.ones((1, 1, 4))      # -phi_j d(phi_i)
                supg_p = tau[:, None, None] * V[:, None, None] * wg[:, :, None] * g[:, :, d][:, None, :]
                blocks[d][3] = self.asm.matrix(grad_p + supg_p)
                div = (V / 4)[:, None, None] * np.ones((1, 4, 1)) * g[:, :, d][:, None, :]         # phi_i d(phi_j)
                pspg_u = tau[:, None, None] * g[:, :, d][:, :, None] * (a0 * (V / 4)[:, None, None]
                                                                        + V[:, None, None] * wg[:, None, :])
                blocks[3][d] = self.asm.matrix(div + pspg_u)
                rhs[d * n:(d + 1) * n] = Mmat @ fb[:, d] + fem.load(cells, (tau * V * f_e[:, d])[:, None] * wg, n)
                if a is not None:
                    rhs[d * n:(d + 1) * n] -= rho * (Mmat @ hist[:, d])
            blocks[3][3] = self.asm.matrix((tau / rho)[:, None, None] * lap)
            rhs[3 * n:] = fem.load(cells, (tau / rho * V)[:, None] * np.einsum("eik,ek->ei", g, f_e), n)
            A = sp.bmat(blocks, format="csr")
            x = np.concatenate([self.u.T.ravel(), self.p])
            x[:3 * n][self.fixed_u.T.ravel()] = self.value_u.T.ravel()[self.fixed_u.T.ravel()]
            if self.p_fix is not None:
                x[3 * n + self.p_fix] = 0.0
            r = rhs - A @ x
            lu = factorize(A[self.free][:, self.free])
            dx = lu(r[self.free])
            if hasattr(lu, "free"):
                lu.free()
            x[self.free] += (relax if it < 6 else 0.5 * relax) * dx      # damp if plain iteration is slow
            u_new = x[:3 * n].reshape(3, n).T
            change = np.abs(u_new - self.u).max()
            scale = max(np.abs(u_new).max(), 1e-12)
            self.u, self.p = u_new, x[3 * n:]
            # The pressure-stabilised continuity equation says that u - (tau/rho)*r, with r the
            # momentum residual, is the field that conserves mass against every test function.
            # The heat equation must be convected by that field or it does not balance.
            ue = self.u[cells]
            res = rho * (a0 * ue.mean(1) + np.einsum("ei,eik->ek", wg, ue)) + np.einsum("ei,eik->ek", self.p[cells], g) - f_e
            self.u_sub = -(tau / rho)[:, None] * res
            if log:
                log(f"    flow iteration {it + 1}: max |u| {scale:.3e} m/s, change {change:.2e}")
            self.last_change, self.last_scale = change, scale
            if change < tol * scale or scale < 1e-14:
                self.converged = True
                return it + 1
        return _not_converged(self, maxit, tol, log, strict)

    def element_velocity(self):
        return self.u[self.cells].mean(1)
