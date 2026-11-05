"""Forced laminar flow with quadratic velocity and linear pressure (Taylor-Hood elements).

Used for the water block, where the coolant is fast compared with the element
size.  With this pair of elements the discrete velocity conserves mass against
every linear test function, and the temperature (linear) is one of those, so
the heat carried by the flow balances exactly: what enters with the coolant
plus what the walls give equals what leaves.  The equal-order element in
``flow.py`` does not have that property.

Stationary incompressible Navier-Stokes,
    rho (u.grad)u = -grad p + div(mu grad u),   div u = 0,
(plus the Boussinesq buoyancy force where the density depends on temperature),
with no-slip walls, a prescribed volume flow at inlets and zero pressure at
outlets, solved by a few fixed-point (Picard) iterations followed by Newton iterations.  A streamline-diffusion
term keeps the iteration stable when convection dominates on the mesh.
"""
import numpy as np
import scipy.sparse as sp

from . import fem
from .flow import outward_normals, _not_converged
from .solve import factorize

# 14-point rule, exact for polynomials of degree 5 (weights sum to 1)
_a1, _w1 = 0.0927352503108912264, 0.01224884051939366 * 6
_a2, _w2 = 0.3108859192633006, 0.01878132095300264 * 6
_b, _w3 = 0.0455037041256496, 0.00709100346284691 * 6
QP = np.array([[1 - 3 * a if i == k else a for i in range(4)] for a in (_a1, _a2) for k in range(4)]
              + [[_b if i in pair else 0.5 - _b for i in range(4)]
                 for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
QW = np.array([_w1] * 4 + [_w2] * 4 + [_w3] * 6)
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def p2_values(lam):
    """Quadratic basis values at barycentric points lam (nq, 4) -> (nq, 10)."""
    v = [lam[:, i] * (2 * lam[:, i] - 1) for i in range(4)] + [4 * lam[:, i] * lam[:, j] for i, j in EDGES]
    return np.stack(v, 1)


def p2_gradients(lam, g):
    """Gradients of the quadratic basis: (ne, nq, 10, 3) from barycentric gradients g (ne, 4, 3)."""
    out = np.empty((g.shape[0], lam.shape[0], 10, 3))
    for i in range(4):
        out[:, :, i] = (4 * lam[:, i] - 1)[None, :, None] * g[:, None, i]
    for k, (i, j) in enumerate(EDGES):
        out[:, :, 4 + k] = 4 * (lam[:, i][None, :, None] * g[:, None, j] + lam[:, j][None, :, None] * g[:, None, i])
    return out


class TaylorHood:
    """One fluid region.  Mirrors the interface of flow.Region where the heat solver needs it."""

    def __init__(self, prob, spec, vol, grad):
        self.spec = spec
        el = np.where(np.isin(prob.dom, spec["domains"]))[0]
        self.el = el
        self.nodes, loc = np.unique(prob.tets[el], return_inverse=True)
        cells = loc.reshape(-1, 4)
        self.n1 = n1 = len(self.nodes)
        self.vol, self.g = vol[el], grad[el]
        self.h = (12.0 * self.vol / np.sqrt(2.0)) ** (1.0 / 3.0)
        self.rho = spec["rho_ref"]
        # edge (mid-side) unknowns
        e = np.sort(np.stack([cells[:, [i, j]] for i, j in EDGES], 1), axis=2)           # (ne, 6, 2)
        key = e[:, :, 0].astype(np.int64) * n1 + e[:, :, 1]
        uk, inv = np.unique(key, return_inverse=True)
        self.edge_nodes = np.stack([uk // n1, uk % n1], 1)
        self.n2 = n2 = n1 + len(uk)
        self.cells1 = cells
        self.cells2 = np.concatenate([cells, n1 + inv.reshape(-1, 6)], 1)                 # (ne, 10)
        self.asm22 = fem.Assembler(self.cells2, n2)
        self.phi = p2_values(QP)                                                          # (nq, 10)
        self.dphi = p2_gradients(QP, self.g)                                              # (ne, nq, 10, 3)
        self.u = np.zeros((n2, 3))
        self.p = np.zeros(n1)
        glob = {int(a): i for i, a in enumerate(self.nodes)}
        P = prob.points

        # ---- boundary faces of the region
        tris, up, down = prob.tris, prob.tri_up, prob.tri_down
        inside = np.isin(up, spec["domains"]).astype(int) + np.isin(down, spec["domains"]).astype(int)
        surface = inside == 1
        is_in = np.isin(prob.tri_bnd, [b for bl, _ in spec["inlets"] for b in bl])
        is_out = np.isin(prob.tri_bnd, spec["outlets"])
        is_sym = np.isin(prob.tri_bnd, spec.get("symmetry", []))
        edge_id = self.edge_of = {(int(a), int(b)): n1 + i for i, (a, b) in enumerate(self.edge_nodes)}

        def face_dofs(face_tris):
            """Local vertex and mid-side unknowns of a set of boundary triangles: (nf, 3), (nf, 3)."""
            v = np.array([[glob[int(a)] for a in t] for t in face_tris], int).reshape(-1, 3)
            m = np.array([[edge_id[tuple(sorted((r[i], r[j])))] for i, j in ((0, 1), (1, 2), (0, 2))] for r in v], int).reshape(-1, 3)
            return v, m

        wall_v, wall_m = face_dofs(tris[surface & ~is_in & ~is_out & ~is_sym])
        wall = np.unique(np.concatenate([wall_v.ravel(), wall_m.ravel()]))
        fixed = np.zeros((n2, 3), bool)
        value = np.zeros((n2, 3))
        st = tris[surface & is_sym]
        if len(st):                                                                       # mirror plane: no flow through it
            x = P[st]
            nn = np.abs(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])).sum(0)
            ax = int(nn.argmax())
            if not np.allclose(np.delete(nn, ax), 0, atol=1e-9 * nn[ax]):
                raise ValueError("mirror plane is not a coordinate plane")
            sv, sm = face_dofs(st)
            fixed[np.unique(np.concatenate([sv.ravel(), sm.ravel()])), ax] = True
        rt = prob.tets[el]
        self.inflow = 0.0
        for bl, U0 in spec["inlets"]:
            ft = tris[surface & np.isin(prob.tri_bnd, bl)]
            if not len(ft):
                continue
            nn = -outward_normals(P, rt, ft)                                              # area vectors into the fluid
            area = np.linalg.norm(nn, axis=1)
            q_target = U0[1] / self.rho if isinstance(U0, tuple) else U0 * area.sum()     # volume flow
            fv, fm = face_dofs(ft)
            unit = (nn / area[:, None]).mean(0)
            unit /= np.linalg.norm(unit)
            idx = np.setdiff1d(np.unique(np.concatenate([fv.ravel(), fm.ravel()])), wall)
            fixed[idx] = True
            value[idx] = unit
            # a quadratic field integrates over a triangle as (area/3) * (sum of its mid-side values)
            q = sum((a / 3.0) * (value[m] @ (n_ / a)).sum() for a, n_, m in zip(area, nn, fm))
            value[idx] *= q_target / q
            self.inflow += q_target
        fixed[wall] = True
        value[wall] = 0.0
        self.fixed, self.value = fixed, value
        ot = tris[surface & is_out]
        self.outlet_tris = ot
        self.p_fix = None if len(ot) else 0
        self.free = np.concatenate([np.where(~fixed[:, c])[0] + c * n2 for c in range(3)]
                                   + [3 * n2 + (np.arange(n1) if len(ot) else np.arange(1, n1))])
        self.n, self.fixed_u, self.value_u = n2, fixed, value
        self.u[fixed] = value[fixed]
        # divergence blocks (pressure test x velocity trial), constant in the iteration
        lam_q = QP                                                                        # linear basis values = lam
        w = QW[None, :] * self.vol[:, None]                                               # (ne, nq)
        r1 = np.repeat(self.cells1, 10, axis=1).ravel()
        c2 = np.tile(self.cells2, (1, 4)).ravel()
        self.B = []
        for d in range(3):
            loc_b = np.einsum("eq,qi,eqa->eia", w, lam_q, self.dphi[:, :, :, d])          # (ne, 4, 10)
            self.B.append(sp.csr_matrix((loc_b.ravel(), (r1, c2)), shape=(n1, n2)))
        self.lap = np.einsum("eq,eqad,eqbd->eab", w, self.dphi, self.dphi)                # (ne, 10, 10)
        self.w = w

    # ------------------------------------------------------------------
    def solve(self, T_glob, T_ref=None, gvec=None, a=None, hist=None, tol=1e-5, maxit=60, relax=1.0, upwind=1.0, picard=3, log=None,
              strict=False):
        """Stationary flow for the temperature field given (it sets the viscosity).

        (2026-10-28) Ending at maxit is no longer silent: see replica.flow._not_converged.

        A few fixed-point (Picard) iterations, then Newton iterations."""
        n1, n2 = self.n1, self.n2
        Te = T_glob[self.nodes][self.cells1].mean(1)
        mu = self.spec["mu"](Te) if callable(self.spec["mu"]) else np.full(len(self.vol), self.spec["mu"])
        rho = self.rho
        visc = mu[:, None, None] * self.lap
        # buoyancy (Boussinesq): body force -rho*beta*(T - T_ref)*g, with the temperature linear on each element
        F = np.zeros(3 * n2 + n1)
        beta = self.spec.get("beta", 0.0)
        if beta and gvec is not None:
            Tq = np.einsum("qi,ei->eq", QP, T_glob[self.nodes][self.cells1]) - T_ref
            fe = -rho * beta * np.einsum("eq,qa->ea", self.w * Tq, self.phi)
            for d in range(3):
                if gvec[d]:
                    F[d * n2:(d + 1) * n2] = np.bincount(self.cells2.ravel(), (fe * gvec[d]).ravel(), n2)
        for it in range(maxit):
            uq = np.einsum("qa,ead->eqd", self.phi, self.u[self.cells2])                  # (ne, nq, 3)
            wgrad = np.einsum("eqd,eqad->eqa", uq, self.dphi)                             # u.grad(phi_a)
            conv = rho * np.einsum("eq,qa,eqb->eab", self.w, self.phi, wgrad)
            speed = np.linalg.norm(uq, axis=2).mean(1)
            he = self.h / 2.0                                                             # node spacing of the quadratic element
            tau = upwind / np.sqrt((2 * speed / he) ** 2 + (4 * (mu / rho) / he ** 2) ** 2)
            sd = rho * tau[:, None, None] * np.einsum("eq,eqa,eqb->eab", self.w, wgrad, wgrad)
            A = self.asm22.matrix(visc + conv + sd)
            Z = None
            K = sp.bmat([[A, Z, Z, -self.B[0].T], [Z, A, Z, -self.B[1].T], [Z, Z, A, -self.B[2].T],
                         [self.B[0], self.B[1], self.B[2], sp.csr_matrix((n1, n1))]], format="csr")
            x = np.concatenate([self.u.T.ravel(), self.p])
            r = F - K @ x
            if it >= picard:
                # Newton step: add the other half of the linearised convection, rho (du.grad) u
                dudx = np.einsum("eci,eqcj->eqij", self.u[self.cells2], self.dphi)        # (ne, nq, 3, 3)
                pp = np.einsum("qa,qb->qab", self.phi, self.phi)
                N = [[self.asm22.matrix(rho * np.einsum("eq,qab,eq->eab", self.w, pp, dudx[:, :, i, j]))
                      for j in range(3)] for i in range(3)]
                K = K + sp.bmat([N[0] + [None], N[1] + [None], N[2] + [None],
                                 [None, None, None, sp.csr_matrix((n1, n1))]], format="csr")
                del N, dudx
            lu = factorize(K[self.free][:, self.free])
            dx = lu(r[self.free])
            lu.free() if hasattr(lu, "free") else None
            if not np.isfinite(dx).all():
                raise FloatingPointError("flow iteration diverged")
            # limit the step: no velocity may change by more than half the largest speed
            nu_free = int((~self.fixed_u).sum())
            big = np.abs(dx[:nu_free]).max()
            cap = 0.5 * max(np.abs(self.u).max(), 1e-12)
            x[self.free] += relax * min(1.0, cap / big if big > 0 and it > 0 else 1.0) * dx
            u_new = x[:3 * n2].reshape(3, n2).T
            change = np.abs(u_new - self.u).max()
            scale = max(np.abs(u_new).max(), 1e-12)
            self.u, self.p = u_new, x[3 * n2:]
            if log:
                log(f"    flow iteration {it + 1}: max speed {np.linalg.norm(u_new, axis=1).max():.4f} m/s, change {change:.2e}")
            self.last_change, self.last_scale = change, scale
            if change < tol * scale:
                self.converged = True
                return it + 1
        return _not_converged(self, maxit, tol, log, strict)

    # ------------------------------------------------------------------
    def heat_convection(self, rc, k, upwind=True):
        """Element matrices (ne, 4, 4) of rho*cp*(u.grad T) for linear temperature, with streamline upwinding."""
        uq = np.einsum("qa,ead->eqd", self.phi, self.u[self.cells2])
        ug = np.einsum("eqd,ejd->eqj", uq, self.g)                                        # u.grad(lam_j) at the points
        rc = rc if np.ndim(rc) else np.full(len(self.vol), rc)
        k = k if np.ndim(k) else np.full(len(self.vol), k)
        loc = rc[:, None, None] * np.einsum("eq,qi,eqj->eij", self.w, QP, ug)
        if not upwind:
            return loc
        speed = np.linalg.norm(uq, axis=2).mean(1)
        tau = 1.0 / np.sqrt((2 * speed / self.h) ** 2 + (4 * k / (rc * self.h ** 2)) ** 2)
        return loc + (tau * rc)[:, None, None] * np.einsum("eq,eqi,eqj->eij", self.w, ug, ug)

    def flows(self, prob):
        """Volume flow leaving through the outlets [m^3/s] (for the mass balance)."""
        if not len(self.outlet_tris):
            return 0.0
        glob = {int(a): i for i, a in enumerate(self.nodes)}
        edge_id = {(int(a), int(b)): self.n1 + i for i, (a, b) in enumerate(self.edge_nodes)}
        nn = outward_normals(prob.points, prob.tets[self.el], self.outlet_tris)
        q = 0.0
        for t, n_ in zip(self.outlet_tris, nn):
            v = [glob[int(a)] for a in t]
            m = [edge_id[tuple(sorted((v[i], v[j])))] for i, j in ((0, 1), (1, 2), (0, 2))]
            q += (self.u[m] @ n_).sum() / 3.0
        return q
