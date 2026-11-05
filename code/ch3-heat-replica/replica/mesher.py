"""Our own tetrahedral meshes of a model, at any resolution, and self-refining meshes.

A new mesh is made by adapting an existing one to a map of target element
sizes with the MMG library (package ``mmgpy``): elements are split, collapsed,
swapped and moved until their edges have the requested length.  The interfaces
between domains and the boundary faces are kept, with their numbers, so
materials and boundary conditions apply to the new mesh unchanged.  MMG treats
the surfaces as smooth and keeps new surface nodes within a set distance
(``surface_tol``) of them.

Two uses:
  * a family of meshes for a convergence study: the element size of the
    reference mesh times a factor (``reference_size`` x factor);
  * self-refining meshes: the size map comes from an error indicator computed
    on the previous solution (``error_indicator`` then ``adapted_size``).

The geometry is whatever the starting mesh describes; this module does not
read CAD files.  (Meshing the STEP file directly was tried with Gmsh and fails
on four surfaces of the CAD model; see the README.)
"""
import numpy as np

from . import fem


def reference_size(mesh):
    """Element size at each node of a mesh: the mean length of the edges that meet there [m]."""
    P, t = mesh["points"], mesh["tets"]
    e = np.concatenate([t[:, [a, b]] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    L = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)
    n = len(P)
    s = np.bincount(e.ravel(), np.concatenate([L, L]), n)
    return s / np.maximum(np.bincount(e.ravel(), minlength=n), 1)


def adapt(mesh, size, gradation=1.4, surface_tol=None, verbose=-1):
    """Adapt ``mesh`` to the nodal target edge length ``size`` [m]; returns a new mesh dict.

    mesh: dict with points [m], tets, tet_dom, tris, tri_bnd (as from the .mph reader).
    gradation: largest ratio between the lengths of neighbouring edges.
    surface_tol: largest distance of a new surface facet from the smooth surface [m]
                 (default: a fifth of the smallest requested size).
    """
    import mmgpy
    from .mphread import _neighbours
    P = np.ascontiguousarray(mesh["points"] * 1e3)                 # MMG works in the model's length unit: use mm
    sz = np.asarray(size, float) * 1e3
    mm = mmgpy.MmgMesh3D()
    mm.set_mesh_size(vertices=len(P), tetrahedra=len(mesh["tets"]), triangles=len(mesh["tris"]))
    mm.set_vertices(P)
    mm.set_tetrahedra(np.ascontiguousarray(mesh["tets"].astype(np.int32)), mesh["tet_dom"].astype(np.int64))
    mm.set_triangles(np.ascontiguousarray(mesh["tris"].astype(np.int32)), mesh["tri_bnd"].astype(np.int64))
    mm["metric"] = sz.reshape(-1, 1)
    tol = float(sz.min()) / 5 if surface_tol is None else surface_tol * 1e3
    mm.remesh(hgrad=gradation, hausd=tol, hmin=float(sz.min()) * 0.7, hmax=float(sz.max()) * 1.5, verbose=verbose)
    v = mm.get_vertices()
    tets, dom = mm.get_elements_with_refs()
    tris, bnd = mm.get_triangles_with_refs()
    # keep only the faces that carry a boundary number of the model
    keep = np.isin(bnd, np.unique(mesh["tri_bnd"]))
    tris, bnd = tris[keep], bnd[keep]
    x = v[tets]
    neg = np.linalg.det(x[:, 1:] - x[:, :1]) < 0
    tets = tets.copy()
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    dom = dom.astype(np.int32)
    up, down = _neighbours(tets, dom, tris)
    return dict(points=v * 1e-3, tets=tets.astype(np.int32), tet_dom=dom, tris=tris.astype(np.int32),
                tri_bnd=bnd.astype(np.int32), tri_up=up, tri_down=down, version=0)


def quality(mesh):
    """Per-element size [m] and quality (3 x inscribed radius / circumscribed radius; 1 = regular, 0 = flat)."""
    x = mesh["points"][mesh["tets"]]
    a, b, c = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6
    area = sum(0.5 * np.linalg.norm(np.cross(x[:, i] - x[:, k], x[:, j] - x[:, k]), axis=1)
               for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    r_in = 3 * vol / area
    num = np.linalg.norm((a * a).sum(1)[:, None] * np.cross(b, c) + (b * b).sum(1)[:, None] * np.cross(c, a)
                         + (c * c).sum(1)[:, None] * np.cross(a, b), axis=1)
    r_out = num / np.maximum(12 * vol, 1e-300)
    return (12 * vol / np.sqrt(2)) ** (1 / 3), 3 * r_in / r_out


# ------------------------------------------------------------------ adaptivity
def error_indicator(prob, T):
    """Flux-recovery error indicator of a temperature field: (eta per element, element volume).

    The heat flux q = -k grad(T) is constant in each element.  A smoother flux
    is recovered at the nodes by volume-weighted averaging within each domain
    (the flux may jump between materials), and the indicator of an element is
    the energy norm of the difference:  eta^2 = (V / k) * mean over its nodes of |q_rec - q_h|^2.
    """
    vol, g = fem.tet_geometry(prob.points, prob.tets)
    Te = T[prob.tets]
    grad = np.einsum("ei,eik->ek", Te, g)
    k = np.empty(len(vol))
    for d in np.unique(prob.dom):
        m = prob.dom == d
        kd = prob.k[d]
        k[m] = kd(Te[m].mean(1)) if callable(kd) else kd
    q = -k[:, None] * grad
    n = prob.points.shape[0]
    eta2 = np.zeros(len(vol))
    for d in np.unique(prob.dom):
        m = np.where(prob.dom == d)[0]
        t = prob.tets[m]
        w = np.bincount(t.ravel(), np.repeat(vol[m], 4), n)
        qr = np.stack([np.bincount(t.ravel(), np.repeat(vol[m] * q[m, c], 4), n) for c in range(3)], 1)
        qr /= np.maximum(w, 1e-300)[:, None]
        diff = qr[t] - q[m][:, None, :]
        eta2[m] = vol[m] / k[m] * (diff ** 2).sum(2).mean(1)
    return np.sqrt(eta2), vol


def adapted_size(prob, eta, vol, reduce=0.6, limits=(0.5, 1.5), h_min=None, h_max=None):
    """Nodal target size for the next mesh from the element indicators.

    The total error is asked to fall by the factor ``reduce`` and to be shared
    equally among the elements.  For linear elements in three dimensions the
    indicator of an element scales as h^2.5, which gives each element's new
    size; the change per pass is limited to ``limits``, and the result is
    passed on to the nodes (smallest neighbouring element).
    """
    h = (12 * vol / np.sqrt(2)) ** (1 / 3)
    total = np.sqrt((eta ** 2).sum())
    target = reduce * total / np.sqrt(len(eta))
    ratio = np.clip((target / np.maximum(eta, 1e-300)) ** (1 / 2.5), *limits)
    h_new = h * ratio
    if h_min is not None or h_max is not None:
        h_new = np.clip(h_new, h_min, h_max)
    size = np.full(prob.points.shape[0], np.inf)
    np.minimum.at(size, prob.tets.ravel(), np.repeat(h_new, 4))
    return size
