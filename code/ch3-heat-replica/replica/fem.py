"""Linear (P1) tetrahedral finite elements: the discretisation COMSOL used (temperature order: linear)."""
import numpy as np
import scipy.sparse as sp

SIGMA = 5.670374419e-8


def tet_geometry(points, tets):
    """Volumes (ne) and basis-function gradients (ne, 4, 3)."""
    x = points[tets]
    J = x[:, 1:] - x[:, :1]                     # rows are edge vectors
    det = np.linalg.det(J)
    inv = np.linalg.inv(J)                      # columns are gradients of lambda_1..3
    g = np.empty((len(tets), 4, 3))
    g[:, 1:] = np.transpose(inv, (0, 2, 1))
    g[:, 0] = -g[:, 1:].sum(1)
    return np.abs(det) / 6.0, g


MASS_REF = (np.ones((4, 4)) + np.eye(4)) / 20.0      # consistent mass, times volume
TRI_MASS_REF = (np.ones((3, 3)) + np.eye(3)) / 12.0  # times area


class Assembler:
    """Assemble element matrices with a fixed sparsity pattern.

    The pattern is computed once; matrices for new coefficients are then a
    single bincount.
    """

    def __init__(self, cells, n):
        m = cells.shape[1]
        r = np.repeat(cells, m, axis=1).ravel().astype(np.int64)
        c = np.tile(cells, (1, m)).ravel().astype(np.int64)
        key, self.inv = np.unique(r * n + c, return_inverse=True)
        self.n = n
        rows = key // n
        self.indices = (key % n).astype(np.int32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))]).astype(np.int32)
        self.nnz = len(key)

    def matrix(self, local):
        """local: (ne, m, m) element matrices."""
        data = np.bincount(self.inv, weights=local.ravel(), minlength=self.nnz)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))


def load(cells, values, n):
    """Lumped nodal load from (ne, m) element-node contributions."""
    return np.bincount(cells.ravel(), weights=values.ravel(), minlength=n)


def tri_area(points, tris):
    x = points[tris]
    return 0.5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1)


def locate(points, tets, xyz, tol=1e-9):
    """Element containing each point and its barycentric coordinates."""
    from scipy.spatial import cKDTree
    x = points[tets]
    tree = cKDTree(x.mean(1))
    out = []
    for p in np.atleast_2d(xyz):
        best = None
        _, cand = tree.query(p, k=64)
        for e in cand:
            A = (x[e, 1:] - x[e, 0]).T
            l = np.linalg.solve(A, p - x[e, 0])
            lam = np.concatenate([[1 - l.sum()], l])
            worst = lam.min()
            if best is None or worst > best[2]:
                best = (e, lam, worst)
            if worst >= -tol:
                break
        out.append(best)
    return out
