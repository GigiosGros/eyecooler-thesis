"""Linear solves with a check (added 2026-10-14, simulation plan, section 9a).

Why: the verification of 2026-10-14 (results/waterblock_research_2026-10-13/verification, problem 3) found that
PARDISO, called through replica.solve.factorize with pypardiso's defaults, often returns a solution of the
time-step systems of the flow (saddle point: velocity and pressure, zero pressure block) that does not satisfy
the system: relative residual |A x - b| / |b| up to 1e5 where SuperLU gives 1e-15 on the same system.  Nothing
in the time stepping checked the residual, so a wrong solve became a velocity spike.

`factorize(A)` returns a solver with the same calling convention as replica.solve.factorize (call it with b,
call .free() at the end) that checks every solution:

  1. PARDISO with explicit settings for this kind of matrix (iparm: weighted matching and scaling on, pivot
     perturbation 1e-8, up to 20 refinement steps inside PARDISO), then iterative refinement with the same
     factors (x <- x + A^-1 (b - A x), up to `refine` times) until |A x - b| / |b| < tol;
  2. if that fails: PARDISO with pypardiso's defaults (as replica.solve.factorize), then the same refinement;
  3. if that fails: SuperLU (scipy) on the same matrix;
  4. if that fails: RuntimeError (never a silent wrong answer).

Counters of what was needed are kept in STATS (read and logged by the callers).  replica/solve.py is not changed.
"""
import numpy as np
import scipy.sparse as sp

STATS = dict(solves=0, refined=0, pardiso_default=0, superlu=0, worst_first_residual=0.0, worst_final_residual=0.0)


def _pardiso(A, explicit=False):
    import pypardiso
    s = pypardiso.PyPardisoSolver()
    if explicit:
        # 1-based numbers as in the MKL documentation; iparm(1) = 1: the values below are used, not the defaults
        for i, v in ((1, 1), (2, 2), (8, 20), (10, 8), (11, 1), (13, 1), (24, 0), (25, 0), (35, 0)):
            s.set_iparm(i, v)
    s.factorize(A)
    f = lambda b: s.solve(A, b)
    f.free = lambda: s.free_memory(everything=True)
    return f


def _superlu(A):
    from scipy.sparse.linalg import splu
    lu = splu(sp.csc_matrix(A))
    f = lambda b: lu.solve(b)
    f.free = lambda: None
    return f


def factorize(A, tol=1e-9, refine=8):
    A = sp.csr_matrix(A)
    if not A.has_sorted_indices:
        A.sort_indices()
    state = dict(f=_pardiso(A, explicit=True), stage=0)

    def rel(x, b, nb):
        return float(np.linalg.norm(A @ x - b) / nb)

    def attempt(f, b, nb):
        try:
            x = f(b)
        except Exception as e:                                        # (2026-10-22) e.g. PyPardisoError: try the next stage
            STATS["solver_errors"] = STATS.get("solver_errors", 0) + 1
            STATS["last_solver_error"] = repr(e)[:200]
            return np.full_like(b, np.nan, dtype=float), np.inf, np.inf, 0
        r0 = rel(x, b, nb) if np.isfinite(x).all() else np.inf
        r = r0
        k = 0
        while r > tol and k < refine and np.isfinite(r):
            x = x + f(b - A @ x)
            r = rel(x, b, nb) if np.isfinite(x).all() else np.inf
            k += 1
        return x, r0, r, k

    def solve(b):
        STATS["solves"] += 1
        nb = max(float(np.linalg.norm(b)), 1e-300)
        x, r0, r, k = attempt(state["f"], b, nb)
        STATS["worst_first_residual"] = max(STATS["worst_first_residual"], r0)
        if k:
            STATS["refined"] += 1
        while not (r <= tol) and state["stage"] < 2:                  # (2026-10-22) a NaN residual no longer passes
            try:
                state["f"].free()
            except Exception:
                pass
            state["stage"] += 1
            try:
                if state["stage"] == 1:
                    STATS["pardiso_default"] += 1
                    state["f"] = _pardiso(A)
                else:
                    STATS["superlu"] += 1
                    state["f"] = _superlu(A)
            except Exception as e:                                    # (2026-10-22) factorisation failed: next stage
                STATS["solver_errors"] = STATS.get("solver_errors", 0) + 1
                STATS["last_solver_error"] = repr(e)[:200]
                state["f"] = lambda b_: (_ for _ in ()).throw(RuntimeError("factorisation failed"))
                state["f"].free = lambda: None
                continue
            x, _, r, _ = attempt(state["f"], b, nb)
        STATS["worst_final_residual"] = max(STATS["worst_final_residual"], r)
        if not (r <= tol):
            raise RuntimeError(f"linear solve failed: relative residual {r:.2e} after PARDISO with explicit settings, PARDISO with the defaults and SuperLU")
        return x

    solve.free = lambda: state["f"].free()
    return solve
