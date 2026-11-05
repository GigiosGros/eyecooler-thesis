"""Partial-stage fix for `replica.solve.Heat.transient`, without editing replica/*.py.

The defect (the 2026-10-08 exact_solutions check): `Heat.transient` takes the number
of steps of a stage as round((t_until - t) / dt) and never checks the end time, so a stage that is not a
whole number of steps ends early or late ([(10, 4)] ends at 8 s, [(10, 3), (30, 6)] at 33 s).

The fix rewrites the stage list before it reaches `Heat.transient`: a stage that is not a whole number of
steps from where the previous one ended is split in two stages, n = floor((t_until - t) / dt) steps of the
requested dt, then one step of the remainder r = t_until - t - n dt, so it ends on t_until.  If r < dt / 2 and
n >= 1 the remainder is merged into the last whole step instead (n - 1 steps of dt, then one of dt + r), so the
last step always lies between dt / 2 and 3 dt / 2 (2026-10-22, before, r had no lower bound, and a
remainder of, say, 1e-4 dt followed by the next stage's step gave a step ratio far above 1 + sqrt 2).  A stage
shorter than one step (n = 0) is still one step of its own length.
The requested step and its grid (t + k dt, where a kink of a boundary condition such as the end of a ramp
usually lies) are kept up to the last one or two steps. (Spreading the stage evenly over ceil(.) equal steps
was tried first and rejected: on [(95, 6)] it moved the step grid off the 90 s end of the test ramp and the
error at 95 s was 0.095 C, against 0.056 C with this split; test 2 of dev/test_transient_fixed.py.) A stage that already is a whole
number of steps is passed on unchanged (same tuple, same float), so the solver performs bit-for-bit the
same arithmetic as before for every such list. The BDF2 formula of `Heat.transient` handles a change of step
with w = dt_new / dt_old and is zero-stable for w < 1 + sqrt 2.  The last step of a split stage has
w = 0.5 to 1.5 against dt; the FIRST step of the following stage has w = dt_next / last step, at most
2 dt_next / dt, which can exceed 1 + sqrt 2 (it already does without this fix: production 15 s after 6 s,
w = 2.5; release_steps 3 s after 1 s, w = 3).  The earlier text here ("a shorter step has w < 1") left out
that following step.

Use either
    T = transient_exact(h, T0, t_end, stages, ...)            # h: a replica.solve.Heat
or the subclass HeatExact (same constructor as Heat), or rewrite a list with exact_stages().

Also: old_end_time() reproduces where the unpatched code ends, for scanning case definitions.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from replica import solve  # noqa: E402

RTOL = 1e-6          # a stage counts as whole when its steps land within RTOL * dt of its end


def _stage_ends(stages, t_end):
    """(t_until clamped as Heat.transient does, dt) for each stage."""
    return [(min(tu, t_end), dt) for tu, dt in stages]


def old_end_time(stages, t_end, t0=0.0):
    """Where the unpatched Heat.transient ends, and the per-stage (t_from, t_until, dt, nst, t_reached)."""
    t, rows = t0, []
    for tu, dt in _stage_ends(stages, t_end):
        nst = int(round((tu - t) / dt))
        t_new = t + max(nst, 0) * dt
        rows.append((t, tu, dt, nst, t_new))
        t = t_new
    return t, rows


def exact_stages(stages, t_end, t0=0.0):
    """Stage list with every non-whole stage split into whole steps of dt and one shorter last step.

    Returns (new_stages, changed) where changed lists (t_from, t_until, dt, n full steps, last step); the last
    step lies in [dt / 2, 3 dt / 2) whenever the stage holds at least one whole step."""
    out, changed, t = [], [], t0
    for (tu_raw, dt), (tu, _) in zip(stages, _stage_ends(stages, t_end)):
        span = tu - t
        if span <= RTOL * dt:                        # nothing left in this stage (or it lies behind t)
            out.append((tu_raw, dt))
            continue
        nst = int(round(span / dt))
        if nst >= 1 and abs(nst * dt - span) <= RTOL * dt:
            out.append((tu_raw, dt))                 # whole: unchanged, identical arithmetic in the solver
        else:
            n = int(math.floor(span / dt + RTOL))
            r = span - n * dt
            if n >= 1 and r < 0.5 * dt:              # merge a short remainder into the last whole step
                n -= 1
                r += dt
            if n >= 1:
                out.append((t + n * dt, dt))         # whole part, the requested step
            out.append((tu_raw, r))                  # one step of the remainder (round(r / r) = 1 in the solver)
            changed.append((t, tu, dt, n, r))
        t = tu
    return out, changed


def transient_exact(h, T0, t_end, stages, **kw):
    """Heat.transient with exact stage ends (see module text)."""
    new, changed = exact_stages(stages, t_end)
    for t, tu, dt, n, dn in changed:
        h.log(f"  stage {t:g}-{tu:g} s is not a whole number of {dt:g} s steps: {n} steps, then one of {dn:.6g} s")
    if new and min(new[-1][0], t_end) < t_end - RTOL * new[-1][1]:
        h.log(f"  note: the stages end at {new[-1][0]:g} s, before t_end = {t_end:g} s (as in Heat.transient)")
    return solve.Heat.transient(h, T0, t_end, new, **kw)


class HeatExact(solve.Heat):
    """replica.solve.Heat whose transient() ends every stage exactly on its t_until."""

    def transient(self, T0, t_end, stages, step="time", callback=None, tol=1e-7):
        return transient_exact(self, T0, t_end, stages, step=step, callback=callback, tol=tol)
