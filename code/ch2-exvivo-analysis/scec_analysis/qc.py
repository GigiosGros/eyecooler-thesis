"""Quality control: which channel traces enter the analysis.

Three objective gates, none of which looks at the outcome (cooling):
  1. hardware  — the raw voltage is flat-lined, or sits at an amplifier rail (either
                 polarity) for most of the run (disconnected or saturated sensor);
  2. fit       — the single-exponential fit to the filtered trace has R^2 < 0.85;
  3. spatial   — the sensor sits too far from where the cohort places it. Each
                 eye's sensor constellation is rigidly aligned (rotation +
                 translation, no scaling) to the cohort mean constellation, and a
                 sensor is flagged when its aligned distance from the mean position
                 exceeds a fixed fence in thermistor lengths (1 length = 3.3 mm,
                 the datasheet maximum of the sensing tip; an optional guard band,
                 qc.spatial_guard_band_mm, is added to the fence and is 0). Only the
                 four tissue-fixed sensors shape the alignment and have an offset fence
                 (qc.spatial_free_channels = ["Vitreous"]): the vitreous sensor floats
                 in the cavity, so its position says nothing about how the eye sat. It
                 is flagged instead when it does not lie in the vitreous cavity: its
                 micro-CT position is more than qc.vitreous_max_centre_fraction (0.75)
                 of the way from the centre of the segmented globe to its wall in
                 the same direction (config/vitreous_position_ct.csv). The earlier
                 rule, a minimum depth inside the convex hull of the globe
                 (qc.vitreous_min_depth_mm), is still available and is off. The alignment
                 removes how each eye sat in the micro-CT frame. qc.spatial_fence
                 = "upper"/"two-sided" restores the earlier Tukey fence on the
                 scale-free inter-sensor-distance measure (notebook v4.28). A
                 spatial breach takes precedence: the reason is recorded as "Space
                 Outlier" even if an earlier gate already failed the trace.
A temperature fence on the final temperature is computed for reporting only; it
never excludes a trace.

After the objective gates, a short documented list of manual exclusions
([[qc.manual_exclusions]] in the config) is applied to traces that passed them:
traces with an obvious transient artifact that the fit gate cannot catch. Each
entry carries its justification; qc.apply_manual = false switches the list off
(the reproduction test does this to match notebook v4.28).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .processing import fit_exponential


def tukey_fence(x, k: float = 1.5, min_n: int = 4):
    """(low, high, outside-mask) for the finite values of x; no fence below min_n."""
    x = np.asarray(x, float)
    fin = x[np.isfinite(x)]
    if len(fin) < min_n:
        return -np.inf, np.inf, np.zeros_like(x, bool)
    q1, q3 = np.percentile(fin, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return lo, hi, (x < lo) | (x > hi)


def hardware_broken(volts, q: dict) -> bool:
    v = pd.to_numeric(pd.Series(volts), errors="coerce").to_numpy()
    v = v[np.isfinite(v)]
    if len(v) < q["hardware_min_samples"]:
        return True
    railed = "hardware_railed_fraction" in q and np.mean(np.abs(v) > q["hardware_railed_V"]) > q["hardware_railed_fraction"]
    return bool(np.std(v) < q["hardware_flat_std_V"]
                or np.mean(np.abs(v - np.median(v)) < 1e-6) > q["hardware_stuck_fraction"]
                or railed)


def hardware_gate(runs: dict, cfg: dict):
    q, device = cfg["qc"], cfg["channels"]["device"]
    for r in runs.values():
        for ch in cfg["channels"]["temperature"]:
            if ch in device or ch not in r.volts.columns:
                continue
            if hardware_broken(r.volts[ch], q):
                r.exclusions[ch] = "Broken (hardware)"
    # Sensors documented as failing although the run itself passes the voltage test
    # ([[qc.hardware_exclusions]] in the config; each entry states its evidence).
    for item in q.get("hardware_exclusions", []):
        if item["run_id"] not in runs:
            raise KeyError(f"hardware exclusion: unknown run {item['run_id']}")
        runs[item["run_id"]].exclusions[item["channel"]] = "Broken (hardware)"


def fit_gate(runs: dict, cfg: dict):
    q = cfg["qc"]
    for r in runs.values():
        t = pd.to_numeric(pd.Series(r.filtered.index), errors="coerce").to_numpy()
        for ch in cfg["channels"]["temperature"]:
            if ch not in r.filtered.columns or r.exclusions.get(ch):
                continue
            y = pd.to_numeric(r.filtered[ch], errors="coerce").to_numpy()
            p = fit_exponential(t, y, q["fit_min_samples"])
            r.fit[ch] = p
            if p is None or p["R2"] < q["fit_r2_min"]:
                r.exclusions[ch] = "Broken (low R2)"


def constellation_deviation(eyes: dict) -> pd.DataFrame:
    """For every (eye, sensor): mean |d / d_mean - 1| over the sensor's distances to
    the other sensors, d_mean being the cohort mean of that pairwise distance.
    Scale-free and frame-invariant: it responds to the SHAPE of the constellation,
    not to where the eye sits or how large it is."""
    pw = {e: {frozenset(p): float(np.linalg.norm(pts[p[0]] - pts[p[1]]))
              for p in combinations(pts, 2)} for e, pts in eyes.items()}
    pairs = set().union(*[set(d) for d in pw.values()])
    mean_d = {pr: float(np.mean([d[pr] for d in pw.values() if pr in d])) for pr in pairs}
    channels = sorted({ch for pts in eyes.values() for ch in pts})
    centroid = {ch: np.mean([pts[ch] for pts in eyes.values() if ch in pts], axis=0) for ch in channels}
    rows = []
    for e, pts in eyes.items():
        for ch in pts:
            dev = [abs(pw[e][frozenset((ch, o))] / mean_d[frozenset((ch, o))] - 1.0)
                   for o in channels if o != ch
                   and frozenset((ch, o)) in pw[e] and mean_d.get(frozenset((ch, o)), 0) > 0]
            off = pts[ch] - centroid[ch]          # descriptive only (figures); not used by the fence
            rows.append({"marker_eye": e, "channel": ch,
                         "deviation": float(np.mean(dev)) if dev else np.nan,
                         "L": float(pts[ch][0]), "P": float(pts[ch][1]), "S": float(pts[ch][2]),
                         "dL": float(off[0]), "dP": float(off[1]), "dS": float(off[2])})
    return pd.DataFrame(rows)


def _rigid_transform(X, Y):
    """Rotation R and centroids (mx, my) that best map X onto Y in the least-squares
    sense (Kabsch: rotation + translation, no reflection, no scaling); a point p
    maps to (p - mx) @ R + my."""
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, mx, my


def _rigid_fit(X, Y):
    R, mx, my = _rigid_transform(X, Y)
    return (X - mx) @ R + my


def aligned_offsets(eyes: dict, iterations: int = 50, free=()) -> pd.DataFrame:
    """Generalised Procrustes (rigid) of every eye's constellation onto the cohort
    mean; returns each sensor's offset from the mean position in the mean's frame.
    Only sensors present in every eye of the cohort enter the alignment; the rest
    are placed with the eye's transform but do not shape it. Channels in `free`
    (sensors not fixed to tissue) never shape the alignment either; their offset is
    measured from the cohort mean of their aligned positions, for description."""
    common = sorted(set.intersection(*[set(p) for p in eyes.values()]) - set(free))
    X = {e: np.array([pts[c] for c in common]) for e, pts in eyes.items()}
    M = np.mean(list(X.values()), axis=0)
    for _ in range(iterations):
        M = np.mean([_rigid_fit(X[e], M) for e in eyes], axis=0)
    placed = {}
    for e, pts in eyes.items():
        R, mx, my = _rigid_transform(X[e], M)
        placed[e] = {ch: (np.asarray(xyz) - mx) @ R + my for ch, xyz in pts.items()}
    free_ref = {ch: np.mean([p[ch] for p in placed.values() if ch in p], axis=0) for ch in free}
    rows = []
    for e, pts in placed.items():
        for ch, a in pts.items():
            ref = M[common.index(ch)] if ch in common else free_ref.get(ch)
            off = a - ref if ref is not None else np.full(3, np.nan)
            rows.append({"marker_eye": e, "channel": ch, "aligned_mm": float(np.linalg.norm(off)),
                         "aL": float(off[0]), "aP": float(off[1]), "aS": float(off[2])})
    return pd.DataFrame(rows)


def fence_mm(channel: str, q: dict) -> float:
    """Decision limit for a channel: thermistor length x number of lengths, plus the
    measurement-uncertainty guard band (qc.spatial_guard_band_mm, 0 if absent)."""
    n = q.get("spatial_fence_lengths_by_channel", {}).get(channel, q["spatial_fence_lengths"])
    return float(n) * q["thermistor_length_mm"] + float(q.get("spatial_guard_band_mm", 0.0))


def vitreous_depths(q: dict) -> dict:
    """{(species, status, marker_eye): depth of the vitreous sensor inside the convex hull
    of the micro-CT globe segmentation, mm}, read from config/<qc.vitreous_depth_file>."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                        q.get("vitreous_depth_file", "vitreous_depth_ct.csv"))
    d = pd.read_csv(path, comment="#")
    return {(r.species, r.status, r.marker_eye): float(r.depth_convex_hull_mm) for r in d.itertuples()}


def vitreous_fractions(q: dict) -> dict:
    """{(species, status, marker_eye): position of the vitreous sensor as a fraction of the way
    from the centre of the micro-CT globe outline to the wall in the same direction}, read from
    config/<qc.vitreous_position_file>."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                        q.get("vitreous_position_file", "vitreous_position_ct.csv"))
    d = pd.read_csv(path, comment="#")
    return {(r.species, r.status, r.marker_eye): float(r.centre_fraction) for r in d.itertuples()}


def runs_for_marker(runs: dict, species: str, status: str, marker_eye: str) -> list[str]:
    """Human marker H<n> belongs to every ring run of eye n (one implanted sensor set);
    porcine marker F<seq>/P<seq> belongs to the run built from file sequence <seq>."""
    n = int(marker_eye[1:])
    if species == "Human":
        return sorted(rid for rid, r in runs.items() if r.species == "Human" and r.eye == n)
    return sorted(rid for rid, r in runs.items()
                  if r.species == "Porcine" and r.status == status and r.source_seq == n)


def spatial_gate(runs: dict, markers: dict, cfg: dict) -> pd.DataFrame:
    q = cfg["qc"]
    mode = q.get("spatial_fence", "two-sided")
    # 2026-10-03: sensors not fixed to tissue neither shape the alignment nor have an offset fence
    free = list(q.get("spatial_free_channels", [])) if mode == "aligned_mm" else []
    table = []
    for (species, status), eyes in markers.items():
        dev = constellation_deviation(eyes).merge(aligned_offsets(eyes, free=free), on=["marker_eye", "channel"])
        dev["species"], dev["status"] = species, status
        dev["fence_mm"] = [np.nan if ch in free else fence_mm(ch, q) for ch in dev["channel"]]
        if mode == "aligned_mm":
            # 2026-10-02: fixed fence in thermistor lengths on the aligned distance
            dev["flagged"] = dev["aligned_mm"] > dev["fence_mm"]
            dev["flag_reason"] = np.where(dev["flagged"], "offset", "")
            # 2026-10-03: a vitreous sensor must lie in the vitreous cavity (micro-CT depth)
            dmin = float(q.get("vitreous_min_depth_mm", 0.0))
            dev["ct_depth_mm"] = np.nan
            if dmin > 0:
                depth = vitreous_depths(q)
                isv = dev["channel"] == "Vitreous"
                dev.loc[isv, "ct_depth_mm"] = [depth.get((species, status, e), np.nan) for e in dev.loc[isv, "marker_eye"]]
                shallow = isv & (dev["ct_depth_mm"] < dmin)
                dev.loc[shallow, "flag_reason"] = [(r + "+depth") if r else "depth" for r in dev.loc[shallow, "flag_reason"]]
                dev["flagged"] = dev["flagged"] | shallow
            # 2026-10-07: ... judged by how far from the centre towards the wall it sits (scales with eye size)
            fmax = float(q.get("vitreous_max_centre_fraction", 0.0))
            dev["ct_centre_fraction"] = np.nan
            if fmax > 0:
                frac = vitreous_fractions(q)
                isv = dev["channel"] == "Vitreous"
                dev.loc[isv, "ct_centre_fraction"] = [frac.get((species, status, e), np.nan) for e in dev.loc[isv, "marker_eye"]]
                outer = isv & (dev["ct_centre_fraction"] > fmax)
                dev.loc[outer, "flag_reason"] = [(r + "+position") if r else "position" for r in dev.loc[outer, "flag_reason"]]
                dev["flagged"] = dev["flagged"] | outer
        else:
            dev["flagged"] = False
            for ch, g in dev.groupby("channel"):
                lo, hi, out = tukey_fence(g["deviation"].to_numpy(), q["tukey_k"], q["fence_min_n"])
                if mode == "upper":
                    # a small deviation means a sensor placed as the cohort places it -- never an
                    # outlier; flag only deviations above the upper fence (2026-10-02)
                    out = g["deviation"].to_numpy() > hi
                dev.loc[g.index[out], "flagged"] = True
        table.append(dev)
        for _, row in dev[dev["flagged"]].iterrows():
            for rid in runs_for_marker(runs, species, status, row["marker_eye"]):
                r, ch = runs[rid], row["channel"]
                prior = r.exclusions.get(ch)
                if prior and not q["spatial_precedence"]:
                    continue
                if prior:
                    r.also_broken[ch] = prior
                r.exclusions[ch] = "Space Outlier"
    return pd.concat(table, ignore_index=True) if table else pd.DataFrame()


def final_temperature(t, y, window_s: float):
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 10:
        return None
    w = m & (t >= t[m].max() - window_s)
    return float(y[w].mean()) if w.sum() > 5 else None


def temperature_fence_report(runs: dict, cfg: dict) -> pd.DataFrame:
    """Final-temperature Tukey fence per group and channel (SCR excluded).
    REPORT ONLY — flagged traces stay in the analysis."""
    q, rows = cfg["qc"], []
    for rid, r in runs.items():
        t = pd.to_numeric(pd.Series(r.filtered.index), errors="coerce").to_numpy()
        for ch in cfg["channels"]["temperature"]:
            if ch in cfg["channels"]["device"] or r.exclusions.get(ch) or ch not in r.filtered:
                continue
            ft = final_temperature(t, pd.to_numeric(r.filtered[ch], errors="coerce").to_numpy(),
                                   q["final_window_s"])
            if ft is not None:
                rows.append({"run_id": rid, "group": r.group, "channel": ch, "final_T": ft})
    df = pd.DataFrame(rows)
    df["breach"] = False
    for _, g in df.groupby(["group", "channel"]):
        _, _, out = tukey_fence(g["final_T"].to_numpy(), q["tukey_k"], q["fence_min_n"])
        df.loc[g.index[out], "breach"] = True
    df["applied"] = False
    return df


MANUAL_REASON = "Visual (artifact)"


def manual_gate(runs: dict, cfg: dict):
    """Documented manual exclusions, applied only to traces that passed the
    objective gates (an objective reason is never overwritten)."""
    q = cfg.get("qc", {})
    if not q.get("apply_manual", True):
        return
    for item in q.get("manual_exclusions", []):
        r = runs.get(item["run_id"])
        if r is None:
            raise KeyError(f"manual exclusion: unknown run {item['run_id']}")
        ch = item["channel"]
        if not r.exclusions.get(ch):
            r.exclusions[ch] = MANUAL_REASON


def run_qc(runs: dict, markers: dict, cfg: dict):
    """Gate order: hardware -> fit -> spatial (spatial reason wins) -> manual."""
    hardware_gate(runs, cfg)
    fit_gate(runs, cfg)
    spatial = spatial_gate(runs, markers, cfg)
    manual_gate(runs, cfg)
    return spatial, temperature_fence_report(runs, cfg)


def exclusion_table(runs: dict, cfg: dict) -> pd.DataFrame:
    rows = [{"run_id": rid, "group": " · ".join(r.group), "channel": ch,
             "reason": r.exclusions.get(ch, ""), "also_broken": r.also_broken.get(ch, "")}
            for rid, r in runs.items() for ch in cfg["channels"]["temperature"]]
    return pd.DataFrame(rows)
