"""Diagnostic figures: calibration, uncertainty, sensor positions, spectra, single
runs, individual traces and the quality-control screens. Colour always identifies
the channel (config [figures.channel_colors])."""
from __future__ import annotations

from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .calibration import human_plateau_levels, human_temperature, kmeans_levels, plateau_levels
from .processing import exponential
from .safe_write import save_figure

COHORT_MARKER = {("Human", "Fresh"): "o", ("Porcine", "Fresh"): "^", ("Porcine", "Preserved"): "s"}
EYE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 3)), (0, (3, 5, 1, 5))]


def _col(cfg, ch):
    return cfg.get("figures", {}).get("channel_colors", {}).get(ch, "#444444")


def _ylim(cfg):
    return cfg.get("figures", {}).get("temperature_ylim_C", (0, 40))


def _cohort(key):
    return f"{key[0]} {key[1].lower()}"


# ── Calibration ──────────────────────────────────────────────────────────────
def calibration_figures(cal_files: dict, sh: dict, current_fit, cfg: dict, folder: str, slug):
    t_ref = cfg["calibration"]["reference_temps_C"]
    seed = cfg["calibration"]["kmeans_seed"]
    chans = [c for c in cfg["channels"]["temperature"]]
    for eye, df in sorted(cal_files.items(), key=lambda kv: int(kv[0][1:])):
        # (a) plateau detection
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, ch in zip(axes.flat, chans):
            if ch not in df.columns:
                ax.set_visible(False)
                continue
            v = pd.to_numeric(df[ch], errors="coerce")
            ax.plot(np.arange(len(v)), v.to_numpy(), color=_col(cfg, ch), lw=0.8)
            for lv, tr in zip(human_plateau_levels(v, cfg), sorted(t_ref, reverse=False)[::-1]):
                ax.axhline(lv, color="#333333", ls="--", lw=1.0)
                ax.text(len(v) * 0.99, lv, f"{lv:.3f} V", ha="right", va="bottom", fontsize=7)
            ax.set_title(ch, fontsize=9)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Voltage (V)")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Calibration plateaus — human eye {eye} (dry block 37 / 20.5 / 4 °C)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_figure(fig, f"{folder}/human_{eye}_plateaus.png")
        plt.close(fig)
        # (b) calibration curves and the three reference points
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for ch in chans:
            c = sh.get(eye, {}).get(ch)
            if c is None or ch not in df.columns:
                continue
            lv = human_plateau_levels(pd.to_numeric(df[ch], errors="coerce"), cfg)
            vg = np.linspace(min(lv) - 0.3, max(lv) + 0.3, 300)
            ax.plot(vg, human_temperature(vg, c), color=_col(cfg, ch), lw=1.8, label=ch)
            ax.scatter(lv, t_ref, color=_col(cfg, ch), edgecolor="black", zorder=3, s=30)
        for tr in t_ref:
            ax.axhline(tr, color="#999999", lw=0.6, ls="-.")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Temperature (°C)")
        linear = cfg["calibration"].get("human_curve", "steinhart_hart") == "linear"
        ax.set_title(f"{'Linear' if linear else 'Steinhart–Hart'} calibration — human eye {eye}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, f"{folder}/human_{eye}_{'linear' if linear else 'steinhart_hart'}.png")
        plt.close(fig)

    # (c) pooled human current calibration
    cur = cfg["channels"]["current"]
    i_on, i_off = cfg["calibration"]["current_reference_A"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    allv = []
    for eye, df in cal_files.items():
        if cur in df.columns:
            lo, hi = kmeans_levels(df[cur].dropna(), 2, seed)
            ax.scatter([lo, hi], [i_on, i_off], color="#5b3fbf", alpha=0.6, s=25)
            allv += [lo, hi]
    if allv:
        vg = np.linspace(min(allv) - 0.3, max(allv) + 0.3, 100)
        m, c = current_fit
        ax.plot(vg, m * vg + c, color="#5b3fbf", lw=2, label=f"pooled fit: I = {m:+.3f}·V {c:+.3f}")
    ax.set_xlabel("Current-sensor output (V)")
    ax.set_ylabel("TEC current (A)")
    ax.set_title("Human TEC-current calibration (per-eye plateaus, pooled fit)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"{folder}/human_current.png")
    plt.close(fig)

    # (d) porcine fixed linear transfer
    a, b = cfg["calibration"]["porcine_slope"], cfg["calibration"]["porcine_intercept"]
    vg = np.linspace(-9, 9, 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(vg, a * vg + b, color="#333333", lw=2, label=f"T = {a}·V + {b}")
    ax.scatter([0.0, -8.0], [a * 0 + b, a * -8 + b], color="#c0392b", zorder=3,
               label="trim points: 0 V ↔ 20.5 °C, −8 V ↔ 37 °C")
    ax.set_xlabel("Amplifier output (V)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Porcine voltage–temperature transfer (all channels)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"{folder}/porcine_transfer.png")
    plt.close(fig)


# ── Uncertainty ──────────────────────────────────────────────────────────────
def uncertainty_figure(budget, path: str, title: str):
    df, uc, U = budget
    d = df[df["u_C"] > 1e-6].sort_values("u_C")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.4))
    ax.barh(range(len(d)), 2 * d["u_C"], color="#6a8caf")
    for i, v in enumerate(d["u_C"]):
        ax.text(2 * v, i, f"  ±{2 * v:.3f}", va="center", fontsize=8)
    ax.axvline(U, color="black", ls="--", lw=1.4, label=f"combined U (k = 2) = ±{U:.3f} °C")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["source"], fontsize=8)
    ax.set_xlabel("Contribution, 2u (°C)")
    ax.set_xlim(0, max(U, 2 * d["u_C"].max()) * 1.25)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


# ── Sensor positions ─────────────────────────────────────────────────────────
def spatial_figures(spatial: pd.DataFrame, cfg: dict, folder: str, slug):
    chans = [c for c in cfg["channels"]["temperature"] if c in set(spatial["channel"])]
    cohorts = list(spatial.groupby(["species", "status"]).groups)
    ch_handles = [Line2D([], [], marker="o", ls="", color=_col(cfg, c), label=c) for c in chans]

    def scatter3d(ax, s, marker, big):
        for ch in chans:
            q = s[s.channel == ch]
            ax.scatter(q.L, q.P, q.S, color=_col(cfg, ch), marker=marker, s=30, alpha=0.8)
            if big:
                ax.scatter(q.L.mean(), q.P.mean(), q.S.mean(), color=_col(cfg, ch), marker="X",
                           s=180, edgecolor="black")

    for key in cohorts:
        s = spatial[(spatial.species == key[0]) & (spatial.status == key[1])]
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        scatter3d(ax, s, COHORT_MARKER.get(key, "o"), True)
        ax.set_xlabel("L (mm)")
        ax.set_ylabel("P (mm)")
        ax.set_zlabel("S (mm)")
        ax.set_title(f"Sensor positions — {_cohort(key)} (n = {s.marker_eye.nunique()} eyes; X = centroid)",
                     fontsize=10)
        ax.legend(handles=ch_handles, fontsize=7, loc="upper left")
        save_figure(fig, f"{folder}/positions_3d_{slug(_cohort(key))}.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, axis, name in zip(axes, ["L", "P", "S"], ["L (left–right)", "P (posterior–anterior)",
                                                             "S (superior–inferior)"]):
            for i, ch in enumerate(chans):
                q = s[s.channel == ch]
                x = np.full(len(q), i) + np.linspace(-0.15, 0.15, len(q))
                ax.scatter(x, q[axis], color=_col(cfg, ch), s=26, alpha=0.85)
                ax.plot([i - 0.3, i + 0.3], [q[axis].mean()] * 2, color=_col(cfg, ch), lw=2.5)
            ax.set_xticks(range(len(chans)))
            ax.set_xticklabels([c.replace(" ", "\n") for c in chans], fontsize=8)
            ax.set_title(name, fontsize=10)
            ax.set_ylabel("mm")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Sensor coordinates per channel — {_cohort(key)} (bar = mean)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, f"{folder}/positions_2d_{slug(_cohort(key))}.png")
        plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for key in cohorts:
        scatter3d(ax, spatial[(spatial.species == key[0]) & (spatial.status == key[1])],
                  COHORT_MARKER.get(key, "o"), False)
    ax.set_xlabel("L (mm)")
    ax.set_ylabel("P (mm)")
    ax.set_zlabel("S (mm)")
    ax.set_title("Sensor positions — all cohorts (co-registered frame)", fontsize=10)
    handles = ch_handles + [Line2D([], [], marker=COHORT_MARKER.get(k, "o"), ls="", color="#555555",
                                   label=_cohort(k)) for k in cohorts]
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    save_figure(fig, f"{folder}/positions_3d_all.png")
    plt.close(fig)

    # inter-thermistor distance matrix (mean pairwise distance, mm)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6.4 * len(cohorts) + 1.5, 5.2))
    fig.subplots_adjust(wspace=0.55, bottom=0.22, left=0.07, right=0.88)
    axes = np.atleast_1d(axes)
    mats = []
    for key in cohorts:
        s = spatial[(spatial.species == key[0]) & (spatial.status == key[1])]
        M = np.zeros((len(chans), len(chans)))
        for i, j in combinations(range(len(chans)), 2):
            d = []
            for _, e in s.groupby("marker_eye"):
                a, b = e[e.channel == chans[i]], e[e.channel == chans[j]]
                if len(a) and len(b):
                    d.append(np.linalg.norm(a[["L", "P", "S"]].to_numpy()[0] - b[["L", "P", "S"]].to_numpy()[0]))
            M[i, j] = M[j, i] = np.mean(d) if d else np.nan
        mats.append(M)
    vmax = np.nanmax([np.nanmax(M) for M in mats])
    short = [c.replace("Equatorial ", "Eq. ").replace("Anterior ", "Ant. ").replace("Posterior ", "Post. ")
             for c in chans]
    for ax, key, M in zip(axes, cohorts, mats):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
        for i in range(len(chans)):
            for j in range(len(chans)):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.6 * vmax else "black")
        ax.set_xticks(range(len(chans)))
        ax.set_yticks(range(len(chans)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        n = spatial[(spatial.species == key[0]) & (spatial.status == key[1])].marker_eye.nunique()
        ax.set_title(f"{_cohort(key)} (n = {n})", fontsize=10)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="Mean distance (mm)")
    save_figure(fig, f"{folder}/distance_matrix.png")
    plt.close(fig)


def spatial_outlier_planes(spatial: pd.DataFrame, cfg: dict, folder: str, slug):
    """Offsets from the cohort mean position in the three planes, per channel; sensors
    flagged by the spatial fence drawn as red diamonds. With the aligned fence the
    offsets are those after rigid alignment (what the fence measures) and the fence
    is drawn as a circle; otherwise the raw offsets from the co-registered centroid."""
    aligned = cfg["qc"].get("spatial_fence", "two-sided") == "aligned_mm"
    p = "a" if aligned else "d"
    chans = [c for c in cfg["channels"]["temperature"] if c in set(spatial["channel"])]
    planes = [(p + "L", p + "P"), (p + "L", p + "S"), (p + "P", p + "S")]
    lim = float(np.nanmax(np.abs(spatial[[p + "L", p + "P", p + "S"]].to_numpy()))) * 1.1
    for key, s in spatial.groupby(["species", "status"]):
        fig, axes = plt.subplots(len(planes), len(chans), figsize=(3.1 * len(chans), 3.0 * len(planes)))
        for c, ch in enumerate(chans):
            q = s[s.channel == ch]
            for r, (x, y) in enumerate(planes):
                ax = axes[r, c]
                ok, bad = q[~q.flagged], q[q.flagged]
                ax.scatter(ok[x], ok[y], color=_col(cfg, ch), s=28, alpha=0.85)
                ax.scatter(bad[x], bad[y], color="#c0392b", marker="D", s=46, edgecolor="black", zorder=3)
                for _, b in bad.iterrows():
                    ax.annotate(b.marker_eye, (b[x], b[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
                ax.axhline(0, color="#999999", lw=0.6)
                ax.axvline(0, color="#999999", lw=0.6)
                if aligned and len(q):
                    # projection of the 3-D fence sphere: a sensor inside this circle can
                    # still be flagged by its offset along the third axis
                    ax.add_patch(plt.Circle((0, 0), q["fence_mm"].iloc[0], fill=False, ls="--",
                                            lw=0.8, color="#c0392b"))
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal")
                ax.grid(alpha=0.3)
                ax.set_xlabel(f"Δ{x[1]} (mm)", fontsize=8)
                ax.set_ylabel(f"Δ{y[1]} (mm)", fontsize=8)
                ax.tick_params(labelsize=7)
                if r == 0:
                    ax.set_title(ch, fontsize=9)
        what = ("offsets from the mean position after rigid alignment" if aligned
                else "offsets from the cohort centroid")
        fig.suptitle(f"Spatial screen — {what}, {_cohort(key)} "
                     f"(red = excluded by the spatial fence)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_figure(fig, f"{folder}/spatial_outliers_2d_{slug(_cohort(key))}.png")
        plt.close(fig)


# ── Temperature fence (report only) ──────────────────────────────────────────
def temperature_fence_figure(tf: pd.DataFrame, cfg: dict, path: str, label):
    chans = [c for c in cfg["channels"]["temperature"] if c in set(tf["channel"])]
    groups = list(dict.fromkeys(tf["group"]))
    fig, axes = plt.subplots(1, len(chans), figsize=(3.2 * len(chans), 5), sharey=True)
    for ax, ch in zip(np.atleast_1d(axes), chans):
        for i, g in enumerate(groups):
            s = tf[(tf.channel == ch) & (tf.group == g)]
            if s.empty:
                continue
            ax.boxplot(s["final_T"], positions=[i], widths=0.55, whis=cfg["qc"]["tukey_k"], showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=_col(cfg, ch), alpha=0.25),
                       medianprops=dict(color="black"))
            x = np.full(len(s), i) + np.linspace(-0.15, 0.15, len(s))
            b = s["breach"].to_numpy()
            ax.scatter(x[~b], s["final_T"].to_numpy()[~b], color=_col(cfg, ch), s=18, zorder=3)
            ax.scatter(x[b], s["final_T"].to_numpy()[b], color=_col(cfg, ch), marker="D", s=40,
                       edgecolor="black", linewidth=1.3, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([label(g) for g in groups], rotation=60, ha="right", fontsize=7)
        ax.set_title(ch, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("Final temperature (°C)")
    fig.suptitle("Final-temperature fence per group — REPORT ONLY, NOT APPLIED "
                 "(black-edged diamond = outside the Tukey fence, kept in the analysis)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path)
    plt.close(fig)


# ── Spectra, single runs, individual traces ──────────────────────────────────
def fft_figure(runs: dict, ids: list, cfg: dict, path: str, title: str):
    """Mean single-sided amplitude spectrum of the calibrated (unfiltered) traces."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for ch in cfg["channels"]["temperature"]:
        specs = []
        for rid in ids:
            r = runs[rid]
            if r.exclusions.get(ch) or ch not in r.temp.columns:
                continue
            t = pd.to_numeric(pd.Series(r.temp.index), errors="coerce").to_numpy()
            y = pd.to_numeric(r.temp[ch], errors="coerce").to_numpy()
            m = np.isfinite(t) & np.isfinite(y)
            if m.sum() < 32:
                continue
            yy = y[m] - y[m].mean()
            dt = np.nanmean(np.diff(t[m]))
            f = np.fft.rfftfreq(len(yy), dt)
            specs.append((f, 2 / len(yy) * np.abs(np.fft.rfft(yy))))
        if not specs:
            continue
        grid = specs[0][0]
        amp = np.mean([np.interp(grid, f, a) for f, a in specs], axis=0)
        ax.loglog(grid[1:], amp[1:], color=_col(cfg, ch), lw=1.2, label=f"{ch} (n = {len(specs)})")
    ax.axvline(cfg["filter"]["lowpass_cutoff_Hz"], color="#c0392b", ls="--", lw=1.2,
               label=f"low-pass cutoff {cfg['filter']['lowpass_cutoff_Hz']} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (°C)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def run_figure(run, cfg: dict, path: str):
    """One run: calibrated, median-filtered and smoothed trace plus the exponential fit
    for every channel; excluded channels are labelled with the reason."""
    chans = cfg["channels"]["temperature"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, ch in zip(axes.flat, chans):
        col = _col(cfg, ch)
        if ch in run.temp.columns:
            t = pd.to_numeric(pd.Series(run.temp.index), errors="coerce").to_numpy() / 60
            ax.plot(t, run.temp[ch].to_numpy(float), color="#bbbbbb", lw=0.6, label="calibrated")
            ax.plot(t, run.median[ch].to_numpy(float), color=col, lw=0.8, alpha=0.6, label="median")
            ax.plot(t, run.filtered[ch].to_numpy(float), color=col, lw=1.8, label="smoothed")
            p = run.fit.get(ch)
            if p:
                ts = t * 60
                ax.plot(t[ts >= 0], exponential(ts[ts >= 0], p["A"], p["k"], p["C"]), color="black", ls=":",
                        lw=1.2, label=f"fit (τ = {p['tau']:.0f} s, R² = {p['R2']:.3f})")
        reason = run.exclusions.get(ch)
        ax.set_title(f"{ch}" + (f"  — EXCLUDED: {reason}" if reason else ""), fontsize=9,
                     color="#c0392b" if reason else "black")
        ax.set_ylim(*_ylim(cfg))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("Time (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (°C)")
    fig.suptitle(f"{run.run_id} — {' · '.join(run.group)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path)
    plt.close(fig)


def individual_traces_figure(runs: dict, ids: list, cfg: dict, path: str, title: str):
    """Every included trace of the group, per channel; one colour per eye, the
    same in every panel, so an eye can be followed across locations."""
    from .figures import channel_label, light_grid
    chans = cfg["channels"]["temperature"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    eyes = sorted({runs[r].eye_id for r in ids}, key=lambda e: int(e[1:]))
    cmap = plt.get_cmap("tab10")
    eye_col = {e: cmap(i % 10) for i, e in enumerate(eyes)}
    for ax, ch in zip(axes.flat, chans):
        n = 0
        for rid in ids:
            r = runs[rid]
            if r.exclusions.get(ch) or ch not in r.filtered.columns:
                continue
            t = pd.to_numeric(pd.Series(r.filtered.index), errors="coerce").to_numpy() / 60
            ax.plot(t, r.filtered[ch].to_numpy(float), color=eye_col[r.eye_id], lw=1.2)
            n += 1
        ax.set_title(f"{ch} (n = {n})", fontsize=9)
        ax.set_ylim(*_ylim(cfg))
        light_grid(ax, x_major=5, x_minor=2.5)
    for ax in axes[-1]:
        ax.set_xlabel("Time (min)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (°C)")
    handles = [Line2D([], [], color=eye_col[e], lw=2.0, label=e) for e in eyes]
    fig.legend(handles=handles, loc="lower center", ncol=len(eyes), fontsize=8, frameon=False, title="Eye")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    save_figure(fig, path)
    plt.close(fig)


def manual_exclusions_figure(runs: dict, cfg: dict, path: str, display=lambda g: " ".join(g)):
    """One panel per manual exclusion: the excluded trace (red) over every
    retained trace of the same group and site (grey), so the artifact can be
    judged against normal behaviour. Drawn at print size for the thesis."""
    from .figures import _print_style, channel_label, light_grid
    items = [m for m in cfg.get("qc", {}).get("manual_exclusions", []) if m["run_id"] in runs]
    if not items or not cfg["qc"].get("apply_manual", True):
        return
    n = len(items)
    ncol = 2 if n > 1 else 1
    nrow = -(-n // ncol)
    with plt.rc_context(_print_style(cfg)):
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.3, 2.6 * nrow + 0.9), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, m in zip(axes.flat, items):
            bad = runs[m["run_id"]]
            ch = m["channel"]
            for rid, r in runs.items():
                if r.group != bad.group or rid == m["run_id"] or r.exclusions.get(ch) or ch not in r.filtered:
                    continue
                t = pd.to_numeric(pd.Series(r.filtered.index), errors="coerce").to_numpy() / 60
                ax.plot(t, r.filtered[ch].to_numpy(float), color="0.65", lw=1.0)
            t = pd.to_numeric(pd.Series(bad.filtered.index), errors="coerce").to_numpy() / 60
            ax.plot(t, bad.filtered[ch].to_numpy(float), color="#C0392B", lw=1.6)
            ax.set_title(f"{display(bad.group)}\n{channel_label(ch, cfg, lower=True)}, eye {bad.eye_id.lstrip('E')}",
                         fontsize=10)
            ax.set_ylim(*_ylim(cfg))
            light_grid(ax, x_major=5, x_minor=2.5)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Time (min)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Temperature (°C)")
        h = [Line2D([], [], color="#C0392B", lw=1.8), Line2D([], [], color="0.65", lw=1.4)]
        fig.legend(h, ["Excluded trace", "Retained traces, same cohort and site"], loc="lower center",
                   ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        save_figure(fig, path)
        plt.close(fig)


def trace_exclusions_figure(runs: dict, cfg: dict, path: str, display=lambda g: " ".join(g), groups=None):
    """Every non-spatial exclusion (hardware, fit, visual) of the given groups, one
    panel each, ordered by reason: the excluded trace (red) over every retained
    trace of the same group and site (grey). A hardware trace carries no
    temperature signal, so its panel states the raw-voltage fault instead."""
    from .figures import _print_style, channel_label, light_grid
    from .qc import MANUAL_REASON
    order = {"Broken (hardware)": 0, "Broken (low R2)": 1, MANUAL_REASON: 2}
    items = [(rid, ch, why) for rid, r in runs.items() for ch, why in r.exclusions.items()
             if why in order and (groups is None or r.group in groups)]
    if not items:
        return
    items.sort(key=lambda x: (order[x[2]], runs[x[0]].group[0] != "Porcine", x[0], x[1]))
    n = len(items)
    ncol = 3 if n > 4 else 2
    nrow = -(-n // ncol)
    red = "#C0392B"
    with plt.rc_context(_print_style(cfg)):
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.3, 2.45 * nrow + 0.8), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, (rid, ch, why) in zip(axes.flat, items):
            bad = runs[rid]
            for other, r in runs.items():
                if r.group != bad.group or other == rid or r.exclusions.get(ch) or ch not in r.filtered:
                    continue
                t = pd.to_numeric(pd.Series(r.filtered.index), errors="coerce").to_numpy() / 60
                ax.plot(t, r.filtered[ch].to_numpy(float), color="0.65", lw=0.9)
            if why == "Broken (hardware)":
                v = pd.to_numeric(bad.volts[ch], errors="coerce").to_numpy()
                v = v[np.isfinite(v)]
                railed = float(np.mean(np.abs(v) > cfg["qc"].get("hardware_railed_V", 9.9)))
                stuck = float(np.mean(np.abs(v - np.median(v)) < 1e-6))
                note = (f"No signal: voltage\nfixed at {np.median(v):+.0f} V" if stuck > 0.5
                        else f"No signal: at the\n±10 V limit for {100 * railed:.0f} %\nof the run")
                # A sensor documented as failing ([[qc.hardware_exclusions]]) has a signal in this
                # run: its trace is drawn, with the note given in the config.
                doc = [h for h in cfg["qc"].get("hardware_exclusions", [])
                       if h["run_id"] == rid and h["channel"] == ch]
                if doc:
                    note = doc[0].get("figure_note", "Sensor failing")
                    if ch in bad.filtered:
                        t = pd.to_numeric(pd.Series(bad.filtered.index), errors="coerce").to_numpy() / 60
                        ax.plot(t, bad.filtered[ch].to_numpy(float), color=red, lw=1.5)
                ax.text(0.5, 0.04, note, transform=ax.transAxes,
                        ha="center", va="bottom", color=red, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=red, lw=0.8))
                label = "Hardware fault"
            else:
                t = pd.to_numeric(pd.Series(bad.filtered.index), errors="coerce").to_numpy() / 60
                ax.plot(t, bad.filtered[ch].to_numpy(float), color=red, lw=1.5)
                label = (f"Poor fit, R² = {bad.fit[ch]['R2']:.2f}" if why != MANUAL_REASON and bad.fit.get(ch)
                         else "Poor fit" if why != MANUAL_REASON else "Visual review")
            cohort = display(bad.group).replace("reduced-bulk aluminum ring", "reduced-bulk ring")
            ax.set_title(f"{label}\n{cohort}\n{channel_label(ch, cfg, lower=True)}, eye {bad.eye_id.lstrip('E')}", fontsize=8)
            ax.set_ylim(*_ylim(cfg))
            light_grid(ax, x_major=5, x_minor=2.5)
        for ax in axes.flat[n:]:
            ax.axis("off")
        for j in range(ncol):                       # x labels on the lowest visible panel of each column
            col = [axes[i, j] for i in range(nrow) if i * ncol + j < n]
            if col:
                col[-1].set_xlabel("Time (min)")
                col[-1].xaxis.set_tick_params(labelbottom=True)
        for ax in axes[:, 0]:
            ax.set_ylabel("Temperature (°C)")
        h = [Line2D([], [], color=red, lw=1.8), Line2D([], [], color="0.65", lw=1.4)]
        labels = ["Excluded trace", "Retained traces,\nsame cohort\nand site"]
        if n < nrow * ncol:                         # legend in the empty panel
            axes.flat[-1].legend(h, labels, loc="center", frameon=False)
            fig.tight_layout()
        else:
            fig.legend(h, [l.replace("\n", " ") for l in labels], loc="lower center", ncol=2, frameon=False,
                       bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.05, 1, 1))
        save_figure(fig, path)
        plt.close(fig)
