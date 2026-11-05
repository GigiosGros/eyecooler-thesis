"""Figures: group temperature curves, pairwise overlays, and the spatial screen."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .safe_write import save_figure

COLORS = ["#1f4e79", "#c0504d", "#4f8a3a", "#7f5fa6", "#d08a1c"]


def display_name(name: str, cfg: dict) -> str:
    """Name shown in titles and legends ([figures.display_names]); the internal
    name, which also sets the output folder, is unchanged."""
    return cfg.get("figures", {}).get("display_names", {}).get(name, name)


def channel_label(ch: str, cfg: dict, lower: bool = False) -> str:
    """Site name shown in figures ([figures.channel_display_names]), as in the thesis
    text: e.g. 'Optic Nerve' -> 'Proximal optic nerve'. lower=True for mid-sentence
    use; an all-capital abbreviation (SCR) is never lowered."""
    name = cfg.get("figures", {}).get("channel_display_names", {}).get(ch, ch)
    if lower and not name.isupper():
        name = name[0].lower() + name[1:]
    return name


def comparison_title(names: list[str], cfg: dict) -> str:
    """'A vs B', factoring out a shared device: 'Fresh porcine vs Preserved
    porcine, SCEC Gen 1' rather than repeating 'SCEC Gen 1' on both sides."""
    parts = [display_name(n, cfg).rsplit(", ", 1) for n in names]
    if all(len(p) == 2 for p in parts) and len({p[1] for p in parts}) == 1:
        heads = [p[0] if i == 0 or p[0].startswith("Human") else p[0][0].lower() + p[0][1:]
                 for i, p in enumerate(parts)]
        return " vs ".join(heads) + ", " + parts[0][1]
    return " vs ".join(display_name(n, cfg) for n in names)


def light_grid(ax, x_major=None, x_minor=None, y_major=10, y_minor=5):
    """Light background grid as in the manuscript figures: major lines plus
    fainter minor lines, beneath the data."""
    from matplotlib.ticker import MultipleLocator
    if x_major:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
    if x_minor:
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    if y_major:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
    if y_minor:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.82", lw=0.5)
    ax.grid(which="minor", color="0.93", lw=0.4)
    ax.tick_params(which="minor", length=2)


def shade(color, t):
    """Tint (t < 0.5, towards white) or shade (t > 0.5, towards black) of a
    colour; used to tell eyes apart while keeping the channel colour."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    if t < 0.5:
        return tuple(rgb + (1 - rgb) * (0.55 * (0.5 - t) / 0.5))
    return tuple(rgb * (1 - 0.45 * (t - 0.5) / 0.5))


def _print_style(cfg: dict) -> dict:
    """Figures are drawn at their printed size (the thesis text width), so a
    12 pt font here is 12 pt on the page, matching the body text."""
    pt = cfg.get("figures", {}).get("font_pt", 12)
    return {"font.size": pt, "axes.titlesize": pt, "axes.labelsize": pt,
            "legend.fontsize": pt, "legend.title_fontsize": pt,
            "xtick.labelsize": pt - 1, "ytick.labelsize": pt - 1,
            "figure.titlesize": pt}


def _legends_below(fig, legends, width, plot_h, top_h, xlab_h, left_in=0.75,
                   right_in=0.12, gap=0.1, **adjust):
    """Stack `legends` (fig.legend, loc='upper center') in a strip below the axes.
    Legend heights are measured after drawing, so the strip always fits them
    and the axes keep the full width and their own height."""
    fig.set_size_inches(width, top_h + plot_h + xlab_h + 2)
    fig.canvas.draw()
    hs = [lg.get_window_extent().height / fig.dpi for lg in legends]
    strip = sum(hs) + gap * (len(legends) + 1)
    H = top_h + plot_h + xlab_h + strip
    fig.set_size_inches(width, H)
    fig.subplots_adjust(left=left_in / width, right=1 - right_in / width,
                        top=1 - top_h / H, bottom=(xlab_h + strip) / H, **adjust)
    y = strip
    for lg, h in zip(legends, hs):
        y -= gap
        lg.set_bbox_to_anchor((0.5, y / H), transform=fig.transFigure)
        y -= h


def _row_major(handles: list, ncol: int) -> list:
    """matplotlib fills legend columns first; reorder so entries read by row."""
    rows = -(-len(handles) // ncol)
    return [handles[r * ncol + c] for c in range(ncol) for r in range(rows)
            if r * ncol + c < len(handles)]


def group_mean(runs: dict, ids: list[str], ch: str):
    """Mean and SD across the included traces of `ch`, per time sample."""
    frames = [runs[r].filtered[ch] for r in ids
              if not runs[r].exclusions.get(ch) and ch in runs[r].filtered.columns]
    if not frames:
        return None
    big = pd.concat(frames)
    g = big.groupby(big.index)
    return g.mean(), g.std(), len(frames)


def plot_groups(runs: dict, groups: dict, cfg: dict, path: str, title: str = "",
                color_by: str = "channel"):
    """One panel per channel (3 rows x 2 columns); one line (mean) and band
    (+/- SD) per group.
    color_by='channel': colour identifies the channel (the same in every figure)
    and line style the group — used for single groups and pairwise comparisons.
    color_by='group': colour identifies the group — used for the all-groups figure."""
    chans = cfg["channels"]["temperature"]
    fcfg = cfg.get("figures", {})
    ch_col = fcfg.get("channel_colors", {})
    styles = fcfg.get("group_linestyles", ["-", "--", ":", "-."])
    with plt.rc_context(_print_style(cfg)):
        fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
        for ax, ch in zip(axes.flat, chans):
            for i, (name, ids) in enumerate(groups.items()):
                res = group_mean(runs, ids, ch)
                if res is None:
                    continue
                mu, sd, n = res
                t = mu.index.to_numpy(float) / 60
                if color_by == "channel":
                    col, ls, band = ch_col.get(ch, "#444444"), styles[i % len(styles)], 0.18 if i == 0 else 0.08
                else:
                    col, ls, band = COLORS[i % len(COLORS)], "-", 0.15
                ax.plot(t, mu.to_numpy(), color=col, ls=ls, lw=1.4)
                ax.fill_between(t, (mu - sd).to_numpy(), (mu + sd).to_numpy(), color=col, alpha=band, lw=0)
            ns = [res[2] for res in (group_mean(runs, ids, ch) for ids in groups.values()) if res]
            ax.set_title(f"{channel_label(ch, cfg)} (n = {', '.join(map(str, ns))})")
            ax.set_ylim(*fcfg.get("temperature_ylim_C", (0, 40)))
            light_grid(ax, x_major=5, x_minor=2.5)
        for ax in axes[-1]:
            ax.set_xlabel("Time (min)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Temperature (°C)")
        h = [Line2D([], [], color="#333333" if color_by == "channel" else COLORS[i % len(COLORS)],
                    ls=styles[i % len(styles)] if color_by == "channel" else "-", lw=1.6)
             for i in range(len(groups))]
        l = [f"{display_name(name, cfg)} ({len(ids)} runs)" for name, ids in groups.items()]
        leg = fig.legend(h, l, loc="upper center", ncol=1, frameon=False)
        if title:
            fig.suptitle(title)
        _legends_below(fig, [leg], fcfg.get("width_in", 6.0), plot_h=5.3,
                       top_h=0.75 if title else 0.35, xlab_h=0.55, hspace=0.45, wspace=0.08)
        save_figure(fig, path)
        plt.close(fig)


def plot_overlay(runs: dict, groups: dict, cfg: dict, path: str, title: str = ""):
    """All channels on ONE plot: colour = channel, line style = group, band = +/- SD.
    Used for single groups (and as an alternative view of each comparison)."""
    fcfg = cfg.get("figures", {})
    ch_col = fcfg.get("channel_colors", {})
    styles = fcfg.get("group_linestyles", ["-", "--", ":", "-."])
    with plt.rc_context(_print_style(cfg)):
        fig, ax = plt.subplots()
        ch_handles = []
        for ch in cfg["channels"]["temperature"]:
            col, ns = ch_col.get(ch, "#444444"), []
            for i, (name, ids) in enumerate(groups.items()):
                res = group_mean(runs, ids, ch)
                if res is None:
                    continue
                mu, sd, n = res
                ns.append(str(n))
                t = mu.index.to_numpy(float) / 60
                ax.plot(t, mu.to_numpy(), color=col, ls=styles[i % len(styles)], lw=1.5)
                ax.fill_between(t, (mu - sd).to_numpy(), (mu + sd).to_numpy(), color=col,
                                alpha=0.15 if i == 0 else 0.07, lw=0)
            if ns:
                ch_handles.append(Line2D([], [], color=col, lw=2.2, label=f"{channel_label(ch, cfg)} (n = {', '.join(ns)})"))
        ax.set_ylim(*fcfg.get("temperature_ylim_C", (0, 40)))
        ax.set_xticks(range(0, 21, 2))
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Temperature (°C)")
        light_grid(ax, x_minor=1)
        ax.set_title(title or comparison_title(list(groups), cfg))

        # Legends go in a strip BELOW the axes, so the plot keeps the full width:
        # channels in 2 columns read by row (anterior -> posterior), then the
        # group line styles, one per row.
        legends = [fig.legend(handles=_row_major(ch_handles, 2), loc="upper center", ncol=2,
                              title="Channel (mean ± SD" + ("; n per group, in legend order)"
                                                          if len(groups) > 1 else ")"),
                              frameon=False, columnspacing=1.2, handlelength=1.8)]
        if len(groups) > 1:
            g_handles = [Line2D([], [], color="#333333", ls=styles[i % len(styles)], lw=1.6,
                                label=f"{display_name(name, cfg)} ({len(ids)} runs)")
                         for i, (name, ids) in enumerate(groups.items())]
            legends.append(fig.legend(handles=g_handles, loc="upper center", ncol=1, frameon=False))
        _legends_below(fig, legends, fcfg.get("width_in", 6.0), plot_h=3.6, top_h=0.4, xlab_h=0.55)
        save_figure(fig, path)
        plt.close(fig)


def _aligned_mode(cfg: dict) -> bool:
    return cfg["qc"].get("spatial_fence", "two-sided") == "aligned_mm"


def plot_spatial_print(spatial: pd.DataFrame, cfg: dict, path: str):
    """Spatial screen at thesis print size, one panel per site, cohorts side by side;
    data in the channel colour, sensors outside the fence as larger diamonds with a
    heavy black edge. Titles and labels in black.
    Aligned fence: each sensor's aligned distance from the cohort mean position
    (mm), with the site's fence as a dashed line. Legacy Tukey fences: the
    inter-sensor distance deviation (%), boxes with 1.5 IQR whiskers."""
    aligned = _aligned_mode(cfg)
    chans = [c for c in cfg["channels"]["temperature"] if c in set(spatial["channel"])]
    cohorts = [(("Porcine", "Preserved"), "Preserved"), (("Porcine", "Fresh"), "Fresh"),
               (("Human", "Fresh"), "Human")]
    col_name, scale, step = ("aligned_mm", 1.0, 2) if aligned else ("deviation", 100.0, 10)
    # 2026-10-07: a sensor screened by its position between the globe centre and the wall (the vitreous)
    # is drawn on that measure, with its own axis and limit, not as an offset from the cohort mean
    fmax = float(cfg["qc"].get("vitreous_max_centre_fraction", 0.0)) if aligned else 0.0
    frac_ch = [c for c in cfg["qc"].get("spatial_free_channels", []) if fmax and "ct_centre_fraction" in spatial
               and spatial.loc[spatial.channel == c, "ct_centre_fraction"].notna().any()]
    with plt.rc_context(_print_style(cfg)):
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 5.6), sharey=False)
        mm = spatial[~spatial.channel.isin(frac_ch)]
        top = mm[col_name].max() * scale
        if aligned:
            top = max(top, mm["fence_mm"].max())
        ymax = step / 2 * np.ceil(top / (step / 2))
        for k, (ax, ch) in enumerate(zip(axes.flat, chans)):
            col = cfg.get("figures", {}).get("channel_colors", {}).get(ch, "#444444")
            is_frac = ch in frac_ch
            for i, (key, _) in enumerate(cohorts):
                s = spatial[(spatial.channel == ch) & (spatial.species == key[0]) & (spatial.status == key[1])]
                d = s["ct_centre_fraction"].to_numpy(float) * 100 if is_frac else s[col_name].to_numpy(float) * scale
                if not aligned:
                    ax.boxplot(d[np.isfinite(d)], positions=[i], widths=0.5, whis=1.5, showfliers=False,
                               medianprops=dict(color="black", lw=1.4), boxprops=dict(lw=1.1, color=col),
                               whiskerprops=dict(lw=1.1, color=col), capprops=dict(lw=1.1, color=col))
                x = i + np.linspace(-0.13, 0.13, len(s)) if len(s) > 1 else np.full(len(s), float(i))
                f = s["flagged"].to_numpy(bool)
                ax.scatter(x[~f], d[~f], s=16, color=col, edgecolor="white", lw=0.3, zorder=3)
                ax.scatter(x[f], d[f], s=46, marker="D", color=col, edgecolor="black", lw=1.4, zorder=4)
            if is_frac:
                ax.axhline(fmax * 100, color="black", ls=":", lw=1.3, zorder=2)
            elif aligned:
                ax.axhline(spatial.loc[spatial.channel == ch, "fence_mm"].iloc[0],
                           color="black", ls="--", lw=1.0, zorder=2)
            ax.set_xticks(range(len(cohorts)))
            ax.set_xticklabels([c[1] for c in cohorts], fontsize=9, rotation=30, ha="right",
                               rotation_mode="anchor")
            ax.set_xlim(-0.6, len(cohorts) - 0.4)
            ax.set_title(channel_label(ch, cfg), color="black")
            if is_frac:
                ax.set_ylim(0, 100); light_grid(ax, y_major=25, y_minor=12.5)
                ax.set_ylabel("Percentage of the distance\nfrom globe center to wall", fontsize=9)
            else:
                ax.set_ylim(0, ymax); light_grid(ax, y_major=step, y_minor=step / 2)
                if k % 3 == 0:
                    ax.set_ylabel("Offset from mean (mm)" if aligned else "Distance deviation (%)")
                else:
                    ax.tick_params(labelleft=False)
            ax.grid(False, axis="x", which="both")
        leg = axes.flat[-1]
        leg.axis("off")
        h = [Line2D([], [], ls="", marker="o", ms=5, color="0.5"),
             Line2D([], [], ls="", marker="D", ms=7, color="0.5", markeredgecolor="black", markeredgewidth=1.4)]
        labels = ["Retained", "Spatial outlier,\nexcluded"]
        if aligned:
            q = cfg["qc"]
            L, n = q["thermistor_length_mm"], q["spatial_fence_lengths"]
            free = q.get("spatial_free_channels", [])
            text = f"Limit: {n:g} thermistor\nlength{'s' if n != 1 else ''} ({n * L:g} mm)"
            for c, k in q.get("spatial_fence_lengths_by_channel", {}).items():
                if c not in free:             # a free channel has no offset limit
                    text += f";\n{channel_label(c, cfg).lower()} {k:g} ({k * L:g} mm)"
            g = float(q.get("spatial_guard_band_mm", 0.0))
            if g:
                text += f"\n+ {g:g} mm guard band"
            if free:
                text += ";\nnone for the " + ", ".join(channel_label(c, cfg).lower() for c in free)
            dm = float(q.get("vitreous_min_depth_mm", 0.0))
            if dm:
                labels[1] = f"Spatial outlier,\nexcluded (beyond the\nlimit, or vitreous\nsensor < {dm:g} mm deep)"
            if frac_ch:
                labels[1] = "Spatial outlier,\nexcluded"
                text = f"Limit, sensors fixed to\ntissue: {n:g} thermistor\nlength ({n * L:g} mm)"
            h.append(Line2D([], [], color="black", ls="--", lw=1.0))
            labels.append(text)
            if frac_ch:
                h.append(Line2D([], [], color="black", ls=":", lw=1.3))
                labels.append(f"Limit, {', '.join(channel_label(c, cfg).lower() for c in frac_ch)}:\n{fmax:g} of the distance from\nglobe center to wall")
        leg.legend(h, labels, loc="center", frameon=False)
        fig.tight_layout()
        save_figure(fig, path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def plot_spatial(spatial: pd.DataFrame, cfg: dict, path: str):
    """Spatial screen per channel and cohort (aligned distance in mm with the fence,
    or the legacy constellation deviation with Tukey boxes); flagged sensors in red."""
    aligned = _aligned_mode(cfg)
    col_name = "aligned_mm" if aligned else "deviation"
    chans = [c for c in cfg["channels"]["temperature"] if c in set(spatial["channel"])]
    cohorts = list(spatial.groupby(["species", "status"]).groups)
    fig, axes = plt.subplots(1, len(chans), figsize=(3 * len(chans), 4), sharey=True)
    for ax, ch in zip(np.atleast_1d(axes), chans):
        for i, (sp, st) in enumerate(cohorts):
            s = spatial[(spatial.channel == ch) & (spatial.species == sp) & (spatial.status == st)]
            if not aligned:
                ax.boxplot(s[col_name].dropna(), positions=[i], widths=0.5, whis=1.5, showfliers=False)
            x = np.full(len(s), i) + np.linspace(-0.12, 0.12, len(s))
            ax.scatter(x, s[col_name], c=np.where(s["flagged"], "#c0392b", "#555555"), s=16, zorder=3)
            for xi, (_, b) in zip(x, s.iterrows()):
                if b["flagged"]:
                    ax.annotate(b.marker_eye, (xi, b[col_name]), fontsize=7, xytext=(3, 2),
                                textcoords="offset points")
        if aligned:
            ax.axhline(spatial.loc[spatial.channel == ch, "fence_mm"].iloc[0], color="#c0392b", ls="--", lw=1)
        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels([f"{sp}\n{st}" for sp, st in cohorts], fontsize=8)
        ax.set_title(channel_label(ch, cfg), fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("Aligned distance from mean position (mm)" if aligned
                                      else "Constellation deviation |d/d̄ − 1|")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
