"""Appendix figures: the vitreous sensor between the center of the globe outline and its wall, on micro-CT.

One section per eye through the center of the segmented globe outline and the vitreous sensor (the line from the
center to the wall is drawn horizontal, wall to the right), for the nine human eyes
(vitreous_position_human.png) and the ten porcine eyes (vitreous_position_porcine.png).

Inputs: one small .npz per eye, written by
  03 - Experiments & Data/Ex-Vivo/CT Segmentation/code/vitreous_position_evidence.py
from the micro-CT volumes and the globe outlines (section image with the grey window applied, the outline as the
zero level of `inside`, the center-to-sensor and center-to-wall distances). They are copied unchanged to
figures/appendix/sources/vitreous_position/ and hashed in figures/appendix/MANIFEST.json. The limit drawn is
qc.vitreous_max_centre_fraction of the thesis-analysis config.

Run:  python code/figures/appendix_vitreous_position.py
"""
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from provenance import THESIS, record  # noqa: E402

BOX = THESIS.parents[2]
CT = BOX / "03 - Experiments & Data" / "Ex-Vivo" / "CT Segmentation" / "outputs" / "_vitreous_position_evidence"
CFG = BOX / "03 - Experiments & Data" / "Analysis Code" / "thesis-analysis" / "config" / "config.toml"
OUT = THESIS / "figures" / "appendix"
SRC = OUT / "sources" / "vitreous_position"

HUMAN = [f"H{i}" for i in range(1, 10)]
PORCINE = [f"P{i}" for i in range(11, 16)] + [f"F{i}" for i in range(15, 20)]
NAME = {**{f"H{i}": f"Human {i}" for i in range(1, 10)},
        **{f"P{i}": f"Preserved {i - 10}" for i in range(11, 16)},      # numbering of the thesis text: 1-5 per cohort
        **{f"F{i}": f"Fresh {i - 14}" for i in range(15, 20)}}
VIEW = 15.5  # mm, half width shown


def panel(ax, z, limit):
    t = z["t_mm"]; r, R = float(z["r_mm"]), float(z["R_mm"])
    ax.imshow(z["image"].T, cmap="gray", vmin=0, vmax=255, origin="lower", extent=[t[0], t[-1], t[0], t[-1]])
    ax.contour(t, t, z["inside"].astype(np.float32).T, [0], colors="red", linewidths=0.8)
    ax.plot([0, R], [0, 0], "-", color="yellow", lw=0.8)
    ax.plot(0, 0, "+", color="yellow", ms=6, mew=1.1)
    ax.plot(R, 0, "|", color="yellow", ms=8, mew=1.3)
    out = r / R > limit
    ax.plot(r, 0, "D" if out else "o", mfc="none", mec="cyan", ms=7, mew=1.4)
    ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW); ax.set_xticks([]); ax.set_yticks([])
    ax.plot([-VIEW + 1.5, -VIEW + 6.5], [-VIEW + 1.6, -VIEW + 1.6], "-", color="white", lw=2)
    ax.text(-VIEW + 4.0, -VIEW + 2.4, "5 mm", color="white", fontsize=6.5, ha="center", va="bottom")
    return out


def figure(ids, nrow, ncol, size, path, limit):
    fig, axes = plt.subplots(nrow, ncol, figsize=size)
    for ax, sid in zip(axes.flat, ids):
        z = np.load(SRC / f"{sid}.npz")
        out = panel(ax, z, limit)
        ax.set_title(f"{NAME[sid]}: {100 * float(z['fraction']):.0f}%" + (", excluded" if out else ""), fontsize=8.5, pad=3)
    fig.tight_layout(pad=0.4, h_pad=0.7, w_pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    limit = float(tomllib.load(open(CFG, "rb"))["qc"]["vitreous_max_centre_fraction"])
    snaps = []
    for sid in HUMAN + PORCINE:
        dst = SRC / f"{sid}.npz"
        dst.write_bytes((CT / f"{sid}.npz").read_bytes())
        snaps.append(dst)
    cfg_snap = OUT / "sources" / "thesis-analysis__config.toml"
    cfg_snap.write_bytes(CFG.read_bytes())
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    jobs = [(HUMAN, 3, 3, (6.3, 6.6), OUT / "vitreous_position_human.png"),
            (PORCINE, 2, 5, (6.3, 2.85), OUT / "vitreous_position_porcine.png")]
    for ids, nr, nc, size, path in jobs:
        figure(ids, nr, nc, size, path, limit)
        used = [SRC / f"{sid}.npz" for sid in ids] + [cfg_snap]
        record(__file__, [path], used, manifest=OUT / "MANIFEST.json",
               origins={**{f"{sid}.npz": "Ex-Vivo/CT Segmentation/outputs/_vitreous_position_evidence/" + f"{sid}.npz" for sid in ids},
                        cfg_snap.name: "thesis-analysis/config/config.toml"},
               notes=f"limit = qc.vitreous_max_centre_fraction = {limit:g}; fractions: "
                     + ", ".join(f"{sid} {float(np.load(SRC / (sid + '.npz'))['fraction']):.3f}" for sid in ids))
        print("wrote", path.name)


if __name__ == "__main__":
    main()
