"""Figure E.1 (figures/appendix/dsc_coolants.png): differential scanning calorimetry of the
candidate coolants (distilled water, 50 % propylene glycol, propylene glycol), 10 C/min ramp.

Inputs: the TA TRIOS exports of the 2024-08-08 runs (Lynch laboratory, operator CP),
`03 - Experiments & Data/Bench Data & Working Notes/Thermal Conductivity DSC/{Distilled water,50%,PG}.xls`,
copied to figures/appendix/sources/dsc/ and hashed in figures/appendix/MANIFEST.json.
(A) normalized heat flow against temperature (endothermic down), from 4 C, after the
start-up transient of the ramp; (B) heat flow relative to water on a 0.1-C grid from 8 to
35 C, with the tabulated specific-heat ratio of propylene glycol to water (2.51/4.18).

Added 2026-10-20: the figure was drawn earlier without a script.

    python code/figures/appendix_dsc_coolants.py
"""
import pathlib
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance import record

THESIS = pathlib.Path(__file__).resolve().parents[2]
OUT = THESIS / "figures" / "appendix"
SRC = OUT / "sources" / "dsc"
RAW = THESIS.parents[2] / "03 - Experiments & Data" / "Bench Data & Working Notes" / "Thermal Conductivity DSC"
FILES = [("Distilled water.xls", "Distilled water", "#0072B2"),
         ("50%.xls", "50% propylene glycol", "#009E73"),
         ("PG.xls", "Propylene glycol", "#D55E00")]
CP_RATIO_PG = 2.51 / 4.18        # tabulated c_p of propylene glycol / water (Appendix E text)


def read_ramp(path):
    xl = pd.read_excel(path, sheet_name=None, header=None)
    sheet = next(v for k, v in xl.items() if k.lower().startswith("ramp"))
    d = sheet.iloc[3:, :3].apply(pd.to_numeric, errors="coerce").dropna()
    d.columns = ["t_min", "T_C", "q_Wg"]
    return d


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    data, snaps = {}, []
    for fname, label, color in FILES:
        dst = SRC / fname
        shutil.copy2(RAW / fname, dst)
        snaps.append(dst)
        d = read_ramp(dst)
        data[label] = d[d.T_C >= 4.0]
    grid = np.arange(8.0, 35.0 + 1e-9, 0.1)
    on = {k: np.interp(grid, v.T_C.to_numpy(), v.q_Wg.to_numpy()) for k, v in data.items()}
    water = on["Distilled water"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for _, label, color in FILES:
        d = data[label]
        a.plot(d.T_C, d.q_Wg, color=color, lw=2, label=label)
    a.set_xlabel("Temperature (°C)")
    a.set_ylabel("Normalized heat flow (W/g), endothermic down")
    a.set_title("A  Heat flow during a 10 °C/min ramp", loc="left", fontweight="bold")
    a.legend(frameon=False, loc="lower left")
    for _, label, color in FILES[1:]:
        b.plot(grid, on[label] / water, color=color, lw=2, label=f"{label}, measured")
    b.axhline(CP_RATIO_PG, color=FILES[2][2], ls="--", lw=1.2, label="Propylene glycol, tabulated $c_p$ ratio")
    b.set_ylim(0, 1)
    b.set_xlabel("Temperature (°C)")
    b.set_ylabel("Ratio to distilled water")
    b.set_title("B  Heat absorbed relative to water", loc="left", fontweight="bold")
    b.legend(frameon=False, loc="lower left", fontsize=9)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUT / "dsc_coolants.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    r = {T: (np.interp(T, grid, on["50% propylene glycol"] / water), np.interp(T, grid, on["Propylene glycol"] / water))
         for T in (10.0, 15.0)}
    pg5 = np.interp(5.0, data["Propylene glycol"].T_C, data["Propylene glycol"].q_Wg) / \
        np.interp(5.0, data["Distilled water"].T_C, data["Distilled water"].q_Wg)
    note = ("ratios to water: 50%% PG / PG at 10 C %.3f / %.3f, at 15 C %.3f / %.3f; PG at 5 C %.3f"
            % (r[10.0][0], r[10.0][1], r[15.0][0], r[15.0][1], pg5))
    record(__file__, [path], snaps, manifest=OUT / "MANIFEST.json",
           origins={s.name: "Bench Data & Working Notes/Thermal Conductivity DSC/" + s.name for s in snaps},
           notes=note)
    print("wrote", path.name, "|", note)


if __name__ == "__main__":
    main()
