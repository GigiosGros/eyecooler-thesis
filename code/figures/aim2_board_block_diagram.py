"""Fig. fig:aim2-board-diagram: block diagram of the integrated control board, revision 1.0, as used.

Drawn by hand in this script from the five schematic sheets (figures/aim2/control-board-schematics.pdf:
Power, ADN8831, Peripherals, Thermistor Channels, Microcontroller) and my own record of the reworks
(2026-10-08): the ring signal to the analog loop of the ADN8831 was cut and the hardware compensation
network removed, the drive is set digitally by the microcontroller, and the 2.5 V reference is read by
the microcontroller. The shutdown line is original. Box positions and line routes are styling.

Run:  python code/figures/aim2_board_block_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from provenance import FIG, record

INK, SIG, PWR, THERM, FILL, FILL2, GONE = "#1a1a1a", "#444444", "#b35900", "#1f5fa8", "#f3f3f3", "#e3ecf7", "#8a8a8a"
DASH, ADDED, CUT = (0, (5, 3)), (0, (6, 2, 1.5, 2)), (0, (1.2, 2.2))
fig, ax = plt.subplots(figsize=(12.5, 7.4), dpi=200)
ax.set_xlim(-5, 129)
ax.set_ylim(-4, 72)
ax.axis("off")


def box(x, y, w, h, title, sub="", fill=FILL, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fill, ec=INK, lw=lw, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=8.2, fontweight="bold", color=INK, zorder=4)
        n = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 1.8 - 2.5 * n, sub, ha="center", va="top", fontsize=6.8, color=INK, linespacing=1.3, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK, zorder=4)


def route(pts, color=SIG, ls="-", both=False, head=True):
    xs, ys = zip(*pts)
    n = len(pts) - (1 if head else 0)
    if n > 1:
        ax.plot(xs[:n], ys[:n], color=color, ls=ls, lw=1.3, zorder=2, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                    arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color, lw=1.3, ls=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=9))


def note(x, y, t, color=SIG, ha="center", va="bottom"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=6.3, color=color, linespacing=1.2, zorder=4,
            bbox=dict(fc="white", ec="none", pad=0.6))


# board outline
ax.add_patch(FancyBboxPatch((22, 5), 80, 61.5, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec="#888888", lw=1.0, ls=(0, (4, 3)), zorder=1))
ax.text(23.5, 65.3, "Integrated control board, revision 1.0", fontsize=7.8, color="#555555", va="top")

# left, outside the board
box(0, 53, 16, 8, "12 V supply")
box(0, 31, 16, 13, "Thermistors", "ring (control)\n5 ocular sites")
box(0, 8, 16, 8, "Operator PC")
# column A
box(26, 53, 18, 8, "Regulators", "5 V (switching)\nand 3.3 V rails")
box(26, 31, 18, 13, "Six thermistor\nchannels", "divider from 2.5 V,\ndigital trim, amplifier")
box(26, 18, 18, 9, "Ambient sensor", "temperature, humidity\n(recorded only)")
# column B
box(53, 53, 18, 8, "References", "2.5 V and 1.25 V,\nbuffered")
box(53, 8, 18, 36, "STM32L476RG\nmicrocontroller", "(Nucleo board)\n\n\nPID loop, 5 Hz\n\nsafety routine,\nevery 50 ms\n\nwatchdog", fill=FILL2, lw=1.6)
# column C
box(81, 53, 17, 8, "Trimmers", "hard limits on\ncurrent and voltage")
box(81, 33, 17, 11, "ADN8831", "module controller\n(analog loop not used)")
box(81, 20, 17, 9, "H-bridge", "4 MOSFETs, 5 V,\n10 mΩ sense")
box(81, 6, 17, 11, "Drivers", "pump and fan,\n12 V, pulse-width\nmodulated")
# right, outside the board
box(108, 34, 18, 9, "Ring on the eye")
box(108, 20, 18, 9, "Thermoelectric\nmodule")
box(108, 12.5, 18, 5, "Fan")
box(108, -3, 18, 12.5, "Coolant loop", "pump\n2 flow and coolant-\ntemperature sensors\n2 pressure inputs")

# power
route([(16, 57), (26, 57)], PWR)
route([(44, 57), (53, 57)], PWR)
route([(71, 57), (81, 57)], PWR)
note(76, 57.8, "2.5 V", PWR)
route([(57, 53), (57, 50), (37, 50), (37, 44)], PWR)
note(47, 50.8, "2.5 V", PWR)
route([(98, 24.5), (108, 24.5)], PWR)
note(103, 25.3, "± current", PWR)
route([(98, 15), (108, 15)], PWR)
route([(98, 8), (108, 8)], PWR)
note(103, 8.8, "pump", PWR)
# signals
route([(16, 37.5), (26, 37.5)])
route([(44, 37.5), (53, 37.5)])
note(48.5, 38.3, "6")
route([(44, 22.5), (53, 22.5)])
note(48.5, 23.3, "I²C")
route([(16, 12), (53, 12)], both=True)
note(34.5, 12.8, "serial link over USB (commands, data)")
route([(71, 38.5), (81, 38.5)])
note(76, 39.0, "shutdown")
route([(81, 35), (71, 35)])
note(76, 30.6, "current,\nvoltage")
route([(89.5, 53), (89.5, 44)])
route([(89.5, 33), (89.5, 29)], both=True)
note(91, 31, "gate drive,\ncurrent sense", ha="left", va="center")
route([(71, 11.5), (81, 11.5)])
route([(108, 0.5), (65, 0.5), (65, 8)])
note(86, 1.3, "flow, coolant temperature (I²C), pressure")
# reworks: added
route([(66, 53), (66, 44)], ls=ADDED)
note(67.2, 49.6, "2.5 V read", ha="left", va="center")
route([(71, 42), (81, 42)], ls=ADDED)
note(76, 42.5, "drive (DAC)")
# reworks: removed (ring signal to the analog loop of the ADN8831)
route([(41, 44), (41, 47), (84, 47), (84, 44)], GONE, ls=CUT, head=False)
ax.plot([62.5], [47], marker="x", ms=7, mew=1.6, color=GONE, zorder=5)
note(49.5, 47.6, "ring signal to the analog loop (cut)", GONE, ha="center")
# thermal
route([(117, 29), (117, 34)], THERM)
note(118.2, 31.5, "heat", THERM, ha="left", va="center")
route([(117, 43), (117, 69.5), (-3, 69.5), (-3, 37.5), (0, 37.5)], THERM, ls=DASH)
note(57, 70.2, "ring temperature (feedback through the ring thermistor)", THERM)

# legend
items = [(SIG, "-", "signal"), (PWR, "-", "power"), (THERM, DASH, "thermal"),
         (SIG, ADDED, "added by rework"), (GONE, CUT, "removed by rework")]
xpos = [0, 13, 26, 40, 62]
for (c, ls, t), x0 in zip(items, xpos):
    ax.plot([x0, x0 + 4.5], [-1.5, -1.5], color=c, ls=ls, lw=1.6)
    ax.text(x0 + 5.5, -1.5, t, va="center", fontsize=7.0, color=INK)
dst = FIG / "board-block-diagram.png"
fig.savefig(dst, bbox_inches="tight", facecolor="white")
sch = FIG / "control-board-schematics.pdf"
record(__file__, [dst], [sch],
       notes="drawn in the script from the schematic sheets and my record of the reworks (2026-10-08); "
             "positions and routes are styling")
print("wrote", dst.name)
