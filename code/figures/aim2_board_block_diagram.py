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
# 2026-11-11: redrawn at the printed width (6.0 in = textwidth, included at width=\textwidth) so that
# the smallest text prints at 7 pt (was 12.5 in wide, text 6.3-8.2 pt, i.e. about 3-4 pt in print). Data units
# are printed points (432 pt = 6.0 in); same boxes, routes, colors and texts, with the texts rewrapped and the
# geometry rescaled to fit them. Saved without a tight bounding box, so the PNG is exactly 6.0 in wide.
W, H = 432, 340
T_TITLE, T_SUB, T_NOTE = 7.5, 7.0, 7.0
LW = 0.8
fig = plt.figure(figsize=(W / 72, H / 72), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(-4, H - 4)
ax.axis("off")


def box(x, y, w, h, title, sub="", fill=FILL, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3", fc=fill, ec=INK, lw=lw, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h - 4, title, ha="center", va="top", fontsize=T_TITLE, fontweight="bold", color=INK, zorder=4)
        n = title.count("\n") + 1
        ax.text(x + w / 2, y + h - 6 - 9.5 * n, sub, ha="center", va="top", fontsize=T_SUB, color=INK, linespacing=1.25, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=T_TITLE, fontweight="bold", color=INK, zorder=4)


def route(pts, color=SIG, ls="-", both=False, head=True):
    xs, ys = zip(*pts)
    n = len(pts) - (1 if head else 0)
    if n > 1:
        ax.plot(xs[:n], ys[:n], color=color, ls=ls, lw=LW, zorder=2, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                    arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=color, lw=LW, ls=ls,
                                    shrinkA=0, shrinkB=0, mutation_scale=6))


def note(x, y, t, color=SIG, ha="center", va="bottom"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=T_NOTE, color=color, linespacing=1.15, zorder=4,
            bbox=dict(fc="white", ec="none", pad=0.6))


# columns (x, width): left outside the board, A, B, C, right outside the board
XL, WL = 7, 56
XA, WA = 75, 64
XB, WB = 154, 70
XC, WC = 261, 64
XR, WR = 364, 67
CA, CC, CR = XA + WA / 2, XC + WC / 2, XR + WR / 2

# board outline
ax.add_patch(FancyBboxPatch((69, 40), 277, 276, boxstyle="round,pad=0,rounding_size=4", fc="none", ec="#888888", lw=0.6, ls=(0, (4, 3)), zorder=1))
ax.text(72, 312, "Integrated control board, revision 1.0", fontsize=T_NOTE, color="#555555", va="top")

# left, outside the board
box(XL, 266, WL, 24, "12 V supply")
box(XL, 149, WL, 36, "Thermistors", "ring (control)\n5 ocular sites")
box(XL, 47, WL, 26, "Operator PC")
# column A
box(XA, 256, WA, 44, "Regulators", "5 V (switching)\nand 3.3 V rails")
box(XA, 140, WA, 54, "Six thermistor\nchannels", "divider from\n2.5 V, digital\ntrim, amplifier")
box(XA, 84, WA, 51, "Ambient\nsensor", "temperature,\nhumidity\n(recorded only)")
# column B
box(XB, 256, WB, 44, "References", "2.5 V and 1.25 V,\nbuffered")
box(XB, 46, WB, 154, "STM32L476RG\nmicrocontroller", "(Nucleo board)\n\n\nPID loop, 5 Hz\n\nsafety routine,\nevery 50 ms\n\nwatchdog", fill=FILL2, lw=1.0)
# column C
box(XC, 256, WC, 44, "Trimmers", "hard limits on\ncurrent and\nvoltage")
box(XC, 152, WC, 44, "ADN8831", "module controller\n(analog loop\nnot used)")
box(XC, 96, WC, 36, "H-bridge", "4 MOSFETs, 5 V,\n10 mΩ sense")
box(XC, 46, WC, 44, "Drivers", "pump and fan,\n12 V, pulse-width\nmodulated")
# right, outside the board
box(XR, 160, WR, 26, "Ring on\nthe eye")
box(XR, 96, WR, 36, "Thermoelectric\nmodule")
box(XR, 74, WR, 12, "Fan")
box(XR, -2, WR, 72, "Coolant loop", "pump\n2 flow and\ncoolant-\ntemperature\nsensors\n2 pressure inputs")

# power
route([(XL + WL, 278), (XA, 278)], PWR)
route([(XA + WA, 278), (XB, 278)], PWR)
route([(XB + WB, 278), (XC, 278)], PWR)
note((XB + WB + XC) / 2, 279, "2.5 V", PWR)
route([(170, 256), (170, 238), (CA, 238), (CA, 194)], PWR)
note(140, 239, "2.5 V", PWR)
route([(XC + WC, 114), (XR, 114)], PWR)
note((XC + WC + XR) / 2, 115, "± current", PWR)
route([(XC + WC, 80), (XR, 80)], PWR)
route([(XC + WC, 52), (XR, 52)], PWR)
note((XC + WC + XR) / 2, 53, "pump", PWR)
# signals
route([(XL + WL, 167), (XA, 167)])
route([(XA + WA, 167), (XB, 167)])
note((XA + WA + XB) / 2, 168, "6")
route([(XA + WA, 113), (XB, 113)])
note((XA + WA + XB) / 2, 114, "I²C")
route([(XL + WL, 60), (XB, 60)], both=True)
note((XL + WL + XB) / 2, 61, "serial link over USB\n(commands, data)")
route([(XB + WB, 172), (XC, 172)])
note((XB + WB + XC) / 2, 173, "shutdown")
route([(XC, 158), (XB + WB, 158)])
note((XB + WB + XC) / 2, 157, "current,\nvoltage", va="top")
route([(CC, 256), (CC, 196)])
route([(CC, 152), (CC, 132)], both=True)
note(CC + 4, 142, "gate drive,\ncurrent sense", ha="left", va="center")
route([(XB + WB, 67), (XC, 67)])
route([(XR, 26), (192, 26), (192, 46)])
note(282, 24.5, "flow, coolant temperature (I²C), pressure", va="top")
# reworks: added
route([(210, 256), (210, 200)], ls=ADDED)
note(212, 232, "2.5 V read", ha="left", va="center")
route([(XB + WB, 184), (XC, 184)], ls=ADDED)
note((XB + WB + XC) / 2, 185, "drive\n(DAC)")
# reworks: removed (ring signal to the analog loop of the ADN8831)
route([(124, 194), (124, 208), (271, 208), (271, 196)], GONE, ls=CUT, head=False)
ax.plot([245], [208], marker="x", ms=4.5, mew=1.0, color=GONE, zorder=5)
note(152, 209, "ring signal to the\nanalog loop (cut)", GONE, ha="center")
# thermal
route([(CR, 132), (CR, 160)], THERM)
note(CR + 4, 146, "heat", THERM, ha="left", va="center")
route([(CR, 186), (CR, 322), (3, 322), (3, 167), (XL, 167)], THERM, ls=DASH)
note(217, 323, "ring temperature (feedback through the ring thermistor)", THERM)

# legend
items = [(SIG, "-", "signal"), (PWR, "-", "power"), (THERM, DASH, "thermal"),
         (SIG, ADDED, "added by rework"), (GONE, CUT, "removed by rework")]
xpos = [7, 60, 112, 172, 258]
for (c, ls, t), x0 in zip(items, xpos):
    ax.plot([x0, x0 + 16], [8, 8], color=c, ls=ls, lw=1.0)
    ax.text(x0 + 19, 8, t, va="center", fontsize=T_NOTE, color=INK)
dst = FIG / "board-block-diagram.png"
fig.savefig(dst, facecolor="white")
sch = FIG / "control-board-schematics.pdf"
record(__file__, [dst], [sch],
       notes="drawn in the script from the schematic sheets and my record of the reworks (2026-10-08); "
             "positions and routes are styling")
print("wrote", dst.name)
