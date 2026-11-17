"""Slide schematic: a zoom into the lithosphere as the forward problem models it (not to scale).
    python figures/schematic_lithosphere.py   -> figures/output/schematic/lithosphere.{pdf,png,svg}

An ocean on the left, a continent with a mountain belt on the right. One crustal layer (bedrock to Moho), a
mantle lid from the Moho to the flow top (a sphere at 105 km), WINTERC-G's LAB crossing the flow top, the
nodes that carry the density unknowns, the perturbations, and the loads on the plate.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fsty  # noqa: E402

fsty.apply()
OUT = HERE / "output" / "schematic"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, BLUE, RUST = "#0b0b0b", "#52514e", "#2a78d6", "#9b4a1c"
WATER, CRUST, LID, MANTLE = "#cfe3f3", "#c49a6c", "#a7a29a", "#f2c4a0"

x = np.linspace(0, 10, 600)
smooth = lambda x0, w: 0.5 * (1 + np.tanh((x - x0) / w))
bump = lambda x0, w: np.exp(-0.5 * ((x - x0) / w) ** 2)

# display units: y up, sea level 0, flow top at -8 (not to scale)
surface = -0.9 + 1.5 * smooth(4.6, 0.35) + 1.1 * bump(8.0, 0.55)  # ocean floor, margin, mountain belt
moho = -1.7 - 2.3 * smooth(4.6, 0.45) - 1.0 * bump(8.0, 0.6)  # thin oceanic crust, thick crust, root
lab = -6.3 - 3.4 * smooth(5.0, 0.8) + 0.25 * np.sin(2.1 * x)  # shallow under the ocean, deep under the continent
TOP, BOTTOM = -8.0, -11.0

fig, ax = plt.subplots(figsize=(11.5, 6.2))
ax.set_xlim(-0.2, 15.4)
ax.set_ylim(-12.4, 2.6)
ax.axis("off")

# ---------------------------------------------------------------- layers
ax.fill_between(x, TOP, BOTTOM, color=MANTLE, lw=0)
ax.fill_between(x, moho, TOP, color=LID, lw=0)
ax.fill_between(x, surface, moho, color=CRUST, lw=0)
ax.fill_between(x, np.minimum(surface, 0), 0, where=surface < 0, color=WATER, lw=0)
ax.plot(x, surface, color=INK, lw=1.4)
ax.plot(x[surface < 0.02], np.zeros((surface < 0.02).sum()), color=BLUE, lw=0.8)
ax.plot(x, moho, color=INK, lw=1.4)
ax.plot(x, lab, color=INK, lw=1.8, ls=(0, (5, 2)))
ax.axhline(TOP, xmin=0.0125, xmax=0.66, color=BLUE, lw=1.6, ls=(0, (7, 4)))
ax.plot([0, 0, 10, 10], [2.0, BOTTOM, BOTTOM, 2.0], color=MUTED, lw=0.6)

# ---------------------------------------------------------------- the spherical reference
# dotted: the spheres at the mean depths, which the reference structure departs from (its load is q_ref) and
# which the unknowns perturb; the mean LAB (104.8 km in WINTERC-G) is the flow top
REF = "#3d3d3a"
for y_ref, name in ((surface.mean(), "mean surface"), (moho.mean(), "mean Moho (23 km)")):
    ax.plot([0, 10], [y_ref, y_ref], color=REF, lw=1.0, ls=(0, (1, 2.2)))
    ax.text(9.95, y_ref + 0.12, name, fontsize=8, color=REF, ha="right", va="bottom")
ax.text(3.0, TOP + 0.12, "mean LAB = flow top (105 km)", fontsize=8, color=BLUE, ha="left", va="bottom")
i3 = np.argmin(abs(x - 3.3))
ax.add_patch(FancyArrowPatch((3.3, moho.mean()), (3.3, moho[i3]), arrowstyle="<|-|>", mutation_scale=8,
                             color=REF, lw=0.9, shrinkA=0, shrinkB=0))
ax.text(3.42, 0.5 * (moho.mean() + moho[i3]), r"reference departure $\rightarrow q_{\mathrm{ref}}$",
        fontsize=8.5, color=REF, va="center")

# ---------------------------------------------------------------- a Moho perturbation (thicker crust)
seg = (x > 6.6) & (x < 7.4)
dm = moho[seg] - 0.55 * np.sin(np.pi * (x[seg] - 6.6) / 0.8)
ax.plot(x[seg], dm, color=RUST, lw=1.4, ls=(0, (2, 1.5)))
ax.add_patch(FancyArrowPatch((7.0, moho[np.argmin(abs(x - 7.0))]), (7.0, dm[np.argmin(abs(x[seg] - 7.0))] + 0.02),
                             arrowstyle="-|>", mutation_scale=9, color=RUST, lw=1.1))
ax.text(7.12, dm.min() + 0.05, r"$\delta r_{\mathrm{Moho}}$", color=RUST, fontsize=10, va="top")

# ---------------------------------------------------------------- nodes in two columns
for xc in (2.0, 8.6):
    i = np.argmin(abs(x - xc))
    s, m = surface[i], moho[i]
    ax.plot([xc, xc], [s, TOP], color=INK, lw=0.5, alpha=0.5)
    ax.plot(xc, 0.5 * (s + m), "s", color=INK, ms=6)  # one crustal value
    for yn in (m, 0.5 * (m + TOP), TOP):  # three GLL nodes across the lid
        ax.plot(xc, yn, "o", color="white", mec=INK, mew=1.4, ms=7)
i = np.argmin(abs(x - 8.6))
ax.text(8.42, 0.5 * (surface[i] + moho[i]), r"$\delta\rho_c$", fontsize=10.5, va="center", ha="right")
for k, yn in enumerate((moho[i], 0.5 * (moho[i] + TOP), TOP)):
    ax.text(8.42, yn + 0.28, rf"$\delta\rho_{{{k}}}$", fontsize=10.5, va="bottom", ha="right")

# ---------------------------------------------------------------- flow and the load on the plate
for xc, sgn in ((2.6, 1), (6.4, -1)):
    t = np.linspace(0.15, 2 * np.pi - 0.15, 120)
    ax.plot(xc + 1.1 * np.cos(t), -9.6 + 0.75 * sgn * np.sin(t), color=RUST, lw=1.0)
ax.text(4.5, -10.6, r"$\delta\rho_m$ drives flow (step 1)", color=RUST, fontsize=10, ha="center")
for xa in (1.0, 3.6, 5.6, 8.0):
    ax.add_patch(FancyArrowPatch((xa, TOP - 0.85), (xa, TOP - 0.08), arrowstyle="-|>", mutation_scale=11,
                                 color=BLUE, lw=1.6))
ax.text(1.12, TOP - 0.5, r"$q_m$", color=BLUE, fontsize=11, va="center")

# water fills the deflection at sea, air on land
ax.text(0.15, -0.45, r"water fills the deflection: $\rho_{\mathrm{LAB}} - \rho_w$", fontsize=8.5,
        color=BLUE, va="center")
ax.text(6.0, 1.65, r"air on land: $\rho_{\mathrm{LAB}}$", fontsize=8.5, color=MUTED, va="center")

# ---------------------------------------------------------------- labels on the right
xl = 10.7
rows = [
    (9.4, surface[np.argmin(abs(x - 9.4))], r"Surface = plate deflection $w$, plus $h_s$ (orogens)", None),
    (9.4, 0.5 * (surface[np.argmin(abs(x - 9.4))] + moho[np.argmin(abs(x - 9.4))]),
     r"Crust: one layer, $\rho_c(x)$ from WINTERC-G", r"unknown $\delta\rho_c(x)$, uniform in depth"),
    (9.4, moho[np.argmin(abs(x - 9.4))], r"Moho, contrast $\rho_{\mathrm{lid}} - \rho_c \approx 300$-$650$",
     r"unknown $\delta r_{\mathrm{Moho}}(x)$"),
    (9.4, -7.0, r"Mantle lid, Moho to the flow top", r"unknowns $\delta\rho_0, \delta\rho_1, \delta\rho_2$ at GLL nodes"),
    (9.4, lab[np.argmin(abs(x - 9.4))], r"LAB (WINTERC-G), jump $\Delta\rho_{\mathrm{LAB}} \approx 8$ kg m$^{-3}$",
     r"unknown $\delta R_{\mathrm{LAB}}(x)$; may lie below the flow top"),
    (9.4, TOP, r"Flow top: a sphere at 105 km, free slip", r"step 3 above, step 1 below"),
    (9.4, -10.2, r"Viscous mantle: $\delta\rho_m$, $\eta(r)$", r"its load on the plate is $q_m$"),
]
rows.sort(key=lambda r: -r[1])  # by depth, so the leaders do not cross
ys = np.linspace(1.4, -10.6, len(rows))
for (xa, ya, title, sub), y in zip(rows, ys):
    ax.plot([xa, xl - 0.12], [ya, y], color=MUTED, lw=0.6)
    ax.plot(xa, ya, "o", color=MUTED, ms=2.5)
    ax.text(xl, y + (0.22 if sub else 0), title, fontsize=10, va="center", color=INK)
    if sub:
        ax.text(xl, y - 0.42, sub, fontsize=8.6, va="center", color=MUTED)

# ---------------------------------------------------------------- the load and the plate
ax.text(0.0, -11.75,
        r"$q_L = -g\,[\,t_c\,\delta\rho_c + \sum_n w_n\,\delta\rho_n + \Delta\rho_{\mathrm{Moho}}\,\delta r_{\mathrm{Moho}}"
        r" + \Delta\rho_{\mathrm{LAB}}\,\delta R_{\mathrm{LAB}}\,]$",
        fontsize=11.5, color=INK)
ax.text(8.2, -11.75,
        r"$(D\nabla^4 + g\,[\rho_{\mathrm{LAB}} - \rho_w C(x)])\,w = q_m + q_L + q_{\mathrm{ref}}$",
        fontsize=11.5, color=INK)
ax.text(0.0, 2.25, "Not to scale; dotted lines: the spherical reference", fontsize=8, color=MUTED)
ax.text(1.0, 1.0, "Ocean", fontsize=11, color=INK, fontweight="bold")
ax.text(6.2, 2.15, "Continent", fontsize=11, color=INK, fontweight="bold")

for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"lithosphere.{ext}", dpi=220, bbox_inches="tight")
print("saved", OUT)
