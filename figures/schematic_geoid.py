"""Slide schematic: the four masses that make the geoid, and their sum (a cartoon, not to scale).
    python figures/schematic_geoid.py   -> figures/output/schematic/geoid_contributions.{pdf,png,svg}

Five panels on the same cross-section (ocean on the left, a continent with a crustal root on the right, a dense
blob in the mantle, which pulls the surface down and pushes the CMB down). Each panel highlights one mass, marks its sign, and draws the geoid it makes above the
surface; the last panel adds them up. The curves are illustrative Gaussians chosen so that the parts nearly
cancel, as in the computed figure 05_geoid/contributions.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fsty  # noqa: E402

fsty.apply()
OUT = HERE / "output" / "schematic"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, BLUE, RED = "#0b0b0b", "#8a8984", "#2a78d6", "#c8432f"
WATER, CRUST, LID, MANTLE, CORE = "#cfe3f3", "#c49a6c", "#a7a29a", "#f2c4a0", "#d9d6cf"
FADE = 0.28

x = np.linspace(0, 10, 500)
g = lambda x0, w: np.exp(-0.5 * ((x - x0) / w) ** 2)
step = lambda x0, w: 0.5 * (1 + np.tanh((x - x0) / w))

# cross-section (display units): surface near 0, Moho, flow top -1.5, CMB -6
BLOB_X, BLOB_Y, ROOT_X = 3.0, -3.6, 7.6
dip = 0.35 * g(BLOB_X, 0.9)  # the plate pulled down over the blob
lift = 0.30 * g(ROOT_X, 0.7) + 0.25 * step(6.0, 0.4)  # pushed up over the root and the continent
surface = -0.25 + lift - dip
moho_ref = -0.55 - 0.45 * step(6.0, 0.4)  # thin oceanic, thick continental crust
moho = moho_ref - 0.35 * g(ROOT_X, 0.45) + lift - dip  # the root, and the column moving with the plate
TOP, CMB = -1.5, -6.0
cmb = CMB - 0.25 * g(BLOB_X, 1.2)  # the CMB pushed down beneath the sinking blob (as flowkernels' cmbtopo)

# geoid curves (arbitrary units)
N_mantle = 1.0 * g(BLOB_X, 1.1) - 0.15 * g(BLOB_X, 2.2)  # the blob's high, less the CMB's broad low
N_lith = -0.35 * g(ROOT_X, 0.5)
N_ref = -0.75 * step(6.0, 0.5)
N_defl = -0.9 * g(BLOB_X, 1.0) + 0.33 * g(ROOT_X, 0.55) + 0.7 * step(6.0, 0.5)
parts = [
    ("Mantle density + CMB", N_mantle, "mantle"),
    ("Lithospheric perturbations", N_lith, "lith"),
    ("Deflected plate", N_defl, "plate"),
    ("Reference lithosphere", N_ref, "ref"),
]
total = sum(p[1] for p in parts)
parts.append(("Total geoid", total, "total"))


def section(ax, focus):
    """The cross-section, with everything but the focused mass faded."""
    a = lambda on: 1.0 if (focus == "total" or on) else FADE
    ax.fill_between(x, CMB - 0.9, cmb, color=CORE, lw=0, alpha=a(focus == "mantle"))
    ax.fill_between(x, cmb, TOP, color=MANTLE, lw=0, alpha=a(focus == "mantle"))
    ax.fill_between(x, TOP, moho, color=LID, lw=0, alpha=a(focus in ("plate", "ref")))
    ax.fill_between(x, moho, surface, color=CRUST, lw=0, alpha=a(focus in ("lith", "plate", "ref")))
    ax.fill_between(x, surface, 0, where=surface < 0, color=WATER, lw=0, alpha=a(focus == "plate"))
    ax.plot(x, surface, color=INK, lw=1.0, alpha=a(focus in ("plate", "ref")))
    ax.plot(x, moho, color=INK, lw=0.8, alpha=a(focus in ("lith", "plate", "ref")))
    ax.plot(x, np.full_like(x, TOP), color=BLUE, lw=1.0, ls=(0, (5, 3)), alpha=a(focus == "mantle"))
    ax.plot(x, cmb, color=INK, lw=1.0, alpha=a(focus == "mantle"))
    ax.add_patch(Ellipse((BLOB_X, BLOB_Y), 0.8, 0.53, fc="#7a3a1a",
                         ec=INK, lw=0.8, alpha=a(focus == "mantle")))


def sign(ax, xy, s, color):
    """A plus or minus drawn with lines (the slide font has no minus glyph)."""
    xs, ys = xy
    dx, dy = 0.22, 0.11
    ax.plot([xs - dx, xs + dx], [ys, ys], color=color, lw=2.2, solid_capstyle="butt")
    if s == "+":
        ax.plot([xs, xs], [ys - dy * 2.0, ys + dy * 2.0], color=color, lw=2.2, solid_capstyle="butt")


fig, axs = plt.subplots(1, 5, figsize=(15.0, 4.4), sharey=True)
for ax, (title, N, focus) in zip(axs, parts):
    section(ax, focus)
    base = 1.25
    ax.axhline(base, color=MUTED, lw=0.6)
    col = INK if focus == "total" else BLUE
    ax.fill_between(x, base, base + N, where=N >= 0, color=RED, alpha=0.25, lw=0)
    ax.fill_between(x, base, base + N, where=N < 0, color=BLUE, alpha=0.25, lw=0)
    ax.plot(x, base + N, color=col, lw=2.0 if focus == "total" else 1.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(CMB - 0.9, 2.5)
    ax.axis("off")

# what each mass is
ax = axs[0]
sign(ax, (BLOB_X, BLOB_Y), "+", "white")
sign(ax, (BLOB_X, CMB - 0.05), "-", BLUE)
ax.add_patch(FancyArrowPatch((BLOB_X, BLOB_Y - 0.7), (BLOB_X, BLOB_Y - 1.4), arrowstyle="-|>",
                             mutation_scale=12, color=INK, lw=1.2))
ax.text(BLOB_X + 0.7, BLOB_Y, "dense,\nsinking", fontsize=8.5, va="center")
ax.text(BLOB_X + 1.0, CMB + 0.35, "CMB pushed down:\nmantle for core", fontsize=8.5, va="center")

ax = axs[1]
sign(ax, (ROOT_X, moho[np.argmin(abs(x - ROOT_X))] + 0.25), "-", BLUE)
ax.text(ROOT_X - 0.2, -2.35, "crustal root:\nlight crust\nfor dense mantle", fontsize=8.5, ha="center",
        va="top")

ax = axs[2]
sign(ax, (BLOB_X, -0.3), "-", BLUE)
sign(ax, (ROOT_X + 0.9, 0.15), "+", RED)
ax.text(BLOB_X - 0.3, -2.3, "pulled down\nover the blob:\nwater for rock", fontsize=8.5, ha="center",
        va="top")
ax.text(ROOT_X + 0.2, -2.3, "pushed up over\nthe continent\nand the root", fontsize=8.5, ha="center",
        va="top")

ax = axs[3]
sign(ax, (8.0, -0.5), "-", BLUE)
ax.text(5.0, -2.3, "thick, light continental crust\nagainst the global mean", fontsize=8.5,
        ha="center", va="top")

ax = axs[4]
ax.text(5.0, -2.3, "a small remainder\nof large terms", fontsize=8.5, ha="center", va="top")

axs[0].text(0.1, 1.4, "geoid", fontsize=8.5, color=MUTED)
axs[0].text(0.1, TOP + 0.08, "flow top", fontsize=7.5, color=BLUE, va="bottom")
axs[0].text(0.1, CMB + 0.1, "CMB", fontsize=7.5, color=INK, va="bottom")
for i in range(4):
    fig.text((i + 1) / 5 + 0.002, 0.55, "+" if i < 3 else "=", fontsize=18, ha="center", va="center",
             color=MUTED)
fig.text(0.01, 0.02, "Cartoon, not to scale. Each panel highlights one mass; red geoid = high, blue = low.",
         fontsize=8, color=MUTED)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.06, wspace=0.08)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"geoid_contributions.{ext}", dpi=220)
print("saved", OUT)
