"""Slide schematic of how the forward problem models the Earth (not to scale).
python figures/schematic.py   -> figures/output/schematic/earth_model.{pdf,png,svg}"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fsty  # noqa: E402

fsty.apply()
OUT = HERE / "output" / "schematic"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#0b0b0b", "#52514e"

A0, A1 = np.radians(48), np.radians(132)  # wedge from 48 to 132 degrees
th = np.linspace(A0, A1, 400)

# display radii (exaggerated near the surface; not to scale)
R_S, R_SED, R_UC, R_LC = 1.0, 0.984, 0.962, 0.938
R_LAB0 = 0.865  # mean LAB: the top of the flow
R_670, R_CMB = 0.74, 0.47
lab = (
    R_LAB0 + 0.028 * np.sin(5.2 * (th - A0)) + 0.014 * np.cos(11 * (th - A0) + 0.6)
)  # laterally varying LAB
surf = R_S + 0.012 * np.exp(
    -0.5 * ((th - np.radians(74)) / np.radians(2.5)) ** 2
)  # an orogen


def ring(r_out, r_in):
    r_out = np.broadcast_to(r_out, th.shape)
    r_in = np.broadcast_to(r_in, th.shape)
    xs = np.r_[r_out * np.cos(th), (r_in * np.cos(th))[::-1]]
    ys = np.r_[r_out * np.sin(th), (r_in * np.sin(th))[::-1]]
    return np.c_[xs, ys]


def arc(r, **kw):
    r = np.broadcast_to(r, th.shape)
    return plt.plot(r * np.cos(th), r * np.sin(th), **kw)


fig, ax = plt.subplots(figsize=(10, 5.0))
ax.set_aspect("equal")
ax.axis("off")

layers = [
    (surf, R_SED, "#e9d7a8"),
    (R_SED, R_UC, "#d9b98c"),
    (R_UC, R_LC, "#c49a6c"),
    (R_LC, lab, "#a7a29a"),
    (lab, R_670, "#f2c4a0"),
    (R_670, R_CMB, "#e7a57c"),
    (R_CMB, 0.0, "#cfcfcf"),
]
for ro, ri, col in layers:
    ax.add_patch(Polygon(ring(ro, ri), closed=True, fc=col, ec="none"))
for r in (R_SED, R_UC, R_LC):
    arc(r, color=INK, lw=0.6, ls=(0, (4, 3)))
arc(surf, color=INK, lw=1.1)
arc(lab, color=INK, lw=1.8)
arc(R_LAB0, color="#2a78d6", lw=1.3, ls=(0, (6, 4)))
arc(R_670, color=INK, lw=0.8, ls=(0, (2, 2)))
arc(R_CMB, color=INK, lw=1.1)
for a in (A0, A1):
    ax.plot([0, np.cos(a)], [0, np.sin(a)], color=INK, lw=1.1)


# flow cells and the load on the plate
def cell(cx_deg, r, w, h, cw):
    t = np.radians(cx_deg)
    c = np.array([r * np.cos(t), r * np.sin(t)])
    e = np.linspace(0, 2 * np.pi, 200)
    tang, norm = np.array([-np.sin(t), np.cos(t)]), np.array([np.cos(t), np.sin(t)])
    pts = c[:, None] + tang[:, None] * w * np.cos(e) + norm[:, None] * h * np.sin(e)
    ax.plot(*pts[:, 20:180], color="#9b4a1c", lw=1.1)
    i0, i1 = (175, 182) if cw else (182, 175)
    ax.add_patch(
        FancyArrowPatch(
            pts[:, i0],
            pts[:, i1],
            arrowstyle="-|>",
            mutation_scale=10,
            color="#9b4a1c",
            lw=1.1,
        )
    )


cell(104, 0.71, 0.10, 0.075, True)
cell(76, 0.71, 0.10, 0.075, False)
t = np.radians(97)
ax.add_patch(
    FancyArrowPatch(
        (0.79 * np.cos(t), 0.79 * np.sin(t)),
        (0.855 * np.cos(t), 0.855 * np.sin(t)),
        arrowstyle="-|>",
        mutation_scale=14,
        color="#2a78d6",
        lw=2,
    )
)
ax.text(
    0.79 * np.cos(t) - 0.02,
    0.80 * np.sin(t),
    r"load $q$",
    color="#2a78d6",
    fontsize=10,
    ha="right",
    va="center",
)
ax.text(
    0.0,
    0.575,
    r"$\delta\rho_m$ drives flow",
    color="#9b4a1c",
    fontsize=9.5,
    ha="center",
)

# labels on the right; leaders start on the wedge's right edge at each feature's radius
xl = np.cos(A0) + 0.16
a_edge = A0 + np.radians(1.5)
rows = [
    (
        1.0,
        r"Surface field $h_s$: orogens, short wavelengths",
        "added to the topography",
        INK,
    ),
    (
        (R_SED + R_LC) / 2,
        r"Sediment, upper and lower crust",
        r"densities $\rho_L$; interfaces perturbed by $\delta r_i$",
        INK,
    ),
    (
        (R_LC + 0.885) / 2,
        r"Lithospheric mantle",
        r"elastic plate, $T_e = 40$ km, flexes under every load",
        INK,
    ),
    (
        float(np.interp(a_edge, th, lab)),
        r"LAB $R(\theta,\phi)$ = reference $+\,\delta R$",
        "an interface load on the plate",
        INK,
    ),
    (R_LAB0, r"Top of the flow: a sphere at the mean LAB", "free slip", "#2a78d6"),
    (
        (R_LAB0 + R_670) / 2 - 0.02,
        r"Viscous mantle: $\delta\rho_m(r,\theta,\phi)$, $\eta(r)$",
        r"flowkernels gives the load $q$ on the plate",
        INK,
    ),
    (R_670, r"670 km: viscosity $\times 30$ below", None, INK),
    (R_CMB, r"Core-mantle boundary, free slip", None, INK),
]
ys = np.linspace(1.0, 0.44, len(rows))
for (r_at, text, sub, col), y in zip(rows, ys):
    p = (r_at * np.cos(a_edge), r_at * np.sin(a_edge))
    ax.plot([p[0], xl - 0.02], [p[1], y], color=MUTED, lw=0.6)
    ax.plot(*p, "o", ms=2.5, color=MUTED)
    ax.text(xl, y + (0.012 if sub else 0), text, fontsize=10.5, va="center", color=col)
    if sub:
        ax.text(xl, y - 0.026, sub, fontsize=8.8, va="center", color=MUTED)

ax.text(
    xl,
    0.31,
    r"$h = \mathsf{F}_{T_e}\,q[\delta\rho_m;\eta] + \mathsf{F}_{T_e}\,\mathsf{L}[\rho_L,\delta r_i] + h_s$",
    ha="left",
    fontsize=12.5,
    color=INK,
)
ax.text(
    xl,
    0.255,
    r"geoid $N$: the potential of every mass",
    ha="left",
    fontsize=10,
    color=MUTED,
)
ax.text(np.cos(A1), 0.27, "not to scale", fontsize=8.5, color=MUTED, ha="left")
ax.set_xlim(np.cos(A1) - 0.02, xl + 0.86)
ax.set_ylim(0.24, 1.04)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"earth_model.{ext}", dpi=220)
print("saved", OUT)
