"""Figures for step 6, the assembled forward model.    python figures/06_forward_model.py

One synthetic Earth through every step: the WINTERC-G reference lithosphere; a 1/k mantle density field (RMS
3 kg/m^3, which gives dynamic topography of realistic size, ~0.7 km RMS); lithospheric perturbations (a 10 km crustal root under Tibet, a depleted lid under the Canadian
Shield); a random surface field with power only above degree 30 (RMS 150 m). Te = 40 km, water at sea, air on
land, full geoid.

  operator.pdf     the block structure of the forward operator: which step feeds which observable
  topography.pdf   (a-d) dynamic, lithospheric, reference and surface contributions, (e) the total, (f) by
                   degree; one colour scale for the maps
  geoid.pdf        (a-d) mantle + CMB, lithospheric perturbations, the deflected plate, the reference
                   lithosphere's own masses, (e) the total, (f) by degree; one colour scale for the maps
Written to figures/output/06_forward_model/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pyshtools as pysh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import cartopy
import cartopy.crs as ccrs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fsty  # noqa: E402
from figstyle import INK, MUTED, BLUE, ORANGE, AQUA, YELLOW  # noqa: E402
from density_inversion.forward import examples as ex  # noqa: E402
from density_inversion.forward.mantle import MantleFlow  # noqa: E402
from density_inversion.forward.model import ForwardModel  # noqa: E402
from density_inversion.forward.surface import high_pass  # noqa: E402

fsty.apply()
if os.environ.get("CARTOPY_DATA_DIR"):
    cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DATA_DIR"]
OUT = HERE / "output" / "06_forward_model"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- the operator's block structure
STEP_COL = {1: "#f2c4a0", 2: "#bcd9f0", 3: "#c49a6c", 4: "#d9d6cf", 5: "#cfe8c8"}
fig, ax = plt.subplots(figsize=(10.5, 4.7))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 4.75)
ax.axis("off")
cols = [
    (r"mantle density $\delta\rho_m$", "step 1"),
    (r"lithosphere $x_L$", "step 3"),
    (r"surface field $h_s$", "step 4"),
]
rows = [(r"topography $h$", 2.55), (r"geoid $N$", 0.75)]
blocks = {
    (0, 0): (r"$P\,A_m$", "plate (2) after flow (1)", [2, 1]),
    (0, 1): (r"$P\,A_L$", "plate (2) after loads (3)", [2, 3]),
    (0, 2): (r"$I$", "added as it is (4)", [4]),
    (1, 0): (r"$G_m + G_w P A_m$", "anomaly + CMB, deflected plate (5)", [5, 1]),
    (1, 1): (r"$G_L + G_w P A_L$", "mass sheets, deflected plate (5)", [5, 3]),
    (1, 2): (r"$0$", "no mass, no geoid", [4]),
}
x0, w, h = 2.2, 2.65, 1.45
for j, (name, step) in enumerate(cols):
    ax.text(x0 + j * (w + 0.12) + w / 2, 4.5, name, ha="center", fontsize=10.5)
    ax.text(
        x0 + j * (w + 0.12) + w / 2, 4.22, step, ha="center", fontsize=8.5, color=MUTED
    )
for i, (name, y) in enumerate(rows):
    ax.text(2.05, y + h / 2, name, ha="right", va="center", fontsize=10.5)
    for j in range(3):
        f, desc, steps = blocks[(i, j)]
        xb = x0 + j * (w + 0.12)
        ax.add_patch(
            FancyBboxPatch(
                (xb, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=STEP_COL[steps[0]],
                ec=INK,
                lw=0.8,
                alpha=0.9,
            )
        )
        ax.text(xb + w / 2, y + h * 0.62, f, ha="center", va="center", fontsize=13)
        ax.text(
            xb + w / 2,
            y + h * 0.25,
            desc,
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK,
        )
ax.text(
    0.0,
    0.18,
    r"plus fixed reference terms: $h_{\mathrm{ref}} = P\,q_{\mathrm{ref}}$, "
    r"$N_{\mathrm{ref}} = N_{\mathrm{ref}}^{\mathrm{structure}} + G_w P\,q_{\mathrm{ref}}$"
    "   (data: total topography, full geoid)",
    fontsize=9.5,
    color=INK,
)
fig.savefig(OUT / "operator.pdf")
fig.savefig(OUT / "operator.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- the synthetic Earth
L = 50
mf = MantleFlow(lmax=L)
fm = ForwardModel(mf, Te=40e3)
Q, X = fm.Q, fm.X
R = float(mf.kernels["R"])
deg = np.asarray(Q.degrees)


def gaussian_cap(lat, lon, width_km, amplitude):
    glat, glon = ex.dh_lat_lon(2 * (4 * L + 1))
    GLA, GLO = np.meshgrid(np.radians(glat), np.radians(glon), indexing="ij")
    la, lo = np.radians(lat), np.radians(lon)
    cosd = np.sin(GLA) * np.sin(la) + np.cos(GLA) * np.cos(la) * np.cos(GLO - lo)
    grid = amplitude * np.exp(
        -0.5 * (R * np.arccos(np.clip(cosd, -1, 1)) / (width_km * 1e3)) ** 2
    )
    return ex.cilm_to_components(
        pysh.expand.SHExpandDH(grid, norm=4, sampling=2, csphase=1, lmax_calc=L), L
    )


x_bot, x_top = X.layer_bounds[0][0], X.layer_bounds[-1][1]
x_m = ex.project(X, ex.power_law(R, x_bot, x_top, rms=3.0, lmax=L, seed=0))
ref = fm.ref
x_L = np.zeros(fm.lithosphere_space().dim)
put = lambda label, comp: x_L.__setitem__(
    slice(ref.index(label) * Q.dim, (ref.index(label) + 1) * Q.dim), comp
)
put("dMoho", gaussian_cap(32.0, 88.0, 400.0, -10e3))
craton = gaussian_cap(60.0, 260.0, 900.0, 1.0)
for n, value in enumerate((-30.0, -15.0, 0.0)):
    put(f"drho lid node {n}", value * craton)
rng = np.random.default_rng(7)
x_s = rng.standard_normal(Q.dim) * high_pass(L, 30)[deg] / np.sqrt(2 * deg + 1)
x_s *= 150.0 / np.sqrt(np.sum(x_s**2) / (4 * np.pi))

parts = fm.parts(x_m, x_L, x_s)
rms = lambda c, lmin: np.sqrt(np.sum(c[deg >= lmin] ** 2) / (4 * np.pi))
ls = np.arange(1, L + 1)
per_degree = lambda c: np.array([np.sqrt(np.sum(c[deg == l] ** 2)) for l in ls])


def panel_figure(name, entries, unit, scale, lmin, title, ymin):
    lat, lon, _ = ex.to_grid(entries[0][0], L)
    LO, LA = np.meshgrid(lon, lat)
    grids = [ex.to_grid(c, L)[2] / scale for c, _ in entries]
    for g, (c, _) in zip(grids, entries):
        if lmin > 0:
            g -= ex.to_grid(np.where(deg < lmin, c, 0.0), L)[2] / scale
    v = max(np.max(np.abs(g)) for g in grids)
    fig = plt.figure(figsize=(13.0, 6.4))
    for i, (F, (c, lab)) in enumerate(zip(grids, entries)):
        ax = fig.add_subplot(
            2, 3, i + 1, projection=ccrs.Robinson(central_longitude=150)
        )
        im = ax.pcolormesh(
            LO,
            LA,
            F,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-v,
            vmax=v,
            shading="auto",
            rasterized=True,
        )
        ax.coastlines(lw=0.4, color=INK)
        ax.set_global()
        ax.set_title(
            f"({'abcde'[i]}) {lab}, RMS {rms(c, max(lmin, 1)) / scale:.2f} {unit}",
            loc="left",
            fontsize=9.5,
        )
    cax = fig.add_axes([0.09, 0.07, 0.52, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{title} ({unit}), one scale for all five maps", fontsize=9)
    ax = fig.add_subplot(2, 3, 6)
    for (c, lab), col in zip(entries, (BLUE, AQUA, YELLOW, ORANGE, INK)):
        y = per_degree(c) / scale
        ax.semilogy(
            ls[ls >= max(lmin, 1)],
            y[ls >= max(lmin, 1)],
            color=col,
            lw=2.0 if col == INK else 1.3,
            label=lab,
        )
    ax.set_xlabel(r"Degree $\ell$")
    ax.set_ylim(bottom=ymin)
    ax.set_ylabel(f"Amplitude per degree ({unit})")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("(f) By degree", loc="left", fontsize=9.5)
    fig.subplots_adjust(
        left=0.03, right=0.95, top=0.93, bottom=0.14, wspace=0.12, hspace=0.25
    )
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


t = parts["topography"]
panel_figure(
    "topography",
    [
        (t["dynamic"], "Dynamic"),
        (t["lithosphere"], "Lithospheric perturbations"),
        (t["reference"], "Reference lithosphere"),
        (t["surface"], "Surface field"),
        (t["total"], "Total topography"),
    ],
    "km",
    1e3,
    1,
    "Topography",
    1e-2,
)
g = parts["geoid"]
panel_figure(
    "geoid",
    [
        (g["mantle"], "Mantle density + CMB"),
        (g["lithosphere"], "Lithospheric perturbations"),
        (g["deflection"], "Deflected plate, all loads"),
        (g["reference"], "Reference lithosphere"),
        (g["total"], "Total geoid"),
    ],
    "m",
    1.0,
    2,
    "Geoid",
    1e-2,
)
print(
    "topography RMS (km): "
    + ", ".join(f"{k} {rms(v, 1) / 1e3:.3f}" for k, v in t.items())
)
print("geoid RMS (m): " + ", ".join(f"{k} {rms(v, 2):.1f}" for k, v in g.items()))
print("saved to", OUT)
