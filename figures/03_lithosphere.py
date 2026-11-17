"""Figures for step 3, lithospheric loads (stages 3a and 3b, on the placeholder mean structure).
    python figures/03_lithosphere.py

  nodes.pdf        (a) the layers and their GLL nodes; (b) the Lagrange functions across the lithospheric mantle
  moho_root.pdf    a 10 km crustal root under Tibet: the Moho perturbation, topography with no plate, with
                   Te = 40 km (both with water at sea and air on land)
  craton.pdf       a depleted cratonic root under the Canadian Shield, -30 kg/m^3 at the Moho tapering to 0 at
                   the LAB: the anomaly at the top node, the load, the topography with Te = 40 km
Written to figures/output/03_lithosphere/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pyshtools as pysh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fsty  # noqa: E402
from figstyle import INK, MUTED, BLUE, ORANGE, AQUA  # noqa: E402
from density_inversion.forward import examples as ex  # noqa: E402
from density_inversion.forward.lithosphere import (
    Lithosphere,
    lagrange,
    gll,
)  # noqa: E402
from density_inversion.forward.mantle import MantleFlow  # noqa: E402
from density_inversion.forward.plate import OceanLoadedPlate  # noqa: E402

fsty.apply()
if os.environ.get("CARTOPY_DATA_DIR"):
    cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DATA_DIR"]
OUT = HERE / "output" / "03_lithosphere"
OUT.mkdir(parents=True, exist_ok=True)

L = 50
mf = MantleFlow(lmax=L)
Q = mf.load_space()
lith = Lithosphere.placeholder(mf)
A = lith.operator(Q)
plates = {Te: OceanLoadedPlate.from_mantle(mf, Te=Te).operator(Q) for Te in (0.0, 40e3)}
apply = lambda op, c: op.codomain.to_components(op(op.domain.from_components(c)))


def gaussian_cap(lat, lon, width_km, amplitude):
    """amplitude * exp(-d^2 / 2 width^2), d the great-circle distance to (lat, lon): sphere components."""
    n = 2 * (4 * L + 1)
    glat, glon = ex.dh_lat_lon(n)
    LA, LO = np.meshgrid(np.radians(glat), np.radians(glon), indexing="ij")
    la, lo = np.radians(lat), np.radians(lon)
    cosd = np.sin(LA) * np.sin(la) + np.cos(LA) * np.cos(la) * np.cos(LO - lo)
    d = mf.kernels["R"] * np.arccos(np.clip(cosd, -1, 1))
    grid = amplitude * np.exp(-0.5 * (d / (width_km * 1e3)) ** 2)
    return ex.cilm_to_components(
        pysh.expand.SHExpandDH(grid, norm=4, sampling=2, csphase=1, lmax_calc=L), L
    )


def model_vector(fields):
    x = np.zeros(A.domain.dim)
    for label, comp in fields.items():
        i = lith.index(label)
        x[i * Q.dim : (i + 1) * Q.dim] = comp
    return x


def maps(name, fields, title, centre=150):
    lat, lon, _ = ex.to_grid(fields[0][0], L)
    LO, LA = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(13.0, 2.9))
    for i, (comp, lab) in enumerate(fields):
        F = ex.to_grid(comp, L)[2]
        v = np.max(np.abs(F))
        ax = fig.add_subplot(
            1, 3, i + 1, projection=ccrs.Robinson(central_longitude=centre)
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
        cb = fig.colorbar(
            im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04, aspect=30
        )
        cb.set_label(lab, fontsize=9)
        ax.set_title(f"({'abc'[i]})", loc="left", fontsize=10)
    fig.suptitle(title, fontsize=9, color=MUTED)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def peak(comp):
    g = ex.to_grid(comp, L)[2]
    return g.flat[np.argmax(np.abs(g))]


# ---------------------------------------------------------------- the layers and nodes
fig, axs = plt.subplots(
    1, 2, figsize=(9.0, 3.8), gridspec_kw={"width_ratios": [1, 1.3]}
)
ax = axs[0]
cols = ["#e9d7a8", "#d9b98c", "#c49a6c", "#a7a29a"]
for layer, c in zip(lith.layers, cols):
    ax.axhspan(layer.top / 1e3, layer.bottom / 1e3, color=c)
    thin = layer.thickness < 5e3
    y = -2.0 if thin else (layer.top + layer.bottom) / 2e3
    text = f"{layer.name} ({layer.top / 1e3:g}-{layer.bottom / 1e3:g} km), {layer.density:g}"
    ax.text(0.03, y, text, va="center", fontsize=8.5, color=INK)
    depths, _ = lith.nodes(layer)
    ax.plot(np.full(depths.size, 0.9), depths / 1e3, "o", color=INK, ms=4, zorder=3)
for name in lith.interface_names:
    d, drho = lith.interface(name)
    ax.axhline(d / 1e3, color=BLUE, lw=1.4)
    ax.text(
        0.03,
        d / 1e3 + 1.0,
        rf"$\delta r$ here, $\Delta\rho = {drho:+.0f}$",
        ha="left",
        va="top",
        fontsize=8,
        color=BLUE,
    )
ax.set_ylim(lith.layers[-1].bottom / 1e3 + 5, -5)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_ylabel("Depth (km)")
ax.set_title("(a)", loc="left", fontsize=10)
ax = axs[1]
layer = lith.layers[-1]
depths, weights = lith.nodes(layer)
z = np.linspace(layer.top, layer.bottom, 200)
x = 2 * (z - layer.top) / layer.thickness - 1
phi = lagrange(gll(layer.nodes)[0], x)
for n, c in zip(range(layer.nodes), (BLUE, ORANGE, AQUA)):
    ax.plot(
        phi[:, n],
        z / 1e3,
        color=c,
        label=rf"$\ell_{n}$, node at {depths[n] / 1e3:.0f} km, weight {weights[n] / 1e3:.1f} km",
    )
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_ylim(layer.bottom / 1e3 + 5, 0)
ax.set_xlabel("Lagrange function across the lithospheric mantle")
ax.legend(fontsize=7.5, loc="lower right")
ax.set_title("(b)", loc="left", fontsize=10)
fig.savefig(OUT / "nodes.pdf")
fig.savefig(OUT / "nodes.png")
plt.close(fig)

# ---------------------------------------------------------------- a crustal root (3a)
root = gaussian_cap(32.0, 88.0, 400.0, -10e3)  # Moho 10 km deeper under Tibet
q = apply(A, model_vector({"dbase of lower crust": root}))
h0, h40 = apply(plates[0.0], q), apply(plates[40e3], q)
maps(
    "moho_root",
    [
        (root / 1e3, r"Moho perturbation $\delta r$ (km)"),
        (h0 / 1e3, "Topography, no plate (km)"),
        (h40 / 1e3, r"Topography, $T_e = 40$ km (km)"),
    ],
    "A 10 km crustal root under Tibet (Gaussian, width 400 km); water at sea, air on land",
    centre=90,
)
_, drho_moho = lith.interface("base of lower crust")
print(
    f"moho root: peak dr {peak(root) / 1e3:.2f} km; topography no plate {peak(h0) / 1e3:.3f} km, "
    f"Te = 40 km {peak(h40) / 1e3:.3f} km; local isostasy on land would give "
    f"{10 * drho_moho / float(mf.kernels['rho_lab']):.3f} km"
)

# ---------------------------------------------------------------- a depleted craton (3b)
cap = gaussian_cap(60.0, 260.0, 900.0, 1.0)
top, bottom = lith.layers[-1].top, lith.layers[-1].bottom
profile = (
    lambda d: -30.0 * (bottom - d) / (bottom - top)
)  # -30 at the Moho, 0 at the LAB
fields = {
    f"drho lithospheric mantle node {n}": profile(d) * cap for n, d in enumerate(depths)
}
q = apply(A, model_vector(fields))
h40 = apply(plates[40e3], q)
maps(
    "craton",
    [
        (
            profile(depths[0]) * cap,
            r"$\delta\rho$ at the Moho node ($\mathrm{kg\,m^{-3}}$)",
        ),
        (q / 1e6, "Load on the plate (MPa)"),
        (h40, r"Topography, $T_e = 40$ km (m)"),
    ],
    r"A depleted root under the Canadian Shield: $-30\ \mathrm{kg\,m^{-3}}$ at the Moho, 0 at the LAB",
    centre=260,
)
print(
    f"craton: peak load {peak(q) / 1e6:.2f} MPa, peak topography {peak(h40):.0f} m "
    f"(column mass {-30 * (bottom - top) / 2:.3g} kg/m^2)"
)
print("saved to", OUT)
