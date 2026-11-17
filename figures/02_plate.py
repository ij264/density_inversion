"""Figures for step 2, the elastic plate (load -> deflection).    python figures/02_plate.py

  filter.pdf            the flexural filter F_l drho g against degree for Te = 0, 20, 40, 60 km
  ocean_function.pdf    the ocean function on its 1 degree grid, and as the model sees it (degrees <= 50)
  flexed_blob.pdf       step 1's dense blob: unflexed deflection, flexed (Te = 40 km, water everywhere), difference
  flexed_power_law.pdf  the same for the 1/k field
  land_sea_blob.pdf     flexed with water everywhere, flexed with air on land, difference
  land_sea_power_law.pdf
  land_sea_uniform.pdf  a uniform 1 MPa upward load with no plate: water everywhere against land and sea
Written to figures/output/02_plate/. Reads only the package's own data.
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
from figstyle import INK, MUTED, BLUE, ORANGE, AQUA, YELLOW  # noqa: E402
from density_inversion.forward import examples as ex  # noqa: E402
from density_inversion.forward.mantle import MantleFlow, viscosity_jump  # noqa: E402
from density_inversion.forward.plate import (  # noqa: E402
    ElasticPlate,
    OceanLoadedPlate,
    ocean_function,
    sample_ocean_function,
)

fsty.apply()
if os.environ.get("CARTOPY_DATA_DIR"):
    cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DATA_DIR"]
OUT = HERE / "output" / "02_plate"
OUT.mkdir(parents=True, exist_ok=True)

L = 50
TE = 40e3
mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=L)
k = mf.kernels
R = float(k["R"])
X = mf.density_space()
A = mf.operator(X)
Q = A.codomain
water = ElasticPlate.from_mantle(mf, Te=TE)
land_sea = OceanLoadedPlate.from_mantle(mf, Te=TE)
P_water, P_land_sea = water.operator(Q), land_sea.operator(Q)
apply = lambda op, c: op.codomain.to_components(op(op.domain.from_components(c)))


def maps(name, fields, title, shared=()):
    """Three maps side by side. Panels listed in `shared` (the same quantity) get one common symmetric
    colour scale; the others each get their own."""
    lat, lon, _ = ex.to_grid(fields[0][0], L)
    LO, LA = np.meshgrid(lon, lat)
    grids = [ex.to_grid(comp, L)[2] for comp, _ in fields]
    common = max((np.max(np.abs(grids[i])) for i in shared), default=None)
    fig = plt.figure(figsize=(13.0, 2.9))
    for i, (comp, lab) in enumerate(fields):
        F = grids[i]
        v = common if i in shared else np.max(np.abs(F))
        ax = fig.add_subplot(
            1, 3, i + 1, projection=ccrs.Robinson(central_longitude=150)
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


def rms(c):
    return np.sqrt(np.sum(c[1:] ** 2) / (4 * np.pi))


# ---------------------------------------------------------------- the filter
fig, ax = plt.subplots(figsize=(6.4, 3.6))
lplot = np.arange(1, 301)
for Te, c in zip((0.0, 20e3, 40e3, 60e3), (MUTED, AQUA, BLUE, ORANGE)):
    p = ElasticPlate(Te, water.drho, water.g, R, lplot[-1])
    ax.plot(
        lplot,
        (p.filter() * p.drho * p.g)[1:],
        color=c,
        label=rf"$T_e = {Te / 1e3:g}$ km",
    )
ax.axvline(L, color=MUTED, lw=0.8, ls="--")
ax.text(L * 1.05, 0.08, rf"model $\ell_{{\max}} = {L}$", fontsize=8, color=MUTED)
ax.set_xscale("log")
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"Degree $\ell$")
ax.set_ylabel(r"$F_\ell\,\Delta\rho\,g$ (1 = no flexure)")
ax.legend(fontsize=8, loc="lower left")
fig.savefig(OUT / "filter.pdf")
fig.savefig(OUT / "filter.png")
plt.close(fig)

# ---------------------------------------------------------------- the ocean function
olat, olon, C = ocean_function()
# expand on a Gauss-Legendre grid sampled from the 1 degree cells (as the plate does)
zero, w = pysh.expand.SHGLQ(3 * L)
glat, glon = pysh.expand.GLQGridCoord(3 * L)
GLA, GLO = np.meshgrid(glat, glon, indexing="ij")
c_sh = ex.cilm_to_components(
    pysh.expand.SHExpandGLQ(
        sample_ocean_function(GLA, GLO), w, zero, norm=4, csphase=1, lmax_calc=L
    ),
    L,
)
fig = plt.figure(figsize=(9.0, 2.9))
for i, (lat, lon, F, lab) in enumerate(
    [
        (olat, olon, C, r"Ocean function $C$, $1^\circ$ cells"),
        (*ex.to_grid(c_sh, L)[:2], ex.to_grid(c_sh, L)[2], rf"$C$ to degree {L}"),
    ]
):
    LO, LA = np.meshgrid(lon, lat)
    ax = fig.add_subplot(1, 2, i + 1, projection=ccrs.Robinson(central_longitude=150))
    im = ax.pcolormesh(
        LO,
        LA,
        F,
        transform=ccrs.PlateCarree(),
        cmap="Blues",
        vmin=-0.1,
        vmax=1.1,
        shading="auto",
        rasterized=True,
    )
    ax.coastlines(lw=0.4, color=INK)
    ax.set_global()
    cb = fig.colorbar(
        im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04, aspect=30
    )
    cb.set_label(lab, fontsize=9)
    ax.set_title(f"({'ab'[i]})", loc="left", fontsize=10)
fig.savefig(OUT / "ocean_function.pdf")
fig.savefig(OUT / "ocean_function.png")
plt.close(fig)

# ---------------------------------------------------------------- the examples
x_bot, x_top = X.layer_bounds[0][0], X.layer_bounds[-1][1]
cases = [
    ("blob", ex.dense_blob(R, depth=300e3, lmax=L)),
    ("power_law", ex.power_law(R, x_bot, x_top, lmax=L, seed=0)),
]
for name, profiles in cases:
    q = apply(A, ex.project(X, profiles))
    h0 = mf.unflexed_deflection(q)
    hw = apply(P_water, q)
    hl = apply(P_land_sea, q)
    label = profiles.label.replace("kg/m^3", r"$\mathrm{kg\,m^{-3}}$")
    maps(
        f"flexed_{name}",
        [
            (h0 / 1e3, "Unflexed deflection (km)"),
            (hw / 1e3, rf"Flexed, $T_e = {TE / 1e3:g}$ km, water everywhere (km)"),
            ((hw - h0) / 1e3, "Flexed minus unflexed (km)"),
        ],
        label,
        shared=(0, 1, 2),
    )
    maps(
        f"land_sea_{name}",
        [
            (hw / 1e3, "Water everywhere (km)"),
            (hl / 1e3, "Water at sea, air on land (km)"),
            ((hl - hw) / 1e3, "Land/sea minus water everywhere (km)"),
        ],
        label + rf", $T_e = {TE / 1e3:g}$ km",
        shared=(0, 1, 2),
    )
    print(
        f"{name}: RMS unflexed {rms(h0) / 1e3:.3f} km, flexed {rms(hw) / 1e3:.3f} km "
        f"(difference {rms(hw - h0):.1f} m); land/sea {rms(hl) / 1e3:.3f} km "
        f"(difference from water everywhere {rms(hl - hw) / 1e3:.3f} km)"
    )

# a uniform upward load with no plate: land rises less
q = np.zeros(Q.dim)
q[0] = 1e6 * np.sqrt(4 * np.pi)
no_plate_w = ElasticPlate.from_mantle(mf, Te=0.0).operator(Q)
no_plate_ls = OceanLoadedPlate.from_mantle(mf, Te=0.0).operator(Q)
a, b = apply(no_plate_w, q), apply(no_plate_ls, q)
maps(
    "land_sea_uniform",
    [
        (a, "Water everywhere (m)"),
        (b, "Water at sea, air on land (m)"),
        (b - a, "Difference (m)"),
    ],
    "A uniform upward load of 1 MPa, no plate",
    shared=(0, 1, 2),
)
rho, rw = float(k["rho_lab"]), float(k["rho_water"])
print(
    f"uniform 1 MPa, no plate: water everywhere {1e6 / ((rho - rw) * water.g):.1f} m, "
    f"air {1e6 / (rho * water.g):.1f} m (ratio {(rho - rw) / rho:.3f})"
)
print("saved to", OUT)
