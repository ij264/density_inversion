"""Figures for step 1, mantle density -> load on the base of the plate.    python figures/01_mantle_flow.py

  kernels.pdf     (a) load kernels K^q_l(r) for the x30 viscosity jump, LAB at 105 km;
                  (b) the same as the unflexed LAB deflection, against a shell running to the surface
  validation.pdf  (a) flowkernels (top at the surface) against the fem_1d kernels (kernels_jump30.npz);
                  (b) relative misfit by degree. Only if DENSITY_INVERSION_FEM_KERNELS points to that file.
  example_blob.pdf       a dense Gaussian blob at 300 km -> load q (MPa) -> unflexed deflection (km)
  example_power_law.pdf  a 1/k random density field (seed 0), the same three panels
The examples are built in density_inversion/forward/examples.py and read no files.
Written to figures/output/01_mantle_flow/.
"""

import os
import sys
from pathlib import Path

import numpy as np
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

fsty.apply()
if os.environ.get(
    "CARTOPY_DATA_DIR"
):  # offline: a folder holding Natural Earth's shapefiles
    cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DATA_DIR"]
FEM_KERNELS = os.environ.get("DENSITY_INVERSION_FEM_KERNELS")
OUT = HERE / "output" / "01_mantle_flow"
OUT.mkdir(parents=True, exist_ok=True)
eta = viscosity_jump(30.0, 670e3)
mf = MantleFlow(viscosity=eta, lab_depth=105e3, lmax=50)
full = MantleFlow(viscosity=eta, lab_depth=1.0, lmax=50, cache=False)
k, kf = mf.kernels, full.kernels
R = float(k["R"])
LS = (2, 5, 10, 20, 40)
COLS = (BLUE, ORANGE, AQUA, YELLOW, MUTED)
depth = lambda kk: (R - kk["r"]) / 1e3

# ---------------------------------------------------------------- kernels
fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True)
for l, c in zip(LS, COLS):
    axs[0].plot(k["K_q"][l - 1] * 1e5 / 1e3, depth(k), color=c, label=rf"$\ell = {l}$")
    axs[1].plot(k["K_h0"][l - 1] * 1e5, depth(k), color=c)
    axs[1].plot(kf["K_h0"][l - 1] * 1e5, depth(kf), color=c, ls=":", lw=1.2)
for ax in axs:
    ax.axhline(670, color=MUTED, lw=0.6, ls="--")
    ax.axhline(105, color=MUTED, lw=0.6)
axs[0].set_ylim(R / 1e3 - float(k["r_cmb"]) / 1e3, 0)
axs[0].set_xlabel(
    "Load on the plate, $K^q_\\ell$\n(kPa per $\\mathrm{kg\\,m^{-3}}$ over 100 km)"
)
axs[1].set_xlabel("Unflexed deflection\n(m per $\\mathrm{kg\\,m^{-3}}$ over 100 km)")
axs[0].set_ylabel(r"Depth of the density anomaly (km)")
axs[0].legend(loc="lower left", fontsize=8)
axs[0].text(
    0.98,
    105,
    "LAB",
    ha="right",
    va="bottom",
    fontsize=8,
    color=MUTED,
    transform=axs[0].get_yaxis_transform(),
)
axs[0].text(
    0.98,
    670,
    r"$\eta \times 30$ below",
    ha="right",
    va="bottom",
    fontsize=8,
    color=MUTED,
    transform=axs[0].get_yaxis_transform(),
)
axs[1].plot([], [], color=INK, label="shell to the LAB")
axs[1].plot([], [], color=INK, ls=":", lw=1.2, label="shell to the surface")
axs[1].legend(loc="lower left", fontsize=8)
for ax, t in zip(axs, "ab"):
    ax.set_title(rf"({t})", loc="left", fontsize=10)
fig.savefig(OUT / "kernels.pdf")
fig.savefig(OUT / "kernels.png")
plt.close(fig)

# ---------------------------------------------------------------- validation against fem_1d
mis = None
if FEM_KERNELS and os.path.exists(FEM_KERNELS):
    fem = np.load(FEM_KERNELS)
    frow = lambda l: list(fem["degrees"]).index(l)
    mis = []
    for l in range(1, 41):
        a = kf["K_h0"][l - 1]
        b = np.interp(kf["r"], fem["r"], fem["K_h_b"][frow(l)])
        mis.append(
            np.sqrt(np.sum(kf["weights"] * (a - b) ** 2) / np.sum(kf["weights"] * b**2))
        )
    fig, axs = plt.subplots(
        1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    for l, c in zip((2, 10, 30), (BLUE, ORANGE, AQUA)):
        axs[0].plot(
            kf["K_h0"][l - 1] * 1e5,
            depth(kf),
            color=c,
            label=rf"flowkernels, $\ell = {l}$",
        )
        axs[0].plot(
            fem["K_h_b"][frow(l)] * 1e5, (R - fem["r"]) / 1e3, color=INK, ls=":", lw=1.1
        )
    axs[0].plot([], [], color=INK, ls=":", lw=1.1, label="fem_1d")
    axs[0].set_ylim(2890, 0)
    axs[0].set_xlabel("Deflection (m per $\\mathrm{kg\\,m^{-3}}$ over 100 km)")
    axs[0].set_ylabel("Depth (km)")
    axs[0].legend(fontsize=8, loc="lower left")
    axs[1].semilogy(range(1, 41), mis, "o-", color=BLUE, ms=3)
    axs[1].set_xlabel(r"Degree $\ell$")
    axs[1].set_ylabel("Relative misfit")
    axs[1].set_ylim(1e-5, 1e-2)
    for ax, t in zip(axs, "ab"):
        ax.set_title(rf"({t})", loc="left", fontsize=10)
    fig.savefig(OUT / "validation.pdf")
    fig.savefig(OUT / "validation.png")
    plt.close(fig)

# ---------------------------------------------------------------- examples: a dense blob and a 1/k field
X = mf.density_space()
A = mf.operator(X)
Q = A.codomain
L = Q.lmax
x_bot, x_top = X.layer_bounds[0][0], X.layer_bounds[-1][1]
cases = [
    ("example_blob", ex.dense_blob(R, depth=300e3, lmax=L), 300),
    ("example_power_law", ex.power_law(R, x_bot, x_top, lmax=L, seed=0), 300),
]
for name, profiles, z in cases:
    c = ex.project(X, profiles)
    q = Q.to_components(A(X.from_components(c)))
    h0 = mf.unflexed_deflection(q)
    rho_z = ex.evaluate_at(X, c, (R - z * 1e3) / R)
    lat, lon, _ = ex.to_grid(q, L)
    LO, LA = np.meshgrid(lon, lat)
    grid = lambda comp: ex.to_grid(comp, L)[2]
    fields = [
        (grid(rho_z), rf"$\delta\rho_m$ at {z} km ($\mathrm{{kg\,m^{{-3}}}}$)"),
        (grid(q) / 1e6, r"Load on the plate, $q$ (MPa)"),
        (grid(h0) / 1e3, r"Unflexed deflection, $q/(\Delta\rho\,g)$ (km)"),
    ]
    fig = plt.figure(figsize=(13.0, 2.9))
    for i, (F, lab) in enumerate(fields):
        v = np.max(np.abs(F))
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
    fig.suptitle(
        profiles.label.replace("kg/m^3", r"$\mathrm{kg\,m^{-3}}$"),
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    G = grid(h0) / 1e3
    print(
        f"{name}: q RMS {np.sqrt(np.sum(q[1:] ** 2) / (4 * np.pi)) / 1e6:.2f} MPa, unflexed deflection "
        f"RMS {np.sqrt(np.sum(h0[1:] ** 2) / (4 * np.pi)) / 1e3:.2f} km, range {G.min():.2f} to {G.max():.2f} km"
    )

if mis is not None:
    print(
        "misfit vs fem_1d, l = 1-40: max %.1e, median %.1e" % (max(mis), np.median(mis))
    )
print("saved to", OUT)
