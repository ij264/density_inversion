"""Figures for step 4, the surface field h_s.    python figures/04_surface_field.py

  residual.pdf   observed bedrock elevation minus what the WINTERC-G reference lithosphere predicts
                 (Te = 40 km, land and sea), split at degree 20: (a) degrees 1-20, (b) degrees 21-50, on one
                 colour scale. The short-wavelength part is where h_s has to act.
  spectra.pdf    (a) observed, predicted and residual topography by degree; (b) the residual against two
                 shapes of h_s prior: a plain Sobolev prior, which puts as much power at long wavelengths as at
                 short and so competes with dynamic topography, and the same prior with a high-pass above
                 degree 30, which leaves the long wavelengths to the mantle
Written to figures/output/04_surface_field/.
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
from figstyle import INK, MUTED, BLUE, ORANGE, AQUA  # noqa: E402
from density_inversion.forward import examples as ex  # noqa: E402
from density_inversion.forward.lithosphere import ReferenceLithosphere  # noqa: E402
from density_inversion.forward.mantle import MantleFlow  # noqa: E402
from density_inversion.forward.plate import OceanLoadedPlate  # noqa: E402
from density_inversion.forward.surface import high_pass  # noqa: E402

fsty.apply()
if os.environ.get("CARTOPY_DATA_DIR"):
    cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DATA_DIR"]
OUT = HERE / "output" / "04_surface_field"
OUT.mkdir(parents=True, exist_ok=True)

L, SPLIT, L0 = 50, 20, 30
mf = MantleFlow(lmax=L)
Q = mf.load_space()
ref = ReferenceLithosphere.from_mantle(mf)
P = OceanLoadedPlate.from_mantle(mf, Te=40e3).operator(Q)
apply = lambda op, c: op.codomain.to_components(op(op.domain.from_components(c)))
deg = np.asarray(Q.degrees)

h_obs = ref.field(Q, "bed_elevation")
h_pred = apply(P, ref.reference_load(Q))
h_obs[0] = h_pred[0] = 0.0
resid = h_obs - h_pred
low = np.where(deg <= SPLIT, resid, 0.0)
high = np.where(deg > SPLIT, resid, 0.0)

# ---------------------------------------------------------------- maps of the residual, split by degree
lat, lon, _ = ex.to_grid(resid, L)
LO, LA = np.meshgrid(lon, lat)
grids = [ex.to_grid(c, L)[2] / 1e3 for c in (low, high)]
v = max(np.max(np.abs(g)) for g in grids)
fig = plt.figure(figsize=(9.0, 3.0))
for i, (F, lab) in enumerate(
    zip(
        grids,
        [
            f"Residual, degrees 1-{SPLIT} (km)",
            f"Residual, degrees {SPLIT + 1}-{L} (km)",
        ],
    )
):
    ax = fig.add_subplot(1, 2, i + 1, projection=ccrs.Robinson(central_longitude=150))
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
    ax.set_title(f"({'ab'[i]})", loc="left", fontsize=10)
fig.suptitle(
    "Observed bedrock elevation minus the WINTERC-G reference prediction",
    fontsize=9,
    color=MUTED,
)
fig.savefig(OUT / "residual.pdf")
fig.savefig(OUT / "residual.png")
plt.close(fig)

# ---------------------------------------------------------------- spectra and prior shapes
ls = np.arange(1, L + 1)
amp = lambda c: np.array([np.sqrt(np.sum(c[deg == l] ** 2)) for l in ls]) / 1e3
lam = 0.05
sob = np.sqrt(
    (2 * ls + 1) * (1 + lam**2 * ls * (ls + 1)) ** (-2.0)
)  # per-degree amplitude, order 2
hp = sob * high_pass(L, L0)[1:]
r = amp(resid)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
for c, lab, col in (
    (h_obs, "observed", INK),
    (h_pred, "reference prediction", BLUE),
    (resid, "residual", ORANGE),
):
    axs[0].semilogy(ls, amp(c), color=col, label=lab)
axs[0].legend(fontsize=8)
axs[0].set_title("(a) Topography by degree", loc="left", fontsize=10)
axs[1].semilogy(ls, r, color=ORANGE, label="residual")
for shape, lab, col, style in (
    (sob, "Sobolev prior (order 2)", AQUA, "--"),
    (hp, f"Sobolev x high-pass at $\\ell = {L0}$", BLUE, "-"),
):
    axs[1].semilogy(ls, shape * r[-1] / shape[-1], color=col, ls=style, label=lab)
axs[1].axvspan(1, L0, color=MUTED, alpha=0.08, lw=0)
axs[1].text(2, 0.013, "left to the mantle and lithosphere", fontsize=8, color=MUTED)
axs[1].legend(fontsize=8, loc="lower right")
axs[1].set_title(
    "(b) Candidate h$_s$ priors, scaled at $\\ell = 50$", loc="left", fontsize=10
)
for ax in axs:
    ax.set_xlabel(r"Degree $\ell$")
axs[0].set_ylabel("Amplitude per degree (km)")
axs[0].set_ylim(1e-2, 10)
fig.savefig(OUT / "spectra.pdf")
fig.savefig(OUT / "spectra.png")
plt.close(fig)
rms = lambda c: np.sqrt(np.sum(c[deg >= 1] ** 2) / (4 * np.pi)) / 1e3
print(
    f"residual RMS {rms(resid):.3f} km: degrees 1-{SPLIT} {rms(low):.3f} km, "
    f"{SPLIT + 1}-{L} {rms(high):.3f} km"
)
print("saved to", OUT)
