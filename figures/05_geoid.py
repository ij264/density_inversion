"""Figures for step 5, the geoid.    python figures/05_geoid.py

  kernels.pdf            (a) the mantle part's geoid kernel (anomaly + CMB), shell to the 105 km flow top;
                         (b) the total with the plate's deflection added (no plate, water everywhere), against
                         flowkernels' own geoid kernel for a shell running to the surface
  blob_geoid.pdf         the dense blob of step 1: mantle part, deflection part, total (Te = 40 km, land and sea)
  reference_geoid.pdf    the reference lithosphere's geoid, that of the deflection it causes, and their sum
  compensation.pdf       a crustal root with no plate: the root alone against root + uplift, by degree
  spectra.pdf            each part of the geoid by degree, for the 1/k field and the reference lithosphere
  contributions.pdf      one synthetic Earth, the geoid of each of the four masses and the total: (1) a 1/k mantle
                         field, (2) lithospheric perturbations (a 10 km crustal root under Tibet, a depleted root
                         under the Canadian Shield), (3) the plate's deflection under all loads, (4) the WINTERC-G
                         reference lithosphere; (f) each by degree. One colour scale for all five maps
Written to figures/output/05_geoid/.
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
from density_inversion.forward.geoid import Geoid  # noqa: E402
from density_inversion.forward.lithosphere import ReferenceLithosphere  # noqa: E402
from density_inversion.forward.mantle import MantleFlow, viscosity_jump  # noqa: E402
from density_inversion.forward.plate import OceanLoadedPlate  # noqa: E402

fsty.apply()
if os.environ.get("CARTOPY_DATA_DIR"):
    cartopy.config["pre_existing_data_dir"] = os.environ["CARTOPY_DATA_DIR"]
OUT = HERE / "output" / "05_geoid"
OUT.mkdir(parents=True, exist_ok=True)

L = 50
mf = MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=L)
k = mf.kernels
R = float(k["R"])
X = mf.density_space()
Q = mf.load_space()
A = mf.operator(X, Q)
ref = ReferenceLithosphere.from_mantle(mf)
geo = Geoid(mf, ref)
P = OceanLoadedPlate.from_mantle(mf, Te=40e3).operator(Q)
G_m, G_w, G_L = (
    geo.mantle_operator(X, Q),
    geo.deflection_operator(Q),
    geo.lithosphere_operator(Q),
)
apply = lambda op, c: op.codomain.to_components(op(op.domain.from_components(c)))
deg = np.asarray(Q.degrees)
rms = lambda c: np.sqrt(np.sum(c[deg >= 2] ** 2) / (4 * np.pi))
per_degree = lambda c: np.array(
    [np.sqrt(np.sum(c[deg == l] ** 2)) for l in range(2, L + 1)]
)


def maps(name, fields, title, shared=(), centre=150):
    """Three maps side by side; panels in `shared` (the same quantity) get one colour scale."""
    lat, lon, _ = ex.to_grid(fields[0][0], L)
    LO, LA = np.meshgrid(lon, lat)
    grids = [ex.to_grid(c, L)[2] for c, _ in fields]
    common = max((np.max(np.abs(grids[i])) for i in shared), default=None)
    fig = plt.figure(figsize=(13.0, 2.9))
    for i, (_, lab) in enumerate(fields):
        F = grids[i]
        v = common if i in shared else np.max(np.abs(F))
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


# ---------------------------------------------------------------- kernels
LS, COLS = (2, 5, 10, 20), (BLUE, ORANGE, AQUA, YELLOW)
full = MantleFlow(
    viscosity=viscosity_jump(30.0, 670e3), lab_depth=1.0, lmax=L, cache=False
)
kf = full.kernels
depth = lambda kk: (R - kk["r"]) / 1e3
r_top, g_top = float(k["r_lab"]), float(k["g_lab"])
# no plate, water everywhere: deflection mass = q / g, spread over the column's contrasts (global means)
LA, LO = np.meshgrid(ref.data["lat"], ref.data["lon"], indexing="ij")
wlat = np.cos(np.radians(LA))
geo_water = Geoid(mf, ref, ocean=lambda la, lo: np.ones_like(la))
sheets = [
    (d, float(np.sum(wlat * f) / np.sum(wlat)))
    for d, f in geo_water.deflection_sheets(LA, LO)
]
total_contrast = sum(f for _, f in sheets)
fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
for l, c in zip(LS, COLS):
    S = sum(
        f / total_contrast * geo.sheet_factor(Q, R - d)[deg == l][0] for d, f in sheets
    )
    K_total = k["K_N"][l - 1] + S * k["K_q"][l - 1] / g_top
    axs[0].plot(k["K_N"][l - 1] * 1e5, depth(k), color=c, label=rf"$\ell = {l}$")
    axs[1].plot(K_total * 1e5, depth(k), color=c)
    axs[1].plot(kf["K_N_fk"][l - 1] * 1e5, depth(kf), color=c, ls=":", lw=1.2)
for ax in axs:
    ax.axhline(670, color=MUTED, lw=0.6, ls="--")
    ax.axhline(105, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
axs[0].set_ylim(R / 1e3 - float(k["r_cmb"]) / 1e3, 0)
axs[0].set_xlabel("Geoid, anomaly + CMB\n(m per $\\mathrm{kg\\,m^{-3}}$ over 100 km)")
axs[1].set_xlabel(
    "Geoid, with the deflected plate\n(m per $\\mathrm{kg\\,m^{-3}}$ over 100 km)"
)
axs[0].set_ylabel("Depth of the density anomaly (km)")
axs[0].legend(fontsize=8, loc="lower left")
axs[1].plot([], [], color=INK, label="this model (no plate)")
axs[1].plot(
    [], [], color=INK, ls=":", lw=1.2, label="flowkernels, shell to the surface"
)
axs[1].legend(fontsize=8, loc="lower left")
for ax, t in zip(axs, "ab"):
    ax.set_title(f"({t})", loc="left", fontsize=10)
fig.savefig(OUT / "kernels.pdf")
fig.savefig(OUT / "kernels.png")
plt.close(fig)

# ---------------------------------------------------------------- the blob
c_blob = ex.project(X, ex.dense_blob(R, depth=300e3, lmax=L))
N_m = apply(G_m, c_blob)
N_w = apply(G_w, apply(P, apply(A, c_blob)))
maps(
    "blob_geoid",
    [
        (N_m, "Mantle part: anomaly + CMB (m)"),
        (N_w, "Deflected plate (m)"),
        (N_m + N_w, "Total geoid (m)"),
    ],
    "The dense blob of step 1 (+30 kg m$^{-3}$, 300 km); viscosity x30 below 670 km; $T_e = 40$ km, land and sea",
    shared=(0, 1, 2),
)
print(
    f"blob: geoid RMS mantle {rms(N_m):.1f} m, deflection {rms(N_w):.1f} m, total {rms(N_m + N_w):.1f} m"
)

# ---------------------------------------------------------------- the reference lithosphere
N_ref = geo.reference(Q)
N_ref_w = apply(G_w, apply(P, ref.reference_load(Q)))
maps(
    "reference_geoid",
    [
        (N_ref, "Reference lithosphere (m)"),
        (N_ref_w, "Its deflection of the plate (m)"),
        (N_ref + N_ref_w, "Sum (m)"),
    ],
    "WINTERC-G lithosphere: its own masses and the uplift they cause nearly cancel",
    shared=(0, 1, 2),
)
print(
    f"reference: geoid RMS structure {rms(N_ref):.0f} m, deflection {rms(N_ref_w):.0f} m, "
    f"sum {rms(N_ref + N_ref_w):.1f} m"
)

# ---------------------------------------------------------------- compensation of a crustal root
A_L = ref.operator(Q)
P0 = OceanLoadedPlate.from_mantle(mf, Te=0.0).operator(Q)
i = ref.index("dMoho")
ratio = []
for l in range(2, L + 1):
    x = np.zeros(A_L.domain.dim)
    x[i * Q.dim : (i + 1) * Q.dim] = np.where(
        (deg == l) & (np.asarray(Q.orders) == 0), -1e3, 0.0
    )
    xs = A_L.domain.from_components(x)
    unc = Q.to_components(G_L(xs))
    com = unc + apply(G_w, Q.to_components(P0(A_L(xs))))
    m = deg == l
    ratio.append(np.linalg.norm(com[m]) / np.linalg.norm(unc[m]))
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.semilogy(range(2, L + 1), ratio, color=BLUE)
ax.set_xlabel(r"Degree $\ell$")
ax.set_ylabel("Geoid of root + uplift / root alone")
ax.set_title("A 1 km crustal root with no plate (Airy)", fontsize=9, color=MUTED)
fig.savefig(OUT / "compensation.pdf")
fig.savefig(OUT / "compensation.png")
plt.close(fig)
print(
    f"compensation ratio at l = 2, 10, 30: {ratio[0]:.3f}, {ratio[8]:.3f}, {ratio[28]:.3f}"
)

# ---------------------------------------------------------------- spectra
x_bot, x_top = X.layer_bounds[0][0], X.layer_bounds[-1][1]
c_pl = ex.project(X, ex.power_law(R, x_bot, x_top, lmax=L, seed=0))
N_pl_m = apply(G_m, c_pl)
N_pl_w = apply(G_w, apply(P, apply(A, c_pl)))
ls = np.arange(2, L + 1)
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
for c, lab, col in (
    (N_pl_m, "mantle part", BLUE),
    (N_pl_w, "deflected plate", ORANGE),
    (N_pl_m + N_pl_w, "total", INK),
):
    axs[0].semilogy(ls, per_degree(c), color=col, label=lab)
for c, lab, col in (
    (N_ref, "reference lithosphere", BLUE),
    (N_ref_w, "its deflection", ORANGE),
    (N_ref + N_ref_w, "sum", INK),
):
    axs[1].semilogy(ls, per_degree(c), color=col, label=lab)
axs[0].set_title("(a) 1/k mantle field, RMS 10 kg m$^{-3}$", loc="left", fontsize=10)
axs[1].set_title("(b) WINTERC-G lithosphere", loc="left", fontsize=10)
for ax in axs:
    ax.set_xlabel(r"Degree $\ell$")
    ax.legend(fontsize=8)
axs[0].set_ylabel("Geoid amplitude per degree (m)")
fig.savefig(OUT / "spectra.pdf")
fig.savefig(OUT / "spectra.png")
plt.close(fig)


# ---------------------------------------------------------------- the four masses and the total
def gaussian_cap(lat, lon, width_km, amplitude):
    """amplitude * exp(-d^2 / 2 width^2), d the great-circle distance to (lat, lon): sphere components."""
    glat, glon = ex.dh_lat_lon(2 * (4 * L + 1))
    GLA, GLO = np.meshgrid(np.radians(glat), np.radians(glon), indexing="ij")
    la, lo = np.radians(lat), np.radians(lon)
    cosd = np.sin(GLA) * np.sin(la) + np.cos(GLA) * np.cos(la) * np.cos(GLO - lo)
    d = R * np.arccos(np.clip(cosd, -1, 1))
    grid = amplitude * np.exp(-0.5 * (d / (width_km * 1e3)) ** 2)
    return ex.cilm_to_components(
        pysh.expand.SHExpandDH(grid, norm=4, sampling=2, csphase=1, lmax_calc=L), L
    )


# (2) lithospheric perturbations in the step-3c unknowns
x_L = np.zeros(A_L.domain.dim)
put = lambda label, comp: x_L.__setitem__(
    slice(ref.index(label) * Q.dim, (ref.index(label) + 1) * Q.dim), comp
)
put("dMoho", gaussian_cap(32.0, 88.0, 400.0, -10e3))  # crust 10 km thicker under Tibet
craton = gaussian_cap(60.0, 260.0, 900.0, 1.0)
for n, value in enumerate(
    (-30.0, -15.0, 0.0)
):  # depleted lid: -30 at the Moho, 0 at the flow top
    put(f"drho lid node {n}", value * craton)
xs_L = A_L.domain.from_components(x_L)

q_total = apply(A, c_pl) + Q.to_components(A_L(xs_L)) + ref.reference_load(Q)
parts = [
    (N_pl_m, "Mantle density + CMB"),
    (Q.to_components(G_L(xs_L)), "Lithospheric perturbations"),
    (apply(G_w, apply(P, q_total)), "Deflected plate, all loads"),
    (N_ref, "Reference lithosphere"),
]
total = sum(c for c, _ in parts)
parts.append((total, "Total geoid"))

lat, lon, _ = ex.to_grid(total, L)
LO, LA = np.meshgrid(lon, lat)
grids = [ex.to_grid(c, L)[2] for c, _ in parts]
v = max(np.max(np.abs(g)) for g in grids)
fig = plt.figure(figsize=(13.0, 6.4))
for i, (F, (_, lab)) in enumerate(zip(grids, parts)):
    ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.Robinson(central_longitude=150))
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
        f"({'abcde'[i]}) {lab}, RMS {rms(parts[i][0]):.0f} m", loc="left", fontsize=9.5
    )
cax = fig.add_axes([0.09, 0.07, 0.52, 0.018])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Geoid (m), one scale for all five maps", fontsize=9)
ax = fig.add_subplot(2, 3, 6)
for (c, lab), col in zip(parts, (BLUE, AQUA, ORANGE, YELLOW, INK)):
    ax.semilogy(
        ls,
        per_degree(c),
        color=col,
        lw=2.0 if col == INK else 1.4,
        label=lab,
    )
ax.set_xlabel(r"Degree $\ell$")
ax.set_ylabel("Amplitude per degree (m)")
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("(f) By degree", loc="left", fontsize=9.5)
fig.subplots_adjust(
    left=0.03, right=0.98, top=0.93, bottom=0.14, wspace=0.12, hspace=0.25
)
fig.savefig(OUT / "contributions.pdf")
fig.savefig(OUT / "contributions.png")
plt.close(fig)
print("contributions, RMS (m): " + ", ".join(f"{lab} {rms(c):.1f}" for c, lab in parts))

print("saved to", OUT)
