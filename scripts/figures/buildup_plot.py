"""Figures of the forward problem's build-up, from outputs/figures_buildup/buildup.npz (buildup_compute.py).
Written by Claude, 24 Nov 2026. Same-quantity panels share a colour scale, except where a caption says otherwise.

Usage: python buildup_plot.py buildup.npz geoid_for_inversion.npz cartopy_dir out_dir
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import pyshtools as pysh

src, geoid_product, cdir, out = sys.argv[1:5]
cartopy.config["pre_existing_data_dir"] = cdir
OUT = Path(out); OUT.mkdir(parents=True, exist_ok=True)
d = np.load(src)
L = int(d["lmax"])
PROJ = ccrs.Robinson(central_longitude=0)


def ordering(lmax):
    deg, order = [], []
    for l in range(lmax + 1):
        deg += [l] * (2 * l + 1)
        order += list(range(l + 1)) + [-m for m in range(1, l + 1)]
    return np.array(deg), np.array(order)


DEG, ORD = ordering(L)


def to_grid(comp, lmax=L):
    cilm = np.zeros((2, L + 1, L + 1))
    cilm[np.where(ORD >= 0, 0, 1), DEG, np.abs(ORD)] = comp
    g = pysh.SHCoeffs.from_array(cilm, normalization="ortho", csphase=1).expand(grid="DH2", lmax=max(lmax, 90))
    return g.lats(), g.lons(), g.to_array()


def spectrum(comp, lmax=L):
    """Root-mean-square over the sphere contributed by each degree (orthonormal components on the unit sphere)."""
    return np.array([np.sqrt(np.sum(comp[DEG == l] ** 2) / (4 * np.pi)) for l in range(lmax + 1)])


def rms(comp, lmin=1):
    return float(np.sqrt(np.sum(comp[DEG >= lmin] ** 2) / (4 * np.pi)))


def panel(fig, pos, comp, title, vmax, cmap="RdBu_r", scale=1.0):
    ax = fig.add_subplot(*pos, projection=PROJ)
    la, lo, v = to_grid(comp)
    im = ax.pcolormesh(lo, la, v * scale, transform=ccrs.PlateCarree(), cmap=cmap, vmin=-vmax, vmax=vmax,
                       shading="auto", rasterized=True)
    ax.coastlines(lw=0.35, color="0.25")
    ax.set_global()
    ax.set_title(title, loc="left", fontsize=8.5)
    return ax, im


def bar(fig, im, axes, label, cax=None):
    if cax is not None:
        cb = fig.colorbar(im, cax=fig.add_axes(cax), orientation="horizontal", extend="both")
    else:
        cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.045, pad=0.04, extend="both", shrink=0.8)
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


km = 1e-3
h1, N1, h2, N2, h3m, N3m = (d[k] for k in ("h1", "N1", "h2", "N2", "h3m", "N3m"))
h_ref, N_ref, N_lift, N_struct, bed = (d[k] for k in ("h_ref", "N_ref", "N_lift", "N_struct", "bed"))
h_meanM, h_L, N_L = d["h_meanM"], d["h_L"], d["N_L"]

# ---- 1. core model
fig = plt.figure(figsize=(7.4, 2.9))
a1, im1 = panel(fig, (1, 2, 1), h1, f"(a) Topography, $h = w - N$ (RMS {rms(h1) * km:.2f} km)", 1.0, scale=km)
bar(fig, im1, [a1], "km")
a2, im2 = panel(fig, (1, 2, 2), N1, f"(b) Geoid, $N$ (RMS {rms(N1, 2):.0f} m)", 40)
bar(fig, im2, [a2], "m")
save(fig, "buildup_1_core")

# ---- 2. plate on land and sea (same mantle load)
fig = plt.figure(figsize=(7.4, 5.6))
axs, ims = [], []
for i, (c, t) in enumerate(((h2, f"(a) Topography with the plate (RMS {rms(h2) * km:.2f} km)"),
                            (h2 - h1, f"(c) Change from the core model (RMS {rms(h2 - h1) * km:.2f} km)"))):
    ax, im = panel(fig, (2, 2, 1 + 2 * i), c, t, 1.0, scale=km); axs.append(ax); ims.append(im)
bar(fig, ims[0], axs, "km")
axs, ims = [], []
for i, (c, t) in enumerate(((N2, f"(b) Geoid with the plate (RMS {rms(N2, 2):.0f} m)"),
                            (N2 - N1, f"(d) Change from the core model (RMS {rms(N2 - N1, 2):.1f} m)"))):
    ax, im = panel(fig, (2, 2, 2 + 2 * i), c, t, 40); axs.append(ax); ims.append(im)
bar(fig, ims[0], axs, "m")
save(fig, "buildup_2_plate")

# ---- 3. reference lithosphere: topography
fig = plt.figure(figsize=(7.4, 5.6))
axs, ims = [], []
for i, (c, t) in enumerate(((bed, f"(a) Observed bedrock elevation, $\\ell \\leq {L}$ (RMS {rms(bed) * km:.1f} km)"),
                            (h_ref, f"(b) Reference lithosphere (RMS {rms(h_ref) * km:.1f} km)"),
                            (h_ref - bed, f"(c) Reference minus observed (RMS {rms(h_ref - bed) * km:.2f} km)"))):
    ax, im = panel(fig, (2, 2, 1 + i), c, t, 5.0, scale=km); axs.append(ax); ims.append(im)
bar(fig, ims[0], axs, "km (all three maps)", cax=[0.14, 0.04, 0.30, 0.018])
ax = fig.add_subplot(2, 2, 4)
for c, lab, st in ((bed, "observed", dict(color="black", lw=1.6)),
                   (h_ref, "reference lithosphere", dict(color="#2a6f97", lw=1.4)),
                   (h_ref - bed, "reference minus observed", dict(color="#c05a2b", lw=1.2)),
                   (h2, "mantle (stage 2)", dict(color="0.5", lw=1.2, ls="--"))):
    s = spectrum(c)
    ax.semilogy(np.arange(1, L + 1), s[1:] * km, label=lab, **st)
ax.set_xlabel("Degree $\\ell$", fontsize=8.5); ax.set_ylabel("RMS per degree (km)", fontsize=8.5)
ax.tick_params(labelsize=8); ax.legend(frameon=False, fontsize=7.5)
ax.set_title("(d) Topography by degree", loc="left", fontsize=8.5)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
save(fig, "buildup_3a_reference_topography")

# ---- 3. reference lithosphere: geoid (the total has its own scale: it is a small remainder)
fig = plt.figure(figsize=(7.4, 6.6))
fig.subplots_adjust(hspace=0.62)
axs, ims = [], []
for i, (c, t) in enumerate(((N_struct, f"(a) Structure, undeflected (RMS {rms(N_struct, 2):.0f} m)"),
                            (N_lift, f"(b) Lift of the columns (RMS {rms(N_lift, 2):.0f} m)"))):
    ax, im = panel(fig, (2, 2, 1 + i), c, t, 600); axs.append(ax); ims.append(im)
bar(fig, ims[0], axs, "m, (a) and (b)", cax=[0.30, 0.545, 0.40, 0.014])
ax, im = panel(fig, (2, 2, 3), N_ref, f"(c) Total, (a) + (b) (RMS {rms(N_ref, 2):.1f} m); own scale", 40)
bar(fig, im, [ax], "m, (c) only", cax=[0.14, 0.04, 0.30, 0.016])
ax = fig.add_subplot(2, 2, 4)
g = np.load(geoid_product)
gl = g["l"]
s_obs = np.array([np.sqrt(np.sum(g["N_obs"][gl == l] ** 2) / (4 * np.pi)) for l in range(2, int(gl.max()) + 1)])
ax.semilogy(np.arange(2, int(gl.max()) + 1), s_obs, color="black", lw=1.6, label="observed (non-hydrostatic)")
for c, lab, st in ((N_struct, "structure", dict(color="#2a6f97", lw=1.2, ls=":")),
                   (N_lift, "lift", dict(color="#2a6f97", lw=1.2, ls="--")),
                   (N_ref, "reference total", dict(color="#2a6f97", lw=1.6)),
                   (N3m, "mantle (stage 3)", dict(color="#c05a2b", lw=1.4))):
    s = spectrum(c)
    ax.semilogy(np.arange(2, L + 1), s[2:], label=lab, **st)
ax.set_xlabel("Degree $\\ell$", fontsize=8.5); ax.set_ylabel("RMS per degree (m)", fontsize=8.5)
ax.tick_params(labelsize=8); ax.legend(frameon=False, fontsize=7.5)
ax.set_title("(d) Geoid by degree", loc="left", fontsize=8.5)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
save(fig, "buildup_3b_reference_geoid")

# ---- 4. sea level
fig = plt.figure(figsize=(4.4, 2.9))
dK = h_meanM - h_ref
dKm = dK.copy(); dKm[0] = 0.0
ax, im = panel(fig, (1, 1, 1), dKm, f"Topography with $K = \\langle M \\rangle$ minus with $K_0$, its mean\n({dK[0] / np.sqrt(4 * np.pi) * km:.2f} km) removed (RMS {rms(dK) * km:.2f} km)", 0.8, scale=km)
bar(fig, im, [ax], "km")
save(fig, "buildup_4_sea_level")

# ---- 5. lithospheric perturbations (example)
fig = plt.figure(figsize=(7.4, 2.9))
a1, im1 = panel(fig, (1, 2, 1), h_L, f"(a) Topography (RMS {rms(h_L) * km:.2f} km)", 2.0, scale=km)
bar(fig, im1, [a1], "km")
a2, im2 = panel(fig, (1, 2, 2), N_L, f"(b) Geoid (RMS {rms(N_L, 2):.1f} m)", 20)
bar(fig, im2, [a2], "m")
save(fig, "buildup_5_perturbations")

# ---- 6. summary: every contribution by degree
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 4.2))
for c, lab, st in ((bed, "observed bedrock", dict(color="black", lw=1.8)),
                   (h_ref, "reference lithosphere", dict(color="#2a6f97", lw=1.4)),
                   (h1, "mantle, core model", dict(color="#c05a2b", lw=1.2, ls="--")),
                   (h3m, "mantle, plate and columns", dict(color="#c05a2b", lw=1.4)),
                   (h_L, "perturbation example", dict(color="0.45", lw=1.2, ls="-.")),
                   (h_ref - bed, "reference minus observed", dict(color="#6a4c93", lw=1.2, ls=":"))):
    top = 31 if lab == "mantle, core model" else L
    ax1.semilogy(np.arange(1, top + 1), spectrum(c)[1:top + 1] * km, label=lab, **st)
ax1.set_xlabel("Degree $\\ell$"); ax1.set_ylabel("RMS per degree (km)"); ax1.set_title("(a) Topography", loc="left", fontsize=9)
ax2.semilogy(np.arange(2, int(gl.max()) + 1), s_obs, color="black", lw=1.8, label="observed (non-hydrostatic)")
for c, lab, st in ((N_struct, "reference: structure", dict(color="#2a6f97", lw=1.1, ls=":")),
                   (N_lift, "reference: lift", dict(color="#2a6f97", lw=1.1, ls="--")),
                   (N_ref, "reference: total", dict(color="#2a6f97", lw=1.5)),
                   (N1, "mantle, core model", dict(color="#c05a2b", lw=1.2, ls="--")),
                   (N3m, "mantle, plate and columns", dict(color="#c05a2b", lw=1.4)),
                   (N_L, "perturbation example", dict(color="0.45", lw=1.2, ls="-."))):
    top = 31 if lab == "mantle, core model" else L
    ax2.semilogy(np.arange(2, top + 1), spectrum(c)[2:top + 1], label=lab, **st)
ax2.set_xlabel("Degree $\\ell$"); ax2.set_ylabel("RMS per degree (m)"); ax2.set_title("(b) Geoid", loc="left", fontsize=9)
for ax in (ax1, ax2):
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
save(fig, "buildup_6_summary_spectra")

# ---- numbers for the notes
rows = [("core: h", rms(h1)), ("core: N", rms(N1, 2)), ("plate: h", rms(h2)), ("plate: N", rms(N2, 2)),
        ("plate - core: h", rms(h2 - h1)), ("plate - core: N", rms(N2 - N1, 2)),
        ("columns: mantle h", rms(h3m)), ("columns: mantle N", rms(N3m, 2)),
        ("reference h", rms(h_ref)), ("bed", rms(bed)), ("reference - bed", rms(h_ref - bed)),
        ("N structure", rms(N_struct, 2)), ("N lift", rms(N_lift, 2)), ("N reference", rms(N_ref, 2)),
        ("perturbation h", rms(h_L)), ("perturbation N", rms(N_L, 2)), ("K step mean", dK[0] / np.sqrt(4 * np.pi)),
        ("K step rms", rms(dK))]
(OUT / "buildup_numbers.txt").write_text("\n".join(f"{k:28s} {v:12.2f}" for k, v in rows) + f"\nK0 - <M> = {float(d['K0']):.3e} kg/m^2\n")
print((OUT / "buildup_numbers.txt").read_text())

# ---- 3c. the asthenosphere the columns are measured against: constant rho_LAB against rho_a(z)
if len(sys.argv) > 6:
    dc = np.load(sys.argv[5]); prof = np.loadtxt(sys.argv[6])
    fig = plt.figure(figsize=(7.4, 9.0))
    fig.subplots_adjust(hspace=0.75, wspace=0.3)
    axs, ims = [], []
    for i, (dd, lab) in enumerate(((dc, "constant $\\rho_{\\mathrm{LAB}}$"), (d, "$\\rho_a(z)$, WINTERC-G 1D"))):
        diff = dd["h_ref"] - dd["bed"]
        ax, im = panel(fig, (3, 2, 1 + i), diff, f"({'ab'[i]}) Reference minus observed, {lab}\n(RMS {rms(diff) * km:.2f} km)", 3.0, scale=km)
        axs.append(ax); ims.append(im)
    bar(fig, ims[0], axs, "km, (a) and (b)")
    axs, ims = [], []
    for i, (dd, lab) in enumerate(((dc, "constant $\\rho_{\\mathrm{LAB}}$"), (d, "$\\rho_a(z)$"))):
        ax, im = panel(fig, (3, 2, 3 + i), dd["N_ref"], f"({'cd'[i]}) Geoid of the reference, {lab}\n(RMS {rms(dd['N_ref'], 2):.1f} m)", 60)
        axs.append(ax); ims.append(im)
    bar(fig, ims[0], axs, "m, (c) and (d)")
    ax = fig.add_subplot(3, 2, 5)
    zz, rr = -prof[:, 0], prof[:, 2]
    m = (zz >= 60) & (zz <= 340)
    ax.plot(rr[m], zz[m], color="black", lw=1.5, label="$\\rho_a(z)$, WINTERC-G 1D")
    ax.axvline(3370.36, color="#c05a2b", lw=1.3, ls="--", label="constant $\\rho_{\\mathrm{LAB}}$ (PREM, 105 km)")
    ax.set_ylim(340, 60); ax.set_xlabel("Density (kg m$^{-3}$)", fontsize=8.5); ax.set_ylabel("Depth (km)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("(e) The asthenosphere displaced by a column", loc="left", fontsize=8.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax = fig.add_subplot(3, 2, 6)
    lab_c = d["lab"] / 1e3  # km
    la, lo, labg = to_grid(lab_c); _, _, dg_c = to_grid(dc["h_ref"] - dc["bed"]); _, _, dg_1 = to_grid(d["h_ref"] - d["bed"])
    w = np.cos(np.radians(la))[:, None] * np.ones_like(labg)
    bins = np.arange(50, 320, 20)
    for dg, lab, st in ((dg_c, "constant", dict(color="#c05a2b", ls="--")), (dg_1, "$\\rho_a(z)$", dict(color="black"))):
        idx = np.digitize(labg, bins)
        mean = [np.sum((w * dg)[idx == j]) / np.sum(w[idx == j]) if np.any(idx == j) else np.nan for j in range(1, len(bins))]
        ax.plot(0.5 * (bins[1:] + bins[:-1]), np.array(mean) * km, lw=1.5, label=lab, **st)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Depth of the LAB (km)", fontsize=8.5); ax.set_ylabel("Mean reference minus observed (km)", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(f) Misfit by depth of the LAB", loc="left", fontsize=8.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "buildup_3c_asthenosphere")
