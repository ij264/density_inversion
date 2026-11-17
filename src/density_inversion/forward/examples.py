"""Example mantle density fields for testing the forward problem.

A density field is described by its coefficient profiles: a function of x = r/R returning, for each x,
the orthonormal real spherical-harmonic coefficients (kg/m^3) in the sphere ordering of pygeoinf
(index l^2 + i, orders 0, 1, ..., l, -1, ..., -l; pyshtools 'ortho' normalisation with csphase=1, i.e. no Condon-Shortley phase).
`project` fits those profiles onto the radial modes of a layered density space.

    dense_blob(...)     a Gaussian dense anomaly at one point in the mantle
    power_law(...)      a random field whose amplitude falls as 1/k (k the 3-D wavenumber)

Usage
-----
    from density_inversion.forward.mantle import MantleFlow
    from density_inversion.forward import examples as ex
    mf = MantleFlow(lmax=50); X = mf.density_space()
    c = ex.project(X, ex.dense_blob(mf.kernels["R"]))     # components of drho_m in X
    rho_300 = ex.evaluate_at(X, c, (R - 300e3) / R)       # sphere coefficients at 300 km depth
"""

from __future__ import annotations

import numpy as np
import pyshtools as pysh

# ------------------------------------------------------------------------- orderings and grids


def sphere_ordering(lmax):
    """(degrees, orders) of the sphere components up to lmax: orders 0, 1, ..., l, -1, ..., -l in each degree."""
    deg, order = [], []
    for l in range(lmax + 1):
        deg += [l] * (2 * l + 1)
        order += list(range(l + 1)) + [-m for m in range(1, l + 1)]
    return np.array(deg), np.array(order)


def cilm_to_components(cilm, lmax):
    """pyshtools array (2, L+1, L+1) -> sphere components up to lmax."""
    deg, order = sphere_ordering(lmax)
    return cilm[np.where(order >= 0, 0, 1), deg, np.abs(order)]


def components_to_cilm(comp, lmax):
    deg, order = sphere_ordering(lmax)
    cilm = np.zeros((2, lmax + 1, lmax + 1))
    cilm[np.where(order >= 0, 0, 1), deg, np.abs(order)] = comp
    return cilm


def to_grid(comp, lmax, grid_lmax=None):
    """Sphere components -> (lat, lon, values) on a Driscoll-Healy grid (sampling 2)."""
    grid_lmax = max(lmax, grid_lmax or 2 * lmax)
    cilm = np.zeros((2, grid_lmax + 1, grid_lmax + 1))
    cilm[:, : lmax + 1, : lmax + 1] = components_to_cilm(comp, lmax)
    values = pysh.expand.MakeGridDH(cilm, norm=4, sampling=2, csphase=1)
    lat, lon = dh_lat_lon(values.shape[0])
    return lat, lon, values


def dh_lat_lon(n):
    """Latitudes and longitudes (degrees) of an n x 2n Driscoll-Healy grid."""
    return 90.0 - 180.0 * np.arange(n) / n, 360.0 * np.arange(2 * n) / (2 * n)


# ---------------------------------------------------------------- radial fit and evaluation


def project(X, profiles, *, points_per_layer=60):
    """Components in the layered density space X of the field whose coefficient profiles are
    `profiles(x) -> array (len(x), (lmax+1)^2)`, fitted by least squares layer by layer.
    """
    lmax = X.layers[0].lmax
    c = np.zeros(X.dim)
    for i, ball in enumerate(X.layers):
        a, b = X.layer_bounds[i]
        x = np.linspace(a, b, points_per_layer)
        C = profiles(x)
        for l in range(lmax + 1):
            n = int(ball.basis.nmodes[l])
            if n == 0:
                continue
            Phi = ball.basis[l].evaluate(x)  # (points, n)
            coef, *_ = np.linalg.lstsq(
                Phi, C[:, l * l : (l + 1) ** 2], rcond=None
            )
            start = X.columns(i).start + ball.basis.block(0, l, 0).start
            c[start : start + (2 * l + 1) * n] = (
                coef.T.ravel()
            )  # (order, mode), mode fastest
    return c


def evaluate_at(X, c, x):
    """Sphere coefficients (kg/m^3) of the field with components c at one radius x = r/R."""
    lmax = X.layers[0].lmax
    i = int(np.searchsorted(np.asarray(X.interfaces), x))
    ball = X.layers[i]
    out = np.zeros((lmax + 1) ** 2)
    for l in range(lmax + 1):
        n = int(ball.basis.nmodes[l])
        if n == 0:
            continue
        start = X.columns(i).start + ball.basis.block(0, l, 0).start
        phi = ball.basis[l].evaluate(np.array([x]))[0]
        out[l * l : (l + 1) ** 2] = (
            c[start : start + (2 * l + 1) * n].reshape(2 * l + 1, n) @ phi
        )
    return out


# ------------------------------------------------------------------------------- the fields


def dense_blob(
    R, *, depth=300e3, lat=20.0, lon=150.0, width=500e3, amplitude=30.0, lmax=50
):
    """A Gaussian density anomaly, amplitude * exp(-d^2 / 2 width^2), d the straight-line distance to a
    point `depth` below (lat, lon). Positive amplitude = dense. Returns coefficient profiles to lmax.
    """
    R = float(R)
    grid_lmax = 4 * lmax
    glat, glon = dh_lat_lon(2 * (grid_lmax + 1))
    LAT, LON = np.meshgrid(np.radians(glat), np.radians(glon), indexing="ij")
    unit = np.stack(
        [np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)]
    )
    la, lo = np.radians(lat), np.radians(lon)
    centre = (R - depth) * np.array(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    )

    def profiles(x):
        out = np.empty((len(x), (lmax + 1) ** 2))
        for j, xj in enumerate(np.atleast_1d(x)):
            d2 = np.sum((xj * R * unit - centre[:, None, None]) ** 2, axis=0)
            grid = amplitude * np.exp(-0.5 * d2 / width**2)
            cilm = pysh.expand.SHExpandDH(
                grid, norm=4, sampling=2, csphase=1, lmax_calc=lmax
            )
            out[j] = cilm_to_components(cilm, lmax)
        return out

    profiles.label = (
        f"{amplitude:+g} kg/m^3 Gaussian, width {width / 1e3:g} km, {depth / 1e3:g} km below "
        f"({lat:g}N, {lon:g}E)"
    )
    return profiles


def power_law(
    R, x_bottom, x_top, *, rms=10.0, slope=1.0, radial_modes=12, lmax=50, seed=0
):
    """A random density field with amplitude proportional to 1/k^slope, k the 3-D wavenumber.

    Each coefficient profile is a sum of cosines across the mantle [x_bottom, x_top],
        c_lm(x) = sum_n a_lmn cos(n pi s),   s = (x - x_bottom) / (x_top - x_bottom),
    with a_lmn ~ N(0, 1/k^(2 slope)), k^2 = (l / r_mid)^2 + (n pi / H)^2, r_mid the mid-mantle radius and
    H the mantle's thickness. Degree 0 is left out. The default 12 cosines are what the default
    density space can hold (its fit error is below 0.1%; at 24 cosines it rises to about 15% in the
    lower mantle). Scaled so the volume RMS is `rms` (kg/m^3). The same seed gives the same field.
    """
    R = float(R)
    H = (x_top - x_bottom) * R
    r_mid = 0.5 * (x_top + x_bottom) * R
    deg, _ = sphere_ordering(lmax)
    n = np.arange(radial_modes + 1)
    k = np.sqrt((deg[:, None] / r_mid) ** 2 + (n[None, :] * np.pi / H) ** 2)
    a = np.random.default_rng(seed).standard_normal(k.shape)
    with np.errstate(divide="ignore"):
        a = np.where(deg[:, None] > 0, a / k**slope, 0.0)

    def raw(x):
        s = (np.atleast_1d(x) - x_bottom) / (x_top - x_bottom)
        return (
            np.cos(np.pi * s[:, None] * n[None, :]) @ a.T
        )  # (points, components)

    # volume RMS: (1 / V) int sum_lm c_lm(r)^2 r^2 dr, over the shell
    xs = np.linspace(x_bottom, x_top, 400)
    w = np.gradient(xs) * xs**2
    ms = np.sum(w * np.sum(raw(xs) ** 2, axis=1)) / (4 * np.pi * np.sum(w))
    scale = rms / np.sqrt(ms)

    def profiles(x):
        return scale * raw(x)

    k_text = "$k^{-1}$" if slope == 1 else f"$k^{-slope:g}"
    profiles.label = f"{k_text} random field, RMS {rms:g} kg/m^3, seed {seed}"
    return profiles
