"""Density spaces and kernel operators in pygeoinf.

Model space: a density anomaly in the mantle, as a ``pygeoinf.ball.Layered`` space split at the viscosity
jump, so density may jump there. In each layer the field is

    drho(r, theta, phi) = sum_lm ( sum_j c_lmj phi_lj(r/R) ) Y_lm(theta, phi)      [kg/m^3]

with Y_lm the real harmonics orthonormal on the unit sphere and phi_lj the layer's radial modes of degree l.
Components run degree by degree (orders 0, 1, ..., l, -1, ..., -l), the radial modes within each harmonic,
layer after layer.

Data space: a ``pygeoinf.symmetric_space.sphere`` space packed in the same order; its components are
``radius`` times the orthonormal coefficients.

Operator, from kernels K_l(r) on quadrature points r_q with weights w_q:

    d_lm = sum_j g_lj c_lmj,      g_lj = sum_q w_q K_l(r_q) phi_lj(r_q/R).

The same row g_l serves every order m, so the operator is a sparse matrix, block-diagonal in degree.
Degrees without kernels map to zero.

``kernels`` is a dict with at least: R, r_cmb, r_jump (m), r and weights (the quadrature), degrees, and
K_<name> (degree, point) for each observable.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from pygeoinf import LinearOperator
from pygeoinf.ball import Layered


def density_space(
    kernels,
    lmax,
    *,
    order=2.0,
    length_scale=0.05,
    radial_modes=(16, 10),
    top_depth=0.0,
    **options,
):
    """A layered Sobolev ball on [r_cmb/R, 1 - top_depth/R], split at the viscosity jump.

    ``top_depth`` (m) is where the flow domain stops. Order 2 lets the field be evaluated at points
    (pygeoinf needs order > 3/2 in a ball). ``radial_modes`` is (lower, upper) modes at every degree;
    ``length_scale`` is in units of R. The forward operator does not depend on the order.
    """
    R = float(kernels["R"])
    top = 1.0 - float(top_depth) / R
    if not float(kernels["r_jump"]) / R < top:
        raise ValueError("top_depth must lie above the viscosity jump")
    bounds = [float(kernels["r_cmb"]) / R, float(kernels["r_jump"]) / R, top]
    return Layered(
        bounds,
        lmax,
        order=order,
        length_scale=length_scale,
        radial_modes=list(radial_modes),
        labels=("lower", "upper"),
        **options,
    )


def _points_by_layer(space, r, R):
    """For each layer: indices of the quadrature points in it, and their radii r/R inside it.

    A point on an interface belongs to the element it came from; the kernels list the lower copy first,
    and ``searchsorted(side='left')`` sends it down and the next one up."""
    x = np.asarray(r, float) / R
    which = np.searchsorted(np.asarray(space.interfaces, float), x, side="left")
    out = []
    for i, (a, b) in enumerate(space.layer_bounds):
        slack = 1e-9 * (b - a)
        idx = np.flatnonzero((which == i) & (x >= a - slack) & (x <= b + slack))
        out.append((idx, np.clip(x[idx], a, b)))
    return out


def _column_block(space, layer, l):
    """(first column, number of radial modes) of degree l of a layer, in the space's components."""
    ball = space.layers[layer]
    return space.columns(layer).start + ball.basis.block(0, l, 0).start, int(
        ball.basis.nmodes[l]
    )


def density_rows(space, kernels, name):
    """g[l] = list over layers of the rows g_lj, from kernels K_<name>."""
    R = float(kernels["R"])
    r, w, K = kernels["r"], kernels["weights"], kernels[f"K_{name}"]
    parts = _points_by_layer(space, r, R)
    rows = {}
    for row, l in enumerate(np.asarray(kernels["degrees"], int)):
        if l > space.layers[0].lmax:
            continue
        rows[int(l)] = [
            (w[idx] * K[row, idx]) @ space.layers[i].basis[l].evaluate(x)
            for i, (idx, x) in enumerate(parts)
        ]
    return rows


def density_operator(space, kernels, name, data_space, rows=None):
    """The observable K_<name> as a LinearOperator from the density space to a sphere space."""
    rows = density_rows(space, kernels, name) if rows is None else rows
    I, J, V = [], [], []
    for l, per_layer in rows.items():
        if l > data_space.lmax:
            continue
        orders = np.arange(2 * l + 1)
        for layer, g in enumerate(per_layer):
            start, n = _column_block(space, layer, l)
            I.append(np.repeat(l * l + orders, n))
            J.append((start + orders[:, None] * n + np.arange(n)[None, :]).ravel())
            V.append(np.tile(data_space.radius * g, 2 * l + 1))
    M = sp.csr_matrix(
        (np.concatenate(V), (np.concatenate(I), np.concatenate(J))),
        shape=(data_space.dim, space.dim),
    )
    return LinearOperator.from_matrix(space, data_space, M, form="components")
