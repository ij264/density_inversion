"""Tests for step 1, mantle density -> load on the plate (density_inversion/forward/mantle.py)."""

import os

import numpy as np
import pytest

from density_inversion.forward.mantle import MantleFlow, viscosity_jump

# kernels from a different solver (fem_1d) for the cross-check; outside this repository
FEM_KERNELS = os.environ.get("DENSITY_INVERSION_FEM_KERNELS")

LMAX = 20


@pytest.fixture(scope="module")
def mf():
    return MantleFlow(viscosity=viscosity_jump(30.0, 670e3), lab_depth=105e3, lmax=LMAX)


@pytest.fixture(scope="module")
def ops(mf):
    X = mf.density_space()
    return X, mf.operator(X)


def profile_components(X, l, m_index, profile):
    """Components of the field profile(r/R) Y_lm in X, fitted layer by layer on the layer's radial modes."""
    c = np.zeros(X.dim)
    for i, ball in enumerate(X.layers):
        a, b = X.layer_bounds[i]
        x = np.linspace(a, b, 400)
        Phi = ball.basis[l].evaluate(x)
        coef, *_ = np.linalg.lstsq(Phi, profile(x), rcond=None)
        n = int(ball.basis.nmodes[l])
        start = X.columns(i).start + ball.basis.block(0, l, 0).start + m_index * n
        c[start : start + n] = coef
    return c


def test_adjoint(ops):
    """<A x, y> = <x, A* y> for random x, y: the adjoint pygeoinf builds is consistent with the operator."""
    X, A = ops
    rng = np.random.default_rng(1)
    for _ in range(3):
        x = X.from_components(rng.standard_normal(X.dim))
        y = A.codomain.from_components(rng.standard_normal(A.codomain.dim))
        lhs = A.codomain.inner_product(A(x), y)
        rhs = X.inner_product(x, A.adjoint(y))
        assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0)


def test_spaces_are_sobolev(ops):
    """Both the density space and the load space are Sobolev spaces."""
    from pygeoinf.ball.space import Sobolev as BallSobolev
    from pygeoinf.symmetric_space import sphere

    X, A = ops
    assert isinstance(A.codomain, sphere.Sobolev)
    assert all(isinstance(ball, BallSobolev) for ball in X.layers)


def test_dense_anomaly_pulls_plate_down(mf):
    """A dense anomaly anywhere in the mantle gives a negative load (the plate is pulled down), at every degree."""
    K = mf.kernels["K_q"][:, 1:-1]  # the boundary nodes carry no load
    assert np.all(K <= 1e-12 * np.abs(K).max())


@pytest.mark.parametrize("depth_km", [300, 1000, 2000])
def test_operator_matches_quadrature(mf, ops, depth_km):
    """A(drho) at (2, 0) equals the kernel integral of the same density profile, and is negative."""
    X, A = ops
    R = float(mf.kernels["R"])
    z0 = (R - depth_km * 1e3) / R
    prof = lambda x: 10.0 * np.exp(-0.5 * ((x - z0) * R / 150e3) ** 2)
    c = profile_components(X, 2, 0, prof)
    q = A.codomain.to_components(A(X.from_components(c)))
    k = mf.kernels
    row = list(k["degrees"]).index(2)
    direct = np.sum(k["weights"] * k["K_q"][row] * prof(k["r"] / R))
    assert q[4] < 0  # sphere ordering: l^2 + (m index) -> (2, 0) is component 4
    assert q[4] == pytest.approx(direct, rel=2e-3)


@pytest.mark.skipif(
    not (FEM_KERNELS and os.path.exists(FEM_KERNELS)),
    reason="set DENSITY_INVERSION_FEM_KERNELS to kernels_jump30.npz (from fem_1d) to run",
)
def test_unflexed_deflection_matches_fem_kernels():
    """With its top at the surface, the shell's deflection kernel matches the fem_1d kernels (kernels_jump30.npz,
    a different solver) to within 1% at every degree 1-40. Degree 1 is in the centre-of-mass frame in both
    solvers (no external dipole); its agreement checks flowkernels' replaced row at l = 1."""
    full = MantleFlow(
        viscosity=viscosity_jump(30.0, 670e3), lab_depth=1.0, lmax=40, cache=False
    )
    fem = np.load(FEM_KERNELS)
    k = full.kernels
    for l in range(1, 41):
        a = k["K_h0"][l - 1]
        b = np.interp(k["r"], fem["r"], fem["K_h_b"][list(fem["degrees"]).index(l)])
        misfit = np.sqrt(
            np.sum(k["weights"] * (a - b) ** 2) / np.sum(k["weights"] * b**2)
        )
        assert misfit < 0.01, (l, misfit)
