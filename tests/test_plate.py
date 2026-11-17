"""Tests for step 2, the elastic plate (density_inversion/forward/plate.py)."""

import numpy as np
import pytest

from density_inversion.forward import examples as ex
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.plate import ElasticPlate, OceanLoadedPlate, rigidity

LMAX = 20


@pytest.fixture(scope="module")
def mf():
    return MantleFlow(lmax=LMAX)


@pytest.fixture(scope="module")
def land_sea(mf):
    return OceanLoadedPlate.from_mantle(mf, Te=40e3)


def apply(op, comp):
    return op.codomain.to_components(op(op.domain.from_components(comp)))


def adjoint_error(op, seed=0):
    rng = np.random.default_rng(seed)
    x = op.domain.from_components(rng.standard_normal(op.domain.dim))
    y = op.codomain.from_components(rng.standard_normal(op.codomain.dim))
    lhs = op.codomain.inner_product(op(x), y)
    rhs = op.domain.inner_product(x, op.adjoint(y))
    return abs(lhs - rhs) / max(abs(lhs), 1e-300)


# ------------------------------------------------------------------ constant water load


def test_filter_is_the_analytic_one(mf):
    """F_l = 1 / (D [l(l+1)/R^2]^2 + drho g) degree by degree, with D from E, nu and Te."""
    p = ElasticPlate.from_mantle(mf, Te=40e3)
    l = np.arange(LMAX + 1)
    D = 90e9 * 40e3**3 / (12 * (1 - 0.25**2))
    assert rigidity(40e3) == pytest.approx(D)
    F = 1.0 / (D * (l * (l + 1) / p.R**2) ** 2 + p.drho * p.g)
    assert np.allclose(p.filter(), F, rtol=1e-14)
    Q = mf.load_space()
    deg, _ = ex.sphere_ordering(LMAX)
    q = np.random.default_rng(1).standard_normal(Q.dim)
    assert np.allclose(apply(p.operator(Q), q), F[deg] * q, rtol=1e-14)


def test_no_plate_gives_step_1_unflexed_deflection(mf):
    """With Te = 0 the plate is local isostasy, the unflexed deflection of step 1."""
    p = ElasticPlate.from_mantle(mf, Te=0.0)
    q = np.random.default_rng(2).standard_normal(mf.load_space().dim) * 1e6
    assert np.allclose(
        apply(p.operator(mf.load_space()), q), mf.unflexed_deflection(q), rtol=1e-14
    )


def test_thicker_plate_damps_more_and_more_at_high_degree(mf):
    F = [ElasticPlate.from_mantle(mf, Te=Te).filter() for Te in (0.0, 20e3, 40e3, 60e3)]
    for thin, thick in zip(F, F[1:]):
        assert np.all(thick[1:] < thin[1:])
    ratio = F[2] / F[0]
    assert np.all(np.diff(ratio[1:]) < 0)


def test_constant_plate_adjoint_and_sign(mf):
    op = ElasticPlate.from_mantle(mf, Te=40e3).operator(mf.load_space())
    assert adjoint_error(op) < 1e-10
    q = np.zeros(op.domain.dim)
    q[0] = 1e6  # a uniform upward load
    assert np.all(apply(op, q)[:1] > 0)


# ------------------------------------------------------------------ land and sea


def test_land_matrix_is_symmetric_and_positive(land_sea):
    K = land_sea.system()
    assert np.allclose(K, K.T, rtol=0, atol=1e-12 * np.abs(K).max())
    assert np.linalg.eigvalsh(K).min() > 0


def test_water_everywhere_is_the_constant_plate(mf):
    """With C = 1 everywhere the land/sea plate is the constant-water plate."""
    Q = mf.load_space()
    ocean = OceanLoadedPlate.from_mantle(
        mf, Te=40e3, ocean=lambda la, lo: np.ones_like(la)
    )
    q = np.random.default_rng(3).standard_normal(Q.dim)
    a = apply(ocean.operator(Q), q)
    b = apply(ElasticPlate.from_mantle(mf, Te=40e3).operator(Q), q)
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10 * np.abs(b).max())


def test_air_everywhere_is_the_plate_with_mantle_density(mf):
    """With C = 0 everywhere the restoring contrast is rho_LAB everywhere."""
    Q = mf.load_space()
    air = OceanLoadedPlate.from_mantle(
        mf, Te=40e3, ocean=lambda la, lo: np.zeros_like(la)
    )
    k = mf.kernels
    p = ElasticPlate(40e3, float(k["rho_lab"]), float(k["g_lab"]), float(k["R"]), LMAX)
    q = np.random.default_rng(4).standard_normal(Q.dim)
    a, b = apply(air.operator(Q), q), apply(p.operator(Q), q)
    assert np.allclose(a, b, rtol=1e-10, atol=1e-10 * np.abs(b).max())


def test_land_sea_adjoint(mf, land_sea):
    assert adjoint_error(land_sea.operator(mf.load_space()), seed=5) < 1e-10


def test_land_rises_less_than_sea_under_the_same_load():
    """No plate and a uniform upward load: away from coasts the deflection is q / (drho g), with
    drho = rho_LAB on land and rho_LAB - rho_w at sea, so land rises by about (rho - rho_w) / rho as much.
    """
    mf = MantleFlow(lmax=50)
    p = OceanLoadedPlate.from_mantle(mf, Te=0.0)
    Q = mf.load_space()
    q = np.zeros(Q.dim)
    q[0] = 1e6 * np.sqrt(4 * np.pi)  # 1 MPa everywhere
    lat, lon, w = ex.to_grid(apply(p.operator(Q), q), 50)
    k = mf.kernels
    rho, rw, g = float(k["rho_lab"]), float(k["rho_water"]), float(k["g_lab"])
    at = lambda la, lo: w[np.argmin(abs(lat - la)), np.argmin(abs(lon - lo))]
    sea = [
        at(0, 200),
        at(-40, 250),
        at(-30, 80),
    ]  # Pacific, south Pacific, Indian Ocean
    land = [at(10, 20), at(60, 100), at(-25, 135)]  # central Africa, Siberia, Australia
    for s in sea:
        assert s == pytest.approx(1e6 / ((rho - rw) * g), rel=0.05)
    for la in land:
        assert la == pytest.approx(1e6 / (rho * g), rel=0.1)
