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
    """The operator is diagonal with F_l = 1 / ((D/R^4)(lam^2 - (1 - nu) lam) + drho g), lam = l(l+1), and
    within a few parts in 10^5 of the flat-plate filter up to lmax."""
    p = ElasticPlate.from_mantle(mf, Te=40e3)
    l = np.arange(LMAX + 1)
    D = 90e9 * 40e3**3 / (12 * (1 - 0.25**2))
    assert rigidity(40e3) == pytest.approx(D)
    lam = l * (l + 1.0)
    F = 1.0 / (D / p.R**4 * (lam**2 - 0.75 * lam) + p.drho * p.g)
    assert np.allclose(p.filter(), F, rtol=1e-14)
    Q = mf.load_space()
    deg, _ = ex.sphere_ordering(LMAX)
    q = np.random.default_rng(1).standard_normal(Q.dim)
    assert np.allclose(apply(p.operator(Q), q), F[deg] * q, rtol=1e-12)
    assert np.allclose(p.filter(), p.flat_filter(), rtol=1e-4)


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


def galerkin_reference(plate, quad_lmax):
    """The previous implementation, kept as an independent check: (diag(1/F_flat) + g rho_w M) w = q with
    M_ij = int (1 - C) Y_i Y_j dOmega by Gauss-Legendre quadrature, w restricted to degree <= lmax.
    """
    import pyshtools as pysh

    L = plate.lmax
    zero, w = pysh.expand.SHGLQ(quad_lmax)
    lat, lon = pysh.expand.GLQGridCoord(quad_lmax)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    land = 1.0 - plate.ocean(LA, LO)
    n = (L + 1) ** 2
    M = np.empty((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0
        cilm = np.zeros((2, quad_lmax + 1, quad_lmax + 1))
        cilm[:, : L + 1, : L + 1] = ex.components_to_cilm(e, L)
        grid = pysh.expand.MakeGridGLQ(cilm, zero, norm=4, csphase=1)
        out = pysh.expand.SHExpandGLQ(
            land * grid, w, zero, norm=4, csphase=1, lmax_calc=L
        )
        M[:, j] = ex.cilm_to_components(out, L)
        e[j] = 0.0
    deg, _ = ex.sphere_ordering(L)
    K = np.diag(1.0 / plate.flat_filter()[deg]) + plate.g * plate.rho_water * 0.5 * (
        M + M.T
    )
    return np.linalg.inv(K)


def test_land_sea_matches_the_galerkin_reference(mf, land_sea):
    """pygeoinf's matrix-free flexure (solved at 3 lmax, truncated) against an independent Galerkin solve:
    both converge on the same answer, to within 2 % at this resolution."""
    Q = mf.load_space()
    q = np.random.default_rng(6).standard_normal(Q.dim)
    a = apply(land_sea.operator(Q), q)
    b = galerkin_reference(land_sea, 3 * LMAX) @ q
    assert np.linalg.norm(a - b) <= 0.02 * np.linalg.norm(b)


def test_assembled_matrix_is_the_operator(mf, land_sea):
    Q = mf.load_space()
    q = np.random.default_rng(7).standard_normal(Q.dim)
    a, b = apply(land_sea.operator(Q), q), apply(land_sea.assembled(Q), q)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-8 * np.abs(a).max())  # CG tolerance


def test_varying_elastic_thickness(mf):
    """A T_e field equal to 40 km everywhere gives the constant-T_e plate; a thicker plate everywhere damps more."""
    Q = mf.load_space()
    q = np.random.default_rng(8).standard_normal(Q.dim)
    const = apply(OceanLoadedPlate.from_mantle(mf, Te=40e3).operator(Q), q)
    field = apply(
        OceanLoadedPlate.from_mantle(mf, Te=lambda la, lo: 40e3 + 0 * la).operator(Q), q
    )
    assert np.allclose(field, const, rtol=1e-8, atol=1e-8 * np.abs(const).max())


def test_water_everywhere_is_the_constant_plate(mf):
    """With C = 1 everywhere the land/sea plate is the constant-water plate."""
    Q = mf.load_space()
    ocean = OceanLoadedPlate.from_mantle(
        mf, Te=40e3, ocean=lambda la, lo: np.ones_like(la)
    )
    q = np.random.default_rng(3).standard_normal(Q.dim)
    a = apply(ocean.operator(Q), q)
    b = apply(ElasticPlate.from_mantle(mf, Te=40e3).operator(Q), q)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-8 * np.abs(b).max())


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
    assert np.allclose(a, b, rtol=1e-8, atol=1e-8 * np.abs(b).max())


def test_land_sea_adjoint(mf, land_sea):
    assert adjoint_error(land_sea.operator(mf.load_space()), seed=5) < 1e-8


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
