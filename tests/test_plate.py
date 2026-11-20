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


def test_land_sea_recovers_a_manufactured_solution(mf):
    """Choose w, build its load independently (bending spectrally, buoyancy as a product on a fine grid) and
    solve: the plate must give w back. The restoring density varies smoothly (degree 1), so the load is
    band-limited and the check is exact up to the solver tolerance. Replaces a comparison with a Galerkin
    solve, which had a different discretisation and needed a 2 % tolerance (audit P14).
    """
    import pyshtools as pysh

    L, L0 = LMAX, 4
    smooth = lambda la, lo: (  # 0.5 + 0.3 x + 0.1 z: degree 1 on the sphere
        0.5
        + 0.3 * np.cos(np.radians(la)) * np.cos(np.radians(lo))
        + 0.1 * np.sin(np.radians(la))
    )
    plate = OceanLoadedPlate.from_mantle(mf, Te=40e3, ocean=smooth, rtol=1e-13)
    Q = mf.load_space()
    deg = np.asarray(Q.degrees)
    w = np.where((deg >= 1) & (deg <= L0), 1.0, 0.0) * np.random.default_rng(
        3
    ).standard_normal(Q.dim)
    lam = deg * (deg + 1.0)
    bending = plate.D / plate.R**4 * (lam**2 - (1 - plate.nu) * lam) * w
    n = 4 * L
    zero, wq = pysh.expand.SHGLQ(n)
    lat, lon = pysh.expand.GLQGridCoord(n)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    b = plate.g * (plate.drho + plate.rho_water * (1.0 - smooth(LA, LO)))
    cilm = np.zeros((2, n + 1, n + 1))
    cilm[:, : L + 1, : L + 1] = ex.components_to_cilm(w, L)
    grid = pysh.expand.MakeGridGLQ(cilm, zero, norm=4, csphase=1)
    buoy = ex.cilm_to_components(
        pysh.expand.SHExpandGLQ(b * grid, wq, zero, norm=4, csphase=1, lmax_calc=L), L
    )
    got = apply(plate.operator(Q), bending + buoy)
    assert np.allclose(got, w, rtol=0, atol=1e-9 * np.abs(w).max())


def test_variable_rigidity_has_no_constant_filter(mf):
    plate = OceanLoadedPlate.from_mantle(mf, Te=lambda la, lo: 40e3 + 0 * la)
    with pytest.raises(ValueError):
        plate.filter()


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
