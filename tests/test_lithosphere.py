"""Tests for step 3, lithospheric loads (density_inversion/forward/lithosphere.py)."""

import numpy as np
import pytest

from density_inversion.forward.lithosphere import Lithosphere, gll, lagrange
from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.plate import ElasticPlate

LMAX = 12


@pytest.fixture(scope="module")
def mf():
    return MantleFlow(lmax=LMAX)


@pytest.fixture(scope="module")
def setup(mf):
    lith = Lithosphere.placeholder(mf)
    Q = mf.load_space()
    return lith, Q, lith.operator(Q)


def uniform(Q, value):
    """Components of a field equal to `value` everywhere (only degree 0)."""
    c = np.zeros(Q.dim)
    c[0] = value * np.sqrt(4 * np.pi)
    return c


def close(a, b):
    """Equal to rounding: pygeoinf's component round trip leaves ~1e-16 relative noise in the zero entries."""
    return np.allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())


def load_of(lith, Q, op, fields):
    """q (components) for {label: components}, every other summand zero."""
    x = np.zeros(op.domain.dim)
    for label, comp in fields.items():
        i = lith.index(label)
        x[i * Q.dim : (i + 1) * Q.dim] = comp
    return Q.to_components(op(op.domain.from_components(x)))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_gll_integrates_polynomials_of_degree_2n_minus_3(n):
    x, w = gll(n)
    assert w.sum() == pytest.approx(2.0)
    assert np.allclose(lagrange(x, x), np.eye(n))
    for p in range(max(2 * n - 2, 1)):
        exact = (1 - (-1) ** (p + 1)) / (p + 1)
        assert np.sum(w * x**p) == pytest.approx(exact, abs=1e-13)


def test_space_labels_and_layout(setup):
    lith, Q, op = setup
    labels = [s[0] for s in lith.summands()]
    assert labels[:3] == [
        "dbase of sediment",
        "dbase of lower crust",
        "dbase of lithospheric mantle",
    ]
    assert (
        len(labels) == 3 + 1 + 1 + 1 + 3
    )  # three interfaces, three uniform layers, three mantle nodes
    assert op.domain == lith.space(Q)


def test_raising_an_interface_gives_minus_g_drho_dr(setup):
    """Raising the Moho by 1 km everywhere puts lithospheric mantle where lower crust was: q = -g Drho dr."""
    lith, Q, op = setup
    depth, drho = lith.interface("base of lower crust")
    assert depth == 35e3 and drho == pytest.approx(3340.0 - 2900.0)
    q = load_of(lith, Q, op, {"dbase of lower crust": uniform(Q, 1e3)})
    assert close(q, uniform(Q, -lith.g * drho * 1e3))


@pytest.mark.parametrize(
    "label", ["drho sediment", "drho upper crust", "drho lower crust"]
)
def test_uniform_layer_density_gives_minus_g_thickness_drho(setup, label):
    lith, Q, op = setup
    layer = next(l for l in lith.layers if f"drho {l.name}" == label)
    q = load_of(lith, Q, op, {label: uniform(Q, 10.0)})
    assert close(q, uniform(Q, -lith.g * layer.thickness * 10.0))


def test_linear_profile_in_the_lithospheric_mantle_is_integrated_exactly(setup):
    """drho = a + b * depth across the lithospheric mantle (three nodes): the load is -g times its integral."""
    lith, Q, op = setup
    layer = lith.layers[-1]
    a, b = -30.0, 2e-4  # -30 kg/m^3 at the surface, rising by 0.2 kg/m^3 per km
    depths, _ = lith.nodes(layer)
    fields = {
        f"drho {layer.name} node {n}": uniform(Q, a + b * d)
        for n, d in enumerate(depths)
    }
    q = load_of(lith, Q, op, fields)
    integral = a * layer.thickness + b * (layer.bottom**2 - layer.top**2) / 2
    assert close(q, uniform(Q, -lith.g * integral))


def test_adjoint(setup):
    lith, Q, op = setup
    rng = np.random.default_rng(0)
    x = op.domain.from_components(rng.standard_normal(op.domain.dim))
    y = Q.from_components(rng.standard_normal(Q.dim))
    lhs, rhs = Q.inner_product(op(x), y), op.domain.inner_product(x, op.adjoint(y))
    assert abs(lhs - rhs) <= 1e-10 * abs(lhs)


def test_airy_isostasy_with_no_plate(mf):
    """A crustal root on land with no plate: h / (root - h) = (rho_m - rho_c) / rho_c, Airy's ratio, when the
    lithospheric mantle and the asthenosphere have the same density."""
    k = mf.kernels
    rho_m = float(k["rho_lab"])
    lith = Lithosphere.placeholder(mf, lithospheric_mantle_density=rho_m)
    Q = mf.load_space()
    op = lith.operator(Q)
    air = ElasticPlate(
        0.0, rho_m, lith.g, lith.R, LMAX
    )  # no plate, nothing fills the deflection on land
    P = air.operator(Q)
    r = 10e3  # thicken the crust by moving the Moho down 10 km
    q = load_of(lith, Q, op, {"dbase of lower crust": uniform(Q, -r)})
    h = Q.to_components(P(Q.from_components(q)))[0] / np.sqrt(4 * np.pi)
    rho_c = 2900.0
    assert h > 0
    assert h / (r - h) == pytest.approx((rho_m - rho_c) / rho_c, rel=1e-12)
