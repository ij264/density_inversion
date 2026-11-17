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


# ------------------------------------------------------------------ 3c and 3d: WINTERC-G reference

from density_inversion.forward.lithosphere import ReferenceLithosphere  # noqa: E402


@pytest.fixture(scope="module")
def ref(mf):
    return ReferenceLithosphere.from_mantle(mf)


@pytest.fixture(scope="module")
def ref_op(mf, ref):
    Q = mf.load_space()
    return Q, ref.operator(Q)


def ref_load(ref, Q, op, fields):
    x = np.zeros(op.domain.dim)
    for label, comp in fields.items():
        i = ref.index(label)
        x[i * Q.dim : (i + 1) * Q.dim] = comp
    return Q.to_components(op(op.domain.from_components(x)))


def test_reference_structure_is_sensible(ref):
    lat, lon = np.array([32.0, 0.0, -75.0]), np.array(
        [88.0, 200.0, 0.0]
    )  # Tibet, Pacific, Antarctica
    s = ref.structure(lat, lon)
    assert (
        s["t_c"][0] > 60e3 and s["t_c"][1] < 12e3
    )  # thick Tibetan crust, thin oceanic crust
    assert np.all(s["t_l"] > 0) and np.all(s["drho_moho"] > 300)
    assert np.allclose(sum(s["node_weights"]), s["t_l"])
    assert ref.sample("ice_thickness", lat, lon)[2] > 1e3  # Antarctic ice


def test_uniform_crustal_density_gives_minus_g_thickness(ref, ref_op):
    """10 kg/m^3 added to the crust everywhere: q = -g t_c(x) 10, projected onto degree <= lmax."""
    Q, op = ref_op
    c = np.zeros(Q.dim)
    c[0] = 10.0 * np.sqrt(4 * np.pi)
    q = ref_load(ref, Q, op, {"drho crust": c})
    expect = ref._analyse(
        Q, lambda LA, LO: -ref.g * 10.0 * ref.structure(LA, LO)["t_c"]
    )
    assert np.allclose(q, expect, rtol=1e-10, atol=1e-10 * np.abs(expect).max())


def test_uniform_moho_raise_gives_minus_g_contrast(ref, ref_op):
    Q, op = ref_op
    c = np.zeros(Q.dim)
    c[0] = 1e3 * np.sqrt(4 * np.pi)
    q = ref_load(ref, Q, op, {"dMoho": c})
    expect = ref._analyse(
        Q, lambda LA, LO: -ref.g * 1e3 * ref.structure(LA, LO)["drho_moho"]
    )
    assert np.allclose(q, expect, rtol=1e-10, atol=1e-10 * np.abs(expect).max())


def test_reference_operator_adjoint(ref_op):
    Q, op = ref_op
    rng = np.random.default_rng(3)
    x = op.domain.from_components(rng.standard_normal(op.domain.dim))
    y = Q.from_components(rng.standard_normal(Q.dim))
    lhs, rhs = Q.inner_product(op(x), y), op.domain.inner_product(x, op.adjoint(y))
    assert abs(lhs - rhs) <= 1e-10 * abs(lhs)


def test_column_mass_by_brute_force(ref):
    """The Gauss quadrature of the lid against a fine trapezoid down one column (central Asia)."""
    lat, lon = np.array([45.0]), np.array([90.0])
    s = ref.structure(lat, lon)
    z = np.linspace(s["t_c"][0], ref.T, 20001)
    rho = ref.mantle_density(
        z - s["bed"][0], np.full(z.size, 45.0), np.full(z.size, 90.0)
    )
    lid = np.sum(0.5 * (rho[1:] + rho[:-1]) * np.diff(z))
    brute = (
        s["rho_c"][0] * s["t_c"][0]
        + lid
        + 910.0 * ref.sample("ice_thickness", lat, lon)[0]
    )
    assert ref.column_mass(lat, lon)[0] == pytest.approx(brute, rel=1e-6)


def test_reference_load_lifts_thick_crust_and_sinks_oceans(mf, ref):
    """q_ref has zero mean, pushes up under Tibet (thick light crust) and down under the Pacific."""
    Q = mf.load_space()
    q = ref.reference_load(Q)
    assert q[0] == 0.0
    lat, lon, g = ex_grid(q)
    at = lambda la, lo: g[np.argmin(abs(lat - la)), np.argmin(abs(lon - lo))]
    assert at(32, 88) > 0 and at(0, 200) < 0


def ex_grid(comp):
    from density_inversion.forward import examples as ex

    return ex.to_grid(comp, LMAX)
