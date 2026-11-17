"""Tests for the example density fields (density_inversion/forward/examples.py)."""

import numpy as np
import pytest

from density_inversion.forward import examples as ex
from density_inversion.forward.mantle import MantleFlow

LMAX = 30


@pytest.fixture(scope="module")
def setup():
    mf = MantleFlow(lmax=LMAX)
    X = mf.density_space()
    return mf, X, mf.operator(X), float(mf.kernels["R"])


def test_ordering_matches_pygeoinf(setup):
    """The coefficient ordering used here is the one pygeoinf's sphere spaces use."""
    Q = setup[0].load_space()
    deg, order = ex.sphere_ordering(LMAX)
    assert np.array_equal(deg, Q.degrees) and np.array_equal(order, Q.orders)


def test_grid_round_trip():
    """Components -> grid -> components returns the same coefficients."""
    import pyshtools as pysh

    comp = np.random.default_rng(3).standard_normal((LMAX + 1) ** 2)
    _, _, g = ex.to_grid(comp, LMAX)
    cilm = pysh.expand.SHExpandDH(g, norm=4, sampling=2, csphase=1, lmax_calc=LMAX)
    assert np.allclose(ex.cilm_to_components(cilm, LMAX), comp, atol=1e-10)


def test_blob_is_where_it_should_be_and_pulls_the_plate_down(setup):
    """The projected blob peaks at its centre with close to its amplitude, and the load is most
    negative directly above it."""
    mf, X, A, R = setup
    c = ex.project(
        X, ex.dense_blob(R, depth=300e3, lat=20.0, lon=150.0, width=600e3, lmax=LMAX)
    )
    lat, lon, g = ex.to_grid(ex.evaluate_at(X, c, (R - 300e3) / R), LMAX)
    i, j = np.unravel_index(np.argmax(g), g.shape)
    assert abs(lat[i] - 20.0) <= 3 and abs(lon[j] - 150.0) <= 3
    assert g[i, j] == pytest.approx(30.0, rel=0.05)
    q = A.codomain.to_components(A(X.from_components(c)))
    _, _, gq = ex.to_grid(q, LMAX)
    k, m = np.unravel_index(np.argmin(gq), gq.shape)
    assert gq[k, m] < 0 and abs(lat[k] - 20.0) <= 3 and abs(lon[m] - 150.0) <= 3


def test_power_law_rms_and_seed(setup):
    """The 1/k field has the volume RMS asked for (to within quadrature) and the same seed gives the
    same field."""
    mf, X, A, R = setup
    a, b = X.layer_bounds[0][0], X.layer_bounds[-1][1]
    p = ex.power_law(R, a, b, rms=10.0, lmax=LMAX, seed=4)
    xs = np.linspace(a, b, 300)
    w = np.gradient(xs) * xs**2
    ms = np.sum(w * np.sum(p(xs) ** 2, axis=1)) / (4 * np.pi * np.sum(w))
    assert np.sqrt(ms) == pytest.approx(10.0, rel=1e-2)  # different quadrature
    assert np.array_equal(p(xs), ex.power_law(R, a, b, rms=10.0, lmax=LMAX, seed=4)(xs))
    c = ex.project(X, p)
    x = (R - 1000e3) / R
    fit, true = ex.evaluate_at(X, c, x), p(np.array([x]))[0]
    assert np.linalg.norm(fit - true) <= 0.01 * np.linalg.norm(true)
