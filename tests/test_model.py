"""Tests for steps 4 and 6: the surface field and the assembled forward model."""

import numpy as np
import pytest

from density_inversion.forward.mantle import MantleFlow
from density_inversion.forward.model import ForwardModel
from density_inversion.forward.plate import sample_ocean_function
from density_inversion.forward.surface import SurfaceField, high_pass

LMAX = 8


@pytest.fixture(scope="module")
def fm():
    return ForwardModel(MantleFlow(lmax=LMAX))


def random_parts(fm, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(fm.X.dim),
        rng.standard_normal(fm.lithosphere_space().dim),
        rng.standard_normal(fm.S.dim),
        1e5 * rng.standard_normal(1),
    )


def test_surface_field_is_the_identity():
    sf = SurfaceField()
    S = sf.space(LMAX)
    x = np.random.default_rng(1).standard_normal(S.dim)
    op = sf.operator(S, S)
    assert np.allclose(S.to_components(op(S.from_components(x))), x, rtol=1e-14)
    hp = high_pass(50, 30)
    assert hp[5] < 1e-4 and hp[50] > 0.98


def test_block_adjoint(fm):
    rng = np.random.default_rng(2)
    M, D = fm.model_space, fm.data_space
    x = M.from_components(rng.standard_normal(M.dim))
    y = D.from_components(rng.standard_normal(D.dim))
    lhs = D.inner_product(fm.operator(x), y)
    assert abs(lhs - M.inner_product(x, fm.operator.adjoint(y))) <= 1e-8 * abs(lhs)


def test_dynamic_topography_is_plate_after_mantle(fm):
    """The mantle column of the topography row is P A_m, and nothing else adds to it."""
    x_m, _, _, _ = random_parts(fm, 3)
    h, _ = fm.forward(x_m=x_m)
    h_ref, _ = fm.offset()
    D = fm.dynamic_topography
    direct = D.codomain.to_components(D(fm.X.from_components(x_m)))
    assert np.allclose(h - h_ref, direct, rtol=1e-10, atol=1e-10 * np.abs(direct).max())


def test_surface_field_passes_through_and_has_no_geoid(fm):
    _, _, x_s, _ = random_parts(fm, 4)
    h, N = fm.forward(x_s=x_s)
    h_ref, N_ref = fm.offset()
    assert np.allclose(h - h_ref, x_s, rtol=1e-12)
    assert np.allclose(N - N_ref, 0.0, atol=1e-12 * np.abs(N_ref).max())


def test_forward_is_the_sum_of_its_parts(fm):
    x_m, x_L, x_s, x_K = random_parts(fm, 5)
    h, N = fm.forward(x_m, x_L, x_s, x_K)
    p = fm.parts(x_m, x_L, x_s, x_K)
    assert np.allclose(
        h, p["topography"]["total"], rtol=1e-8, atol=1e-8 * np.abs(h).max()
    )
    assert np.allclose(N, p["geoid"]["total"], rtol=1e-8, atol=1e-8 * np.abs(N).max())


def test_reference_offsets(fm):
    """With no perturbations the model returns the reference lithosphere's topography and geoid; excluding
    the reference makes them zero."""
    h, N = fm.forward()
    h_ref, N_ref = fm.offset()
    assert np.allclose(h, h_ref) and np.allclose(N, N_ref)
    assert (
        np.sqrt(np.sum(h_ref[1:] ** 2) / (4 * np.pi)) > 1e3
    )  # km-scale isostatic topography
    off = ForwardModel(fm.mf, include_reference=False)
    h0, N0 = off.forward()
    assert np.all(h0 == 0) and np.all(N0 == 0)


# ---------------------------------------------------------------- the sea-level constant (audit L1 / P12 / G8)


def test_uniform_load_is_a_constant(fm):
    """U maps dK (kg/m^2) to the load g dK at every point."""
    Q = fm.Q
    q = fm.U(fm.K_space.from_components(np.array([2.0e5])))
    values = np.asarray(Q.grid_values(q))
    assert np.allclose(values, fm.ref.g * 2.0e5, rtol=1e-12)


def test_sea_level_constant_raises_every_column(fm):
    """dK is an upward uniform pressure: the mean topography rises, oceans further than continents
    (restoring contrast rho_LAB - rho_w at sea, rho_LAB on land)."""
    h = fm.parts(x_K=np.array([1.0e5]))["topography"]["sea_level"]
    assert h[0] > 0
    sea, land = _grid_means(fm, h, inside_only=False)
    assert sea > land > 0


def _grid_means(fm, comp, inside_only=True):
    """Area means of a field on the fine grid: (ocean, land), by default over columns whose whole
    lithosphere lies inside the 105 km column."""
    Q, ref = fm.Q, fm.ref
    H = ref._fine(Q)
    values = np.asarray(
        H.grid_values(Q.degree_transfer_operator(H)(Q.from_components(comp)))
    )
    LA, LO = ref._grid(H)
    s = ref.structure(LA, LO)
    inside = (s["lab"] <= ref.T - s["bed"]) if inside_only else np.ones(LA.shape, bool)
    _, wlat = np.polynomial.legendre.leggauss(LA.shape[0])
    w = np.broadcast_to(wlat[:, None], LA.shape)
    sea = sample_ocean_function(LA, LO) > 0.5
    mean = lambda mask: float(np.sum((w * values)[mask]) / np.sum(w[mask]))
    return mean(sea & inside), mean(~sea & inside)


@pytest.fixture(scope="module")
def fm20():
    return ForwardModel(MantleFlow(lmax=20))


def test_reference_sits_at_sea_level(fm20):
    """Where the reference holds the whole lithosphere, the reference topography matches observed bed
    elevation on average, at sea and on land. With the old zero-mean reference load (K = <M>) the oceans
    were ~2 km too shallow."""
    fm = fm20
    K0 = fm.sea_level_constant()
    assert -7e6 < K0 < -4e6
    misfit = fm.offset()[0] - fm.ref.field(fm.Q, "bed_elevation")
    sea, land = _grid_means(fm, misfit)
    assert abs(sea) < 50.0
    assert abs(land) < 150.0
    old = fm._apply(fm.P, fm.ref.reference_load(fm.Q)) - fm.ref.field(
        fm.Q, "bed_elevation"
    )
    assert _grid_means(fm, old)[0] > 1500.0


# ---------------------------------------------------------------- the ocean that follows the geoid (audit M1 / G1 / P4 / P5)


def test_ocean_series_contracts(fm):
    """The water's load and its mass nearly cancel, so the ocean's self-gravitation T is a small operator
    and the Neumann series for (I + T)^-1 converges fast."""
    assert fm.ocean.contraction() < 0.3


def test_ocean_inverse_adjoint(fm):
    rng = np.random.default_rng(6)
    Q, B = fm.Q, fm.ocean.inverse
    x = Q.from_components(rng.standard_normal(Q.dim))
    y = Q.from_components(rng.standard_normal(Q.dim))
    lhs = Q.inner_product(B(x), y)
    assert abs(lhs - Q.inner_product(x, B.adjoint(y))) <= 1e-9 * abs(lhs)


def test_geoid_and_deflection_satisfy_the_sea_level_equations(fm):
    """For every source together: N = (direct masses) + G_w w + rho_w S(R) (C N), and
    w = P (q - rho_w g C N), h = w - N. The geoid parts are the right-hand side of the first equation, so
    they must sum to the model's N; the deflection is recomputed from the second."""
    x_m, x_L, x_s, x_K = random_parts(fm, 7)
    h, N = fm.forward(x_m, x_L, x_s, x_K)
    p = fm.parts(x_m, x_L, x_s, x_K)
    assert np.allclose(p["geoid"]["total"], N, rtol=0, atol=1e-9 * np.abs(N).max())
    a = fm._apply
    q = (
        a(fm.A_m, x_m)
        + fm.Q.to_components(fm.A_L(fm.A_L.domain.from_components(x_L)))
        + a(fm.U, x_K)
        + fm.ref.reference_load(fm.Q)
        + a(fm.U, np.array([fm.sea_level_constant()]))
    )
    w = a(fm.P, q + a(fm.ocean.load, N))
    assert np.allclose(h, w - N + x_s, rtol=0, atol=1e-9 * np.abs(h).max())


def test_ocean_reproduces_flowkernels_in_its_own_geometry():
    """The limit in which the model and flowkernels describe the same Earth: no plate, water everywhere,
    the whole deflected column lumped at the top of the shell, the sea surface there too. Then the rock
    deflection must be flowkernels' free-surface topography (K_h0) and the geoid flowkernels' own (K_N_fk),
    exactly: the mass flowkernels removes from the top (-sigma_rr / g, rock and water together) is the mass
    the plate and the ocean put back. Without the ocean (the old model) the geoid was ~11% low at degree 2.
    """
    from density_inversion.forward.ocean import SelfGravitatingOcean
    from pygeoinf.symmetric_space import sphere

    mf = MantleFlow(lmax=LMAX)
    k = mf.kernels
    Q = sphere.Sobolev(LMAX, 2.0, 0.05)
    l = np.asarray(Q.degrees, float)
    G = float(mf.shell.constants.G)
    r_top, R = float(k["r_lab"]), float(k["R"])
    g_top, g0 = float(k["g_lab"]), float(k["g0"])
    rho, rho_w = float(k["rho_lab"]), float(k["rho_water"])
    S_top = np.where(l >= 2, 4 * np.pi * G * r_top / ((2 * l + 1) * g_top), 0.0)
    diag = lambda d: Q.spectral_operator(np.asarray(d, float))
    P = diag(np.full(Q.dim, 1.0 / ((rho - rho_w) * g_top)))
    G_w = diag(S_top * (rho - rho_w))
    ocean = SelfGravitatingOcean(
        Q, P, G_w, diag(np.ones(Q.dim)), diag(S_top), rho_w, g_top
    )
    j = np.argmin(abs(k["r"] - (R - 700e3)))  # a source at 700 km depth
    per_degree = lambda kern: np.concatenate([[0.0], kern[:, j]])[l.astype(int)]
    cont = (g_top / g0) * (r_top / R) ** (
        l + 1
    )  # the code's continuation from r_top to R
    K_N_top = np.where(l >= 2, per_degree(k["K_N"]) / np.where(l >= 2, cont, 1.0), 0.0)
    q = per_degree(k["K_q"])
    a = lambda op, c: Q.to_components(op(Q.from_components(c)))
    N = a(ocean.inverse, K_N_top + a(G_w @ P, q))
    w = a(P, q + a(ocean.load, N))
    m = l >= 2
    assert np.allclose(N[m], per_degree(k["K_N_fk"])[m], rtol=1e-10)
    assert np.allclose(w[m], per_degree(k["K_h0"])[m], rtol=1e-10)
    old = K_N_top + a(
        G_w, per_degree(k["K_h0"])
    )  # water load in q, water mass never added
    deficit = 1 - old[l == 2][0] / per_degree(k["K_N_fk"])[l == 2][0]
    assert 0.05 < deficit < 0.2
