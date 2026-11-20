"""Tests for step 5, the geoid (density_inversion/forward/geoid.py)."""

import os

import numpy as np
import pytest

from density_inversion.forward.geoid import Geoid
from density_inversion.forward.lithosphere import ReferenceLithosphere
from density_inversion.forward.mantle import MantleFlow, viscosity_jump
from density_inversion.forward.plate import ElasticPlate

LMAX = 12
FEM_KERNELS = os.environ.get("DENSITY_INVERSION_FEM_KERNELS")


@pytest.fixture(scope="module")
def setup():
    mf = MantleFlow(lmax=LMAX)
    ref = ReferenceLithosphere.from_mantle(mf)
    return mf, ref, Geoid(mf, ref), mf.load_space()


def apply(op, comp):
    return op.codomain.to_components(op(op.domain.from_components(comp)))


def adjoint_error(op, seed=0):
    rng = np.random.default_rng(seed)
    x = op.domain.from_components(rng.standard_normal(op.domain.dim))
    y = op.codomain.from_components(rng.standard_normal(op.codomain.dim))
    lhs = op.codomain.inner_product(op(x), y)
    return abs(lhs - op.domain.inner_product(x, op.adjoint(y))) / abs(lhs)


def test_sheet_at_the_surface_is_the_textbook_one(setup):
    """S_l(R) = 3 / ((2l + 1) rho_mean), and degrees 0 and 1 carry nothing."""
    mf, ref, geo, Q = setup
    c = mf.shell.constants
    rho_mean = c.earth_mass / (4 / 3 * np.pi * geo.R**3)
    l = np.asarray(Q.degrees, float)
    S = geo.sheet_factor(Q, geo.R)
    assert np.allclose(S[l >= 2], 3 / ((2 * l[l >= 2] + 1) * rho_mean), rtol=1e-12)
    assert np.all(S[l < 2] == 0)
    deeper = geo.sheet_factor(Q, geo.R - 100e3)
    assert np.all(deeper[l >= 2] < S[l >= 2])  # deeper mass, smaller geoid


def test_top_sheet_identity_in_the_mantle_kernels(setup):
    """An identity of mantle.py's algebra only: K_N plus the sheet of flowkernels' deflected top is
    flowkernels' geoid. It does not go through the plate or the ocean; the full limit test, which does, is
    test_ocean_reproduces_flowkernels_in_its_own_geometry in test_model.py (audit G9).
    """
    mf, ref, geo, Q = setup
    k = mf.kernels
    l = np.asarray(k["degrees"], float)[:, None]
    r_top = float(k["r_lab"])
    fk_at_surface = (
        k["K_N_fk"] * (float(k["g_lab"]) / geo.g0) * (r_top / geo.R) ** (l + 1)
    )
    S = 4 * np.pi * geo.G * r_top * (r_top / geo.R) ** (l + 1) / ((2 * l + 1) * geo.g0)
    rebuilt = k["K_N"] + S * k["K_m_top"]
    assert np.allclose(
        rebuilt[1:],
        fk_at_surface[1:],
        rtol=1e-12,
        atol=1e-12 * np.abs(fk_at_surface).max(),
    )


@pytest.mark.skipif(
    not (FEM_KERNELS and os.path.exists(FEM_KERNELS)),
    reason="set DENSITY_INVERSION_FEM_KERNELS to kernels_jump30.npz (from fem_1d) to run",
)
def test_flowkernels_geoid_matches_fem_kernels():
    """With its top at the surface, flowkernels' geoid kernel matches fem_1d's to within 1 % (l = 2-12)."""
    full = MantleFlow(
        viscosity=viscosity_jump(30.0, 670e3), lab_depth=1.0, lmax=LMAX, cache=False
    )
    fem = np.load(FEM_KERNELS)
    k = full.kernels
    for l in range(2, LMAX + 1):
        a = k["K_N_fk"][l - 1]
        b = np.interp(k["r"], fem["r"], fem["K_N"][list(fem["degrees"]).index(l)])
        assert (
            np.sqrt(np.sum(k["weights"] * (a - b) ** 2) / np.sum(k["weights"] * b**2))
            < 0.01
        )


def test_deflection_contrasts_add_up_to_the_restoring_contrast(setup):
    """Lifting the column by w moves rho_LAB - rho_w C of mass in all, as the plate's restoring force assumes."""
    mf, ref, geo, Q = setup
    lat, lon = np.array([32.0, 0.0, 60.0]), np.array([88.0, 200.0, 260.0])
    total = sum(f for _, f in geo.deflection_sheets(lat, lon))
    C = geo.ocean(lat, lon)
    assert np.allclose(total, geo.rho_lab - geo.rho_water * C, rtol=1e-12)


def test_operators_adjoint(setup):
    mf, ref, geo, Q = setup
    X = mf.density_space()
    assert adjoint_error(geo.mantle_operator(X, Q)) < 1e-10
    assert adjoint_error(geo.lithosphere_operator(Q), seed=1) < 1e-10
    assert adjoint_error(geo.deflection_operator(Q), seed=2) < 1e-10


def test_compensated_crust_has_a_much_smaller_geoid(setup):
    """Airy: thicken the crust (Moho down) with no plate. The root's mass deficit and the uplifted column's
    excess nearly cancel; the compensated geoid is a small fraction of the root's alone (dipole cancellation).
    """
    mf, ref, geo, Q = setup
    from density_inversion.forward.plate import OceanLoadedPlate

    A = ref.operator(Q)
    GL = geo.lithosphere_operator(Q)
    Gw = geo.deflection_operator(Q)
    P = OceanLoadedPlate.from_mantle(mf, Te=0.0).operator(Q)
    l = np.asarray(Q.degrees)
    x = np.zeros(A.domain.dim)
    i = ref.index("dMoho")
    root = np.where(l == 6, -1e3, 0.0)  # 1 km of extra crust, degree 6
    x[i * Q.dim : (i + 1) * Q.dim] = root
    xs = A.domain.from_components(x)
    uncompensated = Q.to_components(GL(xs))
    w = Q.to_components(P(A(xs)))
    compensated = uncompensated + apply(Gw, w)
    m = l == 6
    assert np.linalg.norm(compensated[m]) < 0.2 * np.linalg.norm(uncompensated[m])


def test_reference_lithosphere_is_nearly_compensated(setup):
    """The reference lithosphere's own geoid and that of the deflection it causes nearly cancel."""
    mf, ref, geo, Q = setup
    from density_inversion.forward.plate import OceanLoadedPlate

    N_ref = geo.reference(Q)
    w = apply(
        OceanLoadedPlate.from_mantle(mf, Te=40e3).operator(Q), ref.reference_load(Q)
    )
    N_w = apply(geo.deflection_operator(Q), w)
    assert np.linalg.norm(N_ref + N_w) < 0.1 * np.linalg.norm(N_ref)


def test_deep_dense_anomaly_lowers_the_geoid_under_uniform_viscosity():
    """With (nearly) uniform viscosity the surface deflection beats the anomaly's own attraction: a dense
    anomaly at 1500 km gives a negative geoid above it, deflection included (water everywhere, no plate).
    """
    mf = MantleFlow(viscosity=viscosity_jump(1.0001, 670e3), lmax=LMAX, cache=False)
    ref = ReferenceLithosphere.from_mantle(mf)
    geo = Geoid(mf, ref, ocean=lambda la, lo: np.ones_like(la))
    Q = mf.load_space()
    X = mf.density_space()
    from density_inversion.forward import examples as ex

    R = float(mf.kernels["R"])
    prof = (
        lambda x: np.where(Q.degrees[None, :] == 2, 1.0, 0.0)
        * 10
        * np.exp(-0.5 * ((x[:, None] - (R - 1500e3) / R) * R / 150e3) ** 2)
        * (np.asarray(Q.orders)[None, :] == 0)
    )
    c = ex.project(X, prof)
    q = apply(mf.operator(X), c)
    w = apply(ElasticPlate.from_mantle(mf, Te=0.0).operator(Q), q)
    N = apply(geo.mantle_operator(X, Q), c) + apply(geo.deflection_operator(Q), w)
    assert N[4] < 0  # degree 2, order 0


# ---------------------------------------------------------------- fix 4 (audit E2 / G2 / G3 / G4 / G5)


def test_variable_radius_sheets_are_exact(setup):
    """A sheet at a laterally varying depth, against the definition evaluated degree by degree:
    N_lm = 4 pi G R / ((2l + 1) g0) [sigma (r / R)^(l + 2)]_lm."""
    mf, ref, geo, Q = setup
    H = ref._fine(Q)
    LA, LO = ref._grid(H)
    s = ref.structure(LA, LO)
    depth, contrast = (
        s["moho"],
        s["drho_moho"],
    )  # 9-68 km, the hardest lithospheric case
    ones = H.from_components(geo._ones(H))
    got = Q.to_components(geo.sheets(Q, [(depth, contrast)], W=H)(ones))
    l = np.asarray(Q.degrees)
    want = np.zeros(Q.dim)
    for d in range(2, Q.lmax + 1):
        field = contrast * ((geo.R - depth) / geo.R) ** (d + 2)
        comp = H.to_components(H.from_grid_values(field))[: Q.dim]
        want[l == d] = (4 * np.pi * geo.G * geo.R / ((2 * d + 1) * geo.g0) * comp)[
            l == d
        ]
    assert np.allclose(got, want, rtol=0, atol=1e-9 * np.abs(want).max())


def test_compensated_root_keeps_its_mass_balance_in_band(setup):
    """Te = 0: the root's mass and the lifted column's must cancel at every degree the model keeps. With the
    deflection truncated before it is multiplied by the column's contrasts they did not (0.5-0.8 % of the
    root, which is 10-23 % of the compensated geoid)."""
    mf, ref, geo, Q = setup
    from density_inversion.forward.plate import OceanLoadedPlate

    H = ref._fine(Q)
    LA, LO = ref._grid(H)
    A = ref.operator(Q)
    P_H = OceanLoadedPlate.from_mantle(mf, Te=0.0, rtol=1e-13).operator(Q, W=H)
    l = np.asarray(Q.degrees)
    x = np.zeros(A.domain.dim)
    i = ref.index("dMoho")
    x[i * Q.dim : (i + 1) * Q.dim] = np.where(l == 12, -1e3, 0.0)
    q = Q.to_components(A(A.domain.from_components(x)))
    w = apply(P_H, q)
    lifted = sum(c for _, c in geo.deflection_sheets(LA, LO))
    moved = H.to_components(
        H.from_grid_values(lifted * np.asarray(H.grid_values(H.from_components(w))))
    )[: Q.dim]
    root = -q / ref.g  # the root's mass, in band (its load is -g times it)
    assert np.linalg.norm(moved + root) < 1e-9 * np.linalg.norm(root)


def test_reference_layers_hold_the_column_mass(setup):
    """The sheets that make the reference geoid carry exactly the mass that makes its load."""
    mf, ref, geo, Q = setup
    lat, lon = np.array([32.0, 0.0, 60.0, -80.0]), np.array([88.0, 200.0, 260.0, 120.0])
    total = sum(m for _, m in geo.reference_layers(lat, lon))
    assert np.allclose(total, ref.column_mass(lat, lon), rtol=1e-12)


def test_reference_geoid_converges_in_quadrature(setup):
    mf, ref, geo, Q = setup
    a = geo.reference(Q)
    b = geo.reference(Q, n_crust=12, n_piece=6, n_ice=4)
    assert np.linalg.norm(a - b) < 1e-6 * np.linalg.norm(b)


def test_deflection_frames_differ_only_by_the_surface_elevation(setup):
    """The column frame is the sea-level frame moved up by the bed elevation, sheet by sheet."""
    mf, ref, geo, Q = setup
    lat, lon = np.array([32.0, 0.0]), np.array([88.0, 200.0])
    bed = ref.sample("bed_elevation", lat, lon)
    for (d1, c1), (d2, c2) in zip(
        geo.deflection_sheets(lat, lon), geo.deflection_sheets(lat, lon, "column")
    ):
        assert np.allclose(d2, d1 + bed) and np.allclose(c1, c2)
